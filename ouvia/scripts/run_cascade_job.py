#!/usr/bin/env python3
"""Cascade job for sender recordings: transcription and text translation.

Supports two independent stages:
  --transcribe: Use Whisper Large V3 to transcribe audio recordings to English text
  --translate-text: Use Tower Plus 9B to translate existing transcripts to Portuguese

Usage:
  python scripts/run_cascade_job.py --transcribe
  python scripts/run_cascade_job.py --translate-text
  python scripts/run_cascade_job.py --transcribe --translate-text  # run both stages
  python scripts/run_cascade_job.py --transcribe --is-mock  # dry run without model calls
"""

from __future__ import annotations

import json
import logging
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import tyro

log = logging.getLogger(__name__)
from sqlalchemy.orm import joinedload, load_only
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup for imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_STUDY_PORTAL_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _STUDY_PORTAL_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ouvia.app import (
    ENTRY_STATUS_RECORDED,
    Conversation,
    ConversationEntry,
    Recording,
    app,
)

TRANSLATIONS_DIR = Path(app.root_path) / "translations"


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------


def load_audio(audio_path: Path, target_sr: Optional[int] = None) -> tuple[Any, int]:
    """Load audio from disk and optionally resample to target sample rate.

    Returns (audio_array, sample_rate).
    """
    import librosa

    path_str = str(audio_path)
    audio, sr = librosa.load(path_str, sr=target_sr)
    return audio, sr


# ---------------------------------------------------------------------------
# Transcriber base class and Whisper implementation
# ---------------------------------------------------------------------------


class BaseTranscriber(ABC):
    """Abstract base class for transcription backends."""

    model_name: str

    @abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        """Transcribe audio at the given path and return the transcribed text."""
        ...

    def cleanup(self) -> None:
        """Optional cleanup hook called after all transcriptions are done."""
        pass


class WhisperTranscriber(BaseTranscriber):
    """OpenAI Whisper Large V3 transcriber using HuggingFace transformers."""

    model_name = "whisper-large-v3"
    HF_MODEL_ID = "openai/whisper-large-v3"

    def __init__(self, is_mock: bool = False):
        self.is_mock = is_mock
        self.pipe = None

        if not is_mock:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            log.info("Loading Whisper model from %s...", self.HF_MODEL_ID)

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.HF_MODEL_ID,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(device)

            processor = AutoProcessor.from_pretrained(self.HF_MODEL_ID)

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
            )
            log.info("Whisper model loaded.")

    def transcribe(self, audio_path: Path) -> str:
        if self.is_mock:
            return "This is a mock transcription (Whisper mock)."

        # Load audio at 16kHz (Whisper's expected sample rate)
        audio, sr = load_audio(audio_path, target_sr=16000)

        # Run transcription
        result = self.pipe(
            {"array": audio, "sampling_rate": sr}, return_timestamps=True
        )
        return result["text"]

    def cleanup(self) -> None:
        """Free GPU memory."""
        if self.pipe is not None:
            import torch

            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Translator base class and Tower implementation
# ---------------------------------------------------------------------------


class BaseTextTranslator(ABC):
    """Abstract base class for text translation backends."""

    model_name: str

    @abstractmethod
    def translate(self, text: str) -> str:
        """Translate text and return the translated text."""
        ...

    def cleanup(self) -> None:
        """Optional cleanup hook called after all translations are done."""
        pass


class TowerTranslator(BaseTextTranslator):
    """Unbabel Tower Plus 9B text translator."""

    model_name = "tower-plus-9b"
    HF_MODEL_ID = "Unbabel/Tower-Plus-9B"

    PROMPT_TEMPLATE = (
        "Translate the following English source text to Portuguese (Portugal):\n"
        "English: {text}\n"
        "Portuguese (Portugal): "
    )

    def __init__(self, is_mock: bool = False):
        self.is_mock = is_mock
        self.pipe = None

        if not is_mock:
            from transformers import pipeline

            log.info("Loading Tower model from %s...", self.HF_MODEL_ID)

            self.pipe = pipeline(
                "text-generation",
                model=self.HF_MODEL_ID,
                device_map="auto",
            )
            log.info("Tower model loaded.")

    def translate(self, text: str) -> str:
        if self.is_mock:
            return "Esta é uma tradução simulada (Tower mock)."

        prompt = self.PROMPT_TEMPLATE.format(text=text)
        messages = [{"role": "user", "content": prompt}]

        outputs = self.pipe(messages, max_new_tokens=512, do_sample=False)

        # Extract the generated text from the response
        generated = outputs[0]["generated_text"]

        # The output is a list of messages; get the assistant's response
        if isinstance(generated, list):
            # Find the assistant's message
            for msg in generated:
                if msg.get("role") == "assistant":
                    return msg.get("content", "").strip()
            # Fallback: return the last message content
            return generated[-1].get("content", "").strip() if generated else ""
        else:
            # If it's a string, return it directly
            return str(generated).strip()

    def cleanup(self) -> None:
        """Free GPU memory."""
        if self.pipe is not None:
            import torch

            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# JSONL persistence
# ---------------------------------------------------------------------------


def get_entry_path(entry_id: int) -> Path:
    """Return the JSONL file path for a conversation entry."""
    return TRANSLATIONS_DIR / f"entry_{entry_id}.jsonl"


def load_existing_records(entry_id: int) -> List[Dict[str, Any]]:
    """Load existing records for an entry from JSONL."""
    path = get_entry_path(entry_id)
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def get_transcribed_models(entry_id: int) -> set[str]:
    """Return the set of model names that have already transcribed an entry."""
    records = load_existing_records(entry_id)
    return {
        r.get("transcription_model", "")
        for r in records
        if r.get("transcription") and r.get("transcription_model")
    }


def get_translated_models(entry_id: int) -> set[str]:
    """Return the set of model names that have already translated an entry."""
    records = load_existing_records(entry_id)
    return {
        r.get("model", "")
        for r in records
        if r.get("translated_text") and r.get("model")
    }


def get_transcription_for_model(entry_id: int, model_name: str) -> Optional[str]:
    """Get the transcription text for a specific model, if it exists."""
    records = load_existing_records(entry_id)
    for r in records:
        if r.get("transcription_model") == model_name and r.get("transcription"):
            return r["transcription"]
    return None


def get_latest_transcription(entry_id: int) -> Optional[tuple[str, str]]:
    """Get the latest transcription (model_name, text) for an entry, if any exists."""
    records = load_existing_records(entry_id)
    # Return the most recent transcription
    for r in reversed(records):
        if r.get("transcription"):
            return (r.get("transcription_model", "unknown"), r["transcription"])
    return None


def append_transcription(
    entry_id: int,
    conv_id: str,
    model_name: str,
    transcription: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Append a transcription record to the entry's JSONL file."""
    TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)

    record: Dict[str, Any] = {
        "conversation_entry_id": entry_id,
        "conversation_id": conv_id,
        "transcription_model": model_name,
        "transcription": transcription,
        "transcription_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        record["metadata"] = metadata

    path = get_entry_path(entry_id)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return path


def append_translation(
    entry_id: int,
    conv_id: str,
    model_name: str,
    translated_text: str,
    source_transcription_model: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Merge translation into the entry's existing JSONL record.

    Instead of appending a separate translation record, this updates the most
    recent record for the given `entry_id` that contains a `transcription` by
    adding only three new keys: `model`, `translated_text`, and `generated_at`.

    If no transcription record exists, this function does NOT create a new
    minimal translation record and will return None.
    """
    TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)

    path = get_entry_path(entry_id)

    records = load_existing_records(entry_id)

    updated = False
    # Find the most recent transcription record and add translation fields to it.
    for r in reversed(records):
        if r.get("transcription"):
            r["model"] = model_name
            r["translated_text"] = translated_text
            r["generated_at"] = datetime.now(timezone.utc).isoformat()
            updated = True
            break

    if not updated:
        # No transcription record found — do not create a new record.
        # Caller is expected not to request translations when no transcription
        # exists; return None to indicate nothing was written.
        return None

    # Write back all records (overwrite file) so the modification is persisted.
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return path


# ---------------------------------------------------------------------------
# Transcription job
# ---------------------------------------------------------------------------


def run_transcription_job(is_mock: bool = False) -> None:
    """Run transcription on all recorded entries using Whisper."""
    log.info("Starting transcription job with Whisper, is_mock=%s", is_mock)

    transcriber = WhisperTranscriber(is_mock=is_mock)

    with app.app_context():
        # Find all recorded entries that have a recording
        entries = (
            ConversationEntry.query.filter(
                ConversationEntry.status == ENTRY_STATUS_RECORDED
            )
            .join(Recording, Recording.conversation_entry_id == ConversationEntry.id)
            .all()
        )

        if not entries:
            log.info("No recorded conversation entries awaiting transcription.")
            return

        recordings_dir = Path(app.config["RECORDINGS_DIR"])
        processed = 0
        skipped_already_done = 0
        skipped_missing = 0
        failed = 0

        for entry in tqdm(entries, desc="Transcribe"):
            # Check if this model already transcribed this entry
            existing_models = get_transcribed_models(entry.id)
            if transcriber.model_name in existing_models:
                skipped_already_done += 1
                continue

            recording = entry.recording
            if recording is None:
                log.debug("Skipping entry %s: no recording metadata present.", entry.id)
                skipped_missing += 1
                continue

            audio_path = recordings_dir / recording.stored_filename
            if not audio_path.exists():
                log.debug(
                    "Skipping entry %s: recorded file missing at %s.",
                    entry.id,
                    audio_path,
                )
                skipped_missing += 1
                continue

            # Get conversation external_id for the JSONL record
            conversation: Optional[Conversation] = entry.conversation
            conv_id = (
                conversation.external_id
                if conversation
                else f"unknown_{entry.conversation_id}"
            )

            # try:
            log.debug("Transcribing entry %s (conv_id=%s)...", entry.id, conv_id)
            transcription = transcriber.transcribe(audio_path)
            log.debug("  Transcribed: %s...", transcription[:100])
            # except Exception as exc:
            #     print(f"Failed to transcribe entry {entry.id}: {exc}")
            #     failed += 1
            #     continue

            try:
                output_path = append_transcription(
                    entry_id=entry.id,
                    conv_id=conv_id,
                    model_name=transcriber.model_name,
                    transcription=transcription,
                )
                log.debug("Stored transcription for entry %s at %s", entry.id, output_path)
                processed += 1
            except OSError as exc:
                log.warning("Failed to persist transcription for entry %s: %s", entry.id, exc)
                failed += 1
                continue

        transcriber.cleanup()

        log.info("--- Transcription Summary ---")
        log.info("Model: %s", transcriber.model_name)
        log.info("Processed: %s", processed)
        log.info("Skipped (already transcribed by this model): %s", skipped_already_done)
        log.info("Skipped (missing recording): %s", skipped_missing)
        log.info("Failed: %s", failed)


# ---------------------------------------------------------------------------
# Text translation job
# ---------------------------------------------------------------------------


def run_text_translation_job(is_mock: bool = False) -> None:
    """Run translation on all existing transcriptions using Tower Plus 9B."""
    log.info("Starting text translation job with Tower, is_mock=%s", is_mock)

    translator = TowerTranslator(is_mock=is_mock)

    with app.app_context():
        # Find all recorded entries that have a recording
        entries = (
            ConversationEntry.query.filter(
                ConversationEntry.status == ENTRY_STATUS_RECORDED
            )
            .join(Recording, Recording.conversation_entry_id == ConversationEntry.id)
            .all()
        )

        if not entries:
            log.info("No recorded conversation entries found.")
            return

        processed = 0
        skipped_already_done = 0
        skipped_no_transcription = 0
        failed = 0

        for entry in tqdm(entries, desc="Translate"):
            # Check if Tower already translated this entry
            existing_models = get_translated_models(entry.id)
            if translator.model_name in existing_models:
                skipped_already_done += 1
                continue

            # Get the latest transcription for this entry
            transcription_data = get_latest_transcription(entry.id)
            if transcription_data is None:
                skipped_no_transcription += 1
                continue

            source_model, transcription_text = transcription_data

            # Get conversation external_id for the JSONL record
            conversation: Optional[Conversation] = entry.conversation
            conv_id = (
                conversation.external_id
                if conversation
                else f"unknown_{entry.conversation_id}"
            )

            # try:
            log.debug("Translating entry %s (conv_id=%s)...", entry.id, conv_id)
            log.debug("  Source (%s): %s...", source_model, transcription_text[:80])
            translated_text = translator.translate(transcription_text)
            # print(
            #     f"  Translated: {translated_text[:80]}..."
            #     if len(translated_text) > 80
            #     else f"  Translated: {translated_text}"
            # )
            # except Exception as exc:
            #     print(f"Failed to translate entry {entry.id}: {exc}")
            #     failed += 1
            #     continue

            try:
                output_path = append_translation(
                    entry_id=entry.id,
                    conv_id=conv_id,
                    model_name=translator.model_name,
                    translated_text=translated_text,
                    source_transcription_model=source_model,
                )
                if output_path is None:
                    # This should not normally happen because we check for
                    # transcriptions earlier, but handle defensively.
                    log.warning(
                        "Skipping storing translation for entry %s: no transcription record found.",
                        entry.id,
                    )
                    skipped_no_transcription += 1
                    continue

                # print(f"Stored translation for entry {entry.id} at {output_path}")
                processed += 1
            except OSError as exc:
                log.warning("Failed to persist translation for entry %s: %s", entry.id, exc)
                failed += 1
                continue

        translator.cleanup()

        log.info("--- Text Translation Summary ---")
        log.info("Model: %s", translator.model_name)
        log.info("Processed: %s", processed)
        log.info("Skipped (already translated by this model): %s", skipped_already_done)
        log.info("Skipped (no transcription available): %s", skipped_no_transcription)
        log.info("Failed: %s", failed)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(
    transcribe: bool = False,
    translate_text: bool = False,
    is_mock: bool = False,
) -> None:
    """Run cascade transcription and/or translation jobs.

    Args:
        transcribe: If True, run Whisper transcription on audio recordings.
        translate_text: If True, run Tower translation on existing transcripts.
        is_mock: If True, skip actual model calls and return mock outputs.
    """
    if not transcribe and not translate_text:
        log.warning("No job specified. Use --transcribe and/or --translate-text.")
        log.info("Run with --help for usage information.")
        return

    if transcribe:
        run_transcription_job(is_mock=is_mock)

    if translate_text:
        run_text_translation_job(is_mock=is_mock)


if __name__ == "__main__":
    tyro.cli(main)
