#!/usr/bin/env python3
"""Translation job for sender recordings using multiple model backends.

Supports:
  - gemini: Google Gemini API (requires GEMINI_API_KEY env var)
  - qwen2audio: Qwen2-Audio-7B Instruct local model (requires transformers, librosa)
  - phi4mm: Microsoft Phi-4-multimodal-instruct local model (requires transformers, librosa)
  - voxtralsmall: Voxtral-Small-24B-2507 local model (requires transformers, torch)
  - desta2: DeSTA2-8B local model (requires transformers, torch, HF_TOKEN env var)
  - googleapi: Google Cloud Speech-to-Text + Translation API (requires GOOGLE_CLOUD_PROJECT env var)

Usage:
  python scripts/run_translation_job.py --model gemini
  python scripts/run_translation_job.py --model qwen2audio
  python scripts/run_translation_job.py --model phi4mm
  python scripts/run_translation_job.py --model desta2
  python scripts/run_translation_job.py --model gemini --is-mock  # dry run without API calls
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
from tqdm import tqdm

import librosa
import soundfile as sf
import tyro

log = logging.getLogger(__name__)

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

ModelChoice = Literal["gemini", "qwen2audio", "phi4mm", "voxtralsmall", "desta2", "googleapi"]


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------


def load_audio(audio_path: Path, target_sr: Optional[int] = None) -> Tuple[Any, int]:
    """Load audio from disk and optionally resample to target sample rate.

    Tries a WAV-specific loader first to avoid audioread/ffmpeg dependency,
    then falls back to librosa's general loader.

    Returns (audio_array, sample_rate).
    """

    path_str = str(audio_path)
    audio, sr = librosa.load(path_str, sr=target_sr)
    return audio, sr


def audio_to_base64(audio: Any, sr: int) -> str:
    """Convert a librosa audio array to base64-encoded WAV format.

    Args:
        audio: Audio array from librosa.load()
        sr: Sample rate of the audio

    Returns:
        Base64-encoded string of the audio in WAV format
    """

    # Write audio to in-memory bytes buffer as WAV
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    buffer.seek(0)

    # Encode to base64
    audio_bytes = buffer.read()
    return base64.b64encode(audio_bytes).decode("utf-8")


# ---------------------------------------------------------------------------
# Translator base class and implementations
# ---------------------------------------------------------------------------


def inference_decorator():
    """Return a decorator that enables torch.inference_mode() when available.

    This avoids importing torch at module import-time in environments without
    PyTorch. If PyTorch is not available, returns a no-op decorator.
    """
    try:
        import torch

        return torch.inference_mode()
    except Exception:
        # Fallback no-op decorator
        def _noop_decorator(fn):
            return fn

        return _noop_decorator


class BaseTranslator(ABC):
    """Abstract base class for translation backends."""

    model_name: str

    @abstractmethod
    def translate(self, audio_path: Path) -> str:
        """Translate audio at the given path and return the translated text."""
        ...

    def cleanup(self) -> None:
        """Optional cleanup hook called after all translations are done."""
        pass


class GeminiTranslator(BaseTranslator):
    """Google Gemini API translator."""

    model_name = "gemini-2.5-flash"

    PROMPT = """You are a professional English-to-Portuguese translator, tasked with providing translations suitable for use in Portugal. Your goal is to accurately convey the meaning and nuances of the original English audio while adhering to Portuguese grammar, vocabulary, and cultural sensitivities.

Listen carefully to the following audio file and return the Portuguese translation without any additional explanations or commentary.
"""

    def __init__(self, is_mock: bool = False):
        self.is_mock = is_mock
        self.client = None
        if not is_mock:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is required")
            self.client = genai.Client(api_key=api_key)

    def translate(self, audio_path: Path) -> str:
        if self.is_mock:
            return "Esta é uma tradução simulada (Gemini mock)."

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                self.PROMPT,
                self.client.files.upload(file=audio_path),
            ],
        )
        return response.text


class Qwen2AudioTranslator(BaseTranslator):
    """Qwen2-Audio-7B-Instruct local model translator."""

    model_name = "qwen2audio"
    HF_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

    def __init__(self, is_mock: bool = False):
        self.is_mock = is_mock
        self.model = None
        self.processor = None

        if not is_mock:
            log.info("Loading Qwen2-Audio model from %s...", self.HF_MODEL_ID)
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(
                self.HF_MODEL_ID,
                trust_remote_code=True,
            )
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.HF_MODEL_ID,
                device_map="auto",
                trust_remote_code=True,
            )
            log.info("Qwen2-Audio model loaded.")

    @inference_decorator()
    def translate(self, audio_path: Path) -> str:
        if self.is_mock:
            return "Esta é uma tradução simulada (Qwen2-Audio mock)."

        # Load audio and resample to processor's expected sample rate
        target_sr = self.processor.feature_extractor.sampling_rate
        audio, sr = load_audio(audio_path, target_sr=target_sr)

        # Build conversation following the Qwen2-Audio-Instruct format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "placeholder"},
                    {"type": "text", "text": "Translate the audio to Portuguese:"},
                ],
            },
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # Collect audios (in this case, just one audio)
        audios = [audio]

        # Process inputs
        inputs = self.processor(
            text=text, audios=audios, return_tensors="pt", padding=True
        )
        inputs.input_ids = inputs.input_ids.to(self.model.device)

        # Generate translation
        generated_ids = self.model.generate(**inputs, max_length=512)
        # Strip input tokens from the generated sequence
        generated_ids = generated_ids[:, inputs.input_ids.size(1) :]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response


class Phi4MMTranslator(BaseTranslator):
    """Microsoft Phi-4-multimodal-instruct local model translator."""

    model_name = "phi4mm"
    HF_MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

    USER_PROMPT = "<|user|>"
    ASSISTANT_PROMPT = "<|assistant|>"
    PROMPT_SUFFIX = "<|end|>"
    SPEECH_PROMPT = "Translate the audio to Portuguese."

    def __init__(self, is_mock: bool = False):
        self.is_mock = is_mock
        self.model = None
        self.processor = None
        self.generation_config = None

        if not is_mock:
            log.info("Loading Phi-4 multimodal model from %s...", self.HF_MODEL_ID)
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoProcessor,
                GenerationConfig,
            )

            self.processor = AutoProcessor.from_pretrained(
                self.HF_MODEL_ID,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.HF_MODEL_ID,
                trust_remote_code=True,
                device_map="cuda",
                torch_dtype="auto",
            )
            self.generation_config = GenerationConfig.from_pretrained(self.HF_MODEL_ID)
            log.info("Phi-4 multimodal model loaded.")

    @inference_decorator()
    def translate(self, audio_path: Path) -> str:
        if self.is_mock:
            return "Esta é uma tradução simulada (Phi-4 mock)."

        import torch

        audio, samplerate = load_audio(audio_path)

        # Build prompt
        prompt = (
            f"{self.USER_PROMPT}<|audio_1|>{self.SPEECH_PROMPT}"
            f"{self.PROMPT_SUFFIX}{self.ASSISTANT_PROMPT}"
        )

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        inputs = self.processor(
            text=prompt,
            audios=[(audio, samplerate)],
            return_tensors="pt",
        ).to(device)

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=self.generation_config,
            num_logits_to_keep=1
        )
        # Strip input tokens
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response


class VoxtralSmallTranslator(BaseTranslator):
    """Voxtral-Small-24B-2507 local model translator."""

    model_name = "voxtralsmall"
    HF_MODEL_ID = "mistralai/Voxtral-Small-24B-2507"

    def __init__(self, is_mock: bool = False):
        self.is_mock = is_mock
        self.model = None
        self.processor = None

        if not is_mock:
            log.info("Loading Voxtral model from %s...", self.HF_MODEL_ID)
            import torch
            from transformers import AutoProcessor, VoxtralForConditionalGeneration

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = AutoProcessor.from_pretrained(self.HF_MODEL_ID)
            self.model = VoxtralForConditionalGeneration.from_pretrained(
                self.HF_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            log.info("Voxtral model loaded.")

    @inference_decorator()
    def translate(self, audio_path: Path) -> str:
        if self.is_mock:
            return "Esta é uma tradução simulada (Voxtral Mini mock)."

        import torch

        prompt = "Translate the audio to Portuguese."
        audio, sr = load_audio(audio_path, target_sr=16000)
        audio_b64 = audio_to_base64(audio, sr)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "base64": audio_b64},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(conversation)
        inputs = inputs.to(self.model.device, dtype=torch.bfloat16)

        outputs = self.model.generate(**inputs, max_new_tokens=500)
        decoded_outputs = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        return decoded_outputs[0]


class GoogleAPITranslator(BaseTranslator):
    """Google Cloud Speech-to-Text + Translation API translator.

    Uses Chirp 3 model for transcription and Cloud Translation for translation.
    Requires GOOGLE_CLOUD_PROJECT environment variable.
    """

    model_name = "googleapi"
    REGION = "us"

    def __init__(self, is_mock: bool = False):
        self.is_mock = is_mock
        self.project_id = None
        self.speech_client = None
        self.translate_client = None

        if not is_mock:
            from google.api_core.client_options import ClientOptions
            from google.cloud import translate_v3
            from google.cloud.speech_v2 import SpeechClient

            self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not self.project_id:
                raise ValueError(
                    "GOOGLE_CLOUD_PROJECT environment variable is required"
                )

            self.speech_client = SpeechClient(
                client_options=ClientOptions(
                    api_endpoint=f"{self.REGION}-speech.googleapis.com",
                )
            )
            self.translate_client = translate_v3.TranslationServiceClient()
            log.info("Google API clients initialized.")

    def _transcribe(self, audio_path: Path) -> str:
        """Transcribe audio to English text using Chirp 3 model."""
        from google.cloud.speech_v2 import types as speech_types

        with open(audio_path, "rb") as f:
            audio_content = f.read()

        config = speech_types.RecognitionConfig(
            auto_decoding_config=speech_types.AutoDetectDecodingConfig(),
            language_codes=["en-US"],
            model="chirp_3",
        )

        request = speech_types.RecognizeRequest(
            recognizer=f"projects/{self.project_id}/locations/{self.REGION}/recognizers/_",
            config=config,
            content=audio_content,
        )

        response = self.speech_client.recognize(request=request)

        transcripts = []
        for result in response.results:
            if result.alternatives:
                transcripts.append(result.alternatives[0].transcript)

        return " ".join(transcripts)

    def _translate(self, text: str) -> str:
        """Translate English text to Portuguese using Cloud Translation API."""
        parent = f"projects/{self.project_id}/locations/global"

        response = self.translate_client.translate_text(
            contents=[text],
            parent=parent,
            mime_type="text/plain",
            source_language_code="en-US",
            target_language_code="pt",
        )

        translated_parts = [t.translated_text for t in response.translations]
        return " ".join(translated_parts)

    def translate(self, audio_path: Path) -> str:
        if self.is_mock:
            return "Esta é uma tradução simulada (Google API mock)."

        # Step 1: Transcribe audio to English
        transcript = self._transcribe(audio_path)
        log.debug("  Transcribed: %s...", transcript[:100])

        # Step 2: Translate English to Portuguese
        translated = self._translate(transcript)
        return translated


class DeSTA2Translator(BaseTranslator):
    """DeSTA2-8B local model translator for audio-to-speech translation."""

    model_name = "desta2"
    HF_MODEL_ID = "DeSTA-ntu/DeSTA2-8B-beta"

    def __init__(self, is_mock: bool = False):
        self.is_mock = is_mock
        self.model = None

        if not is_mock:
            log.info("Loading DeSTA2 model from %s...", self.HF_MODEL_ID)
            from transformers import AutoModel

            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                raise ValueError(
                    "HF_TOKEN environment variable is required for DeSTA2 model access"
                )

            self.model = AutoModel.from_pretrained(
                self.HF_MODEL_ID,
                trust_remote_code=True,
                token=hf_token,
            )
            log.info("DeSTA2 model loaded.")

    @inference_decorator()
    def translate(self, audio_path: Path) -> str:
        if self.is_mock:
            return "Esta é uma tradução simulada (DeSTA2 mock)."

        messages = [
            {"role": "audio", "content": str(audio_path)},
            {"role": "user", "content": "Translate this audio to Portuguese. Produce only the Portuguese translation, without any additional explanations or commentary."},
        ]

        generated_ids = self.model.chat(
            messages,
            max_new_tokens=512,
        )

        response = self.model.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return response


def get_translator(model: ModelChoice, is_mock: bool = False) -> BaseTranslator:
    """Factory function to instantiate the appropriate translator."""
    if model == "gemini":
        return GeminiTranslator(is_mock=is_mock)
    elif model == "qwen2audio":
        return Qwen2AudioTranslator(is_mock=is_mock)
    elif model == "phi4mm":
        return Phi4MMTranslator(is_mock=is_mock)
    elif model == "voxtralsmall":
        return VoxtralSmallTranslator(is_mock=is_mock)
    elif model == "desta2":
        return DeSTA2Translator(is_mock=is_mock)
    elif model == "googleapi":
        return GoogleAPITranslator(is_mock=is_mock)
    else:
        raise ValueError(f"Unknown model: {model}")


# ---------------------------------------------------------------------------
# JSONL persistence
# ---------------------------------------------------------------------------


def get_translation_path(entry_id: int) -> Path:
    """Return the JSONL file path for a conversation entry."""
    return TRANSLATIONS_DIR / f"entry_{entry_id}.jsonl"


def load_existing_translations(entry_id: int) -> List[Dict[str, Any]]:
    """Load existing translation records for an entry from JSONL."""
    path = get_translation_path(entry_id)
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


def get_translated_models(entry_id: int) -> Set[str]:
    """Return the set of model names that have already translated an entry."""
    records = load_existing_translations(entry_id)
    return {r.get("model", "") for r in records if r.get("model")}


def append_translation(
    entry_id: int,
    conv_id: str,
    model_name: str,
    translated_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Append a translation record to the entry's JSONL file."""
    TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)

    record: Dict[str, Any] = {
        "conversation_entry_id": entry_id,
        "conversation_id": conv_id,
        "model": model_name,
        "translated_text": translated_text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        record["metadata"] = metadata

    path = get_translation_path(entry_id)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(
    model: ModelChoice,
    is_mock: bool = False,
    all_entries: bool = False,
) -> None:
    """Iterate over conversation entries and generate translations.

    Args:
        model: Which translation model to use (gemini, qwen2audio, phi4mm).
        is_mock: If True, skip actual model calls and return mock translations.
        all_entries: If True, process all entries regardless of status (default
            only processes entries with ENTRY_STATUS_RECORDED).
    """
    log.info("Starting translation job with model=%s, is_mock=%s, all_entries=%s", model, is_mock, all_entries)

    # Initialize translator (loads model weights once for local models)
    translator = get_translator(model, is_mock=is_mock)

    with app.app_context():
        # Find entries that have a recording
        query = ConversationEntry.query.join(
            Recording, Recording.conversation_entry_id == ConversationEntry.id
        )
        if not all_entries:
            query = query.filter(ConversationEntry.status == ENTRY_STATUS_RECORDED)
        entries = query.all()

        if not entries:
            log.info("No recorded conversation entries awaiting translation.")
            return

        recordings_dir = Path(app.config["RECORDINGS_DIR"])
        processed = 0
        skipped_already_done = 0
        skipped_missing = 0
        failed = 0

        for entry in tqdm(entries, desc="Translating entries", total=len(entries)):
            # Check if this model already translated this entry
            existing_models = get_translated_models(entry.id)
            if translator.model_name in existing_models:
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

            try:
                # print(f"Translating entry {entry.id} (conv_id={conv_id})...")
                translated_text = translator.translate(audio_path)
            except Exception as exc:  # noqa: BLE001 - allow inspection of failures
                log.warning("Failed to translate entry %s: %s", entry.id, exc)
                failed += 1
                continue

            try:
                output_path = append_translation(
                    entry_id=entry.id,
                    conv_id=conv_id,
                    model_name=translator.model_name,
                    translated_text=translated_text,
                )
                # print(f"Stored translation for entry {entry.id} at {output_path}")
                processed += 1
            except OSError as exc:
                log.warning("Failed to persist translation for entry %s: %s", entry.id, exc)
                failed += 1
                continue

        # Cleanup (e.g., free GPU memory)
        translator.cleanup()

        log.info("--- Summary ---")
        log.info("Model: %s", translator.model_name)
        log.info("Processed: %s", processed)
        log.info("Skipped (already translated by this model): %s", skipped_already_done)
        log.info("Skipped (missing recording): %s", skipped_missing)
        log.info("Failed: %s", failed)


if __name__ == "__main__":
    tyro.cli(main)
