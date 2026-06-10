#!/usr/bin/env python3
"""Apply stored translations to conversation entries."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_this_file = Path(__file__).resolve()
_study_portal_dir = _this_file.parents[1]
_project_root = _study_portal_dir.parent
sys.path.insert(0, str(_project_root))

from ouvia.app import (
    ENTRY_STATUS_RECORDED,
    ENTRY_STATUS_TRANSLATED,
    ConversationEntry,
    app,
    db,
)

def load_translations(entry_id: int, translations_dir: Path) -> list[dict]:
    """Load all translation payloads from disk for the given entry.

    Reads a JSONL file where each line is a JSON object representing
    a translation from a different model.
    """
    translation_path = translations_dir / f"entry_{entry_id}.jsonl"
    if not translation_path.exists():
        return []

    translations = []
    try:
        for line in translation_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                translations.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Skipping malformed line in entry {entry_id}: {exc}")
    except OSError as exc:
        print(f"Error reading translation file for entry {entry_id}: {exc}")
        return []

    return translations


def main() -> None:
    """Promote recorded conversation entries that have translations available."""

    parser = argparse.ArgumentParser(
        description="Apply stored translations to conversation entries."
    )
    parser.add_argument(
        "translations_dir",
        type=Path,
        help="Path to the directory containing translation JSONL files.",
    )
    args = parser.parse_args()
    translations_dir: Path = args.translations_dir

    with app.app_context():
        entries = ConversationEntry.query.filter(
            ConversationEntry.status == ENTRY_STATUS_RECORDED
        ).all()

        if not entries:
            print("No recorded conversation entries awaiting translation application.")
            return

        if not translations_dir.exists():
            print("Translations directory does not exist; nothing to apply.")
            return

        updated = 0

        for entry in entries:
            translations = load_translations(entry.id, translations_dir)
            if not translations:
                print(
                    f"Translation not found for entry {entry.id}; leaving in recorded state."
                )
                continue

            # Randomly select one translation from the available options
            selected = random.choice(translations)
            entry.translation_text = selected.get("translated_text")
            entry.translation_model = selected.get("model")
            entry.status = ENTRY_STATUS_TRANSLATED
            updated += 1
            print(
                f"Entry {entry.id}: selected translation from model '{entry.translation_model}'"
            )

        if not updated:
            print("No entries were updated.")
            return

        try:
            db.session.commit()
        except Exception as exc:  # noqa: BLE001 - surface for follow-up
            db.session.rollback()
            print(f"Failed to update entry statuses: {exc}")
            return

        print(f"Marked {updated} conversation entries as translated.")


if __name__ == "__main__":
    main()
