"""
Builds the first training corpus for our language model.

Rather than downloading one massive file, we stream documents from FineWeb Edu,
clean them, remove obvious junk, and write a local corpus that we control.

This gives us a reproducible dataset that we can later tokenize and train on.
"""

import json
import hashlib
from pathlib import Path

from tqdm import tqdm

from src.fineweb_source import stream_fineweb_documents
from src.text_cleaner import clean_document


TARGET_CHARACTERS = 50_000_000

OUTPUT_FILE = Path("data/processed/fineweb_edu.txt")
REPORT_FILE = Path("data/reports/fineweb_report.json")


def sha256(text: str) -> str:
    """Create a stable fingerprint for duplicate detection."""
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def main():

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    seen = set()

    documents_examined = 0
    documents_written = 0
    documents_rejected = 0
    duplicate_documents = 0
    characters_written = 0

    with OUTPUT_FILE.open("w", encoding="utf8") as outfile:

        progress = tqdm(
            total=TARGET_CHARACTERS,
            unit="chars",
            unit_scale=True,
            desc="Building Corpus",
        )

        for record in stream_fineweb_documents():

            documents_examined += 1

            text = clean_document(record["text"])

            if text is None:
                documents_rejected += 1
                continue

            fingerprint = sha256(text)

            if fingerprint in seen:
                duplicate_documents += 1
                continue

            seen.add(fingerprint)

            outfile.write(text)
            outfile.write("\n\n")

            documents_written += 1
            characters_written += len(text)

            progress.update(len(text))

            if characters_written >= TARGET_CHARACTERS:
                break

        progress.close()

    report = {
        "characters_written": characters_written,
        "documents_examined": documents_examined,
        "documents_written": documents_written,
        "documents_rejected": documents_rejected,
        "duplicate_documents": duplicate_documents,
    }

    with REPORT_FILE.open("w") as f:
        json.dump(report, f, indent=4)

    print()
    print("Corpus complete.")
    print(report)


if __name__ == "__main__":
    main()