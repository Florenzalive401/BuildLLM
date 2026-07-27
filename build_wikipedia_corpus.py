"""Build a clean, reproducible Wikipedia corpus from an official Wikimedia dump."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tqdm import tqdm

from src.text_cleaner import clean_document
from src.wikipedia_source import (
    download_dump,
    expected_md5,
    stream_articles,
    verify_md5,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract Wikipedia article text.")
    parser.add_argument("--project", default="enwiki", help="Wikimedia project, for example enwiki or simplewiki.")
    parser.add_argument("--dump", type=Path, default=None, help="Use an existing pages articles XML BZ2 dump.")
    parser.add_argument("--download-directory", type=Path, default=Path("data/raw/wikipedia"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/wikipedia_en.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/wikipedia_en_report.json"))
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--max-characters", type=int, default=None)
    parser.add_argument("--minimum-characters", type=int, default=500)
    parser.add_argument("--verify-checksum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def fingerprint(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    if args.max_articles is not None and args.max_articles <= 0:
        raise ValueError("max_articles must be greater than zero")
    if args.max_characters is not None and args.max_characters <= 0:
        raise ValueError("max_characters must be greater than zero")

    if args.dump is None:
        dump_path = args.download_directory / f"{args.project}-latest-pages-articles-multistream.xml.bz2"
        dump_path = download_dump(dump_path, project=args.project)
    else:
        dump_path = args.dump
        if not dump_path.exists():
            raise FileNotFoundError(f"Wikipedia dump not found: {dump_path}")

    if args.verify_checksum and args.dump is None:
        checksum = expected_md5(args.project)
        if checksum is not None:
            print("Verifying Wikimedia checksum. This reads the entire compressed dump once.")
            if not verify_md5(dump_path, checksum):
                raise IOError("Wikipedia dump checksum verification failed")

    if args.download_only:
        print(f"Downloaded: {dump_path}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    examined = written = rejected = duplicates = characters = 0
    seen: set[str] = set()

    with args.output.open("w", encoding="utf-8") as output, tqdm(
        unit="article", desc=f"Extracting {args.project}"
    ) as progress:
        for article in stream_articles(dump_path):
            examined += 1
            cleaned = clean_document(article.text)
            if cleaned is None or len(cleaned) < args.minimum_characters:
                rejected += 1
                continue

            digest = fingerprint(cleaned)
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)

            record = {
                "source": args.project,
                "page_id": article.page_id,
                "title": article.title,
                "text": cleaned,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            characters += len(cleaned)
            progress.update(1)
            progress.set_postfix(chars=f"{characters:,}", rejected=rejected)

            if args.max_articles is not None and written >= args.max_articles:
                break
            if args.max_characters is not None and characters >= args.max_characters:
                break

    report = {
        "project": args.project,
        "dump": str(dump_path),
        "output": str(args.output),
        "articles_examined": examined,
        "articles_written": written,
        "articles_rejected": rejected,
        "exact_duplicates": duplicates,
        "characters_written": characters,
        "complete_dump_extracted": args.max_articles is None and args.max_characters is None,
    }
    args.report.write_text(json.dumps(report, indent=4), encoding="utf-8")
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()
