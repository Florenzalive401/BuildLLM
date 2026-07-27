from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator

import torch
from tqdm import tqdm

from src.tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode text or JSONL corpus into train and validation token files.")
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/fineweb_edu.txt"))
    parser.add_argument("--format", choices=("auto", "text", "jsonl"), default="auto")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tokenizer.json"))
    parser.add_argument("--output-directory", type=Path, default=Path("data/tokens"))
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--max-characters", type=int, default=None)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=1729)
    return parser.parse_args()


def detect_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    return "jsonl" if path.suffix.lower() in {".jsonl", ".json"} else "text"


def iter_documents(path: Path, format_name: str, text_field: str) -> Iterator[str]:
    if format_name == "jsonl":
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at line {line_number}: {error}") from error
                text = record.get(text_field)
                if isinstance(text, str) and text.strip():
                    yield text
        return

    current: list[str] = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                current.append(line.rstrip("\n"))
            elif current:
                yield "\n".join(current)
                current = []
        if current:
            yield "\n".join(current)


def is_validation_document(text: str, fraction: float, seed: int) -> bool:
    digest = hashlib.blake2b(
        text[:4096].encode("utf-8"),
        digest_size=8,
        person=seed.to_bytes(8, "little", signed=False),
    ).digest()
    value = int.from_bytes(digest, "big") / float(2**64)
    return value < fraction


def main() -> None:
    args = parse_args()
    if not args.corpus.exists():
        raise FileNotFoundError(f"Corpus does not exist: {args.corpus}")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")

    tokenizer = BPETokenizer(args.tokenizer)
    format_name = detect_format(args.corpus, args.format)
    training_chunks: list[torch.Tensor] = []
    validation_chunks: list[torch.Tensor] = []
    training_documents = validation_documents = characters = 0

    documents = iter_documents(args.corpus, format_name, args.text_field)
    for document_index, document in enumerate(tqdm(documents, desc="Encoding documents", unit="doc"), start=1):
        if args.max_documents is not None and document_index > args.max_documents:
            break
        if args.max_characters is not None and characters >= args.max_characters:
            break

        token_ids = tokenizer.encode(document, add_document_end=True)
        chunk = torch.tensor(token_ids, dtype=torch.int32)
        if is_validation_document(document, args.validation_fraction, args.split_seed):
            validation_chunks.append(chunk)
            validation_documents += 1
        else:
            training_chunks.append(chunk)
            training_documents += 1
        characters += len(document)

    if not training_chunks or not validation_chunks:
        raise ValueError("Corpus split must produce both training and validation documents")

    training_tokens = torch.cat(training_chunks)
    validation_tokens = torch.cat(validation_chunks)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    training_path = args.output_directory / "train_tokens.pt"
    validation_path = args.output_directory / "validation_tokens.pt"
    report_path = args.output_directory / "encoding_report.json"
    torch.save(training_tokens, training_path)
    torch.save(validation_tokens, validation_path)

    report = {
        "corpus": str(args.corpus),
        "format": format_name,
        "tokenizer": str(args.tokenizer),
        "vocabulary_size": tokenizer.vocabulary_size,
        "training_documents": training_documents,
        "validation_documents": validation_documents,
        "training_tokens": int(training_tokens.numel()),
        "validation_tokens": int(validation_tokens.numel()),
        "characters_encoded": characters,
        "split_seed": args.split_seed,
        "training_file": str(training_path),
        "validation_file": str(validation_path),
    }
    report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()
