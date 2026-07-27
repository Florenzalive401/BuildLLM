"""Train a byte level BPE tokenizer from text or JSONL corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["<|padding|>", "<|unknown|>", "<|document_end|>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the project tokenizer.")
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/fineweb_edu.txt"))
    parser.add_argument("--format", choices=("auto", "text", "jsonl"), default="auto")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output", type=Path, default=Path("tokenizer/tokenizer.json"))
    parser.add_argument("--vocabulary-size", type=int, default=32_768)
    parser.add_argument("--minimum-frequency", type=int, default=2)
    return parser.parse_args()


def corpus_format(path: Path, requested: str) -> str:
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


def create_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<|unknown|>"))
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def verify_tokenizer(tokenizer: Tokenizer) -> None:
    sample = "Artificial intelligence can support cybersecurity engineering and software development."
    encoding = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoding.ids)
    if decoded != sample:
        raise ValueError("Tokenizer verification failed")
    print(f"Verification token count: {len(encoding.ids)}")


def main() -> None:
    args = parse_args()
    if not args.corpus.exists() or args.corpus.stat().st_size == 0:
        raise FileNotFoundError(f"Corpus is missing or empty: {args.corpus}")
    if args.vocabulary_size < 256:
        raise ValueError("vocabulary_size must be at least 256")

    format_name = corpus_format(args.corpus, args.format)
    tokenizer = create_tokenizer()
    trainer = BpeTrainer(
        vocab_size=args.vocabulary_size,
        min_frequency=args.minimum_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"Training tokenizer from {args.corpus} ({format_name})")
    tokenizer.train_from_iterator(
        iter_documents(args.corpus, format_name, args.text_field),
        trainer=trainer,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output), pretty=True)
    verify_tokenizer(tokenizer)
    print(f"Saved tokenizer: {args.output}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")


if __name__ == "__main__":
    main()
