"""Build, tokenize, and encode the 800 million character training corpus."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIG = Path("configs/corpus_800m.yaml")
DEFAULT_CORPUS = Path("data/processed/training_corpus_800m.jsonl")
DEFAULT_TOKENIZER = Path("tokenizer/800m_tokenizer.json")
DEFAULT_TOKEN_DIRECTORY = Path("data/tokens/800m")
DEFAULT_REPORT = Path("data/reports/training_corpus_800m_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--token-directory", type=Path, default=DEFAULT_TOKEN_DIRECTORY)
    parser.add_argument("--vocabulary-size", type=int, default=32_768)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--skip-encode", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\nRunning:")
    print(" ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def print_summary(
    corpus: Path,
    report: Path,
    tokenizer: Path,
    token_directory: Path,
) -> None:
    print("\n800M corpus preparation complete")
    if corpus.exists():
        print(f"Corpus: {corpus}")
        print(f"Corpus size: {corpus.stat().st_size / (1024 ** 3):.2f} GiB")
    if report.exists():
        print(f"Corpus report: {report}")
    if tokenizer.exists():
        print(f"Tokenizer: {tokenizer}")
    encoding_report = token_directory / "encoding_report.json"
    if encoding_report.exists():
        encoding_data = json.loads(
            encoding_report.read_text(encoding="utf8")
        )
        print(
            f"Training documents: "
            f"{encoding_data['training_documents']:,}"
        )
        print(
            f"Validation documents: "
            f"{encoding_data['validation_documents']:,}"
        )
        print(f"Training tokens: {encoding_data['training_tokens']:,}")
        print(
            f"Validation tokens: "
            f"{encoding_data['validation_tokens']:,}"
        )
        print(f"Training token file: {encoding_data['training_file']}")
        print(
            f"Validation token file: "
            f"{encoding_data['validation_file']}"
        )

    print("\nStart a fresh 18 epoch Iteration 3 run with:")
    print(
        "python run_lab.py `\n"
        "  --iteration 3 `\n"
        "  --device cuda `\n"
        "  --epochs 18 `\n"
        "  --training-examples 0 `\n"
        "  --validation-examples 0 `\n"
        f"  --train-tokens {token_directory / 'train_tokens.pt'} `\n"
        f"  --validation-tokens {token_directory / 'validation_tokens.pt'} `\n"
        f"  --tokenizer {tokenizer} `\n"
        "  --checkpoint-directory checkpoints/iteration_3_800m"
    )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")

    if not args.skip_build:
        if not args.config.is_file():
            raise FileNotFoundError(f"Corpus config not found: {args.config}")
        run([sys.executable, "build_training_corpus.py", "--config", str(args.config)])

    if not args.corpus.is_file():
        raise FileNotFoundError(
            f"Corpus not found: {args.corpus}. Build it first or correct --corpus."
        )

    if not args.skip_tokenizer:
        run([
            sys.executable,
            "train_tokenizer.py",
            "--corpus",
            str(args.corpus),
            "--format",
            "jsonl",
            "--vocabulary-size",
            str(args.vocabulary_size),
            "--output",
            str(args.tokenizer),
        ])
    elif not args.tokenizer.is_file():
        raise FileNotFoundError(
            f"Tokenizer not found: {args.tokenizer}. "
            "Remove --skip-tokenizer or correct --tokenizer."
        )

    if not args.skip_encode:
        if not args.tokenizer.is_file():
            raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
        run([
            sys.executable,
            "encode_corpus.py",
            "--corpus",
            str(args.corpus),
            "--format",
            "jsonl",
            "--tokenizer",
            str(args.tokenizer),
            "--output-directory",
            str(args.token_directory),
            "--validation-fraction",
            str(args.validation_fraction),
        ])

    print_summary(
        args.corpus,
        args.report,
        args.tokenizer,
        args.token_directory,
    )


if __name__ == "__main__":
    main()
