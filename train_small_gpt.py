from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.model_profiles import ModelProfile


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a configured BuildLLM GPT training run."
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/models/gpt_42m.json"),
        help="JSON model profile. CLI values below override profile defaults.",
    )
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=None)
    parser.add_argument("--sequence-length", type=positive_int, default=None)
    parser.add_argument(
        "--training-examples",
        type=nonnegative_int,
        default=0,
        help="Maximum training examples. Use 0 for the full dataset.",
    )
    parser.add_argument(
        "--validation-examples",
        type=nonnegative_int,
        default=0,
        help="Maximum validation examples. Use 0 for the full dataset.",
    )
    parser.add_argument("--embedding-dimension", type=positive_int, default=None)
    parser.add_argument("--layers", type=positive_int, default=None)
    parser.add_argument("--attention-heads", type=positive_int, default=None)
    parser.add_argument("--feed-forward-dimension", type=positive_int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--workers", type=nonnegative_int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16", "fp16"),
        default=None,
    )
    parser.add_argument(
        "--train-tokens",
        default="data/tokens/balanced/train_tokens.pt",
    )
    parser.add_argument(
        "--validation-tokens",
        default="data/tokens/balanced/validation_tokens.pt",
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizer/balanced_tokenizer.json",
    )
    parser.add_argument(
        "--checkpoint-directory",
        default="checkpoints/iteration_2_balanced",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from latest or a specified checkpoint path.",
    )
    return parser


def require_file(path_value: str | Path, label: str) -> None:
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def choose(override: object | None, profile_value: object) -> object:
    return profile_value if override is None else override


def main() -> int:
    args = build_parser().parse_args()

    require_file("train.py", "Training program")
    require_file(args.model_config, "Model configuration")
    require_file(args.train_tokens, "Training token file")
    require_file(args.validation_tokens, "Validation token file")
    require_file(args.tokenizer, "Tokenizer")

    profile = ModelProfile.load(args.model_config)
    batch_size = int(choose(args.batch_size, profile.batch_size))
    sequence_length = int(choose(args.sequence_length, profile.sequence_length))
    embedding_dimension = int(
        choose(args.embedding_dimension, profile.embedding_dimension)
    )
    layers = int(choose(args.layers, profile.layers))
    attention_heads = int(choose(args.attention_heads, profile.attention_heads))
    feed_forward_dimension = int(
        choose(args.feed_forward_dimension, profile.feed_forward_dimension)
    )
    learning_rate = float(choose(args.learning_rate, profile.learning_rate))
    weight_decay = float(choose(args.weight_decay, profile.weight_decay))
    precision = str(choose(args.precision, profile.precision))

    command = [
        sys.executable,
        "train.py",
        "--model-config",
        str(args.model_config),
        "--train-tokens",
        args.train_tokens,
        "--validation-tokens",
        args.validation_tokens,
        "--tokenizer",
        args.tokenizer,
        "--checkpoint-directory",
        args.checkpoint_directory,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(batch_size),
        "--sequence-length",
        str(sequence_length),
        "--embedding-dimension",
        str(embedding_dimension),
        "--layers",
        str(layers),
        "--attention-heads",
        str(attention_heads),
        "--feed-forward-dimension",
        str(feed_forward_dimension),
        "--learning-rate",
        str(learning_rate),
        "--weight-decay",
        str(weight_decay),
        "--workers",
        str(args.workers),
        "--device",
        args.device,
        "--precision",
        precision,
        "--training-examples",
        str(args.training_examples),
        "--validation-examples",
        str(args.validation_examples),
    ]

    if args.resume is not None:
        command.extend(["--resume", args.resume])

    print()
    print("Starting BuildLLM training")
    print(f"Model profile: {profile.name}")
    print(f"Model configuration: {args.model_config}")
    print(f"Checkpoint directory: {args.checkpoint_directory}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {sequence_length}")
    print(f"Embedding dimension: {embedding_dimension}")
    print(f"Layers: {layers}")
    print(f"Attention heads: {attention_heads}")
    print(f"Feed forward dimension: {feed_forward_dimension}")
    print("Training examples: " + ("all" if args.training_examples == 0 else str(args.training_examples)))
    print("Validation examples: " + ("all" if args.validation_examples == 0 else str(args.validation_examples)))
    print(f"Device: {args.device}")
    print(f"Precision: {precision}")
    print()

    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
