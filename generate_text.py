from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import ModelConfig
from src.generation import GenerationConfig, Generator
from src.model import GPTModel
from src.runtime import resolve_device
from src.tokenizer import BPETokenizer
from src.training.checkpoint import CheckpointManager

DEFAULT_STOP_TEXTS = (
    "\nCategory:",
    "\nthumb|",
    "\nFile:",
    "\nImage:",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained checkpoint and generate text."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/default_run/best_checkpoint.pt"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("tokenizer/tokenizer.json"),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Artificial intelligence",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.15,
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--stop-text",
        action="append",
        default=[],
        help=(
            "Stop generation after a decoded text fragment appears. "
            "May be supplied more than once."
        ),
    )
    parser.add_argument(
        "--allow-wiki-markup",
        action="store_true",
        help=(
            "Disable the default stop text filters for common "
            "Wikipedia artifacts."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def build_stop_sequences(
    tokenizer: BPETokenizer,
    *,
    stop_texts: list[str],
) -> tuple[tuple[int, ...], ...]:
    encoded_sequences: list[tuple[int, ...]] = []

    for stop_text in stop_texts:
        token_ids = tokenizer.encode(stop_text)

        if not token_ids:
            raise ValueError(
                f"stop text produced no tokens: {stop_text!r}"
            )

        encoded_sequences.append(tuple(token_ids))

    return tuple(dict.fromkeys(encoded_sequences))


def trim_trailing_stop_sequence(
    token_ids: list[int],
    stop_sequences: tuple[tuple[int, ...], ...],
) -> list[int]:
    for stop_sequence in stop_sequences:
        if len(token_ids) < len(stop_sequence):
            continue

        if tuple(token_ids[-len(stop_sequence) :]) == stop_sequence:
            return token_ids[: -len(stop_sequence)]

    return token_ids


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint does not exist: {args.checkpoint}"
        )

    tokenizer = BPETokenizer(args.tokenizer)
    device = resolve_device(args.device)
    stop_texts = list(args.stop_text)

    if not args.allow_wiki_markup:
        stop_texts = [
            *DEFAULT_STOP_TEXTS,
            *stop_texts,
        ]

    stop_sequences = build_stop_sequences(
        tokenizer,
        stop_texts=stop_texts,
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model_config = ModelConfig(**checkpoint["model_config"])
    model_config.validate()

    if tokenizer.vocabulary_size != model_config.vocabulary_size:
        raise ValueError(
            "tokenizer vocabulary size does not match checkpoint configuration"
        )

    model = GPTModel(model_config).to(device)
    checkpoint_manager = CheckpointManager(args.checkpoint.parent)
    checkpoint_manager.load(
        checkpoint_path=args.checkpoint,
        model=model,
        map_location=device,
    )

    prompt_ids = tokenizer.encode(args.prompt)

    if not prompt_ids:
        raise ValueError("prompt produced no tokens")

    maximum_prompt_length = model_config.maximum_sequence_length - 1
    prompt_ids = prompt_ids[-maximum_prompt_length:]
    available_new_tokens = (
        model_config.maximum_sequence_length - len(prompt_ids)
    )
    max_new_tokens = min(args.max_new_tokens, available_new_tokens)

    if max_new_tokens <= 0:
        raise ValueError("prompt leaves no room for generated tokens")

    prompt_tensor = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    generator = Generator(model, model_config)
    result = generator.generate(
        prompt_tensor,
        GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=not args.greedy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_token_ids=(tokenizer.document_end_token_id,),
            stop_sequences=stop_sequences,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            random_seed=args.seed,
            use_kv_cache=True,
        ),
    )

    result_token_ids = trim_trailing_stop_sequence(
        result.sequences[0].tolist(),
        stop_sequences,
    )
    generated_text = tokenizer.decode(result_token_ids)

    print()
    print("Generated Text")
    print("=" * 70)
    print(generated_text)
    print()
    print(f"Finish reason: {result.finish_reasons[0]}")
    print(f"Generated tokens: {result.generated_lengths[0]}")


if __name__ == "__main__":
    main()
