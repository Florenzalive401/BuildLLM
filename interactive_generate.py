from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import ModelConfig
from src.generation import GenerationConfig, Generator
from src.model import GPTModel
from src.tokenizer import BPETokenizer
from src.training.checkpoint import CheckpointManager

DEFAULT_STOP_TEXTS = (
    "\nCategory:",
    "\nthumb|",
    "\nFile:",
    "\nImage:",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive completion interface for a trained GPT checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tokenizer.json"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument(
        "--stop-text",
        action="append",
        default=[],
        help="Stop generation after a decoded text fragment appears. May be supplied more than once.",
    )
    parser.add_argument(
        "--allow-wiki-markup",
        action="store_true",
        help="Disable the default stop text filters for common Wikipedia artifacts.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


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
    device = resolve_device(args.device)
    tokenizer = BPETokenizer(args.tokenizer)
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
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = GPTModel(model_config).to(device)
    CheckpointManager(args.checkpoint.parent).load(
        checkpoint_path=args.checkpoint, model=model, map_location=device
    )
    generator = Generator(model, model_config)

    print("Interactive GPT completion")
    print("Enter a prompt. Type /quit to exit.")
    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"/quit", "/exit"}:
            break
        if not prompt:
            continue
        prompt_ids = tokenizer.encode(prompt)
        prompt_ids = prompt_ids[-(model_config.maximum_sequence_length - 1):]
        room = model_config.maximum_sequence_length - len(prompt_ids)
        result = generator.generate(
            torch.tensor([prompt_ids], dtype=torch.long, device=device),
            GenerationConfig(
                max_new_tokens=min(args.max_new_tokens, room),
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                stop_token_ids=(tokenizer.document_end_token_id,),
                stop_sequences=stop_sequences,
                random_seed=None,
                use_kv_cache=True,
            ),
        )
        generated_ids = trim_trailing_stop_sequence(
            result.sequences[0].tolist(),
            stop_sequences,
        )
        print("\n" + tokenizer.decode(generated_ids))


if __name__ == "__main__":
    main()
