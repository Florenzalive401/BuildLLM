from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.config import config
from src.model import GPTModel
from src.model_profiles import ModelProfile


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a BuildLLM model profile.")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--vocabulary-size", type=int, default=32_768)
    args = parser.parse_args()

    profile = ModelProfile.load(args.model_config)
    model_config = replace(
        config,
        vocabulary_size=args.vocabulary_size,
        maximum_sequence_length=profile.sequence_length,
        embedding_dimension=profile.embedding_dimension,
        number_of_layers=profile.layers,
        number_of_attention_heads=profile.attention_heads,
        feed_forward_dimension=profile.feed_forward_dimension,
        batch_size=profile.batch_size,
        learning_rate=profile.learning_rate,
        weight_decay=profile.weight_decay,
    )
    model_config.validate()
    model = GPTModel(model_config)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print(f"Model: {profile.name}")
    print(f"Configuration: {args.model_config}")
    print(f"Parameters: {parameters:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Layers: {profile.layers}")
    print(f"Embedding dimension: {profile.embedding_dimension}")
    print(f"Attention heads: {profile.attention_heads}")
    print(f"Head dimension: {profile.embedding_dimension // profile.attention_heads}")
    print(f"Feed forward dimension: {profile.feed_forward_dimension}")
    print(f"Sequence length: {profile.sequence_length}")
    print(f"Default batch size: {profile.batch_size}")
    print(f"Default precision: {profile.precision}")


if __name__ == "__main__":
    main()
