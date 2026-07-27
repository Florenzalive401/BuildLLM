"""
Global configuration for the language model.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    vocabulary_size: int = 16_384
    maximum_sequence_length: int = 256

    embedding_dimension: int = 256
    number_of_layers: int = 6
    number_of_attention_heads: int = 8
    feed_forward_dimension: int = 1_024

    dropout: float = 0.10
    layer_norm_epsilon: float = 1e-5
    bias: bool = True
    weight_tying: bool = True

    batch_size: int = 16
    learning_rate: float = 3e-4
    epochs: int = 10
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    use_gradient_checkpointing: bool = False

    device: str = "cpu"
    random_seed: int = 42

    checkpoint_directory: str = "checkpoints"
    checkpoint_frequency: int = 1
    run_directory: str = "runs"

    generation_temperature: float = 1.0
    generation_top_k: int = 50
    generation_top_p: float = 0.95
    generation_max_tokens: int = 128

    @property
    def attention_head_dimension(self) -> int:
        return (
            self.embedding_dimension
            // self.number_of_attention_heads
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        positive_integer_fields = {
            "vocabulary_size": self.vocabulary_size,
            "maximum_sequence_length": self.maximum_sequence_length,
            "embedding_dimension": self.embedding_dimension,
            "number_of_layers": self.number_of_layers,
            "number_of_attention_heads": (
                self.number_of_attention_heads
            ),
            "feed_forward_dimension": (
                self.feed_forward_dimension
            ),
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "checkpoint_frequency": (
                self.checkpoint_frequency
            ),
            "generation_max_tokens": (
                self.generation_max_tokens
            ),
        }

        for field_name, value in positive_integer_fields.items():
            if value <= 0:
                raise ValueError(
                    f"{field_name} must be greater than zero"
                )

        if (
            self.embedding_dimension
            % self.number_of_attention_heads
            != 0
        ):
            raise ValueError(
                "embedding_dimension must be divisible by "
                "number_of_attention_heads"
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must be greater than or equal to zero "
                "and less than one"
            )

        if self.layer_norm_epsilon <= 0:
            raise ValueError(
                "layer_norm_epsilon must be greater than zero"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero"
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative"
            )

        if self.gradient_clip_norm <= 0:
            raise ValueError(
                "gradient_clip_norm must be greater than zero"
            )

        if self.generation_temperature <= 0:
            raise ValueError(
                "generation_temperature must be greater than zero"
            )

        if self.generation_top_k < 0:
            raise ValueError(
                "generation_top_k cannot be negative"
            )

        if not 0.0 < self.generation_top_p <= 1.0:
            raise ValueError(
                "generation_top_p must be greater than zero "
                "and less than or equal to one"
            )


config = ModelConfig()
config.validate()