from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingState:
    epoch: int = 0
    global_step: int = 0

    training_loss: float = 0.0
    validation_loss: float = 0.0
    best_validation_loss: float = float("inf")

    learning_rate: float = 0.0

    tokens_processed: int = 0
    examples_processed: int = 0

    elapsed_seconds: float = 0.0

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "TrainingState":
        return cls(**data)

    def update_learning_rate(
        self,
        learning_rate: float,
    ) -> None:
        if learning_rate < 0:
            raise ValueError(
                "learning_rate cannot be negative"
            )

        self.learning_rate = learning_rate

    def update_training_loss(
        self,
        loss: float,
    ) -> None:
        if loss < 0:
            raise ValueError(
                "training_loss cannot be negative"
            )

        self.training_loss = loss

    def update_validation_loss(
        self,
        loss: float,
    ) -> None:
        if loss < 0:
            raise ValueError(
                "validation_loss cannot be negative"
            )

        self.validation_loss = loss

        if loss < self.best_validation_loss:
            self.best_validation_loss = loss

    def increment_step(
        self,
        tokens: int,
        examples: int,
    ) -> None:
        if tokens < 0:
            raise ValueError(
                "tokens cannot be negative"
            )

        if examples < 0:
            raise ValueError(
                "examples cannot be negative"
            )

        self.global_step += 1
        self.tokens_processed += tokens
        self.examples_processed += examples

    def increment_epoch(
        self,
    ) -> None:
        self.epoch += 1