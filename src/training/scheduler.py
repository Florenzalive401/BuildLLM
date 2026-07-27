from __future__ import annotations

import math
from dataclasses import dataclass

from torch.optim import Optimizer


@dataclass(frozen=True)
class SchedulerConfig:
    scheduler_type: str = "cosine"
    warmup_steps: int = 0
    maximum_training_steps: int = 100_000
    minimum_learning_rate: float = 0.0

    def validate(
        self,
        base_learning_rate: float,
    ) -> None:
        valid_scheduler_types = {
            "constant",
            "linear",
            "cosine",
        }

        if self.scheduler_type not in valid_scheduler_types:
            raise ValueError(
                "scheduler_type must be constant, linear, or cosine"
            )

        if base_learning_rate <= 0:
            raise ValueError(
                "base_learning_rate must be greater than zero"
            )

        if self.warmup_steps < 0:
            raise ValueError(
                "warmup_steps cannot be negative"
            )

        if self.maximum_training_steps <= 0:
            raise ValueError(
                "maximum_training_steps must be greater than zero"
            )

        if self.warmup_steps >= self.maximum_training_steps:
            raise ValueError(
                "warmup_steps must be less than maximum_training_steps"
            )

        if self.minimum_learning_rate < 0:
            raise ValueError(
                "minimum_learning_rate cannot be negative"
            )

        if self.minimum_learning_rate > base_learning_rate:
            raise ValueError(
                "minimum_learning_rate cannot exceed base_learning_rate"
            )


class LearningRateScheduler:
    def __init__(
        self,
        optimizer: Optimizer,
        scheduler_config: SchedulerConfig,
    ) -> None:
        if not optimizer.param_groups:
            raise ValueError(
                "optimizer must contain at least one parameter group"
            )

        self.optimizer = optimizer
        self.scheduler_config = scheduler_config
        self.current_step = 0

        self.base_learning_rates = [
            float(parameter_group["lr"])
            for parameter_group in optimizer.param_groups
        ]

        for learning_rate in self.base_learning_rates:
            scheduler_config.validate(
                learning_rate
            )

        self._apply_learning_rates(
            self._learning_rates_for_step(
                self.current_step
            )
        )

    @property
    def learning_rate(
        self,
    ) -> float:
        return float(
            self.optimizer.param_groups[0]["lr"]
        )

    def step(
        self,
    ) -> float:
        self.current_step += 1

        learning_rates = self._learning_rates_for_step(
            self.current_step
        )

        self._apply_learning_rates(
            learning_rates
        )

        return self.learning_rate

    def state_dict(
        self,
    ) -> dict[str, int]:
        return {
            "current_step": self.current_step,
        }

    def load_state_dict(
        self,
        state_dict: dict[str, int],
    ) -> None:
        if "current_step" not in state_dict:
            raise KeyError(
                "state_dict must contain current_step"
            )

        current_step = state_dict["current_step"]

        if not isinstance(current_step, int):
            raise TypeError(
                "current_step must be an integer"
            )

        if current_step < 0:
            raise ValueError(
                "current_step cannot be negative"
            )

        self.current_step = current_step

        self._apply_learning_rates(
            self._learning_rates_for_step(
                self.current_step
            )
        )

    def _learning_rates_for_step(
        self,
        step: int,
    ) -> list[float]:
        return [
            self._calculate_learning_rate(
                base_learning_rate,
                step,
            )
            for base_learning_rate in self.base_learning_rates
        ]

    def _calculate_learning_rate(
        self,
        base_learning_rate: float,
        step: int,
    ) -> float:
        if (
            self.scheduler_config.warmup_steps > 0
            and step < self.scheduler_config.warmup_steps
        ):
            return (
                base_learning_rate
                * (step + 1)
                / self.scheduler_config.warmup_steps
            )

        if self.scheduler_config.scheduler_type == "constant":
            return base_learning_rate

        if step >= self.scheduler_config.maximum_training_steps:
            return self.scheduler_config.minimum_learning_rate

        decay_steps = (
            self.scheduler_config.maximum_training_steps
            - self.scheduler_config.warmup_steps
        )

        decay_progress = (
            step
            - self.scheduler_config.warmup_steps
        ) / decay_steps

        decay_progress = min(
            max(decay_progress, 0.0),
            1.0,
        )

        if self.scheduler_config.scheduler_type == "linear":
            multiplier = 1.0 - decay_progress

        elif self.scheduler_config.scheduler_type == "cosine":
            multiplier = 0.5 * (
                1.0
                + math.cos(
                    math.pi * decay_progress
                )
            )

        else:
            raise RuntimeError(
                "unsupported scheduler type"
            )

        return (
            self.scheduler_config.minimum_learning_rate
            + (
                base_learning_rate
                - self.scheduler_config.minimum_learning_rate
            )
            * multiplier
        )

    def _apply_learning_rates(
        self,
        learning_rates: list[float],
    ) -> None:
        if len(learning_rates) != len(
            self.optimizer.param_groups
        ):
            raise ValueError(
                "learning rate count does not match optimizer groups"
            )

        for parameter_group, learning_rate in zip(
            self.optimizer.param_groups,
            learning_rates,
            strict=True,
        ):
            parameter_group["lr"] = learning_rate