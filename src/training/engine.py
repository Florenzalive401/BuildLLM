from __future__ import annotations

import math
import time
from dataclasses import dataclass
from collections.abc import Callable
from typing import Iterable

import torch
from torch import Tensor
from torch import nn
from torch.optim import Optimizer

from src.loss import LanguageModelLoss
from src.training.scheduler import (
    LearningRateScheduler,
)
from src.training_state import TrainingState


@dataclass(frozen=True)
class TrainingStepResult:
    loss: float
    gradient_norm: float
    learning_rate: float
    examples_processed: int
    tokens_processed: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "loss": self.loss,
            "gradient_norm": self.gradient_norm,
            "learning_rate": self.learning_rate,
            "examples_processed": (
                self.examples_processed
            ),
            "tokens_processed": (
                self.tokens_processed
            ),
            "elapsed_seconds": (
                self.elapsed_seconds
            ),
        }


@dataclass(frozen=True)
class TrainingEpochResult:
    average_loss: float
    average_gradient_norm: float
    ending_learning_rate: float
    batch_count: int
    examples_processed: int
    tokens_processed: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "average_loss": self.average_loss,
            "average_gradient_norm": (
                self.average_gradient_norm
            ),
            "ending_learning_rate": (
                self.ending_learning_rate
            ),
            "batch_count": self.batch_count,
            "examples_processed": (
                self.examples_processed
            ),
            "tokens_processed": (
                self.tokens_processed
            ),
            "elapsed_seconds": (
                self.elapsed_seconds
            ),
        }


class TrainingEngine:
    def __init__(
        self,
        loss_function: LanguageModelLoss,
        device: str | torch.device,
        gradient_clip_norm: float | None = None,
        autocast_dtype: torch.dtype | None = None,
        use_gradient_scaler: bool = False,
        step_callbacks: Iterable[Callable[[TrainingStepResult, TrainingState], None]] | None = None,
    ) -> None:
        if not isinstance(
            loss_function,
            LanguageModelLoss,
        ):
            raise TypeError(
                "loss_function must be a "
                "LanguageModelLoss"
            )

        if (
            gradient_clip_norm is not None
            and gradient_clip_norm <= 0
        ):
            raise ValueError(
                "gradient_clip_norm must be "
                "greater than zero"
            )

        self.loss_function = loss_function
        self.device = torch.device(
            device
        )
        self.gradient_clip_norm = (
            gradient_clip_norm
        )
        self.autocast_dtype = autocast_dtype
        self.use_gradient_scaler = bool(use_gradient_scaler)
        if self.use_gradient_scaler and self.device.type != "cuda":
            raise ValueError("gradient scaling requires a CUDA device")
        if self.use_gradient_scaler and self.autocast_dtype != torch.float16:
            raise ValueError("gradient scaling is only used with FP16 autocast")
        self.gradient_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_gradient_scaler,
        )
        self.step_callbacks = tuple(step_callbacks or ())

        for callback in self.step_callbacks:
            if not callable(callback):
                raise TypeError("every step callback must be callable")

    def train_step(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LearningRateScheduler,
        training_state: TrainingState,
        batch: tuple[Tensor, Tensor]
        | list[Tensor],
    ) -> TrainingStepResult:
        self._validate_components(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
        )

        input_ids, target_ids = (
            self._validate_batch(
                batch
            )
        )

        input_ids = input_ids.to(
            self.device,
            non_blocking=True,
        )

        target_ids = target_ids.to(
            self.device,
            non_blocking=True,
        )

        start_time = time.perf_counter()

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None,
        ):
            logits = model(
                input_ids
            )

            self._validate_logits(
                logits=logits,
                target_ids=target_ids,
            )

            loss = self.loss_function(
                logits,
                target_ids,
            )

        self._validate_loss(
            loss
        )

        if self.use_gradient_scaler:
            self.gradient_scaler.scale(loss).backward()
            self.gradient_scaler.unscale_(optimizer)
        else:
            loss.backward()

        gradient_norm = (
            self._process_gradients(
                model
            )
        )

        if self.use_gradient_scaler:
            self.gradient_scaler.step(optimizer)
            self.gradient_scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        token_count = (
            self._count_valid_tokens(
                target_ids
            )
        )

        example_count = int(
            input_ids.size(0)
        )

        loss_value = float(
            loss.detach().item()
        )

        learning_rate = float(
            scheduler.learning_rate
        )

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        training_state.increment_step(
            tokens=token_count,
            examples=example_count,
        )

        training_state.update_training_loss(
            loss_value
        )

        training_state.update_learning_rate(
            learning_rate
        )

        training_state.elapsed_seconds += (
            elapsed_seconds
        )

        result = TrainingStepResult(
            loss=loss_value,
            gradient_norm=gradient_norm,
            learning_rate=learning_rate,
            examples_processed=example_count,
            tokens_processed=token_count,
            elapsed_seconds=elapsed_seconds,
        )

        for callback in self.step_callbacks:
            callback(result, training_state)

        return result

    def train_epoch(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LearningRateScheduler,
        training_state: TrainingState,
        data_loader: Iterable[
            tuple[Tensor, Tensor]
        ],
    ) -> TrainingEpochResult:
        self._validate_components(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
        )

        if data_loader is None:
            raise TypeError(
                "data_loader cannot be None"
            )

        start_time = time.perf_counter()

        total_weighted_loss = 0.0
        total_gradient_norm = 0.0
        total_examples = 0
        total_tokens = 0
        batch_count = 0

        for batch in data_loader:
            result = self.train_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                training_state=training_state,
                batch=batch,
            )

            total_weighted_loss += (
                result.loss
                * result.tokens_processed
            )

            total_gradient_norm += (
                result.gradient_norm
            )

            total_examples += (
                result.examples_processed
            )

            total_tokens += (
                result.tokens_processed
            )

            batch_count += 1

        if batch_count == 0:
            raise ValueError(
                "training data loader produced "
                "no batches"
            )

        if total_tokens == 0:
            raise ValueError(
                "training data loader produced "
                "no valid target tokens"
            )

        average_loss = (
            total_weighted_loss
            / total_tokens
        )

        average_gradient_norm = (
            total_gradient_norm
            / batch_count
        )

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        training_state.increment_epoch()

        training_state.update_training_loss(
            average_loss
        )

        return TrainingEpochResult(
            average_loss=average_loss,
            average_gradient_norm=(
                average_gradient_norm
            ),
            ending_learning_rate=float(
                scheduler.learning_rate
            ),
            batch_count=batch_count,
            examples_processed=total_examples,
            tokens_processed=total_tokens,
            elapsed_seconds=elapsed_seconds,
        )

    def _process_gradients(
        self,
        model: nn.Module,
    ) -> float:
        if self.gradient_clip_norm is not None:
            gradient_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=(
                        self.gradient_clip_norm
                    ),
                    error_if_nonfinite=True,
                )
            )

            gradient_norm_value = float(
                gradient_norm.item()
                if isinstance(
                    gradient_norm,
                    Tensor,
                )
                else gradient_norm
            )

        else:
            gradient_norm_value = (
                self._calculate_gradient_norm(
                    model
                )
            )

        if not math.isfinite(
            gradient_norm_value
        ):
            raise FloatingPointError(
                "gradient norm is not finite"
            )

        return gradient_norm_value

    @staticmethod
    def _calculate_gradient_norm(
        model: nn.Module,
    ) -> float:
        squared_norm = 0.0
        gradient_found = False

        for parameter in model.parameters():
            if parameter.grad is None:
                continue

            gradient_found = True

            parameter_norm = (
                parameter.grad.detach().norm(
                    p=2
                )
            )

            squared_norm += float(
                parameter_norm.item()
            ) ** 2

        if not gradient_found:
            raise RuntimeError(
                "no gradients were produced"
            )

        return math.sqrt(
            squared_norm
        )

    def _count_valid_tokens(
        self,
        target_ids: Tensor,
    ) -> int:
        ignore_index = (
            self.loss_function.ignore_index
        )

        if ignore_index is None:
            return int(
                target_ids.numel()
            )

        return int(
            target_ids.ne(
                ignore_index
            ).sum().item()
        )

    @staticmethod
    def _validate_components(
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LearningRateScheduler,
        training_state: TrainingState,
    ) -> None:
        if not isinstance(
            model,
            nn.Module,
        ):
            raise TypeError(
                "model must be a torch module"
            )

        if not isinstance(
            optimizer,
            Optimizer,
        ):
            raise TypeError(
                "optimizer must be a "
                "torch optimizer"
            )

        if not isinstance(
            scheduler,
            LearningRateScheduler,
        ):
            raise TypeError(
                "scheduler must be a "
                "LearningRateScheduler"
            )

        if not isinstance(
            training_state,
            TrainingState,
        ):
            raise TypeError(
                "training_state must be a "
                "TrainingState"
            )

    @staticmethod
    def _validate_batch(
        batch: object,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(
            batch,
            (tuple, list),
        ):
            raise TypeError(
                "each training batch must be "
                "a tuple or list"
            )

        if len(batch) != 2:
            raise ValueError(
                "each training batch must contain "
                "input_ids and target_ids"
            )

        input_ids = batch[0]
        target_ids = batch[1]

        if not isinstance(
            input_ids,
            Tensor,
        ):
            raise TypeError(
                "input_ids must be a tensor"
            )

        if not isinstance(
            target_ids,
            Tensor,
        ):
            raise TypeError(
                "target_ids must be a tensor"
            )

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape "
                "[batch_size, sequence_length]"
            )

        if target_ids.ndim != 2:
            raise ValueError(
                "target_ids must have shape "
                "[batch_size, sequence_length]"
            )

        if input_ids.shape != target_ids.shape:
            raise ValueError(
                "input_ids and target_ids must "
                "have matching shapes"
            )

        if input_ids.numel() == 0:
            raise ValueError(
                "input_ids cannot be empty"
            )

        if target_ids.numel() == 0:
            raise ValueError(
                "target_ids cannot be empty"
            )

        if input_ids.dtype != torch.long:
            raise TypeError(
                "input_ids must use "
                "torch.long dtype"
            )

        if target_ids.dtype != torch.long:
            raise TypeError(
                "target_ids must use "
                "torch.long dtype"
            )

        return (
            input_ids,
            target_ids,
        )

    @staticmethod
    def _validate_logits(
        logits: object,
        target_ids: Tensor,
    ) -> None:
        if not isinstance(
            logits,
            Tensor,
        ):
            raise TypeError(
                "model must return a tensor"
            )

        if logits.ndim != 3:
            raise ValueError(
                "model logits must have shape "
                "[batch_size, sequence_length, "
                "vocabulary_size]"
            )

        if logits.size(0) != target_ids.size(0):
            raise ValueError(
                "logits and target_ids must have "
                "matching batch sizes"
            )

        if logits.size(1) != target_ids.size(1):
            raise ValueError(
                "logits and target_ids must have "
                "matching sequence lengths"
            )

        if logits.size(2) <= 0:
            raise ValueError(
                "logits vocabulary dimension "
                "must be greater than zero"
            )

        if not logits.is_floating_point():
            raise TypeError(
                "model logits must use a "
                "floating point dtype"
            )

    @staticmethod
    def _validate_loss(
        loss: object,
    ) -> None:
        if not isinstance(
            loss,
            Tensor,
        ):
            raise TypeError(
                "loss_function must return "
                "a tensor"
            )

        if loss.ndim != 0:
            raise ValueError(
                "loss_function must return "
                "a scalar tensor"
            )

        if not torch.isfinite(
            loss
        ):
            raise FloatingPointError(
                "training loss is not finite"
            )