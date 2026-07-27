from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch import nn

from src.loss import LanguageModelLoss


@dataclass(frozen=True)
class ValidationResult:
    average_loss: float
    perplexity: float
    batch_count: int
    examples_processed: int
    tokens_processed: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "average_loss": self.average_loss,
            "perplexity": self.perplexity,
            "batch_count": self.batch_count,
            "examples_processed": self.examples_processed,
            "tokens_processed": self.tokens_processed,
            "elapsed_seconds": self.elapsed_seconds,
        }


class Validator:
    def __init__(
        self,
        loss_function: LanguageModelLoss,
        device: str | torch.device,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        if not isinstance(
            loss_function,
            LanguageModelLoss,
        ):
            raise TypeError(
                "loss_function must be a LanguageModelLoss"
            )

        self.loss_function = loss_function
        self.device = torch.device(
            device
        )
        self.autocast_dtype = autocast_dtype

    def validate(
        self,
        model: nn.Module,
        data_loader: Iterable[
            tuple[Tensor, Tensor]
        ],
    ) -> ValidationResult:
        if not isinstance(
            model,
            nn.Module,
        ):
            raise TypeError(
                "model must be a torch module"
            )

        if data_loader is None:
            raise TypeError(
                "data_loader cannot be None"
            )

        was_training = model.training

        total_loss = 0.0
        total_tokens = 0
        total_examples = 0
        batch_count = 0

        start_time = time.perf_counter()

        try:
            model.eval()

            with torch.inference_mode():
                for batch in data_loader:
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

                    if loss.ndim != 0:
                        raise ValueError(
                            "loss_function must return "
                            "a scalar tensor"
                        )

                    if not torch.isfinite(
                        loss
                    ):
                        raise FloatingPointError(
                            "validation loss is not finite"
                        )

                    token_count = (
                        self._count_valid_tokens(
                            target_ids
                        )
                    )

                    if token_count == 0:
                        continue

                    total_loss += (
                        float(loss.item())
                        * token_count
                    )

                    total_tokens += token_count

                    total_examples += int(
                        input_ids.size(0)
                    )

                    batch_count += 1

        finally:
            model.train(
                was_training
            )

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        if batch_count == 0:
            raise ValueError(
                "validation data loader produced "
                "no usable batches"
            )

        if total_tokens == 0:
            raise ValueError(
                "validation data loader produced "
                "no valid target tokens"
            )

        average_loss = (
            total_loss
            / total_tokens
        )

        perplexity = self._calculate_perplexity(
            average_loss
        )

        return ValidationResult(
            average_loss=average_loss,
            perplexity=perplexity,
            batch_count=batch_count,
            examples_processed=total_examples,
            tokens_processed=total_tokens,
            elapsed_seconds=elapsed_seconds,
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
    def _validate_batch(
        batch: object,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(
            batch,
            (tuple, list),
        ):
            raise TypeError(
                "each validation batch must be "
                "a tuple or list"
            )

        if len(batch) != 2:
            raise ValueError(
                "each validation batch must contain "
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
                "input_ids must use torch.long dtype"
            )

        if target_ids.dtype != torch.long:
            raise TypeError(
                "target_ids must use torch.long dtype"
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
                "model logits must use a floating "
                "point dtype"
            )

    @staticmethod
    def _calculate_perplexity(
        average_loss: float,
    ) -> float:
        try:
            return math.exp(
                average_loss
            )

        except OverflowError:
            return math.inf