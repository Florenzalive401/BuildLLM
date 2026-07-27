from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


class LanguageModelLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                "label_smoothing must be between 0 and 1"
            )

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:

        self._validate(
            logits,
            targets,
        )

        vocabulary_size = logits.shape[-1]

        logits = logits.reshape(
            -1,
            vocabulary_size,
        )

        targets = targets.reshape(
            -1,
        )

        return F.cross_entropy(
            logits,
            targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

    @staticmethod
    def _validate(
        logits: Tensor,
        targets: Tensor,
    ) -> None:

        if logits.ndim != 3:
            raise ValueError(
                "logits must have shape "
                "(batch, sequence, vocabulary)"
            )

        if targets.ndim != 2:
            raise ValueError(
                "targets must have shape "
                "(batch, sequence)"
            )

        if logits.shape[:2] != targets.shape:
            raise ValueError(
                "logits and targets "
                "must have matching batch and sequence dimensions"
            )

        if targets.dtype != torch.long:
            raise TypeError(
                "targets must use torch.long"
            )