from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """Autoregressive language modeling windows from one token region."""

    def __init__(
        self,
        token_file: str | Path,
        sequence_length: int,
        *,
        stride: int = 1,
        maximum_examples: int | None = None,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero")

        if stride <= 0:
            raise ValueError("stride must be greater than zero")

        if maximum_examples is not None and maximum_examples < 0:
            raise ValueError("maximum_examples cannot be negative")

        if maximum_examples == 0:
            maximum_examples = None

        self.sequence_length = sequence_length
        self.stride = stride
        self.maximum_examples = maximum_examples
        self.token_file = Path(token_file)

        if not self.token_file.exists():
            raise FileNotFoundError(
                f"token file does not exist: {self.token_file}"
            )

        self.tokens = torch.load(
            self.token_file,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(self.tokens, torch.Tensor):
            raise TypeError("token file must contain a tensor")

        if self.tokens.ndim != 1:
            raise ValueError("Expected one dimensional token tensor.")

        if not self.tokens.dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("token tensor must use an integer dtype")

        if len(self.tokens) <= sequence_length:
            raise ValueError("Corpus is too small.")

        available_examples = (
            (len(self.tokens) - sequence_length - 1) // stride
        ) + 1

        self._length = (
            min(available_examples, maximum_examples)
            if maximum_examples is not None
            else available_examples
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError("dataset index is out of range")

        start = index * self.stride
        end = start + self.sequence_length

        input_ids = self.tokens[start:end]
        target_ids = self.tokens[start + 1 : end + 1]

        return input_ids.long(), target_ids.long()
