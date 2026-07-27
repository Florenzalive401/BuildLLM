from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AttentionKVCache:
    key: Tensor
    value: Tensor

    def __post_init__(self) -> None:
        self._validate()

    @property
    def batch_size(self) -> int:
        return int(self.key.shape[0])

    @property
    def number_of_heads(self) -> int:
        return int(self.key.shape[1])

    @property
    def sequence_length(self) -> int:
        return int(self.key.shape[2])

    @property
    def head_dimension(self) -> int:
        return int(self.key.shape[3])

    def _validate(self) -> None:
        if not isinstance(self.key, Tensor):
            raise TypeError("key must be a torch.Tensor")

        if not isinstance(self.value, Tensor):
            raise TypeError("value must be a torch.Tensor")

        if self.key.ndim != 4:
            raise ValueError(
                "key must have shape batch_size, number_of_heads, "
                "sequence_length, head_dimension"
            )

        if self.value.ndim != 4:
            raise ValueError(
                "value must have shape batch_size, number_of_heads, "
                "sequence_length, head_dimension"
            )

        if self.key.shape != self.value.shape:
            raise ValueError("key and value must have matching shapes")

        if self.key.shape[0] <= 0:
            raise ValueError("batch_size must be greater than zero")

        if self.key.shape[1] <= 0:
            raise ValueError("number_of_heads must be greater than zero")

        if self.key.shape[2] <= 0:
            raise ValueError("sequence_length must be greater than zero")

        if self.key.shape[3] <= 0:
            raise ValueError("head_dimension must be greater than zero")

        if not self.key.is_floating_point():
            raise TypeError("key must use a floating point dtype")

        if not self.value.is_floating_point():
            raise TypeError("value must use a floating point dtype")

        if self.key.dtype != self.value.dtype:
            raise TypeError("key and value must use the same dtype")

        if self.key.device != self.value.device:
            raise ValueError("key and value must use the same device")


@dataclass(frozen=True)
class ModelKVCache:
    layers: tuple[AttentionKVCache, ...]

    def __post_init__(self) -> None:
        self._validate()

    @property
    def number_of_layers(self) -> int:
        return len(self.layers)

    @property
    def batch_size(self) -> int:
        return self.layers[0].batch_size

    @property
    def sequence_length(self) -> int:
        return self.layers[0].sequence_length

    def _validate(self) -> None:
        if not isinstance(self.layers, tuple):
            raise TypeError("layers must be a tuple")

        if not self.layers:
            raise ValueError("layers cannot be empty")

        first_layer = self.layers[0]

        for layer in self.layers:
            if not isinstance(layer, AttentionKVCache):
                raise TypeError(
                    "layers must contain AttentionKVCache instances"
                )

            if layer.batch_size != first_layer.batch_size:
                raise ValueError(
                    "all cache layers must have matching batch sizes"
                )

            if layer.sequence_length != first_layer.sequence_length:
                raise ValueError(
                    "all cache layers must have matching sequence lengths"
                )

            if layer.key.device != first_layer.key.device:
                raise ValueError(
                    "all cache layers must use the same device"
                )

    def validate_for_model(
        self,
        *,
        batch_size: int,
        number_of_layers: int,
        number_of_heads: int,
        head_dimension: int,
        maximum_sequence_length: int,
        device: torch.device,
    ) -> None:
        if self.batch_size != batch_size:
            raise ValueError(
                "cache batch_size does not match the input batch_size"
            )

        if self.number_of_layers != number_of_layers:
            raise ValueError(
                "cache layer count does not match the model"
            )

        if self.sequence_length > maximum_sequence_length:
            raise ValueError(
                "cache sequence_length exceeds maximum_sequence_length"
            )

        for layer in self.layers:
            if layer.number_of_heads != number_of_heads:
                raise ValueError(
                    "cache number_of_heads does not match the model"
                )

            if layer.head_dimension != head_dimension:
                raise ValueError(
                    "cache head_dimension does not match the model"
                )

            if layer.key.device != device:
                raise ValueError(
                    "cache device does not match the input device"
                )
