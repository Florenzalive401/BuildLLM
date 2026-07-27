from typing import Any

import torch
from torch import nn

from src.config import config


class FeedForwardNetwork(nn.Module):
    def __init__(self, model_config: Any = config) -> None:
        super().__init__()

        self.embedding_dimension = (
            model_config.embedding_dimension
        )

        self.feed_forward_dimension = (
            model_config.feed_forward_dimension
        )

        if self.embedding_dimension <= 0:
            raise ValueError(
                "embedding_dimension must be greater than zero"
            )

        if self.feed_forward_dimension <= 0:
            raise ValueError(
                "feed_forward_dimension must be greater than zero"
            )

        self.input_projection = nn.Linear(
            self.embedding_dimension,
            self.feed_forward_dimension,
        )

        self.activation = nn.GELU()

        self.output_projection = nn.Linear(
            self.feed_forward_dimension,
            self.embedding_dimension,
        )

        self.dropout = nn.Dropout(
            model_config.dropout
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        self._validate_input(hidden_states)

        hidden_states = self.input_projection(
            hidden_states
        )

        hidden_states = self.activation(
            hidden_states
        )

        hidden_states = self.output_projection(
            hidden_states
        )

        hidden_states = self.dropout(
            hidden_states
        )

        return hidden_states

    def _validate_input(
        self,
        hidden_states: torch.Tensor,
    ) -> None:

        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape "
                "batch_size, sequence_length, embedding_dimension"
            )

        if (
            hidden_states.shape[2]
            != self.embedding_dimension
        ):
            raise ValueError(
                "Input embedding dimension does not match "
                "the configured embedding_dimension"
            )