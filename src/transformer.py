from typing import Any

import torch
from torch import nn

from src.attention import CausalSelfAttention
from src.config import config
from src.feed_forward import FeedForwardNetwork
from src.kv_cache import AttentionKVCache


class TransformerBlock(nn.Module):
    """Pre LayerNorm GPT transformer block."""

    def __init__(
        self,
        model_config: Any = config,
    ) -> None:
        super().__init__()

        self.embedding_dimension = model_config.embedding_dimension

        if self.embedding_dimension <= 0:
            raise ValueError(
                "embedding_dimension must be greater than zero"
            )

        self.attention_layer_norm = nn.LayerNorm(
            self.embedding_dimension
        )
        self.attention = CausalSelfAttention(model_config)
        self.feed_forward_layer_norm = nn.LayerNorm(
            self.embedding_dimension
        )
        self.feed_forward = FeedForwardNetwork(model_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: AttentionKVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionKVCache]:
        self._validate_input(hidden_states)

        attention_input = self.attention_layer_norm(hidden_states)

        if use_cache:
            attention_output, updated_cache = self.attention(
                attention_input,
                cache=cache,
                use_cache=True,
            )
        else:
            attention_output = self.attention(
                attention_input,
                cache=cache,
                use_cache=False,
            )
            updated_cache = None

        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.feed_forward(
            self.feed_forward_layer_norm(hidden_states)
        )

        if use_cache:
            return hidden_states, updated_cache

        return hidden_states

    def _validate_input(
        self,
        hidden_states: torch.Tensor,
    ) -> None:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape batch_size, "
                "sequence_length, embedding_dimension"
            )

        if hidden_states.shape[2] != self.embedding_dimension:
            raise ValueError(
                "Input embedding dimension does not match "
                "the configured embedding_dimension"
            )
