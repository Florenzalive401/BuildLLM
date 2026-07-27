import math
from typing import Any

import torch
from torch import nn

from src.config import config
from src.kv_cache import AttentionKVCache


class CausalSelfAttention(nn.Module):
    def __init__(self, model_config: Any = config) -> None:
        super().__init__()

        self.embedding_dimension = model_config.embedding_dimension
        self.number_of_attention_heads = (
            model_config.number_of_attention_heads
        )
        self.maximum_sequence_length = (
            model_config.maximum_sequence_length
        )

        if self.embedding_dimension <= 0:
            raise ValueError(
                "embedding_dimension must be greater than zero"
            )

        if self.number_of_attention_heads <= 0:
            raise ValueError(
                "number_of_attention_heads must be greater than zero"
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

        self.head_dimension = (
            self.embedding_dimension
            // self.number_of_attention_heads
        )

        self.query_key_value_projection = nn.Linear(
            self.embedding_dimension,
            self.embedding_dimension * 3,
            bias=False,
        )

        self.output_projection = nn.Linear(
            self.embedding_dimension,
            self.embedding_dimension,
            bias=False,
        )

        self.attention_dropout = nn.Dropout(model_config.dropout)
        self.output_dropout = nn.Dropout(model_config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: AttentionKVCache | None = None,
        use_cache: bool = False,
        return_attention: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, AttentionKVCache]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            AttentionKVCache,
        ]
    ):
        self._validate_input(hidden_states)

        batch_size, query_length, _ = hidden_states.shape
        past_length = 0

        if cache is not None:
            self._validate_cache(
                cache=cache,
                batch_size=batch_size,
                hidden_states=hidden_states,
            )
            past_length = cache.sequence_length

        total_sequence_length = past_length + query_length

        if total_sequence_length > self.maximum_sequence_length:
            raise ValueError(
                "cached sequence_length exceeds "
                "maximum_sequence_length"
            )

        query_key_value = self.query_key_value_projection(
            hidden_states
        )

        query_key_value = query_key_value.view(
            batch_size,
            query_length,
            3,
            self.number_of_attention_heads,
            self.head_dimension,
        )

        query_key_value = query_key_value.permute(2, 0, 3, 1, 4)
        query, key, value = query_key_value.unbind(dim=0)

        if cache is not None:
            key = torch.cat((cache.key, key), dim=2)
            value = torch.cat((cache.value, value), dim=2)

        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(
            self.head_dimension
        )

        active_mask = self._causal_mask(
            query_length=query_length,
            total_sequence_length=total_sequence_length,
            past_length=past_length,
            device=hidden_states.device,
        )

        attention_scores = attention_scores.masked_fill(
            ~active_mask,
            torch.finfo(attention_scores.dtype).min,
        )

        attention_probabilities = torch.softmax(
            attention_scores,
            dim=-1,
        )

        dropped_probabilities = self.attention_dropout(
            attention_probabilities
        )

        context = dropped_probabilities @ value
        context = context.transpose(1, 2).contiguous()
        context = context.view(
            batch_size,
            query_length,
            self.embedding_dimension,
        )

        output = self.output_projection(context)
        output = self.output_dropout(output)

        updated_cache = None
        if use_cache:
            updated_cache = AttentionKVCache(
                key=key.detach(),
                value=value.detach(),
            )

        if return_attention and use_cache:
            return (
                output,
                attention_probabilities,
                updated_cache,
            )

        if return_attention:
            return output, attention_probabilities

        if use_cache:
            return output, updated_cache

        return output

    @staticmethod
    def _causal_mask(
        *,
        query_length: int,
        total_sequence_length: int,
        past_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        query_positions = torch.arange(
            past_length,
            past_length + query_length,
            device=device,
        ).view(1, 1, query_length, 1)

        key_positions = torch.arange(
            total_sequence_length,
            device=device,
        ).view(1, 1, 1, total_sequence_length)

        return key_positions <= query_positions

    def _validate_input(
        self,
        hidden_states: torch.Tensor,
    ) -> None:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape batch_size, "
                "sequence_length, embedding_dimension"
            )

        sequence_length = hidden_states.shape[1]
        embedding_dimension = hidden_states.shape[2]

        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be greater than zero"
            )

        if sequence_length > self.maximum_sequence_length:
            raise ValueError(
                "sequence_length exceeds maximum_sequence_length"
            )

        if embedding_dimension != self.embedding_dimension:
            raise ValueError(
                "Input embedding dimension does not match "
                "the configured embedding_dimension"
            )

    def _validate_cache(
        self,
        *,
        cache: AttentionKVCache,
        batch_size: int,
        hidden_states: torch.Tensor,
    ) -> None:
        if not isinstance(cache, AttentionKVCache):
            raise TypeError(
                "cache must be an AttentionKVCache or None"
            )

        if cache.batch_size != batch_size:
            raise ValueError(
                "cache batch_size does not match hidden_states"
            )

        if cache.number_of_heads != self.number_of_attention_heads:
            raise ValueError(
                "cache number_of_heads does not match attention"
            )

        if cache.head_dimension != self.head_dimension:
            raise ValueError(
                "cache head_dimension does not match attention"
            )

        if cache.key.device != hidden_states.device:
            raise ValueError(
                "cache device does not match hidden_states"
            )

        if cache.key.dtype != hidden_states.dtype:
            raise TypeError(
                "cache dtype does not match hidden_states"
            )
