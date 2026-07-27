from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.config import config
from src.kv_cache import AttentionKVCache
from src.kv_cache import ModelKVCache
from src.transformer import TransformerBlock


class TransformerStack(nn.Module):
    def __init__(
        self,
        model_config: Any = config,
    ) -> None:
        super().__init__()

        self.number_of_layers = model_config.number_of_layers
        self.use_gradient_checkpointing = (
            model_config.use_gradient_checkpointing
        )

        if self.number_of_layers <= 0:
            raise ValueError(
                "number_of_layers must be greater than zero"
            )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(model_config)
                for _ in range(self.number_of_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: ModelKVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ModelKVCache]:
        self._validate_input(hidden_states)

        if use_cache and self.use_gradient_checkpointing and self.training:
            raise RuntimeError(
                "KV caching cannot be used with gradient checkpointing "
                "during training"
            )

        if cache is not None and cache.number_of_layers != self.number_of_layers:
            raise ValueError(
                "cache layer count does not match number_of_layers"
            )

        updated_layers: list[AttentionKVCache] = []

        for layer_index, layer in enumerate(self.layers):
            layer_cache = (
                cache.layers[layer_index]
                if cache is not None
                else None
            )

            if self.use_gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    use_reentrant=False,
                )
            elif use_cache:
                hidden_states, updated_cache = layer(
                    hidden_states,
                    cache=layer_cache,
                    use_cache=True,
                )
                updated_layers.append(updated_cache)
            else:
                hidden_states = layer(
                    hidden_states,
                    cache=layer_cache,
                    use_cache=False,
                )

        if use_cache:
            return hidden_states, ModelKVCache(
                layers=tuple(updated_layers)
            )

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
