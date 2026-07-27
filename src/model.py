from typing import Any

import torch
from torch import nn

from src.config import config
from src.embeddings import InputEmbedding
from src.kv_cache import ModelKVCache
from src.stack import TransformerStack


class GPTModel(nn.Module):
    def __init__(
        self,
        model_config: Any = config,
    ) -> None:
        super().__init__()

        model_config.validate()
        self.model_config = model_config

        self.embedding = InputEmbedding(model_config)
        self.transformer_stack = TransformerStack(model_config)
        self.final_layer_norm = nn.LayerNorm(
            model_config.embedding_dimension,
            eps=model_config.layer_norm_epsilon,
        )
        self.output_projection = nn.Linear(
            model_config.embedding_dimension,
            model_config.vocabulary_size,
            bias=False,
        )

        if model_config.weight_tying:
            self.output_projection.weight = (
                self.embedding.token_embedding_weight
            )

        self.apply(self._initialize_weights)

    def forward(
        self,
        token_ids: torch.Tensor,
        cache: ModelKVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ModelKVCache]:
        past_length = self._validate_input(
            token_ids=token_ids,
            cache=cache,
        )

        hidden_states = self.embedding(
            token_ids,
            position_offset=past_length,
        )

        if use_cache:
            hidden_states, updated_cache = self.transformer_stack(
                hidden_states,
                cache=cache,
                use_cache=True,
            )
        else:
            hidden_states = self.transformer_stack(
                hidden_states,
                cache=cache,
                use_cache=False,
            )
            updated_cache = None

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.output_projection(hidden_states)

        if use_cache:
            return logits, updated_cache

        return logits

    def _initialize_weights(
        self,
        module: nn.Module,
    ) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _validate_input(
        self,
        *,
        token_ids: torch.Tensor,
        cache: ModelKVCache | None,
    ) -> int:
        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have shape batch_size, sequence_length"
            )

        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long")

        if token_ids.numel() == 0:
            raise ValueError("token_ids cannot be empty")

        sequence_length = token_ids.shape[1]
        past_length = 0

        if cache is not None:
            if not isinstance(cache, ModelKVCache):
                raise TypeError(
                    "cache must be a ModelKVCache or None"
                )

            cache.validate_for_model(
                batch_size=token_ids.shape[0],
                number_of_layers=self.model_config.number_of_layers,
                number_of_heads=(
                    self.model_config.number_of_attention_heads
                ),
                head_dimension=(
                    self.model_config.attention_head_dimension
                ),
                maximum_sequence_length=(
                    self.model_config.maximum_sequence_length
                ),
                device=token_ids.device,
            )
            past_length = cache.sequence_length

        if (
            past_length + sequence_length
            > self.model_config.maximum_sequence_length
        ):
            raise ValueError(
                "combined cache and input sequence_length exceeds "
                "maximum_sequence_length"
            )

        minimum_token_id = int(token_ids.min().item())
        maximum_token_id = int(token_ids.max().item())

        if minimum_token_id < 0:
            raise ValueError(
                "token_ids cannot contain negative values"
            )

        if maximum_token_id >= self.model_config.vocabulary_size:
            raise ValueError(
                "token_ids contain a value outside "
                "the configured vocabulary"
            )

        return past_length
