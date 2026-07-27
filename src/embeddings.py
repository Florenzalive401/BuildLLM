from typing import Any

import torch
from torch import nn

from src.config import config


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        model_config: Any = config,
    ) -> None:
        super().__init__()

        self.vocabulary_size = model_config.vocabulary_size
        self.embedding_dimension = model_config.embedding_dimension

        self.embedding = nn.Embedding(
            num_embeddings=self.vocabulary_size,
            embedding_dim=self.embedding_dimension,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long")

        if token_ids.numel() == 0:
            raise ValueError("token_ids cannot be empty")

        minimum_token_id = int(token_ids.min().item())
        maximum_token_id = int(token_ids.max().item())

        if minimum_token_id < 0:
            raise ValueError(
                "token_ids cannot contain negative values"
            )

        if maximum_token_id >= self.vocabulary_size:
            raise ValueError(
                "token_ids contain a value outside "
                "the configured vocabulary"
            )

        return self.embedding(token_ids)


class PositionEmbedding(nn.Module):
    def __init__(
        self,
        model_config: Any = config,
    ) -> None:
        super().__init__()

        self.maximum_sequence_length = (
            model_config.maximum_sequence_length
        )
        self.embedding_dimension = model_config.embedding_dimension

        self.embedding = nn.Embedding(
            num_embeddings=self.maximum_sequence_length,
            embedding_dim=self.embedding_dimension,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have shape "
                "batch_size, sequence_length"
            )

        if not isinstance(position_offset, int):
            raise TypeError("position_offset must be an integer")

        if position_offset < 0:
            raise ValueError("position_offset cannot be negative")

        sequence_length = token_ids.shape[1]

        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be greater than zero"
            )

        if (
            position_offset + sequence_length
            > self.maximum_sequence_length
        ):
            raise ValueError(
                "Position range exceeds maximum_sequence_length"
            )

        position_ids = torch.arange(
            position_offset,
            position_offset + sequence_length,
            device=token_ids.device,
            dtype=torch.long,
        )

        return self.embedding(position_ids)


class InputEmbedding(nn.Module):
    def __init__(
        self,
        model_config: Any = config,
    ) -> None:
        super().__init__()

        self.token_embedding = TokenEmbedding(model_config)
        self.position_embedding = PositionEmbedding(model_config)
        self.dropout = nn.Dropout(model_config.dropout)

    @property
    def token_embedding_weight(
        self,
    ) -> nn.Parameter:
        return self.token_embedding.embedding.weight

    def forward(
        self,
        token_ids: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have shape "
                "batch_size, sequence_length"
            )

        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(
            token_ids,
            position_offset=position_offset,
        )

        return self.dropout(token_vectors + position_vectors)
