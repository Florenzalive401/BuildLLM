import pytest
import torch

from src.config import config
from src.embeddings import InputEmbedding
from src.embeddings import PositionEmbedding
from src.embeddings import TokenEmbedding


@pytest.fixture
def token_ids() -> torch.Tensor:
    return torch.tensor(
        [
            [10, 25, 900, 7],
            [41, 1000, 82, 13],
        ],
        dtype=torch.long,
    )


def test_token_embedding_output_shape(
    token_ids: torch.Tensor,
) -> None:
    layer = TokenEmbedding()

    output = layer(token_ids)

    assert output.shape == (
        2,
        4,
        config.embedding_dimension,
    )


def test_token_embedding_uses_configured_dimensions() -> None:
    layer = TokenEmbedding()

    assert layer.embedding.weight.shape == (
        config.vocabulary_size,
        config.embedding_dimension,
    )


def test_token_embedding_rejects_invalid_dtype() -> None:
    layer = TokenEmbedding()

    invalid_tokens = torch.tensor(
        [[1.0, 2.0]],
        dtype=torch.float32,
    )

    with pytest.raises(
        TypeError,
        match="torch.long",
    ):
        layer(invalid_tokens)


def test_token_embedding_rejects_unknown_token() -> None:
    layer = TokenEmbedding()

    invalid_tokens = torch.tensor(
        [[config.vocabulary_size]],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="outside the configured vocabulary",
    ):
        layer(invalid_tokens)


def test_token_embedding_rejects_negative_token() -> None:
    layer = TokenEmbedding()

    invalid_tokens = torch.tensor(
        [[0, -1]],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        layer(invalid_tokens)


def test_token_embedding_rejects_empty_input() -> None:
    layer = TokenEmbedding()

    invalid_tokens = torch.empty(
        (1, 0),
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        layer(invalid_tokens)


def test_position_embedding_output_shape(
    token_ids: torch.Tensor,
) -> None:
    layer = PositionEmbedding()

    output = layer(token_ids)

    assert output.shape == (
        token_ids.shape[1],
        config.embedding_dimension,
    )


def test_position_embedding_rejects_invalid_shape() -> None:
    layer = PositionEmbedding()

    invalid_tokens = torch.tensor(
        [1, 2, 3],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="batch_size, sequence_length",
    ):
        layer(invalid_tokens)


def test_position_embedding_rejects_empty_sequence() -> None:
    layer = PositionEmbedding()

    invalid_tokens = torch.empty(
        (1, 0),
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        layer(invalid_tokens)


def test_position_embedding_rejects_long_sequence() -> None:
    layer = PositionEmbedding()

    invalid_tokens = torch.zeros(
        (
            1,
            config.maximum_sequence_length + 1,
        ),
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="maximum_sequence_length",
    ):
        layer(invalid_tokens)


def test_input_embedding_output_shape(
    token_ids: torch.Tensor,
) -> None:
    layer = InputEmbedding()
    layer.eval()

    output = layer(token_ids)

    assert output.shape == (
        2,
        4,
        config.embedding_dimension,
    )


def test_input_embedding_is_deterministic_in_evaluation_mode(
    token_ids: torch.Tensor,
) -> None:
    layer = InputEmbedding()
    layer.eval()

    first_output = layer(token_ids)
    second_output = layer(token_ids)

    assert torch.equal(
        first_output,
        second_output,
    )


def test_input_embedding_supports_gradients(
    token_ids: torch.Tensor,
) -> None:
    layer = InputEmbedding()

    output = layer(token_ids)
    loss = output.sum()
    loss.backward()

    assert (
        layer.token_embedding.embedding.weight.grad
        is not None
    )

    assert (
        layer.position_embedding.embedding.weight.grad
        is not None
    )


def test_input_embedding_exposes_token_weight() -> None:
    layer = InputEmbedding()

    assert (
        layer.token_embedding_weight
        is layer.token_embedding.embedding.weight
    )


def test_input_embedding_rejects_invalid_shape() -> None:
    layer = InputEmbedding()

    invalid_tokens = torch.tensor(
        [1, 2, 3],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="batch_size, sequence_length",
    ):
        layer(invalid_tokens)