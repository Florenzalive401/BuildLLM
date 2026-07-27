from dataclasses import replace

import pytest
import torch

from src.config import config
from src.feed_forward import FeedForwardNetwork


@pytest.fixture
def hidden_states() -> torch.Tensor:
    torch.manual_seed(42)

    return torch.randn(
        2,
        8,
        config.embedding_dimension,
    )


def test_output_shape(
    hidden_states: torch.Tensor,
) -> None:

    network = FeedForwardNetwork()
    network.eval()

    output = network(hidden_states)

    assert output.shape == hidden_states.shape


def test_deterministic_eval(
    hidden_states: torch.Tensor,
) -> None:

    network = FeedForwardNetwork()
    network.eval()

    first = network(hidden_states)
    second = network(hidden_states)

    assert torch.equal(first, second)


def test_gradients(
    hidden_states: torch.Tensor,
) -> None:

    network = FeedForwardNetwork()

    output = network(hidden_states)

    loss = output.sum()

    loss.backward()

    assert (
        network.input_projection.weight.grad
        is not None
    )

    assert (
        network.output_projection.weight.grad
        is not None
    )


def test_invalid_rank() -> None:

    network = FeedForwardNetwork()

    invalid = torch.randn(
        8,
        config.embedding_dimension,
    )

    with pytest.raises(
        ValueError,
        match="hidden_states",
    ):
        network(invalid)


def test_invalid_embedding_dimension() -> None:

    network = FeedForwardNetwork()

    invalid = torch.randn(
        2,
        8,
        config.embedding_dimension + 1,
    )

    with pytest.raises(
        ValueError,
        match="embedding_dimension",
    ):
        network(invalid)


def test_invalid_feed_forward_dimension() -> None:

    invalid_config = replace(
        config,
        feed_forward_dimension=0,
    )

    with pytest.raises(
        ValueError,
        match="feed_forward_dimension",
    ):
        FeedForwardNetwork(invalid_config)