from dataclasses import replace

import pytest
import torch

from src.config import config
from src.transformer import TransformerBlock


@pytest.fixture
def hidden_states():

    torch.manual_seed(42)

    return torch.randn(
        2,
        8,
        config.embedding_dimension,
    )


def test_output_shape(
    hidden_states,
):

    block = TransformerBlock()

    block.eval()

    output = block(hidden_states)

    assert output.shape == hidden_states.shape


def test_output_changes_values(
    hidden_states,
):

    block = TransformerBlock()

    block.eval()

    output = block(hidden_states)

    assert not torch.equal(
        output,
        hidden_states,
    )


def test_is_deterministic(
    hidden_states,
):

    block = TransformerBlock()

    block.eval()

    first = block(hidden_states)

    second = block(hidden_states)

    assert torch.equal(
        first,
        second,
    )


def test_gradients(
    hidden_states,
):

    block = TransformerBlock()

    output = block(hidden_states)

    loss = output.sum()

    loss.backward()

    grads = [
        p.grad
        for p in block.parameters()
        if p.requires_grad
    ]

    assert all(
        g is not None
        for g in grads
    )


def test_invalid_rank():

    block = TransformerBlock()

    invalid = torch.randn(
        4,
        config.embedding_dimension,
    )

    with pytest.raises(
        ValueError,
        match="hidden_states",
    ):
        block(invalid)


def test_invalid_embedding_dimension():

    block = TransformerBlock()

    invalid = torch.randn(
        2,
        8,
        config.embedding_dimension + 1,
    )

    with pytest.raises(
        ValueError,
        match="embedding_dimension",
    ):
        block(invalid)


def test_invalid_config():

    invalid = replace(
        config,
        embedding_dimension=0,
    )

    with pytest.raises(
        ValueError,
        match="embedding_dimension",
    ):
        TransformerBlock(
            invalid
        )