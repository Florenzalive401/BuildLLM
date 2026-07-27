from dataclasses import replace

import pytest
import torch

from src.attention import CausalSelfAttention
from src.config import config


@pytest.fixture
def hidden_states() -> torch.Tensor:
    torch.manual_seed(42)

    return torch.randn(
        2,
        6,
        config.embedding_dimension,
    )


def test_attention_output_shape(
    hidden_states: torch.Tensor,
) -> None:
    layer = CausalSelfAttention()
    layer.eval()

    output = layer(hidden_states)

    assert output.shape == hidden_states.shape


def test_attention_probability_shape(
    hidden_states: torch.Tensor,
) -> None:
    layer = CausalSelfAttention()
    layer.eval()

    _, probabilities = layer(
        hidden_states,
        return_attention=True,
    )

    assert probabilities.shape == (
        hidden_states.shape[0],
        config.number_of_attention_heads,
        hidden_states.shape[1],
        hidden_states.shape[1],
    )


def test_attention_probabilities_sum_to_one(
    hidden_states: torch.Tensor,
) -> None:
    layer = CausalSelfAttention()
    layer.eval()

    _, probabilities = layer(
        hidden_states,
        return_attention=True,
    )

    row_totals = probabilities.sum(dim=-1)

    assert torch.allclose(
        row_totals,
        torch.ones_like(row_totals),
        atol=1e-6,
    )


def test_attention_blocks_future_tokens(
    hidden_states: torch.Tensor,
) -> None:
    layer = CausalSelfAttention()
    layer.eval()

    _, probabilities = layer(
        hidden_states,
        return_attention=True,
    )

    future_mask = torch.triu(
        torch.ones(
            hidden_states.shape[1],
            hidden_states.shape[1],
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    future_probabilities = probabilities[
        :,
        :,
        future_mask,
    ]

    assert torch.count_nonzero(
        future_probabilities
    ) == 0


def test_future_input_does_not_change_past_output() -> None:
    torch.manual_seed(42)

    layer = CausalSelfAttention()
    layer.eval()

    original = torch.randn(
        1,
        6,
        config.embedding_dimension,
    )

    modified = original.clone()

    modified[:, 3:, :] = torch.randn_like(
        modified[:, 3:, :]
    )

    original_output = layer(original)
    modified_output = layer(modified)

    assert torch.allclose(
        original_output[:, :3, :],
        modified_output[:, :3, :],
        atol=1e-6,
    )


def test_attention_is_deterministic_in_evaluation_mode(
    hidden_states: torch.Tensor,
) -> None:
    layer = CausalSelfAttention()
    layer.eval()

    first_output = layer(hidden_states)
    second_output = layer(hidden_states)

    assert torch.equal(
        first_output,
        second_output,
    )


def test_attention_supports_gradients(
    hidden_states: torch.Tensor,
) -> None:
    layer = CausalSelfAttention()

    output = layer(hidden_states)
    loss = output.sum()
    loss.backward()

    assert (
        layer.query_key_value_projection.weight.grad
        is not None
    )

    assert (
        layer.output_projection.weight.grad
        is not None
    )


def test_attention_rejects_invalid_input_shape() -> None:
    layer = CausalSelfAttention()

    invalid_input = torch.randn(
        4,
        config.embedding_dimension,
    )

    with pytest.raises(
        ValueError,
        match="hidden_states must have shape",
    ):
        layer(invalid_input)


def test_attention_rejects_wrong_embedding_dimension() -> None:
    layer = CausalSelfAttention()

    invalid_input = torch.randn(
        1,
        4,
        config.embedding_dimension + 1,
    )

    with pytest.raises(
        ValueError,
        match="Input embedding dimension",
    ):
        layer(invalid_input)


def test_attention_rejects_long_sequence() -> None:
    layer = CausalSelfAttention()

    invalid_input = torch.randn(
        1,
        config.maximum_sequence_length + 1,
        config.embedding_dimension,
    )

    with pytest.raises(
        ValueError,
        match="sequence_length exceeds",
    ):
        layer(invalid_input)


def test_attention_rejects_incompatible_head_count() -> None:
    invalid_head_count = 3

    if (
        config.embedding_dimension
        % invalid_head_count
        == 0
    ):
        invalid_head_count = 5

    invalid_config = replace(
        config,
        number_of_attention_heads=invalid_head_count,
    )

    with pytest.raises(
        ValueError,
        match="must be divisible",
    ):
        CausalSelfAttention(invalid_config)