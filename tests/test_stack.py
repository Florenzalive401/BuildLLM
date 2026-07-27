from dataclasses import replace

import pytest
import torch

from src.config import config
from src.stack import TransformerStack


@pytest.fixture
def hidden_states() -> torch.Tensor:
    torch.manual_seed(42)

    return torch.randn(
        2,
        8,
        config.embedding_dimension,
    )


def test_stack_output_shape(
    hidden_states: torch.Tensor,
) -> None:
    stack = TransformerStack()
    stack.eval()

    output = stack(hidden_states)

    assert output.shape == hidden_states.shape


def test_stack_contains_configured_layers() -> None:
    stack = TransformerStack()

    assert (
        len(stack.layers)
        == config.number_of_layers
    )


def test_stack_is_deterministic_in_evaluation_mode(
    hidden_states: torch.Tensor,
) -> None:
    stack = TransformerStack()
    stack.eval()

    first_output = stack(hidden_states)
    second_output = stack(hidden_states)

    assert torch.equal(
        first_output,
        second_output,
    )


def test_stack_supports_gradients(
    hidden_states: torch.Tensor,
) -> None:
    stack = TransformerStack()

    hidden_states.requires_grad = True

    output = stack(hidden_states)
    loss = output.sum()
    loss.backward()

    assert hidden_states.grad is not None

    gradients = [
        parameter.grad
        for parameter in stack.parameters()
        if parameter.requires_grad
    ]

    assert all(
        gradient is not None
        for gradient in gradients
    )


def test_stack_rejects_invalid_rank() -> None:
    stack = TransformerStack()

    invalid = torch.randn(
        8,
        config.embedding_dimension,
    )

    with pytest.raises(
        ValueError,
        match="hidden_states must have shape",
    ):
        stack(invalid)


def test_stack_rejects_zero_layers() -> None:
    invalid_config = replace(
        config,
        number_of_layers=0,
    )

    with pytest.raises(
        ValueError,
        match="number_of_layers",
    ):
        TransformerStack(
            invalid_config
        )


def test_gradient_checkpointing() -> None:
    checkpoint_config = replace(
        config,
        use_gradient_checkpointing=True,
    )

    stack = TransformerStack(
        checkpoint_config
    )

    stack.train()

    hidden_states = torch.randn(
        2,
        8,
        checkpoint_config.embedding_dimension,
        requires_grad=True,
    )

    output = stack(hidden_states)
    output.sum().backward()

    assert hidden_states.grad is not None