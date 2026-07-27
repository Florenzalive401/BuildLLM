from dataclasses import replace

import pytest
import torch

from src.config import config
from src.model import GPTModel


@pytest.fixture
def token_ids() -> torch.Tensor:
    torch.manual_seed(42)

    return torch.randint(
        low=0,
        high=config.vocabulary_size,
        size=(2, 8),
        dtype=torch.long,
    )


def test_model_output_shape(
    token_ids: torch.Tensor,
) -> None:
    model = GPTModel()
    model.eval()

    logits = model(token_ids)

    assert logits.shape == (
        token_ids.shape[0],
        token_ids.shape[1],
        config.vocabulary_size,
    )


def test_model_is_deterministic_in_evaluation_mode(
    token_ids: torch.Tensor,
) -> None:
    model = GPTModel()
    model.eval()

    first_output = model(token_ids)
    second_output = model(token_ids)

    assert torch.equal(
        first_output,
        second_output,
    )


def test_model_supports_gradients(
    token_ids: torch.Tensor,
) -> None:
    model = GPTModel()

    logits = model(token_ids)
    loss = logits.mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert all(
        gradient is not None
        for gradient in gradients
    )


def test_weight_tying() -> None:
    model = GPTModel()

    assert (
        model.output_projection.weight
        is model.embedding.token_embedding_weight
    )


def test_weight_tying_can_be_disabled() -> None:
    untied_config = replace(
        config,
        weight_tying=False,
    )

    model = GPTModel(
        untied_config
    )

    assert (
        model.output_projection.weight
        is not model.embedding.token_embedding_weight
    )


def test_model_rejects_invalid_rank() -> None:
    model = GPTModel()

    invalid = torch.randint(
        0,
        config.vocabulary_size,
        (8,),
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="token_ids must have shape",
    ):
        model(invalid)


def test_model_rejects_invalid_dtype() -> None:
    model = GPTModel()

    invalid = torch.randn(
        2,
        8,
    )

    with pytest.raises(
        TypeError,
        match="torch.long",
    ):
        model(invalid)


def test_model_rejects_empty_input() -> None:
    model = GPTModel()

    invalid = torch.empty(
        (1, 0),
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        model(invalid)


def test_model_rejects_negative_tokens() -> None:
    model = GPTModel()

    invalid = torch.tensor(
        [[0, 1, -1]],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        model(invalid)


def test_model_rejects_unknown_tokens() -> None:
    model = GPTModel()

    invalid = torch.tensor(
        [[0, config.vocabulary_size]],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="configured vocabulary",
    ):
        model(invalid)


def test_model_rejects_long_sequence() -> None:
    model = GPTModel()

    invalid = torch.zeros(
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
        model(invalid)