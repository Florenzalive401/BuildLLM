import math

import pytest
import torch
from torch import nn

from src.metrics import bytes_to_megabytes
from src.metrics import calculate_examples_per_second
from src.metrics import calculate_gradient_norm
from src.metrics import calculate_perplexity
from src.metrics import calculate_tokens_per_second
from src.metrics import count_parameters
from src.metrics import count_trainable_parameters
from src.metrics import estimate_model_size_bytes
from src.metrics import summarize_model


def test_parameter_count() -> None:
    model = nn.Linear(
        4,
        3,
        bias=True,
    )

    assert count_parameters(model) == 15


def test_trainable_parameter_count() -> None:
    model = nn.Linear(
        4,
        3,
        bias=True,
    )

    model.bias.requires_grad = False

    assert (
        count_trainable_parameters(model)
        == 12
    )


def test_model_size_estimation() -> None:
    model = nn.Linear(
        4,
        3,
        bias=False,
    )

    expected_bytes = (
        model.weight.numel()
        * model.weight.element_size()
    )

    assert (
        estimate_model_size_bytes(model)
        == expected_bytes
    )


def test_bytes_to_megabytes() -> None:
    assert bytes_to_megabytes(
        1024 ** 2
    ) == 1.0


def test_bytes_to_megabytes_rejects_negative_value() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        bytes_to_megabytes(-1)


def test_perplexity() -> None:
    assert calculate_perplexity(
        0.0
    ) == 1.0

    assert calculate_perplexity(
        math.log(10.0)
    ) == pytest.approx(10.0)


def test_perplexity_rejects_negative_loss() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        calculate_perplexity(-1.0)


def test_gradient_norm() -> None:
    model = nn.Linear(
        2,
        1,
        bias=False,
    )

    inputs = torch.ones(1, 2)

    output = model(inputs)
    output.sum().backward()

    assert calculate_gradient_norm(model) > 0


def test_tokens_per_second() -> None:
    assert calculate_tokens_per_second(
        1_000,
        2.0,
    ) == 500.0


def test_tokens_per_second_rejects_invalid_time() -> None:
    with pytest.raises(
        ValueError,
        match="elapsed_seconds",
    ):
        calculate_tokens_per_second(
            100,
            0.0,
        )


def test_examples_per_second() -> None:
    assert calculate_examples_per_second(
        100,
        2.0,
    ) == 50.0


def test_summarize_model() -> None:
    model = nn.Linear(
        4,
        3,
        bias=True,
    )

    summary = summarize_model(model)

    assert summary["total_parameters"] == 15
    assert summary["trainable_parameters"] == 15
    assert summary["model_size_bytes"] > 0
    assert summary["model_size_megabytes"] > 0