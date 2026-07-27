import math

from torch import nn


def count_parameters(
    model: nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def estimate_model_size_bytes(
    model: nn.Module,
) -> int:
    parameter_bytes = sum(
        parameter.numel()
        * parameter.element_size()
        for parameter in model.parameters()
    )

    buffer_bytes = sum(
        buffer.numel()
        * buffer.element_size()
        for buffer in model.buffers()
    )

    return parameter_bytes + buffer_bytes


def bytes_to_megabytes(
    size_bytes: int,
) -> float:
    if size_bytes < 0:
        raise ValueError(
            "size_bytes cannot be negative"
        )

    return size_bytes / (1024 ** 2)


def calculate_perplexity(
    loss: float,
) -> float:
    if not math.isfinite(loss):
        raise ValueError(
            "loss must be finite"
        )

    if loss < 0:
        raise ValueError(
            "loss cannot be negative"
        )

    return math.exp(
        min(loss, 20.0)
    )


def calculate_gradient_norm(
    model: nn.Module,
) -> float:
    squared_norm = 0.0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        gradient_norm = (
            parameter.grad
            .detach()
            .norm(2)
            .item()
        )

        squared_norm += gradient_norm ** 2

    return squared_norm ** 0.5


def calculate_tokens_per_second(
    token_count: int,
    elapsed_seconds: float,
) -> float:
    if token_count < 0:
        raise ValueError(
            "token_count cannot be negative"
        )

    if elapsed_seconds <= 0:
        raise ValueError(
            "elapsed_seconds must be greater than zero"
        )

    return token_count / elapsed_seconds


def calculate_examples_per_second(
    example_count: int,
    elapsed_seconds: float,
) -> float:
    if example_count < 0:
        raise ValueError(
            "example_count cannot be negative"
        )

    if elapsed_seconds <= 0:
        raise ValueError(
            "elapsed_seconds must be greater than zero"
        )

    return example_count / elapsed_seconds


def summarize_model(
    model: nn.Module,
) -> dict[str, int | float]:
    total_parameters = count_parameters(
        model
    )

    trainable_parameters = (
        count_trainable_parameters(model)
    )

    model_size_bytes = estimate_model_size_bytes(
        model
    )

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "model_size_bytes": model_size_bytes,
        "model_size_megabytes": bytes_to_megabytes(
            model_size_bytes
        ),
    }