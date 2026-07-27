import pytest
import torch

from src.loss import LanguageModelLoss


def test_loss_returns_scalar():

    criterion = LanguageModelLoss()

    logits = torch.randn(
        2,
        8,
        100,
        requires_grad=True,
    )

    targets = torch.randint(
        0,
        100,
        (2, 8),
    )

    loss = criterion(
        logits,
        targets,
    )

    assert loss.ndim == 0


def test_backward():

    criterion = LanguageModelLoss()

    logits = torch.randn(
        2,
        8,
        50,
        requires_grad=True,
    )

    targets = torch.randint(
        0,
        50,
        (2, 8),
    )

    loss = criterion(
        logits,
        targets,
    )

    loss.backward()

    assert logits.grad is not None


def test_invalid_logits_rank():

    criterion = LanguageModelLoss()

    logits = torch.randn(
        8,
        100,
    )

    targets = torch.randint(
        0,
        100,
        (8,),
    )

    with pytest.raises(
        ValueError,
    ):
        criterion(
            logits,
            targets,
        )


def test_invalid_target_rank():

    criterion = LanguageModelLoss()

    logits = torch.randn(
        2,
        8,
        100,
    )

    targets = torch.randint(
        0,
        100,
        (16,),
    )

    with pytest.raises(
        ValueError,
    ):
        criterion(
            logits,
            targets,
        )


def test_invalid_dtype():

    criterion = LanguageModelLoss()

    logits = torch.randn(
        2,
        8,
        100,
    )

    targets = torch.randn(
        2,
        8,
    )

    with pytest.raises(
        TypeError,
    ):
        criterion(
            logits,
            targets,
        )


def test_label_smoothing():

    criterion = LanguageModelLoss(
        label_smoothing=0.1,
    )

    logits = torch.randn(
        2,
        8,
        20,
    )

    targets = torch.randint(
        0,
        20,
        (2, 8),
    )

    loss = criterion(
        logits,
        targets,
    )

    assert loss.item() > 0


def test_invalid_label_smoothing():

    with pytest.raises(
        ValueError,
    ):
        LanguageModelLoss(
            label_smoothing=2.0,
        )