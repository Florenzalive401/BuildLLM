import pytest
import torch
from torch import nn
from torch.optim import AdamW

from src.training.scheduler import LearningRateScheduler
from src.training.scheduler import SchedulerConfig


def create_optimizer(
    learning_rate: float = 1.0,
) -> AdamW:
    model = nn.Linear(
        4,
        2,
    )

    return AdamW(
        model.parameters(),
        lr=learning_rate,
    )


def test_constant_learning_rate() -> None:
    optimizer = create_optimizer(
        learning_rate=0.001,
    )

    scheduler = LearningRateScheduler(
        optimizer,
        SchedulerConfig(
            scheduler_type="constant",
            warmup_steps=0,
            maximum_training_steps=10,
        ),
    )

    for _ in range(5):
        scheduler.step()

    assert scheduler.learning_rate == pytest.approx(
        0.001
    )


def test_linear_warmup() -> None:
    optimizer = create_optimizer(
        learning_rate=1.0,
    )

    scheduler = LearningRateScheduler(
        optimizer,
        SchedulerConfig(
            scheduler_type="constant",
            warmup_steps=4,
            maximum_training_steps=10,
        ),
    )

    assert scheduler.learning_rate == pytest.approx(
        0.25
    )

    scheduler.step()

    assert scheduler.learning_rate == pytest.approx(
        0.5
    )

    scheduler.step()

    assert scheduler.learning_rate == pytest.approx(
        0.75
    )

    scheduler.step()

    assert scheduler.learning_rate == pytest.approx(
        1.0
    )


def test_linear_decay_reaches_minimum() -> None:
    optimizer = create_optimizer(
        learning_rate=1.0,
    )

    scheduler = LearningRateScheduler(
        optimizer,
        SchedulerConfig(
            scheduler_type="linear",
            warmup_steps=0,
            maximum_training_steps=4,
            minimum_learning_rate=0.1,
        ),
    )

    for _ in range(4):
        scheduler.step()

    assert scheduler.learning_rate == pytest.approx(
        0.1
    )


def test_cosine_decay_reaches_minimum() -> None:
    optimizer = create_optimizer(
        learning_rate=1.0,
    )

    scheduler = LearningRateScheduler(
        optimizer,
        SchedulerConfig(
            scheduler_type="cosine",
            warmup_steps=0,
            maximum_training_steps=4,
            minimum_learning_rate=0.1,
        ),
    )

    for _ in range(4):
        scheduler.step()

    assert scheduler.learning_rate == pytest.approx(
        0.1
    )


def test_learning_rate_does_not_fall_below_minimum() -> None:
    optimizer = create_optimizer(
        learning_rate=1.0,
    )

    scheduler = LearningRateScheduler(
        optimizer,
        SchedulerConfig(
            scheduler_type="cosine",
            warmup_steps=0,
            maximum_training_steps=2,
            minimum_learning_rate=0.2,
        ),
    )

    for _ in range(10):
        scheduler.step()

    assert scheduler.learning_rate == pytest.approx(
        0.2
    )


def test_multiple_parameter_groups() -> None:
    first_layer = nn.Linear(
        4,
        4,
    )

    second_layer = nn.Linear(
        4,
        2,
    )

    optimizer = AdamW(
        [
            {
                "params": first_layer.parameters(),
                "lr": 1.0,
            },
            {
                "params": second_layer.parameters(),
                "lr": 0.5,
            },
        ]
    )

    scheduler = LearningRateScheduler(
        optimizer,
        SchedulerConfig(
            scheduler_type="linear",
            warmup_steps=0,
            maximum_training_steps=2,
            minimum_learning_rate=0.0,
        ),
    )

    scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        0.5
    )

    assert optimizer.param_groups[1]["lr"] == pytest.approx(
        0.25
    )


def test_scheduler_state_can_be_restored() -> None:
    optimizer = create_optimizer(
        learning_rate=1.0,
    )

    scheduler = LearningRateScheduler(
        optimizer,
        SchedulerConfig(
            scheduler_type="linear",
            warmup_steps=0,
            maximum_training_steps=10,
        ),
    )

    scheduler.step()
    scheduler.step()
    scheduler.step()

    saved_state = scheduler.state_dict()
    saved_learning_rate = scheduler.learning_rate

    restored_optimizer = create_optimizer(
        learning_rate=1.0,
    )

    restored_scheduler = LearningRateScheduler(
        restored_optimizer,
        SchedulerConfig(
            scheduler_type="linear",
            warmup_steps=0,
            maximum_training_steps=10,
        ),
    )

    restored_scheduler.load_state_dict(
        saved_state
    )

    assert restored_scheduler.current_step == 3

    assert restored_scheduler.learning_rate == pytest.approx(
        saved_learning_rate
    )


def test_invalid_scheduler_type() -> None:
    optimizer = create_optimizer()

    with pytest.raises(
        ValueError,
        match="scheduler_type",
    ):
        LearningRateScheduler(
            optimizer,
            SchedulerConfig(
                scheduler_type="invalid",
            ),
        )


def test_invalid_warmup_steps() -> None:
    optimizer = create_optimizer()

    with pytest.raises(
        ValueError,
        match="warmup_steps",
    ):
        LearningRateScheduler(
            optimizer,
            SchedulerConfig(
                warmup_steps=-1,
            ),
        )


def test_warmup_must_be_less_than_maximum_steps() -> None:
    optimizer = create_optimizer()

    with pytest.raises(
        ValueError,
        match="warmup_steps",
    ):
        LearningRateScheduler(
            optimizer,
            SchedulerConfig(
                warmup_steps=10,
                maximum_training_steps=10,
            ),
        )


def test_minimum_learning_rate_cannot_exceed_base_rate() -> None:
    optimizer = create_optimizer(
        learning_rate=0.001,
    )

    with pytest.raises(
        ValueError,
        match="minimum_learning_rate",
    ):
        LearningRateScheduler(
            optimizer,
            SchedulerConfig(
                minimum_learning_rate=0.01,
            ),
        )


def test_invalid_restored_step() -> None:
    optimizer = create_optimizer()

    scheduler = LearningRateScheduler(
        optimizer,
        SchedulerConfig(),
    )

    with pytest.raises(
        ValueError,
        match="current_step",
    ):
        scheduler.load_state_dict(
            {
                "current_step": -1,
            }
        )


def test_scheduler_updates_optimizer() -> None:
    model = nn.Linear(
        2,
        1,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1.0,
    )

    scheduler = LearningRateScheduler(
        optimizer,
        SchedulerConfig(
            scheduler_type="linear",
            maximum_training_steps=2,
        ),
    )

    scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        0.5
    )