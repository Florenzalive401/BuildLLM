from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from src.config import config
from src.training.checkpoint import (
    CheckpointManager,
)
from src.training.scheduler import (
    LearningRateScheduler,
)
from src.training.scheduler import (
    SchedulerConfig,
)
from src.training_state import TrainingState


def create_training_components():
    model = nn.Sequential(
        nn.Linear(
            4,
            8,
        ),
        nn.GELU(),
        nn.Linear(
            8,
            2,
        ),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=0.001,
    )

    scheduler = LearningRateScheduler(
        optimizer,
        SchedulerConfig(
            scheduler_type="linear",
            warmup_steps=0,
            maximum_training_steps=100,
            minimum_learning_rate=0.0001,
        ),
    )

    state = TrainingState()

    return (
        model,
        optimizer,
        scheduler,
        state,
    )


def run_optimizer_step(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LearningRateScheduler,
    state: TrainingState,
) -> None:
    inputs = torch.randn(
        3,
        4,
    )

    targets = torch.randn(
        3,
        2,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    outputs = model(inputs)

    loss = torch.nn.functional.mse_loss(
        outputs,
        targets,
    )

    loss.backward()

    optimizer.step()
    scheduler.step()

    state.increment_step(
        tokens=12,
        examples=3,
    )

    state.update_training_loss(
        float(loss.item())
    )

    state.update_learning_rate(
        scheduler.learning_rate
    )


def clone_parameters(
    model: nn.Module,
) -> list[torch.Tensor]:
    return [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]


def test_checkpoint_is_saved(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path
    )

    (
        model,
        optimizer,
        scheduler,
        state,
    ) = create_training_components()

    checkpoint_path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        model_config=config,
    )

    assert checkpoint_path.exists()

    assert checkpoint_path.name == (
        "checkpoint_step_000000000000.pt"
    )


def test_checkpoint_restores_model_weights(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path
    )

    (
        model,
        optimizer,
        scheduler,
        state,
    ) = create_training_components()

    run_optimizer_step(
        model,
        optimizer,
        scheduler,
        state,
    )

    expected_parameters = clone_parameters(
        model
    )

    checkpoint_path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        model_config=config,
    )

    for parameter in model.parameters():
        parameter.data.zero_()

    manager.load(
        checkpoint_path=checkpoint_path,
        model=model,
    )

    restored_parameters = clone_parameters(
        model
    )

    for expected, restored in zip(
        expected_parameters,
        restored_parameters,
        strict=True,
    ):
        assert torch.equal(
            expected,
            restored,
        )


def test_checkpoint_restores_training_state(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path
    )

    (
        model,
        optimizer,
        scheduler,
        state,
    ) = create_training_components()

    state.epoch = 3
    state.global_step = 17
    state.training_loss = 2.5
    state.validation_loss = 2.2
    state.best_validation_loss = 2.1
    state.learning_rate = 0.0007
    state.tokens_processed = 5000
    state.examples_processed = 200
    state.elapsed_seconds = 42.5

    checkpoint_path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        model_config=config,
    )

    restored_state, _, _ = manager.load(
        checkpoint_path=checkpoint_path,
        model=model,
    )

    assert restored_state == state


def test_checkpoint_restores_optimizer_state(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path
    )

    (
        model,
        optimizer,
        scheduler,
        state,
    ) = create_training_components()

    run_optimizer_step(
        model,
        optimizer,
        scheduler,
        state,
    )

    checkpoint_path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        model_config=config,
    )

    restored_model = nn.Sequential(
        nn.Linear(
            4,
            8,
        ),
        nn.GELU(),
        nn.Linear(
            8,
            2,
        ),
    )

    restored_optimizer = AdamW(
        restored_model.parameters(),
        lr=0.001,
    )

    manager.load(
        checkpoint_path=checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
    )

    assert restored_optimizer.state_dict()[
        "state"
    ]


def test_checkpoint_restores_scheduler_state(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path
    )

    (
        model,
        optimizer,
        scheduler,
        state,
    ) = create_training_components()

    for _ in range(4):
        scheduler.step()

    checkpoint_path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        model_config=config,
    )

    restored_model = nn.Sequential(
        nn.Linear(
            4,
            8,
        ),
        nn.GELU(),
        nn.Linear(
            8,
            2,
        ),
    )

    restored_optimizer = AdamW(
        restored_model.parameters(),
        lr=0.001,
    )

    restored_scheduler = (
        LearningRateScheduler(
            restored_optimizer,
            SchedulerConfig(
                scheduler_type="linear",
                warmup_steps=0,
                maximum_training_steps=100,
                minimum_learning_rate=0.0001,
            ),
        )
    )

    manager.load(
        checkpoint_path=checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert (
        restored_scheduler.current_step
        == scheduler.current_step
    )

    assert (
        restored_scheduler.learning_rate
        == pytest.approx(
            scheduler.learning_rate
        )
    )


def test_checkpoint_restores_metadata(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path
    )

    (
        model,
        optimizer,
        scheduler,
        state,
    ) = create_training_components()

    checkpoint_path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        model_config=config,
        metadata={
            "run_name": "test_run",
            "notes": "checkpoint test",
        },
    )

    _, _, metadata = manager.load(
        checkpoint_path=checkpoint_path,
        model=model,
    )

    assert metadata == {
        "run_name": "test_run",
        "notes": "checkpoint test",
    }


def test_checkpoint_restores_model_config(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path
    )

    (
        model,
        optimizer,
        scheduler,
        state,
    ) = create_training_components()

    test_config = replace(
        config,
        embedding_dimension=128,
        number_of_attention_heads=4,
    )

    checkpoint_path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        model_config=test_config,
    )

    _, restored_config, _ = manager.load(
        checkpoint_path=checkpoint_path,
        model=model,
    )

    assert (
        restored_config[
            "embedding_dimension"
        ]
        == 128
    )

    assert (
        restored_config[
            "number_of_attention_heads"
        ]
        == 4
    )


def test_best_checkpoint_is_saved(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path
    )

    (
        model,
        optimizer,
        scheduler,
        state,
    ) = create_training_components()

    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        model_config=config,
        is_best=True,
    )

    assert (
        manager.best_checkpoint_path.exists()
    )


def test_best_checkpoint_can_be_loaded(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path
    )

    (
        model,
        optimizer,
        scheduler,
        state,
    ) = create_training_components()

    state.global_step = 12

    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        model_config=config,
        is_best=True,
    )

    restored_state, _, _ = (
        manager.load_best(
            model=model
        )
    )

    assert restored_state.global_step == 12


def test_latest_checkpoint_is_returned(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path
    )

    (
        model,
        optimizer,
        scheduler,
        state,
    ) = create_training_components()

    for step in (
        2,
        10,
        4,
    ):
        state.global_step = step

        manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=state,
            model_config=config,
        )

    latest_path = (
        manager.latest_checkpoint_path()
    )

    assert latest_path is not None

    assert latest_path.name == (
        "checkpoint_step_000000000010.pt"
    )


def test_latest_checkpoint_can_be_loaded(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path
    )

    (
        model,
        optimizer,
        scheduler,
        state,
    ) = create_training_components()

    state.global_step = 5

    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        model_config=config,
    )

    state.global_step = 9

    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        model_config=config,
    )

    restored_state, _, _ = (
        manager.load_latest(
            model=model
        )
    )

    assert restored_state.global_step == 9


def test_checkpoint_retention(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path,
        maximum_checkpoints=3,
    )

    (
        model,
        optimizer,
        scheduler,
        state,
    ) = create_training_components()

    for step in range(1, 6):
        state.global_step = step

        manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=state,
            model_config=config,
        )

    checkpoints = (
        manager.list_checkpoints()
    )

    assert len(checkpoints) == 3

    assert [
        path.name
        for path in checkpoints
    ] == [
        "checkpoint_step_000000000003.pt",
        "checkpoint_step_000000000004.pt",
        "checkpoint_step_000000000005.pt",
    ]


def test_retention_does_not_delete_best_checkpoint(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path,
        maximum_checkpoints=1,
    )

    (
        model,
        optimizer,
        scheduler,
        state,
    ) = create_training_components()

    state.global_step = 1

    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        model_config=config,
        is_best=True,
    )

    state.global_step = 2

    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=state,
        model_config=config,
    )

    assert (
        manager.best_checkpoint_path.exists()
    )

    assert len(
        manager.list_checkpoints()
    ) == 1


def test_missing_checkpoint_is_rejected(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path
    )

    (
        model,
        _,
        _,
        _,
    ) = create_training_components()

    with pytest.raises(
        FileNotFoundError,
        match="checkpoint does not exist",
    ):
        manager.load(
            checkpoint_path=(
                tmp_path
                / "missing.pt"
            ),
            model=model,
        )


def test_load_latest_rejects_empty_directory(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path
    )

    (
        model,
        _,
        _,
        _,
    ) = create_training_components()

    with pytest.raises(
        FileNotFoundError,
        match="no periodic checkpoints",
    ):
        manager.load_latest(
            model=model
        )


def test_load_best_rejects_missing_checkpoint(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path
    )

    (
        model,
        _,
        _,
        _,
    ) = create_training_components()

    with pytest.raises(
        FileNotFoundError,
        match="best checkpoint",
    ):
        manager.load_best(
            model=model
        )


def test_invalid_retention_count(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="maximum_checkpoints",
    ):
        CheckpointManager(
            tmp_path,
            maximum_checkpoints=0,
        )


def test_corrupt_checkpoint_is_rejected(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path
    )

    (
        model,
        _,
        _,
        _,
    ) = create_training_components()

    corrupt_path = (
        tmp_path
        / "corrupt.pt"
    )

    torch.save(
        {
            "invalid": True,
        },
        corrupt_path,
    )

    with pytest.raises(
        ValueError,
        match="missing required fields",
    ):
        manager.load(
            checkpoint_path=corrupt_path,
            model=model,
        )