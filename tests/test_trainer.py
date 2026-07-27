import math
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import Tensor
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from src.loss import LanguageModelLoss
from src.training.engine import TrainingEngine
from src.training.scheduler import LearningRateScheduler
from src.training.scheduler import SchedulerConfig
from src.training.trainer import Trainer
from src.training.trainer import TrainerConfig
from src.training.trainer import TrainerEpochRecord
from src.training.trainer import TrainerResult
from src.training_state import TrainingState


class TrainerModel(nn.Module):
    def __init__(
        self,
        vocabulary_size: int = 16,
        embedding_dimension: int = 8,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dimension,
        )

        self.output_projection = nn.Linear(
            embedding_dimension,
            vocabulary_size,
        )

    def forward(
        self,
        input_ids: Tensor,
    ) -> Tensor:
        return self.output_projection(
            self.embedding(
                input_ids
            )
        )


@dataclass(frozen=True)
class ValidationResult:
    average_loss: float


class ValidationRecorder:
    def __init__(
        self,
        losses: list[float],
    ) -> None:
        self.losses = list(
            losses
        )

        self.call_count = 0

    def __call__(
        self,
        model: nn.Module,
        data_loader: Any,
    ) -> ValidationResult:
        del model
        del data_loader

        index = min(
            self.call_count,
            len(
                self.losses
            )
            - 1,
        )

        self.call_count += 1

        return ValidationResult(
            average_loss=(
                self.losses[index]
            )
        )


class CheckpointRecorder:
    def __init__(
        self,
    ) -> None:
        self.saved_names: list[
            str
        ] = []

        self.saved_metadata: list[
            dict[str, Any]
        ] = []

        self.load_count = 0

    def save(
        self,
        **kwargs: Any,
    ) -> None:
        self.saved_names.append(
            kwargs[
                "checkpoint_name"
            ]
        )

        self.saved_metadata.append(
            kwargs[
                "metadata"
            ]
        )

    def load(
        self,
        **kwargs: Any,
    ) -> None:
        self.load_count += 1

        training_state = (
            kwargs[
                "training_state"
            ]
        )

        training_state.epoch = 1


class InterruptingEngine(
    TrainingEngine
):
    def train_epoch(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        del args
        del kwargs

        raise KeyboardInterrupt


class Clock:
    def __init__(
        self,
    ) -> None:
        self.value = 0.0

    def __call__(
        self,
    ) -> float:
        self.value += 0.25
        return self.value


def create_data_loader(
    batch_size: int = 2,
) -> DataLoader:
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
        ],
        dtype=torch.long,
    )

    target_ids = torch.tensor(
        [
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 8],
        ],
        dtype=torch.long,
    )

    return DataLoader(
        TensorDataset(
            input_ids,
            target_ids,
        ),
        batch_size=batch_size,
        shuffle=False,
    )


def create_training_components() -> tuple[
    TrainerModel,
    AdamW,
    LearningRateScheduler,
    TrainingState,
    TrainingEngine,
]:
    model = TrainerModel()

    optimizer = AdamW(
        model.parameters(),
        lr=0.001,
    )

    scheduler = LearningRateScheduler(
        optimizer=optimizer,
        scheduler_config=SchedulerConfig(
            scheduler_type="linear",
            warmup_steps=0,
            maximum_training_steps=100,
            minimum_learning_rate=0.0001,
        ),
    )

    training_state = TrainingState()

    training_engine = TrainingEngine(
        loss_function=LanguageModelLoss(),
        device="cpu",
        gradient_clip_norm=1.0,
    )

    return (
        model,
        optimizer,
        scheduler,
        training_state,
        training_engine,
    )


def create_trainer(
    *,
    maximum_epochs: int = 2,
    validation_callable: Any = None,
    checkpoint_recorder: CheckpointRecorder | None = None,
    early_stopping_patience: int | None = None,
    minimum_improvement: float = 0.0,
    validation_frequency: int = 1,
    checkpoint_frequency: int = 1,
    restore_checkpoint_before_training: bool = False,
    training_engine: TrainingEngine | None = None,
    callbacks: list[Any] | None = None,
) -> Trainer:
    if training_engine is None:
        training_engine = (
            create_training_components()[4]
        )

    save_callable = None
    load_callable = None

    if checkpoint_recorder is not None:
        save_callable = (
            checkpoint_recorder.save
        )

        load_callable = (
            checkpoint_recorder.load
        )

    return Trainer(
        training_engine=training_engine,
        trainer_config=TrainerConfig(
            maximum_epochs=maximum_epochs,
            validation_frequency=(
                validation_frequency
            ),
            checkpoint_frequency=(
                checkpoint_frequency
            ),
            early_stopping_patience=(
                early_stopping_patience
            ),
            minimum_improvement=(
                minimum_improvement
            ),
            save_best_checkpoint=(
                checkpoint_recorder
                is not None
            ),
            save_last_checkpoint=(
                checkpoint_recorder
                is not None
            ),
            save_periodic_checkpoints=(
                checkpoint_recorder
                is not None
            ),
            save_interrupted_checkpoint=(
                checkpoint_recorder
                is not None
            ),
            restore_checkpoint_before_training=(
                restore_checkpoint_before_training
            ),
        ),
        validation_callable=(
            validation_callable
        ),
        checkpoint_save_callable=(
            save_callable
        ),
        checkpoint_load_callable=(
            load_callable
        ),
        epoch_callbacks=callbacks,
        clock=Clock(),
    )


def test_config_defaults_are_valid() -> None:
    TrainerConfig().validate()


@pytest.mark.parametrize(
    "maximum_epochs",
    [
        -1,
        -100,
    ],
)
def test_negative_maximum_epochs_is_rejected(
    maximum_epochs: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="maximum_epochs",
    ):
        TrainerConfig(
            maximum_epochs=maximum_epochs,
        ).validate()


def test_non_integer_maximum_epochs_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="maximum_epochs",
    ):
        TrainerConfig(
            maximum_epochs=1.5,
        ).validate()


@pytest.mark.parametrize(
    "validation_frequency",
    [
        0,
        -1,
    ],
)
def test_invalid_validation_frequency_is_rejected(
    validation_frequency: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="validation_frequency",
    ):
        TrainerConfig(
            validation_frequency=(
                validation_frequency
            ),
        ).validate()


@pytest.mark.parametrize(
    "checkpoint_frequency",
    [
        0,
        -1,
    ],
)
def test_invalid_checkpoint_frequency_is_rejected(
    checkpoint_frequency: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="checkpoint_frequency",
    ):
        TrainerConfig(
            checkpoint_frequency=(
                checkpoint_frequency
            ),
        ).validate()


def test_negative_early_stopping_patience_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="early_stopping_patience",
    ):
        TrainerConfig(
            early_stopping_patience=-1,
        ).validate()


def test_negative_minimum_improvement_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="minimum_improvement",
    ):
        TrainerConfig(
            minimum_improvement=-0.1,
        ).validate()


def test_nonfinite_minimum_improvement_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="minimum_improvement",
    ):
        TrainerConfig(
            minimum_improvement=math.inf,
        ).validate()


def test_invalid_training_engine_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="TrainingEngine",
    ):
        Trainer(
            training_engine="invalid",
            trainer_config=TrainerConfig(),
            checkpoint_save_callable=(
                lambda **kwargs: None
            ),
        )


def test_invalid_trainer_config_is_rejected() -> None:
    training_engine = (
        create_training_components()[4]
    )

    with pytest.raises(
        TypeError,
        match="TrainerConfig",
    ):
        Trainer(
            training_engine=training_engine,
            trainer_config="invalid",
            checkpoint_save_callable=(
                lambda **kwargs: None
            ),
        )


def test_checkpoint_callable_required_when_saving_enabled() -> None:
    training_engine = (
        create_training_components()[4]
    )

    with pytest.raises(
        ValueError,
        match="checkpoint_save_callable",
    ):
        Trainer(
            training_engine=training_engine,
            trainer_config=TrainerConfig(),
        )


def test_zero_epoch_training() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        training_engine,
    ) = create_training_components()

    trainer = create_trainer(
        maximum_epochs=0,
        training_engine=training_engine,
    )

    result = trainer.fit(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        training_data_loader=(
            create_data_loader()
        ),
    )

    assert result.completed_epochs == 0
    assert result.final_epoch == 0
    assert result.global_step == 0
    assert result.epoch_records == ()


def test_single_epoch_training() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        training_engine,
    ) = create_training_components()

    trainer = create_trainer(
        maximum_epochs=1,
        training_engine=training_engine,
    )

    result = trainer.fit(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        training_data_loader=(
            create_data_loader()
        ),
    )

    assert result.completed_epochs == 1
    assert result.final_epoch == 1
    assert result.global_step == 2
    assert len(
        result.epoch_records
    ) == 1


def test_multiple_epoch_training() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        training_engine,
    ) = create_training_components()

    trainer = create_trainer(
        maximum_epochs=3,
        training_engine=training_engine,
    )

    result = trainer.fit(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        training_data_loader=(
            create_data_loader()
        ),
    )

    assert result.completed_epochs == 3
    assert result.final_epoch == 3
    assert result.global_step == 6
    assert len(
        result.epoch_records
    ) == 3


def test_validation_runs_each_epoch() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        training_engine,
    ) = create_training_components()

    validator = ValidationRecorder(
        [
            3.0,
            2.0,
            1.0,
        ]
    )

    trainer = create_trainer(
        maximum_epochs=3,
        validation_callable=validator,
        training_engine=training_engine,
    )

    result = trainer.fit(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        training_data_loader=(
            create_data_loader()
        ),
        validation_data_loader=(
            create_data_loader()
        ),
    )

    assert validator.call_count == 3
    assert (
        result.best_validation_loss
        == pytest.approx(
            1.0
        )
    )


def test_validation_frequency_is_respected() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        training_engine,
    ) = create_training_components()

    validator = ValidationRecorder(
        [
            3.0,
            2.0,
        ]
    )

    trainer = create_trainer(
        maximum_epochs=4,
        validation_callable=validator,
        validation_frequency=2,
        training_engine=training_engine,
    )

    result = trainer.fit(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        training_data_loader=(
            create_data_loader()
        ),
        validation_data_loader=(
            create_data_loader()
        ),
    )

    assert validator.call_count == 2

    assert [
        record.validation_loss
        for record in result.epoch_records
    ] == [
        None,
        3.0,
        None,
        2.0,
    ]


def test_best_validation_checkpoint_is_saved() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        training_engine,
    ) = create_training_components()

    validator = ValidationRecorder(
        [
            3.0,
            2.0,
            2.5,
        ]
    )

    checkpoints = CheckpointRecorder()

    trainer = create_trainer(
        maximum_epochs=3,
        validation_callable=validator,
        checkpoint_recorder=checkpoints,
        training_engine=training_engine,
    )

    trainer.fit(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        training_data_loader=(
            create_data_loader()
        ),
        validation_data_loader=(
            create_data_loader()
        ),
    )

    assert (
        checkpoints.saved_names.count(
            "best"
        )
        == 2
    )


def test_periodic_checkpoints_are_saved() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        training_engine,
    ) = create_training_components()

    checkpoints = CheckpointRecorder()

    trainer = create_trainer(
        maximum_epochs=4,
        checkpoint_frequency=2,
        checkpoint_recorder=checkpoints,
        training_engine=training_engine,
    )

    trainer.fit(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        training_data_loader=(
            create_data_loader()
        ),
    )

    assert "epoch_2" in (
        checkpoints.saved_names
    )

    assert "epoch_4" in (
        checkpoints.saved_names
    )

    assert "epoch_1" not in (
        checkpoints.saved_names
    )


def test_last_checkpoint_is_saved() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        training_engine,
    ) = create_training_components()

    checkpoints = CheckpointRecorder()

    trainer = create_trainer(
        maximum_epochs=1,
        checkpoint_recorder=checkpoints,
        training_engine=training_engine,
    )

    trainer.fit(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        training_data_loader=(
            create_data_loader()
        ),
    )

    assert (
        checkpoints.saved_names[-1]
        == "last"
    )


def test_checkpoint_restoration_runs_before_training() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        training_engine,
    ) = create_training_components()

    checkpoints = CheckpointRecorder()

    trainer = create_trainer(
        maximum_epochs=3,
        checkpoint_recorder=checkpoints,
        restore_checkpoint_before_training=True,
        training_engine=training_engine,
    )

    result = trainer.fit(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        training_data_loader=(
            create_data_loader()
        ),
    )

    assert checkpoints.load_count == 1
    assert result.restored_checkpoint is True
    assert result.starting_epoch == 1
    assert result.final_epoch == 3
    assert result.completed_epochs == 2


def test_early_stopping() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        training_engine,
    ) = create_training_components()

    validator = ValidationRecorder(
        [
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
        ]
    )

    trainer = create_trainer(
        maximum_epochs=10,
        validation_callable=validator,
        early_stopping_patience=1,
        training_engine=training_engine,
    )

    result = trainer.fit(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        training_data_loader=(
            create_data_loader()
        ),
        validation_data_loader=(
            create_data_loader()
        ),
    )

    assert result.stopped_early is True
    assert result.completed_epochs == 3
    assert result.best_validation_loss == 1.0


def test_minimum_improvement_is_respected() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        training_engine,
    ) = create_training_components()

    validator = ValidationRecorder(
        [
            1.0,
            0.95,
            0.89,
        ]
    )

    trainer = create_trainer(
        maximum_epochs=3,
        validation_callable=validator,
        minimum_improvement=0.1,
        training_engine=training_engine,
    )

    result = trainer.fit(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        training_data_loader=(
            create_data_loader()
        ),
        validation_data_loader=(
            create_data_loader()
        ),
    )

    assert [
        record.improved
        for record in result.epoch_records
    ] == [
        True,
        False,
        True,
    ]


def test_epoch_callback_runs() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        training_engine,
    ) = create_training_components()

    records: list[
        TrainerEpochRecord
    ] = []

    trainer = create_trainer(
        maximum_epochs=2,
        training_engine=training_engine,
        callbacks=[
            records.append
        ],
    )

    trainer.fit(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        training_data_loader=(
            create_data_loader()
        ),
    )

    assert len(
        records
    ) == 2

    assert records[0].epoch == 1
    assert records[1].epoch == 2


def test_keyboard_interrupt_is_recorded() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        _,
    ) = create_training_components()

    checkpoints = CheckpointRecorder()

    interrupting_engine = InterruptingEngine(
        loss_function=LanguageModelLoss(),
        device="cpu",
    )

    trainer = create_trainer(
        maximum_epochs=2,
        checkpoint_recorder=checkpoints,
        training_engine=interrupting_engine,
    )

    result = trainer.fit(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        training_data_loader=(
            create_data_loader()
        ),
    )

    assert result.interrupted is True
    assert "interrupted" in (
        checkpoints.saved_names
    )
    assert "last" in (
        checkpoints.saved_names
    )


def test_invalid_model_is_rejected() -> None:
    (
        _,
        optimizer,
        scheduler,
        training_state,
        training_engine,
    ) = create_training_components()

    trainer = create_trainer(
        training_engine=training_engine,
    )

    with pytest.raises(
        TypeError,
        match="torch module",
    ):
        trainer.fit(
            model="invalid",
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            training_data_loader=(
                create_data_loader()
            ),
        )


def test_invalid_optimizer_is_rejected() -> None:
    (
        model,
        _,
        scheduler,
        training_state,
        training_engine,
    ) = create_training_components()

    trainer = create_trainer(
        training_engine=training_engine,
    )

    with pytest.raises(
        TypeError,
        match="torch optimizer",
    ):
        trainer.fit(
            model=model,
            optimizer="invalid",
            scheduler=scheduler,
            training_state=training_state,
            training_data_loader=(
                create_data_loader()
            ),
        )


def test_invalid_scheduler_is_rejected() -> None:
    (
        model,
        optimizer,
        _,
        training_state,
        training_engine,
    ) = create_training_components()

    trainer = create_trainer(
        training_engine=training_engine,
    )

    with pytest.raises(
        TypeError,
        match="LearningRateScheduler",
    ):
        trainer.fit(
            model=model,
            optimizer=optimizer,
            scheduler="invalid",
            training_state=training_state,
            training_data_loader=(
                create_data_loader()
            ),
        )


def test_invalid_training_state_is_rejected() -> None:
    (
        model,
        optimizer,
        scheduler,
        _,
        training_engine,
    ) = create_training_components()

    trainer = create_trainer(
        training_engine=training_engine,
    )

    with pytest.raises(
        TypeError,
        match="TrainingState",
    ):
        trainer.fit(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state="invalid",
            training_data_loader=(
                create_data_loader()
            ),
        )


def test_none_training_data_loader_is_rejected() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        training_engine,
    ) = create_training_components()

    trainer = create_trainer(
        training_engine=training_engine,
    )

    with pytest.raises(
        TypeError,
        match="training_data_loader",
    ):
        trainer.fit(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            training_data_loader=None,
        )


def test_scheduler_must_reference_optimizer() -> None:
    (
        model,
        optimizer,
        _,
        training_state,
        training_engine,
    ) = create_training_components()

    other_optimizer = AdamW(
        model.parameters(),
        lr=0.001,
    )

    scheduler = LearningRateScheduler(
        optimizer=other_optimizer,
        scheduler_config=SchedulerConfig(
            maximum_training_steps=100,
        ),
    )

    trainer = create_trainer(
        training_engine=training_engine,
    )

    with pytest.raises(
        ValueError,
        match="provided optimizer",
    ):
        trainer.fit(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            training_data_loader=(
                create_data_loader()
            ),
        )


def test_validation_data_requires_validator() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        training_engine,
    ) = create_training_components()

    trainer = create_trainer(
        training_engine=training_engine,
    )

    with pytest.raises(
        ValueError,
        match="validation_callable",
    ):
        trainer.fit(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            training_data_loader=(
                create_data_loader()
            ),
            validation_data_loader=(
                create_data_loader()
            ),
        )


def test_nonfinite_validation_loss_is_rejected() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        training_engine,
    ) = create_training_components()

    validator = ValidationRecorder(
        [
            math.nan
        ]
    )

    trainer = create_trainer(
        maximum_epochs=1,
        validation_callable=validator,
        training_engine=training_engine,
    )

    with pytest.raises(
        FloatingPointError,
        match="validation loss",
    ):
        trainer.fit(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            training_data_loader=(
                create_data_loader()
            ),
            validation_data_loader=(
                create_data_loader()
            ),
        )


def test_trainer_result_to_dict() -> None:
    result = TrainerResult(
        starting_epoch=0,
        completed_epochs=2,
        final_epoch=2,
        global_step=4,
        examples_processed=8,
        tokens_processed=32,
        best_validation_loss=1.25,
        stopped_early=False,
        interrupted=False,
        restored_checkpoint=False,
        elapsed_seconds=2.5,
        epoch_records=(),
    )

    assert result.to_dict() == {
        "starting_epoch": 0,
        "completed_epochs": 2,
        "final_epoch": 2,
        "global_step": 4,
        "examples_processed": 8,
        "tokens_processed": 32,
        "best_validation_loss": 1.25,
        "stopped_early": False,
        "interrupted": False,
        "restored_checkpoint": False,
        "elapsed_seconds": 2.5,
        "epoch_records": [],
    }