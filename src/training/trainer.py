from __future__ import annotations

import math
import time
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Protocol

from torch import nn
from torch.optim import Optimizer

from src.training.engine import TrainingEngine
from src.training.engine import TrainingEpochResult
from src.training.scheduler import LearningRateScheduler
from src.training_state import TrainingState


class ValidationCallable(Protocol):
    def __call__(
        self,
        model: nn.Module,
        data_loader: Iterable[Any],
    ) -> Any:
        ...


class CheckpointSaveCallable(Protocol):
    def __call__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LearningRateScheduler,
        training_state: TrainingState,
        checkpoint_name: str,
        metadata: dict[str, Any],
    ) -> Any:
        ...


class CheckpointLoadCallable(Protocol):
    def __call__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LearningRateScheduler,
        training_state: TrainingState,
    ) -> Any:
        ...


class EpochCallback(Protocol):
    def __call__(
        self,
        record: TrainerEpochRecord,
    ) -> None:
        ...


@dataclass(frozen=True)
class TrainerConfig:
    maximum_epochs: int = 1
    validation_frequency: int = 1
    checkpoint_frequency: int = 1
    early_stopping_patience: int | None = None
    minimum_improvement: float = 0.0
    save_best_checkpoint: bool = True
    save_last_checkpoint: bool = True
    save_periodic_checkpoints: bool = True
    save_interrupted_checkpoint: bool = True
    restore_checkpoint_before_training: bool = False
    fail_on_nonfinite_training_loss: bool = True
    fail_on_nonfinite_validation_loss: bool = True

    def validate(
        self,
    ) -> None:
        if not isinstance(
            self.maximum_epochs,
            int,
        ):
            raise TypeError(
                "maximum_epochs must be an integer"
            )

        if self.maximum_epochs < 0:
            raise ValueError(
                "maximum_epochs cannot be negative"
            )

        if not isinstance(
            self.validation_frequency,
            int,
        ):
            raise TypeError(
                "validation_frequency must be an integer"
            )

        if self.validation_frequency <= 0:
            raise ValueError(
                "validation_frequency must be greater than zero"
            )

        if not isinstance(
            self.checkpoint_frequency,
            int,
        ):
            raise TypeError(
                "checkpoint_frequency must be an integer"
            )

        if self.checkpoint_frequency <= 0:
            raise ValueError(
                "checkpoint_frequency must be greater than zero"
            )

        if self.early_stopping_patience is not None:
            if not isinstance(
                self.early_stopping_patience,
                int,
            ):
                raise TypeError(
                    "early_stopping_patience must be an integer or None"
                )

            if self.early_stopping_patience < 0:
                raise ValueError(
                    "early_stopping_patience cannot be negative"
                )

        if not isinstance(
            self.minimum_improvement,
            int | float,
        ):
            raise TypeError(
                "minimum_improvement must be numeric"
            )

        minimum_improvement = float(
            self.minimum_improvement
        )

        if not math.isfinite(
            minimum_improvement
        ):
            raise ValueError(
                "minimum_improvement must be finite"
            )

        if minimum_improvement < 0:
            raise ValueError(
                "minimum_improvement cannot be negative"
            )

        boolean_fields = {
            "save_best_checkpoint": (
                self.save_best_checkpoint
            ),
            "save_last_checkpoint": (
                self.save_last_checkpoint
            ),
            "save_periodic_checkpoints": (
                self.save_periodic_checkpoints
            ),
            "save_interrupted_checkpoint": (
                self.save_interrupted_checkpoint
            ),
            "restore_checkpoint_before_training": (
                self.restore_checkpoint_before_training
            ),
            "fail_on_nonfinite_training_loss": (
                self.fail_on_nonfinite_training_loss
            ),
            "fail_on_nonfinite_validation_loss": (
                self.fail_on_nonfinite_validation_loss
            ),
        }

        for field_name, field_value in boolean_fields.items():
            if not isinstance(
                field_value,
                bool,
            ):
                raise TypeError(
                    f"{field_name} must be a boolean"
                )


@dataclass(frozen=True)
class TrainerEpochRecord:
    epoch: int
    training_result: TrainingEpochResult
    validation_loss: float | None
    validation_result: Any | None
    improved: bool
    best_validation_loss: float | None
    epochs_without_improvement: int
    elapsed_seconds: float

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "training_result": (
                self.training_result.to_dict()
            ),
            "validation_loss": (
                self.validation_loss
            ),
            "validation_result": (
                self.validation_result
            ),
            "improved": self.improved,
            "best_validation_loss": (
                self.best_validation_loss
            ),
            "epochs_without_improvement": (
                self.epochs_without_improvement
            ),
            "elapsed_seconds": (
                self.elapsed_seconds
            ),
        }


@dataclass(frozen=True)
class TrainerResult:
    starting_epoch: int
    completed_epochs: int
    final_epoch: int
    global_step: int
    examples_processed: int
    tokens_processed: int
    best_validation_loss: float | None
    stopped_early: bool
    interrupted: bool
    restored_checkpoint: bool
    elapsed_seconds: float
    epoch_records: tuple[
        TrainerEpochRecord,
        ...
    ] = field(
        default_factory=tuple
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "starting_epoch": (
                self.starting_epoch
            ),
            "completed_epochs": (
                self.completed_epochs
            ),
            "final_epoch": (
                self.final_epoch
            ),
            "global_step": (
                self.global_step
            ),
            "examples_processed": (
                self.examples_processed
            ),
            "tokens_processed": (
                self.tokens_processed
            ),
            "best_validation_loss": (
                self.best_validation_loss
            ),
            "stopped_early": (
                self.stopped_early
            ),
            "interrupted": (
                self.interrupted
            ),
            "restored_checkpoint": (
                self.restored_checkpoint
            ),
            "elapsed_seconds": (
                self.elapsed_seconds
            ),
            "epoch_records": [
                record.to_dict()
                for record in self.epoch_records
            ],
        }


class Trainer:
    def __init__(
        self,
        training_engine: TrainingEngine,
        trainer_config: TrainerConfig,
        validation_callable: ValidationCallable | None = None,
        checkpoint_save_callable: CheckpointSaveCallable | None = None,
        checkpoint_load_callable: CheckpointLoadCallable | None = None,
        epoch_callbacks: Iterable[
            EpochCallback
        ] | None = None,
        clock: Callable[
            [],
            float,
        ] = time.perf_counter,
    ) -> None:
        if not isinstance(
            training_engine,
            TrainingEngine,
        ):
            raise TypeError(
                "training_engine must be a TrainingEngine"
            )

        if not isinstance(
            trainer_config,
            TrainerConfig,
        ):
            raise TypeError(
                "trainer_config must be a TrainerConfig"
            )

        trainer_config.validate()

        if (
            validation_callable is not None
            and not callable(
                validation_callable
            )
        ):
            raise TypeError(
                "validation_callable must be callable or None"
            )

        if (
            checkpoint_save_callable is not None
            and not callable(
                checkpoint_save_callable
            )
        ):
            raise TypeError(
                "checkpoint_save_callable must be callable or None"
            )

        if (
            checkpoint_load_callable is not None
            and not callable(
                checkpoint_load_callable
            )
        ):
            raise TypeError(
                "checkpoint_load_callable must be callable or None"
            )

        if not callable(
            clock
        ):
            raise TypeError(
                "clock must be callable"
            )

        callbacks = tuple(
            epoch_callbacks or ()
        )

        for callback in callbacks:
            if not callable(
                callback
            ):
                raise TypeError(
                    "every epoch callback must be callable"
                )

        if (
            trainer_config.restore_checkpoint_before_training
            and checkpoint_load_callable is None
        ):
            raise ValueError(
                "checkpoint_load_callable is required when checkpoint restoration is enabled"
            )

        checkpoint_saving_enabled = any(
            (
                trainer_config.save_best_checkpoint,
                trainer_config.save_last_checkpoint,
                trainer_config.save_periodic_checkpoints,
                trainer_config.save_interrupted_checkpoint,
            )
        )

        if (
            checkpoint_saving_enabled
            and checkpoint_save_callable is None
        ):
            raise ValueError(
                "checkpoint_save_callable is required when checkpoint saving is enabled"
            )

        self.training_engine = (
            training_engine
        )

        self.trainer_config = (
            trainer_config
        )

        self.validation_callable = (
            validation_callable
        )

        self.checkpoint_save_callable = (
            checkpoint_save_callable
        )

        self.checkpoint_load_callable = (
            checkpoint_load_callable
        )

        self.epoch_callbacks = callbacks

        self.clock = clock

    def fit(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LearningRateScheduler,
        training_state: TrainingState,
        training_data_loader: Iterable[Any],
        validation_data_loader: Iterable[Any] | None = None,
    ) -> TrainerResult:
        self._validate_fit_arguments(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            training_data_loader=training_data_loader,
            validation_data_loader=validation_data_loader,
        )

        started_at = self.clock()

        starting_epoch = (
            training_state.epoch
        )

        restored_checkpoint = False
        interrupted = False
        stopped_early = False

        best_validation_loss = (
            self._initial_best_validation_loss(
                training_state
            )
        )

        epochs_without_improvement = 0

        epoch_records: list[
            TrainerEpochRecord
        ] = []

        if (
            self.trainer_config
            .restore_checkpoint_before_training
        ):
            self._load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                training_state=training_state,
            )

            restored_checkpoint = True

            starting_epoch = (
                training_state.epoch
            )

            best_validation_loss = (
                self._initial_best_validation_loss(
                    training_state
                )
            )

        try:
            while (
                training_state.epoch
                < self.trainer_config.maximum_epochs
            ):
                epoch_started_at = (
                    self.clock()
                )

                training_result = (
                    self.training_engine
                    .train_epoch(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        training_state=training_state,
                        data_loader=training_data_loader,
                    )
                )

                self._validate_training_result(
                    training_result
                )

                validation_loss: (
                    float | None
                ) = None

                validation_result: (
                    Any | None
                ) = None

                improved = False

                if self._should_validate(
                    training_state.epoch
                ):
                    (
                        validation_loss,
                        validation_result,
                    ) = self._run_validation(
                        model=model,
                        validation_data_loader=validation_data_loader,
                    )

                    improved = (
                        best_validation_loss
                        is None
                        or validation_loss
                        < (
                            best_validation_loss
                            - self.trainer_config
                            .minimum_improvement
                        )
                    )

                    if improved:
                        best_validation_loss = (
                            validation_loss
                        )

                        epochs_without_improvement = 0

                        self._set_best_validation_loss(
                            training_state,
                            best_validation_loss,
                        )

                        if (
                            self.trainer_config
                            .save_best_checkpoint
                        ):
                            self._save_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                training_state=training_state,
                                checkpoint_name="best",
                                metadata={
                                    "checkpoint_type": (
                                        "best"
                                    ),
                                    "epoch": (
                                        training_state
                                        .epoch
                                    ),
                                    "validation_loss": (
                                        validation_loss
                                    ),
                                },
                            )

                    else:
                        epochs_without_improvement += 1

                if self._should_save_periodic_checkpoint(
                    training_state.epoch
                ):
                    self._save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        training_state=training_state,
                        checkpoint_name=(
                            f"epoch_{training_state.epoch}"
                        ),
                        metadata={
                            "checkpoint_type": (
                                "periodic"
                            ),
                            "epoch": (
                                training_state.epoch
                            ),
                            "validation_loss": (
                                validation_loss
                            ),
                        },
                    )

                epoch_elapsed_seconds = (
                    self.clock()
                    - epoch_started_at
                )

                record = TrainerEpochRecord(
                    epoch=training_state.epoch,
                    training_result=training_result,
                    validation_loss=validation_loss,
                    validation_result=validation_result,
                    improved=improved,
                    best_validation_loss=(
                        best_validation_loss
                    ),
                    epochs_without_improvement=(
                        epochs_without_improvement
                    ),
                    elapsed_seconds=(
                        epoch_elapsed_seconds
                    ),
                )

                epoch_records.append(
                    record
                )

                self._run_epoch_callbacks(
                    record
                )

                if self._should_stop_early(
                    epochs_without_improvement=(
                        epochs_without_improvement
                    ),
                    validation_performed=(
                        validation_loss is not None
                    ),
                ):
                    stopped_early = True
                    break

        except KeyboardInterrupt:
            interrupted = True

            if (
                self.trainer_config
                .save_interrupted_checkpoint
            ):
                self._save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    training_state=training_state,
                    checkpoint_name="interrupted",
                    metadata={
                        "checkpoint_type": (
                            "interrupted"
                        ),
                        "epoch": (
                            training_state.epoch
                        ),
                    },
                )

        finally:
            if (
                self.trainer_config
                .save_last_checkpoint
            ):
                self._save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    training_state=training_state,
                    checkpoint_name="last",
                    metadata={
                        "checkpoint_type": (
                            "last"
                        ),
                        "epoch": (
                            training_state.epoch
                        ),
                        "interrupted": (
                            interrupted
                        ),
                        "stopped_early": (
                            stopped_early
                        ),
                    },
                )

        elapsed_seconds = (
            self.clock()
            - started_at
        )

        return TrainerResult(
            starting_epoch=starting_epoch,
            completed_epochs=(
                training_state.epoch
                - starting_epoch
            ),
            final_epoch=(
                training_state.epoch
            ),
            global_step=(
                training_state.global_step
            ),
            examples_processed=(
                training_state
                .examples_processed
            ),
            tokens_processed=(
                training_state
                .tokens_processed
            ),
            best_validation_loss=(
                best_validation_loss
            ),
            stopped_early=(
                stopped_early
            ),
            interrupted=(
                interrupted
            ),
            restored_checkpoint=(
                restored_checkpoint
            ),
            elapsed_seconds=(
                elapsed_seconds
            ),
            epoch_records=tuple(
                epoch_records
            ),
        )

    def _validate_fit_arguments(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LearningRateScheduler,
        training_state: TrainingState,
        training_data_loader: Iterable[Any],
        validation_data_loader: Iterable[Any] | None,
    ) -> None:
        if not isinstance(
            model,
            nn.Module,
        ):
            raise TypeError(
                "model must be a torch module"
            )

        if not isinstance(
            optimizer,
            Optimizer,
        ):
            raise TypeError(
                "optimizer must be a torch optimizer"
            )

        if not isinstance(
            scheduler,
            LearningRateScheduler,
        ):
            raise TypeError(
                "scheduler must be a LearningRateScheduler"
            )

        if not isinstance(
            training_state,
            TrainingState,
        ):
            raise TypeError(
                "training_state must be a TrainingState"
            )

        if training_data_loader is None:
            raise TypeError(
                "training_data_loader cannot be None"
            )

        try:
            iter(
                training_data_loader
            )
        except TypeError as error:
            raise TypeError(
                "training_data_loader must be iterable"
            ) from error

        if validation_data_loader is not None:
            try:
                iter(
                    validation_data_loader
                )
            except TypeError as error:
                raise TypeError(
                    "validation_data_loader must be iterable or None"
                ) from error

        if (
            validation_data_loader is not None
            and self.validation_callable is None
        ):
            raise ValueError(
                "validation_callable is required when validation data is provided"
            )

        if (
            self.validation_callable is not None
            and validation_data_loader is None
            and self.trainer_config
            .early_stopping_patience
            is not None
        ):
            raise ValueError(
                "validation_data_loader is required when early stopping is enabled"
            )

        if (
            scheduler.optimizer
            is not optimizer
        ):
            raise ValueError(
                "scheduler must reference the provided optimizer"
            )

    def _validate_training_result(
        self,
        training_result: TrainingEpochResult,
    ) -> None:
        if not isinstance(
            training_result,
            TrainingEpochResult,
        ):
            raise TypeError(
                "training engine must return TrainingEpochResult"
            )

        if (
            self.trainer_config
            .fail_on_nonfinite_training_loss
            and not math.isfinite(
                training_result.average_loss
            )
        ):
            raise FloatingPointError(
                "training loss is not finite"
            )

    def _should_validate(
        self,
        epoch: int,
    ) -> bool:
        return (
            self.validation_callable
            is not None
            and epoch
            % self.trainer_config
            .validation_frequency
            == 0
        )

    def _run_validation(
        self,
        *,
        model: nn.Module,
        validation_data_loader: Iterable[Any] | None,
    ) -> tuple[
        float,
        Any,
    ]:
        if self.validation_callable is None:
            raise RuntimeError(
                "validation callable is not configured"
            )

        if validation_data_loader is None:
            raise ValueError(
                "validation_data_loader is required for validation"
            )

        validation_result = (
            self.validation_callable(
                model,
                validation_data_loader,
            )
        )

        validation_loss = (
            self._extract_validation_loss(
                validation_result
            )
        )

        if (
            self.trainer_config
            .fail_on_nonfinite_validation_loss
            and not math.isfinite(
                validation_loss
            )
        ):
            raise FloatingPointError(
                "validation loss is not finite"
            )

        return (
            validation_loss,
            validation_result,
        )

    @staticmethod
    def _extract_validation_loss(
        validation_result: Any,
    ) -> float:
        if isinstance(
            validation_result,
            int | float,
        ):
            return float(
                validation_result
            )

        possible_attribute_names = (
            "average_loss",
            "validation_loss",
            "loss",
        )

        for attribute_name in possible_attribute_names:
            if hasattr(
                validation_result,
                attribute_name,
            ):
                value = getattr(
                    validation_result,
                    attribute_name,
                )

                if not isinstance(
                    value,
                    int | float,
                ):
                    raise TypeError(
                        "validation loss must be numeric"
                    )

                return float(
                    value
                )

        if isinstance(
            validation_result,
            dict,
        ):
            possible_keys = (
                "average_loss",
                "validation_loss",
                "loss",
            )

            for key in possible_keys:
                if key in validation_result:
                    value = (
                        validation_result[key]
                    )

                    if not isinstance(
                        value,
                        int | float,
                    ):
                        raise TypeError(
                            "validation loss must be numeric"
                        )

                    return float(
                        value
                    )

        raise TypeError(
            "validation result must expose average_loss, validation_loss, or loss"
        )

    def _should_save_periodic_checkpoint(
        self,
        epoch: int,
    ) -> bool:
        return (
            self.trainer_config
            .save_periodic_checkpoints
            and epoch
            % self.trainer_config
            .checkpoint_frequency
            == 0
        )

    def _save_checkpoint(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LearningRateScheduler,
        training_state: TrainingState,
        checkpoint_name: str,
        metadata: dict[str, Any],
    ) -> Any:
        if self.checkpoint_save_callable is None:
            raise RuntimeError(
                "checkpoint save callable is not configured"
            )

        return self.checkpoint_save_callable(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            checkpoint_name=checkpoint_name,
            metadata=metadata,
        )

    def _load_checkpoint(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LearningRateScheduler,
        training_state: TrainingState,
    ) -> Any:
        if self.checkpoint_load_callable is None:
            raise RuntimeError(
                "checkpoint load callable is not configured"
            )

        return self.checkpoint_load_callable(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
        )

    def _run_epoch_callbacks(
        self,
        record: TrainerEpochRecord,
    ) -> None:
        for callback in self.epoch_callbacks:
            callback(
                record
            )

    def _should_stop_early(
        self,
        *,
        epochs_without_improvement: int,
        validation_performed: bool,
    ) -> bool:
        patience = (
            self.trainer_config
            .early_stopping_patience
        )

        if (
            patience is None
            or not validation_performed
        ):
            return False

        return (
            epochs_without_improvement
            > patience
        )

    @staticmethod
    def _initial_best_validation_loss(
        training_state: TrainingState,
    ) -> float | None:
        if not hasattr(
            training_state,
            "best_validation_loss",
        ):
            return None

        value = getattr(
            training_state,
            "best_validation_loss",
        )

        if value is None:
            return None

        if not isinstance(
            value,
            int | float,
        ):
            raise TypeError(
                "best_validation_loss must be numeric or None"
            )

        numeric_value = float(
            value
        )

        if not math.isfinite(
            numeric_value
        ):
            return None

        return numeric_value

    @staticmethod
    def _set_best_validation_loss(
        training_state: TrainingState,
        validation_loss: float,
    ) -> None:
        if not hasattr(
            training_state,
            "best_validation_loss",
        ):
            return

        try:
            setattr(
                training_state,
                "best_validation_loss",
                validation_loss,
            )
        except (
            AttributeError,
            TypeError,
        ):
            return