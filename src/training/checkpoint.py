from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from src.config import ModelConfig
from src.training.scheduler import LearningRateScheduler
from src.training_state import TrainingState


class CheckpointManager:
    def __init__(
        self,
        checkpoint_directory: str | Path,
        maximum_checkpoints: int = 5,
    ) -> None:
        if maximum_checkpoints <= 0:
            raise ValueError(
                "maximum_checkpoints must be greater than zero"
            )

        self.checkpoint_directory = Path(
            checkpoint_directory
        )

        self.maximum_checkpoints = (
            maximum_checkpoints
        )

        self.checkpoint_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_checkpoint_path = (
            self.checkpoint_directory
            / "best_checkpoint.pt"
        )

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LearningRateScheduler,
        training_state: TrainingState,
        model_config: ModelConfig,
        metadata: dict[str, Any] | None = None,
        is_best: bool = False,
    ) -> Path:
        self._validate_save_inputs(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            model_config=model_config,
        )

        checkpoint_path = (
            self.checkpoint_directory
            / self._checkpoint_filename(
                training_state.global_step
            )
        )

        checkpoint = {
            "format_version": 1,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
            ),
            "training_state": (
                training_state.to_dict()
            ),
            "model_config": (
                model_config.to_dict()
            ),
            "metadata": metadata or {},
        }

        self._atomic_save(
            checkpoint,
            checkpoint_path,
        )

        if is_best:
            self._atomic_save(
                checkpoint,
                self.best_checkpoint_path,
            )

        self._enforce_retention()

        return checkpoint_path

    def load(
        self,
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LearningRateScheduler | None = None,
        map_location: str | torch.device = "cpu",
    ) -> tuple[
        TrainingState,
        dict[str, Any],
        dict[str, Any],
    ]:
        resolved_path = Path(
            checkpoint_path
        )

        if not resolved_path.exists():
            raise FileNotFoundError(
                f"checkpoint does not exist: "
                f"{resolved_path}"
            )

        if not resolved_path.is_file():
            raise ValueError(
                "checkpoint_path must reference a file"
            )

        checkpoint = torch.load(
            resolved_path,
            map_location=map_location,
            weights_only=False,
        )

        self._validate_checkpoint(
            checkpoint
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if optimizer is not None:
            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        if scheduler is not None:
            scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        training_state = (
            TrainingState.from_dict(
                checkpoint["training_state"]
            )
        )

        model_config = dict(
            checkpoint["model_config"]
        )

        metadata = dict(
            checkpoint.get(
                "metadata",
                {},
            )
        )

        return (
            training_state,
            model_config,
            metadata,
        )

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LearningRateScheduler | None = None,
        map_location: str | torch.device = "cpu",
    ) -> tuple[
        TrainingState,
        dict[str, Any],
        dict[str, Any],
    ]:
        latest_path = (
            self.latest_checkpoint_path()
        )

        if latest_path is None:
            raise FileNotFoundError(
                "no periodic checkpoints were found"
            )

        return self.load(
            checkpoint_path=latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location,
        )

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LearningRateScheduler | None = None,
        map_location: str | torch.device = "cpu",
    ) -> tuple[
        TrainingState,
        dict[str, Any],
        dict[str, Any],
    ]:
        if not self.best_checkpoint_path.exists():
            raise FileNotFoundError(
                "best checkpoint does not exist"
            )

        return self.load(
            checkpoint_path=(
                self.best_checkpoint_path
            ),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location,
        )

    def latest_checkpoint_path(
        self,
    ) -> Path | None:
        checkpoints = (
            self._periodic_checkpoints()
        )

        if not checkpoints:
            return None

        return checkpoints[-1]

    def list_checkpoints(
        self,
    ) -> list[Path]:
        return list(
            self._periodic_checkpoints()
        )

    def _atomic_save(
        self,
        checkpoint: dict[str, Any],
        destination: Path,
    ) -> None:
        temporary_path = (
            destination.with_suffix(
                destination.suffix + ".tmp"
            )
        )

        try:
            torch.save(
                checkpoint,
                temporary_path,
            )

            temporary_path.replace(
                destination
            )

        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _enforce_retention(
        self,
    ) -> None:
        checkpoints = (
            self._periodic_checkpoints()
        )

        excess_count = (
            len(checkpoints)
            - self.maximum_checkpoints
        )

        if excess_count <= 0:
            return

        for checkpoint_path in checkpoints[
            :excess_count
        ]:
            checkpoint_path.unlink()

    def _periodic_checkpoints(
        self,
    ) -> list[Path]:
        checkpoints = []

        for path in (
            self.checkpoint_directory.glob(
                "checkpoint_step_*.pt"
            )
        ):
            if self._extract_step(path) is not None:
                checkpoints.append(path)

        return sorted(
            checkpoints,
            key=lambda path: (
                self._extract_step(path)
                or 0
            ),
        )

    @staticmethod
    def _checkpoint_filename(
        global_step: int,
    ) -> str:
        if global_step < 0:
            raise ValueError(
                "global_step cannot be negative"
            )

        return (
            f"checkpoint_step_"
            f"{global_step:012d}.pt"
        )

    @staticmethod
    def _extract_step(
        checkpoint_path: Path,
    ) -> int | None:
        match = re.fullmatch(
            r"checkpoint_step_(\d+)\.pt",
            checkpoint_path.name,
        )

        if match is None:
            return None

        return int(
            match.group(1)
        )

    @staticmethod
    def _validate_checkpoint(
        checkpoint: Any,
    ) -> None:
        if not isinstance(
            checkpoint,
            dict,
        ):
            raise ValueError(
                "checkpoint must contain a dictionary"
            )

        required_keys = {
            "format_version",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "training_state",
            "model_config",
        }

        missing_keys = (
            required_keys
            - checkpoint.keys()
        )

        if missing_keys:
            missing = ", ".join(
                sorted(missing_keys)
            )

            raise ValueError(
                f"checkpoint is missing required "
                f"fields: {missing}"
            )

        if checkpoint["format_version"] != 1:
            raise ValueError(
                "unsupported checkpoint format version"
            )

        if not isinstance(
            checkpoint["training_state"],
            dict,
        ):
            raise ValueError(
                "training_state must be a dictionary"
            )

        if not isinstance(
            checkpoint["model_config"],
            dict,
        ):
            raise ValueError(
                "model_config must be a dictionary"
            )

    @staticmethod
    def _validate_save_inputs(
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LearningRateScheduler,
        training_state: TrainingState,
        model_config: ModelConfig,
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
                "scheduler must be a "
                "LearningRateScheduler"
            )

        if not isinstance(
            training_state,
            TrainingState,
        ):
            raise TypeError(
                "training_state must be a "
                "TrainingState"
            )

        if not isinstance(
            model_config,
            ModelConfig,
        ):
            raise TypeError(
                "model_config must be a "
                "ModelConfig"
            )