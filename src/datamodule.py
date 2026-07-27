from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass(frozen=True)
class DataModuleConfig:
    batch_size: int
    validation_fraction: float = 0.1
    number_of_workers: int = 0
    random_seed: int = 42
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last_training_batch: bool = True

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError(
                "validation_fraction must be between zero and one"
            )

        if self.number_of_workers < 0:
            raise ValueError("number_of_workers cannot be negative")


class LanguageModelDataModule:
    """Creates loaders from pre separated token regions.

    Passing only one dataset retains the original deterministic random split
    behavior for compatibility. Production training should pass a distinct
    validation_dataset so overlapping windows never cross the split boundary.
    """

    def __init__(
        self,
        dataset: Dataset[Any],
        data_config: DataModuleConfig,
        validation_dataset: Dataset[Any] | None = None,
    ) -> None:
        data_config.validate()

        if len(dataset) < 2:
            raise ValueError("dataset must contain at least two examples")

        if validation_dataset is not None and len(validation_dataset) < 1:
            raise ValueError(
                "validation_dataset must contain at least one example"
            )

        self.dataset = dataset
        self.supplied_validation_dataset = validation_dataset
        self.data_config = data_config
        self.training_dataset: Dataset[Any] | None = None
        self.validation_dataset: Dataset[Any] | None = None

    def setup(self) -> None:
        if self.supplied_validation_dataset is not None:
            self.training_dataset = self.dataset
            self.validation_dataset = self.supplied_validation_dataset
            return

        validation_size = max(
            1,
            round(len(self.dataset) * self.data_config.validation_fraction),
        )
        training_size = len(self.dataset) - validation_size

        if training_size < 1:
            raise ValueError("validation split leaves no training examples")

        generator = torch.Generator()
        generator.manual_seed(self.data_config.random_seed)

        self.training_dataset, self.validation_dataset = random_split(
            self.dataset,
            lengths=[training_size, validation_size],
            generator=generator,
        )

    def training_dataloader(self) -> DataLoader[Any]:
        self._validate_setup()
        return DataLoader(
            self.training_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.number_of_workers,
            pin_memory=self.data_config.pin_memory,
            persistent_workers=(
                self.data_config.persistent_workers
                and self.data_config.number_of_workers > 0
            ),
            drop_last=self.data_config.drop_last_training_batch,
            worker_init_fn=self._seed_worker,
            generator=self._create_generator(),
        )

    def validation_dataloader(self) -> DataLoader[Any]:
        self._validate_setup()
        return DataLoader(
            self.validation_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.number_of_workers,
            pin_memory=self.data_config.pin_memory,
            persistent_workers=(
                self.data_config.persistent_workers
                and self.data_config.number_of_workers > 0
            ),
            drop_last=False,
            worker_init_fn=self._seed_worker,
            generator=self._create_generator(),
        )

    def _validate_setup(self) -> None:
        if self.training_dataset is None or self.validation_dataset is None:
            raise RuntimeError(
                "setup must be called before requesting dataloaders"
            )

    def _create_generator(self) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.data_config.random_seed)
        return generator

    @staticmethod
    def _seed_worker(worker_id: int) -> None:
        worker_seed = (torch.initial_seed() + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
