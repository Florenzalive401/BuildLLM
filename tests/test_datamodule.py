from pathlib import Path

import pytest
import torch

from src.datamodule import DataModuleConfig
from src.datamodule import LanguageModelDataModule
from src.dataset import TokenDataset


@pytest.fixture
def dataset(
    tmp_path: Path,
) -> TokenDataset:
    tokens = torch.arange(
        200,
        dtype=torch.long,
    )

    token_path = tmp_path / "tokens.pt"

    torch.save(
        tokens,
        token_path,
    )

    return TokenDataset(
        token_file=token_path,
        sequence_length=8,
    )


@pytest.fixture
def data_config() -> DataModuleConfig:
    return DataModuleConfig(
        batch_size=4,
        validation_fraction=0.2,
        number_of_workers=0,
        random_seed=42,
        pin_memory=False,
        persistent_workers=False,
        drop_last_training_batch=False,
    )


def test_setup_creates_training_and_validation_datasets(
    dataset: TokenDataset,
    data_config: DataModuleConfig,
) -> None:
    data_module = LanguageModelDataModule(
        dataset,
        data_config,
    )

    data_module.setup()

    assert data_module.training_dataset is not None
    assert data_module.validation_dataset is not None

    assert (
        len(data_module.training_dataset)
        + len(data_module.validation_dataset)
        == len(dataset)
    )


def test_validation_split_size(
    dataset: TokenDataset,
    data_config: DataModuleConfig,
) -> None:
    data_module = LanguageModelDataModule(
        dataset,
        data_config,
    )

    data_module.setup()

    expected_validation_size = round(
        len(dataset)
        * data_config.validation_fraction
    )

    assert data_module.validation_dataset is not None

    assert (
        len(data_module.validation_dataset)
        == expected_validation_size
    )


def test_training_batch_shapes(
    dataset: TokenDataset,
    data_config: DataModuleConfig,
) -> None:
    data_module = LanguageModelDataModule(
        dataset,
        data_config,
    )

    data_module.setup()

    input_ids, target_ids = next(
        iter(
            data_module.training_dataloader()
        )
    )

    assert input_ids.shape == (
        data_config.batch_size,
        dataset.sequence_length,
    )

    assert target_ids.shape == (
        data_config.batch_size,
        dataset.sequence_length,
    )


def test_validation_batch_shapes(
    dataset: TokenDataset,
    data_config: DataModuleConfig,
) -> None:
    data_module = LanguageModelDataModule(
        dataset,
        data_config,
    )

    data_module.setup()

    input_ids, target_ids = next(
        iter(
            data_module.validation_dataloader()
        )
    )

    assert input_ids.ndim == 2
    assert target_ids.ndim == 2

    assert input_ids.shape[1] == dataset.sequence_length
    assert target_ids.shape[1] == dataset.sequence_length


def test_setup_is_deterministic(
    dataset: TokenDataset,
    data_config: DataModuleConfig,
) -> None:
    first_module = LanguageModelDataModule(
        dataset,
        data_config,
    )

    second_module = LanguageModelDataModule(
        dataset,
        data_config,
    )

    first_module.setup()
    second_module.setup()

    assert first_module.training_dataset is not None
    assert second_module.training_dataset is not None

    assert (
        first_module.training_dataset.indices
        == second_module.training_dataset.indices
    )

    assert first_module.validation_dataset is not None
    assert second_module.validation_dataset is not None

    assert (
        first_module.validation_dataset.indices
        == second_module.validation_dataset.indices
    )


def test_dataloaders_require_setup(
    dataset: TokenDataset,
    data_config: DataModuleConfig,
) -> None:
    data_module = LanguageModelDataModule(
        dataset,
        data_config,
    )

    with pytest.raises(
        RuntimeError,
        match="setup must be called",
    ):
        data_module.training_dataloader()

    with pytest.raises(
        RuntimeError,
        match="setup must be called",
    ):
        data_module.validation_dataloader()


def test_invalid_batch_size() -> None:
    invalid_config = DataModuleConfig(
        batch_size=0,
    )

    with pytest.raises(
        ValueError,
        match="batch_size",
    ):
        invalid_config.validate()


def test_invalid_validation_fraction() -> None:
    invalid_config = DataModuleConfig(
        batch_size=4,
        validation_fraction=1.0,
    )

    with pytest.raises(
        ValueError,
        match="validation_fraction",
    ):
        invalid_config.validate()


def test_invalid_worker_count() -> None:
    invalid_config = DataModuleConfig(
        batch_size=4,
        number_of_workers=-1,
    )

    with pytest.raises(
        ValueError,
        match="number_of_workers",
    ):
        invalid_config.validate()


def test_small_dataset_is_rejected() -> None:
    small_dataset = torch.utils.data.TensorDataset(
        torch.tensor([1])
    )

    with pytest.raises(
        ValueError,
        match="at least two examples",
    ):
        LanguageModelDataModule(
            small_dataset,
            DataModuleConfig(
                batch_size=1,
            ),
        )


def test_persistent_workers_disabled_when_no_workers(
    dataset: TokenDataset,
) -> None:
    data_config = DataModuleConfig(
        batch_size=4,
        number_of_workers=0,
        persistent_workers=True,
        pin_memory=False,
    )

    data_module = LanguageModelDataModule(
        dataset,
        data_config,
    )

    data_module.setup()

    dataloader = (
        data_module.training_dataloader()
    )

    assert dataloader.persistent_workers is False