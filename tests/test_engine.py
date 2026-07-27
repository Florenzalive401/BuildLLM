import math

import pytest
import torch
from torch import Tensor
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from src.loss import LanguageModelLoss
from src.training.engine import TrainingEngine
from src.training.engine import TrainingEpochResult
from src.training.engine import TrainingStepResult
from src.training.scheduler import LearningRateScheduler
from src.training.scheduler import SchedulerConfig
from src.training_state import TrainingState


class EngineModel(nn.Module):
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

        self.forward_training_modes: list[bool] = []

    def forward(
        self,
        input_ids: Tensor,
    ) -> Tensor:
        self.forward_training_modes.append(
            self.training
        )

        hidden_states = self.embedding(
            input_ids
        )

        return self.output_projection(
            hidden_states
        )


class InvalidOutputModel(nn.Module):
    def forward(
        self,
        input_ids: Tensor,
    ) -> Tensor:
        return torch.zeros(
            input_ids.size(0),
            input_ids.size(1),
            dtype=torch.float32,
        )


class NonTensorOutputModel(nn.Module):
    def forward(
        self,
        input_ids: Tensor,
    ) -> str:
        return "invalid"


def create_scheduler(
    optimizer: AdamW,
    scheduler_type: str = "linear",
    maximum_training_steps: int = 100,
    minimum_learning_rate: float = 0.0001,
) -> LearningRateScheduler:
    return LearningRateScheduler(
        optimizer=optimizer,
        scheduler_config=SchedulerConfig(
            scheduler_type=scheduler_type,
            warmup_steps=0,
            maximum_training_steps=maximum_training_steps,
            minimum_learning_rate=minimum_learning_rate,
        ),
    )


def create_components(
    gradient_clip_norm: float | None = 1.0,
) -> tuple[
    EngineModel,
    AdamW,
    LearningRateScheduler,
    TrainingState,
    TrainingEngine,
]:
    model = EngineModel()

    optimizer = AdamW(
        model.parameters(),
        lr=0.001,
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
    )

    training_state = TrainingState()

    engine = TrainingEngine(
        loss_function=LanguageModelLoss(),
        device="cpu",
        gradient_clip_norm=gradient_clip_norm,
    )

    return (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    )


def create_batch() -> tuple[
    Tensor,
    Tensor,
]:
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
        ],
        dtype=torch.long,
    )

    target_ids = torch.tensor(
        [
            [2, 3, 4, 5],
            [3, 4, 5, 6],
        ],
        dtype=torch.long,
    )

    return (
        input_ids,
        target_ids,
    )


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


def clone_parameters(
    model: nn.Module,
) -> list[Tensor]:
    return [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]


def test_train_step_returns_result() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    result = engine.train_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        batch=create_batch(),
    )

    assert isinstance(
        result,
        TrainingStepResult,
    )


def test_train_step_updates_model_parameters() -> None:
    torch.manual_seed(42)

    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    original_parameters = clone_parameters(
        model
    )

    engine.train_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        batch=create_batch(),
    )

    updated_parameters = clone_parameters(
        model
    )

    assert any(
        not torch.equal(
            original,
            updated,
        )
        for original, updated in zip(
            original_parameters,
            updated_parameters,
            strict=True,
        )
    )


def test_train_step_uses_training_mode() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    model.eval()

    engine.train_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        batch=create_batch(),
    )

    assert model.training is True
    assert model.forward_training_modes == [True]


def test_train_step_returns_finite_loss() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    result = engine.train_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        batch=create_batch(),
    )

    assert math.isfinite(
        result.loss
    )

    assert result.loss > 0


def test_train_step_returns_gradient_norm() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    result = engine.train_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        batch=create_batch(),
    )

    assert math.isfinite(
        result.gradient_norm
    )

    assert result.gradient_norm >= 0


def test_train_step_counts_examples() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    result = engine.train_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        batch=create_batch(),
    )

    assert result.examples_processed == 2


def test_train_step_counts_tokens() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    result = engine.train_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        batch=create_batch(),
    )

    assert result.tokens_processed == 8


def test_train_step_advances_training_state() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    engine.train_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        batch=create_batch(),
    )

    assert training_state.global_step == 1
    assert training_state.tokens_processed == 8
    assert training_state.examples_processed == 2
    assert training_state.training_loss is not None
    assert training_state.learning_rate > 0
    assert training_state.elapsed_seconds >= 0


def test_train_step_advances_scheduler() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    initial_step = scheduler.current_step

    engine.train_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        batch=create_batch(),
    )

    assert scheduler.current_step == initial_step + 1


def test_train_step_clears_existing_gradients() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(
            parameter
        )

    engine.train_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        batch=create_batch(),
    )

    for parameter in model.parameters():
        if parameter.grad is not None:
            assert not torch.all(
                parameter.grad == 1
            )


def test_train_step_without_gradient_clipping() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components(
        gradient_clip_norm=None
    )

    result = engine.train_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        batch=create_batch(),
    )

    assert result.gradient_norm >= 0


def test_ignore_index_tokens_are_excluded() -> None:
    model = EngineModel()

    optimizer = AdamW(
        model.parameters(),
        lr=0.001,
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type="constant",
        maximum_training_steps=10,
        minimum_learning_rate=0.0,
    )

    training_state = TrainingState()

    engine = TrainingEngine(
        loss_function=LanguageModelLoss(
            ignore_index=-100
        ),
        device="cpu",
    )

    input_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
        ],
        dtype=torch.long,
    )

    target_ids = torch.tensor(
        [
            [2, 3, -100, -100],
            [3, 4, 5, -100],
        ],
        dtype=torch.long,
    )

    result = engine.train_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        batch=(
            input_ids,
            target_ids,
        ),
    )

    assert result.tokens_processed == 5


def test_training_step_result_to_dict() -> None:
    result = TrainingStepResult(
        loss=2.0,
        gradient_norm=1.5,
        learning_rate=0.001,
        examples_processed=4,
        tokens_processed=32,
        elapsed_seconds=0.5,
    )

    assert result.to_dict() == {
        "loss": 2.0,
        "gradient_norm": 1.5,
        "learning_rate": 0.001,
        "examples_processed": 4,
        "tokens_processed": 32,
        "elapsed_seconds": 0.5,
    }


def test_train_epoch_returns_result() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    result = engine.train_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        data_loader=create_data_loader(),
    )

    assert isinstance(
        result,
        TrainingEpochResult,
    )


def test_train_epoch_counts_batches() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    result = engine.train_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        data_loader=create_data_loader(
            batch_size=2
        ),
    )

    assert result.batch_count == 2


def test_train_epoch_counts_examples() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    result = engine.train_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        data_loader=create_data_loader(
            batch_size=3
        ),
    )

    assert result.examples_processed == 4


def test_train_epoch_counts_tokens() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    result = engine.train_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        data_loader=create_data_loader(),
    )

    assert result.tokens_processed == 16


def test_train_epoch_increments_epoch() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    engine.train_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        data_loader=create_data_loader(),
    )

    assert training_state.epoch == 1


def test_train_epoch_updates_global_step() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    engine.train_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        data_loader=create_data_loader(
            batch_size=2
        ),
    )

    assert training_state.global_step == 2


def test_train_epoch_returns_average_loss() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    result = engine.train_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        data_loader=create_data_loader(),
    )

    assert math.isfinite(
        result.average_loss
    )

    assert result.average_loss > 0


def test_training_epoch_result_to_dict() -> None:
    result = TrainingEpochResult(
        average_loss=2.0,
        average_gradient_norm=1.0,
        ending_learning_rate=0.001,
        batch_count=4,
        examples_processed=8,
        tokens_processed=32,
        elapsed_seconds=2.0,
    )

    assert result.to_dict() == {
        "average_loss": 2.0,
        "average_gradient_norm": 1.0,
        "ending_learning_rate": 0.001,
        "batch_count": 4,
        "examples_processed": 8,
        "tokens_processed": 32,
        "elapsed_seconds": 2.0,
    }


def test_empty_data_loader_is_rejected() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    empty_inputs = torch.empty(
        (0, 4),
        dtype=torch.long,
    )

    empty_targets = torch.empty(
        (0, 4),
        dtype=torch.long,
    )

    data_loader = DataLoader(
        TensorDataset(
            empty_inputs,
            empty_targets,
        ),
        batch_size=2,
    )

    with pytest.raises(
        ValueError,
        match="no batches",
    ):
        engine.train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            data_loader=data_loader,
        )


def test_invalid_gradient_clip_norm_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="gradient_clip_norm",
    ):
        TrainingEngine(
            loss_function=LanguageModelLoss(),
            device="cpu",
            gradient_clip_norm=0.0,
        )


def test_invalid_loss_function_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="LanguageModelLoss",
    ):
        TrainingEngine(
            loss_function="invalid",
            device="cpu",
        )


def test_invalid_model_is_rejected() -> None:
    (
        _,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    with pytest.raises(
        TypeError,
        match="torch module",
    ):
        engine.train_step(
            model="invalid",
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            batch=create_batch(),
        )


def test_invalid_optimizer_is_rejected() -> None:
    (
        model,
        _,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    with pytest.raises(
        TypeError,
        match="torch optimizer",
    ):
        engine.train_step(
            model=model,
            optimizer="invalid",
            scheduler=scheduler,
            training_state=training_state,
            batch=create_batch(),
        )


def test_invalid_scheduler_is_rejected() -> None:
    (
        model,
        optimizer,
        _,
        training_state,
        engine,
    ) = create_components()

    with pytest.raises(
        TypeError,
        match="LearningRateScheduler",
    ):
        engine.train_step(
            model=model,
            optimizer=optimizer,
            scheduler="invalid",
            training_state=training_state,
            batch=create_batch(),
        )


def test_invalid_training_state_is_rejected() -> None:
    (
        model,
        optimizer,
        scheduler,
        _,
        engine,
    ) = create_components()

    with pytest.raises(
        TypeError,
        match="TrainingState",
    ):
        engine.train_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state="invalid",
            batch=create_batch(),
        )


def test_invalid_batch_is_rejected() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    with pytest.raises(
        TypeError,
        match="tuple or list",
    ):
        engine.train_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            batch=torch.ones(
                2,
                4,
                dtype=torch.long,
            ),
        )


def test_mismatched_batch_shapes_are_rejected() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    input_ids = torch.ones(
        2,
        4,
        dtype=torch.long,
    )

    target_ids = torch.ones(
        2,
        3,
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="matching shapes",
    ):
        engine.train_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            batch=(
                input_ids,
                target_ids,
            ),
        )


def test_non_tensor_model_output_is_rejected() -> None:
    (
        _,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    model = NonTensorOutputModel()

    with pytest.raises(
        TypeError,
        match="model must return a tensor",
    ):
        engine.train_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            batch=create_batch(),
        )


def test_invalid_model_output_shape_is_rejected() -> None:
    (
        _,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    model = InvalidOutputModel()

    with pytest.raises(
        ValueError,
        match="model logits must have shape",
    ):
        engine.train_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            batch=create_batch(),
        )


def test_none_data_loader_is_rejected() -> None:
    (
        model,
        optimizer,
        scheduler,
        training_state,
        engine,
    ) = create_components()

    with pytest.raises(
        TypeError,
        match="data_loader cannot be None",
    ):
        engine.train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            data_loader=None,
        )