import math

import pytest
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from src.loss import LanguageModelLoss
from src.training.validator import (
    ValidationResult,
)
from src.training.validator import (
    Validator,
)


class ValidationModel(nn.Module):
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

        self.dropout = nn.Dropout(
            p=0.5
        )

        self.output_projection = nn.Linear(
            embedding_dimension,
            vocabulary_size,
        )

        self.forward_training_modes: list[
            bool
        ] = []

        self.grad_enabled_states: list[
            bool
        ] = []

    def forward(
        self,
        input_ids: Tensor,
    ) -> Tensor:
        self.forward_training_modes.append(
            self.training
        )

        self.grad_enabled_states.append(
            torch.is_grad_enabled()
        )

        hidden_states = self.embedding(
            input_ids
        )

        hidden_states = self.dropout(
            hidden_states
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


class IntegerOutputModel(nn.Module):
    def forward(
        self,
        input_ids: Tensor,
    ) -> Tensor:
        return torch.zeros(
            input_ids.size(0),
            input_ids.size(1),
            16,
            dtype=torch.long,
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

    dataset = TensorDataset(
        input_ids,
        target_ids,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def create_validator() -> Validator:
    return Validator(
        loss_function=LanguageModelLoss(),
        device="cpu",
    )


def test_validator_returns_validation_result() -> None:
    model = ValidationModel()
    validator = create_validator()

    result = validator.validate(
        model=model,
        data_loader=create_data_loader(),
    )

    assert isinstance(
        result,
        ValidationResult,
    )


def test_validator_computes_finite_loss() -> None:
    torch.manual_seed(42)

    model = ValidationModel()
    validator = create_validator()

    result = validator.validate(
        model=model,
        data_loader=create_data_loader(),
    )

    assert math.isfinite(
        result.average_loss
    )

    assert result.average_loss > 0


def test_validator_computes_perplexity() -> None:
    torch.manual_seed(42)

    model = ValidationModel()
    validator = create_validator()

    result = validator.validate(
        model=model,
        data_loader=create_data_loader(),
    )

    assert result.perplexity == pytest.approx(
        math.exp(
            result.average_loss
        )
    )


def test_validator_counts_batches() -> None:
    model = ValidationModel()
    validator = create_validator()

    result = validator.validate(
        model=model,
        data_loader=create_data_loader(
            batch_size=2
        ),
    )

    assert result.batch_count == 2


def test_validator_counts_examples() -> None:
    model = ValidationModel()
    validator = create_validator()

    result = validator.validate(
        model=model,
        data_loader=create_data_loader(
            batch_size=3
        ),
    )

    assert result.examples_processed == 4


def test_validator_counts_tokens() -> None:
    model = ValidationModel()
    validator = create_validator()

    result = validator.validate(
        model=model,
        data_loader=create_data_loader(),
    )

    assert result.tokens_processed == 16


def test_validator_records_elapsed_time() -> None:
    model = ValidationModel()
    validator = create_validator()

    result = validator.validate(
        model=model,
        data_loader=create_data_loader(),
    )

    assert result.elapsed_seconds >= 0.0


def test_validator_uses_evaluation_mode() -> None:
    model = ValidationModel()
    model.train()

    validator = create_validator()

    validator.validate(
        model=model,
        data_loader=create_data_loader(),
    )

    assert model.forward_training_modes

    assert all(
        mode is False
        for mode in model.forward_training_modes
    )


def test_validator_restores_training_mode() -> None:
    model = ValidationModel()
    model.train()

    validator = create_validator()

    validator.validate(
        model=model,
        data_loader=create_data_loader(),
    )

    assert model.training is True


def test_validator_restores_evaluation_mode() -> None:
    model = ValidationModel()
    model.eval()

    validator = create_validator()

    validator.validate(
        model=model,
        data_loader=create_data_loader(),
    )

    assert model.training is False


def test_validator_disables_gradients() -> None:
    model = ValidationModel()
    validator = create_validator()

    validator.validate(
        model=model,
        data_loader=create_data_loader(),
    )

    assert model.grad_enabled_states

    assert all(
        state is False
        for state in model.grad_enabled_states
    )


def test_validator_does_not_create_parameter_gradients() -> None:
    model = ValidationModel()
    validator = create_validator()

    validator.validate(
        model=model,
        data_loader=create_data_loader(),
    )

    for parameter in model.parameters():
        assert parameter.grad is None


def test_validation_result_converts_to_dictionary() -> None:
    result = ValidationResult(
        average_loss=2.0,
        perplexity=math.exp(2.0),
        batch_count=4,
        examples_processed=8,
        tokens_processed=32,
        elapsed_seconds=1.5,
    )

    result_dictionary = result.to_dict()

    assert result_dictionary == {
        "average_loss": 2.0,
        "perplexity": math.exp(2.0),
        "batch_count": 4,
        "examples_processed": 8,
        "tokens_processed": 32,
        "elapsed_seconds": 1.5,
    }


def test_ignore_index_tokens_are_excluded() -> None:
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

    data_loader = DataLoader(
        TensorDataset(
            input_ids,
            target_ids,
        ),
        batch_size=2,
    )

    validator = Validator(
        loss_function=LanguageModelLoss(
            ignore_index=-100
        ),
        device="cpu",
    )

    model = ValidationModel()

    result = validator.validate(
        model=model,
        data_loader=data_loader,
    )

    assert result.tokens_processed == 5


def test_empty_data_loader_is_rejected() -> None:
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

    validator = create_validator()
    model = ValidationModel()

    with pytest.raises(
        ValueError,
        match="no usable batches",
    ):
        validator.validate(
            model=model,
            data_loader=data_loader,
        )


def test_batch_must_be_tuple_or_list() -> None:
    validator = create_validator()
    model = ValidationModel()

    invalid_loader = [
        torch.ones(
            2,
            4,
            dtype=torch.long,
        )
    ]

    with pytest.raises(
        TypeError,
        match="tuple or list",
    ):
        validator.validate(
            model=model,
            data_loader=invalid_loader,
        )


def test_batch_must_contain_two_values() -> None:
    validator = create_validator()
    model = ValidationModel()

    invalid_loader = [
        (
            torch.ones(
                2,
                4,
                dtype=torch.long,
            ),
        )
    ]

    with pytest.raises(
        ValueError,
        match="input_ids and target_ids",
    ):
        validator.validate(
            model=model,
            data_loader=invalid_loader,
        )


def test_input_ids_must_be_tensor() -> None:
    validator = create_validator()
    model = ValidationModel()

    invalid_loader = [
        (
            "invalid",
            torch.ones(
                2,
                4,
                dtype=torch.long,
            ),
        )
    ]

    with pytest.raises(
        TypeError,
        match="input_ids must be a tensor",
    ):
        validator.validate(
            model=model,
            data_loader=invalid_loader,
        )


def test_target_ids_must_be_tensor() -> None:
    validator = create_validator()
    model = ValidationModel()

    invalid_loader = [
        (
            torch.ones(
                2,
                4,
                dtype=torch.long,
            ),
            "invalid",
        )
    ]

    with pytest.raises(
        TypeError,
        match="target_ids must be a tensor",
    ):
        validator.validate(
            model=model,
            data_loader=invalid_loader,
        )


def test_input_ids_must_have_two_dimensions() -> None:
    validator = create_validator()
    model = ValidationModel()

    invalid_loader = [
        (
            torch.ones(
                4,
                dtype=torch.long,
            ),
            torch.ones(
                4,
                dtype=torch.long,
            ),
        )
    ]

    with pytest.raises(
        ValueError,
        match="input_ids must have shape",
    ):
        validator.validate(
            model=model,
            data_loader=invalid_loader,
        )


def test_target_ids_must_have_two_dimensions() -> None:
    validator = create_validator()
    model = ValidationModel()

    invalid_loader = [
        (
            torch.ones(
                2,
                4,
                dtype=torch.long,
            ),
            torch.ones(
                8,
                dtype=torch.long,
            ),
        )
    ]

    with pytest.raises(
        ValueError,
        match="target_ids must have shape",
    ):
        validator.validate(
            model=model,
            data_loader=invalid_loader,
        )


def test_input_and_target_shapes_must_match() -> None:
    validator = create_validator()
    model = ValidationModel()

    invalid_loader = [
        (
            torch.ones(
                2,
                4,
                dtype=torch.long,
            ),
            torch.ones(
                2,
                3,
                dtype=torch.long,
            ),
        )
    ]

    with pytest.raises(
        ValueError,
        match="matching shapes",
    ):
        validator.validate(
            model=model,
            data_loader=invalid_loader,
        )


def test_input_ids_must_use_long_dtype() -> None:
    validator = create_validator()
    model = ValidationModel()

    invalid_loader = [
        (
            torch.ones(
                2,
                4,
                dtype=torch.float32,
            ),
            torch.ones(
                2,
                4,
                dtype=torch.long,
            ),
        )
    ]

    with pytest.raises(
        TypeError,
        match="input_ids must use torch.long",
    ):
        validator.validate(
            model=model,
            data_loader=invalid_loader,
        )


def test_target_ids_must_use_long_dtype() -> None:
    validator = create_validator()
    model = ValidationModel()

    invalid_loader = [
        (
            torch.ones(
                2,
                4,
                dtype=torch.long,
            ),
            torch.ones(
                2,
                4,
                dtype=torch.float32,
            ),
        )
    ]

    with pytest.raises(
        TypeError,
        match="target_ids must use torch.long",
    ):
        validator.validate(
            model=model,
            data_loader=invalid_loader,
        )


def test_model_output_must_be_tensor() -> None:
    validator = create_validator()
    model = NonTensorOutputModel()

    with pytest.raises(
        TypeError,
        match="model must return a tensor",
    ):
        validator.validate(
            model=model,
            data_loader=create_data_loader(),
        )


def test_model_output_must_have_three_dimensions() -> None:
    validator = create_validator()
    model = InvalidOutputModel()

    with pytest.raises(
        ValueError,
        match="model logits must have shape",
    ):
        validator.validate(
            model=model,
            data_loader=create_data_loader(),
        )


def test_model_output_must_be_floating_point() -> None:
    validator = create_validator()
    model = IntegerOutputModel()

    with pytest.raises(
        TypeError,
        match="floating point dtype",
    ):
        validator.validate(
            model=model,
            data_loader=create_data_loader(),
        )


def test_invalid_model_is_rejected() -> None:
    validator = create_validator()

    with pytest.raises(
        TypeError,
        match="torch module",
    ):
        validator.validate(
            model="invalid",
            data_loader=create_data_loader(),
        )


def test_invalid_loss_function_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="LanguageModelLoss",
    ):
        Validator(
            loss_function="invalid",
            device="cpu",
        )


def test_none_data_loader_is_rejected() -> None:
    validator = create_validator()
    model = ValidationModel()

    with pytest.raises(
        TypeError,
        match="data_loader cannot be None",
    ):
        validator.validate(
            model=model,
            data_loader=None,
        )