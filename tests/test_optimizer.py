import pytest
import torch
from torch import nn
from torch.optim import AdamW

from src.training.optimizer import (
    OptimizerConfig,
)
from src.training.optimizer import (
    OptimizerFactory,
)


class OptimizerTestModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            32,
            8,
        )

        self.layer_norm = nn.LayerNorm(
            8
        )

        self.linear = nn.Linear(
            8,
            16,
        )

        self.output = nn.Linear(
            16,
            32,
            bias=False,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embedding(
            token_ids
        )

        hidden_states = self.layer_norm(
            hidden_states
        )

        hidden_states = self.linear(
            hidden_states
        )

        return self.output(
            hidden_states
        )


class TiedWeightModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            32,
            8,
        )

        self.output = nn.Linear(
            8,
            32,
            bias=False,
        )

        self.output.weight = (
            self.embedding.weight
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.output(
            self.embedding(
                token_ids
            )
        )


def test_optimizer_factory_creates_adamw() -> None:
    model = OptimizerTestModel()

    optimizer = OptimizerFactory.create(
        model=model,
        optimizer_config=OptimizerConfig(),
    )

    assert isinstance(
        optimizer,
        AdamW,
    )


def test_optimizer_uses_configured_learning_rate() -> None:
    model = OptimizerTestModel()

    optimizer = OptimizerFactory.create(
        model=model,
        optimizer_config=OptimizerConfig(
            learning_rate=0.002,
        ),
    )

    for parameter_group in optimizer.param_groups:
        assert parameter_group["lr"] == pytest.approx(
            0.002
        )


def test_optimizer_uses_configured_betas() -> None:
    model = OptimizerTestModel()

    optimizer = OptimizerFactory.create(
        model=model,
        optimizer_config=OptimizerConfig(
            beta_one=0.8,
            beta_two=0.9,
        ),
    )

    for parameter_group in optimizer.param_groups:
        assert parameter_group["betas"] == (
            0.8,
            0.9,
        )


def test_optimizer_uses_configured_epsilon() -> None:
    model = OptimizerTestModel()

    optimizer = OptimizerFactory.create(
        model=model,
        optimizer_config=OptimizerConfig(
            epsilon=1e-6,
        ),
    )

    for parameter_group in optimizer.param_groups:
        assert parameter_group["eps"] == pytest.approx(
            1e-6
        )


def test_optimizer_creates_decay_groups() -> None:
    model = OptimizerTestModel()

    optimizer = OptimizerFactory.create(
        model=model,
        optimizer_config=OptimizerConfig(
            weight_decay=0.25,
        ),
    )

    groups_by_name = {
        parameter_group["group_name"]: (
            parameter_group
        )
        for parameter_group
        in optimizer.param_groups
    }

    assert set(
        groups_by_name
    ) == {
        "decay",
        "no_decay",
    }

    assert groups_by_name[
        "decay"
    ]["weight_decay"] == pytest.approx(
        0.25
    )

    assert groups_by_name[
        "no_decay"
    ]["weight_decay"] == pytest.approx(
        0.0
    )


def test_matrix_parameters_receive_weight_decay() -> None:
    model = OptimizerTestModel()

    parameter_groups = (
        OptimizerFactory.create_parameter_groups(
            model=model,
            weight_decay=0.1,
        )
    )

    groups_by_name = {
        parameter_group["group_name"]: (
            parameter_group
        )
        for parameter_group
        in parameter_groups
    }

    decay_parameter_ids = {
        id(parameter)
        for parameter in groups_by_name[
            "decay"
        ]["params"]
    }

    assert id(
        model.embedding.weight
    ) in decay_parameter_ids

    assert id(
        model.linear.weight
    ) in decay_parameter_ids

    assert id(
        model.output.weight
    ) in decay_parameter_ids


def test_biases_do_not_receive_weight_decay() -> None:
    model = OptimizerTestModel()

    parameter_groups = (
        OptimizerFactory.create_parameter_groups(
            model=model,
            weight_decay=0.1,
        )
    )

    groups_by_name = {
        parameter_group["group_name"]: (
            parameter_group
        )
        for parameter_group
        in parameter_groups
    }

    no_decay_parameter_ids = {
        id(parameter)
        for parameter in groups_by_name[
            "no_decay"
        ]["params"]
    }

    assert id(
        model.linear.bias
    ) in no_decay_parameter_ids


def test_layer_norm_parameters_do_not_receive_weight_decay() -> None:
    model = OptimizerTestModel()

    parameter_groups = (
        OptimizerFactory.create_parameter_groups(
            model=model,
            weight_decay=0.1,
        )
    )

    groups_by_name = {
        parameter_group["group_name"]: (
            parameter_group
        )
        for parameter_group
        in parameter_groups
    }

    no_decay_parameter_ids = {
        id(parameter)
        for parameter in groups_by_name[
            "no_decay"
        ]["params"]
    }

    assert id(
        model.layer_norm.weight
    ) in no_decay_parameter_ids

    assert id(
        model.layer_norm.bias
    ) in no_decay_parameter_ids


def test_all_trainable_parameters_are_in_optimizer() -> None:
    model = OptimizerTestModel()

    optimizer = OptimizerFactory.create(
        model=model,
        optimizer_config=OptimizerConfig(),
    )

    model_parameter_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    optimizer_parameter_ids = {
        id(parameter)
        for parameter_group
        in optimizer.param_groups
        for parameter
        in parameter_group["params"]
    }

    assert (
        optimizer_parameter_ids
        == model_parameter_ids
    )


def test_parameters_are_not_duplicated() -> None:
    model = OptimizerTestModel()

    optimizer = OptimizerFactory.create(
        model=model,
        optimizer_config=OptimizerConfig(),
    )

    optimizer_parameters = [
        parameter
        for parameter_group
        in optimizer.param_groups
        for parameter
        in parameter_group["params"]
    ]

    parameter_ids = [
        id(parameter)
        for parameter
        in optimizer_parameters
    ]

    assert len(
        parameter_ids
    ) == len(
        set(parameter_ids)
    )


def test_tied_weights_are_not_duplicated() -> None:
    model = TiedWeightModel()

    optimizer = OptimizerFactory.create(
        model=model,
        optimizer_config=OptimizerConfig(),
    )

    optimizer_parameter_ids = [
        id(parameter)
        for parameter_group
        in optimizer.param_groups
        for parameter
        in parameter_group["params"]
    ]

    tied_weight_id = id(
        model.embedding.weight
    )

    assert (
        optimizer_parameter_ids.count(
            tied_weight_id
        )
        == 1
    )


def test_frozen_parameters_are_excluded() -> None:
    model = OptimizerTestModel()

    model.linear.weight.requires_grad = False
    model.linear.bias.requires_grad = False

    optimizer = OptimizerFactory.create(
        model=model,
        optimizer_config=OptimizerConfig(),
    )

    optimizer_parameter_ids = {
        id(parameter)
        for parameter_group
        in optimizer.param_groups
        for parameter
        in parameter_group["params"]
    }

    assert id(
        model.linear.weight
    ) not in optimizer_parameter_ids

    assert id(
        model.linear.bias
    ) not in optimizer_parameter_ids


def test_optimizer_can_update_parameters() -> None:
    torch.manual_seed(42)

    model = OptimizerTestModel()

    optimizer = OptimizerFactory.create(
        model=model,
        optimizer_config=OptimizerConfig(
            learning_rate=0.01,
        ),
    )

    token_ids = torch.randint(
        low=0,
        high=32,
        size=(2, 4),
        dtype=torch.long,
    )

    original_weight = (
        model.linear.weight
        .detach()
        .clone()
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    output = model(
        token_ids
    )

    loss = output.mean()

    loss.backward()

    optimizer.step()

    assert not torch.equal(
        original_weight,
        model.linear.weight,
    )


def test_model_without_trainable_parameters_is_rejected() -> None:
    model = nn.Linear(
        4,
        2,
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    with pytest.raises(
        ValueError,
        match="no trainable parameters",
    ):
        OptimizerFactory.create(
            model=model,
            optimizer_config=OptimizerConfig(),
        )


@pytest.mark.parametrize(
    "learning_rate",
    [
        0.0,
        -0.001,
    ],
)
def test_invalid_learning_rate(
    learning_rate: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="learning_rate",
    ):
        OptimizerConfig(
            learning_rate=learning_rate,
        ).validate()


def test_negative_weight_decay_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="weight_decay",
    ):
        OptimizerConfig(
            weight_decay=-0.1,
        ).validate()


@pytest.mark.parametrize(
    "beta_one",
    [
        -0.1,
        1.0,
        2.0,
    ],
)
def test_invalid_beta_one(
    beta_one: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="beta_one",
    ):
        OptimizerConfig(
            beta_one=beta_one,
        ).validate()


@pytest.mark.parametrize(
    "beta_two",
    [
        -0.1,
        1.0,
        2.0,
    ],
)
def test_invalid_beta_two(
    beta_two: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="beta_two",
    ):
        OptimizerConfig(
            beta_two=beta_two,
        ).validate()


def test_invalid_epsilon_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="epsilon",
    ):
        OptimizerConfig(
            epsilon=0.0,
        ).validate()


def test_foreach_and_fused_cannot_both_be_enabled() -> None:
    with pytest.raises(
        ValueError,
        match="foreach and fused",
    ):
        OptimizerConfig(
            foreach=True,
            fused=True,
        ).validate()


def test_invalid_model_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="torch module",
    ):
        OptimizerFactory.create(
            model="invalid",
            optimizer_config=OptimizerConfig(),
        )


def test_negative_group_weight_decay_is_rejected() -> None:
    model = OptimizerTestModel()

    with pytest.raises(
        ValueError,
        match="weight_decay",
    ):
        OptimizerFactory.create_parameter_groups(
            model=model,
            weight_decay=-1.0,
        )
