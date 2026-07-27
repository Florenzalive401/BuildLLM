from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torch.optim import AdamW
from torch.optim import Optimizer


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta_one: float = 0.9
    beta_two: float = 0.95
    epsilon: float = 1e-8
    amsgrad: bool = False
    foreach: bool | None = None
    fused: bool | None = None

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero"
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative"
            )

        if not 0.0 <= self.beta_one < 1.0:
            raise ValueError(
                "beta_one must be between zero and one"
            )

        if not 0.0 <= self.beta_two < 1.0:
            raise ValueError(
                "beta_two must be between zero and one"
            )

        if self.epsilon <= 0:
            raise ValueError(
                "epsilon must be greater than zero"
            )

        if (
            self.foreach is True
            and self.fused is True
        ):
            raise ValueError(
                "foreach and fused cannot both be enabled"
            )


class OptimizerFactory:
    @staticmethod
    def create(
        model: nn.Module,
        optimizer_config: OptimizerConfig,
    ) -> Optimizer:
        if not isinstance(
            model,
            nn.Module,
        ):
            raise TypeError(
                "model must be a torch module"
            )

        optimizer_config.validate()

        parameter_groups = (
            OptimizerFactory.create_parameter_groups(
                model=model,
                weight_decay=(
                    optimizer_config.weight_decay
                ),
            )
        )

        optimizer_arguments = {
            "lr": optimizer_config.learning_rate,
            "betas": (
                optimizer_config.beta_one,
                optimizer_config.beta_two,
            ),
            "eps": optimizer_config.epsilon,
            "amsgrad": optimizer_config.amsgrad,
        }

        if optimizer_config.foreach is not None:
            optimizer_arguments["foreach"] = (
                optimizer_config.foreach
            )

        if optimizer_config.fused is not None:
            optimizer_arguments["fused"] = (
                optimizer_config.fused
            )

        return AdamW(
            parameter_groups,
            **optimizer_arguments,
        )

    @staticmethod
    def create_parameter_groups(
        model: nn.Module,
        weight_decay: float,
    ) -> list[dict[str, object]]:
        if not isinstance(
            model,
            nn.Module,
        ):
            raise TypeError(
                "model must be a torch module"
            )

        if weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative"
            )

        decay_parameters: list[nn.Parameter] = []
        no_decay_parameters: list[nn.Parameter] = []

        seen_parameter_ids: set[int] = set()

        for parameter_name, parameter in (
            model.named_parameters()
        ):
            if not parameter.requires_grad:
                continue

            parameter_id = id(parameter)

            if parameter_id in seen_parameter_ids:
                continue

            seen_parameter_ids.add(
                parameter_id
            )

            if OptimizerFactory._uses_weight_decay(
                parameter_name=parameter_name,
                parameter=parameter,
            ):
                decay_parameters.append(
                    parameter
                )

            else:
                no_decay_parameters.append(
                    parameter
                )

        if not decay_parameters and not no_decay_parameters:
            raise ValueError(
                "model has no trainable parameters"
            )

        parameter_groups: list[
            dict[str, object]
        ] = []

        if decay_parameters:
            parameter_groups.append(
                {
                    "params": decay_parameters,
                    "weight_decay": weight_decay,
                    "group_name": "decay",
                }
            )

        if no_decay_parameters:
            parameter_groups.append(
                {
                    "params": no_decay_parameters,
                    "weight_decay": 0.0,
                    "group_name": "no_decay",
                }
            )

        return parameter_groups

    @staticmethod
    def _uses_weight_decay(
        parameter_name: str,
        parameter: nn.Parameter,
    ) -> bool:
        if parameter_name.endswith(
            ".bias"
        ):
            return False

        if parameter.ndim < 2:
            return False

        return True