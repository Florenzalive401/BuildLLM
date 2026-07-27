from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import run_lab


def argument_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_checkpoint_directory_defaults_to_iteration_directory() -> None:
    args = run_lab.parser().parse_args(["--iteration", "2"])

    assert run_lab.checkpoint_directory_for(args) == Path(
        "checkpoints/iteration_2"
    )


def test_checkpoint_directory_override_is_passed_to_training() -> None:
    args = run_lab.parser().parse_args(
        [
            "--iteration",
            "1",
            "--checkpoint-directory",
            "checkpoints/iteration_1_learning_50m",
        ]
    )

    command = run_lab.build_training_command(
        args,
        config_path=run_lab.ITERATIONS[1],
        device="cpu",
    )

    assert Path(argument_value(command, "--checkpoint-directory")) == Path(
        "checkpoints/iteration_1_learning_50m"
    )


def test_learning_run_paths_are_passed_to_training() -> None:
    args = run_lab.parser().parse_args(
        [
            "--iteration",
            "1",
            "--device",
            "cpu",
            "--epochs",
            "3",
            "--training-examples",
            "20000",
            "--validation-examples",
            "1000",
            "--train-tokens",
            "data/tokens/learning_50m/train_tokens.pt",
            "--validation-tokens",
            "data/tokens/learning_50m/validation_tokens.pt",
            "--tokenizer",
            "tokenizer/learning_50m_tokenizer.json",
            "--checkpoint-directory",
            "checkpoints/iteration_1_learning_50m",
            "--resume",
        ]
    )

    command = run_lab.build_training_command(
        args,
        config_path=run_lab.ITERATIONS[1],
        device="cpu",
    )

    assert Path(argument_value(command, "--model-config")) == Path(
        "configs/models/gpt_first_cpu.json"
    )
    assert argument_value(command, "--train-tokens").endswith(
        "learning_50m/train_tokens.pt"
    )
    assert argument_value(command, "--validation-tokens").endswith(
        "learning_50m/validation_tokens.pt"
    )
    assert argument_value(command, "--tokenizer") == (
        "tokenizer/learning_50m_tokenizer.json"
    )
    assert command[-1] == "--resume"


class FakeDevice:
    def __init__(self, device_type: str) -> None:
        self.type = device_type

    def __str__(self) -> str:
        return self.type


@pytest.mark.parametrize(
    (
        "iteration",
        "device",
        "epochs",
        "train_tokens",
        "validation_tokens",
        "tokenizer",
        "checkpoint_directory",
    ),
    [
        (
            "1",
            "cpu",
            "1",
            "data/tokens/learning_50m/train_tokens.pt",
            "data/tokens/learning_50m/validation_tokens.pt",
            "tokenizer/learning_50m_tokenizer.json",
            "checkpoints/iteration_1_pipeline_verification",
        ),
        (
            "2",
            "cuda",
            "3",
            "data/tokens/learning_50m/train_tokens.pt",
            "data/tokens/learning_50m/validation_tokens.pt",
            "tokenizer/learning_50m_tokenizer.json",
            "checkpoints/iteration_2_learning_50m",
        ),
        (
            "2",
            "cuda",
            "3",
            "data/tokens/balanced/train_tokens.pt",
            "data/tokens/balanced/validation_tokens.pt",
            "tokenizer/balanced_tokenizer.json",
            "checkpoints/iteration_2_balanced",
        ),
        (
            "3",
            "cuda",
            "3",
            "data/tokens/learning_50m/train_tokens.pt",
            "data/tokens/learning_50m/validation_tokens.pt",
            "tokenizer/learning_50m_tokenizer.json",
            "checkpoints/iteration_3_learning_50m",
        ),
        (
            "3",
            "cuda",
            "18",
            "data/tokens/800m/train_tokens.pt",
            "data/tokens/800m/validation_tokens.pt",
            "tokenizer/800m_tokenizer.json",
            "checkpoints/iteration_3_800m",
        ),
    ],
)
def test_documented_dry_runs_resolve_without_starting_training(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    iteration: str,
    device: str,
    epochs: str,
    train_tokens: str,
    validation_tokens: str,
    tokenizer: str,
    checkpoint_directory: str,
) -> None:
    runtime_module = types.ModuleType("src.runtime")
    runtime_module.resolve_device = lambda requested: FakeDevice(requested)
    monkeypatch.setitem(sys.modules, "src.runtime", runtime_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_lab.py",
            "--iteration",
            iteration,
            "--device",
            device,
            "--epochs",
            epochs,
            "--train-tokens",
            train_tokens,
            "--validation-tokens",
            validation_tokens,
            "--tokenizer",
            tokenizer,
            "--checkpoint-directory",
            checkpoint_directory,
            "--dry-run",
        ],
    )

    assert run_lab.main() == 0

    output = capsys.readouterr().out
    assert (
        f"--checkpoint-directory {Path(checkpoint_directory)}" in output
    )
    assert f"--train-tokens {train_tokens}" in output
    assert f"--validation-tokens {validation_tokens}" in output
    assert f"--tokenizer {tokenizer}" in output
