import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.config import config
from src.experiment import ExperimentRun


@pytest.fixture
def experiment_config(
    tmp_path: Path,
):
    return replace(
        config,
        run_directory=str(tmp_path),
    )


def test_experiment_creation(
    experiment_config,
) -> None:

    run = ExperimentRun(
        experiment_config
    )

    assert run.run_directory.exists()

    assert run.config_path.exists()

    assert run.metrics_path.exists() is False


def test_logging(
    experiment_config,
) -> None:

    run = ExperimentRun(
        experiment_config
    )

    run.log(
        "Hello World"
    )

    text = run.log_path.read_text(
        encoding="utf-8"
    )

    assert "Hello World" in text


def test_metrics(
    experiment_config,
) -> None:

    run = ExperimentRun(
        experiment_config
    )

    run.record_metrics(
        {
            "epoch": 1,
            "loss": 2.35,
        }
    )

    text = run.metrics_path.read_text(
        encoding="utf-8"
    )

    assert "epoch" in text

    assert "loss" in text

    assert "2.35" in text


def test_config_written(
    experiment_config,
) -> None:

    run = ExperimentRun(
        experiment_config
    )

    data = json.loads(
        run.config_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        data["embedding_dimension"]
        ==
        experiment_config.embedding_dimension
    )

    assert (
        data["number_of_layers"]
        ==
        experiment_config.number_of_layers
    )


def test_unique_run_directories(
    experiment_config,
) -> None:

    run1 = ExperimentRun(
        experiment_config
    )

    run2 = ExperimentRun(
        experiment_config
    )

    assert (
        run1.run_directory
        !=
        run2.run_directory
    )


def test_multiple_metric_rows(
    experiment_config,
) -> None:

    run = ExperimentRun(
        experiment_config
    )

    run.record_metrics(
        {
            "epoch": 1,
            "loss": 2.5,
        }
    )

    run.record_metrics(
        {
            "epoch": 2,
            "loss": 2.1,
        }
    )

    lines = (
        run.metrics_path
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    assert len(lines) == 3