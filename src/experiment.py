import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import ModelConfig


class ExperimentRun:
    def __init__(
        self,
        model_config: ModelConfig,
    ) -> None:

        timestamp = datetime.now().strftime(
            "%Y_%m_%d_%H%M%S_%f"
        )

        run_id = uuid.uuid4().hex[:8]

        self.run_directory = (
            Path(model_config.run_directory)
            / f"{timestamp}_{run_id}"
        )

        self.run_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        self.config_path = (
            self.run_directory
            / "config.json"
        )

        self.metrics_path = (
            self.run_directory
            / "metrics.csv"
        )

        self.log_path = (
            self.run_directory
            / "training.log"
        )

        self.checkpoint_path = (
            self.run_directory
            / "checkpoint.pt"
        )

        self._write_config(
            model_config
        )

    def _write_config(
        self,
        model_config: ModelConfig,
    ) -> None:

        with self.config_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                model_config.to_dict(),
                file,
                indent=4,
            )

    def log(
        self,
        message: str,
    ) -> None:

        timestamp = datetime.now().strftime(
            "%H:%M:%S"
        )

        with self.log_path.open(
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                f"[{timestamp}] {message}\n"
            )

    def record_metrics(
        self,
        metrics: dict[str, Any],
    ) -> None:

        file_exists = (
            self.metrics_path.exists()
        )

        with self.metrics_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(metrics.keys()),
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(metrics)