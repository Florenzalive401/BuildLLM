from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeDefaults:
    batch_size: int
    precision: str
    workers: int = 0

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any] | None,
        *,
        fallback_batch_size: int,
        fallback_precision: str,
    ) -> "RuntimeDefaults":
        payload = payload or {}
        runtime = cls(
            batch_size=int(payload.get("batch_size", fallback_batch_size)),
            precision=str(payload.get("precision", fallback_precision)),
            workers=int(payload.get("workers", 0)),
        )
        runtime.validate()
        return runtime

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("runtime batch_size must be greater than zero")
        if self.workers < 0:
            raise ValueError("runtime workers cannot be negative")
        if self.precision not in {"auto", "fp32", "bf16", "fp16"}:
            raise ValueError("runtime precision must be auto, fp32, bf16, or fp16")


@dataclass(frozen=True)
class ModelProfile:
    name: str
    description: str
    course_iteration: int
    learning_objective: str
    sequence_length: int
    embedding_dimension: int
    layers: int
    attention_heads: int
    feed_forward_dimension: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    precision: str
    cpu: RuntimeDefaults
    gpu: RuntimeDefaults
    source_path: Path

    @classmethod
    def load(cls, path: Path) -> "ModelProfile":
        if not path.is_file():
            raise FileNotFoundError(f"model configuration does not exist: {path}")

        with path.open("r", encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)

        required = {
            "name",
            "sequence_length",
            "embedding_dimension",
            "layers",
            "attention_heads",
            "feed_forward_dimension",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "precision",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(
                "model configuration is missing required fields: "
                + ", ".join(missing)
            )

        default_batch_size = int(payload["batch_size"])
        default_precision = str(payload["precision"])
        runtimes = payload.get("runtime", {})

        profile = cls(
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
            course_iteration=int(payload.get("course_iteration", 0)),
            learning_objective=str(payload.get("learning_objective", "")),
            sequence_length=int(payload["sequence_length"]),
            embedding_dimension=int(payload["embedding_dimension"]),
            layers=int(payload["layers"]),
            attention_heads=int(payload["attention_heads"]),
            feed_forward_dimension=int(payload["feed_forward_dimension"]),
            batch_size=default_batch_size,
            learning_rate=float(payload["learning_rate"]),
            weight_decay=float(payload["weight_decay"]),
            precision=default_precision,
            cpu=RuntimeDefaults.from_payload(
                runtimes.get("cpu"),
                fallback_batch_size=min(default_batch_size, 2),
                fallback_precision="fp32",
            ),
            gpu=RuntimeDefaults.from_payload(
                runtimes.get("gpu"),
                fallback_batch_size=default_batch_size,
                fallback_precision=default_precision,
            ),
            source_path=path,
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        positive_values = {
            "sequence_length": self.sequence_length,
            "embedding_dimension": self.embedding_dimension,
            "layers": self.layers,
            "attention_heads": self.attention_heads,
            "feed_forward_dimension": self.feed_forward_dimension,
            "batch_size": self.batch_size,
        }
        for field_name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than zero")

        if self.course_iteration < 0:
            raise ValueError("course_iteration cannot be negative")
        if self.embedding_dimension % self.attention_heads != 0:
            raise ValueError(
                "embedding_dimension must be divisible by attention_heads"
            )
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.precision not in {"auto", "fp32", "bf16", "fp16"}:
            raise ValueError("precision must be auto, fp32, bf16, or fp16")

    def runtime_for(self, device_type: str) -> RuntimeDefaults:
        if device_type == "cuda":
            return self.gpu
        if device_type == "cpu":
            return self.cpu
        raise ValueError(f"unsupported device type: {device_type}")
