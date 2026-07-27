from pathlib import Path

import pytest
import yaml


def test_learning_corpus_config_is_bounded_and_ordered() -> None:
    config_path = Path("configs/corpus_learning_50m.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf8"))

    assert config["limits"]["maximum_characters"] == 50_000_000
    assert config["output"] == {
        "corpus": "data/processed/training_corpus_learning_50m.jsonl",
        "report": "data/reports/training_corpus_learning_50m_report.json",
    }

    enabled_sources = [
        source for source in config["sources"] if source.get("enabled", True)
    ]
    assert [source["name"] for source in enabled_sources] == [
        "wikipedia",
        "rfc",
        "fineweb_edu",
    ]
    assert [
        source["target_characters"] for source in enabled_sources
    ] == [15_000_000, 10_000_000, 25_000_000]
    assert sum(
        source["target_characters"] for source in enabled_sources
    ) == config["limits"]["maximum_characters"]
    assert enabled_sources[0]["path"] == (
        "data/processed/wikipedia_simple.jsonl"
    )
    assert config["deduplication"]["near_duplicates"] is True
    assert config["balancing"]["enabled"] is True
    assert sum(config["balancing"]["topic_weights"].values()) == pytest.approx(
        1.0
    )


def test_pipeline_verification_config_uses_explicit_artifact_names() -> None:
    config = yaml.safe_load(
        Path("configs/corpus_pipeline_verification.yaml").read_text(
            encoding="utf8"
        )
    )

    assert config["limits"]["maximum_documents"] == 1_000
    assert config["output"] == {
        "corpus": (
            "data/processed/"
            "training_corpus_pipeline_verification.jsonl"
        ),
        "report": (
            "data/reports/"
            "training_corpus_pipeline_verification_report.json"
        ),
    }


def test_balanced_config_uses_explicit_artifact_names() -> None:
    config = yaml.safe_load(
        Path("configs/corpus_balanced.yaml").read_text(encoding="utf8")
    )

    assert config["output"] == {
        "corpus": "data/processed/training_corpus_balanced.jsonl",
        "report": "data/reports/training_corpus_balanced_report.json",
    }
