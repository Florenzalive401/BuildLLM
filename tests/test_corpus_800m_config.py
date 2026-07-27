from pathlib import Path

import yaml


def test_800m_corpus_config_targets_800m_characters():
    config_path = Path("configs/corpus_800m.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf8"))

    assert config["limits"]["maximum_characters"] == 800_000_000
    assert config["output"] == {
        "corpus": "data/processed/training_corpus_800m.jsonl",
        "report": "data/reports/training_corpus_800m_report.json",
    }
    enabled = [source for source in config["sources"] if source.get("enabled", True)]
    assert [source["name"] for source in enabled] == ["wikipedia", "rfc", "fineweb_edu"]
    assert sum(source["target_characters"] for source in enabled) >= 800_000_000
