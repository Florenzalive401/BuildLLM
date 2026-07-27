"""Build a curated, deduplicated, balanced multi source pretraining corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.corpus import CorpusPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/corpus_balanced.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"Corpus configuration does not exist: {args.config}")
    config = yaml.safe_load(args.config.read_text(encoding="utf8"))
    if not isinstance(config, dict):
        raise ValueError("Corpus configuration must contain a YAML mapping")
    result = CorpusPipeline(config).run()
    print(json.dumps({
        "corpus": str(result.output_path),
        "report": str(result.report_path),
        "documents_written": result.documents_written,
        "characters_written": result.characters_written,
    }, indent=2))


if __name__ == "__main__":
    main()
