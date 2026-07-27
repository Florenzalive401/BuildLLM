"""End to end corpus construction pipeline."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.text_cleaner import normalize_unicode, normalize_whitespace, remove_control_characters

from .dedup import DuplicateIndex
from .document import CorpusDocument
from .quality import assess_quality
from .sources import create_source
from .topics import classify_topic


@dataclass(frozen=True, slots=True)
class PipelineResult:
    output_path: Path
    report_path: Path
    documents_written: int
    characters_written: int


class CorpusPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        output = config.get("output", {})
        self.output_path = Path(
            output.get(
                "corpus",
                "data/processed/training_corpus_balanced.jsonl",
            )
        )
        self.report_path = Path(
            output.get(
                "report",
                "data/reports/training_corpus_balanced_report.json",
            )
        )
        quality = config.get("quality", {})
        self.minimum_score = float(quality.get("minimum_score", 55))
        self.minimum_characters = int(quality.get("minimum_characters", 500))
        self.maximum_characters = int(quality.get("maximum_characters", 2_000_000))
        dedup = config.get("deduplication", {})
        self.duplicate_index = DuplicateIndex(int(dedup.get("maximum_hamming_distance", 3)))
        self.near_dedup_enabled = bool(dedup.get("near_duplicates", True))
        self.seed = int(config.get("seed", 1729))
        balancing = config.get("balancing", {})
        self.topic_balancing_enabled = bool(balancing.get("enabled", False))
        self.topic_weights = {str(key): float(value) for key, value in balancing.get("topic_weights", {}).items()}
        self.topic_overflow_factor = float(balancing.get("overflow_factor", 1.10))

    @staticmethod
    def _clean(text: str) -> str:
        return normalize_whitespace(remove_control_characters(normalize_unicode(text)))

    def run(self) -> PipelineResult:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        totals = Counter()
        source_stats: dict[str, Counter] = defaultdict(Counter)
        topic_stats: Counter = Counter()
        topic_characters: Counter = Counter()
        rejection_reasons: Counter = Counter()
        source_stop_reasons: dict[str, str] = {}
        global_character_limit = self.config.get("limits", {}).get("maximum_characters")
        global_document_limit = self.config.get("limits", {}).get("maximum_documents")

        enabled_sources = [source for source in self.config.get("sources", []) if source.get("enabled", True)]
        if not enabled_sources:
            raise ValueError("At least one corpus source must be enabled")

        with self.output_path.open("w", encoding="utf8") as output:
            for source_config in enabled_sources:
                source_name = source_config["name"]
                source_character_limit = source_config.get("target_characters")
                source_document_limit = source_config.get("maximum_documents")
                progress_by_characters = source_character_limit is not None
                progress_total = (
                    int(source_character_limit)
                    if progress_by_characters
                    else int(source_document_limit) if source_document_limit is not None else None
                )
                progress = tqdm(
                    desc=f"Building {source_name}",
                    total=progress_total,
                    unit="char" if progress_by_characters else "doc",
                    unit_scale=progress_by_characters,
                )
                stop_reason = "source_exhausted"
                try:
                    for document in create_source(source_config):
                        totals["documents_examined"] += 1
                        source_stats[source_name]["documents_examined"] += 1
                        cleaned = self._clean(document.text)
                        assessment = assess_quality(
                            cleaned,
                            minimum_score=float(source_config.get("minimum_quality_score", self.minimum_score)),
                            minimum_characters=self.minimum_characters,
                            maximum_characters=self.maximum_characters,
                        )
                        if not assessment.accepted:
                            totals["documents_rejected"] += 1
                            source_stats[source_name]["documents_rejected"] += 1
                            for reason in assessment.reasons:
                                rejection_reasons[reason] += 1
                            continue

                        duplicate_type = self.duplicate_index.classify(cleaned)
                        if duplicate_type == "exact_duplicate" or (duplicate_type == "near_duplicate" and self.near_dedup_enabled):
                            totals[duplicate_type] += 1
                            source_stats[source_name][duplicate_type] += 1
                            continue

                        document.text = cleaned
                        document.quality_score = assessment.score
                        document.topic = classify_topic(cleaned)
                        characters = len(cleaned)
                        if self.topic_balancing_enabled and global_character_limit and self.topic_weights:
                            weight = self.topic_weights.get(document.topic, self.topic_weights.get("general", 0.0))
                            topic_limit = int(float(global_character_limit) * weight * self.topic_overflow_factor)
                            if topic_limit > 0 and topic_characters[document.topic] + characters > topic_limit:
                                totals["topic_quota_rejected"] += 1
                                source_stats[source_name]["topic_quota_rejected"] += 1
                                rejection_reasons["topic_quota_reached"] += 1
                                continue
                        document.metadata["quality_metrics"] = assessment.metrics
                        output.write(json.dumps(document.to_record(), ensure_ascii=False) + "\n")

                        totals["documents_written"] += 1
                        totals["characters_written"] += characters
                        source_stats[source_name]["documents_written"] += 1
                        source_stats[source_name]["characters_written"] += characters
                        topic_stats[document.topic] += 1
                        topic_characters[document.topic] += characters
                        progress.update(characters if progress_by_characters else 1)
                        progress.set_postfix(
                            docs=f"{source_stats[source_name]['documents_written']:,}",
                            chars=f"{source_stats[source_name]['characters_written']:,}",
                            score=assessment.score,
                        )

                        if source_document_limit is not None and source_stats[source_name]["documents_written"] >= int(source_document_limit):
                            stop_reason = "maximum_documents_reached"
                            break
                        if source_character_limit is not None and source_stats[source_name]["characters_written"] >= int(source_character_limit):
                            stop_reason = "target_characters_reached"
                            break
                        if global_document_limit is not None and totals["documents_written"] >= int(global_document_limit):
                            stop_reason = "global_maximum_documents_reached"
                            break
                        if global_character_limit is not None and totals["characters_written"] >= int(global_character_limit):
                            stop_reason = "global_maximum_characters_reached"
                            break
                finally:
                    progress.close()
                    source_stop_reasons[source_name] = stop_reason

                if global_document_limit is not None and totals["documents_written"] >= int(global_document_limit):
                    break
                if global_character_limit is not None and totals["characters_written"] >= int(global_character_limit):
                    break

        report = {
            "version": 1,
            "output": str(self.output_path),
            "elapsed_seconds": round(time.time() - started, 2),
            "totals": dict(totals),
            "sources": {name: dict(stats) for name, stats in source_stats.items()},
            "source_stop_reasons": source_stop_reasons,
            "topics": dict(topic_stats.most_common()),
            "topic_characters": dict(topic_characters.most_common()),
            "rejection_reasons": dict(rejection_reasons.most_common()),
            "configuration": self.config,
        }
        self.report_path.write_text(json.dumps(report, indent=2), encoding="utf8")
        return PipelineResult(
            output_path=self.output_path,
            report_path=self.report_path,
            documents_written=int(totals["documents_written"]),
            characters_written=int(totals["characters_written"]),
        )
