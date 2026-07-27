import json
from pathlib import Path

from src.corpus.pipeline import CorpusPipeline


def test_pipeline_builds_jsonl_and_report(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    good = (
        "Cybersecurity engineering combines system design, threat analysis, testing, and operations. "
        "Teams use evidence to identify weaknesses, prioritize remediation, and verify that controls work. "
        "The process depends on clear documentation and repeatable technical validation. "
    ) * 8
    records = [
        {"title": "One", "page_id": 1, "text": good},
        {"title": "Duplicate", "page_id": 2, "text": good},
        {"title": "Short", "page_id": 3, "text": "short"},
    ]
    input_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf8")
    output_path = tmp_path / "output.jsonl"
    report_path = tmp_path / "report.json"
    config = {
        "output": {"corpus": str(output_path), "report": str(report_path)},
        "quality": {"minimum_score": 35, "minimum_characters": 300},
        "sources": [{
            "name": "test",
            "type": "jsonl",
            "path": str(input_path),
            "text_field": "text",
            "title_field": "title",
            "id_field": "page_id",
        }],
    }
    result = CorpusPipeline(config).run()
    assert result.documents_written == 1
    output_record = json.loads(output_path.read_text(encoding="utf8").strip())
    assert output_record["source"] == "test"
    assert output_record["quality_score"] >= 35
    report = json.loads(report_path.read_text(encoding="utf8"))
    assert report["totals"]["exact_duplicate"] == 1
    assert report["totals"]["documents_rejected"] == 1


def _write_distinct_records(path: Path, count: int) -> None:
    labels = [
        "alphaone", "bravotwo", "charliethree", "deltafour", "echofive",
        "foxtrotsix", "golfseven", "hoteleight", "indianine", "julietten",
    ]
    records = []
    for index in range(count):
        label = labels[index]
        text = (
            f"{label} cybersecurity engineering combines system design, testing, operations, "
            "evidence, remediation, governance, and technical controls. "
        ) * 12
        records.append({"title": label, "page_id": index, "text": text})
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf8")


def test_pipeline_stops_at_source_character_target(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    _write_distinct_records(input_path, 10)
    output_path = tmp_path / "output.jsonl"
    report_path = tmp_path / "report.json"
    config = {
        "output": {"corpus": str(output_path), "report": str(report_path)},
        "quality": {"minimum_score": 0, "minimum_characters": 100},
        "deduplication": {"near_duplicates": False},
        "sources": [{
            "name": "test",
            "type": "jsonl",
            "path": str(input_path),
            "text_field": "text",
            "title_field": "title",
            "id_field": "page_id",
            "target_characters": 2_000,
        }],
    }

    result = CorpusPipeline(config).run()
    report = json.loads(report_path.read_text(encoding="utf8"))

    assert result.documents_written == 2
    assert result.characters_written >= 2_000
    assert report["source_stop_reasons"]["test"] == "target_characters_reached"


def test_pipeline_stops_at_source_document_limit(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    _write_distinct_records(input_path, 10)
    output_path = tmp_path / "output.jsonl"
    report_path = tmp_path / "report.json"
    config = {
        "output": {"corpus": str(output_path), "report": str(report_path)},
        "quality": {"minimum_score": 0, "minimum_characters": 100},
        "deduplication": {"near_duplicates": False},
        "sources": [{
            "name": "test",
            "type": "jsonl",
            "path": str(input_path),
            "text_field": "text",
            "title_field": "title",
            "id_field": "page_id",
            "maximum_documents": 3,
        }],
    }

    result = CorpusPipeline(config).run()
    report = json.loads(report_path.read_text(encoding="utf8"))

    assert result.documents_written == 3
    assert report["source_stop_reasons"]["test"] == "maximum_documents_reached"
