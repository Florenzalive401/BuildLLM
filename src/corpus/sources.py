"""Streaming corpus source adapters."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests
from .document import CorpusDocument


USER_AGENT = "BuildLLM-CorpusBuilder/1.0 (research corpus acquisition; contact: local-user)"


def _first_text(record: dict[str, Any], fields: list[str]) -> str | None:
    for field in fields:
        value: Any = record
        for part in field.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        if isinstance(value, str) and value.strip():
            return value
    return None


def stream_jsonl_source(config: dict[str, Any]) -> Iterator[CorpusDocument]:
    path = Path(config["path"])
    text_field = config.get("text_field", "text")
    source_name = config["name"]
    with path.open("r", encoding="utf8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get(text_field)
            if not isinstance(text, str) or not text.strip():
                continue
            yield CorpusDocument(
                text=text,
                source=source_name,
                title=record.get(config.get("title_field", "title")),
                document_id=str(record.get(config.get("id_field", "page_id"), line_number)),
                url=record.get(config.get("url_field", "url")),
                license=config.get("license"),
                metadata={"input_path": str(path)},
            )


def stream_text_source(config: dict[str, Any]) -> Iterator[CorpusDocument]:
    path = Path(config["path"])
    separator = config.get("document_separator", "\n\n")
    source_name = config["name"]
    text = path.read_text(encoding="utf8")
    for index, document in enumerate(text.split(separator), start=1):
        if document.strip():
            yield CorpusDocument(
                text=document,
                source=source_name,
                document_id=str(index),
                license=config.get("license"),
                metadata={"input_path": str(path)},
            )


def stream_huggingface_source(config: dict[str, Any]) -> Iterator[CorpusDocument]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("The datasets package is required for Hugging Face corpus sources") from error

    dataset = load_dataset(
        config["dataset"],
        name=config.get("configuration"),
        split=config.get("split", "train"),
        streaming=True,
        trust_remote_code=False,
    )
    shuffle_buffer = int(config.get("shuffle_buffer", 10_000))
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=int(config.get("seed", 42)), buffer_size=shuffle_buffer)

    text_fields = list(config.get("text_fields", ["text", "content", "body"]))
    title_fields = list(config.get("title_fields", ["title", "name"]))
    id_fields = list(config.get("id_fields", ["id", "document_id", "url"]))
    for index, record in enumerate(dataset, start=1):
        text = _first_text(record, text_fields)
        if text is None:
            continue
        title = _first_text(record, title_fields)
        document_id = _first_text(record, id_fields) or str(index)
        yield CorpusDocument(
            text=text,
            source=config["name"],
            title=title,
            document_id=str(document_id),
            url=record.get("url") if isinstance(record.get("url"), str) else None,
            license=config.get("license"),
            metadata={"dataset": config["dataset"], "configuration": config.get("configuration")},
        )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str | None:
    for child in element:
        if _local_name(child.tag) == name:
            return child.text
    return None


def _strip_rfc_boilerplate(text: str) -> str:
    text = text.replace("\f", "\n")
    text = re.sub(r"(?m)^\s*RFC \d+.*?\[Page \d+\]\s*$", "", text)
    text = re.sub(r"(?m)^\s*Internet Engineering Task Force.*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stream_rfc_source(config: dict[str, Any]) -> Iterator[CorpusDocument]:
    cache_dir = Path(config.get("cache_directory", "data/raw/rfc"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "rfc-index.xml"
    session = requests.Session()
    session.headers.update({"User-Agent": config.get("user_agent", USER_AGENT)})

    if not index_path.exists() or config.get("refresh_index", False):
        response = session.get("https://www.rfc-editor.org/rfc-index.xml", timeout=120)
        response.raise_for_status()
        index_path.write_bytes(response.content)

    root = ET.parse(index_path).getroot()
    include_obsolete = bool(config.get("include_obsolete", False))
    maximum_rfcs = config.get("maximum_documents")
    emitted = 0
    for entry in root.iter():
        if _local_name(entry.tag) != "rfc-entry":
            continue
        doc_id = _child_text(entry, "doc-id")
        title = _child_text(entry, "title")
        current_status = _child_text(entry, "current-status")
        obsoleted_by = next((child for child in entry if _local_name(child.tag) == "obsoleted-by"), None)
        if not doc_id or not re.fullmatch(r"RFC\d+", doc_id):
            continue
        if obsoleted_by is not None and not include_obsolete:
            continue
        number = int(doc_id[3:])
        local_path = cache_dir / f"rfc{number}.txt"
        url = f"https://www.rfc-editor.org/rfc/rfc{number}.txt"
        if not local_path.exists():
            response = session.get(url, timeout=120)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            local_path.write_bytes(response.content)
        text = _strip_rfc_boilerplate(local_path.read_text(encoding="utf8", errors="replace"))
        yield CorpusDocument(
            text=text,
            source=config["name"],
            title=title,
            document_id=doc_id,
            url=url,
            license=config.get("license", "IETF Trust Legal Provisions"),
            metadata={"status": current_status},
        )
        emitted += 1
        if maximum_rfcs is not None and emitted >= int(maximum_rfcs):
            break


def create_source(config: dict[str, Any]) -> Iterator[CorpusDocument]:
    source_type = config["type"]
    if source_type == "jsonl":
        return stream_jsonl_source(config)
    if source_type == "text":
        return stream_text_source(config)
    if source_type == "huggingface":
        return stream_huggingface_source(config)
    if source_type == "rfc":
        return stream_rfc_source(config)
    raise ValueError(f"Unsupported source type: {source_type}")
