"""Common document representation used by every corpus source."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CorpusDocument:
    text: str
    source: str
    title: str | None = None
    document_id: str | None = None
    url: str | None = None
    license: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    quality_score: float | None = None
    topic: str | None = None

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["metadata"] = {key: value for key, value in self.metadata.items() if value is not None}
        return {key: value for key, value in record.items() if value is not None}
