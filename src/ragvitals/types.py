"""Shared types: Trace, Window, Severity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Severity(str, Enum):
    OK = "ok"
    WARN = "warn"
    DEGRADED = "degraded"


@dataclass
class Trace:
    """A single RAG event ingested into the detector.

    All fields are optional so a Trace can serve a heterogeneous pipeline (e.g.,
    an embedding-drift dimension only needs `query_embedding` while a
    retrieval-relevance dimension needs `retrieval_scores`). Dimensions report
    `INSUFFICIENT_DATA` when their required fields are missing.
    """

    timestamp: datetime
    query: str | None = None
    query_embedding: list[float] | None = None
    retrieved_doc_ids: list[str] | None = None
    retrieval_scores: list[float] | None = None
    relevance_labels: list[int] | None = None  # binary 0/1 per retrieved doc
    response: str | None = None
    judge_scores: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DimensionReport:
    name: str
    severity: Severity
    value: float | None
    baseline: float | None
    z_score: float | None
    sample_size: int
    detail: str = ""


@dataclass
class DetectorReport:
    window_start: datetime
    window_end: datetime
    dimensions: list[DimensionReport]

    @property
    def degraded(self) -> list[str]:
        return [d.name for d in self.dimensions if d.severity is Severity.DEGRADED]

    @property
    def warned(self) -> list[str]:
        return [d.name for d in self.dimensions if d.severity is Severity.WARN]

    @property
    def healthy(self) -> bool:
        return not self.degraded and not self.warned
