"""ragvitals: 5-dimensional production drift detection for RAG systems."""

from ragvitals.detector import (
    CloudWatchSink,
    Detector,
    InMemorySink,
    JSONLSink,
    Sink,
)
from ragvitals.dimensions import (
    EmbeddingDrift,
    JudgeDrift,
    QueryDistribution,
    ResponseQuality,
    RetrievalRelevance,
)
from ragvitals.types import DetectorReport, DimensionReport, Severity, Trace

__all__ = [
    "Detector",
    "Trace",
    "Severity",
    "DimensionReport",
    "DetectorReport",
    "QueryDistribution",
    "RetrievalRelevance",
    "EmbeddingDrift",
    "ResponseQuality",
    "JudgeDrift",
    "Sink",
    "InMemorySink",
    "JSONLSink",
    "CloudWatchSink",
]
__version__ = "0.2.0"
