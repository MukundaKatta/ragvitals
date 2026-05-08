"""ragdrift: 5-dimensional production drift detection for RAG systems."""

from ragdrift.detector import (
    CloudWatchSink,
    Detector,
    InMemorySink,
    JSONLSink,
    Sink,
)
from ragdrift.dimensions import (
    EmbeddingDrift,
    JudgeDrift,
    QueryDistribution,
    ResponseQuality,
    RetrievalRelevance,
)
from ragdrift.types import DetectorReport, DimensionReport, Severity, Trace

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
__version__ = "0.1.0"
