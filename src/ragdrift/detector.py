"""Detector: orchestrate the dimensions, ingest traces, emit reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Iterator, Protocol

from ragdrift.types import DetectorReport, DimensionReport, Severity, Trace


class _Dimension(Protocol):
    name: str
    def update(self, trace: Trace) -> None: ...
    def report(self) -> DimensionReport: ...
    def commit_window(self) -> None: ...


class Sink(Protocol):
    def emit(self, report: DetectorReport) -> None: ...


@dataclass
class InMemorySink:
    history: list[DetectorReport] = field(default_factory=list)

    def emit(self, report: DetectorReport) -> None:
        self.history.append(report)


@dataclass
class JSONLSink:
    path: str

    def emit(self, report: DetectorReport) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(_report_to_dict(report)) + "\n")


@dataclass
class CloudWatchSink:
    """Stub: emits via boto3 cloudwatch.put_metric_data when boto3 is present.

    Kept as a thin wrapper so v0.1 ships without a hard boto dependency. If
    boto3 isn't installed at the call site, this raises a clear ImportError.
    """

    namespace: str

    def emit(self, report: DetectorReport) -> None:
        try:
            import boto3  # type: ignore
        except ImportError as e:
            raise ImportError(
                "CloudWatchSink requires boto3. Install with `pip install boto3`."
            ) from e
        cw = boto3.client("cloudwatch")
        for d in report.dimensions:
            if d.value is None:
                continue
            cw.put_metric_data(
                Namespace=self.namespace,
                MetricData=[{
                    "MetricName": d.name,
                    "Value": d.value,
                    "Timestamp": report.window_end,
                }],
            )


@dataclass
class Detector:
    """Compose dimensions and sinks; ingest traces; emit reports.

    Typical lifecycle:
        det = Detector(dimensions=[QueryDistribution(), RetrievalRelevance()],
                       sinks=[InMemorySink()])
        for trace in stream:
            det.ingest(trace)
        report = det.report()      # current window
        det.commit_window()        # roll baselines forward
    """

    dimensions: list[_Dimension]
    sinks: list[Sink] = field(default_factory=list)
    _window_start: datetime = field(default_factory=datetime.now)
    _last_trace_ts: datetime | None = None

    def ingest(self, trace: Trace) -> None:
        self._last_trace_ts = trace.timestamp
        for d in self.dimensions:
            d.update(trace)

    def ingest_jsonl(self, path: str) -> None:
        for trace in _read_jsonl_traces(path):
            self.ingest(trace)

    def report(self) -> DetectorReport:
        report = DetectorReport(
            window_start=self._window_start,
            window_end=self._last_trace_ts or datetime.now(),
            dimensions=[d.report() for d in self.dimensions],
        )
        for s in self.sinks:
            s.emit(report)
        return report

    def commit_window(self) -> None:
        for d in self.dimensions:
            d.commit_window()
        self._window_start = datetime.now()


# ---------- helpers ----------


def _report_to_dict(r: DetectorReport) -> dict:
    return {
        "window_start": r.window_start.isoformat(),
        "window_end": r.window_end.isoformat(),
        "degraded": r.degraded,
        "warned": r.warned,
        "dimensions": [
            {
                "name": d.name,
                "severity": d.severity.value,
                "value": d.value,
                "baseline": d.baseline,
                "z_score": d.z_score,
                "sample_size": d.sample_size,
                "detail": d.detail,
            }
            for d in r.dimensions
        ],
    }


def _read_jsonl_traces(path: str) -> Iterator[Trace]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            yield Trace(
                timestamp=datetime.fromisoformat(d["timestamp"]),
                query=d.get("query"),
                query_embedding=d.get("query_embedding"),
                retrieved_doc_ids=d.get("retrieved_doc_ids"),
                retrieval_scores=d.get("retrieval_scores"),
                relevance_labels=d.get("relevance_labels"),
                response=d.get("response"),
                judge_scores=d.get("judge_scores"),
                metadata=d.get("metadata") or {},
            )
