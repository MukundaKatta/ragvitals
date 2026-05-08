"""Tests for Detector composition + sinks."""

import json
import os
import tempfile
from datetime import datetime, timedelta

import pytest

from ragdrift import (
    Detector,
    InMemorySink,
    JSONLSink,
    JudgeDrift,
    QueryDistribution,
    ResponseQuality,
    RetrievalRelevance,
    Severity,
    Trace,
)


def _ts(i: int = 0) -> datetime:
    return datetime(2026, 1, 1) + timedelta(minutes=i)


def test_detector_composes_dimensions_and_emits_to_sink():
    rq = RetrievalRelevance(metric="hit_rate", k=5)
    sink = InMemorySink()
    det = Detector(dimensions=[rq], sinks=[sink])
    for _ in range(8):
        for _ in range(20):
            det.ingest(Trace(timestamp=_ts(0), relevance_labels=[1, 1, 0, 0, 0]))
        det.commit_window()
    for _ in range(20):
        det.ingest(Trace(timestamp=_ts(0), relevance_labels=[0] * 5))
    report = det.report()
    assert len(sink.history) == 1
    assert "RetrievalRelevance" in [d.name for d in report.dimensions]


def test_detector_jsonl_sink_writes_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.jsonl")
        sink = JSONLSink(path=path)
        det = Detector(dimensions=[RetrievalRelevance()], sinks=[sink])
        det.ingest(Trace(timestamp=_ts(0), relevance_labels=[1, 0, 0]))
        det.report()
        with open(path) as f:
            line = f.readline().strip()
        record = json.loads(line)
        assert "dimensions" in record
        assert any(d["name"] == "RetrievalRelevance" for d in record["dimensions"])


def test_detector_ingest_jsonl_round_trip():
    with tempfile.TemporaryDirectory() as d:
        traces_path = os.path.join(d, "traces.jsonl")
        with open(traces_path, "w") as f:
            for i in range(5):
                f.write(json.dumps({
                    "timestamp": _ts(i).isoformat(),
                    "relevance_labels": [1, 0, 0, 0, 0],
                    "judge_scores": {"faithfulness": 0.9},
                }) + "\n")
        det = Detector(dimensions=[RetrievalRelevance(), ResponseQuality()])
        det.ingest_jsonl(traces_path)
        report = det.report()
        rq = next(d for d in report.dimensions if d.name == "RetrievalRelevance")
        assert rq.sample_size == 5


def test_detector_report_healthy_property():
    det = Detector(dimensions=[RetrievalRelevance()])
    det.ingest(Trace(timestamp=_ts(0), relevance_labels=[1, 0, 0]))
    report = det.report()
    assert report.healthy
    assert not report.degraded
    assert not report.warned


def test_cloudwatch_sink_raises_clearly_without_boto():
    # We don't expect boto3 to be importable in the test env. If it is, skip.
    try:
        import boto3  # noqa
        pytest.skip("boto3 is installed; skipping the negative-import test")
    except ImportError:
        pass
    from ragdrift import CloudWatchSink, DetectorReport
    sink = CloudWatchSink(namespace="rag/test")
    report = DetectorReport(window_start=_ts(0), window_end=_ts(1), dimensions=[])
    with pytest.raises(ImportError) as exc:
        sink.emit(report)
    assert "boto3" in str(exc.value)
