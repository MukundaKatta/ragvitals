"""Tests for Detector composition + sinks.

Uses only the Python standard library (``unittest``) so the suite runs with::

    python -m unittest discover -s tests

without any third-party test dependencies. The tests are still discovered and
run correctly by pytest as well.
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

# Make the package importable straight from a checkout if it isn't installed.
try:
    import ragvitals  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - import shim
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    )

from ragvitals import (  # noqa: E402
    Detector,
    InMemorySink,
    JSONLSink,
    ResponseQuality,
    RetrievalRelevance,
    Trace,
)


def _ts(i: int = 0) -> datetime:
    return datetime(2026, 1, 1) + timedelta(minutes=i)


class DetectorTest(unittest.TestCase):
    def test_detector_composes_dimensions_and_emits_to_sink(self):
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
        self.assertEqual(len(sink.history), 1)
        self.assertIn("RetrievalRelevance", [d.name for d in report.dimensions])

    def test_detector_jsonl_sink_writes_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.jsonl")
            sink = JSONLSink(path=path)
            det = Detector(dimensions=[RetrievalRelevance()], sinks=[sink])
            det.ingest(Trace(timestamp=_ts(0), relevance_labels=[1, 0, 0]))
            det.report()
            with open(path) as f:
                line = f.readline().strip()
            record = json.loads(line)
            self.assertIn("dimensions", record)
            self.assertTrue(
                any(d["name"] == "RetrievalRelevance" for d in record["dimensions"])
            )

    def test_detector_ingest_jsonl_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            traces_path = os.path.join(d, "traces.jsonl")
            with open(traces_path, "w") as f:
                for i in range(5):
                    f.write(
                        json.dumps(
                            {
                                "timestamp": _ts(i).isoformat(),
                                "relevance_labels": [1, 0, 0, 0, 0],
                                "judge_scores": {"faithfulness": 0.9},
                            }
                        )
                        + "\n"
                    )
            det = Detector(dimensions=[RetrievalRelevance(), ResponseQuality()])
            det.ingest_jsonl(traces_path)
            report = det.report()
            rq = next(d for d in report.dimensions if d.name == "RetrievalRelevance")
            self.assertEqual(rq.sample_size, 5)

    def test_detector_report_healthy_property(self):
        det = Detector(dimensions=[RetrievalRelevance()])
        det.ingest(Trace(timestamp=_ts(0), relevance_labels=[1, 0, 0]))
        report = det.report()
        self.assertTrue(report.healthy)
        self.assertEqual(report.degraded, [])
        self.assertEqual(report.warned, [])

    def test_cloudwatch_sink_raises_clearly_without_boto(self):
        # We don't expect boto3 to be importable in the test env. If it is, skip.
        try:
            import boto3  # noqa: F401

            self.skipTest("boto3 is installed; skipping the negative-import test")
        except ImportError:
            pass
        from ragvitals import CloudWatchSink, DetectorReport

        sink = CloudWatchSink(namespace="rag/test")
        report = DetectorReport(window_start=_ts(0), window_end=_ts(1), dimensions=[])
        with self.assertRaises(ImportError) as ctx:
            sink.emit(report)
        self.assertIn("boto3", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
