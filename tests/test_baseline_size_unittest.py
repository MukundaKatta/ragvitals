"""Standard-library ``unittest`` coverage for ragvitals.

These tests deliberately use only the Python standard library (no pytest),
so they run with::

    python -m unittest discover -s tests

in any environment, including ones without third-party test dependencies
installed. They complement the existing pytest suite and exercise the real
public API.

The headline test here is a regression guard for the ``baseline_size`` bug:
``QueryDistribution``, ``RetrievalRelevance``, and ``EmbeddingDrift`` accept a
``baseline_size`` argument, but it used to be silently ignored because the
rolling baseline was built with a hard-coded size in a ``default_factory``
lambda that could not see the instance's field. The fix wires the field
through ``__post_init__``; these tests pin that behavior down.
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta

# Make the package importable whether or not it has been ``pip install``-ed.
# CI installs it (``pip install -e .``), but running the suite straight from a
# checkout should also work.
try:
    import ragvitals  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - import shim
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    )

from ragvitals import (  # noqa: E402
    Detector,
    EmbeddingDrift,
    InMemorySink,
    JudgeDrift,
    QueryDistribution,
    ResponseQuality,
    RetrievalRelevance,
    Severity,
    Trace,
)


def _ts(i: int = 0) -> datetime:
    return datetime(2026, 1, 1) + timedelta(minutes=i)


class BaselineSizeRespectedTest(unittest.TestCase):
    """Regression: a user-supplied ``baseline_size`` must actually be used."""

    def test_query_distribution_honors_baseline_size(self):
        dim = QueryDistribution(baseline_size=42)
        self.assertEqual(dim.baseline_size, 42)
        self.assertEqual(dim._baseline.size, 42)

    def test_retrieval_relevance_honors_baseline_size(self):
        dim = RetrievalRelevance(baseline_size=500)
        self.assertEqual(dim.baseline_size, 500)
        self.assertEqual(dim._baseline.size, 500)

    def test_embedding_drift_honors_baseline_size(self):
        dim = EmbeddingDrift(baseline_size=7)
        self.assertEqual(dim.baseline_size, 7)
        self.assertEqual(dim._baseline.size, 7)

    def test_defaults_are_unchanged(self):
        self.assertEqual(QueryDistribution()._baseline.size, 1000)
        self.assertEqual(RetrievalRelevance()._baseline.size, 100)
        self.assertEqual(EmbeddingDrift()._baseline.size, 1000)

    def test_baseline_actually_caps_at_configured_size(self):
        # Push more committed windows than the baseline can hold; the rolling
        # buffer must never exceed ``baseline_size``. This proves the size is
        # wired into the real buffer, not just stored on the field.
        dim = RetrievalRelevance(metric="hit_rate", k=5, baseline_size=3)
        for _ in range(10):
            dim.update(Trace(timestamp=_ts(0), relevance_labels=[1, 0, 0, 0, 0]))
            dim.commit_window()
        self.assertEqual(len(dim._baseline.values), 3)


class RetrievalRelevanceMetricsTest(unittest.TestCase):
    def test_metric_values(self):
        cases = {"hit_rate": 1.0, "precision_at_k": 0.4, "mrr": 1.0}
        for metric, expected in cases.items():
            with self.subTest(metric=metric):
                dim = RetrievalRelevance(metric=metric, k=5)
                dim.update(Trace(timestamp=_ts(0), relevance_labels=[1, 0, 1, 0, 0]))
                self.assertAlmostEqual(dim.report().value, expected)

    def test_unknown_metric_raises(self):
        dim = RetrievalRelevance(metric="totally-made-up", k=5)
        with self.assertRaises(ValueError):
            dim.update(Trace(timestamp=_ts(0), relevance_labels=[1]))

    def test_no_traces_reports_ok_with_empty_sample(self):
        dim = RetrievalRelevance()
        report = dim.report()
        self.assertIs(report.severity, Severity.OK)
        self.assertEqual(report.sample_size, 0)
        self.assertIsNone(report.value)


class JudgeDriftTest(unittest.TestCase):
    def test_systematic_shift_is_flagged(self):
        dim = JudgeDrift(score_key="faithfulness", warn_abs=0.05, degraded_abs=0.15)
        dim.set_reference({"ref-1": 0.9, "ref-2": 0.85})
        for ref_id, base in [("ref-1", 0.9), ("ref-2", 0.85)]:
            for _ in range(20):
                dim.update(
                    Trace(
                        timestamp=_ts(0),
                        judge_scores={"faithfulness": base + 0.2},
                        metadata={"reference_id": ref_id},
                    )
                )
        report = dim.report()
        self.assertIs(report.severity, Severity.DEGRADED)
        self.assertIsNotNone(report.value)
        self.assertGreater(report.value, 0.15)

    def test_unknown_reference_is_ignored(self):
        dim = JudgeDrift(score_key="faithfulness")
        dim.set_reference({"a": 0.9})
        dim.update(
            Trace(
                timestamp=_ts(0),
                judge_scores={"faithfulness": 0.5},
                metadata={"reference_id": "not-in-reference"},
            )
        )
        self.assertEqual(dim.report().sample_size, 0)


class ResponseQualityTest(unittest.TestCase):
    def test_worst_named_score_wins(self):
        dim = ResponseQuality(
            score_keys=["faithfulness", "relevance"], warn_z=1.0, degraded_z=2.0
        )
        for _ in range(8):
            for _ in range(50):
                dim.update(
                    Trace(
                        timestamp=_ts(0),
                        judge_scores={"faithfulness": 0.95, "relevance": 0.95},
                    )
                )
            dim.commit_window()
        for _ in range(50):
            dim.update(
                Trace(
                    timestamp=_ts(0),
                    judge_scores={"faithfulness": 0.5, "relevance": 0.95},
                )
            )
        report = dim.report()
        self.assertIs(report.severity, Severity.DEGRADED)
        self.assertIn("faithfulness", report.name)

    def test_baseline_size_propagates_to_each_named_score(self):
        dim = ResponseQuality(score_keys=["faithfulness", "relevance"], baseline_size=5)
        for b in dim._baselines.values():
            self.assertEqual(b.size, 5)


class DetectorIntegrationTest(unittest.TestCase):
    def test_detector_emits_one_report_per_call_to_each_sink(self):
        sink = InMemorySink()
        det = Detector(
            dimensions=[RetrievalRelevance(metric="hit_rate", k=5)], sinks=[sink]
        )
        det.ingest(Trace(timestamp=_ts(0), relevance_labels=[1, 0, 0, 0, 0]))
        report = det.report()
        self.assertEqual(len(sink.history), 1)
        self.assertIn("RetrievalRelevance", [d.name for d in report.dimensions])

    def test_report_healthy_when_nothing_drifts(self):
        det = Detector(dimensions=[RetrievalRelevance()])
        det.ingest(Trace(timestamp=_ts(0), relevance_labels=[1, 0, 0]))
        report = det.report()
        self.assertTrue(report.healthy)
        self.assertEqual(report.degraded, [])
        self.assertEqual(report.warned, [])


if __name__ == "__main__":
    unittest.main()
