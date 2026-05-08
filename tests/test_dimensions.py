"""Tests for individual drift dimensions."""

import math
import random
from datetime import datetime, timedelta

import pytest

from ragdrift import (
    EmbeddingDrift,
    JudgeDrift,
    QueryDistribution,
    ResponseQuality,
    RetrievalRelevance,
    Severity,
    Trace,
)


def _ts(i: int) -> datetime:
    return datetime(2026, 1, 1) + timedelta(minutes=i)


# ---------- QueryDistribution ----------


def test_query_distribution_no_drift_when_same_distribution():
    dim = QueryDistribution()
    centroid_seed = [[1.0, 0.0]] * 50
    dim.set_reference(centroid_seed)
    # Establish a baseline window of "normal" behavior — slightly off-centroid
    # but consistent.
    rng = random.Random(0)
    for _ in range(5):
        for _ in range(50):
            x = 1.0 + rng.gauss(0, 0.01)
            y = rng.gauss(0, 0.01)
            dim.update(Trace(timestamp=_ts(0), query_embedding=[x, y]))
        dim.commit_window()
    # Now feed a comparable window
    for _ in range(50):
        x = 1.0 + rng.gauss(0, 0.01)
        y = rng.gauss(0, 0.01)
        dim.update(Trace(timestamp=_ts(0), query_embedding=[x, y]))
    report = dim.report()
    assert report.severity is Severity.OK
    assert report.sample_size == 50


def test_query_distribution_flags_when_distribution_shifts():
    dim = QueryDistribution(degraded_z=2.0, warn_z=1.5)
    dim.set_reference([[1.0, 0.0]] * 20)
    rng = random.Random(1)
    for _ in range(8):
        for _ in range(40):
            dim.update(Trace(timestamp=_ts(0), query_embedding=[1.0 + rng.gauss(0, 0.001),
                                                                rng.gauss(0, 0.001)]))
        dim.commit_window()
    # Now drift hard — orthogonal direction
    for _ in range(40):
        dim.update(Trace(timestamp=_ts(0), query_embedding=[rng.gauss(0, 0.001),
                                                            1.0 + rng.gauss(0, 0.001)]))
    report = dim.report()
    assert report.severity is Severity.DEGRADED
    assert report.value is not None and report.value > 0.5


def test_query_distribution_no_traces_returns_ok():
    dim = QueryDistribution()
    dim.set_reference([[1.0, 0.0]])
    assert dim.report().severity is Severity.OK


# ---------- RetrievalRelevance ----------


def test_retrieval_relevance_drop_in_hit_rate_alerts():
    dim = RetrievalRelevance(metric="hit_rate", k=5, warn_z=1.5, degraded_z=2.5)
    # 8 baseline windows of 90% hit rate
    rng = random.Random(2)
    for _ in range(8):
        for _ in range(50):
            hits = [1] * 5 if rng.random() < 0.9 else [0] * 5
            dim.update(Trace(timestamp=_ts(0), relevance_labels=hits))
        dim.commit_window()
    # Now hit rate collapses
    for _ in range(50):
        dim.update(Trace(timestamp=_ts(0), relevance_labels=[0] * 5))
    report = dim.report()
    assert report.severity is Severity.DEGRADED
    assert report.value == 0.0


def test_retrieval_relevance_metrics():
    for metric, expected in [("hit_rate", 1.0), ("precision_at_k", 0.4), ("mrr", 1.0)]:
        dim = RetrievalRelevance(metric=metric, k=5)
        dim.update(Trace(timestamp=_ts(0), relevance_labels=[1, 0, 1, 0, 0]))
        r = dim.report()
        assert r.value == pytest.approx(expected)


def test_retrieval_relevance_unknown_metric_raises():
    dim = RetrievalRelevance(metric="totally-made-up", k=5)
    with pytest.raises(ValueError):
        dim.update(Trace(timestamp=_ts(0), relevance_labels=[1]))


# ---------- ResponseQuality ----------


def test_response_quality_worst_dimension_wins():
    dim = ResponseQuality(score_keys=["faithfulness", "relevance"], warn_z=1.0, degraded_z=2.0)
    # Establish baseline: faithfulness=0.95, relevance=0.95
    for _ in range(8):
        for _ in range(50):
            dim.update(Trace(timestamp=_ts(0),
                             judge_scores={"faithfulness": 0.95, "relevance": 0.95}))
        dim.commit_window()
    # Drop faithfulness only
    for _ in range(50):
        dim.update(Trace(timestamp=_ts(0),
                         judge_scores={"faithfulness": 0.5, "relevance": 0.95}))
    r = dim.report()
    assert r.severity is Severity.DEGRADED
    assert "faithfulness" in r.name


def test_response_quality_no_traces():
    dim = ResponseQuality()
    assert dim.report().severity is Severity.OK


# ---------- JudgeDrift ----------


def test_judge_drift_flags_systematic_shift():
    dim = JudgeDrift(score_key="faithfulness", warn_abs=0.05, degraded_abs=0.15)
    dim.set_reference({"ref-1": 0.9, "ref-2": 0.85, "ref-3": 0.92})
    # Today's judge scores them all 0.2 higher
    for ref_id, base in [("ref-1", 0.9), ("ref-2", 0.85), ("ref-3", 0.92)]:
        for _ in range(20):
            dim.update(Trace(timestamp=_ts(0),
                             judge_scores={"faithfulness": base + 0.2},
                             metadata={"reference_id": ref_id}))
    r = dim.report()
    assert r.severity is Severity.DEGRADED
    assert r.value is not None and r.value > 0.15


def test_judge_drift_stable_judge_ok():
    dim = JudgeDrift(score_key="faithfulness")
    dim.set_reference({"a": 0.9, "b": 0.8})
    for _ in range(40):
        dim.update(Trace(timestamp=_ts(0),
                         judge_scores={"faithfulness": 0.9},
                         metadata={"reference_id": "a"}))
        dim.update(Trace(timestamp=_ts(0),
                         judge_scores={"faithfulness": 0.8},
                         metadata={"reference_id": "b"}))
    assert dim.report().severity is Severity.OK


def test_judge_drift_ignores_unknown_reference():
    dim = JudgeDrift(score_key="faithfulness")
    dim.set_reference({"a": 0.9})
    dim.update(Trace(timestamp=_ts(0), judge_scores={"faithfulness": 0.5},
                     metadata={"reference_id": "not-in-reference"}))
    assert dim.report().sample_size == 0


# ---------- EmbeddingDrift ----------


def test_embedding_drift_basic():
    dim = EmbeddingDrift()
    dim.set_reference([[1.0, 0.0]] * 5)
    # Drift gradually
    for _ in range(8):
        for _ in range(30):
            dim.update(Trace(timestamp=_ts(0), query_embedding=[1.0, 0.001]))
        dim.commit_window()
    for _ in range(30):
        dim.update(Trace(timestamp=_ts(0), query_embedding=[0.0, 1.0]))
    r = dim.report()
    assert r.severity is Severity.DEGRADED
