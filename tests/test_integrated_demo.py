"""Tests for the integrated drift dashboard example.

These cover the load-bearing demo entry points so that future library
changes can't silently break the dashboard. The CLI is shipped under
``examples/`` to stay outside the installed wheel, so we import it via a
file path rather than the package namespace.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_PATH = REPO_ROOT / "examples" / "integrated_drift_dashboard.py"


@pytest.fixture(scope="module")
def demo():
    """Load examples/integrated_drift_dashboard.py as a module."""
    spec = importlib.util.spec_from_file_location("integrated_drift_demo", DEMO_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["integrated_drift_demo"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_corpus_is_deterministic_given_seed(demo):
    a = demo.build_corpus(seed=42, n_docs=100)
    b = demo.build_corpus(seed=42, n_docs=100)
    assert len(a) == len(b) == 100
    # Two builds with the same seed must produce byte-identical embeddings.
    # If they don't, the demo isn't reproducible and the dev.to post's
    # output sample is a lie.
    for da, db in zip(a, b):
        assert da.doc_id == db.doc_id
        assert da.topic == db.topic
        assert da.embedding == db.embedding


def test_run_age_zero_is_healthy(demo):
    corpus = demo.build_corpus(seed=42, n_docs=200)
    row = demo.run_age(0, corpus, seed=42)
    # age=0 means the comparison window mirrors the baseline conditions;
    # no dimension should be degraded.
    assert row.degraded == [], f"unexpected degradation at age=0: {row.degraded}"


def test_run_age_thirty_shows_widespread_drift(demo):
    corpus = demo.build_corpus(seed=42, n_docs=200)
    row = demo.run_age(30, corpus, seed=42)
    # The whole point of the demo is that age=30 trips most of the
    # detectors. If this stops being true, the dashboard story is broken.
    assert len(row.degraded) >= 3, (
        f"expected >=3 degraded dimensions at age=30, got {row.degraded}"
    )
    # Latency p50 has to climb visibly between age=0 and age=30 too.
    age0 = demo.run_age(0, corpus, seed=42)
    assert row.latencies_p50_ms > age0.latencies_p50_ms * 2


def test_render_table_includes_all_dimensions(demo):
    corpus = demo.build_corpus(seed=42, n_docs=100)
    rows = [demo.run_age(a, corpus, seed=42) for a in (0, 15, 30)]
    table = demo.render_table(rows)
    for dim_name in (
        "QueryDistribution",
        "RetrievalRelevance",
        "EmbeddingDrift",
        "ResponseQuality",
        "JudgeDrift",
        "latency p50",
        "latency p95",
    ):
        assert dim_name in table, f"{dim_name} missing from rendered table"


def test_main_runs_end_to_end(demo):
    buf = io.StringIO()
    with redirect_stdout(buf):
        demo.main(["--ages", "0", "15", "30", "--seed", "7", "--n-docs", "120"])
    out = buf.getvalue()
    assert "ragvitals integrated drift demo" in out
    assert "QueryDistribution" in out
    assert "age= 0" in out
    assert "age=30" in out
