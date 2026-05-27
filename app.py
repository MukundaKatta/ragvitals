"""Streamlit dashboard for ragvitals — RAG drift detection live demo.

Deployed to Cloud Run so judges can interact with the library without
installing it locally. Generates synthetic Traces against pre-set
reference baselines, runs all five drift dimensions through Detector,
and renders the report.

Run locally:
    streamlit run app.py

Deploy to Cloud Run:
    gcloud run deploy ragvitals-demo --source . --region us-central1 \\
        --allow-unauthenticated --project geminilens-vertex-3010
"""

from __future__ import annotations

import os
import random
import textwrap
from datetime import datetime, timedelta

import streamlit as st

from ragvitals import (
    Detector,
    InMemorySink,
    JudgeDrift,
    QueryDistribution,
    ResponseQuality,
    RetrievalRelevance,
    Trace,
)

# Force the Cloud Run port environment to bind correctly.
PORT = int(os.getenv("PORT", "8080"))


st.set_page_config(
    page_title="ragvitals — RAG drift detection",
    layout="wide",
)

st.title("ragvitals — 5-dim RAG drift detector")
st.caption(
    "Library, not a platform. This page wraps the same `Detector` / `Trace` "
    "you import from PyPI. github.com/MukundaKatta/ragvitals"
)


# ---- reference fixtures --------------------------------------------------------

REFERENCE_EMBEDDINGS = [
    [0.10, 0.20, 0.30, 0.40],
    [0.12, 0.22, 0.31, 0.39],
    [0.09, 0.21, 0.29, 0.41],
    [0.11, 0.19, 0.28, 0.42],
    [0.10, 0.20, 0.31, 0.40],
]
REFERENCE_JUDGE_SCORES = {
    "ref-1": 0.92,
    "ref-2": 0.88,
    "ref-3": 0.95,
}


# ---- trace generator -----------------------------------------------------------


def synth_trace(
    *,
    drift_label: str,
    when: datetime,
    ref_id: str,
) -> Trace:
    """Build a synthetic Trace under one of three drift labels:
    'healthy', 'embedding-drift', 'retrieval-drift', 'judge-drift'."""
    if drift_label == "embedding-drift":
        emb = [v + random.uniform(0.35, 0.55) for v in REFERENCE_EMBEDDINGS[0]]
    else:
        base = random.choice(REFERENCE_EMBEDDINGS)
        emb = [v + random.uniform(-0.02, 0.02) for v in base]

    if drift_label == "retrieval-drift":
        # Almost everything irrelevant.
        labels = [0, 0, 0, 0, 0]
        scores = [random.uniform(0.05, 0.25) for _ in range(5)]
    else:
        labels = [1, 1, 0, 0, 0]
        scores = [random.uniform(0.70, 0.95) for _ in range(5)]

    judge = REFERENCE_JUDGE_SCORES[ref_id]
    if drift_label == "judge-drift":
        # Judge got a lot stricter.
        judge_scores = {
            "faithfulness": max(0.0, judge - random.uniform(0.30, 0.50)),
            "relevance": max(0.0, judge - random.uniform(0.30, 0.50)),
        }
    else:
        judge_scores = {
            "faithfulness": min(1.0, judge + random.uniform(-0.05, 0.05)),
            "relevance": min(1.0, judge + random.uniform(-0.05, 0.05)),
        }

    return Trace(
        timestamp=when,
        query=f"What is the policy for {ref_id}?",
        query_embedding=emb,
        retrieved_doc_ids=["d1", "d2", "d3", "d4", "d5"],
        retrieval_scores=scores,
        relevance_labels=labels,
        response=f"Synthetic response under {drift_label}",
        judge_scores=judge_scores,
        metadata={"reference_id": ref_id},
    )


def make_detector() -> Detector:
    q = QueryDistribution()
    q.set_reference(REFERENCE_EMBEDDINGS)
    j = JudgeDrift()
    j.set_reference(REFERENCE_JUDGE_SCORES)
    return Detector(
        dimensions=[
            q,
            RetrievalRelevance(metric="hit_rate", k=10),
            ResponseQuality(score_keys=["faithfulness", "relevance"]),
            j,
        ],
        sinks=[InMemorySink()],
    )


# ---- UI ------------------------------------------------------------------------


with st.sidebar:
    st.header("Settings")
    drift_label = st.selectbox(
        "Drift scenario",
        ["healthy", "embedding-drift", "retrieval-drift", "judge-drift"],
        index=0,
        help="What kind of degradation to inject into the synthetic traces.",
    )
    n_traces = st.slider("Traces to ingest", min_value=5, max_value=200, value=60)
    seed = st.number_input("RNG seed", min_value=0, max_value=10_000, value=42)
    run_button = st.button("Run drift check", type="primary", use_container_width=True)


if run_button:
    random.seed(int(seed))
    det = make_detector()
    now = datetime.utcnow()
    for i in range(n_traces):
        ref_id = f"ref-{(i % 3) + 1}"
        t = synth_trace(
            drift_label=drift_label,
            when=now - timedelta(minutes=(n_traces - i)),
            ref_id=ref_id,
        )
        det.ingest(t)

    report = det.report()

    col_a, col_b = st.columns(2)
    col_a.metric("Healthy?", "✅ yes" if report.healthy else "❌ NO")
    col_b.metric(
        "Degraded dimensions",
        ", ".join(report.degraded) if report.degraded else "none",
    )

    st.subheader("Per-dimension verdicts")
    for d in report.dimensions:
        with st.expander(f"{d.name}  ·  {d.severity.value}"):
            st.json(
                {
                    "severity": d.severity.value,
                    "value": d.value,
                    "baseline": d.baseline,
                    "z_score": d.z_score,
                    "sample_size": d.sample_size,
                    "detail": d.detail,
                }
            )

    st.subheader("Arize Phoenix export")
    st.markdown(
        "Send this report to your Phoenix project with the built-in "
        "`PhoenixSink` (one OpenTelemetry span per dimension):"
    )
    st.code(
        "from ragvitals import Detector, PhoenixSink\n"
        "\n"
        "det = Detector(\n"
        "    dimensions=[...],\n"
        "    sinks=[PhoenixSink(\n"
        "        endpoint='http://localhost:4317',\n"
        "        project_name='ragvitals-prod',\n"
        "    )],\n"
        ")",
        language="python",
    )

else:
    st.info(
        textwrap.dedent(
            """
            **What this is.** A live demo of the [ragvitals](https://github.com/MukundaKatta/ragvitals)
            Python library wrapped in a Streamlit dashboard so judges can
            interact with the drift detector without `pip install`-ing it.

            **What to try.**
            1. Pick a drift scenario in the sidebar (start with `healthy`, then
               try `embedding-drift` to see the QueryDistribution dimension flag).
            2. Click **Run drift check**.
            3. Inspect the per-dimension verdicts. Each is the same
               `Verdict` object the library returns in production.

            **Architecture.** Streamlit talks to ragvitals.Detector, which
            composes five dimension instances. The same code paths run here as
            in `pip install ragvitals`; only the trace source (synthetic vs
            live RAG pipeline) differs.

            **Arize integration.** Every `DetectorReport` can be streamed to
            Arize Phoenix via the built-in `PhoenixSink` (OpenTelemetry spans).
            Install with `pip install ragvitals[phoenix]` to enable it.
            """
        ).strip()
    )
