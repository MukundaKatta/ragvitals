"""Integrated drift dashboard for ragvitals (Streamlit version).

Same synthetic 500-document corpus as ``integrated_drift_dashboard.py``,
but driven by a Streamlit slider so judges can watch all five drift
dimensions react live as the index "ages".

Run locally
-----------
    pip install streamlit
    streamlit run examples/streamlit_drift_dashboard.py

Or, against the existing Cloud Run deployment, just open the URL printed
by ``gcloud run services list``.

What's on screen
----------------
- One slider (top): ``days_aged``, 0 to 30.
- Five live-updating panels, one per detector: QueryDistribution,
  RetrievalRelevance, EmbeddingDrift, ResponseQuality, JudgeDrift.
- A "DriftReport" summary panel at the bottom: degraded list, warned
  list, healthy flag, plus the same latency p50 / p95 the CLI version
  prints. The summary reflects exactly what ``Detector.report()``
  returned, with no UI-side massaging.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import streamlit as st

from ragvitals import Severity

# We deliberately import the simulation, detector wiring, and corpus
# builder from the CLI demo to keep one source of truth. The Streamlit
# entrypoint is a thin UI shell on top.
_HERE = Path(__file__).resolve().parent
_CLI_PATH = _HERE / "integrated_drift_dashboard.py"
_spec = importlib.util.spec_from_file_location("integrated_drift_demo", _CLI_PATH)
assert _spec is not None and _spec.loader is not None
demo = importlib.util.module_from_spec(_spec)
sys.modules["integrated_drift_demo"] = demo
_spec.loader.exec_module(demo)


# ---- page ---------------------------------------------------------------------

st.set_page_config(page_title="ragvitals integrated drift", layout="wide")
st.title("ragvitals: integrated 5-dimension drift demo")
st.caption(
    "One synthetic 500-doc corpus. One slider that ages the index. Five "
    "drift dimensions wired through the same `Detector` / `Trace` "
    "objects you import from PyPI. github.com/MukundaKatta/ragvitals"
)


with st.sidebar:
    st.header("Simulation")
    seed = st.number_input("RNG seed", min_value=0, max_value=10_000, value=42)
    n_docs = st.slider("Corpus size", min_value=100, max_value=1000, value=500, step=100)
    days_aged = st.slider(
        "days_aged",
        min_value=0,
        max_value=30,
        value=0,
        help="How many simulated days of drift to apply to the query distribution, "
             "judge scores, retrieval staleness, and latency.",
    )


# ---- run ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_corpus(seed: int, n_docs: int):
    return demo.build_corpus(seed=int(seed), n_docs=int(n_docs))


corpus = _cached_corpus(int(seed), int(n_docs))
row = demo.run_age(int(days_aged), corpus, seed=int(seed))


# ---- per-dimension panels ----------------------------------------------------


def severity_color(label: str) -> str:
    if "[degraded]" in label:
        return "red"
    if "[warn]" in label:
        return "orange"
    return "green"


cols = st.columns(5)
panels = [
    ("QueryDistribution",            "Mean cosine distance from reference centroid"),
    ("RetrievalRelevance",           "hit_rate@10 vs trailing baseline"),
    ("EmbeddingDrift",               "Mean centroid drift"),
    ("ResponseQuality.faithfulness", "LLM-judge faithfulness vs baseline"),
    ("JudgeDrift",                   "Mean judge delta on frozen reference set"),
]
for col, (name, helptext) in zip(cols, panels):
    val = row.metrics.get(name, "n/a")
    color = severity_color(val)
    with col:
        st.metric(label=name, value=val.split(" ")[0] if val != "n/a" else "n/a")
        st.caption(helptext)
        st.markdown(f":{color}[**{val}**]")


# ---- summary -----------------------------------------------------------------


st.divider()
st.subheader("DriftReport summary")

summary_cols = st.columns(4)
summary_cols[0].metric(
    "Healthy?",
    "yes" if (not row.degraded and not row.warned) else "no",
)
summary_cols[1].metric("Degraded dims", len(row.degraded))
summary_cols[2].metric("latency p50 (ms)", f"{row.latencies_p50_ms:.0f}")
summary_cols[3].metric("latency p95 (ms)", f"{row.latencies_p95_ms:.0f}")

st.write("**Degraded:**", ", ".join(row.degraded) if row.degraded else "_none_")
st.write("**Warned:**",   ", ".join(row.warned)   if row.warned   else "_none_")


with st.expander("How the simulation works"):
    st.markdown(
        """
        Move the `days_aged` slider and watch all five dimensions react.

        Behind the scenes:
        - A fresh `Detector` is built per slider value so each reading
          is independent (no cross-contamination).
        - Seven days of healthy baseline traffic are streamed through
          the detector first to warm the rolling baselines.
        - Then one day of "aged" traffic is streamed in:
            1. Query distribution shifts as the emergent topic
               (`Q2 reorg`) ramps in.
            2. Retrievals stale because older docs decay.
            3. Judge scores drift downward on the frozen reference set.
            4. Latencies climb (simulated infra degradation).
        - The detector emits one `DetectorReport` and we render its
          per-dimension verdicts above.

        The CLI version (`examples/integrated_drift_dashboard.py`)
        prints the same numbers as an ASCII table for screenshots and
        CI logs.
        """
    )
