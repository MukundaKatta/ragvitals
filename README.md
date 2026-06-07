# ragvitals

[![ci](https://github.com/MukundaKatta/ragvitals/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/ragvitals/actions/workflows/ci.yml)
[![pypi](https://img.shields.io/pypi/v/ragvitals.svg)](https://pypi.org/project/ragvitals/)
[![python](https://img.shields.io/pypi/pyversions/ragvitals.svg)](https://pypi.org/project/ragvitals/)

Five-dimensional production drift detection for RAG systems. Library, not a platform — bring your own time-series store.

## Why

Production RAG rots in five dimensions:

1. **Query distribution** — users start asking different questions
2. **Retrieval relevance** — top-k recall silently falls after a re-index
3. **Embedding drift** — corpus or query embeddings shift vs the snapshot you tuned on
4. **Response quality** — LLM-as-judge scores degrade
5. **Judge drift** — the judge itself drifts, and you can't tell whether the system improved or the ruler moved

Existing tools cover one or two of these. `ragvitals` composes the five with the same time-series store, alarming, and replay path. No platform lock-in.

## Install

```bash
pip install ragvitals
# optional: CloudWatch sink
pip install "ragvitals[aws]"
```

## Quickstart

```python
from datetime import datetime
from ragvitals import (
    Detector, Trace,
    QueryDistribution, RetrievalRelevance, ResponseQuality, JudgeDrift,
    InMemorySink,
)

# Reference set: queries the system was tuned on
reference_embeddings = [...]
reference_judge_scores = {"ref-1": 0.92, "ref-2": 0.88, "ref-3": 0.95}

q = QueryDistribution(); q.set_reference(reference_embeddings)
j = JudgeDrift(); j.set_reference(reference_judge_scores)

det = Detector(
    dimensions=[
        q,
        RetrievalRelevance(metric="hit_rate", k=10),
        ResponseQuality(score_keys=["faithfulness", "relevance"]),
        j,
    ],
    sinks=[InMemorySink()],
)

# Ingest traces from your live pipeline
for trace in stream_of_traces():
    det.ingest(trace)

report = det.report()
print(report.degraded)   # ['RetrievalRelevance']
print(report.healthy)    # False
det.commit_window()      # roll trailing baselines forward at end of comparison interval
```

## What a Trace looks like

```python
Trace(
    timestamp=datetime.utcnow(),
    query="What's the baggage allowance on a Wanna Get Away fare?",
    query_embedding=[...],            # required by QueryDistribution / EmbeddingDrift
    retrieved_doc_ids=["d1", "d2"],
    retrieval_scores=[0.91, 0.83],
    relevance_labels=[1, 0, 0, 0, 0], # binary 0/1 per retrieved doc; required by RetrievalRelevance
    response="Up to 2 free checked bags...",
    judge_scores={"faithfulness": 0.92, "relevance": 0.88},  # required by ResponseQuality / JudgeDrift
    metadata={"reference_id": "ref-1"},                       # required by JudgeDrift
)
```

Each dimension only needs the fields it cares about. Missing fields produce `OK`-with-empty-sample reports rather than errors.

## Tuning detectors

Every dimension exposes the same alarming knobs:

```python
RetrievalRelevance(
    metric="hit_rate",   # "hit_rate" | "precision_at_k" | "mrr"
    k=10,                # top-k cutoff applied to relevance_labels
    baseline_size=200,   # how many committed windows the rolling baseline keeps
    warn_z=2.0,          # |z| at which a dimension warns
    degraded_z=3.0,      # |z| at which a dimension degrades
)
```

- `baseline_size` controls the length of the trailing baseline each
  `commit_window()` feeds. Larger = smoother, slower to react; smaller =
  twitchier. It applies to `QueryDistribution`, `RetrievalRelevance`,
  `EmbeddingDrift`, and `ResponseQuality`.
- `JudgeDrift` alarms on the absolute mean delta against the frozen
  reference set instead of a z-score, so it uses `warn_abs` / `degraded_abs`.
- `QueryDistribution` and `EmbeddingDrift` require a reference centroid via
  `set_reference(embeddings)` before they report anything meaningful.

## Sinks

```python
from ragvitals import InMemorySink, JSONLSink, CloudWatchSink, PhoenixSink

InMemorySink()                                  # tests, REPL
JSONLSink(path="/var/log/ragvitals.jsonl")       # cheap, append-only
CloudWatchSink(namespace="rag/prod")            # boto3-backed, requires `pip install ragvitals[aws]`
PhoenixSink(endpoint="http://localhost:4317",   # Arize Phoenix / Arize Cloud (OTLP)
            project_name="ragvitals")            # requires `pip install ragvitals[phoenix]`
```

## Arize Phoenix integration

`PhoenixSink` ships every `DetectorReport` to Arize Phoenix as an
OpenTelemetry span tree: one parent span per detection window
(`ragvitals.detector.report`) with one child span per dimension
(`ragvitals.dimension.<name>`). Each child carries the drift score,
severity, baseline, z-score, sample size, and detail string as span
attributes Phoenix indexes and renders. Drift events land on the same
timeline as your existing Phoenix-instrumented LLM and retrieval calls,
so on-call sees the correlation in one UI.

```python
from ragvitals import Detector, PhoenixSink, QueryDistribution

det = Detector(
    dimensions=[QueryDistribution()],
    sinks=[PhoenixSink(
        endpoint="http://localhost:4317",        # local Phoenix collector
        project_name="ragvitals-prod",
    )],
)
```

For **Arize Cloud (managed Phoenix)**, point `endpoint` at the OTLP URL
from your Arize space settings and add `headers={"api-key": "..."}`.

`PhoenixSink` is the bridge between ragvitals (the drift library) and
Phoenix (the platform). Install with `pip install ragvitals[phoenix]`.

## Replay against a frozen pipeline

```python
det.ingest_jsonl("s3-or-local-path-to/traces.jsonl")
report = det.report()
```

## Integrated drift demo

The five dimensions are also wired together against one synthetic
500-document corpus, so you can see them react to the same event:

- **CLI:** `python examples/integrated_drift_dashboard.py`
  Prints a side-by-side ASCII table at ages 0, 7, 14, 21, 30.
- **Streamlit:** `streamlit run examples/streamlit_drift_dashboard.py`
  One slider for `days_aged` (0 to 30), five live panels, one DriftReport.

Both demos share the same corpus, topic distribution, and drift simulation
in `examples/integrated_drift_dashboard.py`. The CLI version is what runs
in CI and pastes into screenshots.

Sample CLI output:

```
metric                       | age= 0           | age= 7           | age=14           | age=21           | age=30
---------------------------------------------------------------------------------------------------------------------
QueryDistribution            | 0.502 [ok]       | 0.619 [degraded] | 0.733 [degraded] | 0.754 [degraded] | 0.851 [degraded]
RetrievalRelevance           | 1.000 [ok]       | 0.920 [warn]     | 0.820 [warn]     | 0.800 [warn]     | 0.700 [degraded]
EmbeddingDrift               | 0.503 [ok]       | 0.619 [degraded] | 0.733 [degraded] | 0.754 [degraded] | 0.851 [degraded]
ResponseQuality.faithfulness | 0.909 [ok]       | 0.867 [degraded] | 0.836 [degraded] | 0.798 [degraded] | 0.755 [degraded]
JudgeDrift                   | -0.001 [ok]      | -0.037 [ok]      | -0.069 [warn]    | -0.112 [degraded]| -0.146 [degraded]
latency p50 (ms)             | 119              | 228              | 347              | 460              | 599
latency p95 (ms)             | 163              | 561              | 726              | 493              | 665
```

## What it explicitly is not

- Not a tracing tool. Bring your own JSONL / OpenTelemetry / Phoenix upstream.
- Not an annotation UI.
- Not a replacement for Ragas (which does *offline* eval on a golden set).
- Not Arize/Phoenix — those are platforms; this is a library that writes to a sink you choose.

## Sibling libraries

If your RAG runs on AWS Bedrock, two companion libraries:

- [**bedrockcache**](https://github.com/MukundaKatta/bedrockcache) — audit Anthropic prompt caching across the Bedrock + LiteLLM + Strands stack.
- [**bedrockstack**](https://github.com/MukundaKatta/bedrockstack) — Bedrock-aware retry policy, cost ledger, streaming-error normalization.
- **ragvitals** (this) — 5-dimensional production drift detection for the RAG pipeline above.

Landing repo with a runnable 50-line example wiring all three together: **[bedrock-production-stack](https://github.com/MukundaKatta/bedrock-production-stack)**.

## Roadmap

- v0.2: pluggable statistical tests (KS, MWU) instead of z-score-only.
- v0.3: `Detector.replay(snapshot=...)` against a saved baseline snapshot.
- v0.4: drift attribution (which docs / users / queries are most responsible).

## Develop

The test suite is pure standard-library `unittest`, so it needs no test
dependencies:

```bash
pip install -e .
python -m unittest discover -s tests -v
```

`pytest` (in the `dev` extra) also runs the same suite if you prefer it:

```bash
pip install -e ".[dev]"
pytest -v
```

## License

MIT
