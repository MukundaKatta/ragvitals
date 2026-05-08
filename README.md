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

## Sinks

```python
from ragvitals import InMemorySink, JSONLSink, CloudWatchSink

InMemorySink()                                  # tests, REPL
JSONLSink(path="/var/log/ragvitals.jsonl")       # cheap, append-only
CloudWatchSink(namespace="rag/prod")            # boto3-backed, requires `pip install ragvitals[aws]`
```

## Replay against a frozen pipeline

```python
det.ingest_jsonl("s3-or-local-path-to/traces.jsonl")
report = det.report()
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

## Roadmap

- v0.2: pluggable statistical tests (KS, MWU) instead of z-score-only.
- v0.3: `Detector.replay(snapshot=...)` against a saved baseline snapshot.
- v0.4: drift attribution (which docs / users / queries are most responsible).

## Develop

```bash
pip install -e ".[dev]"
pytest -v
```

## License

MIT
