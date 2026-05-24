"""Integrated drift dashboard for ragvitals (CLI version).

Ties all five ragvitals detectors (QueryDistribution, RetrievalRelevance,
EmbeddingDrift, ResponseQuality, JudgeDrift) to ONE production-shaped
synthetic corpus, and prints a single side-by-side table that shows how
each metric reacts as the index "ages".

Usage
-----
    python examples/integrated_drift_dashboard.py
    python examples/integrated_drift_dashboard.py --ages 0 5 10 20 30
    python examples/integrated_drift_dashboard.py --seed 7

The simulation
--------------
- A deterministic 500-document corpus spread across four topics
  ("Q1 launch plan", "infra updates", "team standup notes",
  "policy & compliance"). Each doc has a low-dimensional embedding.
- 30 days of production query traces, ~50 queries/day, drawn from the
  topic distribution.
- One knob, ``days_aged``:
    1. Shifts the query distribution: a fifth topic ("Q2 reorg") slowly
       ramps in while "Q1 launch plan" decays.
    2. Slows response latencies (simulated infra degradation).
    3. Stales retrievals: older docs increasingly miss the new topic, so
       hit-rate decays.
    4. Drifts judge scores downward on the frozen reference set.

The script runs all five detectors at each requested age and prints a
single ASCII table comparing them. ASCII only, no external dependencies
beyond ragvitals itself.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from ragvitals import (
    Detector,
    EmbeddingDrift,
    InMemorySink,
    JudgeDrift,
    QueryDistribution,
    ResponseQuality,
    RetrievalRelevance,
    Trace,
)


# ---------- corpus & topic model ------------------------------------------------

# Four base topics plus one emergent topic that ramps in as the index ages.
TOPICS = [
    "Q1 launch plan",
    "infra updates",
    "team standup notes",
    "policy & compliance",
]
EMERGENT_TOPIC = "Q2 reorg"

# Each topic gets a fixed 4-D "embedding centroid". Numbers are arbitrary
# but chosen so the cosine distance between topics is non-trivial. Keeping
# the dimensionality small means the script stays fast and the math is
# easy to follow.
TOPIC_CENTROIDS: dict[str, list[float]] = {
    "Q1 launch plan":         [1.0, 0.0, 0.0, 0.0],
    "infra updates":          [0.0, 1.0, 0.0, 0.0],
    "team standup notes":     [0.0, 0.0, 1.0, 0.0],
    "policy & compliance":    [0.0, 0.0, 0.0, 1.0],
    # Emergent topic deliberately sits *outside* the base centroid so
    # that as it absorbs more traffic, QueryDistribution + EmbeddingDrift
    # see a real cosine-distance climb (rather than collapsing toward the
    # base-topic mean). Negative components push it past 90° from each
    # base axis.
    EMERGENT_TOPIC:           [-0.7, -0.7, 0.0, 0.0],
}

# Per-day query budget. Picked so each window has enough samples that the
# z-score is meaningful but small enough that the demo runs in <1 second.
QUERIES_PER_DAY = 50

# Baseline judge scores on the three reference items the JudgeDrift
# dimension is set against.
REFERENCE_IDS = ["ref-q1", "ref-infra", "ref-policy"]
REFERENCE_JUDGE_SCORES = {
    "ref-q1":     0.92,
    "ref-infra":  0.88,
    "ref-policy": 0.95,
}


@dataclass
class CorpusDoc:
    doc_id: str
    topic: str
    age_days: int          # how many days old this doc is at simulation start
    embedding: list[float]


def build_corpus(seed: int, n_docs: int = 500) -> list[CorpusDoc]:
    """Deterministically synthesize an N-document corpus over the four
    base topics. Document age is uniform over [0, 60] days so some docs
    are already "stale" before we even start aging the index."""
    rng = random.Random(seed)
    docs: list[CorpusDoc] = []
    for i in range(n_docs):
        topic = TOPICS[i % len(TOPICS)]
        centroid = TOPIC_CENTROIDS[topic]
        # Jitter around the topic centroid.
        emb = [v + rng.gauss(0.0, 0.04) for v in centroid]
        age = rng.randint(0, 60)
        docs.append(CorpusDoc(doc_id=f"doc-{i:04d}", topic=topic, age_days=age, embedding=emb))
    return docs


# ---------- query / trace generator ---------------------------------------------


def topic_mix_for_age(days_aged: int) -> dict[str, float]:
    """Return the {topic: weight} distribution to draw queries from.

    The mix is parameterized by `days_aged`. At age=0 the distribution
    matches the base corpus. As age grows, the "Q2 reorg" topic ramps in
    while "Q1 launch plan" decays. By age=30 about a third of traffic is
    the new topic.
    """
    frac_new = min(0.35, days_aged / 90.0)  # caps at 35% emergent topic
    decay_q1 = min(0.20, days_aged / 150.0)
    base = {t: 0.25 for t in TOPICS}
    base["Q1 launch plan"] = max(0.05, 0.25 - decay_q1)
    base[EMERGENT_TOPIC] = frac_new
    # Re-normalize.
    total = sum(base.values())
    return {k: v / total for k, v in base.items()}


def sample_topic(rng: random.Random, mix: dict[str, float]) -> str:
    r = rng.random()
    acc = 0.0
    for topic, w in mix.items():
        acc += w
        if r <= acc:
            return topic
    return next(iter(mix))


def synthesize_query_embedding(
    rng: random.Random,
    topic: str,
    days_aged: int,
) -> list[float]:
    """Pull a query embedding near the topic centroid, with extra jitter
    as the index ages (queries become noisier as the world drifts)."""
    centroid = TOPIC_CENTROIDS[topic]
    jitter = 0.04 + 0.005 * days_aged
    return [v + rng.gauss(0.0, jitter) for v in centroid]


def retrieve(
    query_embedding: list[float],
    corpus: list[CorpusDoc],
    days_aged: int,
    k: int = 10,
) -> tuple[list[CorpusDoc], list[float]]:
    """Top-k cosine retrieval against the synthetic corpus. As the index
    ages, retrieval quality degrades for two reasons we explicitly model:
    (1) older docs get a small score penalty proportional to their age,
    (2) the emergent topic has fewer real docs near its centroid, so
    retrievals for it look "stale" by construction."""
    scored: list[tuple[float, CorpusDoc]] = []
    for doc in corpus:
        sim = _cosine_similarity(query_embedding, doc.embedding)
        # Older docs decay slightly: simulates index staleness.
        sim -= 0.001 * doc.age_days * (days_aged / 30.0 + 0.1)
        scored.append((sim, doc))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:k]
    return [d for _, d in top], [s for s, _ in top]


def relevance_labels_for(
    topic: str,
    retrieved: list[CorpusDoc],
    days_aged: int,
) -> list[int]:
    """Binary relevance labels: a retrieved doc is relevant iff its topic
    matches the query topic. The emergent topic has no canonical docs in
    the corpus, so retrievals for it are mostly irrelevant, which is the
    point: the system is silently degrading on new traffic."""
    if topic == EMERGENT_TOPIC:
        # Some neighborhood docs (Q1 + infra) count as "loosely relevant"
        # half the time, scaled down as the topic itself ages.
        loose = max(0.1, 0.4 - 0.01 * days_aged)
        rng = random.Random(hash(tuple(d.doc_id for d in retrieved)) % (2**32))
        return [1 if (d.topic in ("Q1 launch plan", "infra updates") and rng.random() < loose) else 0
                for d in retrieved]
    return [1 if d.topic == topic else 0 for d in retrieved]


def synthesize_latency_ms(rng: random.Random, days_aged: int) -> float:
    """Latency increases with age: simulated infra degradation.

    Mean climbs from ~120ms at age=0 to ~600ms at age=30. p99 tail
    spikes also grow. Not consumed by ragvitals dimensions (there is no
    LatencyDrift detector in v0.2), but reported in the demo table so
    judges can see the infra story too.
    """
    base = 120.0 + 16.0 * days_aged
    spike = 0.0
    if rng.random() < (0.02 + 0.01 * (days_aged / 5)):
        spike = rng.uniform(200.0, 800.0 + 30.0 * days_aged)
    return max(20.0, rng.gauss(base, 25.0)) + spike


def synthesize_judge_scores(
    rng: random.Random,
    ref_id: str,
    days_aged: int,
) -> dict[str, float]:
    """Judge scores drift downward on the reference set as the index ages.

    This models the JudgeDrift narrative. The *judge itself* is not
    moving, but the system's response quality on the frozen reference
    set is. The dimension catches systematic shift on that set.
    """
    base = REFERENCE_JUDGE_SCORES[ref_id]
    drift = 0.005 * days_aged
    return {
        "faithfulness": max(0.0, min(1.0, base - drift + rng.gauss(0.0, 0.02))),
        "relevance":    max(0.0, min(1.0, base - drift + rng.gauss(0.0, 0.02))),
    }


def synthesize_day(
    *,
    day_index: int,
    days_aged: int,
    corpus: list[CorpusDoc],
    rng: random.Random,
) -> list[tuple[Trace, float]]:
    """Generate one day's worth of (Trace, latency_ms) pairs."""
    mix = topic_mix_for_age(days_aged)
    out: list[tuple[Trace, float]] = []
    ts0 = datetime(2026, 5, 1) + timedelta(days=day_index)
    for q in range(QUERIES_PER_DAY):
        topic = sample_topic(rng, mix)
        qemb = synthesize_query_embedding(rng, topic, days_aged)
        retrieved, scores = retrieve(qemb, corpus, days_aged=days_aged, k=10)
        labels = relevance_labels_for(topic, retrieved, days_aged=days_aged)
        latency = synthesize_latency_ms(rng, days_aged)

        # Roughly 1 in 6 traces lands on a reference item, so JudgeDrift
        # has signal but doesn't dominate the sample.
        ref_id = REFERENCE_IDS[q % len(REFERENCE_IDS)] if q % 6 == 0 else None
        if ref_id is not None:
            judge = synthesize_judge_scores(rng, ref_id, days_aged)
            meta = {"reference_id": ref_id, "topic": topic, "latency_ms": latency}
        else:
            judge = {
                "faithfulness": max(0.0, min(1.0, 0.9 - 0.005 * days_aged + rng.gauss(0.0, 0.03))),
                "relevance":    max(0.0, min(1.0, 0.9 - 0.005 * days_aged + rng.gauss(0.0, 0.03))),
            }
            meta = {"topic": topic, "latency_ms": latency}

        trace = Trace(
            timestamp=ts0 + timedelta(minutes=q),
            query=f"q about {topic}",
            query_embedding=qemb,
            retrieved_doc_ids=[d.doc_id for d in retrieved],
            retrieval_scores=scores,
            relevance_labels=labels,
            response=f"synthetic response for {topic}",
            judge_scores=judge,
            metadata=meta,
        )
        out.append((trace, latency))
    return out


# ---------- detector wiring -----------------------------------------------------


def make_detector(corpus: list[CorpusDoc]) -> Detector:
    """Build a Detector wired with the five drift dimensions, with
    references pre-loaded from the synthetic corpus + reference set."""
    base_topic_embeddings = [TOPIC_CENTROIDS[t] for t in TOPICS]

    q = QueryDistribution()
    q.set_reference(base_topic_embeddings)

    e = EmbeddingDrift()
    e.set_reference([d.embedding for d in corpus])

    j = JudgeDrift(score_key="faithfulness", warn_abs=0.05, degraded_abs=0.10)
    j.set_reference(REFERENCE_JUDGE_SCORES)

    return Detector(
        dimensions=[
            q,
            RetrievalRelevance(metric="hit_rate", k=10, warn_z=1.5, degraded_z=2.5),
            e,
            ResponseQuality(score_keys=["faithfulness", "relevance"],
                            warn_z=1.5, degraded_z=2.5),
            j,
        ],
        sinks=[InMemorySink()],
    )


def warm_baselines(det: Detector, corpus: list[CorpusDoc], seed: int) -> None:
    """Run 7 healthy-baseline windows through the detector so each
    dimension has a non-empty rolling baseline before the aged comparison
    windows arrive. Without this, ``stdev≈0`` and z-scores stay at zero.

    The warmup uses ``days_aged=0`` consistently so the baseline truly
    represents "healthy" behavior."""
    rng = random.Random(seed + 1000)
    for day in range(7):
        for trace, _ in synthesize_day(
            day_index=day,
            days_aged=0,
            corpus=corpus,
            rng=rng,
        ):
            det.ingest(trace)
        det.commit_window()


# ---------- per-age run + reporting --------------------------------------------


@dataclass
class AgeRow:
    days_aged: int
    metrics: dict[str, str]
    latencies_p50_ms: float
    latencies_p95_ms: float
    degraded: list[str]
    warned: list[str]


def run_age(
    days_aged: int,
    corpus: list[CorpusDoc],
    seed: int,
) -> AgeRow:
    """Build a fresh detector, warm baselines, then run a single aged
    comparison window. Returns a row for the side-by-side table.

    We rebuild the detector per age so each column is independent: no
    cross-contamination between successive `days_aged` values. This is
    what makes the table fair to read.
    """
    det = make_detector(corpus)
    warm_baselines(det, corpus, seed=seed)

    rng = random.Random(seed + days_aged * 17)
    latencies: list[float] = []
    for trace, latency in synthesize_day(
        day_index=10,
        days_aged=days_aged,
        corpus=corpus,
        rng=rng,
    ):
        det.ingest(trace)
        latencies.append(latency)

    report = det.report()

    metrics: dict[str, str] = {}
    for d in report.dimensions:
        if d.value is None:
            metrics[d.name] = "n/a"
        else:
            metrics[d.name] = f"{d.value:.3f} [{d.severity.value}]"

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]
    return AgeRow(
        days_aged=days_aged,
        metrics=metrics,
        latencies_p50_ms=p50,
        latencies_p95_ms=p95,
        degraded=report.degraded,
        warned=report.warned,
    )


# ---------- table rendering -----------------------------------------------------


def render_table(rows: list[AgeRow]) -> str:
    """Render the per-age results as a side-by-side ASCII table.

    Columns are ages, rows are detector metrics + latency percentiles.
    A separate "summary" block prints the degraded/warned dimension
    names per age, since those strings are too long to fit a fixed-width
    column without wrapping. Plain ASCII, no Unicode box-drawing, so it
    runs everywhere (CI logs, Cloud Run logs, screenshots).
    """
    # Collect dimension names in stable order from the first row.
    dim_names = list(rows[0].metrics.keys())
    metric_rows: list[tuple[str, list[str]]] = []
    for dn in dim_names:
        metric_rows.append((dn, [r.metrics.get(dn, "n/a") for r in rows]))
    metric_rows.append((
        "latency p50 (ms)",
        [f"{r.latencies_p50_ms:.0f}" for r in rows],
    ))
    metric_rows.append((
        "latency p95 (ms)",
        [f"{r.latencies_p95_ms:.0f}" for r in rows],
    ))

    label_w = max(len(name) for name, _ in metric_rows)
    col_w = max(
        max(len(v) for v in vals) for _, vals in metric_rows
    )
    col_w = max(col_w, len("age=00"))
    header = "metric".ljust(label_w) + " | " + " | ".join(
        f"age={r.days_aged:>2d}".ljust(col_w) for r in rows
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for name, vals in metric_rows:
        line = name.ljust(label_w) + " | " + " | ".join(v.ljust(col_w) for v in vals)
        lines.append(line)

    # Summary block keeps long verdict strings out of the fixed-width grid.
    lines.append("")
    lines.append("Verdict by age:")
    for r in rows:
        deg = ", ".join(r.degraded) or "-"
        warn = ", ".join(r.warned) or "-"
        lines.append(f"  age={r.days_aged:>2d}  degraded: {deg}")
        lines.append(f"         warned:   {warn}")
    return "\n".join(lines)


# ---------- helpers -------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------- entrypoint ----------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--ages",
        type=int,
        nargs="+",
        default=[0, 7, 14, 21, 30],
        help="Ages (in simulated days) to score. Default: 0 7 14 21 30.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-docs", type=int, default=500)
    args = parser.parse_args(argv)

    print(f"# ragvitals integrated drift demo (seed={args.seed}, corpus={args.n_docs} docs)")
    print(f"# topics: {', '.join(TOPICS)} (+ emergent: {EMERGENT_TOPIC})")
    print()

    corpus = build_corpus(seed=args.seed, n_docs=args.n_docs)

    rows: list[AgeRow] = []
    for age in args.ages:
        rows.append(run_age(age, corpus, seed=args.seed))

    print(render_table(rows))
    print()
    print("Read the table left to right. Each column is an independent run;")
    print("the only thing that changes across columns is `days_aged`. Values")
    print("are the dimension metric and the severity verdict in [brackets].")


if __name__ == "__main__":
    main()
