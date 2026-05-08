"""Drift dimensions: each computes a single metric over a sliding window
and compares it to a baseline.

Five dimensions matching the canonical 5-axis RAG production-rot story:

  1. QueryDistribution  — input query embedding distribution shift (mean cosine
                          distance from a reference centroid).
  2. RetrievalRelevance — top-k retrieval quality, e.g. mean reciprocal rank
                          or hit rate, vs trailing baseline.
  3. EmbeddingDrift     — drift of the corpus or retrieved-doc embeddings vs
                          a snapshot.
  4. ResponseQuality    — LLM-as-judge score(s) (faithfulness, relevance, etc.)
                          vs trailing baseline.
  5. JudgeDrift         — judge consistency on a frozen reference set
                          (catches the case where the judge itself drifts).

All dimensions compute online with z-score-vs-baseline alarming. No external
dependencies.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from ragvitals.types import DimensionReport, Severity, Trace


# ---------- base ----------


@dataclass
class _RollingBaseline:
    """Fixed-size rolling buffer used as a baseline for z-score comparisons."""

    size: int
    values: deque[float] = field(default_factory=deque)

    def push(self, v: float) -> None:
        self.values.append(v)
        while len(self.values) > self.size:
            self.values.popleft()

    def stats(self) -> tuple[float, float, int]:
        if len(self.values) < 2:
            return (0.0, 0.0, len(self.values))
        return (statistics.fmean(self.values), statistics.pstdev(self.values), len(self.values))


def _z_score(value: float, mean: float, stdev: float) -> float:
    if stdev <= 1e-12:
        return 0.0
    return (value - mean) / stdev


# Relative-change threshold used when the baseline is too stable to z-score
# against (stdev≈0). Without this, a perfectly-stable baseline can never
# alarm, which is exactly backwards: a stable baseline that suddenly drops
# is the strongest signal there is.
_CONSTANT_BASELINE_STDEV = 1e-9
_CONSTANT_BASELINE_WARN_FRAC = 0.05      # 5% relative change warns
_CONSTANT_BASELINE_DEGRADED_FRAC = 0.20  # 20% relative change degrades


def _classify(z: float, warn_z: float = 2.0, degraded_z: float = 3.0,
              direction: str = "lower-is-degraded") -> Severity:
    """Map a z-score to a severity.

    direction='lower-is-degraded' means low values are bad (e.g. recall@k
    falling). direction='higher-is-degraded' means high values are bad (e.g.
    drift distance climbing).
    """
    if direction == "lower-is-degraded":
        if z <= -degraded_z:
            return Severity.DEGRADED
        if z <= -warn_z:
            return Severity.WARN
        return Severity.OK
    else:
        if z >= degraded_z:
            return Severity.DEGRADED
        if z >= warn_z:
            return Severity.WARN
        return Severity.OK


def _classify_with_constant_baseline(
    value: float, mean: float, stdev: float, n_baseline: int,
    warn_z: float, degraded_z: float, direction: str,
) -> tuple[Severity, float]:
    """Classify with a graceful fallback when the baseline is near-constant.

    Returns (severity, z_score_used). When stdev > epsilon, this is just a
    z-score. When stdev is effectively zero AND we have at least 2 baseline
    samples to call it "stable" (not just "no data"), we fall back to relative
    change against the baseline mean.
    """
    if stdev > _CONSTANT_BASELINE_STDEV:
        z = _z_score(value, mean, stdev)
        return _classify(z, warn_z, degraded_z, direction), z

    if n_baseline < 2 or mean == 0:
        return Severity.OK, 0.0

    rel = (value - mean) / abs(mean)
    if direction == "lower-is-degraded":
        if rel <= -_CONSTANT_BASELINE_DEGRADED_FRAC:
            return Severity.DEGRADED, rel
        if rel <= -_CONSTANT_BASELINE_WARN_FRAC:
            return Severity.WARN, rel
    else:
        if rel >= _CONSTANT_BASELINE_DEGRADED_FRAC:
            return Severity.DEGRADED, rel
        if rel >= _CONSTANT_BASELINE_WARN_FRAC:
            return Severity.WARN, rel
    return Severity.OK, rel


# ---------- 1. Query distribution ----------


@dataclass
class QueryDistribution:
    """Mean cosine distance of query embeddings from a reference centroid.

    Pre-load the reference centroid via `set_reference(embeddings)` before
    streaming live traces in. Each trace's `query_embedding` is compared
    against the centroid and the rolling-window mean of those distances is
    z-scored against an earlier baseline.
    """

    name: str = "QueryDistribution"
    baseline_size: int = 1000
    warn_z: float = 2.0
    degraded_z: float = 3.0
    _ref_centroid: list[float] | None = None
    _baseline: _RollingBaseline = field(default_factory=lambda: _RollingBaseline(size=1000))
    _window: list[float] = field(default_factory=list)

    def set_reference(self, embeddings: Iterable[Iterable[float]]) -> None:
        self._ref_centroid = _centroid(embeddings)

    def update(self, trace: Trace) -> None:
        if trace.query_embedding is None or self._ref_centroid is None:
            return
        d = _cosine_distance(trace.query_embedding, self._ref_centroid)
        self._window.append(d)

    def report(self) -> DimensionReport:
        n = len(self._window)
        if n == 0 or self._ref_centroid is None:
            return DimensionReport(self.name, Severity.OK, None, None, None, n,
                                   "no embeddings ingested or no reference set")
        value = statistics.fmean(self._window)
        mean, stdev, n_base = self._baseline.stats()
        sev, z = _classify_with_constant_baseline(
            value, mean, stdev, n_base, self.warn_z, self.degraded_z,
            direction="higher-is-degraded",
        )
        return DimensionReport(self.name, sev, value, mean, z, n,
                               "mean cosine distance from reference centroid")

    def commit_window(self) -> None:
        """Move the current window's metric into the rolling baseline.

        Call this at the end of each comparison interval (e.g. nightly) so the
        next window compares against the trailing distribution.
        """
        if self._window:
            self._baseline.push(statistics.fmean(self._window))
        self._window.clear()


# ---------- 2. Retrieval relevance ----------


@dataclass
class RetrievalRelevance:
    """Hit rate or precision@k from binary `relevance_labels` on each trace."""

    name: str = "RetrievalRelevance"
    metric: str = "hit_rate"  # or "precision_at_k", "mrr"
    k: int = 10
    baseline_size: int = 100
    warn_z: float = 2.0
    degraded_z: float = 3.0
    _baseline: _RollingBaseline = field(default_factory=lambda: _RollingBaseline(size=100))
    _window_scores: list[float] = field(default_factory=list)

    def update(self, trace: Trace) -> None:
        if not trace.relevance_labels:
            return
        labels = trace.relevance_labels[: self.k]
        if self.metric == "hit_rate":
            score = 1.0 if any(labels) else 0.0
        elif self.metric == "precision_at_k":
            score = sum(labels) / max(1, len(labels))
        elif self.metric == "mrr":
            score = next((1 / (i + 1) for i, lbl in enumerate(labels) if lbl), 0.0)
        else:
            raise ValueError(f"unknown metric: {self.metric}")
        self._window_scores.append(score)

    def report(self) -> DimensionReport:
        n = len(self._window_scores)
        if n == 0:
            return DimensionReport(self.name, Severity.OK, None, None, None, n,
                                   "no traces with relevance_labels")
        value = statistics.fmean(self._window_scores)
        mean, stdev, n_base = self._baseline.stats()
        sev, z = _classify_with_constant_baseline(
            value, mean, stdev, n_base, self.warn_z, self.degraded_z,
            direction="lower-is-degraded",
        )
        return DimensionReport(self.name, sev, value, mean, z, n,
                               f"{self.metric}@{self.k}")

    def commit_window(self) -> None:
        if self._window_scores:
            self._baseline.push(statistics.fmean(self._window_scores))
        self._window_scores.clear()


# ---------- 3. Embedding drift ----------


@dataclass
class EmbeddingDrift:
    """Centroid drift of retrieved-doc embeddings vs a reference snapshot."""

    name: str = "EmbeddingDrift"
    warn_z: float = 2.0
    degraded_z: float = 3.0
    _ref_centroid: list[float] | None = None
    _baseline: _RollingBaseline = field(default_factory=lambda: _RollingBaseline(size=1000))
    _window: list[float] = field(default_factory=list)

    def set_reference(self, embeddings: Iterable[Iterable[float]]) -> None:
        self._ref_centroid = _centroid(embeddings)

    def update(self, trace: Trace) -> None:
        # Use query_embedding as a proxy for retrieved-doc embedding
        # distribution because that's what most pipelines actually log. If a
        # caller wants per-doc drift they can pass each retrieved doc as its
        # own Trace.
        if trace.query_embedding is None or self._ref_centroid is None:
            return
        d = _cosine_distance(trace.query_embedding, self._ref_centroid)
        self._window.append(d)

    def report(self) -> DimensionReport:
        n = len(self._window)
        if n == 0:
            return DimensionReport(self.name, Severity.OK, None, None, None, n,
                                   "no embeddings ingested")
        value = statistics.fmean(self._window)
        mean, stdev, n_base = self._baseline.stats()
        sev, z = _classify_with_constant_baseline(
            value, mean, stdev, n_base, self.warn_z, self.degraded_z,
            direction="higher-is-degraded",
        )
        return DimensionReport(self.name, sev, value, mean, z, n,
                               "mean centroid drift")

    def commit_window(self) -> None:
        if self._window:
            self._baseline.push(statistics.fmean(self._window))
        self._window.clear()


# ---------- 4. Response quality ----------


@dataclass
class ResponseQuality:
    """LLM-as-judge score drift, averaged across one or more named judges.

    Each Trace carries `judge_scores={"faithfulness": 0.92, ...}`. The
    dimension tracks each named score independently and reports the worst.
    """

    name: str = "ResponseQuality"
    score_keys: list[str] = field(default_factory=lambda: ["faithfulness", "relevance"])
    baseline_size: int = 100
    warn_z: float = 2.0
    degraded_z: float = 3.0
    _baselines: dict[str, _RollingBaseline] = field(default_factory=dict)
    _windows: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for k in self.score_keys:
            self._baselines.setdefault(k, _RollingBaseline(size=self.baseline_size))
            self._windows.setdefault(k, [])

    def update(self, trace: Trace) -> None:
        if not trace.judge_scores:
            return
        for k in self.score_keys:
            v = trace.judge_scores.get(k)
            if v is not None:
                self._windows[k].append(float(v))

    def report(self) -> DimensionReport:
        worst: DimensionReport | None = None
        total_n = 0
        for k in self.score_keys:
            scores = self._windows[k]
            total_n += len(scores)
            if not scores:
                continue
            value = statistics.fmean(scores)
            mean, stdev, n_base = self._baselines[k].stats()
            sev, z = _classify_with_constant_baseline(
                value, mean, stdev, n_base, self.warn_z, self.degraded_z,
                direction="lower-is-degraded",
            )
            r = DimensionReport(f"{self.name}.{k}", sev, value, mean, z, len(scores), k)
            if worst is None or _severity_rank(sev) > _severity_rank(worst.severity):
                worst = r
        if worst is None:
            return DimensionReport(self.name, Severity.OK, None, None, None, 0,
                                   "no judge_scores in any trace")
        return worst

    def commit_window(self) -> None:
        for k in self.score_keys:
            scores = self._windows[k]
            if scores:
                self._baselines[k].push(statistics.fmean(scores))
            self._windows[k] = []


# ---------- 5. Judge drift ----------


@dataclass
class JudgeDrift:
    """Score drift on a frozen reference set.

    Set a fixed map of `reference_id -> reference_score` (the score the judge
    gave on day 0 for each reference example). On each new trace whose
    `metadata['reference_id']` is in the map, accumulate the delta. A drifting
    judge produces a non-zero mean delta; a stable judge produces ~0.
    """

    name: str = "JudgeDrift"
    score_key: str = "faithfulness"
    warn_abs: float = 0.05
    degraded_abs: float = 0.15
    _reference: dict[str, float] = field(default_factory=dict)
    _deltas: list[float] = field(default_factory=list)

    def set_reference(self, reference: dict[str, float]) -> None:
        self._reference = dict(reference)

    def update(self, trace: Trace) -> None:
        if not trace.judge_scores or not trace.metadata:
            return
        ref_id = trace.metadata.get("reference_id")
        if ref_id is None or ref_id not in self._reference:
            return
        new = trace.judge_scores.get(self.score_key)
        if new is None:
            return
        self._deltas.append(float(new) - self._reference[ref_id])

    def report(self) -> DimensionReport:
        n = len(self._deltas)
        if n == 0:
            return DimensionReport(self.name, Severity.OK, None, None, None, n,
                                   "no traces matched the reference set")
        value = statistics.fmean(self._deltas)
        absv = abs(value)
        if absv >= self.degraded_abs:
            sev = Severity.DEGRADED
        elif absv >= self.warn_abs:
            sev = Severity.WARN
        else:
            sev = Severity.OK
        return DimensionReport(self.name, sev, value, 0.0, None, n,
                               f"mean delta on {self.score_key} vs reference")

    def commit_window(self) -> None:
        self._deltas.clear()


# ---------- helpers ----------


def _centroid(embeddings: Iterable[Iterable[float]]) -> list[float]:
    rows = [list(e) for e in embeddings]
    if not rows:
        raise ValueError("centroid of empty embedding set")
    dim = len(rows[0])
    sums = [0.0] * dim
    for r in rows:
        if len(r) != dim:
            raise ValueError("ragged embeddings")
        for i, v in enumerate(r):
            sums[i] += v
    return [s / len(rows) for s in sums]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


_SEV_RANK = {Severity.OK: 0, Severity.WARN: 1, Severity.DEGRADED: 2}


def _severity_rank(s: Severity) -> int:
    return _SEV_RANK[s]
