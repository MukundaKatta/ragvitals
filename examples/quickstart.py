"""Minimal end-to-end ragvitals example.

Run:  python examples/quickstart.py
"""

import random
from datetime import datetime, timedelta

from ragvitals import (
    Detector,
    InMemorySink,
    QueryDistribution,
    ResponseQuality,
    RetrievalRelevance,
    Trace,
)


def main() -> None:
    rng = random.Random(0)

    # Set up dimensions. QueryDistribution needs a reference centroid; we
    # synthesize one in 2D for the example.
    q = QueryDistribution()
    q.set_reference([[1.0, 0.0]] * 50)

    det = Detector(
        dimensions=[
            q,
            RetrievalRelevance(metric="hit_rate", k=5, warn_z=1.5, degraded_z=2.5),
            ResponseQuality(score_keys=["faithfulness"], warn_z=1.5, degraded_z=2.5),
        ],
        sinks=[InMemorySink()],
    )

    base_ts = datetime(2026, 5, 1)

    # Eight days of healthy baseline traffic
    for day in range(8):
        for _ in range(50):
            det.ingest(Trace(
                timestamp=base_ts + timedelta(days=day),
                query_embedding=[1.0 + rng.gauss(0, 0.01), rng.gauss(0, 0.01)],
                relevance_labels=[1, 0, 0, 0, 0] if rng.random() < 0.9 else [0]*5,
                judge_scores={"faithfulness": rng.gauss(0.92, 0.02)},
            ))
        det.commit_window()

    # Today: retrieval collapses, judge scores drop, queries shift orthogonally
    for _ in range(50):
        det.ingest(Trace(
            timestamp=base_ts + timedelta(days=8),
            query_embedding=[rng.gauss(0, 0.01), 1.0 + rng.gauss(0, 0.01)],
            relevance_labels=[0] * 5,
            judge_scores={"faithfulness": 0.5},
        ))

    report = det.report()
    for d in report.dimensions:
        print(f"  {d.name:35s}  {d.severity.value:9s}  value={d.value!r}  z={d.z_score!r}  n={d.sample_size}")
    print()
    print(f"degraded: {report.degraded}")
    print(f"warned:   {report.warned}")
    print(f"healthy:  {report.healthy}")


if __name__ == "__main__":
    main()
