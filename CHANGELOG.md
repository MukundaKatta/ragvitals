# Changelog

## Unreleased

- **Fix:** `QueryDistribution`, `RetrievalRelevance`, and `EmbeddingDrift`
  now honor a user-supplied `baseline_size`. Previously the rolling baseline
  was constructed with a hard-coded size in the dataclass `default_factory`,
  so passing `baseline_size=...` was silently ignored (the buffer always
  stayed at its default of 1000 / 100). `EmbeddingDrift` also gains a
  `baseline_size` field for parity with the other dimensions.
- Test suite converted to pure standard-library `unittest` so it runs with
  `python -m unittest discover -s tests` without any third-party test
  dependencies. pytest still runs the same suite. Added regression coverage
  for the `baseline_size` fix.
- CI now installs only the package (no test extras) and runs
  `python -m py_compile` plus the `unittest` suite across Python 3.10–3.13.
- README: documented the per-dimension tuning knobs (`baseline_size`,
  `warn_z`/`degraded_z`, `warn_abs`/`degraded_abs`).

## 0.2.0 — 2026-05-24

- Integrated 5-dimension drift demo + Streamlit dashboard.
- `examples/integrated_drift_dashboard.py`: CLI demo that wires all
  five drift dimensions (QueryDistribution, RetrievalRelevance,
  EmbeddingDrift, ResponseQuality, JudgeDrift) against a deterministic
  500-document synthetic corpus. One `days_aged` knob shifts the query
  distribution, stales retrievals, slows latencies, and drifts judge
  scores. Prints a side-by-side ASCII table at ages 0, 7, 14, 21, 30.
- `examples/streamlit_drift_dashboard.py`: thin Streamlit shell on top
  of the same simulation. One slider, five live panels, one
  DriftReport summary.
- 5 new tests covering the integrated demo entrypoints (22 total).

## 0.1.0 — initial release

- `Detector` orchestrating five composable drift dimensions:
  - `QueryDistribution` — input query embedding distribution shift
  - `RetrievalRelevance` — hit-rate / precision@k / MRR vs trailing baseline
  - `EmbeddingDrift` — centroid drift vs reference snapshot
  - `ResponseQuality` — LLM-as-judge score drift across one or more named judges
  - `JudgeDrift` — judge consistency on a frozen reference set
- Sinks: `InMemorySink`, `JSONLSink`, `CloudWatchSink` (boto3 optional, raises clean ImportError without it).
- Trace ingestion via `Detector.ingest()` or `Detector.ingest_jsonl(path)`.
- z-score-vs-baseline alarming with configurable warn/degraded thresholds.
- Zero required runtime dependencies. boto3 only required if you use `CloudWatchSink`.
- 18 tests across dimensions and detector.
