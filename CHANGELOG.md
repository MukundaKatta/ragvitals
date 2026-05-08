# Changelog

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
