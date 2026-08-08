---
title: Fake News Detection API
emoji: 📰
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Fake News Detection API — 6-class LIAR

A fine-tuned BERT claim classifier served with FastAPI, containerised, and
benchmarked. This Space runs the same image and the same `predict()` function as
the repository: the page you are looking at and the JSON endpoint are one code
path, so the demo cannot drift from the API.

## ⚠️ Accuracy, stated up front

**25.7% on the 6-class LIAR benchmark**, against a 16.7% chance floor (n=1267).

It is better than guessing and nowhere near good enough to judge a real claim.
Do not use it to decide whether anything is true. Fine-grained political
fact-checking asks annotators to separate `barely-true` from `half-true`, a
distinction they disagree on among themselves; the ceiling is set by the dataset.

The project is about the serving engineering. The accuracy is published rather
than omitted because that is the honest headline.

## Measured serving performance

CPU, torch 2.10.0+cpu. Full method in [`BENCHMARK.md`](https://github.com/Nishant8677/fake-news-detection-api/blob/main/BENCHMARK.md);
raw figures in `results/benchmark_metrics.json`.

| Metric | Value |
|---|---|
| Mean latency | 8.5 ms |
| P95 latency | 12.8 ms |
| Batch throughput | 396.5 req/s |
| Cold model load | 0.67 s |
| Model on disk | 418.4 MB |

## API

```bash
curl -X POST https://<space-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Our state has the lowest unemployment rate in the entire country."}'
```

```json
{
  "request_id": "…",
  "prediction": "mostly-true",
  "confidence": 0.5455,
  "dataset_version": "v1.0-liar",
  "needs_review": false,
  "timestamp": "…"
}
```

`needs_review` is set when confidence falls below 0.4, so the service flags
low-trust predictions instead of answering with false certainty.

`GET /health` returns `200 {"status":"ok"}` when weights are loaded and
`503 {"status":"degraded"}` when they are not — the container healthcheck polls it.

Prediction logging is disabled here (`LOG_PREDICTIONS=0`): submitted text belongs
to whoever typed it.

[Source](https://github.com/Nishant8677/fake-news-detection-api)
