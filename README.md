# Fake News Detection (6-Class LIAR Model)

**[▶ Live demo](https://fake-news-detection-api-ktb2chsjvps7cqgsmknbcn.streamlit.app/)** ·
[Model weights](https://huggingface.co/Nishant8677/fake-news-liar-bert) ·
[Benchmarks](BENCHMARK.md)

> The demo sleeps when idle. A first visit pays container start plus a 0.67 s
> model load, so give it a moment — that is hosting, not the model.

An end-to-end, production-grade Natural Language Processing (NLP) repository that explores multi-class fake news detection using transformer-based deep learning models.

This repository is built with machine learning engineering best practices in mind, featuring strict data splitting, comprehensive experiment tracking, unit testing, and advanced inference benchmarking.

## 🔍 Project Overview

Fake news detection is a challenging NLP problem due to linguistic ambiguity, source bias, and evolving writing styles. Instead of treating fake news as a simple binary (Fake vs. Real), this project tackles the much harder **6-class classification problem** using the LIAR dataset.

The labels are:
- `pants-fire`
- `false`
- `barely-true`
- `half-true`
- `mostly-true`
- `true`

## 🧠 Architecture

The architecture represents a full ML lifecycle from raw data to a deployed inference API.

- **Model:** BERT (`bert-base-uncased`) with a 6-class sequence classification head.
- **Framework:** PyTorch & HuggingFace Transformers
- **Inference Layer:** FastAPI
- **Experiment Tracking:** Custom JSON/Plotting pipeline

See [docs/architecture.md](docs/architecture.md) for a detailed data flow breakdown.

## 📊 Dataset

We use the **LIAR Dataset (PolitiFact)**, which consists of short political statements (1–2 sentences). 

> **Note:** The dataset `.tsv` files are not included in this repository due to size and licensing constraints. To train or benchmark the model, place `train.tsv`, `valid.tsv`, and `test.tsv` into `data/raw/` and run `preprocess.py`.

## 🚀 Training

To ensure reproducibility, training configurations (learning rate, batch size, seed, etc.) are automatically saved alongside the model weights.

```bash
python train.py
```

See [docs/training.md](docs/training.md) for full instructions.

## 🧪 Evaluation & Benchmarks

We emphasize **Strict Test-Set Evaluation**. The benchmark script never evaluates on training data and captures extensive system metrics to prove deployment readiness.

```bash
python benchmark.py
```

**Benchmarked Metrics Include:**
- Macro & Weighted F1, Precision, Recall
- Confusion Matrix & Classification Reports
- Single & Batch Inference Latency (Average, Median, P95)
- Throughput (req/sec)
- CPU, Peak RAM, Model Size & Load Time

### Measured results

All figures below are committed in [`results/`](results/) and reproduced by
`python benchmark.py`. Full methodology and environment: [BENCHMARK.md](BENCHMARK.md).

**Model quality** — 6-class LIAR, n=1267 test claims:

| Metric | Value |
|---|---|
| Accuracy | **0.2573** |
| Macro F1 | 0.2601 |
| Weighted F1 | 0.2553 |

Random chance on a balanced 6-way task is 0.167. Per-class F1 ranges from 0.19
(`mostly-true`) to 0.31 (`pants-fire`). See [`results/classification_report.txt`](results/classification_report.txt).

**Serving performance:**

| Metric | Value |
|---|---|
| Average latency | 8.5 ms |
| P95 latency | 12.8 ms |
| Batch throughput | 396.5 req/s |
| Single-request throughput | 117.9 req/s |
| Cold-start model load | 0.67 s |
| Model size on disk | 418.4 MB |
| Peak memory | 1344.3 MB |

See [docs/benchmark.md](docs/benchmark.md) for details on our evaluation methodology.

## 🌐 API Deployment

The model is deployed via a FastAPI REST endpoint that accepts text and returns a predicted class along with a confidence score.

```bash
uvicorn inference.app:app --reload
```

Or with Docker (model weights are mounted, not baked into the image):

```bash
docker build -t fake-news-api .
docker run -v $(pwd)/model:/app/model -p 8000:8000 fake-news-api
```

Interactive docs are at `http://localhost:8000/docs`.

### Example session

Real output from a running server, not illustrative:

```console
$ curl -s http://localhost:8000/health
{"status":"ok","model_loaded":true,"dataset_version":"v1.0-liar"}

$ curl -s -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"text": "Our state has the lowest unemployment rate in the entire country."}'
{"request_id":"ac2b3199-c8af-4289-930d-985bd77a6cb7","prediction":"mostly-true",
 "confidence":0.5455,"dataset_version":"v1.0-liar","needs_review":false,
 "timestamp":"2026-08-06T11:28:43.342496"}
```

`needs_review` flags any prediction below a 0.4 confidence threshold. `/health`
returns `503` with `"status": "degraded"` when the weights are not mounted, which
is what the container healthcheck polls.

> The label is the model's, not an endorsement. At 25.7% accuracy it is wrong far
> more often than it is right — see the Limitations section below.

## ⚠️ Limitations & Realities of the Task

The headline number for this project is **25.7% accuracy**, and it is stated up front
deliberately.

Multi-class political fact-checking is substantially harder than binary fake-news
classification. Distinguishing `barely-true` from `half-true` is a judgement call that
annotators themselves disagree on, and the fine-grained 6-way LIAR task is known to sit
far below what people expect from a BERT fine-tune. A model at 25.7% is above the 16.7%
chance floor but is **not fit for any real fact-checking use**, and nothing here claims
otherwise.

What this project is actually about is the **serving layer**: containerised deployment,
sub-13 ms P95 latency, batch throughput, confidence-scored responses, structured logging,
and a benchmark harness whose output is committed rather than described. The model's
ceiling is set by the dataset; the engineering around it is the work.

## 🛠️ Tech Stack
- **Deep Learning:** PyTorch, HuggingFace Transformers
- **API:** FastAPI, Uvicorn
- **Data Engineering:** Pandas, Scikit-learn
- **Testing:** Pytest

## 👤 Author
**Nishant Choudhary**  
BTech Electronics & Computer Engineering, VIT Chennai  
- Email: nishantchoudhary8677@gmail.com
- LinkedIn: [Nishant Choudhary](https://www.linkedin.com/in/nishant-choudhary-7a0a97282)
- GitHub: [Nishant8677](https://github.com/Nishant8677)
