# Fake News Detection (6-Class LIAR Model)

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

See [docs/benchmark.md](docs/benchmark.md) for details on our evaluation methodology.

## 🌐 API Deployment

The model is deployed via a FastAPI REST endpoint that accepts text and returns a predicted class along with a confidence score.

```bash
uvicorn inference.app:app --reload
```

## ⚠️ Limitations & Realities of the Task

Multi-class political fact-checking is significantly harder than binary fake news classification. Label ambiguity (e.g., distinguishing `barely-true` from `half-true`) means that accuracy metrics will be naturally lower than highly separable binary datasets. This project focuses on **engineering robustness** rather than chasing artificial 99% accuracy scores.

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
