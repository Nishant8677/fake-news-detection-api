# Benchmarking Guide

## Overview

The `benchmark.py` script is designed to provide an extensive view of both the model's predictive capability and its system-level performance. This prevents evaluating purely on accuracy and provides real-world engineering metrics.

## Strict Test-Set Evaluation

> **No Data Leakage:** This benchmark is hardcoded to use `data/processed/test_processed.csv`. It never sees the training data.

## Metrics Captured

### 1. Classification Metrics
- **Accuracy**: Overall correct predictions.
- **Macro Precision/Recall/F1**: Unweighted mean across all 6 classes. Crucial for detecting performance drops in minority classes (e.g., `pants-fire`).
- **Weighted Precision/Recall/F1**: Mean across all classes, weighted by their support.
- **Confusion Matrix**: Visualized and saved to `plots/confusion_matrix.png`.

### 2. Inference Performance
- **Single Prediction Latency (Average, Median, P95)**: How fast the model predicts a single statement (in seconds).
- **Batch Prediction Latency**: Throughput when passing multiple items at once (useful for offline processing).
- **Warmup Time**: Time taken for the first prediction, which includes memory allocation overhead.
- **Throughput**: Requests per second.

### 3. System Metrics
- **Peak RAM Usage**: Maximum memory consumed during the benchmark.
- **CPU Usage**: Average CPU utilization during inference.
- **Model Size**: Total disk space required by the `model/` directory (safetensors, config, tokenizer).
- **Model Loading Time**: Time taken to load the model from disk into memory.

## Output

All results are saved comprehensively into `results/benchmark_metrics.json`.
