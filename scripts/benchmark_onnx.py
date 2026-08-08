"""Compares torch, ONNX fp32 and ONNX int8 on latency, throughput and accuracy.

The point of this script is the third column. Quantisation trades accuracy for
speed, and a benchmark that reports only the speed is reporting half of a
trade. Every arm is scored on the same 1,267-row LIAR test split that produced
the 0.2573 figure in results/classification_report.txt, so the cost of the
speedup is measured rather than assumed.

Latency is reported as p50/p95/p99 rather than a mean. A mean hides the tail,
and the tail is what a caller actually waits for.

    python scripts/benchmark_onnx.py
    python scripts/benchmark_onnx.py --latency-runs 300 --batch-size 32
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
from scipy.stats import binomtest
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))

MODEL_DIR = "model"
ONNX_DIR = Path("model_onnx")
TEST_CSV = Path("data/processed/test_processed.csv")
OUTPUT = Path("results/onnx_benchmark.json")

MAX_LENGTH = 64  # matches inference/app.py and training
LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]

# The committed figure every claim about this model refers to.
BASELINE_ACCURACY = 0.2573


def percentiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(float(np.percentile(ordered, 95)), 3),
        "p99_ms": round(float(np.percentile(ordered, 99)), 3),
        "mean_ms": round(statistics.fmean(ordered), 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
        "n": len(ordered),
    }


class TorchRunner:
    name = "torch_fp32"

    def __init__(self, model_dir: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    def logits(self, encoded: dict) -> np.ndarray:
        with torch.no_grad():
            out = self.model(**{k: torch.as_tensor(v) for k, v in encoded.items()})
        return out.logits.numpy()


class OnnxRunner:
    def __init__(self, path: Path, name: str, threads: int):
        import onnxruntime as ort

        self.name = name
        options = ort.SessionOptions()
        # Matched to torch.get_num_threads() by default. An earlier run pinned
        # this to 1 while torch used all 8, which is an 8:1 handicap: it made
        # int8's win understated and made fp32 look slower than torch when the
        # only difference was the thread count.
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.inputs = {i.name for i in self.session.get_inputs()}

    def logits(self, encoded: dict) -> np.ndarray:
        feeds = {k: np.asarray(v, dtype=np.int64) for k, v in encoded.items() if k in self.inputs}
        return self.session.run(None, feeds)[0]


def encode(tokenizer, texts: list[str]) -> dict:
    out = tokenizer(
        texts,
        return_tensors="np",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    return dict(out)


def measure_latency(runner, tokenizer, text: str, runs: int, warmup: int) -> dict:
    encoded = encode(tokenizer, [text])

    for _ in range(warmup):
        runner.logits(encoded)

    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        runner.logits(encoded)
        samples.append((time.perf_counter() - start) * 1000.0)
    return percentiles(samples)


def measure_throughput(runner, tokenizer, texts: list[str], batch_size: int) -> dict:
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    encoded_batches = [encode(tokenizer, b) for b in batches]

    runner.logits(encoded_batches[0])  # warm-up

    start = time.perf_counter()
    for batch in encoded_batches:
        runner.logits(batch)
    elapsed = time.perf_counter() - start

    return {
        "batch_size": batch_size,
        "rows": len(texts),
        "seconds": round(elapsed, 3),
        "rows_per_second": round(len(texts) / elapsed, 1),
    }


def measure_accuracy(runner, tokenizer, texts: list[str], labels: np.ndarray, batch_size: int) -> dict:
    predictions = []
    for i in range(0, len(texts), batch_size):
        encoded = encode(tokenizer, texts[i : i + batch_size])
        predictions.extend(runner.logits(encoded).argmax(-1).tolist())

    predictions = np.asarray(predictions)
    correct = predictions == labels

    per_class = {}
    for idx, label_name in enumerate(LABELS):
        mask = labels == idx
        per_class[label_name] = {
            "support": int(mask.sum()),
            "recall": round(float(correct[mask].mean()), 4) if mask.any() else None,
        }

    return {
        "accuracy": round(float(correct.mean()), 4),
        "n": len(labels),
        "per_class_recall": per_class,
        "predictions": predictions.tolist(),
    }


def peak_rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2


def environment(onnx_threads: int) -> dict[str, Any]:
    import onnxruntime as ort

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "cpu_count": psutil.cpu_count(),
        "onnx_threads": onnx_threads,
        "torch_threads": torch.get_num_threads(),
        "note": (
            "onnx_threads is matched to torch_threads by default so the arms "
            "differ only in runtime and precision. Running onnxruntime at 1 "
            "thread against torch at 8 understates ONNX by roughly the thread "
            "ratio and is not a like-for-like comparison."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency-runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--onnx-threads", type=int, default=torch.get_num_threads(),
                        help="default: match torch.get_num_threads() for a like-for-like run")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if not TEST_CSV.exists():
        print(f"error: {TEST_CSV} not found (data/ is gitignored)", file=sys.stderr)
        return 1
    fp32_path, int8_path = ONNX_DIR / "model.onnx", ONNX_DIR / "model.int8.onnx"
    for path in (fp32_path, int8_path):
        if not path.exists():
            print(f"error: {path} not found. Run scripts/export_onnx.py first.", file=sys.stderr)
            return 1

    df = pd.read_csv(TEST_CSV)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].to_numpy()
    sample_text = texts[0]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    arms = [
        ("torch_fp32", lambda: TorchRunner(MODEL_DIR)),
        ("onnx_fp32", lambda: OnnxRunner(fp32_path, "onnx_fp32", args.onnx_threads)),
        ("onnx_int8", lambda: OnnxRunner(int8_path, "onnx_int8", args.onnx_threads)),
    ]

    results: dict[str, Any] = {}
    print(f"test set: {len(texts)} rows  |  latency runs: {args.latency_runs}  "
          f"|  batch size: {args.batch_size}\n")

    for name, build in arms:
        gc.collect()
        before_mb = peak_rss_mb()
        load_start = time.perf_counter()
        runner = build()
        load_seconds = time.perf_counter() - load_start
        after_mb = peak_rss_mb()

        latency = measure_latency(runner, tokenizer, sample_text, args.latency_runs, args.warmup)
        throughput = measure_throughput(runner, tokenizer, texts, args.batch_size)
        accuracy = measure_accuracy(runner, tokenizer, texts, labels, args.batch_size)

        results[name] = {
            "load_seconds": round(load_seconds, 3),
            "rss_delta_mb": round(after_mb - before_mb, 1),
            "latency_single": latency,
            "throughput_batched": throughput,
            "accuracy": accuracy,
        }

        print(f"{name:<12} p50 {latency['p50_ms']:7.2f} ms  p95 {latency['p95_ms']:7.2f} ms  "
              f"p99 {latency['p99_ms']:7.2f} ms  |  {throughput['rows_per_second']:7.1f} rows/s  "
              f"|  acc {accuracy['accuracy']:.4f}", flush=True)

        del runner
        gc.collect()

    # Agreement matters as much as accuracy: two models can score the same and
    # still disagree on which rows they get right.
    base = np.asarray(results["torch_fp32"]["accuracy"]["predictions"])
    base_correct = base == labels
    for name in ("onnx_fp32", "onnx_int8"):
        other = np.asarray(results[name]["accuracy"]["predictions"])
        other_correct = other == labels

        # McNemar on the paired predictions. An accuracy delta of a few rows on
        # n=1267 sits inside the sampling error of a proportion, so "accuracy
        # dropped 1.2 points" is not by itself evidence that anything changed.
        # The discordant pairs answer that; the agreement figure answers the
        # different question of whether it is the same model.
        b = int((base_correct & ~other_correct).sum())   # torch right, arm wrong
        c = int((~base_correct & other_correct).sum())   # arm right, torch wrong
        p_value = float(binomtest(b, b + c, 0.5).pvalue) if (b + c) else 1.0

        results[name]["agreement_with_torch"] = {
            "identical_predictions": int((base == other).sum()),
            "of": len(base),
            "percent": round(float((base == other).mean()) * 100, 2),
            "changed_but_both_wrong": int(((base != other) & ~base_correct & ~other_correct).sum()),
            "mcnemar": {
                "torch_only_correct": b,
                "arm_only_correct": c,
                "exact_p_value": round(p_value, 4),
                "accuracy_difference_significant": bool(p_value < 0.05),
            },
        }

    torch_acc = results["torch_fp32"]["accuracy"]["accuracy"]
    int8_acc = results["onnx_int8"]["accuracy"]["accuracy"]
    torch_p95 = results["torch_fp32"]["latency_single"]["p95_ms"]
    int8_p95 = results["onnx_int8"]["latency_single"]["p95_ms"]

    payload = {
        "environment": environment(args.onnx_threads),
        "test_set": {"path": str(TEST_CSV), "rows": len(texts)},
        "committed_baseline_accuracy": BASELINE_ACCURACY,
        "arms": results,
        "summary": {
            "p95_speedup_int8_vs_torch": round(torch_p95 / int8_p95, 2) if int8_p95 else None,
            "accuracy_delta_int8_vs_torch": round(int8_acc - torch_acc, 4),
            "torch_reproduces_committed_baseline": abs(torch_acc - BASELINE_ACCURACY) < 0.005,
        },
    }

    # Predictions are bulky and only needed for the agreement figures above.
    for arm in payload["arms"].values():
        arm["accuracy"].pop("predictions", None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print("=" * 68)
    print(f"  p95 speedup, int8 vs torch : {payload['summary']['p95_speedup_int8_vs_torch']}x")
    print(f"  accuracy delta             : {payload['summary']['accuracy_delta_int8_vs_torch']:+.4f}"
          f"  ({torch_acc:.4f} -> {int8_acc:.4f})")
    print(f"  int8 agrees with torch on  : "
          f"{results['onnx_int8']['agreement_with_torch']['percent']}% of rows")
    print(f"  torch reproduces the committed {BASELINE_ACCURACY} baseline: "
          f"{payload['summary']['torch_reproduces_committed_baseline']}")
    print("=" * 68)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
