"""Exports the fine-tuned classifier to ONNX and quantises it to int8.

Uses torch.onnx.export directly rather than optimum. optimum[onnxruntime]
resolves transformers back to 4.57.x, which would silently replace the pinned
5.2.0 that BENCHMARK.md was measured on -- so the "before" numbers would no
longer describe the environment they claim to. onnx and onnxruntime alone are
purely additive.

Produces, under model_onnx/:

    model.onnx          fp32 export, the like-for-like comparison against torch
    model.int8.onnx     dynamically quantised, weights int8

Dynamic quantisation is the right form here: it quantises weights ahead of time
and activations per-batch at run time, needs no calibration dataset, and is
well suited to transformer inference where matmuls dominate. Static
quantisation would need a calibration set and buys little for BERT on CPU.

    python scripts/export_onnx.py
    python scripts/export_onnx.py --opset 17 --output-dir model_onnx
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))

# torch.onnx prints a U+2705 on success. Piped on Windows, stdout defaults to
# cp1252, which cannot encode it -- so a successful export dies in its own
# progress message. Reconfiguring here covers this script's output and anything
# a library prints through it.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_MODEL_DIR = "model"
DEFAULT_OUTPUT_DIR = "model_onnx"

# Matches inference/app.py, which tokenises with max_length=64 to match training.
MAX_LENGTH = 64


def directory_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2


def export_fp32(model_dir: str, out_path: Path, opset: int) -> float:
    """Traces the model to ONNX. Returns seconds taken."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # A representative input. Dynamic axes make batch and sequence variable, so
    # one trace serves both the single-item and batched serving paths.
    sample = tokenizer(
        "Our state has the lowest unemployment rate in the entire country.",
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
            str(out_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
            # The dynamo exporter (torch 2.10's default) writes weights to an
            # external model.onnx.data sidecar. onnxruntime's dynamic quantiser
            # runs shape inference over the model path and fails on that layout
            # with "Inferred shape and existing shape differ". The legacy
            # TorchScript exporter emits one self-contained file, which is both
            # what the quantiser expects and simpler to ship.
            dynamo=False,
        )
    elapsed = time.perf_counter() - start

    # The tokenizer travels with the model: an ONNX graph without the vocabulary
    # that produced its inputs is not a servable artifact.
    tokenizer.save_pretrained(out_path.parent)
    return elapsed


def quantise(fp32_path: Path, int8_path: Path) -> float:
    """Dynamic int8 quantisation. Returns seconds taken."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    start = time.perf_counter()
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    return time.perf_counter() - start


def verify_parity(model_dir: str, fp32_path: Path, int8_path: Path) -> dict:
    """Checks the exports still agree with torch on a fixed set of claims.

    An export that runs but returns different logits is a broken export, and
    the failure is silent unless something compares them.
    """
    import numpy as np
    import onnxruntime as ort

    claims = [
        "Our state has the lowest unemployment rate in the entire country.",
        "Crime has fallen every single year for the past decade.",
        "We spend more on foreign aid than on education.",
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    encoded = tokenizer(
        claims, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH
    )
    with torch.no_grad():
        torch_logits = model(**encoded).logits.numpy()

    feeds = {k: v.numpy() for k, v in encoded.items()}

    def onnx_logits(path: Path):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        wanted = {i.name for i in session.get_inputs()}
        return session.run(None, {k: v for k, v in feeds.items() if k in wanted})[0]

    fp32_logits = onnx_logits(fp32_path)
    int8_logits = onnx_logits(int8_path)

    return {
        "torch_predictions": torch_logits.argmax(-1).tolist(),
        "onnx_fp32_predictions": fp32_logits.argmax(-1).tolist(),
        "onnx_int8_predictions": int8_logits.argmax(-1).tolist(),
        "fp32_max_abs_logit_diff": float(np.abs(torch_logits - fp32_logits).max()),
        "int8_max_abs_logit_diff": float(np.abs(torch_logits - int8_logits).max()),
        "fp32_predictions_match_torch": bool((torch_logits.argmax(-1) == fp32_logits.argmax(-1)).all()),
        "int8_predictions_match_torch": bool((torch_logits.argmax(-1) == int8_logits.argmax(-1)).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if not Path(args.model_dir).exists():
        print(f"error: {args.model_dir} not found. The weights are gitignored; "
              "download them from the Hub or train first.", file=sys.stderr)
        return 1

    fp32_path = args.output_dir / "model.onnx"
    int8_path = args.output_dir / "model.int8.onnx"

    print(f"exporting {args.model_dir} -> {fp32_path} (opset {args.opset})", flush=True)
    export_seconds = export_fp32(args.model_dir, fp32_path, args.opset)
    print(f"  done in {export_seconds:.1f}s", flush=True)

    print(f"quantising -> {int8_path}", flush=True)
    quantise_seconds = quantise(fp32_path, int8_path)
    print(f"  done in {quantise_seconds:.1f}s", flush=True)

    print("checking the exports still agree with torch", flush=True)
    parity = verify_parity(args.model_dir, fp32_path, int8_path)

    torch_mb = directory_size_mb(Path(args.model_dir))
    fp32_mb = fp32_path.stat().st_size / 1024**2
    int8_mb = int8_path.stat().st_size / 1024**2

    summary = {
        "opset": args.opset,
        "max_length": MAX_LENGTH,
        "export_seconds": round(export_seconds, 2),
        "quantise_seconds": round(quantise_seconds, 2),
        "size_mb": {
            "torch_directory": round(torch_mb, 2),
            "onnx_fp32": round(fp32_mb, 2),
            "onnx_int8": round(int8_mb, 2),
        },
        "size_reduction_vs_torch": {
            "onnx_fp32_percent": round((1 - fp32_mb / torch_mb) * 100, 2),
            "onnx_int8_percent": round((1 - int8_mb / torch_mb) * 100, 2),
        },
        "parity": parity,
    }

    (args.output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"  torch directory : {torch_mb:8.1f} MB")
    print(f"  onnx fp32       : {fp32_mb:8.1f} MB")
    print(f"  onnx int8       : {int8_mb:8.1f} MB  "
          f"({summary['size_reduction_vs_torch']['onnx_int8_percent']:.0f}% smaller)")
    print()
    print(f"  fp32 matches torch predictions: {parity['fp32_predictions_match_torch']} "
          f"(max logit diff {parity['fp32_max_abs_logit_diff']:.2e})")
    print(f"  int8 matches torch predictions: {parity['int8_predictions_match_torch']} "
          f"(max logit diff {parity['int8_max_abs_logit_diff']:.2e})")
    print()
    print("Prediction parity on three claims is a smoke test, not an accuracy")
    print("measurement. Run scripts/benchmark_onnx.py for the full test set.")

    if shutil.which("git"):
        print(f"\nwrote {args.output_dir / 'export_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
