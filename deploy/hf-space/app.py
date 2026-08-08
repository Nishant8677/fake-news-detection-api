"""Hugging Face Space entrypoint: a Gradio UI mounted on the existing API.

The UI calls inference.app.predict() -- the same function the /predict endpoint
calls -- rather than reimplementing inference. There is one code path, so the
demo cannot drift from the API, and the numbers in BENCHMARK.md keep describing
the thing that is actually running.

Serves at /        the Gradio interface
          /predict the JSON endpoint (unchanged)
          /health  the readiness probe (unchanged)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference.app import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    DATASET_VERSION,
    MODEL_LOADED,
    NewsInput,
    predict,
)
from inference.app import (
    app as api,
)

ACCURACY = 0.2573          # results/classification_report.txt, n=1267
CHANCE_FLOOR = 1 / 6       # 6-class LIAR

DISCLAIMER = f"""
## ⚠️ This model is wrong more often than it is right — read this first

It scores **{ACCURACY:.1%} accuracy** on the 6-class LIAR benchmark, against a
**{CHANCE_FLOOR:.1%}** chance floor. It is better than guessing and nowhere near
good enough to judge whether a real claim is true.

**Do not use this to decide whether anything is actually true or false.**

It exists to demonstrate the *serving* engineering — latency, throughput,
containerisation, health checking — on a model whose accuracy ceiling is set by
a genuinely hard dataset. Fine-grained political fact-checking asks annotators to
separate `barely-true` from `half-true`, a distinction they themselves disagree
on. The number is published rather than hidden because it is the honest headline.
"""

EXAMPLES = [
    "Our state has the lowest unemployment rate in the entire country.",
    "Crime has fallen every single year for the past decade.",
    "We spend more on foreign aid than on education.",
    "The new policy will create two million jobs by next year.",
]


def classify(text: str):
    """Runs a claim through the same path the REST endpoint uses."""
    if not text or not text.strip():
        return {}, "Enter a claim above."

    if not MODEL_LOADED:
        return {}, "Model weights are not loaded — the Space is degraded."

    result = predict(NewsInput(text=text))

    label = result["prediction"]
    confidence = result["confidence"]

    note = f"**{label}** at {confidence:.1%} confidence."
    if result["needs_review"]:
        note += (
            f"\n\n`needs_review` is set: confidence is below the "
            f"{CONFIDENCE_THRESHOLD:.0%} threshold, so the service flags this "
            f"as low-trust rather than answering with false certainty."
        )
    note += f"\n\nRemember the model is only {ACCURACY:.1%} accurate overall."

    return {label: confidence}, note


with gr.Blocks(title="Fake News Detection API — 6-class LIAR") as demo:
    gr.Markdown("# Fake News Detection API")
    gr.Markdown(
        "A fine-tuned BERT classifier over the 6-class LIAR dataset, served with "
        "FastAPI. This page and the `/predict` endpoint run the same function."
    )
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                label="Claim",
                placeholder="Paste a political claim…",
                lines=3,
            )
            go = gr.Button("Classify", variant="primary")
            gr.Examples(EXAMPLES, inputs=text)
        with gr.Column():
            out_label = gr.Label(label="Prediction")
            out_note = gr.Markdown()

    go.click(classify, inputs=text, outputs=[out_label, out_note])
    text.submit(classify, inputs=text, outputs=[out_label, out_note])

    gr.Markdown(
        f"""
---
**Measured serving performance** (CPU, torch 2.10.0+cpu — see `BENCHMARK.md`):
8.5 ms mean · 12.8 ms P95 · 396 req/s batch · 0.67 s cold load · 418 MB on disk.

Dataset version `{DATASET_VERSION}`. API: `POST /predict`, `GET /health`.
[Source](https://github.com/Nishant8677/fake-news-detection-api)
"""
    )

# Gradio at /, the FastAPI routes preserved underneath.
app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
