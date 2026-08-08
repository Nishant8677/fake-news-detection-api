"""Streamlit Community Cloud entrypoint for the fake-news classifier.

Calls inference.app.predict() -- the same function the REST endpoint calls --
rather than reimplementing inference. The Gradio Space does likewise, so all
three surfaces share one code path and none of them can drift from the numbers
in BENCHMARK.md.

Streamlit reruns the script top to bottom on every interaction. That is safe
here because inference.app loads the model at import time and Python caches
modules, so the 418 MB of weights are read once per process rather than once
per click.

Entrypoint: deploy/streamlit/streamlit_app.py
Requires:   MODEL_DIR set to the Hub model repo, via .streamlit/secrets.toml
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st


def _secret(name: str) -> str | None:
    """Reads a secret, tolerating the absence of any secrets file.

    `name in st.secrets` does not return False when no secrets.toml exists
    anywhere -- it raises StreamlitSecretNotFoundError while trying to locate
    one. Guarding on membership therefore crashes the app on startup in exactly
    the case it was meant to handle: a deployment where secrets have not been
    set yet, and a local run where they never will be.
    """
    try:
        return st.secrets[name]
    except Exception:
        return None


# MODEL_DIR must be set before inference.app is imported, because the module
# loads the model at import time. On Community Cloud it arrives via secrets;
# locally it comes from the environment.
if not os.getenv("MODEL_DIR"):
    from_secrets = _secret("MODEL_DIR")
    if from_secrets:
        os.environ["MODEL_DIR"] = from_secrets

# Submitted text belongs to whoever typed it, and this filesystem is ephemeral.
os.environ.setdefault("LOG_PREDICTIONS", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

st.set_page_config(page_title="Fake News Detection API", page_icon="📰")

ACCURACY = 0.2573      # results/classification_report.txt, n=1267
CHANCE_FLOOR = 1 / 6   # 6-class LIAR

EXAMPLES = [
    "Our state has the lowest unemployment rate in the entire country.",
    "Crime has fallen every single year for the past decade.",
    "We spend more on foreign aid than on education.",
    "The new policy will create two million jobs by next year.",
]


@st.cache_resource(show_spinner="Loading the model…")
def _load():
    """Imports the serving module, which loads weights as a side effect.

    Wrapped in cache_resource so a failure surfaces once with its real error
    rather than on every rerun.
    """
    from inference.app import CONFIDENCE_THRESHOLD, MODEL_LOADED, NewsInput, predict

    return predict, NewsInput, MODEL_LOADED, CONFIDENCE_THRESHOLD


st.title("Fake News Detection API")
st.write(
    "A fine-tuned BERT classifier over the 6-class LIAR dataset. This page calls "
    "the same `predict()` function the REST service exposes."
)

st.error(
    f"""**This model is wrong more often than it is right.**

It scores **{ACCURACY:.1%}** on the 6-class LIAR benchmark against a
**{CHANCE_FLOOR:.1%}** chance floor. Do not use it to decide whether anything is
actually true.

It exists to demonstrate the serving engineering — latency, throughput,
containerisation, health checking — on a model whose ceiling is set by a hard
dataset: fine-grained fact-checking asks annotators to separate `barely-true`
from `half-true`, which they disagree on themselves. The number is published
rather than hidden because it is the honest headline."""
)

try:
    predict, NewsInput, model_loaded, threshold = _load()
except Exception as exc:  # noqa: BLE001 - surfaced to the user deliberately
    st.exception(exc)
    st.stop()

if not model_loaded:
    st.warning(
        "Model weights did not load. Set `MODEL_DIR` in the app's secrets to the "
        "Hub repo id, e.g. `Nishant8677/fake-news-liar-bert`."
    )
    st.stop()

if "claim" not in st.session_state:
    st.session_state.claim = ""

st.subheader("Try a claim")
cols = st.columns(len(EXAMPLES))
# No strict=True: it is Python 3.10+, and Community Cloud lets the deployer
# choose 3.9. The lists are the same length by construction anyway.
for col, example in zip(cols, EXAMPLES):  # noqa: B905
    if col.button(example[:22] + "…", help=example, use_container_width=True):
        st.session_state.claim = example

claim = st.text_area("Claim", key="claim", height=100, placeholder="Paste a political claim…")

if st.button("Classify", type="primary", disabled=not claim.strip()):
    result = predict(NewsInput(text=claim))

    left, right = st.columns(2)
    left.metric("Prediction", result["prediction"])
    right.metric("Confidence", f"{result['confidence']:.1%}")

    if result["needs_review"]:
        st.info(
            f"`needs_review` is set: confidence is below the {threshold:.0%} "
            "threshold, so the service flags this as low-trust rather than "
            "answering with false certainty."
        )

    st.caption(f"The model is {ACCURACY:.1%} accurate overall. Treat the label accordingly.")

    with st.expander("Raw response (identical to `POST /predict`)"):
        st.json(result)

st.divider()
st.caption(
    "Measured serving performance (CPU, torch 2.10.0+cpu): 8.5 ms mean · 12.8 ms P95 · "
    "396 req/s batch · 0.67 s cold load · 418 MB on disk. Method in BENCHMARK.md, "
    "raw figures in results/benchmark_metrics.json. "
    "[Source](https://github.com/Nishant8677/fake-news-detection-api)"
)
