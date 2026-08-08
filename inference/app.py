import os
import uuid
from datetime import datetime

import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------- CONFIG ----------------
# Folder with model.safetensors, config.json, tokenizer files. Read from the
# environment so the container can point at the mounted volume; "model" is the
# right default when running from the repo root.
# Accepts either a local directory containing model.safetensors/config.json/
# tokenizer files, or a Hugging Face Hub repo id -- from_pretrained resolves
# both, which is how the hosted demo loads weights without shipping 419 MB.
MODEL_DIR = os.getenv("MODEL_DIR", "model")
DATASET_VERSION = "v1.0-liar"
LOG_FILE = "logs/predictions.csv"
CONFIDENCE_THRESHOLD = 0.4           # Lowered for 6-class problem

# Prediction logging writes the submitted text to disk. That is useful locally
# and wrong on a public deployment, where the text belongs to whoever typed it
# and the filesystem is ephemeral anyway. Off unless explicitly enabled.
LOG_PREDICTIONS = os.getenv("LOG_PREDICTIONS", "1") not in ("0", "false", "False", "")
# --------------------------------------

LABEL_MAP_INV = {
    0: "pants-fire",
    1: "false",
    2: "barely-true",
    3: "half-true",
    4: "mostly-true",
    5: "true"
}

app = FastAPI(title="Fake News Detection API (6-Class)")

# ---------------- LOAD MODEL ----------------
# We only load if the directory has the config, otherwise we might fail on startup
# in a production environment this should be robust.
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    MODEL_LOADED = True
except Exception as e:
    print(f"Warning: Model could not be loaded from {MODEL_DIR}. Error: {e}")
    MODEL_LOADED = False
# --------------------------------------------

# ---------------- INPUT SCHEMA ----------------
class NewsInput(BaseModel):
    text: str
# ---------------------------------------------

# ---------------- LOGGING ----------------
def log_prediction(text, label, confidence, needs_review, request_id):
    if not LOG_PREDICTIONS:
        return

    os.makedirs("logs", exist_ok=True)

    row = {
        "request_id": request_id,
        "text": text.strip()[:200] if text else "N/A",
        "prediction": label,
        "confidence": confidence,
        "needs_review": needs_review,
        "dataset_version": DATASET_VERSION,
        "timestamp": datetime.now().isoformat()
    }

    df = pd.DataFrame([row])

    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)
# ---------------------------------------------

# ---------------- HEALTH ENDPOINT ----------------
@app.get("/health")
def health():
    """Liveness/readiness probe used by the container HEALTHCHECK.

    Returns 503 when the weights are absent: the process is up but cannot
    serve a prediction, which is the failure worth surfacing since the model
    is mounted at runtime rather than baked into the image.
    """
    if MODEL_LOADED:
        return {"status": "ok", "model_loaded": True, "dataset_version": DATASET_VERSION}

    return JSONResponse(
        status_code=503,
        content={"status": "degraded", "model_loaded": False, "dataset_version": DATASET_VERSION},
    )
# -------------------------------------------------

# ---------------- INFERENCE ENDPOINT ----------------
@app.post("/predict")
def predict(news: NewsInput):
    if not MODEL_LOADED:
        return {"error": "Model is not loaded. Please train the model first."}
        
    request_id = str(uuid.uuid4())

    # Tokenize input
    inputs = tokenizer(
        news.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64  # Match training max_length
    )

    # Model inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    confidence_tensor, predicted_class = torch.max(probs, dim=1)

    confidence = round(confidence_tensor.item(), 4)
    label = LABEL_MAP_INV.get(predicted_class.item(), "UNKNOWN")
    needs_review = confidence < CONFIDENCE_THRESHOLD

    # Log prediction
    log_prediction(
        text=news.text,
        label=label,
        confidence=confidence,
        needs_review=needs_review,
        request_id=request_id
    )

    # Final response
    return {
        "request_id": request_id,
        "prediction": label,
        "confidence": confidence,
        "dataset_version": DATASET_VERSION,
        "needs_review": needs_review,
        "timestamp": datetime.now().isoformat()
    }
# ---------------------------------------------------
