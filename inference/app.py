from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from datetime import datetime
import os
import uuid
import json

# ---------------- CONFIG ----------------
MODEL_DIR = "model"                  # Folder with model.safetensors, config.json, tokenizer files
DATASET_VERSION = "v1.0-liar"
LOG_FILE = "logs/predictions.csv"
CONFIDENCE_THRESHOLD = 0.4           # Lowered for 6-class problem
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
