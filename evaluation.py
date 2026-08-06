import json
import os

import matplotlib

# These plots are only ever written to disk. Selecting the non-interactive
# backend before pyplot is imported keeps that true on a headless CI runner and
# stops matplotlib reaching for Tk, which made the suite fail intermittently.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

RESULTS_DIR = "results"
PLOTS_DIR = "plots"
DEFAULT_METRICS_JSON = os.path.join(RESULTS_DIR, "evaluation_metrics.json")

CLASS_NAMES = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]

def plot_confusion_matrix(y_true, y_pred, output_path="plots/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate_predictions(
    y_true,
    y_pred,
    output_json=DEFAULT_METRICS_JSON,
    results_dir=RESULTS_DIR,
    plots_dir=PLOTS_DIR,
):
    """
    Evaluates predictions for a 6-class problem and saves metrics to JSON.

    output_json, results_dir and plots_dir are all parameters because this
    function writes three files, not one. Callers that redirect only the JSON
    used to overwrite the committed classification report and confusion matrix
    with whatever data they passed in.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Macro_Precision": round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "Macro_Recall": round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "Macro_F1": round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "Weighted_Precision": round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        "Weighted_Recall": round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        "Weighted_F1": round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
    }
    
    cm = confusion_matrix(y_true, y_pred).tolist()
    cr = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    
    results = {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": cr
    }
    
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
        
    # Save the classification report as a text file for easy reading
    cr_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(cr_text)

    # Generate and save confusion matrix plot
    plot_confusion_matrix(y_true, y_pred, os.path.join(plots_dir, "confusion_matrix.png"))
        
    return results
