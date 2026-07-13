import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

RESULTS_DIR = "results"
PLOTS_DIR = "plots"

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

def evaluate_predictions(y_true, y_pred, output_json="results/evaluation_metrics.json"):
    """
    Evaluates predictions for a 6-class problem and saves metrics to JSON.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
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
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(cr_text)
        
    # Generate and save confusion matrix plot
    plot_confusion_matrix(y_true, y_pred, os.path.join(PLOTS_DIR, "confusion_matrix.png"))
        
    return results
