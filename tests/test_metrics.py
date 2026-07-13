import pytest
import os
import json
from evaluation import evaluate_predictions

def test_metrics_generation():
    y_true = [0, 1, 2, 3, 4, 5, 0, 1]
    y_pred = [0, 1, 2, 3, 4, 5, 1, 0]
    
    output_file = "tests/test_metrics.json"
    os.makedirs("tests", exist_ok=True)
    
    results = evaluate_predictions(y_true, y_pred, output_file)
    
    assert "metrics" in results
    assert "confusion_matrix" in results
    assert "classification_report" in results
    
    assert "Macro_F1" in results["metrics"]
    assert "Weighted_F1" in results["metrics"]
    
    if os.path.exists(output_file):
        os.remove(output_file)
