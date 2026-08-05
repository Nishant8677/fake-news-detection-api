import json
import os
import time

import numpy as np
import pandas as pd
import psutil
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Import from the repository
from evaluation import evaluate_predictions

TEST_DATA_PATH = "data/processed/test_processed.csv"
MODEL_DIR = "model"
RESULTS_DIR = "results"

def get_dir_size(path="."):
    total = 0
    if not os.path.exists(path):
        return 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total

def benchmark_resources():
    model_size_mb = get_dir_size(MODEL_DIR) / (1024 * 1024)
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = psutil.cpu_percent(interval=1.0)
    
    return {
        "Model_Size_MB": round(model_size_mb, 2),
        "Memory_Usage_MB": round(memory_mb, 2),
        "CPU_Usage_Percent": round(cpu_percent, 2)
    }

def benchmark_inference(df, model, tokenizer, device):
    latencies = []
    y_true = df['label'].tolist()
    y_pred = []
    
    texts = df['text'].tolist()
    
    print(f"Running inference benchmark on {len(texts)} test samples...")
    start_total = time.time()
    
    # Warm-up
    if len(texts) > 0:
        inputs = tokenizer(texts[0], return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
        with torch.no_grad():
            model(**inputs)
            
    warmup_end = time.time()
    warmup_time = warmup_end - start_total
    
    start_infer = time.time()
    for text in texts:
        t0 = time.time()
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            
        confidence, predicted_class = torch.max(probs, dim=1)
        
        t1 = time.time()
        latencies.append(t1 - t0)
        y_pred.append(predicted_class.item())
        
    end_infer = time.time()
    
    latencies = np.array(latencies)
    throughput = len(texts) / (end_infer - start_infer) if len(texts) > 0 else 0
    
    inference_metrics = {
        "Warmup_Time_s": round(warmup_time, 4),
        "Average_Latency_s": round(np.mean(latencies), 4) if len(latencies) > 0 else 0,
        "Median_Latency_s": round(np.median(latencies), 4) if len(latencies) > 0 else 0,
        "P95_Latency_s": round(np.percentile(latencies, 95), 4) if len(latencies) > 0 else 0,
        "Throughput_req_per_s": round(throughput, 2)
    }
    
    return inference_metrics, latencies, y_true, y_pred

def benchmark_batch_inference(df, model, tokenizer, device, batch_size=32):
    print(f"Running batch inference benchmark (batch_size={batch_size})...")
    texts = df['text'].tolist()
    
    start_total = time.time()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # The result is not consumed here, but /predict runs this softmax on
            # every request, so it stays inside the timed region. Dropping it
            # would inflate the throughput figure by timing less work than the
            # API actually does.
            _ = torch.softmax(outputs.logits, dim=1)
    end_total = time.time()
    
    throughput = len(texts) / (end_total - start_total) if len(texts) > 0 else 0
    return {"Batch_Throughput_req_per_s": round(throughput, 2)}

def main():
    print("Loading test dataset...")
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Test data not found at {TEST_DATA_PATH}. Please run preprocess.py first.")
        return
        
    df = pd.read_csv(TEST_DATA_PATH)
    
    print("Loading model and tokenizer...")
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    t1 = time.time()
    model_loading_time = round(t1 - t0, 4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 1. Resource Usage (Baseline)
    resource_metrics = benchmark_resources()
    
    # 2. Inference Benchmark
    inference_metrics, latencies, y_true, y_pred = benchmark_inference(df, model, tokenizer, device)
    inference_metrics["Model_Loading_Time_s"] = model_loading_time
    
    # 3. Evaluation Metrics
    print("Computing classification metrics...")
    eval_results = evaluate_predictions(y_true, y_pred, os.path.join(RESULTS_DIR, "evaluation_metrics.json"))
    
    # 4. Batch Inference Benchmark
    batch_metrics = benchmark_batch_inference(df, model, tokenizer, device)
    
    # Check max resources reached
    resource_metrics_end = benchmark_resources()
    resource_metrics["Peak_Memory_Usage_MB"] = max(resource_metrics["Memory_Usage_MB"], resource_metrics_end["Memory_Usage_MB"])
    
    # Compile Results
    final_report = {
        "Classification_Metrics": eval_results["metrics"],
        "Inference_Performance": inference_metrics,
        "Batch_Inference_Performance": batch_metrics,
        "System_Performance": resource_metrics
    }
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "benchmark_metrics.json"), "w") as f:
        json.dump(final_report, f, indent=4)
        
    print(f"Benchmarking complete. Results saved to {RESULTS_DIR}/benchmark_metrics.json")

if __name__ == "__main__":
    main()
