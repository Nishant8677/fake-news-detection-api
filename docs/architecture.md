# Fake News Detection Architecture

## Data Flow

This project follows a strict pipeline to ensure reproducibility, avoid data leakage, and maintain a professional machine learning engineering standard.

1. **Dataset (`data/raw/`)**: The LIAR dataset is stored here in its raw `.tsv` format (`train.tsv`, `valid.tsv`, `test.tsv`).
2. **Cleaning & Preprocessing (`preprocess.py`)**: 
   - Reads the raw data.
   - Maps the original string labels to integer classes (0 to 5).
   - Strips whitespace and removes null values.
   - Saves clean `.csv` files into `data/processed/`.
3. **Tokenizer (`transformers.BertTokenizer`)**: 
   - Converts the text statements into token IDs and attention masks with a maximum sequence length of 64.
4. **BERT Model (`transformers.BertForSequenceClassification`)**: 
   - Utilizes `bert-base-uncased` with a 6-class sequence classification head.
5. **Training (`train.py`)**: 
   - Fine-tunes the model on the training set.
   - Evaluates on the validation set after every epoch.
   - Saves model weights, tokenizer, and hyperparameter configuration (`training_config.json`) to the `model/` directory.
   - Saves training metrics to `results/training_history.json`.
6. **Evaluation & Plotting (`evaluation.py`)**: 
   - Provides functions to compute macro/weighted multi-class metrics (Accuracy, Precision, Recall, F1).
   - Automatically saves plots like confusion matrices to `plots/`.
7. **FastAPI Inference (`inference/app.py`)**: 
   - Exposes a robust REST API `/predict` endpoint.
   - Automatically logs incoming requests and confidence scores to `logs/predictions.csv`.
8. **Benchmark (`benchmark.py`)**: 
   - Evaluates model performance **strictly on the test set**.
   - Calculates system-level metrics such as latency, throughput, model loading time, RAM, and CPU usage.
