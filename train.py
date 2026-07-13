import os
import json
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
TRAIN_DATA = "data/processed/train_processed.csv"
VALID_DATA = "data/processed/valid_processed.csv"
MODEL_SAVE_DIR = "model"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

CONFIG = {
    "learning_rate": 2e-5,
    "batch_size": 16,
    "epochs": 3,
    "optimizer": "AdamW",
    "scheduler": "None",
    "max_length": 64,
    "seed": 42,
    "model_name": "bert-base-uncased",
    "num_labels": 6
}
# ----------------------------------------

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def save_plot(history, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(history, marker='o')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()

def main():
    torch.manual_seed(CONFIG["seed"])
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Save training configuration
    with open(os.path.join(MODEL_SAVE_DIR, "training_config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)
        
    print("Loading datasets...")
    if not os.path.exists(TRAIN_DATA) or not os.path.exists(VALID_DATA):
        print(f"Error: Processed data not found. Run preprocess.py first.")
        return
        
    train_df = pd.read_csv(TRAIN_DATA)
    val_df = pd.read_csv(VALID_DATA)
    
    tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])
    
    train_dataset = NewsDataset(train_df["text"], train_df["label"], tokenizer, CONFIG["max_length"])
    val_dataset = NewsDataset(val_df["text"], val_df["label"], tokenizer, CONFIG["max_length"])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BertForSequenceClassification.from_pretrained(
        CONFIG["model_name"], 
        num_labels=CONFIG["num_labels"]
    )
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    
    training_history = []
    validation_history = []
    
    print("Starting training...")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_train_loss = 0
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if step % 100 == 0:
                print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        avg_train_loss = total_train_loss / len(train_loader)
        training_history.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        validation_history.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} finished | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
    # Save the model
    print(f"Saving model to {MODEL_SAVE_DIR}...")
    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    
    # Save training history
    history_dict = {
        "train_loss": training_history,
        "val_loss": validation_history
    }
    with open(os.path.join(RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history_dict, f, indent=4)
        
    # Generate Plots
    save_plot(training_history, "Training Loss over Epochs", "Loss", "training_loss.png")
    save_plot(validation_history, "Validation Loss over Epochs", "Loss", "validation_loss.png")
    print("Training complete. Results and plots saved.")

if __name__ == "__main__":
    main()
