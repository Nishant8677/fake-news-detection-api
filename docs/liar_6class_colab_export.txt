from google.colab import drive
drive.mount('/content/drive')

---
import pandas as pd

columns = [
    "id", "label", "statement", "subject", "speaker", "speaker_job",
    "state", "party", "barely_true", "false", "half_true",
    "mostly_true", "pants_on_fire", "context"
]

base_path = "/content/drive/MyDrive/Datasets/LIAR_Dataset/"

train_df = pd.read_csv(base_path + "train.tsv", sep="\t", names=columns)
val_df   = pd.read_csv(base_path + "valid.tsv", sep="\t", names=columns)
test_df  = pd.read_csv(base_path + "test.tsv",  sep="\t", names=columns)

---
train_df.head()

---
#keeping only the required columns
train_df = train_df[["statement","label"]]
test_df = test_df[["statement","label"]]
val_df = val_df[["statement","label"]]
print(train_df.label.value_counts())
print(test_df.label.value_counts())
print(val_df.label.value_counts())
---
train_df.columns = ["text", "label"]
val_df.columns   = ["text", "label"]
test_df.columns  = ["text", "label"]









---
label_map = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}

---
train_df["label"] = train_df["label"].map(label_map)
test_df["label"] = test_df["label"].map(label_map)
val_df["label"] = val_df["label"].map(label_map)
---
for df in [train_df, val_df, test_df]:
    df.dropna(inplace=True)
    df["text"] = df["text"].str.strip()


---
texts = df["text"].values
labels = df["label"].values

---
X_train = train_df["text"].values
y_train = train_df["label"].values

X_val = val_df["text"].values
y_val = val_df["label"].values

---
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

---
def tokenize_texts(tokenizer):
  return tokenizer(
      list(texts),
      padding = True,
      truncation = True,
      max_length = 64,
      return_tensors = "pt"
  )

---
import torch
from torch.utils.data import Dataset




class NewsDataset(Dataset):
  def __init__(self,texts,labels,tokenizer,max_length =64):
    self.texts = list(texts)
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.texts)

  def __getitem__(self,idx):
    encoding = self.tokenizer(
      self.texts[idx],
      padding = "max_length",
      truncation = True,
      max_length = self.max_length,
      return_tensors = "pt"
  )
    return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
    }


---
train_dataset = NewsDataset(X_train, y_train, tokenizer)
val_dataset   = NewsDataset(X_val, y_val, tokenizer)

---
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

---
import torch
from transformers import BertForSequenceClassification
from torch.optim import AdamW

---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

---
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=6
)

---
model.to(device)

---
optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)

---
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

---
epochs = 3   # you can set 1, 2, or 3 depending on stability

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for step, batch in enumerate(train_loader):

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # progress update
        if step % 100 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Step {step}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} finished | Avg loss: {avg_loss:.4f}")

---
save_path = "/content/drive/MyDrive/fake_news_bert_liar"

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

---
import os
os.listdir("/content/drive/MyDrive/fake_news_bert_liar")

---
model.eval()

---
import torch
from sklearn.metrics import accuracy_score, classification_report

predictions = []
true_labels = []

---
with torch.no_grad():
    for batch in val_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

---
print("Accuracy:", accuracy_score(true_labels, predictions))
print(classification_report(true_labels, predictions))
