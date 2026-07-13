import pandas as pd
import os

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "speaker_job",
    "state", "party", "barely_true", "false", "half_true",
    "mostly_true", "pants_on_fire", "context"
]

LABEL_MAP = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}

def preprocess_split(split_name):
    raw_path = os.path.join(RAW_DATA_DIR, f"{split_name}.tsv")
    processed_path = os.path.join(PROCESSED_DATA_DIR, f"{split_name}_processed.csv")
    
    if not os.path.exists(raw_path):
        print(f"Warning: {raw_path} not found. Please place the LIAR dataset files in the data/raw/ directory.")
        return False
        
    print(f"Processing {split_name}...")
    df = pd.read_csv(raw_path, sep="\t", names=LIAR_COLUMNS)
    
    # Keep only statement and label
    df = df[["statement", "label"]]
    df.columns = ["text", "label"]
    
    # Map string labels to integers
    df["label"] = df["label"].map(LABEL_MAP)
    
    # Clean data
    df.dropna(inplace=True)
    df["text"] = df["text"].str.strip()
    df["label"] = df["label"].astype(int)
    
    # Save to processed directory
    df.to_csv(processed_path, index=False)
    print(f"Saved {len(df)} samples to {processed_path}")
    return True

def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    splits = ["train", "valid", "test"]
    success = True
    for split in splits:
        if not preprocess_split(split):
            success = False
            
    if success:
        print("Preprocessing completed successfully.")
    else:
        print("Preprocessing encountered missing files.")

if __name__ == "__main__":
    main()
