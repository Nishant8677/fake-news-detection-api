# Training Guide

## 1. Setup Data

Download the LIAR dataset and place the following files into the `data/raw/` directory:
- `train.tsv`
- `valid.tsv`
- `test.tsv`

## 2. Preprocess

Run the preprocessing script to clean the data and map labels.
```bash
python preprocess.py
```
This will generate `train_processed.csv`, `valid_processed.csv`, and `test_processed.csv` in `data/processed/`.

## 3. Train the Model

Run the training script.
```bash
python train.py
```

### What happens during training?
1. The script loads the hyperparameter configuration defined in `train.py`.
2. It saves this configuration to `model/training_config.json` for reproducibility.
3. The dataset is tokenized on the fly using `NewsDataset`.
4. The model trains for the specified number of epochs, evaluating against `valid_processed.csv` at the end of each epoch.
5. Loss metrics are logged and plotted automatically to `plots/training_loss.png` and `plots/validation_loss.png`.
6. The final model is saved to the `model/` directory.

### Configuration
You can adjust the `CONFIG` dictionary at the top of `train.py`:
- `learning_rate`
- `batch_size`
- `epochs`
- `max_length`

## 4. Evaluate

To get a full breakdown of performance on the unseen test set, run the benchmark script:
```bash
python benchmark.py
```
