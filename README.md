# MNIST Digit Classification – ECE 503 Project

This project implements three pipelines for classifying 28×28 grayscale digits (Kaggle-style CSVs and MNIST-in-CSV):
- **PCA + Softmax**: classical baseline (PCA for dimensionality reduction + multinomial logistic regression).
- **CNN**: baseline 2D convolutional network on raw pixels.
- **PCA + CNN**: CNNs trained on PCA-reduced features, in 1D (Conv1d) or 2D (Conv2d over a padded PCA grid).

All training/evaluation scripts are in `src/` and share configurable hyperparameters via `src/config.py`.

## Data
- Kaggle CSVs: `data/KAGGLE_CSV/train.csv`, `data/KAGGLE_CSV/test.csv` (train has a `label` column; test does not).
- MNIST CSVs: `data/MNIST_CSV/mnist_train.csv`, `data/MNIST_CSV/mnist_test.csv` (no header; first column is the label, even in the “test” CSV).
- Choose dataset at runtime with `--dataset kaggle` or `--dataset mnist` (defaults to Kaggle).

## Environment Setup
1) Create/activate a virtual environment (recommended).  
2) Install dependencies:
```bash
pip install -r requirements.txt
```

## Key Scripts
- `src/train_pca.py`: trains PCA + Logistic Regression baseline.
- `src/train_cnn.py`: trains the baseline 2D CNN on raw pixels.
- `src/train_cnn_pca.py`: trains CNNs on PCA-reduced data; supports 1D or 2D layouts via `--layout {1d,2d}`.
- `src/predict.py`: generates predictions using either the CNN or PCA baseline (`--model cnn|pca`).

## Running Training
From the project root:
```bash
# PCA + Softmax
python3 src/train_pca.py --dataset kaggle

# CNN on raw pixels
python3 src/train_cnn.py --dataset mnist

# PCA + CNN (1D layout over component sequence)
python3 src/train_cnn_pca.py --dataset mnist --pca-components 100 --layout 1d

# PCA + CNN (2D layout over padded sqrt(k)×sqrt(k) grid)
python3 src/train_cnn_pca.py --dataset kaggle --pca-components 100 --layout 2d
```

## Inference / Predictions
```bash
# Using CNN checkpoint
python3 src/predict.py --dataset kaggle --model cnn

# Using PCA + Logistic Regression baseline
python3 src/predict.py --dataset mnist --model pca
```
Predictions are written to `submission_<dataset>.csv` in the project root (e.g., `submission_kaggle.csv`). CNN checkpoints are saved under `models/cnn_model_<dataset>.pt`; PCA baselines under `models/pca_model_<dataset>.pkl`; PCA+CNN variants under `models/cnn_pca_model_<dataset>_1d.pt` or `_2d.pt`.

## Hyperparameters and Tuning
Edit `src/config.py` for global defaults:
- Learning rate (`LEARNING_RATE`), weight decay (`WEIGHT_DECAY`), epochs (`NUM_EPOCHS`), batch size (`BATCH_SIZE`), validation fraction (`VALIDATION_FRACTION`), PCA components (`PCA_N_COMPONENTS`), number of classes (`NUM_CLASSES`), data/model paths.
- Use CLI flags to override where available:
  - `--dataset {kaggle,mnist}` on all train scripts.
  - `--pca-components <int>` and `--layout {1d,2d}` on `train_cnn_pca.py`.
  - `--model {cnn,pca}` on `predict.py`.

Tips:
- Increase `PCA_N_COMPONENTS` (e.g., 150) if accuracy drops; decrease for faster/lighter models.
- For CPU runs, smaller batch sizes (32) can help memory; on GPU, try 64–128.
- Weight decay of `1e-4` is a good default; raise to reduce overfitting if needed.

## Outputs
- Checkpoints: saved in `models/` (per-dataset, per-layout filenames as noted above).
- Submissions: `submission_<dataset>.csv` in the project root for kaggle.
- Logs: timing per epoch printed to stdout; best-model saves announced during training.

## Project Structure
- `src/config.py`: paths and hyperparameters.
- `src/data_utils.py`: dataset-aware CSV loaders (Kaggle vs MNIST).
- `src/models.py`: model definitions (`SimpleCNN`, `PCACNN`, `PCACNN2D`).
- `src/train_*.py`: training entry points (PCA baseline, CNN, PCA+CNN).
- `src/predict.py`: inference for CNN or PCA baseline.
- `data/`: Kaggle and MNIST CSVs.
- `models/`: saved checkpoints.

## Repro Notes
- Training scripts time each epoch and report total/average duration.
- PCA is fit on the training split only (never on validation/test).
- Validation splits are stratified by label for balanced evaluation.
