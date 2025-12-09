
"""data_utils.py

Dataset-aware CSV loaders for Kaggle and MNIST formats.

The two datasets look similar (28x28 grayscale digits) but their CSVs
are not:
    * Kaggle CSVs have a header row. The training CSV has a ``label``
      column plus pixel0..pixel783; the test CSV omits ``label``.
    * MNIST CSVs from ``mnist-in-csv`` have **no header row**. The first
      column of every row is the label; even the MNIST test CSV includes
      labels, which we discard for inference DataLoaders.

This module centralizes the parsing rules and returns either raw numpy
arrays or PyTorch DataLoaders, so callers never have to remember which
CSV has headers or embedded labels.
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    get_dataset_config,
    DEFAULT_DATASET,
    AVAILABLE_DATASETS,
    RANDOM_SEED,
    VALIDATION_FRACTION,
    BATCH_SIZE,
    NUM_WORKERS,
)


class ImageCSVDataset(Dataset):
    """Wrap flattened pixel arrays (and optional labels) as a Dataset.

    The Dataset handles reshaping to (1, 28, 28) and normalizing pixels
    to [0, 1] so training loops receive ready-to-use tensors.
    """

    def __init__(self, features: np.ndarray, labels: Optional[np.ndarray] = None):
        # Persist features as float32 for PyTorch; keep labels if provided
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64) if labels is not None else None

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        # Reshape flat vector -> (C=1, H=28, W=28) and normalize
        image = self.features[idx].reshape(1, 28, 28) / 255.0
        image_tensor = torch.from_numpy(image)

        if self.labels is None:
            return image_tensor

        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return image_tensor, label_tensor


# -------------------------------------------------------------------
# Raw array loaders (CSV -> numpy). These encode the per-dataset quirks.
# -------------------------------------------------------------------


def load_raw_train_arrays(dataset: str = DEFAULT_DATASET) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for the requested dataset with correct CSV parsing."""
    cfg = get_dataset_config(dataset)

    if cfg.name == "kaggle":
        # Header row present; labels live in an explicit "label" column
        df = pd.read_csv(cfg.train_csv)
        labels = df["label"].values
        pixels = df.drop(columns=["label"]).values
    elif cfg.name == "mnist":
        # No header row; column 0 is label, remainder are pixels
        df = pd.read_csv(cfg.train_csv, header=None)
        labels = df.iloc[:, 0].values
        pixels = df.iloc[:, 1:].values
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose from {AVAILABLE_DATASETS}.")

    return pixels.astype(np.float32), labels.astype(np.int64)


def load_raw_test_arrays(
    dataset: str = DEFAULT_DATASET,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (X, y_optional) for test data, preserving MNIST labels if present."""
    cfg = get_dataset_config(dataset)

    if cfg.name == "kaggle":
        # Kaggle test CSV has no labels at all
        df = pd.read_csv(cfg.test_csv)
        labels = None
        pixels = df.values
    elif cfg.name == "mnist":
        # MNIST test CSV still contains labels in the first column
        df = pd.read_csv(cfg.test_csv, header=None)
        labels = df.iloc[:, 0].values.astype(np.int64)
        pixels = df.iloc[:, 1:].values
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose from {AVAILABLE_DATASETS}.")

    return pixels.astype(np.float32), labels


# -------------------------------------------------------------------
# PyTorch DataLoader helpers for train/validation and test splits.
# -------------------------------------------------------------------


def load_train_val_dataloaders(
    dataset: str = DEFAULT_DATASET,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/validation DataLoaders honoring dataset-specific CSV shapes."""
    X, y = load_raw_train_arrays(dataset)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    train_dataset = ImageCSVDataset(X_train, y_train)
    val_dataset = ImageCSVDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # shuffle each epoch for SGD
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # validation set does not need shuffling
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader


def load_test_dataloader(
    dataset: str = DEFAULT_DATASET,
) -> DataLoader:
    """Create a test DataLoader; ignores labels even if MNIST test includes them."""
    X_test, _ = load_raw_test_arrays(dataset)
    test_dataset = ImageCSVDataset(X_test, labels=None)

    return DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )


__all__ = [
    "ImageCSVDataset",
    "load_raw_train_arrays",
    "load_raw_test_arrays",
    "load_train_val_dataloaders",
    "load_test_dataloader",
    "DEFAULT_DATASET",
]
