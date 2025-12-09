
"""data_utils.py

Data loading and preprocessing utilities.

This module is responsible for:
    - Loading train.csv and test.csv from disk.
    - Splitting the training set into training and validation subsets.
    - Wrapping data in PyTorch Dataset and DataLoader objects.

By separating these responsibilities from the model/training code, we:
    * avoid duplicating CSV-loading code,
    * keep the training scripts compact, and
    * can easily swap in a different dataset loader in the future.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    TRAIN_CSV_PATH,
    TEST_CSV_PATH,
    RANDOM_SEED,
    VALIDATION_FRACTION,
    BATCH_SIZE,
    NUM_WORKERS,
)


class ImageCSVDataset(Dataset):
    """Generic Dataset for image data stored in CSV format.

    Expected CSV format
    -------------------
    * Training/validation:
        - Column "label" contains integer class labels (0..NUM_CLASSES-1).
        - All other columns are pixel values (one row per image).
    * Test:
        - No "label" column; only pixel columns.

    The dataset keeps the raw features as a NumPy array and performs
    light preprocessing (reshape, normalization) in ``__getitem__``.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray | None = None):
        """Create a dataset from raw feature and label arrays.

        Parameters
        ----------
        features:
            2D array of shape (N_samples, N_features) containing flattened
            pixel data.
        labels:
            1D array of shape (N_samples,) with integer labels, or ``None``
            if this is a test set.
        """
        # Store data as float32 for compatibility with PyTorch tensors
        self.features = features.astype(np.float32)
        # If labels are provided, cast to int64 (PyTorch's default for classes)
        self.labels = labels.astype(np.int64) if labels is not None else None

    def __len__(self) -> int:
        """Return total number of samples in the dataset."""
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        """Return a single sample at index ``idx``.

        Returns
        -------
        (image_tensor, label_tensor)
            For training/validation datasets (labels is not None).
        image_tensor
            For test datasets (labels is None).
        """
        # Extract the flattened pixel vector
        x = self.features[idx]

        # Here we assume 28x28 grayscale images. Adjust if your image
        # resolution or channel count is different.
        image = x.reshape(1, 28, 28)  # (channels=1, height=28, width=28)

        # Normalize pixel values from [0, 255] to [0, 1]
        image = image / 255.0

        # Convert to a torch.FloatTensor
        image_tensor = torch.from_numpy(image)

        if self.labels is not None:
            label = self.labels[idx]
            label_tensor = torch.tensor(label, dtype=torch.long)
            return image_tensor, label_tensor

        # For test data, we do not have labels
        return image_tensor


def load_train_val_dataloaders() -> tuple[DataLoader, DataLoader]:
    """Load training data and return train/validation DataLoaders.

    Steps
    -----
    1. Read ``TRAIN_CSV_PATH`` into a pandas DataFrame.
    2. Separate labels (y) from features (X).
    3. Split into train and validation sets using ``train_test_split``.
    4. Wrap each split in ``ImageCSVDataset``.
    5. Wrap datasets in PyTorch ``DataLoader`` objects.

    Returns
    -------
    train_loader, val_loader
        DataLoaders for the training and validation subsets.
    """
    # 1. Read CSV file
    df = pd.read_csv(TRAIN_CSV_PATH)

    # 2. Separate label column (y) and pixel columns (X)
    y = df["label"].values
    X = df.drop(columns=["label"]).values

    # 3. Split into training and validation subsets
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y,  # preserves class distribution
    )

    # 4. Create Dataset objects
    train_dataset = ImageCSVDataset(X_train, y_train)
    val_dataset = ImageCSVDataset(X_val, y_val)

    # 5. Wrap in DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,      # shuffle each epoch for SGD
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,     # no need to shuffle validation set
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader


def load_test_dataloader() -> DataLoader:
    """Load test data and return a DataLoader.

    The test CSV is assumed to contain only pixel columns (no labels).

    Returns
    -------
    test_loader : DataLoader
        DataLoader delivering only image tensors.
    """
    df_test = pd.read_csv(TEST_CSV_PATH)
    X_test = df_test.values

    test_dataset = ImageCSVDataset(X_test, labels=None)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return test_loader
