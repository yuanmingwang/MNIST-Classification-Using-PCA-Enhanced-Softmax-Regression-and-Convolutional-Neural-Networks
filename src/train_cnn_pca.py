"""train_cnn_pca.py

Train a CNN with PCA preprocessing (1D or 2D layout).

Workflow:
    1) Fit PCA on the *training* split only (after normalizing pixels).
    2) Transform train/val splits into low-dimensional PCA components.
    3) Feed either:
       - a 1D CNN over the component sequence, or
       - a 2D CNN over a zero-padded square grid of components.

This keeps the CNN architecture unchanged while letting PCA act as a
dimensionality-reduction + denoising step. Each dataset (Kaggle vs
MNIST) is parsed with its own CSV rules via ``data_utils``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from config import (
    PCA_N_COMPONENTS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    AVAILABLE_DATASETS,
    DEFAULT_DATASET,
    RANDOM_SEED,
    VALIDATION_FRACTION,
    BATCH_SIZE,
    NUM_WORKERS,
    get_dataset_config,
)
from data_utils import load_raw_train_arrays
from models import PCACNN, PCACNN2D


class PCAFeatureDataset(Dataset):
    """Dataset for PCA component vectors (kept 1D) plus labels."""

    def __init__(self, components: np.ndarray, labels: np.ndarray):
        # Keep the PCA-reduced features exactly as-is; shape becomes (N, 1, L)
        # to satisfy Conv1d's expected (N, C, L) input ordering.
        self.features = components.astype(np.float32)[:, None, :]
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        # CNN expects float tensor and int64 labels
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class PCAFeatureDataset2D(Dataset):
    """Dataset for PCA component vectors packed into a square grid."""

    def __init__(self, components: np.ndarray, labels: np.ndarray, grid_side: int):
        # Pad components into a square grid (row-wise); zeros fill unused cells.
        num_samples, num_features = components.shape
        grid_size = grid_side * grid_side
        padded = np.zeros((num_samples, grid_size), dtype=np.float32)
        padded[:, :num_features] = components.astype(np.float32)
        self.features = padded.reshape(num_samples, 1, grid_side, grid_side)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def build_component_loaders(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA on normalized train data and return splits + timer.

    Parameters
    ----------
    X, y:
        Full dataset of flattened pixels (shape: [N, 784]) and labels.
    n_components:
        Number of PCA components to retain before reshaping into a grid.
    """
    # Split first so PCA is fit only on the training subset
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Normalize to [0, 1] before PCA to keep scale consistent
    X_train_norm = X_train.astype(np.float32) / 255.0
    X_val_norm = X_val.astype(np.float32) / 255.0

    # Fit PCA on the training split only, then transform both splits
    pca = PCA(
        n_components=n_components,
        random_state=RANDOM_SEED,
    )

    pca_start = time.perf_counter()
    X_train_pca = pca.fit_transform(X_train_norm)
    X_val_pca = pca.transform(X_val_norm)
    pca_fit_time = time.perf_counter() - pca_start
    print(
        f"[PCA] Fit completed with {n_components} components in {pca_fit_time:.2f} sec "
        f"(variance retained not printed; tune via PCA_N_COMPONENTS)."
    )

    return X_train_pca, y_train, X_val_pca, y_val, pca_fit_time


def train_cnn_with_pca(
    dataset: str = DEFAULT_DATASET,
    n_components: int = PCA_N_COMPONENTS,
    layout: str = "1d",
) -> None:
    """Train the CNN using PCA-reconstructed images as input."""
    dataset_config = get_dataset_config(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using device: {device} | dataset: {dataset_config.name} | "
        f"PCA components: {n_components} | layout: {layout}"
    )

    # Load raw arrays with dataset-aware CSV parsing
    X, y = load_raw_train_arrays(dataset_config.name)

    # PCA fit/transform once up front
    (
        X_train_pca,
        y_train,
        X_val_pca,
        y_val,
        pca_fit_time,
    ) = build_component_loaders(X, y, n_components)
    print(f"[PCA] Fit + transform time: {pca_fit_time:.2f} sec")
    print(f"[Data] Train samples: {len(y_train)} | Val samples: {len(y_val)}")

    # Build loaders depending on layout choice
    if layout == "1d":
        train_dataset = PCAFeatureDataset(X_train_pca, y_train)
        val_dataset = PCAFeatureDataset(X_val_pca, y_val)
    elif layout == "2d":
        grid_side = int(np.ceil(np.sqrt(n_components)))
        train_dataset = PCAFeatureDataset2D(X_train_pca, y_train, grid_side)
        val_dataset = PCAFeatureDataset2D(X_val_pca, y_val, grid_side)
        print(f"[PCA] 2D grid side: {grid_side} (padded with zeros if needed)")
    else:
        raise ValueError("layout must be '1d' or '2d'")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # CNN sized to the PCA sequence length, not the original 28x28 image
    print(f"[PCA] Component length after reduction: {n_components}")
    if layout == "1d":
        model = PCACNN(input_length=n_components).to(device)
        checkpoint_path = dataset_config.cnn_pca_model_path.replace(".pt", "_1d.pt")
    else:
        grid_side = int(np.ceil(np.sqrt(n_components)))
        model = PCACNN2D(input_side=grid_side).to(device)
        checkpoint_path = dataset_config.cnn_pca_model_path.replace(".pt", "_2d.pt")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_accuracy = 0.0
    epoch_durations: list[float] = []
    train_start = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.perf_counter()

        # ---------------------------
        # Training phase
        # ---------------------------
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = running_loss / total_train
        train_accuracy = correct_train / total_train

        # ---------------------------
        # Validation phase
        # ---------------------------
        model.eval()
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, dim=1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_accuracy = correct_val / total_val

        print(
            f"[PCA+CNN] Epoch [{epoch}/{NUM_EPOCHS}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Save to a dedicated PCA+CNN checkpoint so plain CNN runs are untouched
            torch.save(model.state_dict(), checkpoint_path)
            print(
                f"  -> New best model saved ({checkpoint_path}) "
                f"with Val Acc = {best_val_accuracy:.4f}"
            )

        epoch_duration = time.perf_counter() - epoch_start
        epoch_durations.append(epoch_duration)
        print(f"  Epoch time: {epoch_duration:.2f} sec")

    total_time = time.perf_counter() - train_start
    avg_epoch_time = sum(epoch_durations) / len(epoch_durations)
    print(
        f"[PCA+CNN] Training complete | Total: {total_time:.2f} sec | "
        f"Avg/epoch: {avg_epoch_time:.2f} sec"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train CNN with PCA preprocessing (Kaggle or MNIST CSV datasets). "
            "Supports 1D (Conv1d) or 2D (Conv2d) layouts over PCA features."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=AVAILABLE_DATASETS,
        default=DEFAULT_DATASET,
        help="Dataset name that determines CSV parsing rules and checkpoint paths.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=PCA_N_COMPONENTS,
        help="Number of PCA components to retain before inverse-transforming.",
    )
    parser.add_argument(
        "--layout",
        choices=("1d", "2d"),
        default="1d",
        help="Choose Conv1d over PCA sequence or Conv2d over a padded square grid.",
    )
    args = parser.parse_args()

    train_cnn_with_pca(
        dataset=args.dataset,
        n_components=args.pca_components,
        layout=args.layout,
    )
