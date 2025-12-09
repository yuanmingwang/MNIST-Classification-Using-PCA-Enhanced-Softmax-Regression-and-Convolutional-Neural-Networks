
"""train_pca.py

Tiny wrapper script to train the PCA baseline.

This keeps the command-line entry point very simple and mirrors the
structure of ``train_cnn.py``.

Usage
-----
From the project root:

    python src/train_pca.py
"""

from __future__ import annotations

import argparse

from pca_utils import train_pca_baseline
from config import AVAILABLE_DATASETS, DEFAULT_DATASET


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PCA + Logistic Regression baseline on Kaggle or MNIST CSV data."
    )
    parser.add_argument(
        "--dataset",
        choices=AVAILABLE_DATASETS,
        default=DEFAULT_DATASET,
        help="Dataset name to load CSV files from.",
    )
    args = parser.parse_args()

    train_pca_baseline(dataset=args.dataset)
