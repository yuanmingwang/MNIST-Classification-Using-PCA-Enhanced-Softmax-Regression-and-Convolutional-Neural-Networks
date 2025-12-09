
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

from pca_utils import train_pca_baseline


if __name__ == "__main__":
    train_pca_baseline()
