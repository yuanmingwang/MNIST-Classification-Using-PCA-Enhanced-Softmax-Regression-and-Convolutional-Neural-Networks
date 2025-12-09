
"""pca_utils.py

Utilities for a PCA-based baseline classifier.

Rationale
---------
In this project, we want to have a comparison between:
    * a classical ML pipeline (PCA + classifier), and
    * a modern deep learning model (CNN).

By putting the PCA baseline into its own module, we can:
    * run classical experiments without touching any PyTorch code,
    * cleanly compare performance.

This module uses scikit-learn (PCA + LogisticRegression).
"""

from __future__ import annotations

import numpy as np
import joblib
import time

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config import (
    get_dataset_config,
    DEFAULT_DATASET,
    RANDOM_SEED,
    VALIDATION_FRACTION,
    PCA_N_COMPONENTS,
)
from data_utils import load_raw_train_arrays


def train_pca_baseline(dataset: str = DEFAULT_DATASET) -> None:
    """Train a PCA + Logistic Regression baseline on the chosen dataset.

    High-level steps
    ----------------
    1. Load train.csv (labels + flattened pixels).
    2. Split into training and validation sets.
    3. Normalize pixels to [0, 1].
    4. Fit PCA on training features only.
    5. Transform both training and validation sets.
    6. Train Logistic Regression on PCA features.
    7. Evaluate on validation set.
    8. Save (PCA, classifier) pair to disk via joblib.

    The saved model can then be used for predictions on test.csv.
    """
    start_time = time.perf_counter()  # overall timer for PCA baseline
    dataset_config = get_dataset_config(dataset)

    # 1. Load the data with dataset-aware parsing (handles Kaggle headers
    #    vs. headerless MNIST CSVs). This keeps PCA training aligned with
    #    the CNN pipeline's preprocessing.
    X, y = load_raw_train_arrays(dataset_config.name)

    # 2. Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_FRACTION,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    # 3. Simple normalization of pixel intensities
    X_train /= 255.0
    X_val /= 255.0

    # 4. Fit PCA on training data only (very important!)
    pca = PCA(
        n_components=PCA_N_COMPONENTS,
        random_state=RANDOM_SEED,
    )
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    # 5. Train a multinomial Logistic Regression classifier
    clf = LogisticRegression(
        max_iter=1000,        # allow more iterations for convergence
        multi_class="multinomial",
        solver="lbfgs",      # good default for multinomial LR
        n_jobs=-1,            # use all CPU cores (if allowed)
    )
    clf.fit(X_train_pca, y_train)

    # 6. Evaluate on validation set
    y_val_pred = clf.predict(X_val_pca)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"[PCA+LR][{dataset_config.name}] Validation accuracy: {val_acc:.4f}")

    # 7. Save both PCA object and classifier as a tuple
    joblib.dump((pca, clf), dataset_config.pca_model_path)
    print(
        f"[PCA+LR][{dataset_config.name}] Saved PCA+LR model to: "
        f"{dataset_config.pca_model_path}"
    )

    total_time = time.perf_counter() - start_time
    print(f"[PCA+LR][{dataset_config.name}] Training time: {total_time:.2f} sec")
