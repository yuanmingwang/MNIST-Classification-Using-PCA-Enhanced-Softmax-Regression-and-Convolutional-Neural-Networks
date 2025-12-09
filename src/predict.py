
"""predict.py

Script to generate predictions on ``test.csv`` using either:
    * the trained CNN model, or
    * the trained PCA + Logistic Regression baseline.

The output is saved to a dataset-specific submission file (see
``get_submission_path``), in a Kaggle-style format:

    ImageId,Label
    1,2
    2,0
    3,9
    ...

"""

from __future__ import annotations

import argparse
import time

import pandas as pd
import torch
import joblib

from config import (
    get_dataset_config,
    get_submission_path,
    AVAILABLE_DATASETS,
    DEFAULT_DATASET,
)
from data_utils import load_test_dataloader
from models import SimpleCNN


def predict_with_cnn(dataset: str = DEFAULT_DATASET) -> None:
    """Load the best CNN model and predict labels for ``test.csv``."""
    dataset_config = get_dataset_config(dataset)
    submission_path = get_submission_path(dataset_config.name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.perf_counter()  # wall-clock timer for inference

    # Instantiate a fresh model and load the saved parameters
    model = SimpleCNN().to(device)
    state_dict = torch.load(dataset_config.cnn_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # set to evaluation mode

    # Load test data
    test_loader = load_test_dataloader(dataset_config.name)

    predictions: list[int] = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)

            # Take argmax over class dimension -> predicted label
            _, predicted = torch.max(outputs, dim=1)
            predictions.extend(predicted.cpu().numpy().tolist())

    # Build submission DataFrame (1-based ImageId index is common)
    submission_df = pd.DataFrame(
        {
            "ImageId": list(range(1, len(predictions) + 1)),
            "Label": predictions,
        }
    )

    submission_df.to_csv(submission_path, index=False)
    duration = time.perf_counter() - start_time
    print(f"[CNN] Saved predictions to: {submission_path} | Inference time: {duration:.2f} sec")


def predict_with_pca(dataset: str = DEFAULT_DATASET) -> None:
    """Load the PCA+LR model and predict labels for ``test.csv``."""
    dataset_config = get_dataset_config(dataset)
    submission_path = get_submission_path(dataset_config.name)
    start_time = time.perf_counter()  # wall-clock timer for inference

    # Load (PCA, classifier) pair
    pca, clf = joblib.load(dataset_config.pca_model_path)

    # Load test data directly with pandas (no PyTorch needed here)
    df_test = pd.read_csv(dataset_config.test_csv)
    X_test = df_test.values.astype("float32") / 255.0

    # Apply PCA transform and classify
    X_test_pca = pca.transform(X_test)
    predictions = clf.predict(X_test_pca)

    submission_df = pd.DataFrame(
        {
            "ImageId": list(range(1, len(predictions) + 1)),
            "Label": predictions,
        }
    )
    submission_df.to_csv(submission_path, index=False)
    duration = time.perf_counter() - start_time
    print(f"[PCA+LR] Saved predictions to: {submission_path} | Inference time: {duration:.2f} sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run predictions on Kaggle or MNIST CSV test data."
    )
    parser.add_argument(
        "--dataset",
        choices=AVAILABLE_DATASETS,
        default=DEFAULT_DATASET,
        help="Dataset name that determines which CSV files and model checkpoints to load.",
    )
    parser.add_argument(
        "--model",
        choices=("cnn", "pca"),
        default="cnn",
        help="Choose which trained model to use for inference.",
    )
    args = parser.parse_args()

    if args.model == "cnn":
        predict_with_cnn(dataset=args.dataset)
    else:
        predict_with_pca(dataset=args.dataset)
