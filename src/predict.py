
"""predict.py

Script to generate predictions on ``test.csv`` using either:
    * the trained CNN model, or
    * the trained PCA + Logistic Regression baseline.

The output is saved to ``SUBMISSION_CSV_PATH`` (see config.py), in a
Kaggle-style format:

    ImageId,Label
    1,2
    2,0
    3,9
    ...

"""

from __future__ import annotations

import pandas as pd
import torch
import joblib

from config import (
    SUBMISSION_CSV_PATH,
    CNN_MODEL_PATH,
    PCA_MODEL_PATH,
)
from data_utils import load_test_dataloader
from models import SimpleCNN
from config import TEST_CSV_PATH


def predict_with_cnn() -> None:
    """Load the best CNN model and predict labels for ``test.csv``."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate a fresh model and load the saved parameters
    model = SimpleCNN().to(device)
    state_dict = torch.load(CNN_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # set to evaluation mode

    # Load test data
    test_loader = load_test_dataloader()

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

    submission_df.to_csv(SUBMISSION_CSV_PATH, index=False)
    print(f"[CNN] Saved predictions to: {SUBMISSION_CSV_PATH}")


def predict_with_pca() -> None:
    """Load the PCA+LR model and predict labels for ``test.csv``."""
    # Load (PCA, classifier) pair
    pca, clf = joblib.load(PCA_MODEL_PATH)

    # Load test data directly with pandas (no PyTorch needed here)
    df_test = pd.read_csv(TEST_CSV_PATH)
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
    submission_df.to_csv(SUBMISSION_CSV_PATH, index=False)
    print(f"[PCA+LR] Saved predictions to: {SUBMISSION_CSV_PATH}")


if __name__ == "__main__":
    # Toggle this flag to choose which model to use for prediction
    USE_CNN = True  # set to False to use PCA baseline instead

    if USE_CNN:
        predict_with_cnn()
    else:
        predict_with_pca()
