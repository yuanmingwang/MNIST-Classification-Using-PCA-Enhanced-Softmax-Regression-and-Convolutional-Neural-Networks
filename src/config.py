
"""config.py

Central place for file paths and hyperparameters.

This file is deliberately small and declarative: it should *not* contain
any heavy logic. Keeping all important "knobs" here makes it much easier
to run controlled experiments and to reproduce results for the report.

"""

import os
from dataclasses import dataclass

# -------------------------------------------------------------
# 1. Get the *project root* directory
#    config.py is inside: project_root/src/
#    so project_root = dirname(dirname(config.py))
# -------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))       # .../project/src
PROJECT_ROOT = os.path.dirname(SRC_DIR)                    # .../project/

# -------------------------------------------------------------
# 2. Build absolute paths for data, models, submission
# -------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
KAGGLE_DATA_DIR = os.path.join(DATA_DIR, "KAGGLE_CSV")
MNIST_DATA_DIR = os.path.join(DATA_DIR, "MNIST_CSV")

# Make sure the models folder exists
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

@dataclass(frozen=True)
class DatasetConfig:
    """Bundle all dataset-specific paths in one place.

    Keeping the paths together makes it trivial to switch between Kaggle
    and MNIST runs (and to add more datasets later without touching the
    training scripts).
    """

    name: str
    train_csv: str
    test_csv: str
    cnn_model_path: str
    pca_model_path: str


# Dataset-specific file locations
DATASETS = {
    "kaggle": DatasetConfig(
        name="kaggle",
        train_csv=os.path.join(KAGGLE_DATA_DIR, "train.csv"),
        test_csv=os.path.join(KAGGLE_DATA_DIR, "test.csv"),
        cnn_model_path=os.path.join(MODEL_DIR, "cnn_model_kaggle.pt"),
        pca_model_path=os.path.join(MODEL_DIR, "pca_model_kaggle.pkl"),
    ),
    "mnist": DatasetConfig(
        name="mnist",
        train_csv=os.path.join(MNIST_DATA_DIR, "mnist_train.csv"),
        test_csv=os.path.join(MNIST_DATA_DIR, "mnist_test.csv"),
        cnn_model_path=os.path.join(MODEL_DIR, "cnn_model_mnist.pt"),
        pca_model_path=os.path.join(MODEL_DIR, "pca_model_mnist.pkl"),
    ),
}

# Default dataset name and helper lists for CLI validation
DEFAULT_DATASET = "kaggle"
AVAILABLE_DATASETS = tuple(DATASETS.keys())


def get_dataset_config(name: str = DEFAULT_DATASET) -> DatasetConfig:
    """Return the configuration for a given dataset name.

    Parameters
    ----------
    name:
        One of ``AVAILABLE_DATASETS`` (case-insensitive).
    """
    normalized = name.lower()
    if normalized not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {', '.join(AVAILABLE_DATASETS)}"
        )
    return DATASETS[normalized]


def get_submission_path(dataset: str = DEFAULT_DATASET) -> str:
    """Build a dataset-specific submission filename to avoid clobbering."""
    dataset_config = get_dataset_config(dataset)
    return os.path.join(PROJECT_ROOT, f"submission_{dataset_config.name}.csv")


# Backwards-compatible shortcuts for the default dataset (Kaggle)
DEFAULT_DATASET_CONFIG = get_dataset_config(DEFAULT_DATASET)
TRAIN_CSV_PATH = DEFAULT_DATASET_CONFIG.train_csv  # training data with labels
TEST_CSV_PATH = DEFAULT_DATASET_CONFIG.test_csv    # test data without labels

# Folder + filenames for saved models (default dataset)
CNN_MODEL_PATH = DEFAULT_DATASET_CONFIG.cnn_model_path      # PyTorch state_dict
PCA_MODEL_PATH = DEFAULT_DATASET_CONFIG.pca_model_path      # joblib dump (PCA + classifier)

# Where to save predictions for the test set (default dataset)
SUBMISSION_CSV_PATH = get_submission_path(DEFAULT_DATASET)

# ---------------------------
# Data / training hyperparameters
# ---------------------------

# Random seed for reproducibility (train/val split, PCA, etc.)
RANDOM_SEED = 3

# Proportion of training data used for validation
VALIDATION_FRACTION = 0.2  # 20% validation, 80% training

# PyTorch DataLoader parameters
BATCH_SIZE = 64
# If you get issues with multi-processing on Windows, set NUM_WORKERS = 0
NUM_WORKERS = 4

# CNN training hyperparameters
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # L2 regularization term (weight decay in Adam)

# PCA parameters
# Number of principal components. You can tune this (e.g. 50, 100, 200).
PCA_N_COMPONENTS = 100

# Number of classes in the classification problem (e.g., 10 for digits 0-9)
NUM_CLASSES = 10
