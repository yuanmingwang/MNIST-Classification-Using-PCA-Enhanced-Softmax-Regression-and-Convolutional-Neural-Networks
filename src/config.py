
"""config.py

Central place for file paths and hyperparameters.

This file is deliberately small and declarative: it should *not* contain
any heavy logic. Keeping all important "knobs" here makes it much easier
to run controlled experiments and to reproduce results for the report.

"""

import os

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

# The CSV files lies in `data/` at the root.
TRAIN_CSV_PATH = os.path.join(KAGGLE_DATA_DIR, "train.csv")  # training data with labels
TEST_CSV_PATH = os.path.join(KAGGLE_DATA_DIR, "test.csv")    # test data without labels

# Folder + filenames for saved models
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.pt")      # PyTorch state_dict
PCA_MODEL_PATH = os.path.join(MODEL_DIR, "pca_model.pkl")     # joblib dump (PCA + classifier)

# Where to save predictions for the test set
SUBMISSION_CSV_PATH = os.path.join(PROJECT_ROOT, "submission.csv")      # written into project root

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
