
"""train_cnn.py

Script to train the convolutional neural network (CNN).

This file glues together:
    * the data loading utilities (from ``data_utils``),
    * the CNN architecture (from ``models``), and
    * the training loop itself (loss, optimizer, evaluation).

This is kept separate from the notebook so that we can:
    * run training from the command line,
    * call this script programmatically from a notebook if desired,
    * keep our Jupyter notebook focused on experiment tracking and plots.

Usage
-----
From the project root:

    python src/train_cnn.py

This will read ``data/train.csv``, train the network, and save the best
weights to ``models/cnn_model.pt``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CNN_MODEL_PATH,
)
from data_utils import load_train_val_dataloaders
from models import SimpleCNN


def train_cnn() -> None:
    """Main training routine for the CNN model."""
    # Use GPU if available; otherwise, fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loaders for training and validation splits
    train_loader, val_loader = load_train_val_dataloaders()

    # Instantiate the CNN and move it to the chosen device
    model = SimpleCNN().to(device)

    # Cross-entropy loss is standard for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer with optional L2 weight decay (regularization)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_accuracy = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---------------------------
        # Training phase
        # ---------------------------
        model.train()  # set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            # Move data to the same device as the model
            images = images.to(device)
            labels = labels.to(device)

            # Reset gradients from the previous iteration
            optimizer.zero_grad()

            # Forward pass: compute predicted scores
            outputs = model(images)

            # Compute training loss
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients w.r.t. weights
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate batch loss (scaled by batch size)
            running_loss += loss.item() * images.size(0)

            # Compute training accuracy for monitoring
            _, predicted = torch.max(outputs, dim=1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        # Compute average loss and accuracy over the epoch
        avg_train_loss = running_loss / total_train
        train_accuracy = correct_train / total_train

        # ---------------------------
        # Validation phase
        # ---------------------------
        model.eval()  # set model to evaluation mode
        correct_val = 0
        total_val = 0

        # No gradients are needed during evaluation
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
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # Save the model if validation accuracy improved
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), CNN_MODEL_PATH)
            print(f"  -> New best model saved with Val Acc = {best_val_accuracy:.4f}")


if __name__ == "__main__":
    train_cnn()
