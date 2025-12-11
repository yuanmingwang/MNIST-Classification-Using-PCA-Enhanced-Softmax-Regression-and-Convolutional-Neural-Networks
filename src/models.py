
"""models.py

Model definitions: at the moment we provide a simple convolutional
neural network suitable for 28x28 grayscale images (e.g., MNIST).

The idea is to keep this file focused purely on network *architecture*.
Training logic (loss functions, optimizers, loops) is handled elsewhere.

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES


class SimpleCNN(nn.Module):
    """A small convolutional neural network for image classification.

    Architecture (example)
    ----------------------
    Input:  (N, 1, 28, 28)

    1. Conv2d(1 -> 32, kernel_size=3) + ReLU
       -> (N, 32, 26, 26)
       MaxPool2d(2x2)
       -> (N, 32, 13, 13)

    2. Conv2d(32 -> 64, kernel_size=3) + ReLU
       -> (N, 64, 11, 11)
       MaxPool2d(2x2)
       -> (N, 64, 5, 5)

    3. Flatten: (N, 64*5*5) = (N, 1600)
       Fully connected layer to 128 units + ReLU
       Fully connected layer to NUM_CLASSES logits.

    This is intentionally simple so it trains fast and is easy to explain.
    Can add BatchNorm, Dropout, more layers, etc in the future.
    """

    def __init__(self):
        super().__init__()

        # Convolutional layers extract local patterns (edges, blobs, etc.)
        self.conv1 = nn.Conv2d(
            in_channels=1,   # grayscale input
            out_channels=32, # number of learned filters
            kernel_size=3,   # 3x3 convolution kernel
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
        )

        # After the two conv + pooling operations, the spatial resolution
        # becomes 5x5 (see docstring above), so there are 64*5*5 features.
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass of the CNN.

        Parameters
        ----------
        x:
            Input tensor of shape (batch_size, 1, 28, 28).

        Returns
        -------
        logits:
            Output tensor of shape (batch_size, NUM_CLASSES). These are
            *unnormalized* scores; typically pass them to
            ``nn.CrossEntropyLoss`` which applies ``softmax`` internally.
        """
        # First conv block: conv -> ReLU -> max pooling
        x = self.conv1(x)              # (B, 32, 26, 26)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)  # (B, 32, 13, 13)

        # Second conv block
        x = self.conv2(x)              # (B, 64, 11, 11)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)  # (B, 64, 5, 5)

        # Flatten all channels and spatial dimensions into a single vector
        x = torch.flatten(x, start_dim=1)   # (B, 64*5*5)

        # Fully connected layers map features to class logits
        x = self.fc1(x)
        x = F.relu(x)
        logits = self.fc2(x)

        return logits


class PCACNN(nn.Module):
    """1D CNN that operates directly on PCA component vectors.

    Input shape: (N, 1, L) where L = number of PCA components. This keeps
    the reduced dimensionality (e.g., 100 comps instead of 784 pixels)
    and applies 1D convolutions over the component sequence.

    Example spatial dimensions (L = 100):
        Input: (N, 1, 100)

        1) Conv1d(1 -> 16, k=3) + ReLU
           -> (N, 16, 98)
           MaxPool1d(2)
           -> (N, 16, 49)

        2) Conv1d(16 -> 32, k=3) + ReLU
           -> (N, 32, 47)
           MaxPool1d(2)
           -> (N, 32, 23)   # floor division for pooling

        Flatten: (N, 32*23)
        FC -> 128 -> NUM_CLASSES logits.

    The lengths above adjust automatically based on the provided
    ``input_length`` so we can choose any PCA_N_COMPONENTS without
    rewriting the network.
    """

    def __init__(self, input_length: int):
        super().__init__()

        # Convolutional stack mirrors SimpleCNN but uses 1D ops over components
        # Widths are smaller than the 2D CNN to reflect the already-compressed input.
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)

        # Compute length after conv/pool stack to size the FC layer
        length_after = self._compute_length(input_length)
        if length_after < 1:
            raise ValueError(
                "PCA component sequence too short for two conv+pool blocks. "
                "Increase PCA components to grow the 1D length."
            )

        self.fc1 = nn.Linear(32 * length_after, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    @staticmethod
    def _compute_length(length: int) -> int:
        """Compute 1D length after two (conv3 + pool2) blocks."""
        length = length - 2   # conv1
        length = length // 2  # pool1
        length = length - 2   # conv2
        length = length // 2  # pool2
        return length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        logits = self.fc2(x)
        return logits
