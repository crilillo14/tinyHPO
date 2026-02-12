"""
            -- mnist.py --

Configurable MNIST classifier using tinygrad.

This module provides a flexible MNISTNet class that can be configured
with different hyperparameters for use with HPO methods.
"""
from tinygrad import nn
from tinygrad.tensor import Tensor
from tinygrad.nn.datasets import mnist
from typing import Optional


class MNISTNet:
    """
    Configurable MNIST classifier.

    Architecture:
        Conv2d(1, 32) -> ReLU -> MaxPool
        Conv2d(32, 64) -> ReLU -> MaxPool
        Flatten -> Dropout -> Linear(hidden_size) -> ReLU -> Dropout -> Linear(10)
    """

    def __init__(self,
                 dropout: float = 0.5) -> None:
        """
        Initialize MNIST network.

        Args:
            hidden_size: Size of the hidden fully-connected layer
            dropout: Dropout probability (0 = no dropout)
        """
        self.hidden_size = 128
        self.dropout = dropout

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))

        # After conv layers: 28->26->13->11->5, channels=64
        # Flattened size: 64 * 5 * 5 = 1600
        self.fc1 = nn.Linear(1600, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 10)

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # Conv block 1
        x = self.conv1(x).relu().max_pool2d((2, 2))
        # Conv block 2
        x = self.conv2(x).relu().max_pool2d((2, 2))
        # Flatten and FC layers
        x = x.flatten(1)
        x = x.dropout(self.dropout)
        x = self.fc1(x).relu()
        x = x.dropout(self.dropout)
        x = self.fc2(x)
        return x


def load_mnist():
    """Load MNIST dataset."""
    X_train, Y_train, X_test, Y_test = mnist()
    return X_train, Y_train, X_test, Y_test


def train_model(model: MNISTNet,
                X_train: Tensor,
                Y_train: Tensor,
                learning_rate: float = 0.001,
                batch_size: int = 128,
                epochs: int = 1,
                verbose: bool = False) -> float:
    """
    Train the model and return final training loss.

    Args:
        model: MNISTNet instance to train
        X_train: Training images
        Y_train: Training labels
        learning_rate: Learning rate for Adam optimizer
        batch_size: Batch size for training
        epochs: Number of training epochs
        verbose: Print progress

    Returns:
        Final training loss
    """
    optim = nn.optim.Adam(nn.state.get_parameters(model), lr=learning_rate)

    n_samples = X_train.shape[0]
    steps_per_epoch = n_samples // batch_size

    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            Tensor.training = True
            samples = Tensor.randint(batch_size, high=n_samples)
            x, y = X_train[samples], Y_train[samples]

            optim.zero_grad()
            loss = model(x).sparse_categorical_crossentropy(y)
            loss.backward()
            optim.step()

            epoch_loss += loss.item()

        epoch_loss /= steps_per_epoch
        final_loss = epoch_loss

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    return final_loss


def evaluate_model(model: MNISTNet,
                   X_test: Tensor,
                   Y_test: Tensor) -> float:
    """
    Evaluate model accuracy on test set.

    Args:
        model: Trained MNISTNet instance
        X_test: Test images
        Y_test: Test labels

    Returns:
        Accuracy as a float between 0 and 1
    """
    Tensor.training = False
    predictions = model(X_test).argmax(axis=1)
    accuracy = (predictions == Y_test).mean().item()
    return accuracy


if __name__ == "__main__":
    print(f"Device: {Device.DEFAULT}")

    # Load data
    print("Loading MNIST...")
    X_train, Y_train, X_test, Y_test = load_mnist()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Create and train model
    model = MNISTNet(dropout=0.5)
    
    config = {
        "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
        # "dropout" : ((0.1, 0.5) , 5)
    }
    
    
    

    print("\nTraining...")
    train_model(model, X_train, Y_train,
                learning_rate=0.001,
                epochs=3,
                verbose=True)

    # Evaluate
    accuracy = evaluate_model(model, X_test, Y_test)
    print(f"\nTest Accuracy: {accuracy:.2%}")
