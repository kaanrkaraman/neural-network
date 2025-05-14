from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the model.

        Parameters:
        - x (np.ndarray): Input tensor.

        Returns:
        - np.ndarray: Output tensor.
        """
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through the model.

        Parameters:
        - grad_output (np.ndarray): Gradient from the loss.

        Returns:
        - np.ndarray: Gradient w.r.t. input.
        """
        pass

    @abstractmethod
    def update(self, learning_rate: float) -> None:
        """
        Update model parameters using computed gradients.

        Parameters:
        - learning_rate (float): Learning rate for gradient descent.
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run inference (without training logic).

        Parameters:
        - x (np.ndarray): Input tensor.

        Returns:
        - np.ndarray: Predicted output.
        """
        pass
