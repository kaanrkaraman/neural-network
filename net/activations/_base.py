from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    """
    Abstract base class for activation functions.
    All activation functions must implement forward and backward methods.
    """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass.
        Stores any necessary internal state for use in backward.

        Parameters:
        - x (np.ndarray): Input tensor.

        Returns:
        - np.ndarray: Output after applying activation.
        """
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass using internal state.

        Parameters:
        - grad_output (np.ndarray): Gradient from the next layer.

        Returns:
        - np.ndarray: Gradient with respect to input.
        """
        pass

    @abstractmethod
    def update(self, learning_rate: float) -> None:
        """
        Update any internal parameters if necessary.

        Parameters:
        - learning_rate (float): Learning rate for updates.
        """
        pass
