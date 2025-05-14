from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    """
    Base class for all layers in the neural network.
    """

    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.

        Parameters:
        - x (np.ndarray): Input tensor.

        Returns:
        - np.ndarray: Output tensor.
        """
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through the layer.

        Parameters:
        - grad_output (np.ndarray): Gradient from next layer.

        Returns:
        - np.ndarray: Gradient w.r.t. input.
        """
        pass

    @abstractmethod
    def update(self, learning_rate: float) -> None:
        """
        Update the layer's parameters.
        """
        pass
