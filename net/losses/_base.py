from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    """
    Abstract base class for all loss functions.
    """

    @abstractmethod
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Compute the scalar loss value.

        Parameters:
        - prediction (np.ndarray): Model outputs.
        - target (np.ndarray): Ground truth labels.

        Returns:
        - float: Computed loss.
        """
        pass

    @abstractmethod
    def backward(self) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the prediction.

        Returns:
        - np.ndarray: Gradient of the loss w.r.t. prediction.
        """
        pass
