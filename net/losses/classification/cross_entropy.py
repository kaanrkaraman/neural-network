from typing import Optional

import numpy as np

from net.losses._base import Loss


class CrossEntropy(Loss):
    """
    Cross-Entropy loss computation class.

    This class is designed to compute and manage the cross-entropy loss for
    multi-class classification tasks. The `forward` method calculates the cross-entropy
    loss based on the predicted probabilities and the ground truth target values. The
    `backward` method computes the derivative of the loss with respect to the predicted
    values. Both methods are essential for training machine learning models via
    backpropagation.

    :ivar pred: Stores the predicted probabilities, which are used during the backward
        pass for gradient computation.
    :type pred: numpy.ndarray
    :ivar target: Holds the ground truth target values, which are used to compute
        the loss during the forward pass and gradients during the backward pass.
    :type target: numpy.ndarray
    """

    def __init__(self):
        self.pred: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Calculates the cross-entropy loss between predicted probabilities and target labels.

        This method computes the cross-entropy loss, a standard loss function
        used for multi-class classification tasks. The loss is calculated using the
        formula:
            loss = -mean(sum(target * log(pred), axis=1))

        Predicted probabilities are clipped to avoid numerical instability in
        logarithmic calculations.

        :param pred: Predicted probabilities. Values should be in the range [0, 1].
        :param target: Ground truth one-hot encoded labels.
        :return: The computed cross-entropy loss value.
        :rtype: float
        """
        pred = np.clip(pred, 1e-7, 1 - 1e-7)
        self.pred = pred
        self.target = target
        loss = -np.mean(np.sum(target * np.log(pred), axis=1))
        return float(loss)

    def backward(self) -> np.ndarray:
        """
        Computes the gradient of the cross-entropy loss with respect to the
        predictions, given the current predicted values and target values. This
        function is typically used in the context of backpropagation in machine
        learning or neural network frameworks.

        :return: The gradient of the loss function with shape and type matching the
                 predictions.
        :rtype: np.ndarray
        """
        return -self.target / (self.pred * self.pred.shape[0])
