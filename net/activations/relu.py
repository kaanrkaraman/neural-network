import numpy as np

from net.activations._base import Activation


class ReLU(Activation):
    """
    ReLU activation function.

    This class implements the ReLU (Rectified Linear Unit) activation function, a
    commonly used activation function in neural networks. It provides methods for
    performing the forward and backward passes, as well as a placeholder for
    parameter updates, though ReLU itself has no trainable parameters.

    :ivar input: Input tensor cached during the forward pass for use in the backward
        pass. Set to None if no forward pass has been conducted.
    :type input: np.ndarray | None
    """

    def __init__(self):
        self.input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the ReLU (Rectified Linear Unit) activation function to the input
        array. This function sets all negative values in the input array to zero
        and retains positive values as is.

        :param x: The input array for which the ReLU activation will be applied.
                   The input array should be a NumPy array with numerical values.
        :return: A NumPy array where all negative values from the input array
                 are replaced with zero, while positive values remain unchanged.
        """
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the loss with respect to the input tensor using the chain rule
        and the derivative of the ReLU activation function.

        :param grad_output: A numpy array representing the gradient of the loss with respect to
            the output of the ReLU activation function.
        :type grad_output: np.ndarray
        :return: A numpy array representing the gradient of the loss with respect to the
            input of the ReLU activation function.
        :rtype: np.ndarray
        :raises ValueError: If the method is called before forward pass was executed and thus
            the input is not available.
        """
        if self.input is None:
            raise ValueError("Cannot call backward before forward.")

        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input

    def update(self, learning_rate: float) -> None:
        # ReLU has no parameters to update
        pass
