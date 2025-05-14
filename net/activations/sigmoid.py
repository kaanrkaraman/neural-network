import numpy as np

from net.activations._base import Activation


class Sigmoid(Activation):
    """
    Implements the Sigmoid activation function for use in neural networks.

    The Sigmoid activation function maps input values to a value between 0 and 1
    using the formula: 1 / (1 + exp(-x)). It is often used in neural networks to
    introduce non-linearity and is particularly common in binary classification tasks.
    The class provides methods for forward computation, backward propagation of
    gradients, and an empty update method as Sigmoid has no trainable parameters.

    :ivar output: The result of applying the sigmoid function during the forward pass.
    :type output: np.ndarray | None
    """

    def __init__(self):
        self.output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the sigmoid activation function.

        This function applies the sigmoid activation to the input array. The sigmoid
        function transforms the input values to outputs in the range (0, 1) and is
        commonly used in neural networks to introduce non-linearity.

        :param x: The input numpy array for the forward pass as a multidimensional
            array, containing the values to be transformed using the sigmoid function.
        :return: The transformed numpy array containing the sigmoid activation
            of the input array.
        """
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass of a sigmoid function. The method calculates the
        gradient of the loss with respect to the input of the sigmoid function.

        :param grad_output: Gradient of the loss with respect to the output of
            the sigmoid function.
        :type grad_output: np.ndarray
        :return: Gradient of the loss with respect to the input of the sigmoid
            function.
        :rtype: np.ndarray
        :raises ValueError: If backward is called before forward, indicating
            that the output is unavailable.
        """
        if self.output is None:
            raise ValueError("Cannot call backward before forward.")
        return grad_output * self.output * (1 - self.output)

    def update(self, learning_rate: float) -> None:
        # Sigmoid has no parameters to update
        pass
