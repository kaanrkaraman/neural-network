import numpy as np

from net.activations._base import Activation


class Tanh(Activation):
    """
    Applies the hyperbolic tangent (tanh) activation function.

    The Tanh class represents the tanh activation commonly used in neural networks
    to introduce non-linearity. It computes the tanh function on the input tensor
    during the forward pass, and calculates the gradient during the backward pass.
    This class has no trainable parameters to update.

    :ivar output: Stores the output of the tanh activation after forward pass.
        This is used in"""

    def __init__(self):
        self.output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the hyperbolic tangent activation for input data.

        The function applies the hyperbolic tangent (tanh) activation function to the
        input array. It simultaneously updates the instance's internal state (`self.output`)
        with the result of the computation and returns the computed value. The tanh
        activation function maps input values to a range between -1 and 1.

        :param x: Input array for the tanh calculation. Expected to be a NumPy array
            of any shape compatible with the method's requirements.
        :type x: np.ndarray

        :return: The result of applying the tanh activation function on the input array.
        :rtype: np.ndarray
        """
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the loss function with respect to the input of a layer based
        on the gradient of the output (grad_output) and the stored forward pass result (output).
        This method performs the backward pass for layers applying the hyperbolic tangent
        (tanh) activation function. The derivative of the tanh function is used to scale
        the `grad_output` appropriately.

        :param grad_output: Gradient of the loss with respect to the output of this layer.
            The input is expected to be a NumPy array of the same dimensionality as the
            output from the forward pass.
        :return: Gradient of the loss with respect to the input of this layer after applying
            the tanh derivative, scaled by the forward pass result.
        :rtype: np.ndarray
        :raises ValueError: If the method is called before the forward pass, indicated by
            the lack of stored output data (``self.output`` is ``None``).
        """
        if self.output is None:
            raise ValueError("Cannot call backward before forward.")
        return grad_output * (1 - self.output**2)

    def update(self, learning_rate: float) -> None:
        # Tanh has no parameters to update
        pass
