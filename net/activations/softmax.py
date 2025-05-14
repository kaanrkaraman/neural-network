import numpy as np
from net.activations._base import Activation


class Softmax(Activation):
    """
    Represents the softmax activation function.

    This class provides a softmax activation layer, commonly used in neural networks
    for converting logits to probabilities. The implementation is batch-wise and
    supports both forward and backward pass computations. The `forward` method computes
    the softmax output, while the `backward` method calculates the gradient of the
    loss with respect to the input logits. This class does not maintain trainable
    parameters and provides a placeholder `update` method for compatibility purposes.

    :ivar output: Stores the result of the forward pass, representing probabilities
        computed from input logits. Shape: (batch_size, num_classes) or None if
        forward has not been called yet.
    :type output: np.ndarray | None
    """

    def __init__(self):
        self.output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the softmax activation function. This method takes
        an input array, performs a numerically stable transformation by shifting the
        input values to avoid potential overflow/underflow issues, and calculates the
        softmax probabilities for each input element. The resulting probabilities are
        stored as the output attribute and returned.

        :param x: Input array for which the softmax probabilities are to be computed.
                  It is expected to be a 2D NumPy array where each row represents a
                  separate sample or data point.
        :type x: np.ndarray
        :return: A 2D NumPy array of the same shape as the input array, containing the
                 softmax probabilities for each input value. Each row sums to 1.
        :rtype: np.ndarray
        """
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the loss with respect to the input of the layer
        using the backward pass. This function assumes that the forward pass has
        already been executed and the output probabilities (softmax) are stored.

        :param grad_output: Gradient of the loss with respect to the output of this
            layer. It is a numpy array of shape (batch_size, num_classes).
        :return: Gradient of the loss with respect to the input of this layer. It
            is a numpy array of the same shape as `grad_output`, i.e.,
            (batch_size, num_classes).
        """
        if self.output is None:
            raise ValueError("Cannot call backward before forward.")

        batch_size, num_classes = grad_output.shape
        dx = np.zeros_like(grad_output)

        for i in range(batch_size):
            s = self.output[i].reshape(-1, 1)  # shape (C, 1)
            jacobian = np.diagflat(s) - s @ s.T  # shape (C, C)
            dx[i] = jacobian @ grad_output[i]

        return dx

    def update(self, learning_rate: float) -> None:
        # Softmax has no parameters to update
        pass