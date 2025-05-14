import numpy as np

from net.layers import Layer


class Dropout(Layer):
    """
    Implements the Dropout layer for neural networks.

    The Dropout layer randomly sets a fraction of input units to 0 during training,
    which helps prevent overfitting. The dropout rate is the fraction of the input
    units to drop, and it is only applied during training.

    :ivar p: Fraction of the input units to drop (between 0 and 1).
    :type p: float
    :ivar training: Boolean indicating whether the layer is in training mode.
    :type training: bool
    """

    def __init__(self, p):
        super().__init__()
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        if not self.training or self.p == 0.0:
            return x
        if self.p >= 1.0:
            raise ValueError("Dropout probability must be < 1.0")

        self.mask = (np.random.rand(*x.shape) >= self.p)
        return x * self.mask / (1 - self.p)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.training:
            return grad_output * self.mask / (1 - self.p)
        else:
            return grad_output

    def update(self, learning_p: float) -> None:
        # Dropout layer does not have parameters to update
        pass

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
