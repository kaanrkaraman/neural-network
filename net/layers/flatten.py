from __future__ import annotations

import numpy as np

from net.layers._base import Layer


class Flatten(Layer):
    """
    Flattens the input tensor while preserving the batch size dimension.

    The purpose of this class is to convert multi-dimensional input tensors
    into a 2D tensor, which is often required before feeding the data into
    fully connected layers in neural networks. The class also supports
    backpropagation where the flattened tensor is reshaped to its original
    dimensions.

    :ivar original_shape: Stores the original shape of the tensor before flattening,
        to enable reshaping during the backward pass.
    :type original_shape: tuple[int, ...] | None
    """

    def __init__(self):
        super().__init__()
        self.original_shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.original_shape = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.original_shape is None:
            raise ValueError("Cannot call backward before forward.")
        return grad_output.reshape(self.original_shape)

    def update(self, learning_rate: float) -> None:
        # Flatten has no parameters to update
        pass
