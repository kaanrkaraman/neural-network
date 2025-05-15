import numpy as np

from net.layers._base import Layer, Parameter


class Dense(Layer):
    """
    Implements a fully connected (dense) layer in a neural network.

    This class represents a dense layer that connects input and output features
    using weights and biases. It performs linear transformations during
    the forward pass and computes gradients for weights and biases during
    the backward pass. The update method applies gradient descent for parameter
    updates.

    :ivar in_features: Number of input features to the layer.
    :type in_features: int
    :ivar out_features: Number of output features from the layer.
    :type out_features: int
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = Parameter(
            np.random.uniform(-limit, limit, (in_features, out_features))
        )
        self.b = Parameter(np.zeros((1, out_features)))

        self.dW = np.zeros_like(self.W.value)
        self.db = np.zeros_like(self.b.value)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x  # (batch_size, in_features)
        self.output = x @ self.W.value + self.b.value  # (batch_size, out_features)
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        grad_output: (batch_size, out_features)
        """
        # Gradient w.r.t. weights and biases
        self.dW = self.input.T @ grad_output  # (in_features, out_features)
        self.db = np.sum(grad_output, axis=0, keepdims=True)  # (1, out_features)

        self.W.grad = self.dW
        self.b.grad = self.db

        grad_input = grad_output @ self.W.value.T  # (batch_size, in_features)
        return grad_input

    def update(self, learning_rate: float) -> None:
        self.W.value -= learning_rate * self.dW
        self.b.value -= learning_rate * self.db

    def train(self) -> None:
        """
        This method is not applicable for the Dense layer as it does not
        perform training directly. Instead, it should be used in a model
        context where the forward and backward passes are handled.
        """
        pass

    def eval(self) -> None:
        """
        This method is not applicable for the Dense layer as it does not
        perform evaluation directly. Instead, it should be used in a model
        context where the forward and backward passes are handled.
        """
        pass
