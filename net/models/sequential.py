import numpy as np

from net.layers import Layer
from net.models._base import Model


class Sequential(Model):
    """
    Sequential model that chains multiple layers together.

    The Sequential class is designed for building and working with a sequence
    of layers that are applied in order. Each layer processes its input and
    produces output for the next layer, facilitating the construction of
    layered architectures such as neural networks. The class provides methods
    for forward and backward passes, weight updates, and prediction.

    :ivar layers: List of layers, each implementing `forward`, `backward`,
        and `update` methods.
    :type layers: list[Layer]
    """

    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the forward pass through a sequence of layers in the network.

        Iterates over each layer in the network's layers and sequentially applies
        the forward pass for each layer using its `forward` method. Returns the
        final computed output after all layers have processed the input.

        :param x: The input array to be processed through the forward pass of
                  the layers in the network.
        :type x: np.ndarray
        :return: The output array after passing through all layers of the network.
        :rtype: np.ndarray
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass through the layers of the model.

        This function takes the gradient of the output from the subsequent layer (or
        the loss function for the final output layer) and propagates it backward
        through the layers of the model in reverse order, computing the gradients
        for the input at each layer as defined by their respective backward
        methods.

        :param grad_output: Gradient of the output from the subsequent layer.
            This gradient is propagated backwards through the model. Typically,
            this is a numpy array with a shape matching the output of the last
            layer.
        :type grad_output: np.ndarray

        :return: Gradient of the input after backpropagating through all layers.
            The resulting value can be used for optimizing or analyzing the input
            to the model.
        :rtype: np.ndarray
        """
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def update(self, lr: float) -> None:
        """
        Updates the parameters of all layers in the model using the given learning rate.

        This method iterates over the list of layers in the model and updates
        their internal parameters based on the provided learning rate. It acts
        as a wrapper to invoke the update method of each individual layer.

        :param lr: Learning rate used to perform the update operation for
                   all layers in the model.
        :type lr: float

        :return: None
        """
        for layer in self.layers:
            layer.update(lr)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Compares the output of the forward function with a threshold value (0.5) to generate predictions
        indicating whether each input element meets the defined criteria.

        :param x: Input data as a numpy ndarray.
        :type x: np.ndarray
        :return: Predicted boolean values represented as a numpy ndarray.
        :rtype: np.ndarray
        """
        return self.forward(x) > 0.5
