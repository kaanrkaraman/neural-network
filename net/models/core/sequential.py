import pickle

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

    def __init__(self, layers: list[Layer]) -> None:
        """

        :rtype: None
        """
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
        Generates predictions from model outputs.

        - For binary classification (output shape [N, 1]), returns 0 or 1 using a 0.5 threshold.
        - For multiclass classification (output shape [N, C]), returns class indices via argmax.

        :param x: Input data as a NumPy array.
        :return: Predicted class labels (0/1 for binary, class indices for multiclass).
        """
        output = self.forward(x)

        if output.shape[1] == 1:
            return (output > 0.5).astype(int)

        return np.argmax(output, axis=1)

    def train(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()

    def eval(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()

    def save(self, filepath: str) -> None:
        """
        Saves the model to the specified file path, including architecture and parameters.
        """
        save_data: dict[str, list] = {
            "layer_types": [layer.__class__.__name__ for layer in self.layers],
            "layer_params": [],
        }

        for layer in self.layers:
            params = {}
            for attr in ["W", "b"]:  # Adjust as needed for more complex layers
                if hasattr(layer, attr):
                    params[attr] = getattr(layer, attr)
            save_data["layer_params"].append(params)

        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)

    def load(self, filepath: str) -> None:
        """
        Loads the model parameters from the specified file.
        Note: Assumes same model structure is used during reloading.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        for layer, params in zip(self.layers, data["layer_params"]):
            for name, value in params.items():
                if hasattr(layer, name):
                    setattr(layer, name, value)
