from abc import ABC, abstractmethod

import numpy as np


class Parameter:
    def __init__(self, value: np.ndarray):
        self.value = value
        self.grad = None


class Layer(ABC):
    """
    Defines an abstract base class for layers to be used in neural networks or other
    similar computational frameworks.

    This class enforces implementation of key methods such as `forward`, `backward`,
    `update`, and others, providing a structured blueprint for building layers of
    varied functionalities. Each subclass must implement the abstract methods defined
    here to define its specific behavior. This design enables scalability, modularity,
    and adaptability for different applications, such as training machine learning models
    or constructing multi-layered architectures.

    :ivar input: The input data to the layer, typically represented as a NumPy array.
    :ivar output: The output data from the layer, computed based on the input and the layer's logic.
    """

    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Represents an abstract method for implementing forward operations
        in a class. The `forward` method takes an input array and returns
        an output array after performing specific operations, which must
        be defined in a subclass.

        This method is designed to serve as a blueprint for derived classes
        and enforces implementation of forward computations. Any class
        inheriting from this abstract base class must implement the
        `forward` method.

        :param x: Input array that the forward operation will process.
                  The type of operations depends on the specific context
                  defined in a subclass.
        :type x: np.ndarray
        :return: Output array resulting from the forward operation performed
                 on the input array.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the loss with respect to the input for the given layer
        during the backpropagation step in a neural network. This method should be
        implemented in subclasses to define the specific backpropagation behavior
        for the layer. It receives the gradient of the loss with respect to the output
        of the layer and computes the corresponding gradient with respect to the input.

        :param grad_output: Gradient of the loss with respect to the output of the layer.
            Shape must match the output shape of the layer.
        :type grad_output: np.ndarray
        :return: Gradient of the loss with respect to the input of the layer.
            Shape must match the input shape of the layer.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def update(self, learning_rate: float) -> None:
        """
        Updates the internal model parameters based on the learning rate.

        This method is expected to be implemented by subclasses to provide
        specific updating logic. The update process modifies the internal state
        of the model to improve its performance by adjusting parameters in
        accordance with the provided learning rate.

        :param learning_rate: The step size used to adjust model parameters.
        :type learning_rate: float
        :return: None
        :rtype: None
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """
        Abstract method for training a model on provided input data and labels. This method
        must be implemented by subclasses to define the specific steps required to train
        a machine learning model or similar. It modifies the state of the object to adapt
        based on the given data.

        :return: None
        :rtype: None
        """
        pass

    @abstractmethod
    def eval(self) -> None:
        """
        Abstract method to evaluate a given set of inputs and compute a corresponding metric or result.

        This method should be implemented by subclasses, enabling them to define their specific
        logic for evaluating the inputs provided and returning a float value as the outcome. The
        evaluation typically considers both x and y inputs, where x could represent the input data
        or features and y could denote labels or target values.

        :return: A computed evaluation result as a float value.
        """
        pass
