from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """
    An abstract base class defining the core structure and operations of a model.

    This class serves as a blueprint for deriving specific model implementations, particularly
    useful in machine learning and related applications. Subclasses are required to implement
    all abstract methods, which include forward and backward computations, parameter updates,
    prediction, and saving/loading of the model's state.
    """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        An abstract method defining the forward computation.

        This method is intended to be overridden in subclasses to define
        specific operations on the input data. It accepts an input array
        and typically returns a transformed array of the same or modified
        shape, depending on the operation being implemented.

        :param x: Input array on which the forward operation will be
            computed.
        :type x: numpy.ndarray

        :return: The resulting array after performing the forward
            operation.
        :rtype: numpy.ndarray
        """
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the output with respect to the input during backpropagation.

        The method must be implemented in a subclass and is responsible for defining how
        to compute the gradients for a specific operation. This operation is fundamental
        for enabling automatic differentiation in machine learning models.

        :param grad_output: The gradient of the loss with respect to the output of the layer.
        :type grad_output: np.ndarray
        :return: The gradient of the loss with respect to the input of the layer.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def update(self, learning_rate: float) -> None:
        """
        Abstract method for updating internal parameters or performing necessary updates
        based on the given learning rate. Implementations should provide the specific
        logic for how the update is handled.

        :param learning_rate: The learning rate to guide the update. Must be a floating-point
            value defining the step size in optimization processes.
        :return: None
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        An abstract method that makes predictions based on input data. This method must
        be implemented by any subclass to define the specific prediction logic.

        :param x: An array of input data represented as a NumPy ndarray. The structure
            and expected formatting of the data will depend on the specific implementation
            of this method in a subclass.
        :return: A NumPy ndarray containing the predicted values or results corresponding
            to the input data.
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """
        An abstract base class method intended to be overridden in subclasses for implementing
        custom training logic. This method must be implemented by any concrete subclass to
        enable the execution of specific training-related processes. The implementation defines
        how the training operation is executed and what resources or techniques it requires.

        :return: None
        """
        pass

    @abstractmethod
    def eval(self):
        """
        An interface for defining the structure of classes that must
        implement an eval method. Classes inheriting from this interface
        are required to provide an implementation of the eval method.

        Classes using this interface are expected to handle their
        own specific logic inside the eval method.

        :return: None
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Saves the current state or data to the specified file at the given file path.
        This method should be implemented in subclasses and is expected to handle
        all necessary actions to persist the data into the file system.

        :param filepath: The path to the file where the data should be saved.
        :type filepath: str
        :return: None
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Loads the state or data from the specified file at the given file path.
        This method should be implemented in subclasses and is expected to handle
        all necessary actions to retrieve the data from the file system.

        :param filepath: The path to the file from which the data should be loaded.
        :type filepath: str
        :return: None
        """
        pass
