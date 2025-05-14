from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Base class for all optimizers.
    """

    def __init__(self, params):
        """
        Initialize the optimizer with parameters.

        Args:
            params (iterable): Parameters to optimize.
        """
        self.params = list(params)

    @abstractmethod
    def step(self):
        """
        Perform a single optimization step.
        """
        pass

    @abstractmethod
    def zero_grad(self):
        """
        Zero the gradients of the parameters.
        """
        for param in self.params:
            if hasattr(param, "grad"):
                param.grad = None
