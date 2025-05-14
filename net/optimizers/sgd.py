from net.optimizers._base import Optimizer


class SGD(Optimizer):
    """
    Implements stochastic gradient descent (SGD) optimization algorithm.

    This optimizer adjusts the parameters of a model by calculating
    gradients and updating the parameters based on those gradients
    scaled by the learning rate. It is widely used in machine learning
    and deep learning for optimizing model performance.

    :ivar params: Parameters of the model to optimize.
    :type params: Iterable
    :ivar lr: Learning rate used for adjusting the model parameters.
    :type lr: float
    """

    def __init__(self, params, lr=1e-03):
        super().__init__(params)
        self.lr = lr

    def step(self):
        """
        Updates parameters using their gradient and a specified learning rate.

        This method iterates over all parameters provided. For parameters that
        have a non-None gradient, it updates their value by subtracting the
        product of the learning rate and their gradient. Parameters with a
        None gradient are skipped in the update process.

        :return: None
        """
        for param in self.params:
            if param.grad is None:
                continue
            param.value -= self.lr * param.grad

    def zero_grad(self):
        """
        Sets the gradients of all parameters to None.
        """
        for param in self.params:
            if hasattr(param, "grad"):
                param.grad = None
