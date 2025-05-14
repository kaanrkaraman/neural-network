import numpy as np

from net.optimizers._base import Optimizer


class Adam(Optimizer):
    """
    Implements the Adam optimization algorithm.

    The Adam optimizer is an adaptive learning rate optimization algorithm
    that combines the advantages of two other extensions of stochastic
    gradient descent, namely AdaGrad and RMSProp. It computes individual
    adaptive learning rates for different parameters from estimates of
    first and second moments of the gradients. The algorithm is particularly
    effective for training deep neural networks.

    :ivar lr: Learning rate for the optimizer (alpha).
    :type lr: float
    :ivar beta1: Exponential decay rate for the first moment estimates.
    :type beta1: float
    :ivar beta2: Exponential decay rate for the second moment estimates.
    :type beta2: float
    :ivar eps: A small value to prevent division by zero during updates.
    :type eps: float
    :ivar t: Time step maintaining iteration count for decay rates.
    :type t: int
    :ivar m: List of first moment estimates for each parameter.
    :type m: list
    :ivar v: List of second moment estimates for each parameter.
    :type v: list
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = [np.zeros_like(p.value) for p in self.params]
        self.v = [np.zeros_like(p.value) for p in self.params]

    def step(self):
        """
        Performs a single optimization step using the Adam optimization algorithm.

        The method implements the Adam algorithm to update the parameters of a model
        based on their gradients. It maintains moving averages of the gradients (`m`)
        and the squared gradients (`v`) for each parameter, which are then bias-corrected
        to compute the update step. The learning rate, as well as algorithm-specific
        hyperparameters beta1, beta2 (exponential decay rates for the moment estimates),
        and epsilon (for numerical stability), influence the update steps.

        :raises AttributeError: If any parameter in self.params doesn't have the
            required attributes (e.g., `grad`, `value`).
        :raises ZeroDivisionError: If the bias correction results in division by zero.

        :param self: Instance of the class invoking this method, where:
            - `self.t` is an integer representing the time step (iteration count).
            - `self.params` is a collection of parameter objects with attributes:
                - `grad`: The gradient of the parameter.
                - `value`: The numerical value of the parameter.
            - `self.m` and `self.v` are lists representing the moment vectors for the
              parameters (moving averages of gradients and squared gradients,
              respectively).
            - `self.beta1` and `self.beta2` are scalars that control the decay rates
              for the moment estimates.
            - `self.lr` is the learning rate (scalar controlling the size of parameter
              updates).
            - `self.eps` is a small scalar added for numerical stability when dividing
              by the square root of `v_hat`.

        :return: None
        """
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad**2)

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            param.value -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.params:
            param.grad = None
