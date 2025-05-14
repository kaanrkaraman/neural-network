import numpy as np

from net.optimizers._base import Optimizer


class SGDMomentum(Optimizer):
    """
    Implements the SGD (Stochastic Gradient Descent) optimizer with momentum.

    This class extends the general Optimizer class to include momentum-based
    gradient descent, improving optimization by smoothing parameter updates.
    Momentum helps accelerate gradients and prevents oscillations by combining
    fractional past updates with current gradient updates. This implementation
    requires initialization of parameters and supports resetting gradients
    to None after updates.

    :ivar lr: Learning rate for gradient updates.
    :type lr: float
    :ivar momentum: Momentum factor, controlling the contribution of past updates.
    :type momentum: float
    :ivar velocities: List of velocity buffers for each parameter, used to store
        momentum updates.
    :type velocities: list
    """

    def __init__(self, params, lr=1e-3, momentum=0.9):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.value) for p in self.params]

    def step(self):
        """
        Updates the parameters using momentum-based gradient descent.

        This method iterates over all parameters in the `params` list. For each parameter
        with non-None gradients, it updates the parameter value based on the momentum and
        learning rate. Momentum is used to smooth updates by combining a fraction of
        the previous updates (scaled by `self.momentum`) with the current gradient's
        scaled value (scaled by `self.lr`). Parameters are modified in place.

        :raises AttributeError: If `self.params` or `self.velocities` are not initialized
            or improperly structured.
        :raises ValueError: If the size of `self.params` and `self.velocities` differ.

        :return: None
        """
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            self.velocities[i] = (
                self.momentum * self.velocities[i] - self.lr * param.grad
            )
            param.value += self.velocities[i]

    def zero_grad(self):
        for param in self.params:
            param.grad = None
