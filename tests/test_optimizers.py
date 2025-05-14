import numpy as np
import pytest

from net.layers import Parameter
from net.optimizers.adam import Adam
from net.optimizers.momentum import SGDMomentum
from net.optimizers.sgd import SGD


def quadratic_loss(param):
    """
    Calculates the quadratic loss for a given parameter and updates its gradient.

    The quadratic loss is calculated as the square of the difference between
    the parameter's value and a constant, here 3.0. The gradient of the loss with
    respect to the parameter is also computed and stored in the parameter's
    `grad` attribute.

    :param param: An object that has `value` and `grad` attributes.
        The `value` attribute should hold the numerical value of the parameter,
        and the `grad` attribute will be updated with the computed gradient.
    :type param: Any object with `value` (float) and `grad` (float) attributes
    :return: The quadratic loss as a float.
    :rtype: float
    """
    loss = (param.value - 3.0) ** 2
    param.grad = 2 * (param.value - 3.0)
    return loss


@pytest.mark.parametrize(
    "optimizer_class,kwargs",
    [
        (SGD, {"lr": 0.1}),
        (SGDMomentum, {"lr": 0.1, "momentum": 0.9}),
        (Adam, {"lr": 0.05}),
    ],
)
def test_optimizer_convergence(optimizer_class, kwargs):
    """
    Test the convergence of various optimizer classes on a quadratic loss function.

    This test ensures that the specified optimizer classes can minimize a
    quadratic loss function and converge to the expected value of `x = 3.0`
    within a defined tolerance.

    :param optimizer_class: The optimizer class to be tested.
    :type optimizer_class: type
    :param kwargs: Dictionary of keyword arguments to configure the optimizer.
    :type kwargs: dict
    :return: None
    """
    param = Parameter(np.array(0.0))
    optimizer = optimizer_class([param], **kwargs)

    losses = []
    for _ in range(100):
        loss = quadratic_loss(param)
        losses.append(loss)
        optimizer.step()
        optimizer.zero_grad()

    final_x = param.value
    assert np.isclose(
        final_x, 3.0, atol=0.1
    ), f"{optimizer_class.__name__} failed to converge. Final x: {final_x}"
