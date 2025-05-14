import numpy as np


def numerical_gradient_check(layer, x: np.ndarray, eps: float = 1e-5):
    """
    Performs a numerical gradient check for a given layer's weight and bias parameters.

    This function is used to verify the correctness of the gradients computed by a layer's
    `backward` method by comparing them with numerically approximated gradients. It perturbs
    each parameter's value slightly in both positive and negative directions, computes the
    resulting change in the output using the `forward` method, and calculates an approximate
    gradient using finite differences. These approximate gradients are then compared with the
    analytical gradients `dW` and `db` provided by the layer after the backward pass.

    :param layer: The layer object for which the gradient check is being performed. The layer
        should have attributes `W` (weights), `b` (biases), and provide `forward` and
        `backward` methods. The gradients computed by the layer must be available in `dW`
        and `db` attributes respectively.
    :type layer: Any
    :param x: Input data to the layer used for testing. It is passed through the layer's
        `forward` and `backward` methods during gradient checking.
    :param eps: Small perturbation used for numerical differentiation to approximate
        gradients. Defaults to 1e-5.
    :return: None. This function either completes successfully or raises an assertion
        error if any of the gradient checks fail.
    """
    grad_output = np.ones_like(layer.forward(x))
    layer.forward(x)
    layer.backward(grad_output)

    # Check dW
    for i in range(layer.W.shape[0]):
        for j in range(layer.W.shape[1]):
            original = layer.W[i, j]
            layer.W[i, j] = original + eps
            plus = layer.forward(x).sum()
            layer.W[i, j] = original - eps
            minus = layer.forward(x).sum()
            layer.W[i, j] = original  # restore

            numerical = (plus - minus) / (2 * eps)
            analytic = layer.dW[i, j]
            np.testing.assert_allclose(
                analytic,
                numerical,
                rtol=1e-4,
                atol=1e-6,
                err_msg=f"Gradient check failed at W[{i},{j}]",
            )

    # Check db
    for j in range(layer.b.shape[1]):
        original = layer.b[0, j]
        layer.b[0, j] = original + eps
        plus = layer.forward(x).sum()
        layer.b[0, j] = original - eps
        minus = layer.forward(x).sum()
        layer.b[0, j] = original  # restore

        numerical = (plus - minus) / (2 * eps)
        analytic = layer.db[0, j]
        np.testing.assert_allclose(
            analytic,
            numerical,
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"Gradient check failed at b[{j}]",
        )
