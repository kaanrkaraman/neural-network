import numpy as np
import pytest


def numerical_gradient_check(layer, x: np.ndarray, eps: float = 1e-5):
    """
    Perform gradient checking on a layer with parameters (W, b).
    This assumes:
      - forward(x): produces scalar output (or sum is used)
      - backward(grad_output): computes gradients
      - layer.dW, layer.db exist after backward
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
                analytic, numerical, rtol=1e-4, atol=1e-6,
                err_msg=f"Gradient check failed at W[{i},{j}]"
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
            analytic, numerical, rtol=1e-4, atol=1e-6,
            err_msg=f"Gradient check failed at b[{j}]"
        )