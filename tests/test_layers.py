import numpy as np
import pytest

from net.layers.dense import Dense
from net.layers.flatten import Flatten


@pytest.fixture
def input_tensor_3d():
    """Simulated 3D input for Flatten: shape (batch=2, channels=3, width=4)"""
    return np.random.randn(2, 3, 4)


@pytest.fixture
def input_tensor_2d():
    """2D input for Dense layer: shape (batch=2, features=4)"""
    return np.random.randn(2, 4)


def numerical_gradient_check(layer, x: np.ndarray, eps: float = 1e-5):
    """
    Perform gradient checking on a layer by comparing its analytical gradients
    with numerically approximated gradients.

    Assumes the layer has been forward-called with x.
    """
    grad_output = np.ones_like(layer.forward(x))
    layer.forward(x)
    layer.backward(grad_output)

    # Check dW
    for i in range(layer.W.shape[0]):
        for j in range(layer.W.shape[1]):
            W_orig = layer.W[i, j]
            layer.W[i, j] = W_orig + eps
            plus = layer.forward(x).sum()
            layer.W[i, j] = W_orig - eps
            minus = layer.forward(x).sum()
            layer.W[i, j] = W_orig

            numerical_grad = (plus - minus) / (2 * eps)
            np.testing.assert_allclose(
                layer.dW[i, j],
                numerical_grad,
                rtol=1e-4,
                atol=1e-6,
                err_msg=f"Gradient check failed for W[{i},{j}]",
            )

    # Check db
    for j in range(layer.b.shape[1]):
        b_orig = layer.b[0, j]
        layer.b[0, j] = b_orig + eps
        plus = layer.forward(x).sum()
        layer.b[0, j] = b_orig - eps
        minus = layer.forward(x).sum()
        layer.b[0, j] = b_orig

        numerical_grad = (plus - minus) / (2 * eps)
        np.testing.assert_allclose(
            layer.db[0, j],
            numerical_grad,
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"Gradient check failed for b[{j}]",
        )


def test_flatten_forward_shape(input_tensor_3d):
    flatten = Flatten()
    output = flatten.forward(input_tensor_3d)
    batch_size = input_tensor_3d.shape[0]
    flattened_dim = np.prod(input_tensor_3d.shape[1:])
    assert output.shape == (
        batch_size,
        flattened_dim,
    ), f"Expected shape {(batch_size, flattened_dim)}, got {output.shape}"


def test_flatten_backward_shape_restoration(input_tensor_3d):
    flatten = Flatten()
    out = flatten.forward(input_tensor_3d)
    grad_output = np.ones_like(out)
    grad_input = flatten.backward(grad_output)
    assert grad_input.shape == input_tensor_3d.shape
    np.testing.assert_array_equal(grad_input.shape, input_tensor_3d.shape)


def test_flatten_backward_value_consistency(input_tensor_3d):
    flatten = Flatten()
    out = flatten.forward(input_tensor_3d)
    grad_output = np.random.randn(*out.shape)
    grad_input = flatten.backward(grad_output)
    assert grad_input.shape == input_tensor_3d.shape
    np.testing.assert_array_almost_equal(
        grad_output.flatten(),
        grad_input.flatten(),
        err_msg="Flatten backward did not preserve values correctly",
    )


def test_dense_forward_output_shape(input_tensor_2d):
    dense = Dense(in_features=4, out_features=3)
    output = dense.forward(input_tensor_2d)
    assert output.shape == (2, 3), f"Expected shape (2, 3), got {output.shape}"


def test_dense_backward_output_shape(input_tensor_2d):
    dense = Dense(in_features=4, out_features=3)
    output = dense.forward(input_tensor_2d)
    grad_output = np.ones_like(output)
    grad_input = dense.backward(grad_output)
    assert grad_input.shape == input_tensor_2d.shape


def test_dense_gradient_check(input_tensor_2d):
    dense = Dense(in_features=4, out_features=3)
    numerical_gradient_check(dense, input_tensor_2d)
