import numpy as np
import pytest

from net.layers.dense import Dense
from net.layers.flatten import Flatten

from tests.conftest import numerical_gradient_check


@pytest.fixture
def input_tensor_3d():
    """Simulated 3D input for Flatten: shape (batch=2, channels=3, width=4)"""
    return np.random.randn(2, 3, 4)


@pytest.fixture
def input_tensor_2d():
    """2D input for Dense layer: shape (batch=2, features=4)"""
    return np.random.randn(2, 4)

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


def test_dense_forward_known_output():
    dense = Dense(in_features=2, out_features=2)
    dense.W = np.array([[1.0, 2.0], [3.0, 4.0]])
    dense.b = np.array([[0.5, -0.5]])
    x = np.array([[1.0, 2.0]])
    expected = np.array([[1.0*1.0 + 2.0*3.0 + 0.5, 1.0*2.0 + 2.0*4.0 - 0.5]])  # shape (1,2)
    output = dense.forward(x)
    np.testing.assert_allclose(output, expected, rtol=1e-6)


def test_dense_gradient_check(input_tensor_2d):
    dense = Dense(in_features=4, out_features=3)
    numerical_gradient_check(dense, input_tensor_2d)
