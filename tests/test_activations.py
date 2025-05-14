import numpy as np
import pytest

from net.activations.relu import ReLU
from net.activations.sigmoid import Sigmoid
from net.activations.tanh import Tanh


@pytest.fixture
def test_tensor():
    return np.array([[1.0, -1.0, 0.0]])


def numerical_gradient_check(activation, x: np.ndarray, eps: float = 1e-5):
    grad_output = np.ones_like(x)
    activation.forward(x)
    analytic = activation.backward(grad_output)

    numerical = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_pos = x.copy()
            x_neg = x.copy()
            x_pos[i, j] += eps
            x_neg[i, j] -= eps

            out_pos = activation.forward(x_pos).sum()
            out_neg = activation.forward(x_neg).sum()
            numerical[i, j] = (out_pos - out_neg) / (2 * eps)

    np.testing.assert_allclose(analytic, numerical, rtol=1e-4, atol=1e-6)


def test_relu_forward_output_shape():
    relu = ReLU()
    x = np.random.randn(4, 5)
    output = relu.forward(x)
    assert output.shape == x.shape, "Output shape must match input shape"


@pytest.mark.parametrize(
    "activation_cls, input_val, expected_output",
    [
        (Sigmoid, np.array([[0.0]]), np.array([[0.5]])),
        (Tanh, np.array([[0.0]]), np.array([[0.0]])),
        (ReLU, np.array([[-1.0, 0.0, 2.0]]), np.array([[0.0, 0.0, 2.0]])),
    ],
)
def test_activation_forward_values(activation_cls, input_val, expected_output):
    act = activation_cls()
    output = act.forward(input_val)
    np.testing.assert_almost_equal(output, expected_output, decimal=6)


def test_relu_backward_correctness(test_tensor):
    relu = ReLU()
    grad_output = np.ones_like(test_tensor)
    relu.forward(test_tensor)
    grad_input = relu.backward(grad_output)

    expected = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(grad_input, expected)


def test_relu_gradient_check():
    relu = ReLU()
    x = np.random.randn(2, 3)
    numerical_gradient_check(relu, x)


def test_sigmoid_gradient_check():
    sigmoid = Sigmoid()
    x = np.random.randn(2, 2)
    numerical_gradient_check(sigmoid, x)


def test_tanh_gradient_check():
    tanh = Tanh()
    x = np.random.randn(2, 2)
    numerical_gradient_check(tanh, x)
