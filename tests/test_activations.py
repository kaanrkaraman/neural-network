import numpy as np
import pytest

from net.activations.relu import ReLU
from net.activations.sigmoid import Sigmoid
from net.activations.tanh import Tanh


@pytest.fixture
def test_tensor():
    """
    Fixture that provides a test tensor as a NumPy array.
    This fixture is commonly used in unit tests that require a sample tensor
    with predefined values for validation purposes.

    The tensor is structured as a 1x3 array containing the following values:
    1.0, -1.0, and 0.0. It can be utilized in tests targeting numerical computations,
    machine learning functionalities, or related operations.

    :yield: NumPy array containing a test tensor with predefined values.
    :rtype: numpy.ndarray
    """
    return np.array([[1.0, -1.0, 0.0]])


def numerical_gradient_check(activation, x: np.ndarray, eps: float = 1e-5):
    """
    Performs a numerical gradient check for a given activation function by comparing
    the analytical gradient computed via the `backward` method of the activation
    function against the numerical gradient obtained via finite difference approximation.
    The function validates the gradients within a defined relative and absolute tolerance.

    :param activation: An instance of an activation function implementing `forward`
        and `backward` methods. The `forward` method computes the output given
        the input, and `backward` computes the gradient of the loss with respect to the input.
    :param x: A 2D NumPy array representing the input tensor for which the numerical
        gradient check will be conducted.
    :param eps: A small float value used for finite difference approximation of the
        gradients. Defaults to 1e-5.
    :return: None. Asserts equivalence between numerical and analytical gradients
        with specified tolerance.
    """
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
    """
    Test the output shape of the ReLU activation function's forward pass to ensure it
    matches the input shape.

    :raises AssertionError: If the output shape does not match the input shape.
    """
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
    """
    Tests the forward pass of various activation functions against expected output values.

    :param activation_cls: The class of the activation function to be tested.
    :param input_val: The input array passed to the forward method of the activation function.
    :param expected_output: The expected output array from the forward method of
        the activation function when provided with the given input_val.
    :return: None
    """
    act = activation_cls()
    output = act.forward(input_val)
    np.testing.assert_almost_equal(output, expected_output, decimal=6)


def test_relu_backward_correctness(test_tensor):
    """
    Tests the correctness of the ReLU backward computation. The function applies
    the ReLU forward pass to the input tensor, computes the gradient during the
    backward pass, and verifies that the computed gradient matches the expected
    values.

    :param test_tensor: Input tensor for testing ReLU backward correctness.
        This tensor is passed through the ReLU operation, and its gradient
        with respect to the backward pass is calculated.
    :type test_tensor: numpy.ndarray

    :return: None. Asserts the correctness of the backward operation by
        comparing the computed gradient with the expected values.
    """
    relu = ReLU()
    grad_output = np.ones_like(test_tensor)
    relu.forward(test_tensor)
    grad_input = relu.backward(grad_output)

    expected = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(grad_input, expected)


def test_relu_gradient_check():
    """
    Performs a gradient check on the ReLU function.

    This function creates an instance of the ReLU class, generates a random input
    array, and uses the `numerical_gradient_check` function to verify the gradient
    calculations of the ReLU operation.

    :return: None
    """
    relu = ReLU()
    x = np.random.randn(2, 3)
    numerical_gradient_check(relu, x)


def test_sigmoid_gradient_check():
    """
    Performs the gradient check for the Sigmoid activation function.

    This function creates a Sigmoid activation function instance and uses random
    inputs to perform a numerical gradient check. The gradient check ensures that
    the analytical gradient of the Sigmoid function matches its numerical gradient.

    :raises ValueError: If the numerical gradient check fails for the Sigmoid
        activation function.
    """
    sigmoid = Sigmoid()
    x = np.random.randn(2, 2)
    numerical_gradient_check(sigmoid, x)


def test_tanh_gradient_check():
    """
    Tests the gradient computation for the Tanh activation function using
    numerical gradient checking. Random input data is generated, and the
    computed gradient of the Tanh function is compared with the numerical
    approximation to ensure correctness of the implementation.

    :return: None
    """
    tanh = Tanh()
    x = np.random.randn(2, 2)
    numerical_gradient_check(tanh, x)
