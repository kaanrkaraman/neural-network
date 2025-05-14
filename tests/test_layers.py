import numpy as np
import pytest

from net.layers.dense import Dense
from net.layers.flatten import Flatten
from tests.conftest import numerical_gradient_check


@pytest.fixture
def input_tensor_3d():
    """
    Fixture for generating a random 3D tensor.

    This fixture creates and returns a randomly generated 3-dimensional
    tensor with shape (2, 3, 4). It can be used for testing purposes
    where a mock 3D input tensor is required.

    :return: A randomly generated 3D tensor with shape (2, 3, 4).
    :rtype: numpy.ndarray
    """
    return np.random.randn(2, 3, 4)


@pytest.fixture
def input_tensor_2d():
    """
    Fixture that provides a sample 2D NumPy array with random values.

    This fixture returns a 2D array of shape (2, 4) with random values from
    a normal distribution (mean 0, variance 1). It can be used in tests
    that require a random 2D tensor for computations or validations.

    :return: Random 2D NumPy array with dimensions (2, 4).
    :rtype: numpy.ndarray
    """
    return np.random.randn(2, 4)


def test_flatten_forward_shape(input_tensor_3d):
    """
    Validates the shape of the output from the `Flatten` operation's forward
    method. Ensures that the flattened output matches the expected shape, which
    is derived from the input tensor's batch size and the product of its other
    dimensions.

    :param input_tensor_3d: Input tensor with three dimensions, where the first
                            dimension represents the batch size.
    :type input_tensor_3d: numpy.ndarray
    :return: None
    """
    flatten = Flatten()
    output = flatten.forward(input_tensor_3d)
    batch_size = input_tensor_3d.shape[0]
    flattened_dim = np.prod(input_tensor_3d.shape[1:])
    assert output.shape == (
        batch_size,
        flattened_dim,
    ), f"Expected shape {(batch_size, flattened_dim)}, got {output.shape}"


def test_flatten_backward_shape_restoration(input_tensor_3d):
    """
    Tests the shape restoration functionality of the backward pass in the Flatten layer.

    This function ensures that during the backward operation of the Flatten layer, the shape
    of the gradient input is restored to match the shape of the input tensor provided to the
    layer during the forward pass. It also verifies that the restored shape is equivalent to
    the original input tensor's shape.

    :param input_tensor_3d: The input tensor with three dimensions, which is passed through
        the Flatten layer for testing backward shape restoration and gradient computation.
    :type input_tensor_3d: numpy.ndarray

    :return: None
    """
    flatten = Flatten()
    out = flatten.forward(input_tensor_3d)
    grad_output = np.ones_like(out)
    grad_input = flatten.backward(grad_output)
    assert grad_input.shape == input_tensor_3d.shape
    np.testing.assert_array_equal(grad_input.shape, input_tensor_3d.shape)


def test_flatten_backward_value_consistency(input_tensor_3d):
    """
    Test the value consistency of the Flatten layer's backward method. This function
    verifies that the backward operation of the Flatten layer maintains the values
    properly when the gradients are propagated back through the layer. It ensures
    that the shape of the computed gradients matches that of the input tensor and
    that the values of the gradients are preserved correctly, providing confidence
    in the reliability of the implementation.

    :param input_tensor_3d: A 3-dimensional tensor passed as input to the Flatten
        layer during the test
    :type input_tensor_3d: numpy.ndarray
    :return: None
    """
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
    """
    Tests the output shape of the Dense layer's forward method to ensure it
    produces the expected shape for the given input tensor. This test is
    specific to a Dense layer initialized with `in_features=4` and `out_features=3`.

    :param input_tensor_2d: A 2D tensor with shape (2, 4) representing the batch
        input for the Dense layer.
    :type input_tensor_2d: torch.Tensor
    :return: None
    :rtype: NoneType

    """
    dense = Dense(in_features=4, out_features=3)
    output = dense.forward(input_tensor_2d)
    assert output.shape == (2, 3), f"Expected shape (2, 3), got {output.shape}"


def test_dense_backward_output_shape(input_tensor_2d):
    """
    Tests if the backward pass of a `Dense` layer produces an input gradient with the
    same shape as the input tensor. It ensures that the `backward` method processes the
    gradient from the output of the Dense layer correctly and returns an input gradient
    that matches the original input tensor's shape.

    :param input_tensor_2d: A 2D tensor representing the input data. Its dimensions are
                            used for validation in the test.
    :type input_tensor_2d: np.ndarray
    :return: None
    """
    dense = Dense(in_features=4, out_features=3)
    output = dense.forward(input_tensor_2d)
    grad_output = np.ones_like(output)
    grad_input = dense.backward(grad_output)
    assert grad_input.shape == input_tensor_2d.shape


def test_dense_forward_known_output():
    """
    Tests the forward function of the Dense layer for correctness by comparing
    its output to a known expected value. The test sets the weights, biases,
    and input manually, then computes the output and validates it against
    the expected output using numpy testing utilities.

    :raises AssertionError: Raised if the computed output does not match the
        expected output within the given tolerance.
    """
    dense = Dense(in_features=2, out_features=2)
    dense.W = np.array([[1.0, 2.0], [3.0, 4.0]])
    dense.b = np.array([[0.5, -0.5]])
    x = np.array([[1.0, 2.0]])
    expected = np.array(
        [[1.0 * 1.0 + 2.0 * 3.0 + 0.5, 1.0 * 2.0 + 2.0 * 4.0 - 0.5]]
    )  # shape (1,2)
    output = dense.forward(x)
    np.testing.assert_allclose(output, expected, rtol=1e-6)


def test_dense_gradient_check(input_tensor_2d):
    """
    Performs numerical gradient check on a Dense layer to ensure that the computed
    gradients are correct by comparing them with numerically approximated gradients.
    Gradient checking is a useful troubleshooting tool for debugging neural network
    implementations.

    :param input_tensor_2d: Input tensor to be passed through the Dense layer for
        gradient computation
    :type input_tensor_2d: torch.Tensor

    :return: None
    """
    dense = Dense(in_features=4, out_features=3)
    numerical_gradient_check(dense, input_tensor_2d)
