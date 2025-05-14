import numpy as np
import pytest

from net.layers import Conv2D, Dense, Flatten, MaxPool2D, Parameter
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


@pytest.fixture
def conv_layer():
    """
    Creates and returns a 2D convolutional layer. The convolutional layer has the
    specified channel configuration, kernel size, stride, and padding. This is set
    up to facilitate consistent testing or injection into other dependent functions
    or classes as a fixture.

    :return: A 2D convolutional layer initialized with in_channels=1,
        out_channels=1, kernel_size=3, stride=1, and padding=1.
    :rtype: Conv2D
    """
    return Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)


@pytest.fixture
def input_tensor():
    """
    Creates a numpy array representing an input tensor for a test or fixture
    with specified shape and values generated from a standard normal
    distribution. This tensor is commonly used in testing scenarios and conforms
    to the expected shape of (batch_size=2, channels=1, height=5, width=5).

    :return: A 4-dimensional numpy array with random values sampled from a standard
             normal distribution. The shape of the array is (2, 1, 5, 5).
    :rtype: numpy.ndarray
    """
    # Shape: (batch_size=2, channels=1, height=5, width=5)
    return np.random.randn(2, 1, 5, 5)


@pytest.fixture
def input_tensor_pooling():
    # Shape: (batch_size=1, channels=1, height=4, width=4)
    return np.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]],
        dtype=np.float32,
    )


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
    dense.W = Parameter(np.array([[1.0, 2.0], [3.0, 4.0]]))
    dense.b = Parameter(np.array([[0.5, -0.5]]))
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


def test_conv2d_forward_output_shape(conv_layer, input_tensor):
    """
    Tests the forward pass output shape of a given convolutional layer with a given input tensor.

    This function ensures that the shape of the output tensor from the convolutional layer's
    forward operation matches the expected dimensions. It raises an assertion error if the
    output shape does not match the expected shape.

    :param conv_layer: The convolutional layer to be tested.
    :type conv_layer: any
    :param input_tensor: The input tensor to be passed through the convolutional layer's forward function.
    :type input_tensor: any
    :return: None
    """
    output = conv_layer.forward(input_tensor)
    assert output.shape == (
        2,
        1,
        5,
        5,
    ), f"Expected output shape (2, 1, 5, 5), got {output.shape}"


def test_conv2d_forward_numerical_result():
    """
    Tests the forward operation of a 2D convolutional layer (Conv2D) with a manually
    set kernel and bias against a numerical expected result. The test ensures that
    the convolutional operation is performed correctly when a single input image of
    size 3x3 with one channel is passed through the layer.

    This test verifies that the computed output matches the expected value within
    a specified decimal tolerance.

    :return: Asserts that the computed output from the Conv2D layer matches the
        expected tensor, within a tolerance value for decimal places.
    """
    # Input: batch=1, channels=1, 3x3 image
    x = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)

    conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)

    conv.W.value[:] = np.array([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]])
    conv.b.value[:] = 0

    output = conv.forward(x)
    expected = np.array([[[[-6.0]]]])
    np.testing.assert_almost_equal(output, expected, decimal=5)


def numerical_gradient(f, x, eps=1e-5):
    """
    Computes the numerical gradient of a given scalar-valued function with respect
    to its input. This function iteratively perturbs each element of the input
    array and estimates the derivative using the central difference formula.

    :param f: Function for which the gradient needs to be computed. The function
        must accept a single parameter, typically a numeric array, and return a
        scalar value.
    :type f: Callable[[np.ndarray], float]
    :param x: Input array for which the gradient is to be calculated.
    :type x: np.ndarray
    :param eps: Small offset used for numerical differentiation. Default is 1e-5.
    :type eps: float
    :return: Computed numerical gradient, which is an array of the same shape as
        the input x.
    :rtype: np.ndarray
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=[["readwrite"]])

    while not it.finished:
        idx = it.multi_index
        original = x[idx]

        x[idx] = original + eps
        loss_plus = f(x)

        x[idx] = original - eps
        loss_minus = f(x)

        x[idx] = original
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()

    return grad


def test_conv2d_gradient_wrt_input():
    """
    Tests the gradient computation of the Conv2D class with respect to its input.

    This function validates the correctness of the backward-pass gradient calculation
    of the Conv2D class by comparing the analytical gradients to numerical gradients.
    It uses a random single-channel 5x5 input, applies the convolution operation with
    specific parameters, and calculates the analytical gradient. The analytical gradient
    is then compared to the numerical gradient using the `allclose` method from
    NumPy's testing module for accuracy verification.

    :raises AssertionError: If the analytical gradient computed does not match the
                            numerical gradient within the specified tolerance parameters.
    :return: None
    """
    x = np.random.randn(1, 1, 5, 5)
    conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

    def loss_fn(input_x):
        return conv.forward(input_x).sum()

    conv.forward(x)
    grad_output = np.ones_like(conv.output)
    analytical = conv.backward(grad_output)
    numerical = numerical_gradient(loss_fn, x.copy())

    np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)


def test_conv2d_gradient_wrt_weights():
    """
    Tests the gradient computation of `Conv2D` weights by comparing the analytical gradient
    calculated during backpropagation with the numerical gradient computed using the finite
    difference method. It verifies the correctness of the implementation of the gradient
    calculation for the convolutional layer's weights.

    :raises AssertionError: If the difference between the analytical and numerical gradients
        exceeds the specified tolerances.
    """
    x = np.random.randn(1, 1, 5, 5)
    conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

    def loss_fn(weight):
        conv.W.value = weight
        return conv.forward(x).sum()

    conv.forward(x)
    grad_output = np.ones_like(conv.output)
    conv.backward(grad_output)

    numerical = numerical_gradient(loss_fn, conv.W.value.copy())
    np.testing.assert_allclose(conv.W.grad, numerical, rtol=1e-4, atol=1e-6)


def test_conv2d_gradient_wrt_bias():
    """
    Tests the gradient of a 2-dimensional convolution operation with respect to the bias term.
    This function verifies that the computed gradient of the bias via backpropagation is
    consistent with the numerically approximated gradient. To achieve this, it sets up a
    Convolutional Layer (Conv2D), computes the forward pass followed by a backward pass,
    and compares the calculated gradient to the numerical gradient using `numpy.testing.assert_allclose`.
    The function ensures the accuracy and correctness of the gradient computation for the bias
    in the Conv2D layer.

    :return: None
    """
    x = np.random.randn(1, 1, 5, 5)
    conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

    def loss_fn(bias):
        conv.b.value = bias
        return conv.forward(x).sum()

    conv.forward(x)
    grad_output = np.ones_like(conv.output)
    conv.backward(grad_output)

    numerical = numerical_gradient(loss_fn, conv.b.value.copy())
    np.testing.assert_allclose(conv.b.grad, numerical, rtol=1e-4, atol=1e-6)


def test_maxpool2d_output_shape(input_tensor_pooling):
    """
    Tests the output shape of a 2D max pooling operation. Applies a 2D max pooling
    operation over an input tensor with a kernel size of 2 and a stride of 2.
    Verifies if the resulting tensor's shape matches the expected dimensions.

    :param input_tensor_pooling: Input tensor to be tested for max pooling. The
        input should be a 4-dimensional tensor, typically representing
        batch size, number of channels, height, and width.
    :type input_tensor_pooling: torch.Tensor
    :return: None
    """
    pool = MaxPool2D(kernel_size=2, stride=2)
    output = pool.forward(input_tensor_pooling)
    assert output.shape == (
        1,
        1,
        2,
        2,
    ), f"Expected output shape (1, 1, 2, 2), got {output.shape}"


def test_maxpool2d_forward_values(input_tensor_pooling):
    """
    Tests the forward pass of the MaxPool2D layer using a specific input tensor. This
    includes comparing the output of the layer with an expected result to ensure the
    pooling operation is functioning correctly.

    :param input_tensor_pooling: The input tensor for the MaxPool2D layer forward pass
        operation. The shape and type should correspond to the requirements of the pooling
        layer.
    :return: None
    """
    pool = MaxPool2D(kernel_size=2, stride=2)
    output = pool.forward(input_tensor_pooling)

    expected = np.array([[[[6, 8], [14, 16]]]], dtype=np.float32)

    np.testing.assert_array_equal(output, expected)


def test_maxpool2d_backward_single_grad():
    """
    Tests the backward operation of the MaxPool2D layer for a single gradient.
    This function verifies the correctness of the gradient computation during the
    backpropagation step by comparing the calculated gradient with an expected result.

    :raises AssertionError: If `grad_input` does not match the `expected` tensor.
    """
    x = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    pool = MaxPool2D(kernel_size=2, stride=2)
    pool.forward(x)

    grad_output = np.array([[[[1]]]], dtype=np.float32)
    grad_input = pool.backward(grad_output)

    expected = np.array([[[[0, 0], [0, 1]]]], dtype=np.float32)

    np.testing.assert_array_equal(grad_input, expected)


def test_maxpool2d_backward_multiple_regions():
    """
    Tests the backward pass of the MaxPool2D operation over an input array with
    multiple pooling regions. The function verifies that the gradient computed
    during the backward pass aligns with the expected gradient array. This test
    ensures the correctness of gradient propagation for MaxPool2D when multiple
    regions are involved.

    :return: None
    """
    x = np.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]],
        dtype=np.float32,
    )
    pool = MaxPool2D(kernel_size=2, stride=2)
    out = pool.forward(x)

    grad_output = np.ones_like(out)
    grad_input = pool.backward(grad_output)

    expected = np.array(
        [[[[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1]]]], dtype=np.float32
    )

    np.testing.assert_array_equal(grad_input, expected)
