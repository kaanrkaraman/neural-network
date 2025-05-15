import numpy as np

from net.losses import Loss
from net.losses.classification.binary_cross_entropy import BinaryCrossEntropy
from net.losses.classification.cross_entropy import CrossEntropy
from net.losses.regression.mse import MeanSquaredError


def numerical_gradient_loss(
    loss_fn: Loss, pred: np.ndarray, target: np.ndarray, eps=1e-5
):
    """
    Compute the numerical gradient of a loss function with respect
    to predictions. This function approximates the gradient using
    finite differences by slightly perturbing the prediction values
    and observing the change in the loss. This is useful for verifying
    the correctness of analytical gradients.

    :param loss_fn: The loss function used for computing the gradient.
    :type loss_fn: Loss
    :param pred: The predictions array for which the gradient is computed.
    :type pred: np.ndarray
    :param target: The ground truth values against which the predictions
        are compared.
    :type target: np.ndarray
    :param eps: The small epsilon value used for numerical differentiation.
        Default is 1e-5.
    :type eps: float
    :return: A numpy array representing the computed numerical gradient
        of the loss with respect to the predictions, having the same shape
        as the `pred` array.
    :rtype: np.ndarray
    """
    grad = np.zeros_like(pred)

    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            pred_pos = pred.copy()
            pred_neg = pred.copy()

            pred_pos[i, j] += eps
            pred_neg[i, j] -= eps

            loss_pos = loss_fn.forward(pred_pos, target)
            loss_neg = loss_fn.forward(pred_neg, target)

            grad[i, j] = (loss_pos - loss_neg) / (2 * eps)

    return grad


def test_mse_forward():
    """
    Tests the forward pass of the Mean Squared Error (MSE) computation.

    This function validates the correctness of the forward method in the
    MeanSquaredError class. It verifies that the calculated MSE loss matches
    the expected value using the provided predictions and target arrays. If
    the calculated loss is not close to the expected value, it raises an
    assertion error.

    :raises AssertionError: If the calculated loss does not match the expected
        mean squared error value.
    """
    mse = MeanSquaredError()
    pred = np.array([[0.5, 0.8]])
    target = np.array([[1.0, 0.0]])
    loss = mse.forward(pred, target)
    expected = np.mean((pred - target) ** 2)
    assert np.isclose(loss, expected), f"Expected {expected}, got {loss}"


def test_mse_backward_shape():
    """
    Tests the `backward` method of the MeanSquaredError class to ensure that
    the gradient's shape matches the shape of the predictions. The test
    involves forward computation followed by the backward computation of
    gradient. It asserts the shape of the gradient computed by the backward
    method.

    :raises AssertionError: If the shape of the gradient does not match the
        shape of the predictions.
    """
    mse = MeanSquaredError()
    pred = np.random.randn(4, 3)
    target = np.random.randn(4, 3)
    mse.forward(pred, target)
    grad = mse.backward()
    assert grad.shape == pred.shape


def test_mse_gradient_check():
    """
    Tests the gradient calculation for a Mean Squared Error (MSE) loss function by
    comparing the analytically computed gradients with numerically estimated
    gradients. Ensures that the two results are close within specified tolerances.

    :raises AssertionError: If the analytical and numerical gradients do not match
        within the relative (`rtol`) and absolute (`atol`) tolerances.
    """
    pred = np.random.rand(3, 2)
    target = np.random.rand(3, 2)
    loss_fn = MeanSquaredError()

    loss_fn.forward(pred, target)
    analytical = loss_fn.backward()
    numerical = numerical_gradient_loss(loss_fn, pred.copy(), target)

    np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)


def test_bce_forward():
    """
    Tests the forward method of the BinaryCrossEntropy class to ensure it computes the
    correct loss value given predictions and target values.

    The function calculates the expected binary cross-entropy loss manually using the
    predicted and target values, clips the predictions to avoid computational errors,
    and compares the result of the BinaryCrossEntropy's forward method with the expected
    loss using an assertion.

    :raises AssertionError: If the calculated loss from the BinaryCrossEntropy's forward
        method does not match the manually calculated expected loss.
    """
    bce = BinaryCrossEntropy()
    pred = np.array([[0.9], [0.1]])
    target = np.array([[1.0], [0.0]])
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    expected = -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))
    loss = bce.forward(pred, target)
    assert np.isclose(loss, expected), f"Expected {expected}, got {loss}"


def test_bce_backward_shape():
    """
    Tests the backward shape computation of the BinaryCrossEntropy class to ensure
    that the gradient shape matches the predictions.

    :raises AssertionError: If the shape of the gradient does not match the shape
        of the predictions.
    """
    bce = BinaryCrossEntropy()
    pred = np.random.rand(4, 1)
    target = np.random.randint(0, 2, (4, 1))
    bce.forward(pred, target)
    grad = bce.backward()
    assert grad.shape == pred.shape


def test_bce_gradient_check():
    """
    Performs a test to verify the correctness of the computed gradient
    of the Binary Cross-Entropy loss function through analytical and
    numerical comparison. The test ensures that the backward pass of
    the loss function produces gradients that closely match those
    calculated using a numerical approximation.

    :raises AssertionError: If the analytical gradients diverge from
                            the numerical gradients beyond the
                            specified tolerances.
    """
    pred = np.clip(np.random.rand(3, 1), 1e-5, 1 - 1e-5)
    target = np.random.randint(0, 2, (3, 1))
    loss_fn = BinaryCrossEntropy()

    loss_fn.forward(pred, target)
    analytical = loss_fn.backward()
    numerical = numerical_gradient_loss(loss_fn, pred.copy(), target)

    np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)


def test_ce_forward():
    """
    Tests the forward pass computation of the CrossEntropy loss function.

    The function creates prediction and target arrays, clips the predictions to avoid numerical
    instabilities, and calculates the expected loss value using the cross-entropy formula. It then
    uses the `forward` method of the `CrossEntropy` class to compute the loss and asserts that
    it matches the expected value. This ensures that the `CrossEntropy` forward pass behaves
    correctly and produces accurate results.

    :raises AssertionError: If the computed loss does not match the expected loss value.
    """
    ce = CrossEntropy()
    pred = np.array([[0.7, 0.2, 0.1]])
    target = np.array([[1.0, 0.0, 0.0]])
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    expected = -np.mean(np.sum(target * np.log(pred), axis=1))
    loss = ce.forward(pred, target)
    assert np.isclose(loss, expected), f"Expected {expected}, got {loss}"


def test_ce_backward_shape():
    """
    Tests the correctness of the backward pass shape in the cross-entropy
    implementation. The function initializes a random predicted probability
    distribution and a one-hot encoded target distribution as input, then
    it computes the forward pass of the cross-entropy and calculates the
    gradient during the backward pass. Finally, it asserts whether the shape
    of the computed gradient matches the shape of the input predictions.

    :raises AssertionError: If the shape of the gradient from the backward pass
        does not equal the shape of the predicted input.
    :return: None
    """
    ce = CrossEntropy()
    pred = np.clip(np.random.rand(4, 3), 1e-5, 1 - 1e-5)
    pred /= pred.sum(axis=1, keepdims=True)
    target = np.zeros_like(pred)
    target[np.arange(4), np.random.randint(0, 3, size=4)] = 1
    ce.forward(pred, target)
    grad = ce.backward()
    assert grad.shape == pred.shape


def test_ce_gradient_check():
    """
    Tests that the gradient of the cross-entropy loss function is correctly computed by comparing
    the analytical gradient against a numerically computed approximation.

    The test initializes random predictions and corresponding one-hot target labels, ensuring
    that the predictions are normalized. It calculates the cross-entropy loss and computes
    the analytical gradient via the backward pass of the loss function. The analytical gradient
    is compared with a numerically estimated gradient using the numerical gradient function.
    The comparison is done within specified relative and absolute tolerances.

    :return: None
    """
    pred = np.clip(np.random.rand(3, 3), 1e-5, 1 - 1e-5)
    pred /= pred.sum(axis=1, keepdims=True)
    target = np.zeros_like(pred)
    target[np.arange(3), np.random.randint(0, 3, size=3)] = 1
    loss_fn = CrossEntropy()

    loss_fn.forward(pred, target)
    analytical = loss_fn.backward()
    numerical = numerical_gradient_loss(loss_fn, pred.copy(), target)

    np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)
