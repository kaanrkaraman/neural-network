import numpy as np
import pytest

from net.utils.metrics import mae, mse, rmse


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        (np.array([[1.0], [2.0], [3.0]]), np.array([[1.0], [2.0], [3.0]]), 0.0),
        (np.array([[1.0], [2.0], [3.0]]), np.array([[2.0], [2.0], [2.0]]), 2 / 3),
        (np.array([[0.0], [0.0], [0.0]]), np.array([[1.0], [1.0], [1.0]]), 1.0),
    ],
)
def test_mse(y_true, y_pred, expected):
    """
    Test the Mean Squared Error (MSE) function by comparing the result with
    expected values for specific test cases. The test uses parameterized inputs
    and asserts the closeness of the calculated MSE to the expected value. If the
    calculated result does not match the expected value within a tolerance, an
    assertion error is raised, indicating an MSE mismatch.

    :param y_true: Ground truth values. It should be a 2D NumPy array.
    :param y_pred: Predicted values corresponding to the ground truth. It should
        also be a 2D NumPy array.
    :param expected: The expected mean squared error derived for the test case.
        Its value is a float.
    :return: None
    """
    result = mse(y_true, y_pred)
    assert np.isclose(
        result, expected
    ), f"MSE mismatch: got {result}, expected {expected}"


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        (np.array([[1.0], [2.0], [3.0]]), np.array([[1.0], [2.0], [3.0]]), 0.0),
        (
            np.array([[1.0], [2.0], [3.0]]),
            np.array([[2.0], [2.0], [2.0]]),
            np.sqrt(2 / 3),
        ),
        (np.array([[0.0], [0.0], [0.0]]), np.array([[1.0], [1.0], [1.0]]), 1.0),
    ],
)
def test_rmse(y_true, y_pred, expected):
    """
    Tests the Root Mean Square Error (RMSE) calculation by comparing the result of
    the `rmse` function with the expected value for given test cases. The test
    cases include various combinations of predictions and ground truths, ensuring
    the correctness of the implementation across different scenarios.

    :param y_true: Ground truth values provided in test cases. Assumes inputs are
        numpy arrays of shape (n, 1).
    :type y_true: numpy.ndarray
    :param y_pred: Predicted values provided in test cases to compare against the
        ground truth values. Assumes inputs are numpy arrays of shape (n, 1).
    :type y_pred: numpy.ndarray
    :param expected: The expected RMSE value to verify against the result of the
        `rmse` function.
    :type expected: float
    :return: None. Assertions verify the correctness of the `rmse` calculations
        based on computed and expected RMSE values.
    """
    result = rmse(y_true, y_pred)
    assert np.isclose(
        result, expected
    ), f"RMSE mismatch: got {result}, expected {expected}"


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        (np.array([[1.0], [2.0], [3.0]]), np.array([[1.0], [2.0], [3.0]]), 0.0),
        (np.array([[1.0], [2.0], [3.0]]), np.array([[2.0], [2.0], [2.0]]), 2 / 3),
        (np.array([[0.0], [0.0], [0.0]]), np.array([[1.0], [1.0], [1.0]]), 1.0),
    ],
)
def test_mae(y_true, y_pred, expected):
    """
    Test the Mean Absolute Error (MAE) function by comparing computed results to
    expected values. The test cases include scenarios with perfectly matching
    predictions, uniform predictions differing from actual values, and completely
    erroneous predictions.

    :param y_true: The ground truth values as a 2D numpy array.
    :param y_pred: The predicted values as a 2D numpy array.
    :param expected: The expected MAE value as a float.
    :return: None
    """
    result = mae(y_true, y_pred)
    assert np.isclose(
        result, expected
    ), f"MAE mismatch: got {result}, expected {expected}"
