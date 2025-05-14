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
    result = mae(y_true, y_pred)
    assert np.isclose(
        result, expected
    ), f"MAE mismatch: got {result}, expected {expected}"
