import numpy as np

from net.losses import Loss
from net.losses.binary_cross_entropy import BinaryCrossEntropy
from net.losses.cross_entropy import CrossEntropy
from net.losses.mse import MeanSquaredError


def numerical_gradient_loss(
    loss_fn: Loss, pred: np.ndarray, target: np.ndarray, eps=1e-5
):
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
    mse = MeanSquaredError()
    pred = np.array([[0.5, 0.8]])
    target = np.array([[1.0, 0.0]])
    loss = mse.forward(pred, target)
    expected = np.mean((pred - target) ** 2)
    assert np.isclose(loss, expected), f"Expected {expected}, got {loss}"


def test_mse_backward_shape():
    mse = MeanSquaredError()
    pred = np.random.randn(4, 3)
    target = np.random.randn(4, 3)
    mse.forward(pred, target)
    grad = mse.backward()
    assert grad.shape == pred.shape


def test_mse_gradient_check():
    pred = np.random.rand(3, 2)
    target = np.random.rand(3, 2)
    loss_fn = MeanSquaredError()

    loss_fn.forward(pred, target)
    analytical = loss_fn.backward()
    numerical = numerical_gradient_loss(loss_fn, pred.copy(), target)

    np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)


def test_bce_forward():
    bce = BinaryCrossEntropy()
    pred = np.array([[0.9], [0.1]])
    target = np.array([[1.0], [0.0]])
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    expected = -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))
    loss = bce.forward(pred, target)
    assert np.isclose(loss, expected), f"Expected {expected}, got {loss}"


def test_bce_backward_shape():
    bce = BinaryCrossEntropy()
    pred = np.random.rand(4, 1)
    target = np.random.randint(0, 2, (4, 1))
    bce.forward(pred, target)
    grad = bce.backward()
    assert grad.shape == pred.shape


def test_bce_gradient_check():
    pred = np.clip(np.random.rand(3, 1), 1e-5, 1 - 1e-5)
    target = np.random.randint(0, 2, (3, 1))
    loss_fn = BinaryCrossEntropy()

    loss_fn.forward(pred, target)
    analytical = loss_fn.backward()
    numerical = numerical_gradient_loss(loss_fn, pred.copy(), target)

    np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)


def test_ce_forward():
    ce = CrossEntropy()
    pred = np.array([[0.7, 0.2, 0.1]])
    target = np.array([[1.0, 0.0, 0.0]])
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    expected = -np.mean(np.sum(target * np.log(pred), axis=1))
    loss = ce.forward(pred, target)
    assert np.isclose(loss, expected), f"Expected {expected}, got {loss}"


def test_ce_backward_shape():
    ce = CrossEntropy()
    pred = np.clip(np.random.rand(4, 3), 1e-5, 1 - 1e-5)
    pred /= pred.sum(axis=1, keepdims=True)
    target = np.zeros_like(pred)
    target[np.arange(4), np.random.randint(0, 3, size=4)] = 1
    ce.forward(pred, target)
    grad = ce.backward()
    assert grad.shape == pred.shape


def test_ce_gradient_check():
    pred = np.clip(np.random.rand(3, 3), 1e-5, 1 - 1e-5)
    pred /= pred.sum(axis=1, keepdims=True)
    target = np.zeros_like(pred)
    target[np.arange(3), np.random.randint(0, 3, size=3)] = 1
    loss_fn = CrossEntropy()

    loss_fn.forward(pred, target)
    analytical = loss_fn.backward()
    numerical = numerical_gradient_loss(loss_fn, pred.copy(), target)

    np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)
