from typing import Optional

import numpy as np

from net.losses._base import Loss


class MeanSquaredError(Loss):
    def __init__(self):
        self.pred: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self.pred = pred
        self.target = target
        loss = np.mean((pred - target) ** 2)
        return float(loss)

    def backward(self) -> np.ndarray:
        assert self.pred is not None and self.target is not None
        return (self.pred - self.target) * (2 / self.pred.size)
