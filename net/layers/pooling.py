import numpy as np

from net.layers import Layer


class MaxPool2D(Layer):
    """
    2D Max Pooling Layer.

    Performs max pooling on 4D input tensors with shape (N, C, H, W).

    :param kernel_size: int or tuple, window size
    :param stride: int or tuple, step size between pooling regions
    """

    def __init__(self, kernel_size, stride=None):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        N, C, H, W = x.shape
        Kh, Kw = self.kernel_size
        Sh, Sw = self.stride

        H_out = (H - Kh) // Sh + 1
        W_out = (W - Kw) // Sw + 1
        out = np.zeros((N, C, H_out, W_out))

        self.cache = {}

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * Sh
                        w_start = w * Sw
                        h_end = h_start + Kh
                        w_end = w_start + Kw

                        region = x[n, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(region)
                        out[n, c, h, w] = max_val

                        # Save mask for backward
                        mask = region == max_val
                        self.cache[(n, c, h, w)] = (mask, h_start, w_start)

        self.output = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        N, C, H, W = self.input.shape
        dx = np.zeros_like(self.input)
        Kh, Kw = self.kernel_size
        Sh, Sw = self.stride

        H_out = grad_output.shape[2]
        W_out = grad_output.shape[3]

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        mask, h_start, w_start = self.cache[(n, c, h, w)]
                        h_end = h_start + Kh
                        w_end = w_start + Kw

                        dx[n, c, h_start:h_end, w_start:w_end] += (
                                grad_output[n, c, h, w] * mask
                        )

        return dx

    def update(self, learning_rate: float) -> None:
        pass  # No parameters to update

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass
