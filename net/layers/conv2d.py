import numpy as np

from net.layers import Layer, Parameter


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        limit = np.sqrt(2 / fan_in)

        weight_shape = (out_channels, in_channels, *kernel_size)
        W_init = np.random.randn(*weight_shape) * limit
        W_init = np.clip(W_init, -1.0, 1.0)

        self.W = Parameter(W_init)
        self.b = Parameter(np.zeros(out_channels))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        N, C, H, W = x.shape
        Kh, Kw = self.kernel_size
        S = self.stride
        P = self.padding

        x_padded = np.pad(
            x, ((0, 0), (0, 0), (P, P), (P, P)), mode="constant", constant_values=0
        )

        H_out = (H + 2 * P - Kh) // S + 1
        W_out = (W + 2 * P - Kw) // S + 1
        out = np.zeros((N, self.out_channels, H_out, W_out))

        for n in range(N):
            for c_out in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * S
                        w_start = w * S
                        h_end = h_start + Kh
                        w_end = w_start + Kw

                        region = x_padded[n, :, h_start:h_end, w_start:w_end]
                        product = region * self.W.value[c_out]
                        product = np.clip(product, -1e6, 1e6)
                        val = np.sum(product)
                        val = np.clip(val, -1e6, 1e6)
                        val += self.b.value[c_out].item()
                        out[n, c_out, h, w] = val

        self.output = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        x = self.input
        N, C, H, W = x.shape
        Kh, Kw = self.kernel_size
        S = self.stride
        P = self.padding

        x_padded = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), mode="constant")
        dx_padded = np.zeros_like(x_padded)
        dW = np.zeros_like(self.W.value)
        db = np.zeros_like(self.b.value)

        H_out = grad_output.shape[2]
        W_out = grad_output.shape[3]

        for n in range(N):
            for c_out in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * S
                        w_start = w * S
                        h_end = h_start + Kh
                        w_end = w_start + Kw

                        region = x_padded[n, :, h_start:h_end, w_start:w_end]

                        dW[c_out] += grad_output[n, c_out, h, w] * region
                        db[c_out] += grad_output[n, c_out, h, w]
                        dx_padded[n, :, h_start:h_end, w_start:w_end] += (
                            grad_output[n, c_out, h, w] * self.W.value[c_out]
                        )

        if P > 0:
            dx = dx_padded[:, :, P:-P, P:-P]
        else:
            dx = dx_padded

        self.W.grad = dW
        self.b.grad = db
        return dx

    def update(self, lr: float) -> None:
        self.W.value -= lr * self.W.grad
        self.b.value -= lr * self.b.grad

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass