"""MLP with a hand-written backward pass. Gradients are checked in tests/test_platform.py."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)


def sigmoid(x: np.ndarray) -> np.ndarray:
    # clip to avoid overflow in exp for large negative x
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


@dataclass
class Layer:
    W: np.ndarray
    b: np.ndarray
    # cached during forward, consumed during backward
    x_in: np.ndarray | None = field(default=None, repr=False)
    z: np.ndarray | None = field(default=None, repr=False)


class MLP:
    """A plain feedforward network: Linear -> ReLU -> ... -> Linear -> Sigmoid."""

    def __init__(self, sizes: list[int], seed: int = 42) -> None:
        if len(sizes) < 2:
            raise ValueError("need at least an input and an output layer")
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.layers: list[Layer] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:], strict=True):
            # He initialisation: scaled for ReLU so activations neither vanish
            # nor explode as depth increases (see CA12's failure mode).
            scale = np.sqrt(2.0 / fan_in)
            W = rng.normal(0, scale, size=(fan_in, fan_out)).astype(np.float64)
            b = np.zeros((1, fan_out), dtype=np.float64)
            self.layers.append(Layer(W=W, b=b))

    def forward(self, X: np.ndarray) -> np.ndarray:
        a = X
        for i, layer in enumerate(self.layers):
            layer.x_in = a
            z = a @ layer.W + layer.b
            layer.z = z
            is_last = i == len(self.layers) - 1
            a = sigmoid(z) if is_last else relu(z)
        return a

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, lr: float) -> dict[str, float]:
        """
        Manual backpropagation. dL/dz at the output layer for sigmoid + binary cross-entropy
        collapses to (y_pred - y_true) -- a standard identity, used here explicitly rather
        than via autodiff, and this collapse is exactly why that loss/activation pairing is
        the conventional choice.
        """
        n = y_true.shape[0]
        grad_norms: dict[str, float] = {}

        dz = (y_pred - y_true) / n  # (batch, out)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            dW = layer.x_in.T @ dz
            db = dz.sum(axis=0, keepdims=True)
            grad_norms[f"layer{i}_dW_norm"] = float(np.linalg.norm(dW))

            if i > 0:
                da_prev = dz @ layer.W.T
                dz = da_prev * relu_grad(self.layers[i - 1].z)

            # gradient descent update (CA3): theta <- theta - eta * grad
            layer.W -= lr * dW
            layer.b -= lr * db

        return grad_norms

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def flat_params(self) -> np.ndarray:
        """Used only by the numerical gradient check in tests."""
        return np.concatenate([layer.W.ravel() for layer in self.layers] +
                               [layer.b.ravel() for layer in self.layers])

    def set_flat_params(self, flat: np.ndarray) -> None:
        i = 0
        for layer in self.layers:
            n = layer.W.size
            layer.W = flat[i:i + n].reshape(layer.W.shape)
            i += n
        for layer in self.layers:
            n = layer.b.size
            layer.b = flat[i:i + n].reshape(layer.b.shape)
            i += n


def train(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 200,
    lr: float = 0.1,
    verbose: bool = False,
) -> list[float]:
    losses = []
    for epoch in range(epochs):
        y_pred = model.forward(X)
        loss = binary_cross_entropy(y, y_pred)
        model.backward(y, y_pred, lr)
        losses.append(loss)
        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"epoch {epoch:4d}  loss {loss:.4f}")
    return losses
