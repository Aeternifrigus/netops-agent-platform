"""Synthetic cell-tower health data and a training script for the anomaly MLP."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .mlp import MLP, train

FEATURE_NAMES = ["latency_ms", "packet_loss_pct", "retransmit_rate", "connected_devices"]
MODEL_PATH = Path(__file__).parent / "anomaly_model.npz"


def generate_dataset(n: int = 2000, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    latency = rng.uniform(5, 150, n)
    packet_loss = rng.uniform(0, 8, n)
    retransmit = rng.uniform(0, 5, n)
    devices = rng.uniform(10, 500, n)

    X = np.column_stack([latency, packet_loss, retransmit, devices])

    # XOR-like interaction between latency and loss, so a linear model can't fit it
    high_latency = latency > 80
    high_loss = packet_loss > 4
    interaction_anomaly = high_latency ^ high_loss  # XOR
    overload_anomaly = (devices > 400) & (retransmit > 2)

    y = (interaction_anomaly | overload_anomaly).astype(np.float64)
    noise_flip = rng.random(n) < 0.03  # 3% label noise, since real telemetry is never clean
    y = np.where(noise_flip, 1 - y, y)

    return X, y.reshape(-1, 1)


def normalise(X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    if mean is None:
        mean, std = X.mean(axis=0), X.std(axis=0)
        std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def train_and_save(epochs: int = 400, lr: float = 0.3, seed: int = 7) -> dict:
    X, y = generate_dataset(seed=seed)
    split = int(len(X) * 0.8)

    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train, mean, std = normalise(X_train_raw)
    X_test, _, _ = normalise(X_test_raw, mean, std)

    model = MLP(sizes=[4, 12, 8, 1], seed=seed)
    losses = train(model, X_train, y_train, epochs=epochs, lr=lr)

    test_pred = model.predict_proba(X_test)
    test_labels = (test_pred >= 0.5).astype(np.float64)
    accuracy = float((test_labels == y_test).mean())

    weights = {}
    for i, layer in enumerate(model.layers):
        weights[f"W{i}"] = layer.W
        weights[f"b{i}"] = layer.b
    weights["mean"] = mean
    weights["std"] = std
    weights["sizes"] = np.array(model.sizes)

    np.savez(MODEL_PATH, **weights)

    return {
        "final_train_loss": losses[-1],
        "test_accuracy": accuracy,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model_path": str(MODEL_PATH),
    }


def load_model() -> tuple[MLP, np.ndarray, np.ndarray]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"no trained model at {MODEL_PATH}; run `python -m app.nn.train_anomaly_model` first"
        )
    data = np.load(MODEL_PATH)
    sizes = list(data["sizes"])
    model = MLP(sizes=sizes)
    for i, layer in enumerate(model.layers):
        layer.W = data[f"W{i}"]
        layer.b = data[f"b{i}"]
    return model, data["mean"], data["std"]


if __name__ == "__main__":
    report = train_and_save()
    print(f"trained: test accuracy {report['test_accuracy']:.3f} "
          f"on {report['n_test']} held-out samples "
          f"(final train loss {report['final_train_loss']:.4f})")
