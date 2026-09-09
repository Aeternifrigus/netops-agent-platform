"""Scaled dot-product and multi-head self-attention, from scratch."""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # subtract the max for numerical stability (IT5's log-sum-exp trick, applied
    # to plain softmax rather than its log form)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """AT1: Attn(Q,K,V) = softmax(QK^T / sqrt(d_k)) V"""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)  # the sqrt(d_k) scaling this file's docstring cites

    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights


class MultiHeadAttention:
    """
    AT2: run several attention operations in parallel on learned projections of
    the same input, then concatenate and mix.
    """

    def __init__(self, d_model: int, n_heads: int, seed: int = 42) -> None:
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must divide evenly by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        rng = np.random.default_rng(seed)
        scale = np.sqrt(1.0 / d_model)
        self.W_q = rng.normal(0, scale, (d_model, d_model))
        self.W_k = rng.normal(0, scale, (d_model, d_model))
        self.W_v = rng.normal(0, scale, (d_model, d_model))
        self.W_o = rng.normal(0, scale, (d_model, d_model))

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)

    def forward(
        self, x: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """x: (seq_len, d_model). Returns (output, per-head attention weights)."""
        Q, K, V = x @ self.W_q, x @ self.W_k, x @ self.W_v
        Qh, Kh, Vh = self._split_heads(Q), self._split_heads(K), self._split_heads(V)

        head_outputs, head_weights = [], []
        for h in range(self.n_heads):
            out, w = scaled_dot_product_attention(Qh[h], Kh[h], Vh[h], mask=mask)
            head_outputs.append(out)
            head_weights.append(w)

        concat = np.concatenate(head_outputs, axis=-1)  # (seq_len, d_model)
        output = concat @ self.W_o
        return output, np.stack(head_weights)  # (n_heads, seq_len, seq_len)


def most_relevant_events(
    event_embeddings: np.ndarray, query_index: int, n_heads: int = 2
) -> np.ndarray:
    """
    Given embedded recent events, return how much attention the event at query_index pays to
    every event including itself, averaged across heads. This is the actual, inspectable
    output an operations agent tool exposes: not a black-box relevance score, but the
    attention distribution itself.
    """
    mha = MultiHeadAttention(d_model=event_embeddings.shape[1], n_heads=n_heads)
    _, weights = mha.forward(event_embeddings)
    avg_weights = weights.mean(axis=0)  # average over heads
    return avg_weights[query_index]
