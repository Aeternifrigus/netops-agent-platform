"""Tool functions for the network-operations agents."""
from __future__ import annotations

import numpy as np

from ..graph.loader import load_topology
from ..graph.network_graph import NetworkGraph, get_graph
from ..nn.attention import most_relevant_events
from ..nn.train_anomaly_model import load_model, normalise

_graph: NetworkGraph | None = None
_model_cache = None


def _get_graph() -> NetworkGraph:
    global _graph
    if _graph is None:
        import os

        _graph = get_graph(os.environ.get("NEO4J_URI"), os.environ.get("NEO4J_USER", "neo4j"),
                            os.environ.get("NEO4J_PASSWORD", ""))
        load_topology(_graph)
    return _graph


def _get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = load_model()
    return _model_cache


def get_downstream_impact(tower_id: str, max_hops: int = 3) -> dict:
    """
    Find every tower that depends on the given tower as an upstream link, up to max_hops
    away. Use this when deciding how serious an outage at one tower is.
    """
    graph = _get_graph()
    impacted = graph.downstream_impact(tower_id, max_hops=max_hops)
    return {
        "tower_id": tower_id,
        "impacted_count": len(impacted),
        "impacted_towers": [n.id for n in impacted],
    }


def score_tower_health(
    latency_ms: float,
    packet_loss_pct: float,
    retransmit_rate: float,
    connected_devices: float,
) -> dict:
    """
    Score whether a tower's current telemetry reading looks anomalous, using a trained
    neural network rather than fixed thresholds, since the real failure pattern is an
    interaction between latency and packet loss, not either alone.
    """
    model, mean, std = _get_model()
    X = np.array([[latency_ms, packet_loss_pct, retransmit_rate, connected_devices]])
    X_norm, _, _ = normalise(X, mean, std)
    proba = float(model.predict_proba(X_norm)[0, 0])
    return {
        "anomaly_probability": round(proba, 4),
        "verdict": "anomalous" if proba >= 0.5 else "normal",
    }


def rank_relevant_events(event_descriptions: list[str], focus_index: int) -> dict:
    """
    Given a short list of recent event descriptions, rank how relevant each one is to the
    event at focus_index, using self-attention over simple bag-of-words embeddings. Use this
    to help decide which recent events are worth mentioning when explaining an incident.
    """
    if not event_descriptions:
        return {"ranked": []}
    if not (0 <= focus_index < len(event_descriptions)):
        raise ValueError(
            f"focus_index {focus_index} out of range for "
            f"{len(event_descriptions)} events"
        )

    # bag-of-words embedding; avoids depending on an embedding model here
    vocab = sorted({w for e in event_descriptions for w in e.lower().split()})
    dim = max(len(vocab), 2)

    def embed(text: str) -> np.ndarray:
        vec = np.zeros(dim)
        for w in text.lower().split():
            if w in vocab:
                vec[vocab.index(w) % dim] += 1.0
        return vec / (np.linalg.norm(vec) + 1e-9)

    embeddings = np.stack([embed(e) for e in event_descriptions])
    weights = most_relevant_events(embeddings, query_index=focus_index)

    ranked = sorted(
        zip(event_descriptions, weights.tolist(), strict=True),
        key=lambda pair: -pair[1],
    )
    return {"ranked": [{"event": e, "weight": round(w, 4)} for e, w in ranked]}
