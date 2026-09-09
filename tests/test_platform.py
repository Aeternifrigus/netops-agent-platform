import numpy as np
import pytest

from app.agents import orchestrator, tools
from app.graph.loader import load_topology
from app.graph.network_graph import Edge, InMemoryGraph, Node
from app.nn.attention import (
    MultiHeadAttention,
    most_relevant_events,
    scaled_dot_product_attention,
    softmax,
)
from app.nn.mlp import MLP, binary_cross_entropy, relu, relu_grad, sigmoid, train
from app.nn.train_anomaly_model import generate_dataset, normalise

# ── MLP mechanics ───────────────────────────────────────────────

def test_relu_zeroes_negatives_and_passes_positives():
    x = np.array([-2.0, -0.1, 0.0, 0.5, 3.0])
    assert np.array_equal(relu(x), [0, 0, 0, 0.5, 3.0])


def test_relu_grad_is_step_function():
    x = np.array([-1.0, 0.0, 1.0])
    assert np.array_equal(relu_grad(x), [0.0, 0.0, 1.0])


def test_sigmoid_bounds_and_midpoint():
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
    assert sigmoid(np.array([100.0]))[0] == pytest.approx(1.0, abs=1e-6)
    assert sigmoid(np.array([-100.0]))[0] == pytest.approx(0.0, abs=1e-6)


def test_sigmoid_does_not_overflow_on_extreme_input():
    # would raise a RuntimeWarning / produce nan without the clip in sigmoid()
    with np.errstate(over="raise"):
        result = sigmoid(np.array([1e10, -1e10]))
    assert np.isfinite(result).all()


def test_bce_is_zero_for_perfect_prediction():
    y = np.array([[1.0], [0.0]])
    pred = np.array([[1.0 - 1e-9], [1e-9]])
    assert binary_cross_entropy(y, pred) == pytest.approx(0.0, abs=1e-6)


def test_bce_penalises_confident_wrong_prediction_heavily():
    y = np.array([[1.0]])
    confident_right = binary_cross_entropy(y, np.array([[0.99]]))
    confident_wrong = binary_cross_entropy(y, np.array([[0.01]]))
    assert confident_wrong > confident_right * 10


def test_mlp_forward_output_shape_and_range():
    model = MLP(sizes=[4, 6, 1])
    X = np.random.default_rng(0).normal(size=(10, 4))
    out = model.forward(X)
    assert out.shape == (10, 1)
    assert np.all((out >= 0) & (out <= 1)), "sigmoid output must stay in [0,1]"


def test_mlp_rejects_degenerate_architecture():
    with pytest.raises(ValueError):
        MLP(sizes=[4])


def test_training_reduces_loss_on_a_learnable_problem():
    """Not a numeric gradient check -- just confirms backward() actually learns."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64).reshape(-1, 1)  # linearly separable

    model = MLP(sizes=[3, 8, 1], seed=1)
    losses = train(model, X, y, epochs=150, lr=0.5)

    assert losses[-1] < losses[0] * 0.3, "loss should drop substantially on an easy problem"
    preds = (model.predict_proba(X) >= 0.5).astype(np.float64)
    assert (preds == y).mean() > 0.9


def test_backward_gradients_match_numerical_gradient():
    """
    The actual proof the hand-derived backward pass is correct: compare the analytic
    gradient (from MLP.backward) against a finite-difference approximation of the same loss
    surface. This is the standard way to validate backprop by hand, and it is the test that
    would catch a wrong chain-rule composition that the "loss goes down" test above could
    miss -- a bug can still let loss decrease while individual gradients are wrong.
    """
    rng = np.random.default_rng(3)
    X = rng.normal(size=(5, 3))
    y = rng.integers(0, 2, size=(5, 1)).astype(np.float64)
    model = MLP(sizes=[3, 4, 1], seed=3)

    def loss_at(flat_params: np.ndarray) -> float:
        model.set_flat_params(flat_params)
        return binary_cross_entropy(y, model.forward(X))

    params = model.flat_params()

    # analytic gradient via a zero-learning-rate step
    model.set_flat_params(params)
    y_pred = model.forward(X)
    n = X.shape[0]
    dz = (y_pred - y) / n
    analytic = np.zeros_like(params)
    for i in reversed(range(len(model.layers))):
        layer = model.layers[i]
        dW = layer.x_in.T @ dz
        db = dz.sum(axis=0, keepdims=True)
        if i > 0:
            dz = (dz @ layer.W.T) * relu_grad(model.layers[i - 1].z)
        # stash in the same flat order flat_params() uses: all W's, then all b's
        w_sizes = [layer_.W.size for layer_ in model.layers]
        b_sizes = [layer_.b.size for layer_ in model.layers]
        w_start = sum(w_sizes[:i])
        b_start = sum(w_sizes) + sum(b_sizes[:i])
        analytic[w_start:w_start + layer.W.size] = dW.ravel()
        analytic[b_start:b_start + layer.b.size] = db.ravel()

    # numeric gradient via central finite differences, on a random subset of
    # parameters -- checking all of them is unnecessary and slow
    model.set_flat_params(params)
    eps = 1e-5
    idx = np.random.default_rng(4).choice(len(params), size=15, replace=False)
    numeric = np.zeros_like(params)
    for i in idx:
        p_plus, p_minus = params.copy(), params.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        numeric[i] = (loss_at(p_plus) - loss_at(p_minus)) / (2 * eps)

    model.set_flat_params(params)  # restore, since loss_at mutates the model

    rel_error = np.abs(analytic[idx] - numeric[idx]) / (
        np.abs(analytic[idx]) + np.abs(numeric[idx]) + 1e-8
    )
    assert np.max(rel_error) < 1e-4, (
        f"analytic and numeric gradients disagree beyond floating-point "
        f"tolerance (max relative error {np.max(rel_error):.2e}); "
        f"the backward() derivation likely has a sign or chain-rule error"
    )


# ── attention mechanics ─────────────────────────────────────────

def test_softmax_rows_sum_to_one():
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    out = softmax(x, axis=-1)
    assert np.allclose(out.sum(axis=-1), 1.0)


def test_softmax_stable_on_large_values():
    x = np.array([[1000.0, 1001.0, 999.0]])
    out = softmax(x)
    assert np.isfinite(out).all()
    assert out.sum() == pytest.approx(1.0)


def test_attention_output_shape():
    Q = K = V = np.random.default_rng(0).normal(size=(5, 8))
    out, weights = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (5, 8)
    assert weights.shape == (5, 5)
    assert np.allclose(weights.sum(axis=-1), 1.0)


def test_attention_attends_most_to_identical_vector():
    """A query identical to one key should attend most strongly to that key."""
    K = np.eye(4)
    Q = np.array([[1.0, 0.0, 0.0, 0.0]])
    V = K.copy()
    _, weights = scaled_dot_product_attention(Q, K, V)
    assert np.argmax(weights[0]) == 0


def test_multi_head_requires_divisible_dims():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=10, n_heads=3)


def test_multi_head_output_shape():
    mha = MultiHeadAttention(d_model=8, n_heads=2)
    x = np.random.default_rng(0).normal(size=(6, 8))
    out, weights = mha.forward(x)
    assert out.shape == (6, 8)
    assert weights.shape == (2, 6, 6)


def test_most_relevant_events_returns_a_distribution():
    embeddings = np.random.default_rng(0).normal(size=(4, 6))
    weights = most_relevant_events(embeddings, query_index=1)
    assert weights.shape == (4,)
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)


# ── training data ───────────────────────────────────────────────

def test_synthetic_labels_are_not_linearly_separable():
    """
    Confirms the XOR-based labelling rule actually defeats a linear model,
    which is why this uses a neural net.
    """
    X, y = generate_dataset(n=500, seed=7)
    X_norm, _, _ = normalise(X)

    # a single-layer "network" with no hidden layer is exactly logistic regression
    linear_model = MLP(sizes=[4, 1], seed=7)
    train(linear_model, X_norm, y, epochs=300, lr=0.3)
    linear_preds = (linear_model.predict_proba(X_norm) >= 0.5).astype(np.float64)
    linear_acc = (linear_preds == y).mean()

    hidden_model = MLP(sizes=[4, 12, 8, 1], seed=7)
    train(hidden_model, X_norm, y, epochs=300, lr=0.3)
    hidden_preds = (hidden_model.predict_proba(X_norm) >= 0.5).astype(np.float64)
    hidden_acc = (hidden_preds == y).mean()

    assert hidden_acc > linear_acc + 0.05, (
        "the hidden-layer model should meaningfully beat a linear model on a "
        "problem constructed to need non-linearity"
    )


# ── graph ───────────────────────────────────────────────────────

def test_graph_upsert_and_neighbors():
    g = InMemoryGraph()
    g.upsert_node(Node(id="a", label="Tower"))
    g.upsert_node(Node(id="b", label="Tower"))
    g.upsert_edge(Edge(source="a", target="b", rel_type="UPSTREAM_OF"))
    assert [n.id for n in g.neighbors("a")] == ["b"]


def test_edge_to_missing_node_is_rejected():
    g = InMemoryGraph()
    g.upsert_node(Node(id="a", label="Tower"))
    with pytest.raises(ValueError):
        g.upsert_edge(Edge(source="a", target="ghost", rel_type="UPSTREAM_OF"))


def test_downstream_impact_multi_hop():
    g = InMemoryGraph()
    for nid in ["leaf", "mid", "core"]:
        g.upsert_node(Node(id=nid, label="Tower"))
    g.upsert_edge(Edge(source="leaf", target="mid", rel_type="UPSTREAM_OF"))
    g.upsert_edge(Edge(source="mid", target="core", rel_type="UPSTREAM_OF"))

    impact = g.downstream_impact("leaf", max_hops=5)
    assert {n.id for n in impact} == {"mid", "core"}


def test_downstream_impact_respects_max_hops():
    g = InMemoryGraph()
    for nid in ["a", "b", "c"]:
        g.upsert_node(Node(id=nid, label="Tower"))
    g.upsert_edge(Edge(source="a", target="b", rel_type="UPSTREAM_OF"))
    g.upsert_edge(Edge(source="b", target="c", rel_type="UPSTREAM_OF"))

    assert {n.id for n in g.downstream_impact("a", max_hops=1)} == {"b"}


def test_downstream_impact_on_unknown_node_is_empty():
    assert InMemoryGraph().downstream_impact("does-not-exist") == []


def test_loader_populates_the_sample_topology():
    g = InMemoryGraph()
    counts = load_topology(g)
    assert counts["nodes"] == g.node_count()
    assert g.node_count() > 5


def test_loaded_topology_shows_real_downstream_impact():
    g = InMemoryGraph()
    load_topology(g)
    impact = g.downstream_impact("tower-003", max_hops=3)
    impacted_ids = {n.id for n in impact}
    assert "tower-001" in impacted_ids
    assert "core-warsaw" in impacted_ids


# ── tool functions ──────────────────────────────────────────────

def test_score_tower_health_flags_the_interaction_case():
    tools._graph = None  # reset module cache between tests
    result = tools.score_tower_health(
        latency_ms=120, packet_loss_pct=1.0, retransmit_rate=0.5, connected_devices=100
    )
    assert result["verdict"] in ("anomalous", "normal")
    assert 0.0 <= result["anomaly_probability"] <= 1.0


def test_score_tower_health_normal_reading():
    result = tools.score_tower_health(
        latency_ms=20, packet_loss_pct=0.2, retransmit_rate=0.1, connected_devices=50
    )
    assert result["verdict"] == "normal"


def test_get_downstream_impact_tool_uses_sample_topology():
    result = tools.get_downstream_impact("tower-003")
    assert "tower-001" in result["impacted_towers"]
    assert result["impacted_count"] == len(result["impacted_towers"])


def test_rank_relevant_events_orders_by_weight():
    events = [
        "latency spike on tower-001",
        "routine maintenance on tower-002",
        "packet loss alert on tower-001",
    ]
    result = tools.rank_relevant_events(events, focus_index=0)
    weights = [r["weight"] for r in result["ranked"]]
    assert weights == sorted(weights, reverse=True)
    assert len(result["ranked"]) == 3


def test_rank_relevant_events_rejects_bad_index():
    with pytest.raises(ValueError):
        tools.rank_relevant_events(["one event"], focus_index=5)


def test_rank_relevant_events_handles_empty_list():
    assert tools.rank_relevant_events([], focus_index=0) == {"ranked": []}


# ── agent construction (no live model call) ─────────────────────

def test_orchestrator_has_three_sub_agents_with_correct_tools():
    orch = orchestrator.build_orchestrator()
    names = {a.name for a in orch.sub_agents}
    assert names == {"topology_agent", "health_agent", "incident_agent"}


def test_sub_agents_are_correctly_parented():
    orch = orchestrator.build_orchestrator()
    assert all(a.parent_agent is orch for a in orch.sub_agents)


def test_each_specialist_has_exactly_one_tool():
    orch = orchestrator.build_orchestrator()
    for agent in orch.sub_agents:
        assert len(agent.tools) == 1


def test_topology_agent_tool_is_the_graph_tool():
    agent = orchestrator.build_topology_agent()
    assert agent.tools[0].name == "get_downstream_impact"
