"""Exercises the FastAPI endpoints that need no model credentials."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)

print("=" * 60)
print("HEALTH")
print("=" * 60)
r = client.get("/health")
print(json.dumps(r.json(), indent=2))
assert r.status_code == 200

print()
print("=" * 60)
print("TOOL: impact  (graph traversal, tower-003 fails)")
print("=" * 60)
r = client.post("/tools/impact", json={"tower_id": "tower-003", "max_hops": 3})
print(json.dumps(r.json(), indent=2))
assert r.status_code == 200
assert "tower-001" in r.json()["impacted_towers"]

print()
print("=" * 60)
print("TOOL: health-score  (interaction anomaly: high latency + low loss)")
print("=" * 60)
cases = [
    {"latency_ms": 120, "packet_loss_pct": 0.5, "retransmit_rate": 0.3, "connected_devices": 100},
    {"latency_ms": 20, "packet_loss_pct": 0.2, "retransmit_rate": 0.1, "connected_devices": 50},
    {"latency_ms": 30, "packet_loss_pct": 6.0, "retransmit_rate": 0.2, "connected_devices": 80},
]
for c in cases:
    r = client.post("/tools/health-score", json=c)
    body = r.json()
    print(f"  {c} -> {body['verdict']} (p={body['anomaly_probability']})")
    assert r.status_code == 200

print()
print("=" * 60)
print("TOOL: relevance  (attention over recent events)")
print("=" * 60)
r = client.post("/tools/relevance", json={
    "event_descriptions": [
        "latency spike on tower-001",
        "routine maintenance on tower-002",
        "packet loss alert on tower-001",
    ],
    "focus_index": 0,
})
print(json.dumps(r.json(), indent=2))
assert r.status_code == 200

print()
print("=" * 60)
print("CHAT without credentials -> should fail clearly, not crash")
print("=" * 60)
r = client.post("/chat", json={"message": "what happens if tower-001 fails?"})
print(f"  status: {r.status_code}")
print(f"  detail: {r.json()['detail']}")
assert r.status_code == 503

print()
print("all endpoint checks passed")
