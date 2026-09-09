"""Loads the sample topology into whichever graph backend is configured."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .network_graph import Edge, NetworkGraph, Node

logger = logging.getLogger(__name__)

DEFAULT_TOPOLOGY_PATH = Path(__file__).resolve().parents[2] / "data" / "network_topology.json"


def load_topology(graph: NetworkGraph, path: Path = DEFAULT_TOPOLOGY_PATH) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))

    for n in data["nodes"]:
        graph.upsert_node(Node(id=n["id"], label=n["label"], properties=n.get("properties", {})))

    edge_count = 0
    for e in data["edges"]:
        graph.upsert_edge(
            Edge(source=e["source"], target=e["target"], rel_type=e["rel_type"],
                 properties=e.get("properties", {}))
        )
        edge_count += 1

    logger.info("loaded topology: %d nodes, %d edges", len(data["nodes"]), edge_count)
    return {"nodes": len(data["nodes"]), "edges": edge_count}
