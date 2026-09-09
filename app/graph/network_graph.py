"""
Network topology as a graph: cell towers, the regions they serve, and the upstream links
between them. A knowledge-graph question ("what else is affected if this tower fails") is a
graph traversal, not a SQL join across foreign keys, which is the actual argument for Neo4j
over a relational store here -- not just using it because it was on a list.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass
class Node:
    id: str
    label: str                 # 'Tower' | 'Region' | 'Link'
    properties: dict = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    rel_type: str               # 'SERVES' | 'UPSTREAM_OF' | 'BACKS_UP'
    properties: dict = field(default_factory=dict)


class NetworkGraph(Protocol):
    def upsert_node(self, node: Node) -> None: ...
    def upsert_edge(self, edge: Edge) -> None: ...
    def neighbors(self, node_id: str, rel_type: str | None = None) -> list[Node]: ...
    def downstream_impact(self, node_id: str, max_hops: int = 3) -> list[Node]: ...
    def node_count(self) -> int: ...
    def healthy(self) -> bool: ...


class InMemoryGraph:
    """Adjacency-list graph. The offline default and what the tests exercise."""

    name = "memory"

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._out_edges: dict[str, list[Edge]] = {}

    def healthy(self) -> bool:
        return True

    def upsert_node(self, node: Node) -> None:
        self._nodes[node.id] = node
        self._out_edges.setdefault(node.id, [])

    def upsert_edge(self, edge: Edge) -> None:
        if edge.source not in self._nodes or edge.target not in self._nodes:
            raise ValueError(
                f"cannot add edge {edge.source}->{edge.target}: both nodes must exist first"
            )
        self._out_edges.setdefault(edge.source, []).append(edge)

    def neighbors(self, node_id: str, rel_type: str | None = None) -> list[Node]:
        edges = self._out_edges.get(node_id, [])
        if rel_type:
            edges = [e for e in edges if e.rel_type == rel_type]
        return [self._nodes[e.target] for e in edges if e.target in self._nodes]

    def downstream_impact(self, node_id: str, max_hops: int = 3) -> list[Node]:
        """
        BFS over UPSTREAM_OF edges: if `node_id` fails, which towers depend on it
        as their upstream link, transitively, up to max_hops away.
        """
        if node_id not in self._nodes:
            return []

        visited: set[str] = {node_id}
        frontier = [node_id]
        impacted: list[Node] = []

        for _ in range(max_hops):
            next_frontier = []
            for nid in frontier:
                for nbr in self.neighbors(nid, rel_type="UPSTREAM_OF"):
                    if nbr.id not in visited:
                        visited.add(nbr.id)
                        impacted.append(nbr)
                        next_frontier.append(nbr.id)
            if not next_frontier:
                break
            frontier = next_frontier

        return impacted

    def node_count(self) -> int:
        return len(self._nodes)


class Neo4jGraph:
    """Real Neo4j-backed implementation, used when NEO4J_URI is configured."""

    name = "neo4j"

    def __init__(self, uri: str, user: str, password: str) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def healthy(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as exc:
            logger.info("neo4j unreachable: %s", exc)
            return False

    def upsert_node(self, node: Node) -> None:
        with self._driver.session() as session:
            session.run(
                f"MERGE (n:{node.label} {{id: $id}}) SET n += $props",
                id=node.id, props=node.properties,
            )

    def upsert_edge(self, edge: Edge) -> None:
        with self._driver.session() as session:
            session.run(
                f"""
                MATCH (a {{id: $source}}), (b {{id: $target}})
                MERGE (a)-[r:{edge.rel_type}]->(b)
                SET r += $props
                """,
                source=edge.source, target=edge.target, props=edge.properties,
            )

    def neighbors(self, node_id: str, rel_type: str | None = None) -> list[Node]:
        rel = f":{rel_type}" if rel_type else ""
        with self._driver.session() as session:
            result = session.run(
                f"MATCH (a {{id: $id}})-[{rel}]->(b) "
                f"RETURN b.id AS id, labels(b) AS labels, b AS props",
                id=node_id,
            )
            return [
                Node(
                    id=r["id"],
                    label=r["labels"][0] if r["labels"] else "Node",
                    properties=dict(r["props"]),
                )
                for r in result
            ]

    def downstream_impact(self, node_id: str, max_hops: int = 3) -> list[Node]:
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (a {{id: $id}})-[:UPSTREAM_OF*1..{max_hops}]->(b)
                RETURN DISTINCT b.id AS id, labels(b) AS labels, b AS props
                """,
                id=node_id,
            )
            return [
                Node(
                    id=r["id"],
                    label=r["labels"][0] if r["labels"] else "Node",
                    properties=dict(r["props"]),
                )
                for r in result
            ]

    def node_count(self) -> int:
        with self._driver.session() as session:
            return session.run("MATCH (n) RETURN count(n) AS c").single()["c"]


def get_graph(uri: str | None, user: str = "neo4j", password: str = "") -> NetworkGraph:
    if uri:
        try:
            g = Neo4jGraph(uri, user, password)
            if g.healthy():
                return g
            logger.warning("neo4j configured but unreachable; using in-memory graph")
        except Exception as exc:
            logger.warning("neo4j driver init failed (%s); using in-memory graph", exc)
    return InMemoryGraph()
