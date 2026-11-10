"""Core knowledge graph implementation."""

from __future__ import annotations

from collections import defaultdict

from .edge import GraphEdge
from .node import GraphNode


class KnowledgeGraph:
    """In-memory biomedical knowledge graph."""

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

        self._outgoing: dict[str, list[GraphEdge]] = defaultdict(list)
        self._incoming: dict[str, list[GraphEdge]] = defaultdict(list)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        if edge.source not in self.nodes:
            raise KeyError(f"Unknown source node: {edge.source}")

        if edge.target not in self.nodes:
            raise KeyError(f"Unknown target node: {edge.target}")

        self.edges.append(edge)

        self._outgoing[edge.source].append(edge)
        self._incoming[edge.target].append(edge)

    def get_node(self, node_id: str) -> GraphNode | None:
        return self.nodes.get(node_id)

    def neighbors(self, node_id: str) -> list[GraphNode]:
        return [
            self.nodes[e.target]
            for e in self._outgoing.get(node_id, [])
        ]

    def outgoing_edges(self, node_id: str) -> list[GraphEdge]:
        return list(self._outgoing.get(node_id, []))

    def incoming_edges(self, node_id: str) -> list[GraphEdge]:
        return list(self._incoming.get(node_id, []))

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self._outgoing.clear()
        self._incoming.clear()
