"""Knowledge graph query interface."""

from __future__ import annotations

from .edge import EdgeType, GraphEdge
from .graph import KnowledgeGraph
from .node import GraphNode


class GraphQuery:
    """High-level query interface for the KnowledgeGraph."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def find_node(self, node_id: str) -> GraphNode | None:
        return self.graph.get_node(node_id)

    def neighbors(self, node_id: str) -> list[GraphNode]:
        return self.graph.neighbors(node_id)

    def outgoing(self, node_id: str) -> list[GraphEdge]:
        return self.graph.outgoing_edges(node_id)

    def incoming(self, node_id: str) -> list[GraphEdge]:
        return self.graph.incoming_edges(node_id)

    def by_type(self, node_type: str) -> list[GraphNode]:
        return [
            node
            for node in self.graph.nodes.values()
            if node.type == node_type
        ]

    def relations(
        self,
        node_id: str,
        relation: EdgeType,
    ) -> list[GraphEdge]:

        return [
            edge
            for edge in self.graph.outgoing_edges(node_id)
            if edge.relation == relation
        ]

    def related_nodes(
        self,
        node_id: str,
        relation: EdgeType,
    ) -> list[GraphNode]:

        return [
            self.graph.get_node(edge.target)
            for edge in self.relations(node_id, relation)
            if self.graph.get_node(edge.target) is not None
        ]

    def exists(self, node_id: str) -> bool:
        return self.graph.has_node(node_id)
