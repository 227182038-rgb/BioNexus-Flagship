"""Graph algorithms for biomedical knowledge graphs."""

from __future__ import annotations

from collections import deque

from .graph import KnowledgeGraph


class GraphAlgorithms:
    """Common graph algorithms."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def bfs(self, start: str) -> list[str]:
        """Breadth-first traversal."""

        if not self.graph.has_node(start):
            return []

        visited: set[str] = set()
        queue: deque[str] = deque([start])
        order: list[str] = []

        while queue:
            node = queue.popleft()

            if node in visited:
                continue

            visited.add(node)
            order.append(node)

            for neighbor in self.graph.neighbors(node):
                if neighbor.id not in visited:
                    queue.append(neighbor.id)

        return order

    def dfs(self, start: str) -> list[str]:
        """Depth-first traversal."""

        if not self.graph.has_node(start):
            return []

        visited: set[str] = set()
        stack: list[str] = [start]
        order: list[str] = []

        while stack:
            node = stack.pop()

            if node in visited:
                continue

            visited.add(node)
            order.append(node)

            neighbors = self.graph.neighbors(node)

            for neighbor in reversed(neighbors):
                if neighbor.id not in visited:
                    stack.append(neighbor.id)

        return order

    def shortest_path(
        self,
        source: str,
        target: str,
    ) -> list[str]:

        if (
            not self.graph.has_node(source)
            or not self.graph.has_node(target)
        ):
            return []

        queue = deque([[source]])
        visited = {source}

        while queue:
            path = queue.popleft()
            node = path[-1]

            if node == target:
                return path

            for neighbor in self.graph.neighbors(node):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append(path + [neighbor.id])

        return []

    def degree(self, node_id: str) -> int:
        """Return node degree."""

        return (
            len(self.graph.outgoing_edges(node_id))
            + len(self.graph.incoming_edges(node_id))
        )

    def degree_centrality(self) -> dict[str, float]:
        """Simple degree centrality."""

        n = self.graph.node_count()

        if n <= 1:
            return {}

        result: dict[str, float] = {}

        for node in self.graph.nodes:
            result[node] = self.degree(node) / (n - 1)

        return result
