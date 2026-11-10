"""Knowledge graph export utilities."""

from __future__ import annotations

import json
from pathlib import Path

from .graph import KnowledgeGraph


class GraphExporter:
    """Export KnowledgeGraph into standard formats."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "type": node.type,
                    "aliases": node.aliases,
                    "properties": node.properties,
                    "confidence": node.confidence,
                }
                for node in self.graph.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation.value,
                    "weight": edge.weight,
                    "confidence": edge.confidence,
                    "directed": edge.directed,
                    "properties": edge.properties,
                    "evidence": edge.evidence,
                }
                for edge in self.graph.edges
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save_json(
        self,
        path: str | Path,
    ) -> Path:

        path = Path(path)

        path.write_text(
            self.to_json(),
            encoding="utf-8",
        )

        return path

    def to_graphml(self) -> str:

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            '<graph id="BioNexus" edgedefault="directed">',
        ]

        for node in self.graph.nodes.values():
            lines.append(
                f'<node id="{node.id}"/>'
            )

        for i, edge in enumerate(self.graph.edges):
            lines.append(
                f'<edge id="e{i}" '
                f'source="{edge.source}" '
                f'target="{edge.target}"/>'
            )

        lines.append("</graph>")
        lines.append("</graphml>")

        return "\n".join(lines)

    def save_graphml(
        self,
        path: str | Path,
    ) -> Path:

        path = Path(path)

        path.write_text(
            self.to_graphml(),
            encoding="utf-8",
        )

        return path
