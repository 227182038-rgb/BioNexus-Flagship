"""Graph node definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GraphNode:
    """Represents a biomedical entity in the knowledge graph."""

    id: str

    label: str

    type: str

    properties: dict[str, Any] = field(default_factory=dict)

    aliases: list[str] = field(default_factory=list)

    evidence: list[str] = field(default_factory=list)

    confidence: float = 1.0

    def add_alias(self, alias: str) -> None:
        if alias and alias not in self.aliases:
            self.aliases.append(alias)

    def add_evidence(self, evidence: str) -> None:
        if evidence and evidence not in self.evidence:
            self.evidence.append(evidence)

    def update_properties(
        self,
        values: dict[str, Any],
    ) -> None:
        self.properties.update(values)
