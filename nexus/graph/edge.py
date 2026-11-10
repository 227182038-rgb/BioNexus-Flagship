"""Graph edge definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EdgeType(str, Enum):
    """Supported biological relationships."""

    ENCODES = "ENCODES"
    INTERACTS_WITH = "INTERACTS_WITH"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    HAS_STRUCTURE = "HAS_STRUCTURE"
    ANNOTATED_BY = "ANNOTATED_BY"
    PARTICIPATES_IN = "PARTICIPATES_IN"
    TARGETED_BY = "TARGETED_BY"
    TREATS = "TREATS"
    CITES = "CITES"
    SUPPORTS = "SUPPORTS"
    EXPRESSED_IN = "EXPRESSED_IN"
    ORTHOLOG_OF = "ORTHOLOG_OF"


@dataclass(slots=True)
class GraphEdge:
    """Represents a typed relationship between two graph nodes."""

    source: str

    target: str

    relation: EdgeType

    weight: float = 1.0

    confidence: float = 1.0

    evidence: list[str] = field(default_factory=list)

    properties: dict[str, Any] = field(default_factory=dict)

    directed: bool = True

    def add_evidence(self, value: str) -> None:
        if value and value not in self.evidence:
            self.evidence.append(value)

    def update_properties(
        self,
        values: dict[str, Any],
    ) -> None:
        self.properties.update(values)
