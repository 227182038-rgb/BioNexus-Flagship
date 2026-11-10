"""API models for the BioNexus scientific pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BioNexusAnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=1)


class BioNexusSubject(BaseModel):
    name: str
    id: str
    type: str


class BioNexusRankedEntity(BaseModel):
    entity_id: str
    score: float


class BioNexusEvidence(BaseModel):
    source: str
    identifier: str
    title: str
    confidence: float
    metadata: dict[str, str] = Field(default_factory=dict)


class BioNexusHypothesis(BaseModel):
    title: str
    description: str
    supporting_entities: list[str]
    confidence: float
    rationale: str


class BioNexusContradiction(BaseModel):
    topic: str
    claim_a: str
    claim_b: str
    source_a: str
    source_b: str
    confidence: float


class BioNexusScientificConfidence(BaseModel):
    evidence_quality: float
    source_diversity: float
    relationship_support: float
    literature_support: float
    consistency: float
    overall: float


class BioNexusGraphNode(BaseModel):
    id: str
    label: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float


class BioNexusGraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    weight: float
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    directed: bool


class BioNexusGraphResponse(BaseModel):
    analysis_id: str
    subject_id: str | None = None
    node_count: int
    edge_count: int
    nodes: list[BioNexusGraphNode] = Field(default_factory=list)
    edges: list[BioNexusGraphEdge] = Field(default_factory=list)


class BioNexusEvidenceResponse(BaseModel):
    analysis_id: str
    count: int
    evidence: list[BioNexusEvidence] = Field(
        default_factory=list
    )


class BioNexusLiteratureItem(BaseModel):
    publication_id: str
    title: str
    relevance: float
    confidence: float
    tier: str | None = None
    evidence: list[str] = Field(default_factory=list)


class BioNexusLiteratureResponse(BaseModel):
    analysis_id: str
    count: int
    literature: list[BioNexusLiteratureItem] = Field(
        default_factory=list
    )


class BioNexusStructureItem(BaseModel):
    structure_id: str
    title: str
    method: str | None = None
    resolution: str | None = None
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class BioNexusStructureResponse(BaseModel):
    analysis_id: str
    count: int
    structures: list[BioNexusStructureItem] = Field(
        default_factory=list
    )


class BioNexusAnalyzeResponse(BaseModel):
    analysis_id: str | None = None
    query: str
    subject: BioNexusSubject | None = None
    summary: str | None = None
    reasoning: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    ranked_entities: list[BioNexusRankedEntity] = Field(
        default_factory=list
    )
    hypotheses: list[BioNexusHypothesis] = Field(
        default_factory=list
    )
    contradictions: list[BioNexusContradiction] = Field(
        default_factory=list
    )
    representation_confidence: float | None = None
    scientific_confidence: BioNexusScientificConfidence | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


__all__ = [
    "BioNexusAnalyzeRequest",
    "BioNexusAnalyzeResponse",
    "BioNexusSubject",
    "BioNexusRankedEntity",
    "BioNexusEvidence",
    "BioNexusEvidenceResponse",
    "BioNexusLiteratureItem",
    "BioNexusLiteratureResponse",
    "BioNexusStructureItem",
    "BioNexusStructureResponse",
    "BioNexusHypothesis",
    "BioNexusContradiction",
    "BioNexusScientificConfidence",
    "BioNexusGraphNode",
    "BioNexusGraphEdge",
    "BioNexusGraphResponse",
]




