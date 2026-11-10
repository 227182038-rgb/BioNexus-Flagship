"""Typed data model for Nexus.

Every artifact that flows through Nexus ΓÇö evidence, claims, inferences,
uncertainty estimates, interpretations ΓÇö is represented here as a Pydantic
model with strict types. The model is the contract between modules: as long
as two modules agree on these types, they can interoperate without coupling
to each other's internals.

The model honours the Prime Directive by distinguishing deterministic
evidence (``EvidenceClass.DETERMINISTIC`` ΓÇö produced by BioKit) from
AI-assisted evidence (every other class). The Kernel refuses to register
claims whose evidence trail does not include at least one deterministic
source when the claim purports to be about a biological fact.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Identifiers
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ


def _new_id(prefix: str) -> str:
    """Generate a content-prefixed unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _fingerprint(content: str) -> str:
    """Deterministic SHA-256 fingerprint for content-addressable storage."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Evidence
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ


class EvidenceClass(str, Enum):
    """Classification of evidence sources by reliability profile.

    The class drives the confidence function applied to the evidence
    (see :class:`Confidence`). Deterministic evidence (BioKit output)
    is treated as ground truth within its domain of validity and never
    decays; every other class is subject to confidence revision over time.
    """

    DETERMINISTIC = "deterministic"  # BioKit calculation
    PUBLICATION = "publication"  # peer-reviewed literature
    ASSAY = "assay"  # experimental assay result
    DATABASE = "database"  # curated database (UniProt, ChEMBL, etc.)
    COMPUTATIONAL = "computational"  # AI/ML prediction
    EXTRACTED = "extracted"  # LLM-extracted assertion from text
    USER = "user"  # user-provided assertion


class Evidence(BaseModel):
    """A single unit of evidence entered into the Evidence Ledger.

    Evidence is append-only: once registered, an :class:`Evidence` record is
    never modified. If a source is retracted or superseded, a new
    :class:`Evidence` record is added and the old one is flagged via a
    :class:`ClaimEdge` of type ``supersedes`` or ``contradicts``.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: _new_id("ev"))
    source_class: EvidenceClass
    source_uri: str = Field(
        description="Stable URI or canonical identifier of the source "
        "(DOI, PubMed ID, UniProt accession, BioKit run ID, etc.)."
    )
    content: str = Field(description="Verbatim or canonical content of the evidence.")
    fingerprint: str = Field(description="SHA-256 fingerprint of (source_uri + content).")
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("fingerprint")
    @classmethod
    def _compute_fingerprint(cls, v: str, info: Any) -> str:
        # The fingerprint is normally set by the factory; if it's empty
        # or wrong, recompute from source_uri + content.
        values = info.data
        src = values.get("source_uri", "")
        content = values.get("content", "")
        expected = _fingerprint(f"{src}\x00{content}")
        return v or expected


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Uncertainty
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ


class UncertaintyType(str, Enum):
    """Three orthogonal sources of uncertainty.

    Following the architecture in the Critical Review, every AI-assisted
    claim carries an :class:`Uncertainty` envelope with up to three
    sub-components. Deterministic claims (BioKit) carry no uncertainty
    envelope: their uncertainty is zero by construction.
    """

    ALEATORIC = "aleatoric"  # irreducible noise in the data-generating process
    EPISTEMIC = "epistemic"  # reducible uncertainty due to limited data
    MODEL_FORM = "model_form"  # uncertainty about model class appropriateness


class Uncertainty(BaseModel):
    """A typed uncertainty envelope attached to a :class:`Claim`.

    Each present component is a number in ``[0, 1]``. ``None`` means the
    component was not estimated for this claim (different from ``0.0``,
    which means it was estimated to be zero).
    """

    model_config = ConfigDict(frozen=True)

    aleatoric: float | None = None
    epistemic: float | None = None
    model_form: float | None = None
    calibration_drift: float = Field(
        default=0.0,
        description="Recent calibration drift observed for the producing "
        "model, in [0, 1]. Non-zero values reduce effective confidence.",
    )

    @field_validator("aleatoric", "epistemic", "model_form", "calibration_drift")
    @classmethod
    def _in_unit_interval(cls, v: float | None) -> float | None:
        if v is None:
            return None
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Uncertainty component must be in [0, 1], got {v}")
        return v

    def total(self) -> float:
        """Aggregate uncertainty, treating missing components as zero.

        Uses root-sum-square so multiple moderate uncertainties do not
        combine linearly into a near-certain negative.
        """
        components = [v for v in (self.aleatoric, self.epistemic, self.model_form) if v is not None]
        if not components:
            return 0.0
        rss: float = sum(v * v for v in components) ** 0.5
        # Calibration drift scales the total up but cannot exceed 1.0.
        return float(min(1.0, rss * (1.0 + self.calibration_drift)))


class Confidence(BaseModel):
    """Confidence value attached to a :class:`Claim`.

    Confidence is in ``[0, 1]`` and is derived from the supporting
    evidence and the claim's uncertainty envelope. It is *not* a
    probability of correctness in the Bayesian sense; it is a calibrated
    reliability score whose interpretation depends on the evidence class.
    """

    model_config = ConfigDict(frozen=True)

    value: float = Field(ge=0.0, le=1.0)
    evidence_class: EvidenceClass
    decay_half_life_days: float | None = Field(
        default=None,
        description="If non-None, confidence decays with this half-life "
        "(in days) since the most recent supporting evidence was added.",
    )


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Provenance
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ


class Provenance(BaseModel):
    """Derivation chain for a :class:`Claim`.

    The provenance chain traces a claim back to its source artifacts:
    evidence ledger entries, model fingerprints, code commits, environment
    hashes. It is the architectural mechanism that makes Nexus
    scientifically accountable.

    The chain is stored as a list of "hops", each hop being either an
    Evidence ID or another Claim ID. The chain is acyclic by construction
    (claims cannot depend on themselves) but the data model does not
    enforce acyclicity at the type level; the Engine enforces it at
    registration time.
    """

    model_config = ConfigDict(frozen=True)

    hops: list[str] = Field(
        description="Ordered list of Evidence IDs and Claim IDs leading "
        "from source artifacts to the present claim."
    )
    model_fingerprint: str | None = Field(
        default=None,
        description="If the claim was produced by an AI model, "
        "SHA-256 fingerprint of (architecture + weights + config).",
    )
    code_commit: str | None = Field(
        default=None, description="Git commit hash of the producing code."
    )
    environment_hash: str | None = Field(
        default=None,
        description="Hash of the execution environment "
        "(Docker image digest or Nix derivation hash).",
    )


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Claims and the Claim Graph
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ


class ClaimEdgeType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REFINES = "refines"
    SUPERSEDES = "supersedes"


class Claim(BaseModel):
    """A scientific claim registered in the Claim Graph.

    A claim is the unit of scientific assertion in Nexus. It is the
    output of either BioKit (deterministic computation) or an AI
    intelligence module (interpretation, reasoning, hypothesis). Every
    claim carries provenance, an uncertainty envelope (or ``None`` for
    deterministic claims), and a confidence value.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: _new_id("cl"))
    statement: str = Field(description="Human-readable statement of the claim.")
    evidence_ids: list[str] = Field(default_factory=list, description="IDs of supporting evidence.")
    provenance: Provenance
    uncertainty: Uncertainty | None = Field(
        default=None,
        description="None for deterministic claims; required for AI-assisted claims.",
    )
    confidence: Confidence
    is_deterministic: bool = Field(
        description="True iff the claim is the direct output of BioKit "
        "or another deterministic computation.",
    )
    timestamp: float = Field(default_factory=time.time)
    tags: frozenset[str] = Field(default_factory=frozenset)


class ClaimEdge(BaseModel):
    """A directed edge in the Claim Graph."""

    model_config = ConfigDict(frozen=True)

    source_claim_id: str
    target_claim_id: str
    edge_type: ClaimEdgeType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str | None = None


class ClaimGraph(BaseModel):
    """The Claim Graph: claims as nodes, inferential relationships as edges."""

    claims: dict[str, Claim] = Field(default_factory=dict)
    edges: list[ClaimEdge] = Field(default_factory=list)

    def add_claim(self, claim: Claim) -> None:
        if claim.id in self.claims:
            raise ValueError(f"Claim {claim.id} already exists")
        self.claims[claim.id] = claim

    def add_edge(self, edge: ClaimEdge) -> None:
        for cid in (edge.source_claim_id, edge.target_claim_id):
            if cid not in self.claims:
                raise ValueError(f"Edge references unknown claim {cid}")
        self.edges.append(edge)

    def neighbors(self, claim_id: str, edge_type: ClaimEdgeType | None = None) -> list[Claim]:
        out: list[Claim] = []
        for e in self.edges:
            if edge_type is not None and e.edge_type != edge_type:
                continue
            other = None
            if e.source_claim_id == claim_id:
                other = e.target_claim_id
            elif e.target_claim_id == claim_id:
                other = e.source_claim_id
            if other and other in self.claims:
                out.append(self.claims[other])
        return out


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Evidence Ledger
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ


class EvidenceLedger(BaseModel):
    """Append-only ledger of all evidence that has entered Nexus.

    The ledger is content-addressable: each evidence entry has a
    fingerprint that uniquely identifies its content. Two entries with
    the same fingerprint are considered the same evidence (and the
    second registration is a no-op).
    """

    entries: dict[str, Evidence] = Field(default_factory=dict)

    def add(self, evidence: Evidence) -> Evidence:
        """Add evidence to the ledger. Idempotent on fingerprint."""
        for existing in self.entries.values():
            if existing.fingerprint == evidence.fingerprint:
                return existing
        self.entries[evidence.id] = evidence
        return evidence

    def get(self, evidence_id: str) -> Evidence | None:
        return self.entries.get(evidence_id)


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Gate State
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ


class GateName(str, Enum):
    CALIBRATION = "calibration"
    BENCHMARK = "benchmark"
    REPLICATION = "replication"
    ADVERSARIAL = "adversarial"
    SHADOW = "shadow"


class GateResult(BaseModel):
    """Result of evaluating a subject (claim or model) against a gate."""

    model_config = ConfigDict(frozen=True)

    gate: GateName
    subject_id: str
    passed: bool
    metric_value: float
    threshold: float
    reviewer_signature: str | None = Field(
        default=None, description="Human reviewer signature, if reviewed."
    )
    timestamp: float = Field(default_factory=time.time)
    notes: str | None = None


class GateState(BaseModel):
    """Record of all gate evaluations performed on subjects."""

    results: list[GateResult] = Field(default_factory=list)

    def add(self, result: GateResult) -> None:
        self.results.append(result)

    def latest(self, subject_id: str, gate: GateName) -> GateResult | None:
        for r in reversed(self.results):
            if r.subject_id == subject_id and r.gate == gate:
                return r
        return None

    def all_passed(self, subject_id: str) -> bool:
        subject_results = [r for r in self.results if r.subject_id == subject_id]
        if not subject_results:
            return False
        seen_gates = {r.gate for r in subject_results}
        required = {GateName.CALIBRATION, GateName.BENCHMARK}
        if not required.issubset(seen_gates):
            return False
        return all(r.passed for r in subject_results)


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Interpretation ΓÇö the canonical AI output contract
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ


class Interpretation(BaseModel):
    """The canonical output of an AI interpretation request.

    Every AI output in Nexus is structured as an :class:`Interpretation`.
    The ten-layer structure mirrors the output protocol in the Nexus
    constitution: observed evidence, retrieved knowledge, inference,
    explanation, alternatives, contradictions, confidence, uncertainty,
    recommended next step, and final recommendation.

    The structure makes it impossible for an AI module to present an
    unsupported conclusion as fact: the ``inference`` field is always
    accompanied by ``evidence_ids``, ``uncertainty``, and ``confidence``.
    """

    model_config = ConfigDict(frozen=False)  # callers may attach runtime metadata

    answer: str = Field(description="The direct answer to the user's question.")
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="IDs of Evidence entries that support the answer.",
    )
    retrieved_knowledge: list[str] = Field(
        default_factory=list, description="Brief summaries of retrieved knowledge, with citations."
    )
    inference: str = Field(
        default="",
        description="The inference step from evidence + knowledge to answer.",
    )
    explanation: str = Field(default="", description="Mechanistic explanation.")
    alternatives: list[str] = Field(
        default_factory=list, description="Alternative interpretations considered."
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Contradictory evidence or interpretations identified.",
    )
    confidence: Confidence
    uncertainty: Uncertainty | None = None
    recommended_next_step: str | None = None
    citations: list[str] = Field(
        default_factory=list,
        description="Citation strings (DOI, PubMed ID, etc.) for the answer.",
    )
    claim_ids: list[str] = Field(
        default_factory=list,
        description="IDs of Claims registered in the Claim Graph as a result "
        "of this interpretation.",
    )
    provider_model: str | None = Field(
        default=None, description="LLM provider + model that produced this interpretation."
    )


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# Protocol: anything that produces interpretations
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ


class Interprets(Protocol):
    """Protocol for any module that produces an :class:`Interpretation`."""

    def interpret(self, question: str, context: dict[str, Any] | None = None) -> Interpretation: ...


# Re-export the literal type used by some signatures.
__all__ = [
    "Claim",
    "ClaimEdge",
    "ClaimEdgeType",
    "ClaimGraph",
    "Confidence",
    "Evidence",
    "EvidenceClass",
    "EvidenceLedger",
    "GateName",
    "GateResult",
    "GateState",
    "Interpretation",
    "Interprets",
    "Literal",
    "Provenance",
    "Uncertainty",
    "UncertaintyType",
]


# Ensure Literal is exported for downstream type hints that need it.
_ = Literal  # re-export marker
