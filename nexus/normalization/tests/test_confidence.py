"""Unit tests for ConfidenceEngine in nexus.normalization.confidence."""

from __future__ import annotations

import pytest

from nexus.fusion.models import BiomedicalEntity, Evidence
from nexus.normalization.confidence import ConfidenceEngine


def test_confidence_engine_score() -> None:
    engine = ConfidenceEngine()
    entity = BiomedicalEntity(
        id="P04637",
        canonical_id="UNIPROT:P04637",
        canonical_source="uniprot",
        name="TP53",
        cross_references={"ensembl": "ENSG00000141510"},
    )
    entity.add_evidence(Evidence(source="uniprot", source_id="P04637"))

    breakdown = engine.breakdown(entity)
    assert 0.0 <= breakdown.overall <= 1.0
    assert breakdown.evidence > 0.0
    assert breakdown.identity > 0.0


def test_confidence_engine_invalid_weights() -> None:
    with pytest.raises(ValueError, match="confidence weights cannot be negative"):
        ConfidenceEngine(evidence_weight=-1.0)
