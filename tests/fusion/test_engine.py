"""Unit tests for FusionEngine."""

from __future__ import annotations

import pytest

from nexus.fusion.engine import FusionEngine
from nexus.fusion.models import BiomedicalEntity
from nexus.knowledge.base import KnowledgeClient, KnowledgeRecord, KnowledgeSource


class DummyClient(KnowledgeClient):
    def __init__(self, source: KnowledgeSource, records: dict[str, KnowledgeRecord] | None = None) -> None:
        super().__init__(base_url="https://example.com")
        self.source = source
        self._records = records or {}

    @classmethod
    def default_base_url(cls) -> str:
        return "https://example.com"

    async def fetch(self, record_id: str) -> KnowledgeRecord:
        if record_id in self._records:
            return self._records[record_id]
        return KnowledgeRecord(
            source=self.source,
            source_id=record_id,
            title=f"Title {record_id}",
            content=f"Content {record_id}",
            uri=f"https://example.com/{record_id}",
            metadata={"id": record_id},
        )

    async def search(self, query: str, limit: int = 10) -> list[KnowledgeRecord]:
        return list(self._records.values())[:limit]

    async def search_gene_symbol(self, query: str, limit: int = 10, reviewed_only: bool = True) -> list[KnowledgeRecord]:
        return list(self._records.values())[:limit]


@pytest.mark.asyncio
async def test_fusion_engine_search_gene_symbol() -> None:
    uniprot_rec = KnowledgeRecord(
        source=KnowledgeSource.UNIPROT,
        source_id="P04637",
        title="TP53 Human",
        content="Cellular tumor antigen p53",
        uri="https://uniprot.org/uniprot/P04637",
        metadata={"accession": "P04637", "gene_symbol": "TP53", "cross_references": {"pdb": "1A1U"}},
    )
    uniprot_client = DummyClient(KnowledgeSource.UNIPROT, {"P04637": uniprot_rec})
    pdb_rec = KnowledgeRecord(
        source=KnowledgeSource.PDB,
        source_id="1A1U",
        title="p53 core domain",
        content="Crystal structure",
        uri="https://rcsb.org/structure/1A1U",
        metadata={"pdb_id": "1A1U", "method": "X-RAY"},
    )
    pdb_client = DummyClient(KnowledgeSource.PDB, {"1A1U": pdb_rec})

    engine = FusionEngine(clients=[uniprot_client, pdb_client])
    entities = await engine.search("TP53")

    assert len(entities) >= 1
    p53_entity = next((e for e in entities if "P04637" in e.proteins or e.id == "P04637"), None)
    assert p53_entity is not None


@pytest.mark.asyncio
async def test_fusion_engine_fetch_exact_identifier() -> None:
    ensembl_rec = KnowledgeRecord(
        source=KnowledgeSource.ENSEMBL,
        source_id="ENSG00000141510",
        title="TP53 Ensembl Gene",
        content="Human TP53 gene",
        uri="https://ensembl.org/id/ENSG00000141510",
        metadata={"ensembl_id": "ENSG00000141510", "symbol": "TP53"},
    )
    ensembl_client = DummyClient(KnowledgeSource.ENSEMBL, {"ENSG00000141510": ensembl_rec})

    engine = FusionEngine(clients=[ensembl_client])
    entities = await engine.fetch("ensembl:ENSG00000141510")

    assert len(entities) == 1
    assert entities[0].name == "TP53"


def test_fusion_engine_extract_identifier() -> None:
    from nexus.knowledge.router import QueryRouter

    router = QueryRouter()
    route_pubmed = router.route("pmid:12345")
    assert FusionEngine._extract_identifier(route_pubmed, KnowledgeSource.PUBMED) == "12345"

    route_uniprot = router.route("uniprot:P04637")
    assert FusionEngine._extract_identifier(route_uniprot, KnowledgeSource.UNIPROT) == "P04637"

    route_ensembl = router.route("ensembl:ENSG00000141510")
    assert FusionEngine._extract_identifier(route_ensembl, KnowledgeSource.ENSEMBL) == "ENSG00000141510"

    route_pdb = router.route("pdb:1A1U")
    assert FusionEngine._extract_identifier(route_pdb, KnowledgeSource.PDB) == "1A1U"

    route_go = router.route("go:0006915")
    assert FusionEngine._extract_identifier(route_go, KnowledgeSource.GO) == "GO:0006915"
