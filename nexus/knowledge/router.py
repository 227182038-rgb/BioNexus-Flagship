"""Biomedical query classification and knowledge-source routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re

from nexus.knowledge.base import KnowledgeSource


class QueryIntent(StrEnum):
    """High-level biomedical query intents."""

    GENE_SYMBOL = "gene_symbol"
    UNIPROT_ID = "uniprot_id"
    ENSEMBL_ID = "ensembl_id"
    PUBMED_ID = "pubmed_id"
    PDB_ID = "pdb_id"
    GO_ID = "go_id"
    NCBI_ID = "ncbi_id"
    FREE_TEXT = "free_text"


@dataclass(frozen=True, slots=True)
class SourcePlan:
    """Retrieval plan for one knowledge source."""

    source: KnowledgeSource
    priority: int
    limit: int
    exact: bool = False


@dataclass(frozen=True, slots=True)
class QueryRoute:
    """Classification and retrieval plan for a biomedical query."""

    original: str
    normalized: str
    intent: QueryIntent
    plans: tuple[SourcePlan, ...]


class QueryRouter:
    """Classify biomedical queries and route them to relevant sources."""

    _UNIPROT = re.compile(
        r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]{5})$",
        re.IGNORECASE,
    )

    _ENSEMBL = re.compile(
        r"^ENS[A-Z]*[0-9]{6,}$",
        re.IGNORECASE,
    )

    _PUBMED = re.compile(
        r"^(?:PMID[:\s]*)?([0-9]{1,9})$",
        re.IGNORECASE,
    )

    _GO = re.compile(
        r"^GO[:\s]?[0-9]{7}$",
        re.IGNORECASE,
    )

    _PDB = re.compile(
        r"^(?:PDB[:\s]*)?[0-9A-Z]{4}$",
        re.IGNORECASE,
    )

    _NCBI_ACCESSION = re.compile(
        r"^(?:[A-Z]{1,4}_)?[0-9]+(?:\.[0-9]+)?$",
        re.IGNORECASE,
    )

    _GENE_SYMBOL = re.compile(
        r"^[A-Z][A-Z0-9-]{1,14}$"
    )

    def route(
        self,
        query: str,
    ) -> QueryRoute:
        """Classify a query and produce a deterministic source plan."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        original = query
        normalized = query.strip()

        if not normalized:
            raise ValueError("query cannot be empty")

        intent = self.classify(normalized)
        plans = self._plans_for(
            intent,
            normalized,
        )

        return QueryRoute(
            original=original,
            normalized=normalized,
            intent=intent,
            plans=plans,
        )

    def classify(
        self,
        query: str,
    ) -> QueryIntent:
        """Classify a biomedical query without performing retrieval."""

        value = query.strip()

        if self._is_pubmed_url(value):
            return QueryIntent.PUBMED_ID

        if self._is_uniprot_url(value):
            return QueryIntent.UNIPROT_ID

        if self._is_pdb_url(value):
            return QueryIntent.PDB_ID

        if self._is_ensembl_url(value):
            return QueryIntent.ENSEMBL_ID

        if self._is_go_url(value):
            return QueryIntent.GO_ID

        if self._GO.fullmatch(value):
            return QueryIntent.GO_ID

        if self._ENSEMBL.fullmatch(value):
            return QueryIntent.ENSEMBL_ID

        if self._UNIPROT.fullmatch(value):
            return QueryIntent.UNIPROT_ID

        # Explicit PDB identifiers take precedence over ambiguous
        # four-character biological symbols.
        if value.upper().startswith("PDB:"):
            return QueryIntent.PDB_ID

        # Gene symbols are checked before the generic four-character
        # PDB pattern. This prevents symbols such as TP53 from being
        # misclassified as PDB IDs.
        if self._GENE_SYMBOL.fullmatch(value):
            return QueryIntent.GENE_SYMBOL

        # Four-character PDB identifiers such as 1TUP or 4HHB.
        # Requiring a leading digit avoids treating TP53/EGFR/etc.
        # as structures.
        if re.fullmatch(r"^[0-9][A-Z0-9]{3}$", value, re.IGNORECASE):
            return QueryIntent.PDB_ID

        if self._PUBMED.fullmatch(value):
            return QueryIntent.PUBMED_ID

        if self._NCBI_ACCESSION.fullmatch(value):
            return QueryIntent.NCBI_ID

        return QueryIntent.FREE_TEXT

    def _plans_for(
        self,
        intent: QueryIntent,
        query: str,
    ) -> tuple[SourcePlan, ...]:
        """Build a source plan appropriate to the query intent."""

        if intent == QueryIntent.UNIPROT_ID:
            return (
                SourcePlan(
                    KnowledgeSource.UNIPROT,
                    priority=100,
                    limit=1,
                    exact=True,
                ),
            )

        if intent == QueryIntent.ENSEMBL_ID:
            return (
                SourcePlan(
                    KnowledgeSource.ENSEMBL,
                    priority=100,
                    limit=1,
                    exact=True,
                ),
            )

        if intent == QueryIntent.PUBMED_ID:
            return (
                SourcePlan(
                    KnowledgeSource.PUBMED,
                    priority=100,
                    limit=1,
                    exact=True,
                ),
            )

        if intent == QueryIntent.PDB_ID:
            return (
                SourcePlan(
                    KnowledgeSource.PDB,
                    priority=100,
                    limit=1,
                    exact=True,
                ),
            )

        if intent == QueryIntent.GO_ID:
            return (
                SourcePlan(
                    KnowledgeSource.GO,
                    priority=100,
                    limit=1,
                    exact=True,
                ),
            )

        if intent == QueryIntent.NCBI_ID:
            return (
                SourcePlan(
                    KnowledgeSource.NCBI,
                    priority=100,
                    limit=1,
                    exact=True,
                ),
            )

        if intent == QueryIntent.GENE_SYMBOL:
            return (
                SourcePlan(
                    KnowledgeSource.ENSEMBL,
                    priority=100,
                    limit=5,
                ),
                SourcePlan(
                    KnowledgeSource.UNIPROT,
                    priority=90,
                    limit=5,
                ),
                SourcePlan(
                    KnowledgeSource.PUBMED,
                    priority=80,
                    limit=10,
                ),
                SourcePlan(
                    KnowledgeSource.PDB,
                    priority=60,
                    limit=5,
                ),
            )

        # Free-text biomedical research query.
        return (
            SourcePlan(
                KnowledgeSource.PUBMED,
                priority=100,
                limit=10,
            ),
            SourcePlan(
                KnowledgeSource.UNIPROT,
                priority=80,
                limit=5,
            ),
            SourcePlan(
                KnowledgeSource.ENSEMBL,
                priority=70,
                limit=5,
            ),
        )

    @staticmethod
    def _is_pubmed_url(value: str) -> bool:
        return bool(
            re.fullmatch(
                r"https?://(?:www\.)?pubmed\.ncbi\.nlm\.nih\.gov/[0-9]+/?",
                value,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_uniprot_url(value: str) -> bool:
        return bool(
            re.fullmatch(
                r"https?://(?:www\.)?uniprot\.org/(?:uniprot/)?[A-Z0-9]+/?",
                value,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_pdb_url(value: str) -> bool:
        return bool(
            re.fullmatch(
                r"https?://(?:www\.)?rcsb\.org/structure/[A-Z0-9]{4}/?",
                value,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_ensembl_url(value: str) -> bool:
        return bool(
            re.fullmatch(
                r"https?://(?:www\.)?ensembl\.org/id/ENS[A-Z0-9]+/?",
                value,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_go_url(value: str) -> bool:
        return bool(
            re.fullmatch(
                r"https?://(?:www\.)?(?:amigo\.)?geneontology\.org/.*/GO:[0-9]{7}/?",
                value,
                re.IGNORECASE,
            )
        )


__all__ = [
    "QueryIntent",
    "QueryRoute",
    "QueryRouter",
    "SourcePlan",
]
