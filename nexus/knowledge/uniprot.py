"""UniProt client."""

from __future__ import annotations

from typing import Any

from nexus.knowledge.base import (
    KnowledgeClient,
    KnowledgeError,
    KnowledgeRecord,
    KnowledgeSource,
)
from nexus.core.exceptions import UniProtError


class UniProtClient(KnowledgeClient):
    """Client for the UniProtKB REST API."""

    source = KnowledgeSource.UNIPROT

    @classmethod
    def default_base_url(cls) -> str:
        return "https://rest.uniprot.org"

    async def fetch(
        self,
        record_id: str,
    ) -> KnowledgeRecord:
        """Fetch a single UniProtKB entry."""
        record_id = record_id.strip()

        if not record_id:
            raise ValueError("record_id cannot be empty")

        try:
            response = await self._request(
                "GET",
                f"/uniprotkb/{record_id}.json",
            )

            data: dict[str, Any] = response.json()

        except Exception as exc:
            raise UniProtError(
                f"UniProt fetch failed for {record_id}: {exc}"
            ) from exc

        return self._parse(
            data,
            record_id,
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[KnowledgeRecord]:
        """Perform a generic UniProt full-text search."""
        query = query.strip()

        if not query:
            return []

        return await self._search(
            query,
            limit=limit,
        )

    async def search_gene_symbol(
        self,
        symbol: str,
        limit: int = 5,
        reviewed_only: bool = True,
    ) -> list[KnowledgeRecord]:
        """Search for proteins associated with a human gene symbol."""
        symbol = symbol.strip()

        if not symbol:
            return []

        clauses = [
            f"gene:{symbol}",
            "organism_id:9606",
        ]

        if reviewed_only:
            clauses.append("reviewed:true")

        query = " AND ".join(
            f"({clause})"
            for clause in clauses
        )

        return await self._search(
            query,
            limit=limit,
        )

    async def _search(
        self,
        query: str,
        limit: int,
    ) -> list[KnowledgeRecord]:
        """Execute a UniProtKB search query."""
        try:
            response = await self._request(
                "GET",
                "/uniprotkb/search",
                params={
                    "query": query,
                    "format": "json",
                    "size": str(limit),
                },
            )

            data = response.json()

        except Exception as exc:
            raise UniProtError(
                f"UniProt search failed for {query!r}: {exc}"
            ) from exc

        results = data.get("results", [])

        records: list[KnowledgeRecord] = []

        for entry in results[:limit]:
            accession = entry.get(
                "primaryAccession",
                "",
            )

            if not accession:
                continue

            records.append(
                self._parse(
                    entry,
                    accession,
                )
            )

        return records

    def _parse(
        self,
        data: dict[str, Any],
        accession: str,
    ) -> KnowledgeRecord:
        """Convert a UniProt JSON record into a KnowledgeRecord."""

        title = (
            data.get("proteinDescription", {})
            .get("recommendedName", {})
            .get("fullName", {})
            .get("value", accession)
        )

        organism = (
            data.get("organism", {})
            .get("scientificName", "unknown organism")
        )

        comments = data.get(
            "comments",
            [],
        )

        function_text = ""

        for comment in comments:
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get(
                    "texts",
                    [],
                )

                function_text = " ".join(
                    text.get("value", "")
                    for text in texts
                )

                break

        genes = data.get("genes", [])

        gene_symbols: list[str] = []
        gene_synonyms: list[str] = []

        for gene in genes:
            gene_name = gene.get("geneName", {})
            value = gene_name.get("value")

            if value:
                gene_symbols.append(str(value))

            for synonym in gene.get("synonyms", []):
                synonym_value = synonym.get("value")

                if synonym_value:
                    gene_synonyms.append(
                        str(synonym_value)
                    )

        cross_references: dict[str, str] = {}

        for reference in data.get(
            "uniProtKBCrossReferences",
            [],
        ):
            database = reference.get("database")
            identifier = reference.get("id")

            if not database or not identifier:
                continue

            key = str(database).lower()

            # Preserve the first identifier for a database as the primary
            # compact cross-reference. Full source payload remains available
            # in metadata if needed later.
            cross_references.setdefault(
                key,
                str(identifier),
            )

        sequence = data.get(
            "sequence",
            {},
        )

        content = (
            f"UniProt entry {accession}: "
            f"{title} ({organism}).\n"
            f"Function: "
            f"{function_text or 'no function annotation available.'}"
        )

        return KnowledgeRecord(
            source=self.source,
            source_id=accession,
            title=f"{title} ({organism})",
            content=content,
            uri=(
                "https://www.uniprot.org/uniprot/"
                f"{accession}"
            ),
            metadata={
                "accession": accession,
                "organism": organism,
                "sequence_length": sequence.get(
                    "length",
                    0,
                ),
                "entry_type": data.get(
                    "entryType",
                    "",
                ),
                "gene_symbols": gene_symbols,
                "gene_symbol": (
                    gene_symbols[0]
                    if gene_symbols
                    else ""
                ),
                "gene_synonyms": gene_synonyms,
                "cross_references": cross_references,
            },
        )


__all__ = [
    "UniProtClient",
]
