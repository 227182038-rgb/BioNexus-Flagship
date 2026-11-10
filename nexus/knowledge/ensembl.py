"""Ensembl REST client."""

from __future__ import annotations

from typing import Any

from nexus.knowledge.base import (
    KnowledgeClient,
    KnowledgeError,
    KnowledgeRecord,
    KnowledgeSource,
)
from nexus.core.exceptions import EnsemblError


class EnsemblClient(KnowledgeClient):
    """Client for the Ensembl REST API."""

    source = KnowledgeSource.ENSEMBL

    @classmethod
    def default_base_url(cls) -> str:
        return "https://rest.ensembl.org"

    async def fetch(
        self,
        record_id: str,
    ) -> KnowledgeRecord:
        """Fetch a gene by Ensembl ID."""

        record_id = record_id.strip()

        if not record_id:
            raise ValueError("record_id cannot be empty")

        try:
            response = await self._request(
                "GET",
                f"/lookup/id/{record_id}",
                headers={
                    "Accept": "application/json",
                },
            )

            data: dict[str, Any] = response.json()

        except Exception as exc:
            raise EnsemblError(
                f"Ensembl fetch failed for {record_id}: {exc}"
            ) from exc

        symbol = str(
            data.get(
                "display_name",
                record_id,
            )
        )

        description = str(
            data.get(
                "description",
                "no description available",
            )
        )

        species = str(
            data.get(
                "species",
                "unknown species",
            )
        )

        biotype = str(
            data.get(
                "biotype",
                "unknown biotype",
            )
        )

        chromosome = str(
            data.get(
                "seq_region_name",
                "unknown chromosome",
            )
        )

        start = data.get(
            "start",
            0,
        )

        end = data.get(
            "end",
            0,
        )

        content = (
            f"Ensembl gene {record_id}: "
            f"{symbol} ({species}).\n"
            f"Biotype: {biotype}; "
            f"location: {chromosome}:{start}-{end}.\n"
            f"Description: {description}."
        )

        return KnowledgeRecord(
            source=self.source,
            source_id=record_id,
            title=f"{symbol} ({species})",
            content=content,
            uri=(
                "https://www.ensembl.org/id/"
                f"{record_id}"
            ),
            metadata={
                "ensembl_id": record_id,
                "symbol": symbol,
                "species": species,
                "biotype": biotype,
                "chromosome": chromosome,
                "start": start,
                "end": end,
            },
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[KnowledgeRecord]:
        """Resolve a human gene symbol through Ensembl."""

        query = query.strip()

        if not query:
            return []

        # Primary strategy: direct human gene-symbol lookup.
        try:
            response = await self._request(
                "GET",
                f"/lookup/symbol/homo_sapiens/{query}",
                headers={
                    "Accept": "application/json",
                },
            )

            data: dict[str, Any] = response.json()

            ensembl_id = data.get("id")

            if ensembl_id:
                try:
                    return [
                        await self.fetch(
                            str(ensembl_id)
                        )
                    ]
                except KnowledgeError:
                    pass

        except Exception:
            pass

        # Fallback: xrefs endpoint with the symbol in the URL path.
        try:
            response = await self._request(
                "GET",
                f"/xrefs/symbol/homo_sapiens/{query}",
                headers={
                    "Accept": "application/json",
                },
            )

            data = response.json()

        except Exception:
            return []

        ids = [
            str(item["id"])
            for item in data
            if item.get("type") == "gene"
            and item.get("id")
        ]

        records: list[KnowledgeRecord] = []

        for ensembl_id in ids[:limit]:
            try:
                records.append(
                    await self.fetch(
                        ensembl_id
                    )
                )
            except KnowledgeError:
                continue

        return records


__all__ = [
    "EnsemblClient",
]
