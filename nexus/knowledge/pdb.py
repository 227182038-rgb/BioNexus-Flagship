"""Protein Data Bank (PDB) client (via RCSB PDB REST API)."""

from __future__ import annotations

from typing import Any

from nexus.knowledge.base import KnowledgeClient, KnowledgeError, KnowledgeRecord, KnowledgeSource
from nexus.core.exceptions import PDBError


class PDBClient(KnowledgeClient):
    source = KnowledgeSource.PDB

    @classmethod
    def default_base_url(cls) -> str:
        return "https://data.rcsb.org/rest/v1"

    async def fetch(self, record_id: str) -> KnowledgeRecord:
        pdb_id = record_id.upper()

        try:
            response = await self._request(
                "GET",
                f"/core/entry/{pdb_id}",
            )
            data: dict[str, Any] = response.json()

        except Exception as exc:
            raise PDBError(
                f"PDB fetch failed for {pdb_id}: {exc}"
            ) from exc

        title = data.get("struct", {}).get("title", pdb_id)
        method = data.get("exptl", [{}])[0].get("method", "unknown method")

        resolution = data.get("rcsb_entry_info", {}).get(
            "resolution_combined",
            [None],
        )
        resolution_str = (
            f"{resolution[0]:.2f} Å"
            if resolution and resolution[0]
            else "N/A"
        )

        organisms: list[str] = []
        source_orgs = data.get("rcsb_entity_source_organism", [])
        if isinstance(source_orgs, list):
            for org in source_orgs:
                if isinstance(org, dict):
                    name = org.get("ncbi_scientific_name") or org.get("scientific_name")
                    if name and name not in organisms:
                        organisms.append(str(name))
        if not organisms:
            host_orgs = data.get("rcsb_host_organism", [])
            if isinstance(host_orgs, list):
                for org in host_orgs:
                    if isinstance(org, dict):
                        name = org.get("ncbi_scientific_name") or org.get("scientific_name")
                        if name and name not in organisms:
                            organisms.append(str(name))

        organism = ", ".join(organisms) if organisms else "unknown"

        deposition_date = data.get(
            "rcsb_accession_info",
            {},
        ).get(
            "initial_deposition_date",
            "unknown",
        )

        content = (
            f"PDB entry {pdb_id}: {title}.\n"
            f"Method: {method}; resolution: {resolution_str}.\n"
            f"Deposited: {deposition_date}."
        )

        return KnowledgeRecord(
            source=self.source,
            source_id=pdb_id,
            title=title,
            content=content,
            uri=f"https://www.rcsb.org/structure/{pdb_id}",
            metadata={
                "pdb_id": pdb_id,
                "method": method,
                "resolution": resolution_str,
                "deposition_date": deposition_date,
                "organism": organism,
            },
        )

    async def search(self, query: str, limit: int = 10) -> list[KnowledgeRecord]:
        try:
            response = await self._request(
                "POST",
                "https://search.rcsb.org/rcsbsearch/v2/query",
                json={
                    "query": {
                        "type": "terminal",
                        "service": "full_text",
                        "parameters": {
                            "value": query,
                        },
                    },
                    "return_type": "entry",
                    "request_options": {
                        "paginated": False,
                        "return_counts": False,
                        "scoring_strategy": "sequence",
                        "results_content_type": [
                            "experimental",
                        ],
                    },
                },
                params={
                    "return_type": "entry",
                },
            )

            data = response.json()
            ids = [
                r.get("identifier", "")
                for r in data.get("result_set", [])
            ]

        except Exception:
            return []

        records: list[KnowledgeRecord] = []

        for pdb_id in ids[:limit]:
            try:
                records.append(await self.fetch(pdb_id))
            except KnowledgeError:
                continue

        return records


__all__ = ["PDBClient"]
