"""Gene Ontology (GO) client (via EMBL-EBI QuickGO)."""

from __future__ import annotations

from nexus.knowledge.base import KnowledgeClient, KnowledgeError, KnowledgeRecord, KnowledgeSource
from nexus.core.exceptions import GOError


class GOClient(KnowledgeClient):
    source = KnowledgeSource.GO

    @classmethod
    def default_base_url(cls) -> str:
        return "https://www.ebi.ac.uk/QuickGO"

    async def fetch(self, record_id: str) -> KnowledgeRecord:
        go_id = record_id.upper()

        if not go_id.startswith("GO:"):
            go_id = f"GO:{go_id}"

        try:
            response = await self._request(
                "GET",
                f"/services/ontology/terms/{go_id}",
                headers={"Accept": "application/json"},
            )

            data = response.json()
            results = data.get("results", [])

            if not results:
                raise GOError(f"GO term {go_id} not found")

            term = results[0]

        except (KnowledgeError, GOError):
            raise

        except Exception as exc:
            raise GOError(
                f"GO fetch failed for {go_id}: {exc}"
            ) from exc

        name = term.get("name", go_id)
        namespace = term.get("aspect", "unknown aspect")
        definition = term.get("definition", "no definition available")
        is_obsolete = term.get("isObsolete", False)

        content = (
            f"Gene Ontology term {go_id}: {name}.\n"
            f"Aspect: {namespace}; obsolete: {is_obsolete}.\n"
            f"Definition: {definition}"
        )

        return KnowledgeRecord(
            source=self.source,
            source_id=go_id,
            title=f"{name} ({namespace})",
            content=content,
            uri=f"http://amigo.geneontology.org/amigo/term/{go_id}",
            metadata={
                "go_id": go_id,
                "name": name,
                "aspect": namespace,
                "is_obsolete": is_obsolete,
            },
        )

    async def search(self, query: str, limit: int = 10) -> list[KnowledgeRecord]:
        try:
            response = await self._request(
                "GET",
                "/services/ontology/search",
                params={
                    "query": query,
                    "limit": str(limit),
                    "page": "1",
                },
                headers={"Accept": "application/json"},
            )

            data = response.json()
            results = data.get("results", [])

        except Exception:
            return []

        records: list[KnowledgeRecord] = []

        for term in results[:limit]:
            go_id = term.get("id", "")

            if not go_id:
                continue

            try:
                records.append(await self.fetch(go_id))
            except KnowledgeError:
                continue

        return records


__all__ = ["GOClient"]
