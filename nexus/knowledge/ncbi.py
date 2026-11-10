"""Generic NCBI client (covers databases other than PubMed, e.g. nuccore, protein)."""

from __future__ import annotations

from nexus.knowledge.base import KnowledgeClient, KnowledgeError, KnowledgeRecord, KnowledgeSource
from nexus.core.exceptions import NCBIError


class NCBIClient(KnowledgeClient):
    source = KnowledgeSource.NCBI

    @classmethod
    def default_base_url(cls) -> str:
        return "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
        rate_limit_per_sec: float = 3.0,
        email: str | None = None,
        db: str = "nuccore",
    ) -> None:
        super().__init__(base_url, timeout, rate_limit_per_sec, email)
        self._db = db

    async def fetch(self, record_id: str) -> KnowledgeRecord:
        try:
            response = await self._request(
                "GET",
                "/efetch.fcgi",
                params={
                    "db": self._db,
                    "id": record_id,
                    "rettype": "fasta",
                    "retmode": "text",
                },
            )
        except Exception as exc:
            raise NCBIError(
                f"NCBI fetch failed for {record_id}: {exc}"
            ) from exc

        content = response.text
        lines = content.splitlines()
        title = lines[0][1:] if lines and lines[0].startswith(">") else record_id

        return KnowledgeRecord(
            source=self.source,
            source_id=record_id,
            title=title,
            content=content,
            uri=f"https://www.ncbi.nlm.nih.gov/{self._db}/{record_id}",
            metadata={
                "db": self._db,
                "format": "fasta",
            },
        )

    async def search(self, query: str, limit: int = 10) -> list[KnowledgeRecord]:
        try:
            response = await self._request(
                "GET",
                "/esearch.fcgi",
                params={
                    "db": self._db,
                    "term": query,
                    "retmax": str(limit),
                    "retmode": "json",
                },
            )

            ids = response.json().get(
                "esearchresult",
                {},
            ).get(
                "idlist",
                [],
            )

        except Exception as exc:
            raise NCBIError(
                f"NCBI search failed for {query!r}: {exc}"
            ) from exc

        records: list[KnowledgeRecord] = []

        for rid in ids[:limit]:
            try:
                records.append(await self.fetch(rid))
            except KnowledgeError:
                continue

        return records


__all__ = ["NCBIClient"]
