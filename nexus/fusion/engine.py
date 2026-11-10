"""Biomedical knowledge fusion engine."""

from __future__ import annotations

from nexus.fusion.models import BiomedicalEntity
from nexus.fusion.resolver import EntityResolver
from nexus.knowledge.base import KnowledgeClient, KnowledgeSource
from nexus.knowledge.router import (
    QueryIntent,
    QueryRoute,
    QueryRouter,
)
from nexus.normalization.resolver import NormalizationResolver
from nexus.core.logging import get_logger, get_correlation_id, set_correlation_id


class FusionEngine:
    """Coordinates routed retrieval, fusion, and normalization."""

    def __init__(
        self,
        clients: list[KnowledgeClient],
        normalization: NormalizationResolver | None = None,
        router: QueryRouter | None = None,
    ) -> None:
        self.logger = get_logger("fusion")
        self._clients = {
            client.source: client
            for client in clients
        }

        self._resolver = EntityResolver()

        self._normalization = (
            normalization
            or NormalizationResolver()
        )

        self._router = (
            router
            or QueryRouter()
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
        correlation_id: str | None = None,
    ) -> list[BiomedicalEntity]:
        """Route a biomedical query and fuse relevant sources."""
        cid = correlation_id or get_correlation_id()
        set_correlation_id(cid)

        self.logger.info("[%s] Fusion searching query: %s", cid, query)

        route = self._router.route(query)

        records = []

        for plan in route.plans:
            client = self._clients.get(
                plan.source
            )

            if client is None:
                continue

            retrieval_limit = min(
                plan.limit,
                limit,
            )

            try:
                if (
                    route.intent == QueryIntent.GENE_SYMBOL
                    and plan.source == KnowledgeSource.UNIPROT
                    and hasattr(
                        client,
                        "search_gene_symbol",
                    )
                ):
                    results = await client.search_gene_symbol(
                        route.normalized,
                        limit=retrieval_limit,
                        reviewed_only=True,
                    )

                elif plan.exact:
                    record = await self._exact_fetch(
                        client,
                        route,
                    )

                    results = (
                        [record]
                        if record is not None
                        else []
                    )

                else:
                    results = await client.search(
                        route.normalized,
                        limit=retrieval_limit,
                    )

                records.extend(results)

            except Exception as exc:
                self.logger.warning(
                    "Fusion retrieval error from source %s [%s]: %s",
                    plan.source,
                    cid,
                    exc,
                )
                continue

        records.extend(
            await self._fetch_authoritative_structure_records(
                records,
                limit=5,
                correlation_id=cid,
            )
        )

        entities = self._resolver.resolve(
            records
        )

        return self._normalization.resolve(
            entities,
            correlation_id=cid,
        )

    async def _fetch_authoritative_structure_records(
        self,
        records,
        limit: int = 5,
        correlation_id: str | None = None,
    ):
        """Fetch PDB structures referenced by authoritative UniProt records."""
        cid = correlation_id or get_correlation_id()

        pdb_client = self._clients.get(
            KnowledgeSource.PDB
        )

        if pdb_client is None:
            return []

        identifiers: list[str] = []
        seen: set[str] = set()

        for record in records:
            if record.source != KnowledgeSource.UNIPROT:
                continue

            cross_references = record.metadata.get(
                "cross_references",
                {},
            )

            if not isinstance(cross_references, dict):
                continue

            pdb_id = cross_references.get(
                "pdb"
            )

            if not pdb_id:
                continue

            pdb_id = str(pdb_id).strip().upper()

            if not pdb_id or pdb_id in seen:
                continue

            seen.add(pdb_id)
            identifiers.append(pdb_id)

            if len(identifiers) >= limit:
                break

        results = []

        for pdb_id in identifiers:
            try:
                results.append(
                    await pdb_client.fetch(
                        pdb_id
                    )
                )
            except Exception as exc:
                self.logger.warning(
                    "Authoritative structure fetch failed for %s [%s]: %s",
                    pdb_id,
                    cid,
                    exc,
                )
                continue

        return results

    async def fetch(
        self,
        identifier: str,
        correlation_id: str | None = None,
    ) -> list[BiomedicalEntity]:
        """Resolve an exact biomedical identifier."""
        cid = correlation_id or get_correlation_id()
        set_correlation_id(cid)

        route = self._router.route(
            identifier
        )

        records = []

        for plan in route.plans:
            client = self._clients.get(
                plan.source
            )

            if client is None:
                continue

            try:
                record = await client.fetch(
                    self._extract_identifier(
                        route,
                        plan.source,
                    )
                )

                records.append(record)

            except Exception as exc:
                self.logger.warning(
                    "Exact fetch failed for source %s [%s]: %s",
                    plan.source,
                    cid,
                    exc,
                )
                continue

        records.extend(
            await self._fetch_authoritative_structure_records(
                records,
                limit=5,
                correlation_id=cid,
            )
        )

        entities = self._resolver.resolve(
            records
        )

        return self._normalization.resolve(
            entities,
            correlation_id=cid,
        )

    async def _exact_fetch(
        self,
        client: KnowledgeClient,
        route: QueryRoute,
    ):
        """Fetch an exact identifier."""
        return await client.fetch(
            self._extract_identifier(
                route,
                client.source,
            )
        )

    @staticmethod
    def _extract_identifier(
        route: QueryRoute,
        source: KnowledgeSource,
    ) -> str:
        """Extract a source-specific identifier."""

        value = route.normalized

        if source == KnowledgeSource.PUBMED:
            if value.lower().startswith("pmid:"):
                return value.split(
                    ":",
                    1,
                )[1].strip()

            if (
                "pubmed.ncbi.nlm.nih.gov/"
                in value.lower()
            ):
                return (
                    value.rstrip("/")
                    .split("/")[-1]
                )

            return value

        if source == KnowledgeSource.UNIPROT:
            if "/uniprot/" in value.lower():
                return (
                    value.rstrip("/")
                    .split("/")[-1]
                )

            return value.removeprefix(
                "uniprot:"
            ).strip()

        if source == KnowledgeSource.ENSEMBL:
            if "/id/" in value.lower():
                return (
                    value.rstrip("/")
                    .split("/")[-1]
                )

            return value.removeprefix(
                "ensembl:"
            ).strip()

        if source == KnowledgeSource.PDB:
            if "/structure/" in value.lower():
                return (
                    value.rstrip("/")
                    .split("/")[-1]
                )

            return (
                value.removeprefix("pdb:")
                .strip()
                .upper()
            )

        if source == KnowledgeSource.GO:
            if "/term/" in value.lower():
                value = (
                    value.rstrip("/")
                    .split("/")[-1]
                )

            value = value.removeprefix(
                "go:"
            ).strip()

            if not value.upper().startswith("GO:"):
                value = f"GO:{value}"

            return value.upper()

        if source == KnowledgeSource.NCBI:
            return value.removeprefix(
                "ncbi:"
            ).strip()

        return value


__all__ = [
    "FusionEngine",
]

