"""Base classes for knowledge source clients.

A :class:`KnowledgeSource` is a tag enum identifying the source
(PubMed, UniProt, etc.). A :class:`KnowledgeClient` is a typed HTTP
client that fetches records from a source and returns
:class:`Document` objects suitable for RAG indexing.
"""

from __future__ import annotations

import abc
import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from nexus.rag.indexing import Document
from nexus.core.exceptions import KnowledgeError


class KnowledgeSource(str, Enum):
    PUBMED = "pubmed"
    NCBI = "ncbi"
    UNIPROT = "uniprot"
    PDB = "pdb"
    ENSEMBL = "ensembl"
    GO = "go"


@dataclass(frozen=True)
class KnowledgeRecord:
    """A single record fetched from a knowledge source."""

    source: KnowledgeSource
    source_id: str
    title: str
    content: str
    uri: str
    metadata: dict[str, Any]

    def to_document(self) -> Document:
        return Document(
            id=f"{self.source.value}_{self.source_id}",
            source_uri=self.uri,
            title=self.title,
            content=self.content,
            metadata={**self.metadata, "source": self.source.value, "source_id": self.source_id},
        )


class KnowledgeClient(abc.ABC):
    """Abstract base class for knowledge source clients.

    Subclasses implement :meth:`fetch` to retrieve a single record by
    ID, and :meth:`search` to find records matching a query. Both
    methods return :class:`KnowledgeRecord` objects that can be
    converted to :class:`Document` objects for indexing.
    """

    source: KnowledgeSource
    base_url: str

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
        rate_limit_per_sec: float = 3.0,
        email: str | None = None,
    ) -> None:
        self.base_url = base_url or self.default_base_url()
        self._timeout = timeout
        self._rate_limit = rate_limit_per_sec
        self._email = email

        self._client: httpx.AsyncClient | None = None
        self._last_request: float = 0.0

    @classmethod
    @abc.abstractmethod
    def default_base_url(cls) -> str: ...

    @abc.abstractmethod
    async def fetch(self, record_id: str) -> KnowledgeRecord:
        """Fetch a single record by ID."""

    @abc.abstractmethod
    async def search(self, query: str, limit: int = 10) -> list[KnowledgeRecord]:
        """Search for records matching ``query``."""

    def _make_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers: dict[str, str] = {}

            if self._email:
                headers["User-Agent"] = f"Nexus-BII/0.1 ({self._email})"
            else:
                headers["User-Agent"] = "Nexus-BII/0.1"

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
                headers=headers,
            )

        return self._client

    async def _throttle(self) -> None:
        """Throttle requests according to the configured rate limit."""
        if self._rate_limit <= 0:
            return

        min_interval = 1.0 / self._rate_limit
        elapsed = time.monotonic() - self._last_request

        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)

        self._last_request = time.monotonic()

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Execute an HTTP request with throttling and retries."""

        self._make_client()

        assert self._client is not None

        retries = 3
        delay = 1.0

        for attempt in range(retries):
            await self._throttle()

            try:
                response = await self._client.request(method, url, **kwargs)
                response.raise_for_status()
                return response

            except (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.NetworkError,
            ):
                if attempt == retries - 1:
                    raise KnowledgeError("Network request failed.")

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code

                if status in (429, 500, 502, 503, 504):
                    if attempt == retries - 1:
                        raise KnowledgeError(f"HTTP {status}: request failed after retries.")
                else:
                    raise KnowledgeError(f"HTTP {status}: {exc.response.text}") from exc

            await asyncio.sleep(delay)
            delay *= 2

        raise KnowledgeError("Unexpected request failure.")

    async def __aenter__(self) -> KnowledgeClient:
        self._make_client()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


__all__ = [
    "KnowledgeClient",
    "KnowledgeError",
    "KnowledgeRecord",
    "KnowledgeSource",
]
