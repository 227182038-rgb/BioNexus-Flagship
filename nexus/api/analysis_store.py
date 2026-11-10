"""In-memory store for reusable BioNexus analysis contexts."""

from __future__ import annotations

import threading
import uuid

from nexus.pipeline.context import ScientificAnalysisContext


class AnalysisStore:
    """Thread-safe in-memory store for recent scientific analyses."""

    def __init__(self) -> None:
        self._items: dict[str, ScientificAnalysisContext] = {}
        self._lock = threading.RLock()

    def create(
        self,
        context: ScientificAnalysisContext,
    ) -> str:
        """Store a context and return a stable analysis identifier."""

        analysis_id = (
            f"ana_{uuid.uuid4().hex}"
        )

        with self._lock:
            self._items[analysis_id] = context

        return analysis_id

    def get(
        self,
        analysis_id: str,
    ) -> ScientificAnalysisContext | None:
        """Retrieve an analysis context."""

        with self._lock:
            return self._items.get(
                analysis_id
            )

    def delete(
        self,
        analysis_id: str,
    ) -> bool:
        """Delete one analysis context."""

        with self._lock:
            return (
                self._items.pop(
                    analysis_id,
                    None,
                )
                is not None
            )

    def clear(self) -> None:
        """Clear all stored contexts."""

        with self._lock:
            self._items.clear()


__all__ = [
    "AnalysisStore",
]
