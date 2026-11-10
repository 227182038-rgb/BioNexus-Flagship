"""Evidence aggregation and scoring."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Evidence:
    """Represents one scientific evidence item."""

    source: str
    identifier: str
    title: str
    confidence: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)


class EvidenceManager:
    """Collect, deduplicate, and rank scientific evidence."""

    def __init__(self) -> None:
        self._items: list[Evidence] = []
        self._keys: set[tuple[str, str, str]] = set()

    @staticmethod
    def _key(
        evidence: Evidence,
    ) -> tuple[str, str, str]:
        return (
            evidence.source.casefold(),
            evidence.identifier.casefold(),
            evidence.title.casefold(),
        )

    def add(
        self,
        evidence: Evidence,
    ) -> None:
        key = self._key(evidence)

        if key in self._keys:
            return

        self._keys.add(key)
        self._items.append(evidence)

    def extend(
        self,
        items: list[Evidence],
    ) -> None:
        for item in items:
            self.add(item)

    def all(self) -> list[Evidence]:
        return list(self._items)

    def sort(self) -> list[Evidence]:
        return sorted(
            self._items,
            key=lambda x: x.confidence,
            reverse=True,
        )

    def top(
        self,
        n: int = 5,
    ) -> list[Evidence]:
        return self.sort()[:n]

    def clear(self) -> None:
        self._items.clear()
        self._keys.clear()


__all__ = [
    "Evidence",
    "EvidenceManager",
]
