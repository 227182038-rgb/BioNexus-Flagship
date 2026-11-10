"""Biomedical alias resolution.

Provides deterministic, network-free resolution of biomedical aliases to
canonical identifiers. Authoritative alias data can be supplied by higher
layers or external adapters without changing this resolver's core API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata


@dataclass(frozen=True, slots=True)
class AliasMatch:
    """Result of resolving an alias."""

    alias: str
    canonical_id: str
    confidence: float = 1.0


@dataclass(slots=True)
class AliasResolver:
    """Resolve normalized aliases to canonical identifiers.

    The resolver maintains two indexes:

    ``alias -> canonical_id``
        Fast forward lookup.

    ``canonical_id -> aliases``
        Reverse lookup for inspection and graph enrichment.

    Alias collisions are rejected by default rather than silently assigning
    an alias to an arbitrary biological entity.
    """

    _aliases: dict[str, str] = field(default_factory=dict)
    _canonical_aliases: dict[str, set[str]] = field(default_factory=dict)

    def register(
        self,
        canonical_id: str,
        aliases: list[str] | tuple[str, ...],
        *,
        allow_collision: bool = False,
    ) -> None:
        """Register aliases for a canonical identifier."""
        canonical = self._normalize_alias(canonical_id)

        if not canonical:
            raise ValueError("canonical_id cannot be empty")

        self._canonical_aliases.setdefault(canonical, set())

        for alias in aliases:
            normalized = self._normalize_alias(alias)

            if not normalized:
                continue

            existing = self._aliases.get(normalized)

            if (
                existing is not None
                and existing != canonical
                and not allow_collision
            ):
                raise ValueError(
                    f"Alias collision: '{alias}' is already mapped to "
                    f"'{existing}'"
                )

            self._aliases[normalized] = canonical
            self._canonical_aliases[canonical].add(normalized)

        # The canonical identifier itself is always resolvable.
        self._aliases.setdefault(canonical, canonical)
        self._canonical_aliases[canonical].add(canonical)

    def register_one(
        self,
        canonical_id: str,
        alias: str,
        *,
        allow_collision: bool = False,
    ) -> None:
        """Register one alias."""
        self.register(
            canonical_id,
            [alias],
            allow_collision=allow_collision,
        )

    def resolve(
        self,
        alias: str,
    ) -> AliasMatch | None:
        """Resolve an alias to its canonical identifier."""
        normalized = self._normalize_alias(alias)

        if not normalized:
            return None

        canonical = self._aliases.get(normalized)

        if canonical is None:
            return None

        return AliasMatch(
            alias=alias,
            canonical_id=canonical,
            confidence=1.0,
        )

    def resolve_id(self, alias: str) -> str | None:
        """Return only the canonical identifier for an alias."""
        match = self.resolve(alias)

        return match.canonical_id if match else None

    def aliases_for(
        self,
        canonical_id: str,
    ) -> tuple[str, ...]:
        """Return all registered aliases for a canonical identifier."""
        canonical = self._normalize_alias(canonical_id)

        return tuple(
            sorted(
                self._canonical_aliases.get(
                    canonical,
                    set(),
                )
            )
        )

    def contains(self, alias: str) -> bool:
        """Return whether an alias is registered."""
        return self._normalize_alias(alias) in self._aliases

    def remove(self, alias: str) -> bool:
        """Remove an alias.

        Returns ``True`` when an alias was removed.
        """
        normalized = self._normalize_alias(alias)

        canonical = self._aliases.pop(normalized, None)

        if canonical is None:
            return False

        aliases = self._canonical_aliases.get(canonical)

        if aliases is not None:
            aliases.discard(normalized)

            if not aliases:
                self._canonical_aliases.pop(canonical, None)

        return True

    def clear(self) -> None:
        """Remove all registered aliases."""
        self._aliases.clear()
        self._canonical_aliases.clear()

    def export(self) -> dict[str, tuple[str, ...]]:
        """Export the current canonical-to-alias mapping."""
        return {
            canonical: tuple(sorted(aliases))
            for canonical, aliases in self._canonical_aliases.items()
        }

    @staticmethod
    def _normalize_alias(alias: str) -> str:
        """Normalize an alias for deterministic matching."""
        if not isinstance(alias, str):
            raise TypeError("alias must be a string")

        value = unicodedata.normalize("NFKC", alias).strip()

        if not value:
            return ""

        value = value.casefold()

        # Treat common separators consistently.
        value = re.sub(r"[\s_\-]+", " ", value)

        # Remove surrounding punctuation while retaining meaningful
        # biomedical characters such as ':' and '.'
        value = value.strip(" \t\r\n.,;")

        return value


__all__ = [
    "AliasMatch",
    "AliasResolver",
]

