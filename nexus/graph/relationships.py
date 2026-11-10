"""Evidence-backed biomedical relationship resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from nexus.fusion.models import BiomedicalEntity
from .edge import EdgeType


@dataclass(frozen=True, slots=True)
class RelationshipCandidate:
    """Candidate relationship between biomedical entities."""

    source: str
    target: str
    relation: EdgeType
    confidence: float
    evidence: tuple[str, ...] = ()


class RelationshipResolver:
    """Resolve relationships supported by explicit source cross-references."""

    def resolve(
        self,
        entities: Iterable[BiomedicalEntity],
    ) -> list[RelationshipCandidate]:
        """Resolve cross-reference-backed relationships."""

        items = list(entities)
        candidates: list[RelationshipCandidate] = []

        for source in items:
            source_id = source.canonical_id or source.id

            if not source_id:
                continue

            for target in items:
                target_id = target.canonical_id or target.id

                if (
                    not target_id
                    or target_id == source_id
                ):
                    continue

                match = self._shared_authoritative_identity(
                    source,
                    target,
                )

                if match is None:
                    continue

                relation, confidence, evidence = match

                candidates.append(
                    RelationshipCandidate(
                        source=source_id,
                        target=target_id,
                        relation=relation,
                        confidence=confidence,
                        evidence=evidence,
                    )
                )

        return self._deduplicate(candidates)

    def _shared_authoritative_identity(
        self,
        source: BiomedicalEntity,
        target: BiomedicalEntity,
    ) -> tuple[EdgeType, float, tuple[str, ...]] | None:
        """Detect explicit shared authoritative identities."""

        source_refs = self._reference_tokens(source)
        target_refs = self._reference_tokens(target)

        shared = source_refs & target_refs

        if not shared:
            return None

        source_type = self._entity_type(source)
        target_type = self._entity_type(target)

        # ENCODES is directional:
        # gene -> protein
        if (
            source_type == "gene"
            and target_type == "protein"
        ):
            return (
                EdgeType.ENCODES,
                0.98,
                tuple(
                    f"authoritative_xref:{value}"
                    for value in sorted(shared)
                ),
            )

        # Never create protein -> gene using ENCODES.
        if (
            source_type == "protein"
            and target_type == "gene"
        ):
            return None

        return (
            EdgeType.ASSOCIATED_WITH,
            0.95,
            tuple(
                f"authoritative_xref:{value}"
                for value in sorted(shared)
            ),
        )

    @classmethod
    def _reference_tokens(
        cls,
        entity: BiomedicalEntity,
    ) -> set[str]:
        """Build comparable identity tokens."""

        tokens: set[str] = set()

        def add(value: str) -> None:
            value = value.strip().casefold()

            if not value:
                return

            tokens.add(value)

            if ":" in value:
                tokens.add(
                    value.split(":", 1)[1]
                )

        if entity.id:
            add(entity.id)

        if entity.canonical_id:
            add(entity.canonical_id)

        for value in entity.cross_references.values():
            if value:
                add(str(value))

        for value in entity.genes:
            if value:
                add(str(value))

        for value in entity.proteins:
            if value:
                add(str(value))

        return tokens

    @staticmethod
    def _entity_type(
        entity: BiomedicalEntity,
    ) -> str:
        value = entity.entity_type

        if hasattr(value, "value"):
            return str(value.value).casefold()

        return str(value).casefold()

    @staticmethod
    def _deduplicate(
        candidates: list[RelationshipCandidate],
    ) -> list[RelationshipCandidate]:
        """Remove duplicate relationships deterministically."""

        seen: set[tuple[str, str, EdgeType]] = set()
        result: list[RelationshipCandidate] = []

        for candidate in candidates:
            key = (
                candidate.source,
                candidate.target,
                candidate.relation,
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(candidate)

        return result


__all__ = [
    "RelationshipCandidate",
    "RelationshipResolver",
]
