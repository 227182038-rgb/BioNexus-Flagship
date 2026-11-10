"""Calibrated literature-to-subject relevance resolution."""

from __future__ import annotations

from dataclasses import dataclass
import re

from nexus.fusion.models import BiomedicalEntity
from nexus.core.logging import get_logger, get_correlation_id, set_correlation_id

from .edge import EdgeType


@dataclass(frozen=True, slots=True)
class LiteratureLink:
    """Evidence-backed link between a subject and a publication."""

    source: str
    target: str
    relation: EdgeType
    relevance: float
    evidence: tuple[str, ...]
    tier: str


class LiteratureLinker:
    """Calibrate literature relevance using explicit evidence tiers."""

    def __init__(self) -> None:
        self.logger = get_logger("literature")

    def link(
        self,
        subject: BiomedicalEntity,
        publications: list[BiomedicalEntity],
        correlation_id: str | None = None,
    ) -> list[LiteratureLink]:
        cid = correlation_id or get_correlation_id()
        set_correlation_id(cid)

        source = subject.canonical_id or subject.id

        if not source:
            return []

        self.logger.debug("[%s] Linking literature for subject %s across %d publications", cid, source, len(publications))

        terms = self._subject_terms(subject)
        links: list[LiteratureLink] = []

        for publication in publications:
            target = publication.canonical_id or publication.id

            if not target:
                continue

            try:
                score, evidence, tier = self._score(
                    publication,
                    terms,
                )
            except Exception as exc:
                self.logger.warning("Literature scoring failed for publication %s [%s]: %s", target, cid, exc)
                continue

            if score <= 0:
                continue

            links.append(
                LiteratureLink(
                    source=source,
                    target=target,
                    relation=EdgeType.SUPPORTS,
                    relevance=score,
                    evidence=tuple(evidence),
                    tier=tier,
                )
            )

        return sorted(
            links,
            key=lambda item: (
                item.relevance,
                item.tier,
                item.target,
            ),
            reverse=True,
        )

    @staticmethod
    def _subject_terms(
        entity: BiomedicalEntity,
    ) -> set[str]:
        terms: set[str] = set()

        candidates = [
            entity.name,
            *entity.aliases,
            *entity.genes,
            *entity.proteins,
        ]

        for value in candidates:
            if not value:
                continue

            cleaned = str(value).strip().casefold()

            if cleaned:
                terms.add(cleaned)

        # Ignore identifiers that are unlikely to be useful text signals.
        return {
            term
            for term in terms
            if not term.startswith(
                ("ensembl:", "uniprot:", "pubmed:", "pdb:")
            )
        }

    def _score(
        self,
        publication: BiomedicalEntity,
        terms: set[str],
    ) -> tuple[float, list[str], str]:

        metadata = publication.metadata

        title = str(
            publication.name or ""
        ).strip().casefold()

        mesh_terms = [
            str(value).strip().casefold()
            for value in metadata.get(
                "mesh_terms",
                [],
            )
            if value
        ]

        publication_types = [
            str(value).strip().casefold()
            for value in metadata.get(
                "publication_types",
                [],
            )
            if value
        ]

        score = 0.0
        evidence: list[str] = []

        exact_title = False
        mesh_match = False
        p53_mesh = False

        for term in terms:
            if term and term in title:
                exact_title = True
                score += 0.55
                evidence.append(
                    f"title_match:{term}"
                )

            if any(
                term == mesh
                or term in mesh
                for mesh in mesh_terms
            ):
                mesh_match = True
                score += 0.25
                evidence.append(
                    f"mesh_match:{term}"
                )

        subject_mesh = False
        for term in terms:
            if not term:
                continue
            escaped = re.escape(term)
            if any(re.search(rf"\b{escaped}\b", mesh) for mesh in mesh_terms):
                subject_mesh = True
                score += 0.20
                evidence.append(f"explicit_subject_mesh_term:{term}")
                break

        # Reviews are valuable contextual evidence but should not
        # automatically outrank direct primary literature.
        if any(
            "review" in value
            for value in publication_types
        ):
            score += 0.05
            evidence.append(
                "publication_type:review"
            )

        # Tier assignment is deliberately conservative.
        if exact_title and subject_mesh:
            tier = "A_direct"
            score += 0.10

        elif exact_title and mesh_match:
            tier = "A_direct"
            score += 0.05

        elif subject_mesh:
            tier = "B_strong_context"

        elif exact_title:
            tier = "B_strong_context"

        elif mesh_match:
            tier = "C_contextual"

        else:
            tier = "D_weak"

        if tier == "D_weak":
            return 0.0, [], tier

        return (
            min(score, 1.0),
            evidence,
            tier,
        )


__all__ = [
    "LiteratureLink",
    "LiteratureLinker",
]
