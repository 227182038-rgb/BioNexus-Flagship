"""Biomedical identifier normalization.

Provides deterministic normalization and classification of identifiers used by
the BioNexus knowledge-source layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from urllib.parse import unquote, urlparse

from nexus.knowledge.base import KnowledgeSource


class IdentifierNamespace(str, Enum):
    """Supported biomedical identifier namespaces."""

    PUBMED = "pubmed"
    NCBI = "ncbi"
    UNIPROT = "uniprot"
    PDB = "pdb"
    ENSEMBL = "ensembl"
    GO = "go"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class NormalizedIdentifier:
    """Canonical representation of a biomedical identifier."""

    namespace: IdentifierNamespace
    value: str
    original: str

    @property
    def canonical(self) -> str:
        """Return a stable namespace-qualified identifier."""
        return f"{self.namespace.value}:{self.value}"

    @property
    def source(self) -> KnowledgeSource | None:
        """Return the corresponding BioNexus knowledge source."""
        mapping = {
            IdentifierNamespace.PUBMED: KnowledgeSource.PUBMED,
            IdentifierNamespace.NCBI: KnowledgeSource.NCBI,
            IdentifierNamespace.UNIPROT: KnowledgeSource.UNIPROT,
            IdentifierNamespace.PDB: KnowledgeSource.PDB,
            IdentifierNamespace.ENSEMBL: KnowledgeSource.ENSEMBL,
            IdentifierNamespace.GO: KnowledgeSource.GO,
        }

        return mapping.get(self.namespace)


class IdentifierNormalizer:
    """Normalize biomedical identifiers deterministically."""

    _PREFIXES = {
        "pmid": IdentifierNamespace.PUBMED,
        "pubmed": IdentifierNamespace.PUBMED,
        "ncbi": IdentifierNamespace.NCBI,
        "geneid": IdentifierNamespace.NCBI,
        "uniprot": IdentifierNamespace.UNIPROT,
        "uniprotkb": IdentifierNamespace.UNIPROT,
        "pdb": IdentifierNamespace.PDB,
        "rcsb": IdentifierNamespace.PDB,
        "ensembl": IdentifierNamespace.ENSEMBL,
        "ensg": IdentifierNamespace.ENSEMBL,
        "ensp": IdentifierNamespace.ENSEMBL,
        "enst": IdentifierNamespace.ENSEMBL,
        "go": IdentifierNamespace.GO,
    }

    @classmethod
    def normalize(
        cls,
        identifier: str,
        namespace: IdentifierNamespace | str | None = None,
    ) -> NormalizedIdentifier:
        """Normalize a raw biomedical identifier."""

        if not isinstance(identifier, str):
            raise TypeError("identifier must be a string")

        original = identifier
        value = identifier.strip()

        if not value:
            raise ValueError("identifier cannot be empty")

        detected_namespace = cls._coerce_namespace(namespace)

        if cls._looks_like_url(value):
            url_namespace, value = cls._from_url(value)

            if detected_namespace is None:
                detected_namespace = url_namespace

        value = unquote(value).strip()

        prefix_namespace, value = cls._remove_prefix(value)

        if detected_namespace is None:
            detected_namespace = prefix_namespace

        if detected_namespace is None:
            detected_namespace = cls._infer_namespace(value)

        if detected_namespace is None:
            detected_namespace = IdentifierNamespace.UNKNOWN

        value = cls._clean_value(value, detected_namespace)

        cls._validate(value, detected_namespace)

        return NormalizedIdentifier(
            namespace=detected_namespace,
            value=value,
            original=original,
        )

    @classmethod
    def normalize_many(
        cls,
        identifiers: list[str],
        namespace: IdentifierNamespace | str | None = None,
    ) -> list[NormalizedIdentifier]:
        """Normalize identifiers and remove canonical duplicates."""

        results = []
        seen = set()

        for identifier in identifiers:
            normalized = cls.normalize(identifier, namespace)

            if normalized.canonical in seen:
                continue

            seen.add(normalized.canonical)
            results.append(normalized)

        return results

    @classmethod
    def detect_namespace(
        cls,
        identifier: str,
    ) -> IdentifierNamespace:
        """Detect the namespace of an identifier."""
        return cls.normalize(identifier).namespace

    @classmethod
    def canonicalize(cls, identifier: str) -> str:
        """Return the canonical namespace-qualified representation."""
        return cls.normalize(identifier).canonical

    @classmethod
    def _coerce_namespace(
        cls,
        namespace: IdentifierNamespace | str | None,
    ) -> IdentifierNamespace | None:
        if namespace is None:
            return None

        if isinstance(namespace, IdentifierNamespace):
            return namespace

        normalized = namespace.strip().lower()

        for candidate in IdentifierNamespace:
            if candidate.value == normalized:
                return candidate

        raise ValueError(
            f"Unsupported identifier namespace: {namespace}"
        )

    @staticmethod
    def _looks_like_url(value: str) -> bool:
        return value.lower().startswith(("http://", "https://"))

    @classmethod
    def _from_url(
        cls,
        value: str,
    ) -> tuple[IdentifierNamespace | None, str]:
        parsed = urlparse(value)

        host = parsed.netloc.lower()
        path = parsed.path.strip("/")

        if "pubmed.ncbi.nlm.nih.gov" in host:
            match = re.fullmatch(r"(\d+)", path)

            if match:
                return IdentifierNamespace.PUBMED, match.group(1)

        if "uniprot.org" in host:
            parts = [part for part in path.split("/") if part]

            if parts and parts[0].lower() in {
                "uniprot",
                "uniprotkb",
            }:
                parts = parts[1:]

            if parts:
                return IdentifierNamespace.UNIPROT, parts[0]

        if "rcsb.org" in host:
            parts = [part for part in path.split("/") if part]

            if len(parts) >= 2 and parts[0].lower() == "structure":
                return IdentifierNamespace.PDB, parts[1]

        if "ebi.ac.uk" in host:
            parts = [part for part in path.split("/") if part]

            if parts:
                candidate = parts[-1]

                if candidate.upper().startswith(
                    ("ENSG", "ENSP", "ENST")
                ):
                    return IdentifierNamespace.ENSEMBL, candidate

        if "ncbi.nlm.nih.gov" in host:
            parts = [part for part in path.split("/") if part]

            if parts:
                return IdentifierNamespace.NCBI, parts[-1]

        if "geneontology.org" in host:
            parts = [part for part in path.split("/") if part]

            if parts:
                candidate = parts[-1]

                if candidate.upper().startswith("GO:"):
                    return IdentifierNamespace.GO, candidate

        raise ValueError(
            f"Unsupported biomedical identifier URL: {value}"
        )

    @classmethod
    def _remove_prefix(
        cls,
        value: str,
    ) -> tuple[IdentifierNamespace | None, str]:
        if ":" not in value:
            return None, value

        prefix, remainder = value.split(":", 1)

        namespace = cls._PREFIXES.get(prefix.strip().lower())

        if namespace is None:
            return None, value

        return namespace, remainder.strip()

    @classmethod
    def _infer_namespace(
        cls,
        value: str,
    ) -> IdentifierNamespace | None:
        if re.fullmatch(r"\d+", value):
            return IdentifierNamespace.PUBMED

        if re.fullmatch(
            r"[OPQ][0-9][A-Z0-9]{3}[0-9]",
            value,
            re.IGNORECASE,
        ):
            return IdentifierNamespace.UNIPROT

        if re.fullmatch(r"[0-9A-Za-z]{4}", value):
            return IdentifierNamespace.PDB

        if re.fullmatch(
            r"ENS[A-Z]{0,2}\d+",
            value,
            re.IGNORECASE,
        ):
            return IdentifierNamespace.ENSEMBL

        if re.fullmatch(
            r"GO:\d+",
            value,
            re.IGNORECASE,
        ):
            return IdentifierNamespace.GO

        return None

    @staticmethod
    def _clean_value(
        value: str,
        namespace: IdentifierNamespace,
    ) -> str:
        value = value.strip().strip("/")

        if namespace == IdentifierNamespace.GO:
            # The namespace is stored separately, so the value is numeric.
            return value.upper().removeprefix("GO:")

        if namespace in {
            IdentifierNamespace.UNIPROT,
            IdentifierNamespace.PDB,
            IdentifierNamespace.ENSEMBL,
        }:
            return value.upper()

        return value

    @staticmethod
    def _validate(
        value: str,
        namespace: IdentifierNamespace,
    ) -> None:
        if not value:
            raise ValueError(
                "normalized identifier cannot be empty"
            )

        if namespace == IdentifierNamespace.PUBMED:
            if not value.isdigit():
                raise ValueError(
                    f"Invalid PubMed identifier: {value}"
                )

        elif namespace == IdentifierNamespace.GO:
            if not re.fullmatch(r"\d+", value):
                raise ValueError(
                    f"Invalid GO identifier: {value}"
                )

        elif namespace == IdentifierNamespace.PDB:
            if not re.fullmatch(r"[0-9A-Z]{4}", value):
                raise ValueError(
                    f"Invalid PDB identifier: {value}"
                )


__all__ = [
    "IdentifierNamespace",
    "IdentifierNormalizer",
    "NormalizedIdentifier",
]
