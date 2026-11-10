from __future__ import annotations

from enum import StrEnum


class EntityType(StrEnum):
    GENE = "gene"
    PROTEIN = "protein"
    DISEASE = "disease"
    PATHWAY = "pathway"
    DRUG = "drug"
    VARIANT = "variant"
    PUBLICATION = "publication"
    STRUCTURE = "structure"
    ORGANISM = "organism"
    CELL = "cell"
    TISSUE = "tissue"
    ONTOLOGY = "ontology"
    COMPOUND = "compound"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "EntityType":
        value = value.strip().lower()

        for member in cls:
            if member.value == value:
                return member

        return cls.UNKNOWN
