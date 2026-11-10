"""Evidence-grounded scientific contradiction detection."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .evidence import Evidence


@dataclass(slots=True)
class Contradiction:
    """Represents conflicting scientific evidence."""

    topic: str

    claim_a: str

    claim_b: str

    source_a: str

    source_b: str

    confidence: float


class ContradictionDetector:
    """Detect explicit conflicts between structured scientific claims."""

    def detect(
        self,
        evidence: list[Evidence],
    ) -> list[Contradiction]:
        """Detect contradictions only when claims are structured."""

        grouped: dict[str, list[Evidence]] = {}

        for item in evidence:
            topic = self._topic(item)
            claim = self._claim(item)

            # Do not manufacture claims from arbitrary titles.
            if not topic or not claim:
                continue

            grouped.setdefault(
                topic,
                [],
            ).append(item)

        contradictions: list[Contradiction] = []

        for topic, items in grouped.items():
            for index, first in enumerate(items):
                for second in items[index + 1:]:
                    contradiction = self._compare(
                        topic,
                        first,
                        second,
                    )

                    if contradiction is not None:
                        contradictions.append(
                            contradiction
                        )

        return self._deduplicate(
            contradictions
        )

    @staticmethod
    def _topic(
        item: Evidence,
    ) -> str | None:
        """Return a structured normalized topic."""

        value = item.metadata.get(
            "topic"
        )

        if value is None:
            return None

        value = str(value).strip().casefold()

        return value or None

    @staticmethod
    def _claim(
        item: Evidence,
    ) -> str | None:
        """Return a structured normalized claim."""

        value = item.metadata.get(
            "claim"
        )

        if value is None:
            return None

        value = str(value).strip()

        return value or None

    def _compare(
        self,
        topic: str,
        first: Evidence,
        second: Evidence,
    ) -> Contradiction | None:
        """Compare two claims for explicit semantic opposition."""

        claim_a = self._normalize_claim(
            self._claim(first)
        )

        claim_b = self._normalize_claim(
            self._claim(second)
        )

        if not claim_a or not claim_b:
            return None

        if claim_a == claim_b:
            return None

        polarity_a = self._polarity(
            claim_a
        )

        polarity_b = self._polarity(
            claim_b
        )

        if (
            polarity_a == "unknown"
            or polarity_b == "unknown"
        ):
            return None

        if polarity_a == polarity_b:
            return None

        confidence = min(
            max(
                0.0,
                min(
                    1.0,
                    float(first.confidence),
                ),
            ),
            max(
                0.0,
                min(
                    1.0,
                    float(second.confidence),
                ),
            ),
        )

        return Contradiction(
            topic=topic,
            claim_a=claim_a,
            claim_b=claim_b,
            source_a=first.source,
            source_b=second.source,
            confidence=confidence,
        )

    @staticmethod
    def _normalize_claim(
        claim: str | None,
    ) -> str | None:
        if not claim:
            return None

        normalized = re.sub(
            r"\s+",
            " ",
            claim.strip(),
        )

        return normalized

    @staticmethod
    def _polarity(
        claim: str,
    ) -> str:
        """Classify only obvious scientific polarity markers."""

        value = claim.casefold()

        positive_patterns = (
            r"\bincrease\b",
            r"\bincreased\b",
            r"\bupregulat(?:e|ed|ion)\b",
            r"\bactivate(?:s|d|ion)\b",
            r"\bpromote(?:s|d)?\b",
            r"\benhance(?:s|d)?\b",
            r"\bassociat(?:e|ed|ion)\b",
            r"\bsupport(?:s|ed)?\b",
            r"\bprotect(?:s|ed|ive)\b",
            r"\bbenefit(?:s|ed)?\b",
        )

        negative_patterns = (
            r"\bdecrease\b",
            r"\bdecreased\b",
            r"\bdownregulat(?:e|ed|ion)\b",
            r"\binhibit(?:s|ed|ion)\b",
            r"\bsuppress(?:es|ed|ion)\b",
            r"\breduce(?:s|d)?\b",
            r"\bimpair(?:s|ed|ment)\b",
            r"\boppose(?:s|d)?\b",
            r"\bharm(?:s|ed|ful)\b",
        )

        if any(
            re.search(
                pattern,
                value,
            )
            for pattern in positive_patterns
        ):
            return "positive"

        if any(
            re.search(
                pattern,
                value,
            )
            for pattern in negative_patterns
        ):
            return "negative"

        return "unknown"

    @staticmethod
    def _deduplicate(
        contradictions: list[Contradiction],
    ) -> list[Contradiction]:
        """Deduplicate equivalent contradiction pairs."""

        seen: set[
            tuple[str, str, str]
        ] = set()

        result: list[Contradiction] = []

        for item in contradictions:
            claims = tuple(
                sorted(
                    (
                        item.claim_a.casefold(),
                        item.claim_b.casefold(),
                    )
                )
            )

            key = (
                item.topic.casefold(),
                claims[0],
                claims[1],
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result


__all__ = [
    "Contradiction",
    "ContradictionDetector",
]
