"""Guide RNA scanner, scorer, and off-target search."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from biokit.crispr.constants import (
    DEFAULT_GUIDE_LENGTH,
    DEFAULT_MAX_MISMATCHES,
    DEFAULT_PAM,
)
from biokit.crispr.models import GuideRNA, OffTarget
from biokit.sequence.dna import DNA
from biokit.sequence.validation import validate_dna
from biokit.statistics.sequence_stats import gc_fraction


def _matches_pam(pam: str, pattern: str = DEFAULT_PAM) -> bool:
    if len(pam) != len(pattern):
        return False
    return all(q == "N" or p == q for p, q in zip(pam, pattern, strict=False))


def iter_guides(
    sequence: str,
    guide_length: int = DEFAULT_GUIDE_LENGTH,
    pam_pattern: str = DEFAULT_PAM,
    require_pam: bool = True,
) -> Iterator[tuple[str, str, int, str]]:
    """Yield ``(spacer, pam, position, strand)`` for candidate guides."""
    validate_dna(sequence)
    seq = sequence.upper()
    rc_seq = DNA(seq).reverse_complement().sequence
    for i in range(len(seq) - guide_length - len(pam_pattern) + 1):
        spacer = seq[i : i + guide_length]
        pam = seq[i + guide_length : i + guide_length + len(pam_pattern)]
        if not require_pam or _matches_pam(pam, pam_pattern):
            yield spacer, pam, i + 1, "+"
    for i in range(len(rc_seq) - guide_length - len(pam_pattern) + 1):
        spacer = rc_seq[i : i + guide_length]
        pam = rc_seq[i + guide_length : i + guide_length + len(pam_pattern)]
        if not require_pam or _matches_pam(pam, pam_pattern):
            forward_start = len(seq) - (i + guide_length + len(pam_pattern)) + 1
            yield spacer, pam, forward_start, "-"


def score_guide(spacer: str, off_targets: int = 0) -> float:
    """Score a guide; higher is better. Score range: [0, 60].

    The ceiling is 60 (achieved when GC fraction is in [0.4, 0.7] and the
    spacer contains no ``TTTT``); off-targets divide the GC score down.
    """
    gc = gc_fraction(spacer)
    if 0.4 <= gc <= 0.7:
        gc_score = 60.0
    else:
        gc_score = max(0.0, 60.0 - abs(gc - 0.55) * 200)
    if "TTTT" in spacer.upper():
        gc_score -= 20.0
    return float(max(0.0, gc_score / (1.0 + off_targets)))


def design_guides(
    sequence: str,
    guide_length: int = DEFAULT_GUIDE_LENGTH,
    pam_pattern: str = DEFAULT_PAM,
    min_score: float = 0.0,
) -> list[GuideRNA]:
    """Design all candidate sgRNAs for ``sequence``.

    Examples
    --------
    >>> guides = design_guides("ATGGCAGGTGACCCGTTGACCGGTAACGCATGCAGTGGACCTAGG")
    >>> all(g.pam.endswith("GG") for g in guides)
    True
    """
    guides: list[GuideRNA] = []
    for spacer, pam, pos, strand in iter_guides(
        sequence, guide_length=guide_length, pam_pattern=pam_pattern
    ):
        gc = gc_fraction(spacer)
        sc = score_guide(spacer)
        if sc < min_score:
            continue
        guides.append(
            GuideRNA(
                spacer=spacer,
                pam=pam,
                strand=strand,
                position=pos,
                gc=gc,
                score=sc,
            )
        )
    guides.sort(key=lambda g: g.score, reverse=True)
    return guides


def _hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b, strict=True) if x != y)


def find_off_targets(
    guide: GuideRNA,
    reference: str,
    max_mismatches: int = DEFAULT_MAX_MISMATCHES,
) -> list[OffTarget]:
    """Find off-target sites for ``guide`` in ``reference``."""
    validate_dna(reference)
    ref = reference.upper()
    hits: list[OffTarget] = []
    spacer_len = len(guide.spacer)
    for spacer, pam, pos, strand in iter_guides(ref, guide_length=spacer_len):
        mm = _hamming(spacer, guide.spacer)
        if mm > max_mismatches:
            continue
        if pos == guide.position and strand == guide.strand and mm == 0:
            continue
        hits.append(
            OffTarget(
                sequence=spacer + pam,
                position=pos,
                strand=strand,
                mismatches=mm,
            )
        )
    return hits


def annotate_guides(
    guides: Sequence[GuideRNA],
    reference: str,
    max_mismatches: int = DEFAULT_MAX_MISMATCHES,
) -> list[GuideRNA]:
    """Annotate each guide with off-target hits and re-score."""
    out: list[GuideRNA] = []
    for g in guides:
        off_targets = find_off_targets(g, reference, max_mismatches=max_mismatches)
        new_score = score_guide(g.spacer, off_targets=len(off_targets))
        out.append(
            GuideRNA(
                spacer=g.spacer,
                pam=g.pam,
                strand=g.strand,
                position=g.position,
                gc=g.gc,
                score=new_score,
                off_targets=off_targets,
            )
        )
    out.sort(key=lambda g: g.score, reverse=True)
    return out


__all__ = [
    "annotate_guides",
    "design_guides",
    "find_off_targets",
    "iter_guides",
    "score_guide",
]
