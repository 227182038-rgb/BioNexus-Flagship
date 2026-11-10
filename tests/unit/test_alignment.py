"""Tests for biokit.alignment."""

from __future__ import annotations

import random

from biokit.alignment import GotohAligner, NeedlemanWunsch, SmithWaterman


class TestNeedlemanWunsch:
    """Needleman-Wunsch global alignment tests."""

    def test_identical_sequences(self):
        result = NeedlemanWunsch(match=1, mismatch=-1, gap=-2).align("ATGC", "ATGC")
        assert result.matches == 4
        assert result.mismatches == 0
        assert result.gaps == 0
        assert result.identity == 100.0
        assert result.aligned_seq1 == "ATGC"
        assert result.aligned_seq2 == "ATGC"

    def test_with_gaps(self):
        result = NeedlemanWunsch(match=1, mismatch=-1, gap=-2).align("GATTACA", "GCATGCU")
        assert result.length > 0
        assert result.matches > 0


class TestSmithWaterman:
    """Smith-Waterman local alignment tests."""

    def test_local_alignment(self):
        result = SmithWaterman(match=2, mismatch=-1, gap=-2).align("ACGTACGT", "ACGT")
        assert result.matches > 0
        assert "ACGT" in result.aligned_seq1


class TestGotoh:
    """Gotoh affine-gap alignment tests."""

    def test_basic(self):
        result = GotohAligner(match=1, mismatch=-1, gap_open=-5, gap_extend=-1).align(
            "ACGTACGT", "ACGTACGT"
        )
        assert result.matches > 0


def _random_acgt_cases(count: int, seed: int = 777) -> list[tuple[str, str]]:
    """Deterministic corpus of small random sequence pairs."""
    rng = random.Random(seed)
    return [
        (
            "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 12))),
            "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 12))),
        )
        for _ in range(count)
    ]


def _affine_rescore(
    aligned1: str,
    aligned2: str,
    match: int = 1,
    mismatch: int = -1,
    gap_open: int = -5,
    gap_extend: int = -1,
) -> int:
    """Rescore an aligned pair under the affine scheme of GotohAligner."""
    total = 0
    in_gap_a = in_gap_b = False
    for ca, cb in zip(aligned1, aligned2, strict=True):
        if ca == "-":
            total += gap_extend if in_gap_a else gap_open
            in_gap_a, in_gap_b = True, False
        elif cb == "-":
            total += gap_extend if in_gap_b else gap_open
            in_gap_a, in_gap_b = False, True
        else:
            in_gap_a = in_gap_b = False
            total += match if ca == cb else mismatch
    return total


class TestGotohTracebackRegression:
    """Regression tests for lead L-10: traceback must be exact, not best-effort.

    Invariants:
    (a) reported score equals the Biopython PairwiseAligner optimum;
    (b) rescoring the RETURNED aligned strings reproduces the reported score;
    (c) removing gaps recovers the original input sequences.
    """

    # Deterministic corpus: fixed seed, small inputs to exercise paths quickly.
    _CASES = _random_acgt_cases(60)

    def test_rescore_matches_reported_score(self):
        aligner = GotohAligner(match=1, mismatch=-1, gap_open=-5, gap_extend=-1)
        for s1, s2 in self._CASES:
            res = aligner.align(s1, s2)
            assert _affine_rescore(res.aligned_seq1, res.aligned_seq2) == res.score, (s1, s2)

    def test_ungapping_recovers_inputs(self):
        aligner = GotohAligner(match=1, mismatch=-1, gap_open=-5, gap_extend=-1)
        for s1, s2 in self._CASES:
            res = aligner.align(s1, s2)
            assert res.aligned_seq1.replace("-", "") == s1
            assert res.aligned_seq2.replace("-", "") == s2

    def test_score_equals_biopython_optimum(self):
        from Bio import Align

        aligner = Align.PairwiseAligner(
            mode="global",
            match_score=1,
            mismatch_score=-1,
            open_gap_score=-5,
            extend_gap_score=-1,
        )
        gotoh = GotohAligner(match=1, mismatch=-1, gap_open=-5, gap_extend=-1)
        for s1, s2 in self._CASES:
            assert gotoh.align(s1, s2).score == aligner.score(s1, s2), (s1, s2)
