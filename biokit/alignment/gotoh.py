"""Gotoh alignment with affine gap penalties."""

from __future__ import annotations

from collections.abc import Callable

from biokit.alignment.result import AlignmentResult
from biokit.alignment.scoring import match_mismatch_scoring


class GotohAligner:
    """Gotoh's algorithm for global alignment with affine gap penalties.

    Affine gap model: ``gap_open + (k - 1) * gap_extend`` for a gap of length k.

    Parameters
    ----------
    match : int, optional
        Match score (default 1).
    mismatch : int, optional
        Mismatch penalty (default -1).
    gap_open : int, optional
        Gap opening penalty (default -5).
    gap_extend : int, optional
        Gap extension penalty (default -1).
    """

    def __init__(
        self,
        match: int = 1,
        mismatch: int = -1,
        gap_open: int = -5,
        gap_extend: int = -1,
    ) -> None:
        self.match = match
        self.mismatch = mismatch
        self.gap_open = gap_open
        self.gap_extend = gap_extend

    def align(
        self,
        seq1: str,
        seq2: str,
        scoring: Callable[[str, str], int] | None = None,
    ) -> AlignmentResult:
        """Align two sequences with affine gap penalties."""
        score_fn = scoring or match_mismatch_scoring(self.match, self.mismatch)
        m, n = len(seq1), len(seq2)
        NEG_INF = float("-inf")
        # M = match/mismatch matrix; Ix = gap in seq2 (consumed seq1); Iy = gap in seq1
        M = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
        Ix = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
        Iy = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
        M[0][0] = 0
        for i in range(1, m + 1):
            Ix[i][0] = self.gap_open + (i - 1) * self.gap_extend
        for j in range(1, n + 1):
            Iy[0][j] = self.gap_open + (j - 1) * self.gap_extend
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                s = score_fn(seq1[i - 1], seq2[j - 1])
                M[i][j] = max(
                    M[i - 1][j - 1] + s,
                    Ix[i - 1][j - 1] + s,
                    Iy[i - 1][j - 1] + s,
                )
                Ix[i][j] = max(
                    M[i - 1][j] + self.gap_open,
                    Ix[i - 1][j] + self.gap_extend,
                )
                Iy[i][j] = max(
                    M[i][j - 1] + self.gap_open,
                    Iy[i][j - 1] + self.gap_extend,
                )
        # Find best score in last row/col
        best_score = max(M[m][n], Ix[m][n], Iy[m][n])
        # Exact traceback: always follow predecessors *in the matrix whose
        # value matches the optimum at the current cell*, tracking which
        # matrix (M / Ix / Iy) the path is in.  Ties (equal predecessors)
        # are resolved deterministically with priority M > Ix > Iy; every
        # choice among equal predecessors yields an equally optimal
        # alignment under the same scoring scheme.
        NEG_INF = float("-inf")
        if M[m][n] == best_score:
            state = "M"
        elif Ix[m][n] == best_score:
            state = "Ix"
        else:
            state = "Iy"
        aligned1: list[str] = []
        aligned2: list[str] = []
        i, j = m, n
        matches = mismatches = gaps = 0
        while i > 0 or j > 0:
            if state == "M":
                # M[i][j] is defined only for i > 0 and j > 0 (M has -inf on
                # the borders), so being in M here implies both are positive.
                s = score_fn(seq1[i - 1], seq2[j - 1])
                diag_target = M[i][j] - s
                for prev_matrix in ("M", "Ix", "Iy"):
                    prev = {"M": M[i - 1][j - 1], "Ix": Ix[i - 1][j - 1],
                            "Iy": Iy[i - 1][j - 1]}[prev_matrix]
                    if prev == diag_target:
                        state = prev_matrix
                        break
                aligned1.append(seq1[i - 1])
                aligned2.append(seq2[j - 1])
                if seq1[i - 1] == seq2[j - 1]:
                    matches += 1
                else:
                    mismatches += 1
                i -= 1
                j -= 1
            elif state == "Ix":
                # Gap in seq2: Ix[i][j] = max(M[i-1][j] + gap_open,
                #                             Ix[i-1][j] + gap_extend)
                opener = M[i - 1][j] + self.gap_open if i > 0 else NEG_INF
                extender = Ix[i - 1][j] + self.gap_extend if i > 0 else NEG_INF
                if Ix[i][j] == opener:
                    state = "M"
                elif Ix[i][j] == extender:
                    state = "Ix"  # extend the existing gap
                else:  # pragma: no cover - optimum inconsistency guard
                    raise ValueError("traceback hit an inconsistent cell in Ix")
                aligned1.append(seq1[i - 1])
                aligned2.append("-")
                gaps += 1
                i -= 1
            else:  # state == "Iy"
                # Gap in seq1: Iy[i][j] = max(M[i][j-1] + gap_open,
                #                             Iy[i][j-1] + gap_extend)
                opener = M[i][j - 1] + self.gap_open if j > 0 else NEG_INF
                extender = Iy[i][j - 1] + self.gap_extend if j > 0 else NEG_INF
                if Iy[i][j] == opener:
                    state = "M"
                elif Iy[i][j] == extender:
                    state = "Iy"
                else:  # pragma: no cover - optimum inconsistency guard
                    raise ValueError("traceback hit an inconsistent cell in Iy")
                aligned1.append("-")
                aligned2.append(seq2[j - 1])
                gaps += 1
                j -= 1
        return AlignmentResult(
            score=best_score,
            aligned_seq1="".join(reversed(aligned1)),
            aligned_seq2="".join(reversed(aligned2)),
            matches=matches,
            mismatches=mismatches,
            gaps=gaps,
        )


__all__ = ["GotohAligner"]
