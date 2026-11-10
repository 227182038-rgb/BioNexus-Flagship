"""Hardy-Weinberg test and Weir-Cockerham F_ST."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from itertools import combinations

from biokit.exceptions import PopGenError
from biokit.popgen.frequencies import allele_frequencies
from biokit.popgen.models import Genotype


def hardy_weinberg_test(genotypes: Iterable[Genotype]) -> tuple[float, float]:
    """χ² test for Hardy-Weinberg equilibrium.

    Returns
    -------
    tuple[float, float]
        ``(chi2_statistic, p_value)``.
    """
    from scipy import stats

    genotypes = list(genotypes)
    freqs = allele_frequencies(genotypes)
    n = len(genotypes)
    alleles = sorted(freqs)
    observed: dict[Genotype, int] = defaultdict(int)
    for gt in genotypes:
        a1, a2 = sorted(gt)
        observed[(a1, a2)] += 1
    expected: list[float] = []
    obs_list: list[int] = []
    for a1, a2 in combinations(alleles, 2):
        exp = 2 * freqs[a1] * freqs[a2] * n
        if exp > 0:
            expected.append(exp)
            obs_list.append(observed.get((a1, a2), 0))
    for a in alleles:
        exp = freqs[a] * freqs[a] * n
        if exp > 0:
            expected.append(exp)
            obs_list.append(observed.get((a, a), 0))
    if not expected:
        return 0.0, 1.0
    chi2 = sum((o - e) ** 2 / e for o, e in zip(obs_list, expected, strict=False))
    # Degrees of freedom for HWE: with k alleles there are k(k+1)/2 unordered
    # genotype classes, and estimating the k allele frequencies from the same
    # sample consumes k-1 free parameters (they are constrained to sum to 1):
    #   df = k(k+1)/2 - 1 - (k-1) = k(k-1)/2
    # (1 for a biallelic locus).  This is the asymptotic chi-square
    # approximation; when many expected counts are small (commonly < 5),
    # an exact test is preferable and this p-value should be treated with
    # caution.
    k = len(alleles)
    df = max(k * (k - 1) // 2, 1)
    p = float(stats.chi2.sf(chi2, df))
    return float(chi2), float(p)


def weir_cockerham_fst(populations: Sequence[Iterable[Genotype]]) -> float:
    """F_ST via Weir & Cockerham (1984).

    Parameters
    ----------
    populations : sequence of iterables of (str, str)
        One iterable of genotypes per sub-population.
    """
    pop_lists = [list(p) for p in populations]
    if len(pop_lists) < 2:
        raise PopGenError("F_ST requires ≥ 2 populations")
    all_alleles: set[str] = set()
    for pop in pop_lists:
        for gt in pop:
            all_alleles.update(gt)
    alleles = sorted(all_alleles)
    if not alleles:
        raise PopGenError("no alleles observed")
    if len(alleles) == 1:
        return 0.0
    a = alleles[0]
    r = len(pop_lists)
    n_per_pop = [len(p) for p in pop_lists]
    n_total = sum(n_per_pop)
    n_bar = n_total / r
    p_per_pop: list[float] = []
    h_per_pop: list[float] = []
    for pop, n_i in zip(pop_lists, n_per_pop, strict=False):
        if n_i == 0:
            p_per_pop.append(0.0)
            h_per_pop.append(0.0)
            continue
        ac: Counter[str] = Counter()
        for gt in pop:
            ac[gt[0]] += 1
            ac[gt[1]] += 1
        p_per_pop.append(ac[a] / (2 * n_i))
        h_per_pop.append(sum(1 for gt in pop if gt[0] != gt[1]) / n_i)
    p_bar = sum(p * n for p, n in zip(p_per_pop, n_per_pop, strict=False)) / n_total
    s2 = sum(n * (p - p_bar) ** 2 for p, n in zip(p_per_pop, n_per_pop, strict=False)) / (
        (r - 1) * n_bar
    )
    h_bar = sum(h * n for h, n in zip(h_per_pop, n_per_pop, strict=False)) / n_total
    n_c = (n_total - sum(n * n for n in n_per_pop) / n_total) / (r - 1)
    a_num = (
        n_bar
        / n_c
        * (s2 - (1 / (n_bar - 1)) * (p_bar * (1 - p_bar) - (r - 1) / r * s2 - h_bar / 4))
    )
    b_num = (
        n_bar
        / (n_bar - 1)
        * (p_bar * (1 - p_bar) - (r - 1) / r * s2 - (1 - 2 * n_bar) / (4 * n_bar) * h_bar)
    )
    c_num = h_bar / 2
    denom = a_num + b_num + c_num
    if denom == 0:
        return 0.0
    return float(a_num / denom)


__all__ = ["hardy_weinberg_test", "weir_cockerham_fst"]
