"""Tests for biokit.popgen."""

from __future__ import annotations

import pytest
from biokit.exceptions import PopGenError
from biokit.popgen import (
    allele_frequencies,
    allele_stats,
    expected_heterozygosity,
    hardy_weinberg_test,
    linkage_disequilibrium,
    observed_heterozygosity,
    weir_cockerham_fst,
)


def test_allele_frequencies_basic():
    assert allele_frequencies([("A", "A"), ("A", "a"), ("a", "a")]) == {"A": 0.5, "a": 0.5}


def test_allele_frequencies_empty():
    assert allele_frequencies([]) == {}


def test_allele_frequencies_invalid_tuple():
    with pytest.raises(PopGenError):
        allele_frequencies([("A", "B", "C")])


def test_observed_heterozygosity():
    assert observed_heterozygosity([("A", "A"), ("A", "a"), ("a", "a")]) == pytest.approx(1 / 3)


def test_expected_heterozygosity():
    assert expected_heterozygosity([("A", "A"), ("A", "a"), ("a", "a")]) == pytest.approx(0.5)


def test_allele_stats():
    stats = allele_stats([("A", "A"), ("A", "a"), ("a", "a")])
    assert stats.num_genotypes == 3
    assert stats.num_alleles == 2


def test_hardy_weinberg_returns_p_in_range():
    genotypes = [("A", "A")] * 25 + [("A", "a")] * 50 + [("a", "a")] * 25
    chi2, p = hardy_weinberg_test(genotypes)
    assert 0.0 <= p <= 1.0


def test_hardy_weinberg_df_biallelic_regression():
    """Regression for lead L-09: df must be k(k-1)/2 = 1 for a biallelic locus.

    Derivation: k alleles give k(k+1)/2 genotype classes; allele frequencies
    estimated from the same sample consume (k-1) free parameters, so
    df = k(k+1)/2 - 1 - (k-1) = k(k-1)/2.
    """
    from scipy import stats as scipy_stats

    genotypes = [
        ("A", "A"), ("A", "T"), ("A", "T"), ("T", "T"), ("A", "A"),
        ("A", "T"), ("A", "A"), ("A", "A"), ("T", "T"), ("A", "T"),
    ]
    chi2, p = hardy_weinberg_test(genotypes)
    expected_p = float(scipy_stats.chi2.sf(chi2, 1))
    assert p == pytest.approx(expected_p, abs=1e-9)


def test_hardy_weinberg_df_multiallelic_regression():
    """Regression for lead L-09: df must be k(k-1)/2 = 3 for k = 3 alleles."""
    from scipy import stats as scipy_stats

    genotypes = [
        ("A", "A"), ("A", "C"), ("C", "G"), ("G", "G"), ("A", "G"),
        ("C", "C"), ("A", "C"), ("C", "G"), ("G", "A"), ("A", "A"),
        ("C", "C"), ("A", "G"),
    ]
    chi2, p = hardy_weinberg_test(genotypes)
    expected_p = float(scipy_stats.chi2.sf(chi2, 3))
    assert p == pytest.approx(expected_p, abs=1e-9)


def test_fst_two_pops():
    pop1 = [("A", "A"), ("A", "a"), ("a", "a")]
    pop2 = [("A", "A"), ("A", "A"), ("A", "a")]
    assert -1.0 < weir_cockerham_fst([pop1, pop2]) < 1.0


def test_fst_single_pop_raises():
    with pytest.raises(PopGenError):
        weir_cockerham_fst([("A", "A")])


def test_linkage_disequilibrium_range():
    ga = [("A", "A"), ("A", "a"), ("a", "a"), ("A", "a")]
    gb = [("B", "B"), ("B", "b"), ("b", "b"), ("B", "b")]
    dp, r2 = linkage_disequilibrium(ga, gb)
    assert -1.0 <= dp <= 1.0
    assert 0.0 <= r2 <= 1.0


def test_linkage_disequilibrium_mismatch():
    with pytest.raises(PopGenError):
        linkage_disequilibrium([("A", "A")], [("B", "B"), ("B", "b")])
