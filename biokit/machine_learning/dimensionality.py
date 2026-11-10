"""Dimensionality reduction (PCA from scratch)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from biokit.exceptions import NotFittedError, ValidationError


class PCA:
    """Principal Component Analysis (NumPy SVD-based, no sklearn dependency).

    Parameters
    ----------
    n_components : int, optional
        Number of principal components to keep (default 2).

    Examples
    --------
    >>> pca = PCA(n_components=2)
    >>> X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    >>> X2 = pca.fit_transform(X)
    >>> X2.shape
    (3, 2)
    """

    def __init__(self, n_components: int = 2) -> None:
        if n_components < 1:
            raise ValidationError("n_components must be ≥ 1")
        self.n_components = n_components
        self._mean: NDArray[np.float64] | None = None
        self._components: NDArray[np.float64] | None = None
        self._explained_variance: NDArray[np.float64] | None = None
        self._total_variance: float | None = None

    def fit(self, X: NDArray[np.float64]) -> PCA:
        """Fit PCA on data."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValidationError("X must be 2-D")
        self._mean = X.mean(axis=0)
        X_centered = X - self._mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        n = min(self.n_components, len(S))
        self._components = Vt[:n]
        var = (S**2) / (X.shape[0] - 1) if X.shape[0] > 1 else S**2
        self._explained_variance = var[:n]
        # Total data variance across ALL components (for ratios).  The sum
        # of squared singular values equals the total variance of the
        # centered data, so the ratio of a selected PC is its share of the
        # TOTAL variance (sklearn convention), not of the selected subset.
        total = float((S**2).sum() / (X.shape[0] - 1)) if X.shape[0] > 1 else float((S**2).sum())
        self._total_variance = total
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project X onto principal components."""
        if self._mean is None or self._components is None:
            raise NotFittedError("PCA must be fitted before transform")
        X = np.asarray(X, dtype=float)
        projected: Any = (X - self._mean) @ self._components.T
        return np.asarray(projected, dtype=float)

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fit and project X."""
        self.fit(X)
        return self.transform(X)

    @property
    def explained_variance_ratio(self) -> NDArray[np.float64]:
        """Fraction of TOTAL data variance explained by each selected PC.

        Follows the sklearn ``PCA.explained_variance_ratio_`` convention:
        each entry is the component's variance divided by the variance of
        the whole dataset, so the entries over all data dimensions sum to 1
        and a subset of selected components sums to less than 1.
        """
        if self._explained_variance is None or self._total_variance is None:
            raise NotFittedError("PCA must be fitted first")
        if self._total_variance <= 0.0:
            raise ValidationError(
                "explained_variance_ratio is undefined for zero-variance data"
            )
        ratio: Any = self._explained_variance / self._total_variance
        return np.asarray(ratio, dtype=float)


__all__ = ["PCA"]
