"""Machine learning for biology.

Submodules:

* :mod:`biokit.machine_learning.classifiers` — supervised classifiers
* :mod:`biokit.machine_learning.clustering` — unsupervised clustering
* :mod:`biokit.machine_learning.dimensionality` — PCA (t-SNE / UMAP wrappers are not implemented)
* :mod:`biokit.machine_learning.encoders` — sequence/label encoders
* :mod:`biokit.machine_learning.features` — feature extraction
* :mod:`biokit.machine_learning.metrics` — evaluation metrics
* :mod:`biokit.machine_learning.splitters` — train/test splitters
* :mod:`biokit.machine_learning.datasets` — toy datasets
"""

from __future__ import annotations

__all__ = [
    "classifiers",
    "clustering",
    "datasets",
    "dimensionality",
    "encoders",
    "features",
    "metrics",
    "splitters",
]
