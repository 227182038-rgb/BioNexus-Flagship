"""Tests for biokit.machine_learning."""

from __future__ import annotations

import numpy as np
import pytest
from biokit.exceptions import NotFittedError
from biokit.machine_learning.classifiers import KNNClassifier
from biokit.machine_learning.clustering import KMeans
from biokit.machine_learning.datasets import make_blobs, make_classification
from biokit.machine_learning.dimensionality import PCA
from biokit.machine_learning.encoders import LabelEncoder, OneHotSequenceEncoder
from biokit.machine_learning.features import gc_content_features, kmer_features
from biokit.machine_learning.metrics import (
    accuracy,
    confusion_matrix,
    precision_recall_f1,
)
from biokit.machine_learning.splitters import k_fold, train_test_split


class TestKNN:
    """KNN classifier tests."""

    def test_predict(self):
        clf = KNNClassifier(k=1)
        X = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
        y = np.array(["a", "a", "b"])
        clf.fit(X, y)
        pred = clf.predict(np.array([[0.1, 0.1]]))
        assert pred[0] == "a"

    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            KNNClassifier(k=1).predict(np.array([[0.0, 0.0]]))


class TestKMeans:
    """KMeans clustering tests."""

    def test_fit_predict(self):
        X, _ = make_blobs(n_samples=30, centers=3, random_state=0)
        kmeans = KMeans(k=3, random_state=0)
        labels = kmeans.fit_predict(X)
        assert len(set(labels.tolist())) == 3

    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            KMeans(k=2).predict(np.array([[0.0, 0.0]]))


class TestPCA:
    """PCA tests."""

    def test_fit_transform(self):
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        X2 = PCA(n_components=2).fit_transform(X)
        assert X2.shape == (3, 2)

    def test_ratio_is_share_of_total_variance_regression(self):
        """Regression for lead L-11a.

        ``explained_variance_ratio`` must express each selected PC as a
        fraction of the TOTAL data variance (sklearn ``PCA`` semantics),
        not as a share of the *selected* components' variance.
        """
        rng = np.random.default_rng(42)
        X = rng.normal(size=(60, 5))
        full = PCA(n_components=5).fit(X).explained_variance_ratio
        partial = PCA(n_components=2).fit(X).explained_variance_ratio
        assert partial[0] == pytest.approx(full[0])  # truth: ~0.28, was [1.0]
        assert abs(partial.sum()) < 1.0  # 2 of 5 components < 100% of total

    def test_transform_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            PCA().transform(np.array([[0.0, 0.0]]))


class TestEncoders:
    """Encoder tests."""

    def test_label_encoder(self):
        enc = LabelEncoder()
        enc.fit(["a", "b", "a", "c"])
        assert enc.transform(["a", "b", "c"]).tolist() == [0, 1, 2]

    def test_one_hot_sequence_encoder(self):
        enc = OneHotSequenceEncoder()
        m = enc.transform("ATGC")
        assert m.shape == (4, 4)
        assert m[0, 0] == 1.0  # A → [1, 0, 0, 0]


class TestFeatures:
    """Feature extraction tests."""

    def test_kmer_features(self):
        X = kmer_features(["AAACCC", "GGGTTT"], k=2)
        assert X.shape == (2, 16)

    def test_gc_content_features(self):
        X = gc_content_features(["ATGC", "GGCC"])
        assert X.shape == (2, 1)


class TestMetrics:
    """Metric tests."""

    def test_accuracy(self):
        assert accuracy(["a", "b", "a"], ["a", "b", "b"]) == pytest.approx(2 / 3)

    def test_confusion_matrix(self):
        labels, mat = confusion_matrix(["a", "b"], ["a", "a"])
        assert labels == ["a", "b"]
        assert mat[0, 0] == 1
        assert mat[1, 0] == 1

    def test_precision_recall_f1(self):
        p, r, f1 = precision_recall_f1(["a", "a", "b"], ["a", "b", "b"], "a")
        assert 0 <= p <= 1
        assert 0 <= r <= 1


class TestSplitters:
    """Splitter tests."""

    def test_train_test_split(self):
        X = [1, 2, 3, 4, 5]
        X_train, X_test = train_test_split(X, test_size=0.4, random_state=0)
        assert len(X_train) + len(X_test) == 5

    def test_k_fold(self):
        splits = k_fold(10, k=5, random_state=0)
        assert len(splits) == 5
        for train_idx, test_idx in splits:
            assert len(train_idx) == 8
            assert len(test_idx) == 2


class TestDatasets:
    """Dataset tests."""

    def test_make_blobs(self):
        X, y = make_blobs(n_samples=30, centers=3, random_state=0)
        assert X.shape == (30, 2)
        assert len(set(y.tolist())) == 3

    def test_make_classification(self):
        X, y = make_classification(n_samples=20, n_classes=2, random_state=0)
        assert X.shape == (20, 4)
        assert len(set(y.tolist())) == 2
