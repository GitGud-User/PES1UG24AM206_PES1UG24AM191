"""Small, inspectable versions of the reference paper's model families."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


class ClassConditionalKMeans(ClassifierMixin, BaseEstimator):
    def __init__(self, n_clusters=8, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError('Both classes are required')
        self.centers_ = [KMeans(n_clusters=min(self.n_clusters, int((y == c).sum())), n_init=10,
                               random_state=self.random_state).fit(X[y == c]).cluster_centers_ for c in self.classes_]
        return self

    def predict_proba(self, X):
        d0, d1 = [pairwise_distances(X, centers).min(axis=1) for centers in self.centers_]
        # Distance-derived ranking score; not a calibrated probability.
        p = (d0 + 1e-12) / (d0 + d1 + 2e-12)
        return np.column_stack([1-p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)


class JointKMeansRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y):
        # X and y are standardized by train-only outer transforms.
        joined = np.column_stack([X, y])
        self.centers_ = KMeans(n_clusters=self.n_clusters, n_init=10,
                              random_state=self.random_state).fit(joined).cluster_centers_
        return self

    def predict(self, X):
        closest = pairwise_distances(X, self.centers_[:, :-1]).argmin(axis=1)
        return self.centers_[closest, -1]
