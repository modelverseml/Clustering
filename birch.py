"""
BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)
--------------------------------------------------------------------
A streaming-friendly clustering algorithm. BIRCH compresses the dataset into
small "Clustering Features" (CFs) on a single pass, then runs an off-the-shelf
clustering algorithm on those CFs at the end. This means it only has to read
each point once and stores at most a few hundred CFs regardless of N.

For each subcluster we keep three statistics:

    CF = (N, LS, SS)
        N  = number of points in this CF
        LS = sum of points        (-> centroid = LS / N)
        SS = sum of squared norms (-> radius = sqrt(SS/N - ||LS/N||^2))

This implementation uses a *flat* list of CFs (the "leaders" simplification of
the full CF tree). Inserting a point either merges it into the closest CF if
that doesn't push the CF radius above `threshold`, or starts a new CF. After
the pass, K-Means on the CF centroids gives the final cluster labels.

This is enough to demonstrate the idea — full CF-tree BIRCH adds a balanced
tree on top for sub-linear search, which is mainly a speed optimisation.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans  # used for the final clustering of CFs


class _CF:
    """One Clustering Feature: a compact summary of a set of points."""

    def __init__(self, point):
        self.N = 1
        self.LS = point.astype(float).copy()
        self.SS = float((point ** 2).sum())

    @property
    def centroid(self):
        return self.LS / self.N

    def radius_after_add(self, point):
        """Cluster radius this CF would have if we added `point` to it."""

        new_N = self.N + 1
        new_LS = self.LS + point
        new_SS = self.SS + float((point ** 2).sum())
        mean = new_LS / new_N
        # radius = sqrt(SS/N - ||LS/N||^2)
        variance = new_SS / new_N - float((mean ** 2).sum())
        return np.sqrt(max(variance, 0))

    def add(self, point):
        self.N += 1
        self.LS += point
        self.SS += float((point ** 2).sum())


class BIRCH:

    def __init__(self, data, n_clusters, threshold=0.5):

        self.data = data
        self.n_clusters = n_clusters
        self.threshold = threshold

    def fit(self):

        X = self.data.to_numpy(dtype=float) if hasattr(self.data, 'to_numpy') else np.asarray(self.data, dtype=float)

        # Single-pass CF construction.
        cfs = []
        assignments = np.empty(len(X), dtype=int)

        for i, point in enumerate(X):
            if not cfs:
                cfs.append(_CF(point))
                assignments[i] = 0
                continue

            # Find the CF whose centroid is closest to this point.
            centroids = np.array([cf.centroid for cf in cfs])
            dists = np.linalg.norm(centroids - point, axis=1)
            best = int(np.argmin(dists))

            # Merge if it keeps the CF tight enough; otherwise start a fresh one.
            if cfs[best].radius_after_add(point) <= self.threshold:
                cfs[best].add(point)
                assignments[i] = best
            else:
                cfs.append(_CF(point))
                assignments[i] = len(cfs) - 1

        # Final clustering: run K-Means on the CF centroids and map labels back.
        cf_centroids = np.array([cf.centroid for cf in cfs])
        if len(cfs) <= self.n_clusters:
            cf_labels = np.arange(len(cfs))
        else:
            km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=0)
            cf_labels = km.fit_predict(cf_centroids)

        self.labels_ = cf_labels[assignments]
        self.cf_centroids_ = cf_centroids
        return self

    def get_clusters(self):

        if not hasattr(self, 'labels_'):
            self.fit()
        index = self.data.index if hasattr(self.data, 'index') else None
        return pd.Series(self.labels_, index=index, name='cluster')
