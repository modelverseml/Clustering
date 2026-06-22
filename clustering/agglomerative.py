"""
Agglomerative (Bottom-Up) Hierarchical Clustering
-------------------------------------------------
Every point starts as its own cluster. At each step we merge the two clusters
that are "closest" under a chosen linkage rule, and repeat until we reach the
target number of clusters (or one cluster containing everything).

Linkage rules supported:
    single   - distance between two clusters = distance of their closest pair
    complete - distance between two clusters = distance of their farthest pair
    average  - distance between two clusters = mean of all pairwise distances

These choices give very different cluster shapes — single linkage chains points
through narrow bridges, complete linkage prefers tight balls, average is in
between. Wards-linkage / variance-minimisation is omitted to keep the code
short; the three above cover most of the conceptual ground.

This is the brute-force O(N^3) implementation — fine for the small datasets
used in the notebook (a few hundred points).
"""

import numpy as np
import pandas as pd


class Agglomerative:

    def __init__(self, data, n_clusters, linkage='average'):

        self.data = data
        self.n_clusters = n_clusters
        self.linkage = linkage

    def _pair_distance(self, points_a, points_b):
        """Distance between two clusters under the chosen linkage rule."""

        # Pairwise Euclidean distances between every point in A and every point in B.
        diff = points_a[:, None, :] - points_b[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))

        if self.linkage == 'single':
            return dists.min()
        if self.linkage == 'complete':
            return dists.max()
        return dists.mean()  # 'average'

    def get_clusters(self):
        """Return a Series of cluster labels in [0, n_clusters)."""

        X = self.data.to_numpy(dtype=float) if hasattr(self.data, 'to_numpy') else np.asarray(self.data)

        # Each cluster is a list of point indices; start with every point alone.
        clusters = [[i] for i in range(len(X))]

        while len(clusters) > self.n_clusters:
            # Find the closest pair of clusters under the current linkage rule.
            best = (np.inf, -1, -1)
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    d = self._pair_distance(X[clusters[i]], X[clusters[j]])
                    if d < best[0]:
                        best = (d, i, j)

            _, i, j = best
            # Merge j into i; remove j (work from the back to keep indices valid).
            clusters[i] = clusters[i] + clusters[j]
            del clusters[j]

        # Convert the list-of-clusters representation back to a flat label array.
        labels = np.empty(len(X), dtype=int)
        for cluster_id, members in enumerate(clusters):
            labels[members] = cluster_id

        index = self.data.index if hasattr(self.data, 'index') else None
        return pd.Series(labels, index=index, name='cluster')
