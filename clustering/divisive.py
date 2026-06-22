"""
Divisive (Top-Down) Hierarchical Clustering — DIANA-style
---------------------------------------------------------
The opposite of Agglomerative. Start with all points in one cluster and at
each step pick the cluster with the largest "diameter" and split it. The
split mechanism is the classical DIANA "splinter group" rule:

    1. Find the point with the largest mean dissimilarity to the rest of the
       cluster. That point seeds a new splinter group S.
    2. For every remaining point, if it is closer (on average) to S than to
       R = C \\ S, move it from R to S.
    3. Repeat (2) until no more points want to move.

We keep splitting until we have `n_clusters` clusters in total.
"""

import numpy as np
import pandas as pd


class Divisive:

    def __init__(self, data, n_clusters):

        self.data = data
        self.n_clusters = n_clusters

    def _diameter(self, X_cluster):
        """Cluster diameter = max pairwise distance — used to pick which cluster to split."""

        if len(X_cluster) <= 1:
            return 0.0
        diff = X_cluster[:, None, :] - X_cluster[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=2)).max()

    def _split(self, member_indices, X):
        """Apply DIANA's splinter-group rule to one cluster, returning two index lists."""

        cluster = X[member_indices]

        # Average dissimilarity of each point to the rest of the cluster.
        diff = cluster[:, None, :] - cluster[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        n = len(cluster)
        mean_dist = dists.sum(axis=1) / max(n - 1, 1)

        # Seed the splinter group with the most-dissimilar point.
        seed = int(np.argmax(mean_dist))
        in_splinter = np.zeros(n, dtype=bool)
        in_splinter[seed] = True

        # Iteratively move points whose avg distance to S is smaller than to R.
        moved = True
        while moved:
            moved = False
            for i in range(n):
                if in_splinter[i]:
                    continue
                splinter_mask = in_splinter
                remaining_mask = ~in_splinter
                remaining_mask[i] = False  # exclude self when averaging over R

                d_to_s = dists[i, splinter_mask].mean() if splinter_mask.any() else np.inf
                d_to_r = dists[i, remaining_mask].mean() if remaining_mask.any() else np.inf

                if d_to_s < d_to_r:
                    in_splinter[i] = True
                    moved = True

        splinter = [member_indices[i] for i in range(n) if in_splinter[i]]
        remaining = [member_indices[i] for i in range(n) if not in_splinter[i]]
        return splinter, remaining

    def get_clusters(self):
        """Return a Series of cluster labels in [0, n_clusters)."""

        X = self.data.to_numpy(dtype=float) if hasattr(self.data, 'to_numpy') else np.asarray(self.data)

        clusters = [list(range(len(X)))]

        while len(clusters) < self.n_clusters:
            # Split the cluster with the largest diameter.
            diameters = [self._diameter(X[c]) for c in clusters]
            target = int(np.argmax(diameters))

            if diameters[target] == 0:
                break  # nothing left to split (all remaining clusters are singletons)

            splinter, remaining = self._split(clusters[target], X)

            # If the splinter or the remainder is empty, splitting can't progress further.
            if not splinter or not remaining:
                break

            clusters[target] = remaining
            clusters.append(splinter)

        labels = np.empty(len(X), dtype=int)
        for cluster_id, members in enumerate(clusters):
            labels[members] = cluster_id

        index = self.data.index if hasattr(self.data, 'index') else None
        return pd.Series(labels, index=index, name='cluster')
