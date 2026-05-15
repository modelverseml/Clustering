"""
Mean Shift Clustering
---------------------
A mode-seeking, non-parametric algorithm. Treats the data as a sample from
some unknown density and shifts every point toward the nearest *mode* of that
density. Points that converge to the same mode form a cluster.

Update rule (flat / uniform kernel of bandwidth h):

    m(x) = mean of points within distance h of x       (the local mean)
    x   <- m(x)

Repeat until x stops moving. Then a clean-up step merges modes that are within
bandwidth `h` of each other into a single cluster.

Mean Shift does **not** require the number of clusters in advance — the
bandwidth `h` is the only knob. Small `h` → many clusters; large `h` → few.
"""

import numpy as np
import pandas as pd


class MeanShift:

    def __init__(self, data, bandwidth, max_iter=300, tol=1e-3):

        self.data = data
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol

    def _shift(self, x, X):
        """Return the mean of points within bandwidth `h` of x. Falls back to x itself."""

        distances = np.linalg.norm(X - x, axis=1)
        neighbours = X[distances <= self.bandwidth]
        if len(neighbours) == 0:
            return x
        return neighbours.mean(axis=0)

    def fit(self):
        """Shift every point until convergence, then merge nearby modes."""

        X = self.data.to_numpy(dtype=float) if hasattr(self.data, 'to_numpy') else np.asarray(self.data, dtype=float)

        # Each point gets its own trajectory; we run them all in parallel.
        positions = X.copy()
        for _ in range(self.max_iter):
            new_positions = np.array([self._shift(p, X) for p in positions])
            if np.max(np.linalg.norm(new_positions - positions, axis=1)) < self.tol:
                positions = new_positions
                break
            positions = new_positions

        # Cluster the converged positions: any two within `h` belong to the same mode.
        labels = -np.ones(len(X), dtype=int)
        modes = []
        for i, p in enumerate(positions):
            assigned = False
            for k, mode in enumerate(modes):
                if np.linalg.norm(p - mode) <= self.bandwidth:
                    labels[i] = k
                    assigned = True
                    break
            if not assigned:
                modes.append(p)
                labels[i] = len(modes) - 1

        self.cluster_centers_ = np.array(modes)
        self.labels_ = labels
        return self

    def get_clusters(self):

        if not hasattr(self, 'labels_'):
            self.fit()
        index = self.data.index if hasattr(self.data, 'index') else None
        return pd.Series(self.labels_, index=index, name='cluster')
