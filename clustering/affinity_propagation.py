"""
Affinity Propagation
--------------------
A message-passing algorithm that decides *both* the number of clusters and
which points act as cluster centres ("exemplars") by exchanging two kinds of
real-valued messages between every pair of points until they converge.

Notation (i, k are point indices):

    s(i, k)  similarity. We use s(i, k) = -||x_i - x_k||^2.
             s(k, k) is the "preference" — higher means k is more likely to
             become an exemplar (more clusters). We use the median of all
             off-diagonal similarities by default, matching scikit-learn.

    r(i, k)  responsibility — how well-suited k is to be the exemplar for i,
             relative to other candidates.

                r(i, k) = s(i, k) - max_{k' != k} ( a(i, k') + s(i, k') )

    a(i, k)  availability — how appropriate it would be for i to pick k as
             its exemplar, given the support k is getting from other points.

                a(i, k)  = min(0, r(k, k) + sum_{i' not in {i,k}} max(0, r(i', k)))    for i != k
                a(k, k)  = sum_{i' != k} max(0, r(i', k))

Damping keeps the updates stable; the final exemplar for point i is the k
that maximises r(i, k) + a(i, k).
"""

import numpy as np
import pandas as pd


class AffinityPropagation:

    def __init__(self, data, damping=0.9, max_iter=200, preference=None):

        self.data = data
        self.damping = damping
        self.max_iter = max_iter
        self.preference = preference

    def fit(self):
        """Iteratively update R and A; return cluster labels."""

        X = self.data.to_numpy(dtype=float) if hasattr(self.data, 'to_numpy') else np.asarray(self.data, dtype=float)
        n = len(X)

        # Negative squared Euclidean — sklearn's default similarity.
        diff = X[:, None, :] - X[None, :, :]
        S = -np.sum(diff ** 2, axis=2)

        # Self-similarity (preference) controls how readily points become exemplars.
        pref = self.preference if self.preference is not None else np.median(S[~np.eye(n, dtype=bool)])
        np.fill_diagonal(S, pref)

        R = np.zeros((n, n))
        A = np.zeros((n, n))

        for _ in range(self.max_iter):
            # Responsibility update.
            AS = A + S
            # For each i, we need max_{k' != k} (a(i,k') + s(i,k')).
            # Trick: find the top-2 values per row; if k is the argmax, use the runner-up.
            max_idx = np.argmax(AS, axis=1)
            max_val = AS[np.arange(n), max_idx]
            AS_masked = AS.copy()
            AS_masked[np.arange(n), max_idx] = -np.inf
            second_max = AS_masked.max(axis=1)

            new_R = S - max_val[:, None]
            new_R[np.arange(n), max_idx] = S[np.arange(n), max_idx] - second_max
            R = self.damping * R + (1 - self.damping) * new_R

            # Availability update.
            Rp = np.maximum(R, 0)
            np.fill_diagonal(Rp, np.diag(R))  # keep r(k,k) (could be negative)
            col_sums = Rp.sum(axis=0)
            new_A = col_sums[None, :] - Rp
            new_A = np.minimum(new_A, 0)
            # Self-availability: sum of positive responsibilities from others toward k.
            new_diag = (np.maximum(R, 0).sum(axis=0) - np.maximum(np.diag(R), 0))
            np.fill_diagonal(new_A, new_diag)
            A = self.damping * A + (1 - self.damping) * new_A

        # Exemplars are points where r(k,k) + a(k,k) > 0.
        criterion = np.diag(A) + np.diag(R)
        exemplar_indices = np.where(criterion > 0)[0]
        if len(exemplar_indices) == 0:
            # Fallback: pick the single best point if the algorithm did not converge to any exemplar.
            exemplar_indices = np.array([int(np.argmax(np.diag(A + R)))])

        # Assign each point to its preferred exemplar.
        labels = np.argmax(S[:, exemplar_indices], axis=1)
        self.exemplars_ = exemplar_indices
        self.labels_ = labels
        return self

    def get_clusters(self):

        if not hasattr(self, 'labels_'):
            self.fit()
        index = self.data.index if hasattr(self.data, 'index') else None
        return pd.Series(self.labels_, index=index, name='cluster')
