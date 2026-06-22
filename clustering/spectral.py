"""
Spectral Clustering
-------------------
Treats clustering as a graph-partitioning problem. The trick is that the
eigenvectors of the graph Laplacian *embed* the data in a low-dimensional
space where the cluster structure becomes a simple K-Means problem.

Algorithm:

    1. Build a similarity graph from the data. We use the Gaussian kernel
            S_{ij} = exp(-||x_i - x_j||^2 / (2 sigma^2))    for i != j
            S_{ii} = 0
    2. Form the degree matrix D = diag(S.sum(axis=1)) and the *normalised*
       symmetric Laplacian
            L_sym = I - D^{-1/2} S D^{-1/2}
    3. Take the first k eigenvectors of L_sym (smallest k eigenvalues).
    4. Stack them as columns of U (shape n x k), normalise each row to unit
       length, then run K-Means on U's rows to get k clusters.

Spectral clustering handles non-convex clusters that K-Means cannot — moons,
circles, etc. — because the embedding step linearises them.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans  # only used for the final step in eigenspace


class SpectralClustering:

    def __init__(self, data, n_clusters, sigma=1.0, random_state=None):

        self.data = data
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.random_state = random_state

    def fit(self):

        X = self.data.to_numpy(dtype=float) if hasattr(self.data, 'to_numpy') else np.asarray(self.data, dtype=float)
        n = len(X)

        # 1. Gaussian similarity matrix (with zero diagonal).
        diff = X[:, None, :] - X[None, :, :]
        sq_dist = (diff ** 2).sum(axis=2)
        S = np.exp(-sq_dist / (2 * self.sigma ** 2))
        np.fill_diagonal(S, 0.0)

        # 2. Symmetric normalised Laplacian: L_sym = I - D^{-1/2} S D^{-1/2}.
        d_inv_sqrt = 1.0 / np.sqrt(S.sum(axis=1))
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L_sym = np.eye(n) - D_inv_sqrt @ S @ D_inv_sqrt

        # 3. Take the k eigenvectors with the smallest eigenvalues. `eigh` is
        # for symmetric matrices and returns eigenvalues in ascending order.
        _, eigvecs = np.linalg.eigh(L_sym)
        U = eigvecs[:, :self.n_clusters]

        # 4. Row-normalise (per Ng-Jordan-Weiss) and cluster the rows.
        row_norm = np.linalg.norm(U, axis=1, keepdims=True)
        row_norm[row_norm == 0] = 1.0
        U_norm = U / row_norm

        km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        self.labels_ = km.fit_predict(U_norm)
        return self

    def get_clusters(self):

        if not hasattr(self, 'labels_'):
            self.fit()
        index = self.data.index if hasattr(self.data, 'index') else None
        return pd.Series(self.labels_, index=index, name='cluster')
