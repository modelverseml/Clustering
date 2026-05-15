"""
Gaussian Mixture Model (GMM) — Expectation Maximisation
-------------------------------------------------------
Models the data as a mixture of K multivariate Gaussians:

    p(x) = sum_k  pi_k * N(x | mu_k, Sigma_k)

Fitting is done by the EM algorithm, alternating two cheap steps until the
log-likelihood stops improving:

    E-step: gamma_{i,k} = pi_k * N(x_i | mu_k, Sigma_k) / sum_j pi_j * N(x_i | mu_j, Sigma_j)
            -- the soft "responsibility" of cluster k for point x_i.

    M-step: N_k    = sum_i gamma_{i,k}
            pi_k   = N_k / n
            mu_k   = sum_i gamma_{i,k} x_i / N_k
            Sigma_k = sum_i gamma_{i,k} (x_i - mu_k)(x_i - mu_k)^T / N_k

Gives soft cluster assignments (probabilities); calling `.get_clusters()` hardens
them with argmax. Covariance matrices are nudged by `reg_covar * I` each step
to stop them collapsing to a single point.
"""

import numpy as np
import pandas as pd


class GMM:

    def __init__(self, data, n_components, max_iter=100, tol=1e-4, reg_covar=1e-6, random_state=None):

        self.data = data
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.weights = None       # shape (K,)
        self.means = None         # shape (K, d)
        self.covariances = None   # shape (K, d, d)
        self.responsibilities = None  # shape (n, K)

    def _gaussian_pdf(self, X, mu, cov):
        """Multivariate-normal density at every row of X, given (mu, cov)."""

        d = X.shape[1]
        # `solve` is more stable than `inv` for the quadratic form below.
        diff = X - mu
        # Tiny ridge keeps Sigma invertible even when a cluster collapses.
        cov = cov + self.reg_covar * np.eye(d)
        try:
            inv_cov = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)
        except np.linalg.LinAlgError:
            return np.zeros(len(X))

        exponent = -0.5 * np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
        norm = 1.0 / np.sqrt(((2 * np.pi) ** d) * max(det_cov, 1e-300))
        return norm * np.exp(exponent)

    def fit(self):
        """Run EM until log-likelihood converges or max_iter is hit."""

        rng = np.random.default_rng(self.random_state)
        X = self.data.to_numpy(dtype=float) if hasattr(self.data, 'to_numpy') else np.asarray(self.data, dtype=float)
        n, d = X.shape
        K = self.n_components

        # Initialise: means by random sampling, covariances as identity, equal weights.
        self.means = X[rng.choice(n, size=K, replace=False)].astype(float)
        self.covariances = np.array([np.eye(d) for _ in range(K)])
        self.weights = np.ones(K) / K

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            # E-step.
            weighted = np.column_stack([
                self.weights[k] * self._gaussian_pdf(X, self.means[k], self.covariances[k])
                for k in range(K)
            ])
            total = weighted.sum(axis=1, keepdims=True)
            total[total == 0] = 1e-300  # guard against numerical zero rows
            gamma = weighted / total
            self.responsibilities = gamma

            # M-step.
            N_k = gamma.sum(axis=0)
            self.weights = N_k / n
            self.means = (gamma.T @ X) / N_k[:, None]
            for k in range(K):
                diff = X - self.means[k]
                self.covariances[k] = (gamma[:, k][:, None] * diff).T @ diff / N_k[k]

            # Convergence check on log-likelihood of the data.
            ll = np.log(total.flatten() + 1e-300).sum()
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def get_clusters(self):
        """Hard cluster labels by argmax over the soft responsibilities."""

        if self.responsibilities is None:
            self.fit()
        labels = self.responsibilities.argmax(axis=1)

        index = self.data.index if hasattr(self.data, 'index') else None
        return pd.Series(labels, index=index, name='cluster')
