from __future__ import annotations
import numpy as np


class DiagonalShrinkage:
    def __init__(self, alpha=0.0):
        self.alpha = float(alpha)

    def __call__(self, S):
        return shrink_to_diagonal(S, self.alpha)


# ---------------------------------------------------------------------
# Core shrinkage
# ---------------------------------------------------------------------

def shrink_to_diagonal(S, alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    diag = np.diag(np.diag(S))

    return (1.0 - alpha) * S + alpha * diag


# ---------------------------------------------------------------------
# Multiple harmonics
# ---------------------------------------------------------------------

def shrink_all(S, alpha):
    alpha = np.asarray(alpha)

    if alpha.ndim == 0:
        alpha = np.full(len(S), alpha)

    return np.asarray([shrink_to_diagonal(M, a) for M, a in zip(S, alpha)])


# ---------------------------------------------------------------------
# Eigen-decomposition
# ---------------------------------------------------------------------

def spectral_eigendecomposition(S):
    values, vectors = np.linalg.eigh(S)
    order = np.argsort(values)[::-1]

    return values[order], vectors[:, order]


def decompose_all(S):
    R = len(S)
    K = S.shape[1]
    values = np.empty((R, K))
    vectors = np.empty((R, K, K), dtype=S.dtype)

    for r in range(R):
        values[r], vectors[r] = spectral_eigendecomposition(S[r])

    return values, vectors
