"""
spectral/shrinkage.py

Shrinkage estimators for cross-spectral matrices.
"""

from __future__ import annotations

import numpy as np


class DiagonalShrinkage:
    """
    Shrink Hermitian matrices toward their diagonal.

    S_shrunk =
        (1-alpha) S
        + alpha diag(S)
    """

    def __init__(self, alpha=0.0):
        self.alpha = float(alpha)

    def __call__(self, S):
        return shrink_to_diagonal(S, self.alpha)


# ---------------------------------------------------------------------
# Core shrinkage
# ---------------------------------------------------------------------

def shrink_to_diagonal(S, alpha):
    """
    Shrink one Hermitian matrix toward its diagonal.

    Parameters
    ----------
    S : (K,K)

    alpha : float
        0 = no shrinkage
        1 = diagonal only
    """

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

    return np.asarray([
        shrink_to_diagonal(M, a)
        for M, a in zip(S, alpha)
    ])


# ---------------------------------------------------------------------
# Eigen-decomposition
# ---------------------------------------------------------------------

def spectral_eigendecomposition(S):
    """
    Eigen-decompose one Hermitian matrix.

    Returns
    -------
    eigenvalues
    eigenvectors

    Sorted descending.
    """

    values, vectors = np.linalg.eigh(S)

    order = np.argsort(values)[::-1]

    return (
        values[order],
        vectors[:, order],
    )


def decompose_all(S):
    """
    Eigen-decompose every harmonic.

    Parameters
    ----------
    S : (R,K,K)

    Returns
    -------
    values
        (R,K)

    vectors
        (R,K,K)
    """

    R = len(S)
    K = S.shape[1]

    values = np.empty((R, K))
    vectors = np.empty((R, K, K), dtype=S.dtype)

    for r in range(R):

        values[r], vectors[r] = spectral_eigendecomposition(S[r])

    return values, vectors
