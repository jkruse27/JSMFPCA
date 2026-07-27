from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd

try:
    from sklearn.utils.extmath import randomized_svd
    HAS_RANDOMIZED_SVD = True
except ImportError:
    HAS_RANDOMIZED_SVD = False


# =============================================================================
# Dataclass
# =============================================================================

@dataclass(slots=True)
class SVDResult:
    """
    Result of a weighted SVD.

    This object stores only the learned FPCA basis.
    Subject scores are computed separately by `project()`.
    """

    mean: np.ndarray
    basis: np.ndarray
    eigenvalues: np.ndarray
    singular_values: np.ndarray
    weights: np.ndarray

    @property
    def n_components(self):
        return self.basis.shape[0]

    @property
    def n_scales(self):
        return self.basis.shape[1]


# =============================================================================
# Internal helpers
# =============================================================================

def _normalize_basis(phi: np.ndarray, weights: np.ndarray):
    """
    Normalize eigenfunctions under the weighted L2 inner product.
    """

    norms = np.sqrt(np.sum(phi**2 * weights, axis=1, keepdims=True))
    norms[norms == 0] = 1.0

    return phi / norms


def _align_signs(phi: np.ndarray):
    """
    Deterministic sign convention.

    The largest absolute loading of every eigenfunction
    is forced to be positive.
    """

    phi = phi.copy()

    for k in range(phi.shape[0]):
        idx = np.argmax(np.abs(phi[k]))

        if phi[k, idx] < 0:
            phi[k] *= -1

    return phi


# =============================================================================
# Weighted SVD
# =============================================================================

def weighted_svd(
    X: np.ndarray,
    weights: np.ndarray,
    n_components: int | None = None,
    randomized: bool = False,
    random_state: int | None = None,
):
    """
    Weighted thin SVD.

    Parameters
    ----------
    X (n_samples, n_scales)
        Raw (uncentered) curves.

    weights
        Quadrature weights.

    n_components
        Number of retained components.
        If None, compute all.

    randomized
        Use randomized SVD.

    random_state
        Random seed.

    Returns
    -------
    SVDResult
    """

    X = np.asarray(X, dtype=float)

    weights = np.asarray(weights, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")

    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional.")

    if X.shape[1] != len(weights):
        raise ValueError(
            "weights length must equal number of columns."
        )

    mean = X.mean(axis=0)
    Xc = X - mean
    sqrt_w = np.sqrt(weights)
    Xw = Xc * sqrt_w

    # ------------------------------------------------------------------

    if randomized:
        if not HAS_RANDOMIZED_SVD:
            raise ImportError("randomized_svd requires scikit-learn.")

        if n_components is None:
            raise ValueError("randomized SVD requires n_components.")

        U, S, Vt = randomized_svd(
            Xw,
            n_components=n_components,
            random_state=random_state,
        )

    else:

        U, S, Vt = svd(
            Xw,
            full_matrices=False,
            overwrite_a=False,
            check_finite=True,
        )

        if n_components is not None:
            U = U[:, :n_components]
            S = S[:n_components]
            Vt = Vt[:n_components]

    # ------------------------------------------------------------------
    # Recover eigenfunctions
    # ------------------------------------------------------------------

    phi = Vt / sqrt_w
    phi = _normalize_basis(phi, weights)
    phi = _align_signs(phi)
    eigenvalues = S**2 / (len(X) - 1)

    return SVDResult(
        mean=mean,
        basis=phi,
        eigenvalues=eigenvalues,
        singular_values=S,
        weights=weights,
    )


# =============================================================================
# Projection
# =============================================================================

def project_scores(curves, mean, basis, weights):
    """
    Project one or more curves onto the weighted FPCA basis.

    Parameters
    ----------
    curves
        Shape (n_scales,) or (n_samples, n_scales)

    Returns
    -------
    scores
        Shape (K,) or (n_samples, K)
    """

    curves = np.asarray(curves)
    single_curve = curves.ndim == 1

    if single_curve:
        curves = curves[None, :]

    centered = curves - mean
    scores = centered @ (basis * weights).T

    if single_curve:
        return scores[0]

    return scores


# =============================================================================
# Reconstruction
# =============================================================================

def reconstruct_curves(scores, mean, basis):
    """
    Reconstruct one or more curves.
    """

    scores = np.asarray(scores)
    single = scores.ndim == 1

    if single:
        scores = scores[None, :]

    curves = mean + scores @ basis

    if single:
        return curves[0]

    return curves


# =============================================================================
# Explained variance
# =============================================================================

def explained_variance_ratio(eigenvalues: np.ndarray):
    """
    Fraction of variance explained.
    """

    return eigenvalues / eigenvalues.sum()


def cumulative_variance(eigenvalues: np.ndarray):
    """
    Cumulative explained variance.
    """

    return np.cumsum(explained_variance_ratio(eigenvalues))


# =============================================================================
# Reconstruction error
# =============================================================================

def reconstruction_error(X: np.ndarray, reconstructed: np.ndarray):
    """
    Mean squared reconstruction error.
    """

    return np.mean((X - reconstructed) ** 2)


# =============================================================================
# Orthogonality check
# =============================================================================

def check_orthogonality(basis, weights, atol=1e-8):
    """
    Verify weighted orthonormality.

    Returns
    -------
    bool
    """

    G = basis @ np.diag(weights) @ basis.T

    return np.allclose(G, np.eye(len(G)), atol=atol)
