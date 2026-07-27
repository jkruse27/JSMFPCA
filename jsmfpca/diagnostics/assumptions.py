from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import subspace_angles


# ---------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------

@dataclass(slots=True)
class A2Result:
    reconstruction_error: float
    mean_principal_angle: float
    max_principal_angle: float
    rv_coefficient: float


@dataclass(slots=True)
class A4Result:
    relative_error: float
    eigenvalue_correlation: float
    mean_principal_angle: float
    max_principal_angle: float


# ---------------------------------------------------------------------
# Shared FPCA basis (A2)
# ---------------------------------------------------------------------

def check_shared_basis(pooled_basis, reference_basis):
    pooled_basis = _orthonormalize(pooled_basis)
    reference_basis = _orthonormalize(reference_basis)

    angles = np.rad2deg(subspace_angles(pooled_basis, reference_basis))
    projection = pooled_basis @ pooled_basis.T
    reconstruction = projection @ reference_basis

    error = (
        np.linalg.norm(reference_basis - reconstruction)
        / np.linalg.norm(reference_basis)
    )

    rv = _rv_coefficient(
        pooled_basis @ pooled_basis.T,
        reference_basis @ reference_basis.T
    )

    return A2Result(
        reconstruction_error=float(error),
        mean_principal_angle=float(np.mean(angles)),
        max_principal_angle=float(np.max(angles)),
        rv_coefficient=float(rv),
    )


def project_block_circulant(C):
    n = 24

    if C.shape[0] % n != 0:
        raise ValueError("Matrix dimension must be divisible by 24.")

    K = C.shape[0] // n
    blocks = C.reshape(n, K, n, K).transpose(0, 2, 1, 3)
    lag_blocks = np.empty((n, K, K))

    for d in range(n):
        lag_blocks[d] = np.mean(
            [blocks[i, (i + d) % n] for i in range(n)], axis=0
        )

    projected = np.empty_like(blocks)

    for i in range(n):
        for j in range(n):
            projected[i, j] = lag_blocks[(j - i) % n]

    return projected.transpose(0, 2, 1, 3).reshape(n * K, n * K)


# ---------------------------------------------------------------------
# Joint stationarity (A4)
# ---------------------------------------------------------------------

def check_stationarity(empirical_covariance, harmonic_covariance=None):
    projected = project_block_circulant(empirical_covariance)
    projection_error = (
        np.linalg.norm(empirical_covariance - projected)
        / np.linalg.norm(empirical_covariance)
    )

    if harmonic_covariance is None:
        return A4Result(
            relative_error=float(projection_error),
            eigenvalue_correlation=np.nan,
            mean_principal_angle=np.nan,
            max_principal_angle=np.nan,
        )

    val1, vec1 = np.linalg.eigh(empirical_covariance)
    val2, vec2 = np.linalg.eigh(harmonic_covariance)

    order1 = np.argsort(val1)[::-1]
    order2 = np.argsort(val2)[::-1]

    val1 = val1[order1]
    val2 = val2[order2]

    vec1 = vec1[:, order1]
    vec2 = vec2[:, order2]

    angles = np.rad2deg(subspace_angles(vec1, vec2))
    corr = np.corrcoef(val1, val2)[0, 1]

    return A4Result(
        relative_error=float(projection_error),
        eigenvalue_correlation=float(corr),
        mean_principal_angle=float(np.mean(angles)),
        max_principal_angle=float(np.max(angles)),
    )


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _orthonormalize(X):
    Q, _ = np.linalg.qr(X)

    return Q


def _rv_coefficient(A, B):
    numerator = np.trace(A @ B)
    denominator = np.sqrt(np.trace(A @ A) * np.trace(B @ B))

    return numerator / denominator
