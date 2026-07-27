from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


# =============================================================================
# Circular utilities
# =============================================================================

def wrap_hour(hours):
    """
    Wrap hours into [0, 24).
    """
    return np.mod(hours, 24.0)


def circular_distance(h1, h2):
    """
    Shortest circular distance (hours).

    Returns values in [-12, 12).
    """
    d = (h1 - h2 + 12.0) % 24.0 - 12.0
    return d


def circular_mean(hours):
    """
    Circular mean of hours.

    Parameters
    ----------
    hours : array-like

    Returns
    -------
    float
        Mean hour in [0,24).
    """

    hours = np.asarray(hours)
    theta = 2 * np.pi * hours / 24
    z = np.mean(np.exp(1j * theta))

    return (np.angle(z) % (2 * np.pi)) * 24 / (2 * np.pi)


# =============================================================================
# Harmonic basis
# =============================================================================

def harmonic_design_matrix(hours, n_harmonics):
    """
    Construct harmonic regression matrix.

    Parameters
    ----------
    hours : array-like

    n_harmonics : int

    Returns
    -------
    X (n_samples, 2*n_harmonics + 1)
    """

    hours = np.asarray(hours)
    omega = 2 * np.pi / 24
    cols = [np.ones_like(hours)]

    for r in range(1, n_harmonics + 1):
        cols.append(np.cos(r * omega * hours))
        cols.append(np.sin(r * omega * hours))

    return np.column_stack(cols)


def harmonic_coefficients(hours, y, n_harmonics):
    """
    Ordinary least-squares harmonic regression.
    """

    X = harmonic_design_matrix(hours, n_harmonics)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    return beta


def harmonic_predict(hours, beta):
    """
    Predict from harmonic coefficients.
    """

    n_harmonics = (len(beta) - 1) // 2

    X = harmonic_design_matrix(hours, n_harmonics)

    return X @ beta


# =============================================================================
# Functional inner products
# =============================================================================

def trapezoidal_weights(x):
    """
    Trapezoidal integration weights.
    """

    x = np.asarray(x)
    w = np.zeros_like(x)
    w[1:-1] = (x[2:] - x[:-2]) / 2
    w[0] = (x[1] - x[0]) / 2
    w[-1] = (x[-1] - x[-2]) / 2

    return w


def weighted_inner_product(x, y, weights):
    """
    Weighted functional inner product.
    """

    return np.sum(weights * x * y)


def project_function(curve, mean, basis, weights):
    """
    FPCA projection.

    Parameters
    ----------
    basis (K, n_scales)

    Returns
    -------
    scores (K,)
    """

    centered = np.asarray(curve) - mean
    return (basis * weights) @ centered


def weighted_norm(x, weights):
    """
    Weighted L2 norm.
    """
    return np.sqrt(np.sum(weights * x**2))


# =============================================================================
# Covariance utilities
# =============================================================================

def covariance_matrix(X):
    """
    Sample covariance.

    X (n_samples, n_features)
    """

    X = np.asarray(X)
    X = X - X.mean(axis=0)

    return X.T @ X / (len(X) - 1)


def nearest_psd(A):
    """
    Project matrix onto positive semidefinite cone.
    """

    vals, vecs = eigh(A)
    vals[vals < 0] = 0

    return vecs @ np.diag(vals) @ vecs.T


# =============================================================================
# FFT utilities
# =============================================================================

def cross_spectrum(covariance_lags):
    """
    FFT of lag covariance sequence.

    Parameters
    ----------
    covariance_lags
        shape (24, K, K)

    Returns
    -------
    ndarray
        shape (24, K, K)
    """

    return np.fft.fft(covariance_lags, axis=0)


def inverse_cross_spectrum(spectrum):
    """
    Inverse FFT.
    """

    return np.real(np.fft.ifft(spectrum, axis=0))


# =============================================================================
# Eigenvector utilities
# =============================================================================

def align_eigenvector_sign(reference, vector):
    """
    Flip eigenvector sign to maximize agreement with reference.
    """

    if np.dot(reference, vector) < 0:
        return -vector

    return vector


def sort_eigensystem(values, vectors):
    """
    Descending eigenvalue order.
    """

    idx = np.argsort(values)[::-1]

    return values[idx], vectors[:, idx]


# =============================================================================
# Validation
# =============================================================================

def reconstruction_error(original, reconstructed):
    """
    Mean squared reconstruction error.
    """

    return np.mean((original - reconstructed) ** 2)
