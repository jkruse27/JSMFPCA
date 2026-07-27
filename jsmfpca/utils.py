from __future__ import annotations
import numpy as np
from scipy.linalg import eigh


# =============================================================================
# Circular utilities
# =============================================================================

def wrap_hour(hours):
    return np.mod(hours, 24.0)


def circular_distance(h1, h2):
    d = (h1 - h2 + 12.0) % 24.0 - 12.0
    return d


def circular_mean(hours):
    hours = np.asarray(hours)
    theta = 2 * np.pi * hours / 24
    z = np.mean(np.exp(1j * theta))

    return (np.angle(z) % (2 * np.pi)) * 24 / (2 * np.pi)


# =============================================================================
# Harmonic basis
# =============================================================================

def harmonic_design_matrix(hours, n_harmonics):
    hours = np.asarray(hours)
    omega = 2 * np.pi / 24
    cols = [np.ones_like(hours)]

    for r in range(1, n_harmonics + 1):
        cols.append(np.cos(r * omega * hours))
        cols.append(np.sin(r * omega * hours))

    return np.column_stack(cols)


def harmonic_coefficients(hours, y, n_harmonics):
    X = harmonic_design_matrix(hours, n_harmonics)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    return beta


def harmonic_predict(hours, beta):
    n_harmonics = (len(beta) - 1) // 2
    X = harmonic_design_matrix(hours, n_harmonics)

    return X @ beta


# =============================================================================
# Functional inner products
# =============================================================================

def trapezoidal_weights(x):
    x = np.asarray(x)
    w = np.zeros_like(x)
    w[1:-1] = (x[2:] - x[:-2]) / 2
    w[0] = (x[1] - x[0]) / 2
    w[-1] = (x[-1] - x[-2]) / 2

    return w


def weighted_inner_product(x, y, weights):
    return np.sum(weights * x * y)


def project_function(curve, mean, basis, weights):
    centered = np.asarray(curve) - mean
    return (basis * weights) @ centered


def weighted_norm(x, weights):
    return np.sqrt(np.sum(weights * x**2))


# =============================================================================
# Covariance utilities
# =============================================================================

def covariance_matrix(X):
    X = np.asarray(X)
    X = X - X.mean(axis=0)

    return X.T @ X / (len(X) - 1)


def nearest_psd(A):
    vals, vecs = eigh(A)
    vals[vals < 0] = 0

    return vecs @ np.diag(vals) @ vecs.T


# =============================================================================
# FFT utilities
# =============================================================================

def cross_spectrum(covariance_lags):
    return np.fft.fft(covariance_lags, axis=0)


def inverse_cross_spectrum(spectrum):
    return np.real(np.fft.ifft(spectrum, axis=0))


# =============================================================================
# Eigenvector utilities
# =============================================================================

def align_eigenvector_sign(reference, vector):
    return -vector if np.dot(reference, vector) < 0 else vector


def sort_eigensystem(values, vectors):
    idx = np.argsort(values)[::-1]

    return values[idx], vectors[:, idx]


# =============================================================================
# Validation
# =============================================================================

def reconstruction_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)
