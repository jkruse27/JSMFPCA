from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------

@dataclass(slots=True)
class HarmonicFit:
    order: int
    coefficients: np.ndarray
    residual_variance: float
    r2: float

    @property
    def intercept(self):
        return self.coefficients[0]

    @property
    def n_coefficients(self):
        return len(self.coefficients)


# ---------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------

def harmonic_design_matrix(hours, order):
    hours = np.asarray(hours, dtype=float)
    omega = 2 * np.pi / 24
    cols = [np.ones_like(hours)]

    for r in range(1, order + 1):
        cols.append(np.cos(r * omega * hours))
        cols.append(np.sin(r * omega * hours))

    return np.column_stack(cols)


# ---------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------

def fit_harmonic(hours, values, order=2):
    hours = np.asarray(hours)
    values = np.asarray(values)

    X = harmonic_design_matrix(hours, order)

    beta, *_ = np.linalg.lstsq(X, values, rcond=None)
    fitted = X @ beta
    residuals = values - fitted

    rss = np.sum(residuals**2)
    tss = np.sum((values - values.mean())**2)

    if len(values) > len(beta):
        sigma2 = rss / (len(values) - len(beta))
    else:
        sigma2 = rss

    r2 = 1.0 if tss == 0 else 1 - rss / tss

    return HarmonicFit(
        order=order, coefficients=beta, residual_variance=sigma2, r2=r2
    )


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

def predict_harmonic(fit, hours):
    X = harmonic_design_matrix(hours, fit.order)
    return X @ fit.coefficients


# ---------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------

def harmonic_residuals(fit, hours, values):
    values = np.asarray(values)
    return values - predict_harmonic(fit, hours)


# ---------------------------------------------------------------------
# Harmonic amplitudes
# ---------------------------------------------------------------------

def harmonic_amplitudes(fit):
    amps = []
    beta = fit.coefficients

    for r in range(fit.order):
        a = beta[1 + 2 * r]
        b = beta[2 + 2 * r]
        amps.append(np.sqrt(a * a + b * b))

    return np.asarray(amps)


# ---------------------------------------------------------------------
# Harmonic phases
# ---------------------------------------------------------------------

def harmonic_phases(fit):
    phases = []
    beta = fit.coefficients

    for r in range(fit.order):
        a = beta[1 + 2 * r]
        b = beta[2 + 2 * r]
        phases.append(np.arctan2(-b, a))

    return np.asarray(phases)


# ---------------------------------------------------------------------
# Hour of maximum
# ---------------------------------------------------------------------

def acrophases(fit):
    omega = 2 * np.pi / 24
    hours = []

    for phase, r in zip(harmonic_phases(fit), range(1, fit.order + 1)):
        h = (-phase / (omega * r)) % 24
        hours.append(h)

    return np.asarray(hours)


# ---------------------------------------------------------------------
# Evaluate on the full day
# ---------------------------------------------------------------------

def daily_curve(fit):
    return predict_harmonic(fit, np.arange(24))


# ---------------------------------------------------------------------
# Information criteria
# ---------------------------------------------------------------------

def aic(fit, n):
    k = fit.n_coefficients
    sigma2 = max(fit.residual_variance, 1e-12)

    return n * np.log(sigma2) + 2 * k


def bic(fit, n):
    k = fit.n_coefficients
    sigma2 = max(fit.residual_variance, 1e-12)

    return n * np.log(sigma2) + k * np.log(n)
