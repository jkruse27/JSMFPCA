"""
Estimation of subject-specific Fourier coefficients.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .fourier import FourierBasis
from .prior import SpectralPrior


# ---------------------------------------------------------------------
# Base estimator
# ---------------------------------------------------------------------

class BaseCoefficientEstimator(ABC):
    """
    Base class for coefficient estimators.
    """

    @abstractmethod
    def fit(
        self,
        basis: FourierBasis,
        hours: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate Fourier coefficients.

        Parameters
        ----------
        basis
            Fourier basis.

        hours
            Observed hours.

        values
            Matrix (n_hours, K).

        Returns
        -------
        coefficients
            Matrix (2R, K).
        """


# ---------------------------------------------------------------------
# Ordinary least squares
# ---------------------------------------------------------------------

@dataclass(slots=True)
class OLSCoefficientEstimator(BaseCoefficientEstimator):

    def fit(
        self,
        basis: FourierBasis,
        hours: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:

        return basis.fit(
            hours,
            values,
        )


# ---------------------------------------------------------------------
# Multivariate BLUP / PACE
# ---------------------------------------------------------------------

@dataclass(slots=True)
class BLUPCoefficientEstimator(BaseCoefficientEstimator):
    """
    Posterior mean estimator of Fourier coefficients.

    Parameters
    ----------
    coefficient_covariance
        Prior covariance of vec(B).

    noise_variance
        Observation noise variance.

    ridge
        Numerical stabilization.
    """

    prior: SpectralPrior
    noise_variance: float = 1.0
    ridge: float = 1e-8

    def fit(
        self,
        basis: FourierBasis,
        hours: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:

        X = basis.design_matrix(hours)

        n_obs = len(hours)
        n_basis = X.shape[1]
        n_modes = values.shape[1]

        y = values.reshape(-1, order="F")

        H = np.kron(np.eye(n_modes), X)

        Sigma_b = self.prior.covariance()

        Sigma_e = self.noise_variance * np.eye(
            n_obs * n_modes
        )

        Sigma_y = (
            H @ Sigma_b @ H.T
            + Sigma_e
        )

        Sigma_y.flat[:: Sigma_y.shape[0] + 1] += self.ridge

        gain = Sigma_b @ H.T @ np.linalg.solve(
            Sigma_y,
            np.eye(Sigma_y.shape[0]),
        )

        b = gain @ y

        return b.reshape(
            n_basis,
            n_modes,
            order="F",
        )
