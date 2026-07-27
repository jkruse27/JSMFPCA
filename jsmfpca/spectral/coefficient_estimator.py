from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from .prior import SpectralPrior


# ---------------------------------------------------------------------
# Base estimator
# ---------------------------------------------------------------------

class BaseCoefficientEstimator(ABC):
    @abstractmethod
    def fit(self, basis, hours, values):
        ...


# ---------------------------------------------------------------------
# Ordinary least squares
# ---------------------------------------------------------------------

@dataclass(slots=True)
class OLSCoefficientEstimator(BaseCoefficientEstimator):
    def fit(self, basis, hours, values):
        return basis.fit(hours, values)


# ---------------------------------------------------------------------
# Multivariate BLUP / PACE
# ---------------------------------------------------------------------

@dataclass(slots=True)
class BLUPCoefficientEstimator(BaseCoefficientEstimator):
    prior: SpectralPrior
    noise_variance: float = 1.0
    ridge: float = 1e-8

    def fit(self, basis, hours, values):
        X = basis.design_matrix(hours)
        n_obs = len(hours)
        n_basis = X.shape[1]
        n_modes = values.shape[1]
        y = values.reshape(-1, order="F")
        H = np.kron(np.eye(n_modes), X)
        Sigma_b = self.prior.covariance()
        Sigma_e = self.noise_variance * np.eye(n_obs * n_modes)
        Sigma_y = H @ Sigma_b @ H.T + Sigma_e
        Sigma_y.flat[:: Sigma_y.shape[0] + 1] += self.ridge
        tmp_I = np.eye(Sigma_y.shape[0])
        gain = Sigma_b @ H.T @ np.linalg.solve(Sigma_y, tmp_I)
        b = gain @ y

        return b.reshape(n_basis, n_modes, order="F")
