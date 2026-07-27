from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class Posterior:
    mean: np.ndarray
    covariance: np.ndarray

    n_basis: int | None = None
    n_modes: int | None = None


class GaussianBLUP:
    def __init__(self, ridge: float = 1e-8):
        self.ridge = ridge

    def estimate(self, H, y, prior_covariance, noise_covariance):
        Sigma_b = np.asarray(prior_covariance)
        Sigma_e = np.asarray(noise_covariance)
        Identity = np.eye(Sigma_e.shape[0])

        inv_prior = np.linalg.solve(Sigma_b + self.ridge, Identity)
        inv_noise = np.linalg.solve(Sigma_e + self.ridge, Identity)

        posterior_precision = inv_prior + H.T @ inv_noise @ H
        posterior_covariance = np.linalg.solve(posterior_precision, Identity)
        posterior_mean = posterior_covariance @ H.T @ inv_noise @ y

        return Posterior(mean=posterior_mean, covariance=posterior_covariance)

    @property
    def std(self):
        return np.sqrt(np.diag(self.covariance))

    def reconstruct(self, H, posterior):
        mean = H @ posterior.mean
        covariance = H @ posterior.covariance @ H.T

        return mean, covariance
