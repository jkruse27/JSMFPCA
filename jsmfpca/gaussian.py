"""
Generic Gaussian BLUP / PACE estimator.

Computes the posterior distribution

    b | y

under

    b ~ N(0, Σ_b)

    y = H b + ε

    ε ~ N(0,R)
"""

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
    """
    Generic multivariate BLUP / PACE estimator.
    """

    def __init__(
        self,
        ridge: float = 1e-8,
    ):
        self.ridge = ridge

    def estimate(
        self,
        H: np.ndarray,
        y: np.ndarray,
        prior_covariance: np.ndarray,
        noise_covariance: np.ndarray,
    ) -> Posterior:
        """
        Parameters
        ----------
        H
            Observation matrix.

        y
            Observation vector.

        prior_covariance
            Cov(b)

        noise_covariance
            Cov(ε)

        Returns
        -------
        Posterior
        """

        Sigma_b = np.asarray(prior_covariance)
        Sigma_e = np.asarray(noise_covariance)

        inv_prior = np.linalg.solve(
            Sigma_b + self.ridge, np.eye(Sigma_e.shape[0])
        )

        inv_noise = np.linalg.solve(
            Sigma_e + self.ridge, np.eye(Sigma_e.shape[0])
        )

        posterior_precision = (
            inv_prior +
            H.T @ inv_noise @ H
        )

        posterior_covariance = np.linalg.solve(
            posterior_precision, np.eye(Sigma_e.shape[0])
        )

        posterior_mean = (
            posterior_covariance
            @ H.T
            @ inv_noise
            @ y
        )

        return Posterior(
            mean=posterior_mean,
            covariance=posterior_covariance,
        )

    @property
    def std(self):
        return np.sqrt(np.diag(self.covariance))

    def reconstruct(
        self,
        H: np.ndarray,
        posterior: Posterior,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Posterior predictive mean and covariance.

        Returns
        -------
        mean
            H E[b|y]

        covariance
            H Cov(b|y) Hᵀ
        """

        mean = H @ posterior.mean

        covariance = (
            H
            @ posterior.covariance
            @ H.T
        )

        return mean, covariance
