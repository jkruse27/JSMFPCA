"""
Stage-2 inference.

Uses the estimated cross-spectral model together with the Gaussian BLUP to
estimate each subject's latent coordinated circadian rhythms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import CurveDataset, SubjectScores
from ..gaussian import GaussianBLUP
from .basis import HarmonicBasis
from .model import SpectralModel
from .operator import ObservationOperator
from .prior import PriorBuilder
from .subject import SpectralSubject


@dataclass(slots=True)
class SpectralInference:

    basis: HarmonicBasis
    model: SpectralModel
    operator: ObservationOperator
    prior_builder: PriorBuilder
    blup: GaussianBLUP

    def estimate_subject(
        self,
        subject: SubjectScores,
        noise_covariance: np.ndarray,
    ) -> SpectralSubject:
        """
        Estimate one subject.

        Parameters
        ----------
        subject
            Stage-1 scores for one subject.

        noise_covariance
            Observation noise covariance.

        Returns
        -------
        SpectralSubject
        """

        prior = self.prior_builder.build(self.model)

        H = self.operator.build(
            hours=subject.hours,
            basis=self.basis,
            model=self.model,
        )

        y = self.operator.response_vector(
            subject=subject,
            model=self.model,
        )

        posterior = self.blup.estimate(
            H=H,
            y=y,
            prior_covariance=prior.covariance(),
            noise_covariance=noise_covariance,
        )

        return SpectralSubject.from_posterior(
            posterior=posterior,
            subject=subject,
            model=self.model,
        )

    def estimate_dataset(
        self,
        dataset: CurveDataset,
        noise_covariance: np.ndarray,
    ) -> list[SpectralSubject]:

        return [
            self.estimate_subject(subject, noise_covariance)
            for subject in dataset.subject_scores()
        ]

    def reconstruct_subject(
        self,
        subject: SpectralSubject,
        hours: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Reconstruct the K-dimensional score trajectories.

        Returns
        -------
        ndarray
            Shape (24,K)
        """

        if hours is None:
            hours = np.arange(24)

        return subject.predict(
            basis=self.basis,
            hours=hours,
        )
