from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..gaussian import GaussianBLUP
from .fourier import FourierBasis
from .model import SpectralModel
from .operator import ObservationOperator
from .prior import PriorBuilder
from .data import SpectralSubject


@dataclass(slots=True)
class SpectralInference:
    basis: FourierBasis
    model: SpectralModel
    operator: ObservationOperator
    prior_builder: PriorBuilder
    blup: GaussianBLUP

    def estimate_subject(self, subject, noise_covariance):
        prior = self.prior_builder.build(self.model)

        H = self.operator.build(
            hours=subject.hours, basis=self.basis, model=self.model
        )

        y = self.operator.response_vector(
            subject=subject, model=self.model
        )

        posterior = self.blup.estimate(
            H=H, y=y, prior_covariance=prior.covariance(),
            noise_covariance=noise_covariance
        )

        return SpectralSubject.from_posterior(
            posterior=posterior, subject=subject, model=self.model
        )

    def estimate_dataset(self, dataset, noise_covariance):
        return [
            self.estimate_subject(subject, noise_covariance)
            for subject in dataset.subject_scores()
        ]

    def reconstruct_subject(self, subject, hours=None):
        if hours is None:
            hours = np.arange(24)

        return subject.predict(basis=self.basis, hours=hours)
