from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

@dataclass(slots=True)
class CosinorSubject:
    subject_id: str | int
    hours: np.ndarray
    observed: np.ndarray
    coefficients: np.ndarray
    reconstructed: np.ndarray
    residuals: np.ndarray


@dataclass(slots=True)
class CosinorDataset:
    subjects: list[CosinorSubject]
    n_components: int
    n_harmonics: int


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class CosinorModel:
    def __init__(self, n_harmonics=3):
        self.n_harmonics = n_harmonics

    def fit_subject(self, subject):
        coef = self._estimate_coefficients(subject.hours, subject.curves)
        reconstruction = self.predict(subject.hours, coef)

        return CosinorSubject(
            subject_id=subject.subject_id,
            hours=subject.hours.copy(),
            observed=subject.curves.copy(),
            coefficients=coef,
            reconstructed=reconstruction,
            residuals=subject.curves - reconstruction
        )

    def fit(self, dataset):
        subjects = [self.fit_subject(s) for s in dataset.subjects]

        return CosinorDataset(
            subjects=subjects,
            n_components=self.n_harmonics * 2,
            n_harmonics=self.n_harmonics,
        )

    def fit_transform(self, dataset, y=None):
        return self.fit(dataset).transform(dataset)

    def predict(self, hours, coefficients):
        X = self.design_matrix(hours)

        return X @ coefficients

    def design_matrix(self, hours):
        hours = np.asarray(hours)
        cols = [np.ones_like(hours)]
        omega = 2.0 * np.pi / 24.0

        for r in range(1, self.n_harmonics + 1):
            cols.append(np.cos(r * omega * hours))
            cols.append(np.sin(r * omega * hours))

        return np.column_stack(cols)

    def _estimate_coefficients(self, hours, centered):
        X = self.design_matrix(hours)
        coef, *_ = np.linalg.lstsq(X, centered, rcond=None)

        return coef

    @staticmethod
    def amplitude_phase(coefficients):
        n_terms = coefficients.shape[0]
        n_harmonics = (n_terms - 1) // 2
        amplitudes = []
        phases = []

        for r in range(n_harmonics):
            a = coefficients[1 + 2 * r]
            b = coefficients[2 + 2 * r]
            amp = np.sqrt(a * a + b * b)
            phase = np.arctan2(-b, a)
            amplitudes.append(amp)
            phases.append(phase)

        return np.asarray(amplitudes), np.asarray(phases)

    @staticmethod
    def mesor(coefficients):
        return coefficients[0]
