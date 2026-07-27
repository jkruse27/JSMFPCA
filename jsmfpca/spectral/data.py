from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np


# ---------------------------------------------------------------------
# One subject
# ---------------------------------------------------------------------

@dataclass(slots=True)
class SpectralSubject:
    subject_id: str | int
    hours: np.ndarray
    offsets: np.ndarray
    centered: np.ndarray
    coefficients: np.ndarray
    rotated_coefficients: np.ndarray

    @property
    def n_harmonics(self):
        return self.coefficients.shape[0] // 2

    @property
    def n_modes(self):
        return self.coefficients.shape[1]

    @property
    def amplitudes(self):
        return np.sqrt(self.rotated_cosine**2 + self.rotated_sine**2)

    @property
    def phases(self):
        return np.arctan2(-self.rotated_sine, self.rotated_cosine)


# ---------------------------------------------------------------------
# Complete dataset
# ---------------------------------------------------------------------

@dataclass(slots=True)
class SpectralDataset:
    subjects: List[SpectralSubject]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    @property
    def n_subjects(self):
        return len(self.subjects)

    @property
    def n_harmonics(self):
        return self.eigenvalues.shape[0]

    @property
    def n_components(self):
        return self.eigenvalues.shape[1]

    @property
    def subject_ids(self):
        return [s.subject_id for s in self.subjects]

    def subject(self, subject_id):
        for s in self.subjects:
            if s.subject_id == subject_id:
                return s

        raise KeyError(subject_id)

    def subset(self, indices):
        return self.__class__(
            [self.subjects[i] for i in indices],
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors
        )

    def stack_offsets(self):
        return np.vstack([s.offsets for s in self.subjects])

    def stack_amplitudes(self):
        return np.stack([s.amplitudes for s in self.subjects])

    def stack_phases(self):
        return np.stack([s.phases for s in self.subjects])
