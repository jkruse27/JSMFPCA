from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

@dataclass(slots=True)
class MFPCAResult:
    mean: np.ndarray
    visit_mean: np.ndarray

    phi: np.ndarray
    psi: np.ndarray

    lambda_phi: np.ndarray
    lambda_psi: np.ndarray

    explained_between: np.ndarray
    explained_within: np.ndarray

    @property
    def n_between(self):
        return self.phi.shape[1]

    @property
    def n_within(self):
        return self.psi.shape[1]

    @property
    def n_timepoints(self):
        return self.phi.shape[0]


# ---------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------

@dataclass(slots=True)
class SubjectScores:
    subject_id: str | int
    xi: np.ndarray
    zeta: np.ndarray


@dataclass(slots=True)
class ScoreDataset:
    subjects: list[SubjectScores]

    @property
    def n_subjects(self):
        return len(self.subjects)

    @property
    def n_between(self):
        return self.subjects[0].xi.size

    @property
    def n_within(self):
        return self.subjects[0].zeta.shape[-1]
