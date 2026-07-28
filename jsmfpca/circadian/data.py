from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np


# ---------------------------------------------------------------------
# One subject
# ---------------------------------------------------------------------

@dataclass(slots=True)
class CircadianSubject:
    subject_id: str | int
    hours: np.ndarray
    scores: np.ndarray
    fitted: np.ndarray
    offsets: np.ndarray
    residuals: np.ndarray
    centered: np.ndarray
    _hour_index: dict = field(init=False, repr=False)
    _lag_pairs: dict = field(init=False, repr=False)

    @property
    def n_hours(self):
        return len(self.hours)

    @property
    def n_components(self):
        return self.scores.shape[1]

    def component(self, k):
        return (
            self.hours,
            self.scores[:, k],
            self.fitted[:, k],
            self.residuals[:, k],
            self.centered[:, k],
        )

    def __post_init__(self):
        self._hour_index = {int(h): i for i, h in enumerate(self.hours)}
        self._lag_pairs = {}
        observed = set(self._hour_index)

        for lag in range(24):
            pairs = []

            for h in observed:
                h2 = (h + lag) % 24

                if h2 in observed:
                    pairs.append((self._hour_index[h], self._hour_index[h2]))

            self._lag_pairs[lag] = pairs


# ---------------------------------------------------------------------
# Complete dataset
# ---------------------------------------------------------------------

@dataclass
class CircadianDataset:
    subjects: List[CircadianSubject]
    harmonic_order: int
    population_coefficients: np.ndarray
    population_curves: np.ndarray
    residual_variances: np.ndarray

    def __post_init__(self):

        if len(self.subjects) == 0:
            raise ValueError("Dataset is empty.")

        self._n_components = self.subjects[0].n_components

    # -------------------------------------------------------------

    @property
    def n_subjects(self):
        return len(self.subjects)

    @property
    def n_components(self):
        return self._n_components

    @property
    def subject_ids(self):
        return [s.subject_id for s in self.subjects]

    # -------------------------------------------------------------

    def subject(self, subject_id):
        for s in self.subjects:
            if s.subject_id == subject_id:
                return s

        raise KeyError(subject_id)

    def subset(self, indices):
        return self.__class__([self.subjects[i] for i in indices])

    # -------------------------------------------------------------
    # Stack quantities
    # -------------------------------------------------------------

    def stack_scores(self):
        return np.vstack([s.scores for s in self.subjects])

    def stack_fitted(self):
        return np.vstack([s.fitted for s in self.subjects])

    def stack_residuals(self):
        return np.vstack([s.residuals for s in self.subjects])

    def stack_centered(self):
        return np.vstack([s.centered for s in self.subjects])

    def stack_hours(self):
        return np.concatenate([s.hours for s in self.subjects])

    def stack_subject_ids(self):
        ids = []

        for s in self.subjects:
            ids.extend([s.subject_id] * s.n_hours)

        return np.asarray(ids)

    # -------------------------------------------------------------
    # Iterators
    # -------------------------------------------------------------

    def iter_component(self, k):
        for s in self.subjects:
            for h, value in zip(s.hours, s.centered[:, k]):
                yield s.subject_id, int(h), float(value)

    def iter_hour(self, hour):
        for s in self.subjects:
            idx = np.where(s.hours == hour)[0]

            if len(idx):
                yield s.subject_id, s.centered[idx[0]]

    def observed_pairs(self, lag):
        lag %= 24

        for subject in self.subjects:
            for i, j in subject._lag_pairs[lag]:
                yield (
                    subject.subject_id, int(subject.hours[i]),
                    subject.centered[i], subject.centered[j]
                )

    # -------------------------------------------------------------

    def summary(self):
        n_obs = sum(s.n_hours for s in self.subjects)

        print("CircadianDataset")
        print("----------------")
        print(f"Subjects      : {self.n_subjects}")
        print(f"Components    : {self.n_components}")
        print(f"Harmonics     : {self.harmonic_order}")
        print(f"Observations  : {n_obs}")
