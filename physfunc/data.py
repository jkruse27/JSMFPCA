from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass(slots=True)
class Observation:
    subject_id: str
    hour: int
    curve: np.ndarray

    def __post_init__(self):
        self.curve = np.asarray(self.curve, dtype=float)

        if self.curve.ndim != 1:
            raise ValueError("curve must be one-dimensional.")

        if not (0 <= self.hour <= 23):
            raise ValueError("hour must be between 0 and 23.")


# =============================================================================
# One subject
# =============================================================================

@dataclass
class SubjectCurves:
    subject_id: str
    hours: np.ndarray
    curves: np.ndarray
    metadata: Optional[dict] = field(default_factory=dict)

    # -------------------------------------------------------------------------

    def __post_init__(self):
        self.hours = np.asarray(self.hours, dtype=int)
        self.curves = np.asarray(self.curves, dtype=float)

        if self.hours.ndim != 1:
            raise ValueError("hours must be one-dimensional.")

        if self.curves.ndim != 2:
            raise ValueError("curves must be two-dimensional.")

        if len(self.hours) != self.curves.shape[0]:
            raise ValueError("hours and curves have inconsistent lengths.")

        if np.any(self.hours < 0) or np.any(self.hours > 23):
            raise ValueError("hours must be in [0,23].")

        if len(np.unique(self.hours)) != len(self.hours):
            raise ValueError(f"Duplicate hours detected in {self.subject_id}.")

        idx = np.argsort(self.hours)
        self.hours = self.hours[idx]
        self.curves = self.curves[idx]

    # -------------------------------------------------------------------------

    @property
    def n_hours(self):
        return len(self.hours)

    # -------------------------------------------------------------------------

    @property
    def n_scales(self):
        return self.curves.shape[1]

    # -------------------------------------------------------------------------

    def has_hour(self, hour: int):
        return hour in self.hours

    # -------------------------------------------------------------------------

    def get_curve(self, hour: int):
        idx = np.where(self.hours == hour)[0]

        if len(idx) == 0:
            return None

        return self.curves[idx[0]]

    def copy_with(self, **kwargs):
        import dataclasses
        return dataclasses.replace(self, **kwargs)

    # -------------------------------------------------------------------------

    def observations(self):
        for h, c in zip(self.hours, self.curves):
            yield Observation(self.subject_id, int(h), c)


# =============================================================================
# Complete dataset
# =============================================================================

class JSMFPCAData:
    def __init__(self, subjects: List[SubjectCurves], scales: np.ndarray):
        self.subjects = list(subjects)
        self.scales = np.asarray(scales, dtype=float)
        self._validate()

    def _validate(self):
        if len(self.subjects) == 0:
            raise ValueError("Dataset is empty.")

        n_scales = self.subjects[0].n_scales

        for subject in self.subjects:
            if subject.n_scales != n_scales:
                raise ValueError("Subjects must have identical scales.")

        if len(self.scales) != n_scales:
            raise ValueError("Scale vector has incompatible length.")

    @property
    def n_subjects(self):
        return len(self.subjects)

    @property
    def n_scales(self):
        return len(self.scales)

    @property
    def subject_ids(self):
        return [s.subject_id for s in self.subjects]

    @property
    def n_hours(self):
        if not self.subjects:
            return 0
        return len(self.subjects[0].hours)

    def stack_curves(self):
        return np.vstack([s.curves for s in self.subjects])

    def stack_hours(self):
        return np.concatenate([s.hours for s in self.subjects])

    def stack_subject_ids(self):
        ids = []
        for s in self.subjects:
            ids.extend([s.subject_id] * s.n_hours)
        return np.asarray(ids)

    def iter_observations(self):
        for subject in self.subjects:
            yield from subject.observations()

    def observed_pairs(self, lag):
        lag %= 24

        for subject in self.subjects:
            hour_map = {h: c for h, c in zip(subject.hours, subject.curves)}

            for h in subject.hours:
                h2 = (h + lag) % 24

                if h2 in hour_map:
                    yield (subject.subject_id, h, hour_map[h], hour_map[h2])

    def subject(self, subject_id):
        for subject in self.subjects:
            if subject.subject_id == subject_id:
                return subject

        raise KeyError(subject_id)

    def subset(self, indices):
        return self.__class__([self.subjects[i] for i in indices], self.scales)

    def copy_with(self, **kwargs):
        new_subjects = kwargs.get("subjects", self.subjects)
        new_scales = kwargs.get("scales", getattr(self, "scales", None))

        return self.__class__(subjects=new_subjects, scales=new_scales)

    def as_array(self):
        return np.array([subject.curves for subject in self.subjects])

    def mean_curve(self):
        if not self.subjects:
            raise ValueError("Empty dataset.")

        all_curves = np.vstack([subject.curves for subject in self.subjects])

        return np.mean(all_curves, axis=0)

    def summary(self):
        print("JSMFPCA Dataset")
        print("----------------")
        print(f"Subjects : {self.n_subjects}")
        print(f"Scales   : {self.n_scales}")

        observations = sum(s.n_hours for s in self.subjects)

        print(f"Observations : {observations}")
        print("Average observed hours :", observations / self.n_subjects)
