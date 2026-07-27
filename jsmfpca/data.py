from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from copy import deepcopy

import numpy as np


# =============================================================================
# Individual observations
# =============================================================================


@dataclass(slots=True)
class BaseEstimator:
    def clone(self):
        return deepcopy(self)

    def get_params(self):
        return vars(self).copy()

    def set_params(self, **params):

        for key, value in params.items():

            setattr(self, key, value)

        return self


@dataclass(slots=True)
class Observation:
    """
    A single observed curve.

    Parameters
    ----------
    subject_id
        Subject identifier.

    hour
        Hour-of-day (0-23).

    curve
        Nonlinear HRV curve evaluated over the scale axis.
        Shape = (n_scales,)
    """

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
    """
    Curves observed for one subject.

    Parameters
    ----------
    subject_id

    hours
        Array of observed hours.

    curves
        Array of observed curves.

        Shape
        -----
        (n_observed_hours,
         n_scales)

    metadata
        Optional subject-level information.
    """

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

    # -------------------------------------------------------------------------

    def observations(self):
        for h, c in zip(self.hours, self.curves):
            yield Observation(self.subject_id, int(h), c)


# =============================================================================
# FPCA score container
# =============================================================================

@dataclass
class ScoreMatrix:
    """
    FPCA scores for one subject.

    Shape
    -----

    scores

        (n_observed_hours,
         n_components)

    """

    subject_id: str
    hours: np.ndarray
    scores: np.ndarray

    def __post_init__(self):
        self.hours = np.asarray(self.hours)
        self.scores = np.asarray(self.scores)

        if self.scores.ndim != 2:
            raise ValueError("scores must be 2D.")

        if len(self.hours) != self.scores.shape[0]:
            raise ValueError("hours and scores have incompatible shapes.")

    @property
    def n_components(self):
        return self.scores.shape[1]


# =============================================================================
# Complete dataset
# =============================================================================

class JSMFPCAData:
    """
    Collection of SubjectCurves.

    Provides convenient access methods used throughout the pipeline.
    """

    # -------------------------------------------------------------------------

    def __init__(self, subjects: List[SubjectCurves], scales: np.ndarray):

        self.subjects = list(subjects)
        self.scales = np.asarray(scales, dtype=float)
        self._validate()

    # -------------------------------------------------------------------------

    def _validate(self):
        if len(self.subjects) == 0:
            raise ValueError("Dataset is empty.")

        n_scales = self.subjects[0].n_scales

        for subject in self.subjects:
            if subject.n_scales != n_scales:
                raise ValueError("Subjects must have identical scales.")

        if len(self.scales) != n_scales:
            raise ValueError("Scale vector has incompatible length.")

    # -------------------------------------------------------------------------

    @property
    def n_subjects(self):
        return len(self.subjects)

    # -------------------------------------------------------------------------

    @property
    def n_scales(self):
        return len(self.scales)

    # -------------------------------------------------------------------------

    @property
    def subject_ids(self):
        return [s.subject_id for s in self.subjects]

    # -------------------------------------------------------------------------

    def stack_curves(self):
        """
        Stack every observed curve.

        Returns
        -------
        ndarray

            Shape

            (total_observations,
             n_scales)
        """

        return np.vstack([s.curves for s in self.subjects])

    # -------------------------------------------------------------------------

    def stack_hours(self):
        return np.concatenate([s.hours for s in self.subjects])

    # -------------------------------------------------------------------------

    def iter_observations(self):

        """
        Iterate over every observed curve.
        """

        for subject in self.subjects:
            yield from subject.observations()

    # -------------------------------------------------------------------------

    def observed_pairs(self, lag):
        """
        Iterate over all valid pairs separated by a circular lag.

        Parameters
        ----------
        lag : int

            Circular lag in hours.

        Yields
        ------
        subject_id
        hour
        curve(h)
        curve(h+lag)
        """

        lag %= 24

        for subject in self.subjects:
            hour_map = {h: c for h, c in zip(subject.hours, subject.curves)}

            for h in subject.hours:
                h2 = (h + lag) % 24

                if h2 in hour_map:

                    yield (subject.subject_id, h, hour_map[h], hour_map[h2])

    # -------------------------------------------------------------------------

    def subject(self, subject_id):
        for subject in self.subjects:
            if subject.subject_id == subject_id:
                return subject

        raise KeyError(subject_id)

    # -------------------------------------------------------------------------

    def subset(self, indices):
        return self.__class__([self.subjects[i] for i in indices])

    def summary(self):
        print("JSMFPCA Dataset")
        print("----------------")
        print(f"Subjects : {self.n_subjects}")
        print(f"Scales   : {self.n_scales}")

        observations = sum(s.n_hours for s in self.subjects)

        print(f"Observations : {observations}")
        print("Average observed hours :", observations / self.n_subjects)


@dataclass
class SubjectScores:
    subject_id: str
    hours: np.ndarray
    scores: np.ndarray        # ξ(h,k)
    offsets: Optional[np.ndarray] = None
    harmonic_coeffs: Optional[np.ndarray] = None


@dataclass
class ScoreDataset:
    """
    Collection of SubjectScores.

    Parameters
    ----------
    subjects : list[SubjectScores]
        FPCA scores for all subjects.
    """

    subjects: List[SubjectScores]

    def __post_init__(self):
        if len(self.subjects) == 0:
            raise ValueError("Dataset is empty.")

        n_components = self.subjects[0].n_components

        for s in self.subjects:
            if s.n_components != n_components:
                raise ValueError(
                    "All subjects must have the same number of components."
                )

        self._n_components = n_components

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_subjects(self):
        return len(self.subjects)

    @property
    def n_components(self):
        return self._n_components

    @property
    def subject_ids(self):
        return [s.subject_id for s in self.subjects]

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def subject(self, subject_id):
        for s in self.subjects:
            if s.subject_id == subject_id:
                return s
        raise KeyError(subject_id)

    def subset(self, indices):
        return self.__class__([self.subjects[i] for i in indices])

    # ------------------------------------------------------------------
    # Iterators
    # ------------------------------------------------------------------

    def iter_observations(self):
        """
        Yield (subject_id, hour, score_vector).
        """
        for subject in self.subjects:
            for h, score in zip(subject.hours, subject.scores):
                yield subject.subject_id, int(h), score

    def iter_component(self, k):
        """
        Yield (subject_id, hour, score) for one FPCA component.
        """
        for subject in self.subjects:
            for h, score in zip(subject.hours, subject.scores[:, k]):
                yield subject.subject_id, int(h), float(score)

    def iter_hour(self, hour):
        """
        Yield (subject_id, score_vector) for one hour.
        """
        for subject in self.subjects:
            idx = np.where(subject.hours == hour)[0]
            if len(idx):
                yield subject.subject_id, subject.scores[idx[0]]

    def observed_pairs(self, lag):
        """
        Yield score-vector pairs separated by a circular lag.

        Returns
        -------
        subject_id, hour, score(h), score(h+lag)
        """
        lag %= 24

        for subject in self.subjects:
            score_map = {
                int(h): s
                for h, s in zip(subject.hours, subject.scores)
            }

            for h in subject.hours:
                h2 = (int(h) + lag) % 24
                if h2 in score_map:
                    yield (
                        subject.subject_id,
                        int(h),
                        score_map[int(h)],
                        score_map[h2],
                    )

    # ------------------------------------------------------------------
    # Stacking
    # ------------------------------------------------------------------

    def stack_scores(self):
        """
        Stack all score vectors.

        Returns
        -------
        ndarray, shape (n_observations, n_components)
        """
        return np.vstack([s.scores for s in self.subjects])

    def stack_hours(self):
        """
        Stack all observed hours.
        """
        return np.concatenate([s.hours for s in self.subjects])

    def stack_subject_ids(self):
        """
        Subject ID for every observation.
        """
        ids = []
        for s in self.subjects:
            ids.extend([s.subject_id] * s.n_hours)
        return np.asarray(ids)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self):
        n_obs = sum(s.n_hours for s in self.subjects)

        print("ScoreDataset")
        print("------------")
        print(f"Subjects     : {self.n_subjects}")
        print(f"Components   : {self.n_components}")
        print(f"Observations : {n_obs}")
        print(f"Avg. hours   : {n_obs / self.n_subjects:.2f}")
