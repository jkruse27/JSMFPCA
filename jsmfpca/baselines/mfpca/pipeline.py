from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..base import FunctionalEstimator
from ...data import JSMFPCAData
from ...fingerprint import FingerprintBuilder
from .model import TraditionalMFPCA
from .scores import estimate_scores
from .data import MFPCAResult, ScoreDataset


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------

@dataclass(slots=True)
class TraditionalMFPCAResult:
    model: MFPCAResult
    scores: ScoreDataset
    fingerprints: list


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

class TraditionalMFPCAPipeline(FunctionalEstimator):
    def __init__(self, explained_variance=0.99, fingerprint_builder=None):
        super().__init__()

        self.model = TraditionalMFPCA(explained_variance=explained_variance)
        self.fingerprint_build = fingerprint_builder or FingerprintBuilder({})
        self.result_: TraditionalMFPCAResult | None = None

    def fit(self, dataset: JSMFPCAData):
        tensor, subject_ids = self._dataset_to_tensor(dataset)

        self.model.fit(tensor)
        scores = estimate_scores(
            self.model.result, tensor, subject_ids=subject_ids
        )
        fingerprints = self.fingerprint_build.transform_dataset(scores)

        self.result_ = TraditionalMFPCAResult(
            model=self.model.result, scores=scores, fingerprints=fingerprints
        )

        return self

    def transform(self, dataset: JSMFPCAData):
        if self.result_ is None:
            raise RuntimeError("Estimator has not been fitted.")

        tensor, subject_ids = self._dataset_to_tensor(dataset)
        scores = estimate_scores(
            self.model.result, tensor, subject_ids=subject_ids
        )

        return self.fingerprint_builder.transform_dataset(scores)

    def fit_transform(self, dataset):
        self.fit(dataset)

        return self.result_.fingerprints

    def reconstruct(self, dataset: JSMFPCAData):
        if self.result_ is None:
            raise RuntimeError("Estimator has not been fitted.")

        tensor, _ = self._dataset_to_tensor(dataset)
        scores = estimate_scores(self.model.result, tensor)
        model = self.model.result
        reconstructed = []

        for subject in scores.subjects:
            xi = subject.xi
            subject_curves = []

            for zeta in subject.zeta:
                if np.isnan(zeta).all():
                    subject_curves.append(np.full(model.n_timepoints, np.nan))
                    continue

                curve = (
                    model.mean
                    + model.visit_mean[len(subject_curves)]
                    + model.phi @ xi
                    + model.psi @ zeta
                )

                subject_curves.append(curve)
            reconstructed.append(np.asarray(subject_curves))

        return reconstructed

    @staticmethod
    def _dataset_to_tensor(dataset: JSMFPCAData):
        n_subjects = dataset.n_subjects
        n_visits = dataset.n_visits
        n_time = dataset.n_timepoints

        tensor = np.full((n_subjects, n_visits, n_time), np.nan)
        subject_ids = []

        for i, subject in enumerate(dataset.subjects):
            subject_ids.append(subject.subject_id)
            tensor[i] = subject.curves

        return tensor, subject_ids

    @property
    def fitted(self):
        return self.result_ is not None

    def get_params(self):
        return {"explained_variance": self.model.explained_variance}

    def set_params(self, **params):
        for key, value in params.items():
            if key == "explained_variance":
                self.model.explained_variance = value

        return self
