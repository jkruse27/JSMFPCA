from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..base import FunctionalEstimator
from ...data import JSMFPCAData
from ...fingerprint import Fingerprint
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


class MFPCAFingerprintBuilder:
    def transform(self, subject):
        return Fingerprint(
            subject_id=subject.subject_id,
            vector=np.asarray(subject.xi),
            feature_names=[f"mfpca_mode_{k+1}" for k in range(len(subject.xi))]
        )

    def transform_dataset(self, subjects):
        return [self.transform(subject) for subject in subjects]


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

class TraditionalMFPCAPipeline(FunctionalEstimator):
    def __init__(self, explained_variance=0.99, fingerprint_builder=None):
        super().__init__()
        self.model = TraditionalMFPCA(explained_variance=explained_variance)
        self.fingerprint_build = (
            fingerprint_builder or MFPCAFingerprintBuilder()
        )
        self.result_: TraditionalMFPCAResult | None = None

    def fit(self, dataset: JSMFPCAData):
        tensor, subject_ids = self._dataset_to_tensor(dataset)
        self.model.fit(tensor)
        scores = estimate_scores(
            self.model.result, tensor, subject_ids=subject_ids
        )
        fingerprints = self.fingerprint_build.transform_dataset(
            scores.subjects
        )

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
        return self.fingerprint_build.transform_dataset(scores.subjects)

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
        for subject_data, scores_subj in zip(
            dataset.subjects, scores.subjects
        ):
            xi = scores_subj.xi
            subject_curves = []

            for h in subject_data.hours:
                zeta = scores_subj.zeta[int(h)]
                if np.isnan(zeta).all():
                    subject_curves.append(np.full(model.n_timepoints, np.nan))
                    continue

                curve = (
                    model.mean
                    + model.visit_mean[int(h)]
                    + model.phi @ xi
                    + model.psi @ zeta
                )
                subject_curves.append(curve)

            reconstructed.append(np.asarray(subject_curves))

        return reconstructed

    @staticmethod
    def _dataset_to_tensor(dataset: JSMFPCAData):
        n_subjects = dataset.n_subjects
        n_visits = 24
        n_time = dataset.n_scales

        tensor = np.full((n_subjects, n_visits, n_time), np.nan)
        subject_ids = []

        for i, subject in enumerate(dataset.subjects):
            subject_ids.append(subject.subject_id)
            for h, curve in zip(subject.hours, subject.curves):
                tensor[i, int(h)] = curve

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
