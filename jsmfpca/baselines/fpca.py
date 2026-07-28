from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..data import JSMFPCAData
from ..fingerprint import FingerprintBuilder
from ..fpca.model import ShapeFPCA
from .base import BaselineEstimator


@dataclass(slots=True)
class FPCAResult:
    fpca: ShapeFPCA
    fingerprints: list


class FPCA(BaselineEstimator):
    def __init__(
        self, n_components="cv", cv=None, fpca=None, fingerprint_builder=None
    ):
        self.fpca = fpca or ShapeFPCA(n_components=n_components, cv=cv)
        self.fingerprint_build = fingerprint_builder or FingerprintBuilder({})
        self.result_: FPCAResult | None = None

    def fit(self, dataset: JSMFPCAData):
        self.fpca.fit(dataset)
        scores = self.fpca.project_scores(dataset)
        subjects = self._subject_scores(scores)
        fingerprints = self.fingerprint_builder.transform_dataset(subjects)
        self.result_ = FPCAResult(fpca=self.fpca, fingerprints=fingerprints)

        return self

    def predict_scores(self, dataset: JSMFPCAData):
        scores = self.result_.fpca.project_scores(dataset)

        return self._subject_scores(scores)

    def transform(self, dataset: JSMFPCAData):
        subjects = self.predict_scores(dataset)

        return self.fingerprint_build.transform_dataset(subjects)

    def reconstruct(self, dataset: JSMFPCAData):
        scores = self.result_.fpca.project_scores(dataset)

        return self.result_.fpca.reconstruct_curves(scores)

    @staticmethod
    def _subject_scores(score_dataset):
        subjects = []

        for subject in score_dataset.subjects:
            mean_scores = np.mean(subject.scores, axis=0)

            subjects.append(subject.copy_with(
                scores=mean_scores[None], hours=np.array([0])
            ))

        return subjects
