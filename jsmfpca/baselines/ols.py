from __future__ import annotations
from dataclasses import dataclass
from ..base import FunctionalEstimator
from ..data import JSMFPCAData
from ..fpca.model import ShapeFPCA
from ..circadian.model import CircadianModel
from ..spectral.model import SpectralModel
from ..fingerprint import FingerprintBuilder


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------

@dataclass(slots=True)
class OLSHarmonicResult:
    stage0: ShapeFPCA
    circadian: CircadianModel
    spectral: SpectralModel
    fingerprints: list


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

class OLSHarmonicPipeline(FunctionalEstimator):
    def __init__(
        self,
        n_modes="cv",
        n_harmonics="cv",
        shrinkage="cv",
        rotate=True,
        cv=None,
        fpca=None,
        circadian=None,
        spectral=None,
        selector=None,
        fingerprint_builder=None,
    ):
        super().__init__()

        self.rotate = rotate
        self.n_modes = n_modes
        self.n_harmonics = n_harmonics
        self.shrinkage = shrinkage
        self.cv = cv

        self.fpca = fpca or ShapeFPCA(n_components=n_modes, cv=cv)
        self.circadian = circadian or CircadianModel()
        self.spectral = spectral or SpectralModel(
            n_harmonics=n_harmonics, shrinkage=shrinkage
        )

        self.selector = selector
        self.fingerprint_build = fingerprint_builder or FingerprintBuilder({})
        self.result_: OLSHarmonicResult | None = None

    def fit(self, dataset: JSMFPCAData):
        self.fpca.fit(dataset)
        scores = self.fpca.project_scores(dataset)

        self.circadian.fit(scores)
        circadian_scores = self.circadian.transform(scores)

        self.spectral.fit(circadian_scores)
        subjects = self.spectral.transform(circadian_scores)

        if not self.rotate:
            for subject in subjects.subjects:
                subject.rotated_coefficients = (subject.coefficients.copy())

        fingerprints = self.fingerprint_build.transform_dataset(subjects)

        self.result_ = OLSHarmonicResult(
            stage0=self.fpca, circadian=self.circadian,
            spectral=self.spectral, fingerprints=fingerprints
        )

        return self

    def transform(self, dataset: JSMFPCAData):
        scores = self.fpca.project_scores(dataset)
        circadian_scores = self.circadian.transform(scores)
        subjects = self.spectral.transform(circadian_scores)

        if not self.rotate:
            for subject in subjects.subjects:
                subject.rotated_coefficients = subject.coefficients.copy()

        return self.fingerprint_build.transform_dataset(subjects)

    def fit_transform(self, dataset):
        self.fit(dataset)

        return self.result_.fingerprints

    def reconstruct(self, dataset):
        scores = self.fpca.project_scores(dataset)
        circadian_scores = self.circadian.transform(scores)
        subjects = self.spectral.transform(circadian_scores)

        return [
            self.spectral.reconstruct_subject(subject, rotated=self.rotate)
            for subject in subjects.subjects
        ]

    @property
    def fitted(self):
        return self.result_ is not None
