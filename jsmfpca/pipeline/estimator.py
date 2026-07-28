from __future__ import annotations
from ..base import FunctionalEstimator
from ..data import JSMFPCAData
from ..fingerprint import FingerprintBuilder
from ..fpca.model import ShapeFPCA
from ..circadian.model import CircadianModel
from ..spectral.model import SpectralModel
from ..spectral.selection import SpectralSelector
from ..spectral.prior import PriorBuilder
from ..spectral.operator import ObservationOperator
from ..spectral.inference import SpectralInference
from ..gaussian import GaussianBLUP
from .model import JSMFPCAModel
from .result import JSMFPCAResult


class JSMFPCA(FunctionalEstimator):
    def __init__(
        self, n_modes="cv", n_harmonics="cv", shrinkage="cv", cv=None,
        fpca=None, circadian=None, spectral=None, selector=None,
        fingerprint_builder=None, prior_builder=None, operator=None, blup=None
    ):
        super().__init__()

        self.fpca = fpca or ShapeFPCA(n_components=n_modes, cv=cv)
        self.circadian = circadian or CircadianModel()

        self.spectral = spectral or SpectralModel(
            n_harmonics=n_harmonics, shrinkage=shrinkage
        )

        self.selector = selector or SpectralSelector(
            shrinkage_grid=(0.0, 0.1, 0.25, 0.5),
            harmonic_grid=(1, 2, 3),
            component_grid=(1, 2, 3),
            scoring=self.reconstruction_error,
        )

        self.prior_builder = prior_builder or PriorBuilder()
        self.operator = operator or ObservationOperator()
        self.blup = blup or GaussianBLUP()
        self.fingerprint_build = fingerprint_builder or FingerprintBuilder({})
        self.result_: JSMFPCAResult | None = None

    def fit(self, dataset: JSMFPCAData):
        self.fpca.fit(dataset)
        scores = self.fpca.project_scores(dataset)

        self.circadian.fit(scores)
        circadian_scores = self.circadian.transform(scores)

        selection = self.selector.select(
            estimator=self.spectral, dataset=circadian_scores
        )

        self.spectral.set_params(
            shrinkage=selection.shrinkage,
            n_harmonics=selection.n_harmonics,
            n_components=selection.n_components
        )

        self.spectral.fit(circadian_scores)
        prior = self.prior_builder.build(self.spectral)
        inference = SpectralInference(
            basis=self.spectral.basis_, model=self.spectral,
            operator=self.operator, prior_builder=self.prior_builder,
            blup=self.blup
        )

        self.fingerprint_build.retained_components = {
            h + 1: k for h, k in enumerate(selection.n_components)
        }

        subjects = inference.estimate_dataset(
            circadian_scores, noise_covariance=self.spectral.noise_covariance_
        )

        fingerprints = self.fingerprint_build.transform_dataset(subjects)

        self.result_ = JSMFPCAResult(
            model=JSMFPCAModel(
                shape=self.fpca, circadian=self.circadian,
                spectral=self.spectral, prior=prior, inference=inference
            ), fingerprints=fingerprints,
        )

        return self

    def _check_fitted(self):
        if self.result_ is None:
            raise RuntimeError("Estimator has not been fitted.")

    def predict_scores(self, dataset):
        self._check_fitted()
        scores = self.result_.model.shape.project_scores(dataset)
        circadian_scores = self.result_.model.circadian.transform(scores)

        return self.result_.model.inference.estimate_dataset(
                circadian_scores,
                noise_covariance=self.result_.model.spectral.noise_covariance_
            )

    def transform(self, dataset):
        subjects = self.predict_scores(dataset)

        return self.fingerprint_build.transform_dataset(subjects)

    def fit_transform(self, dataset):
        self.fit(dataset)

        return self.result_.fingerprints

    def reconstruct(self, dataset):
        subjects = self.predict_scores(dataset)

        return [subject.reconstruct() for subject in subjects]

    @staticmethod
    def reconstruction_error(observed, predicted):
        return predicted.reconstruction_error(observed)
