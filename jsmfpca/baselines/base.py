from __future__ import annotations
from abc import ABC, abstractmethod
from ..base import FunctionalEstimator
from ..data import JSMFPCAData


class BaselineEstimator(FunctionalEstimator, ABC):
    @abstractmethod
    def fit(self, dataset: JSMFPCAData):
        ...

    @abstractmethod
    def predict_scores(self, dataset: JSMFPCAData):
        ...

    @abstractmethod
    def reconstruct(self, dataset: JSMFPCAData):
        ...

    def transform(self, dataset: JSMFPCAData):
        return self.predict_scores(dataset)

    def fit_transform(self, dataset: JSMFPCAData):
        self.fit(dataset)
        return self.transform(dataset)

    @staticmethod
    def reconstruction_error(observed, predicted):
        return predicted.reconstruction_error(observed)
