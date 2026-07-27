from __future__ import annotations
from abc import ABC, abstractmethod
from ..base import FunctionalEstimator
from ..data import CurveDataset


class BaselineEstimator(FunctionalEstimator, ABC):
    @abstractmethod
    def fit(self, dataset: CurveDataset):
        ...

    @abstractmethod
    def predict_scores(self, dataset: CurveDataset):
        ...

    @abstractmethod
    def reconstruct(self, dataset: CurveDataset):
        ...

    def transform(self, dataset: CurveDataset):
        return self.predict_scores(dataset)

    def fit_transform(self, dataset: CurveDataset):
        self.fit(dataset)
        return self.transform(dataset)

    @staticmethod
    def reconstruction_error(observed, predicted):
        return predicted.reconstruction_error(observed)
