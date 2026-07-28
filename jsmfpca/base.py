from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy


class FunctionalEstimator(ABC):
    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def transform(self, dataset):
        pass

    @abstractmethod
    def reconstruct(self, dataset):
        pass

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

    @property
    def fitted(self) -> bool:
        return getattr(
            self, "result_", None
        ) is not None or getattr(self, "_is_fitted", False)

    def clone(self):
        return deepcopy(self)

    def get_params(self):
        return {
            key: value for key, value in self.__dict__.items()
            if not key.endswith("_") and not key.startswith("_")
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
