from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
import numpy as np


@dataclass(slots=True)
class BootstrapResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray | None = None

    @property
    def mean_eigenvalues(self):
        return self.eigenvalues.mean(axis=0)

    @property
    def std_eigenvalues(self):
        return self.eigenvalues.std(axis=0, ddof=1)

    @property
    def ci95(self):
        return np.percentile(self.eigenvalues, [2.5, 97.5], axis=0)


class Bootstrap:
    def __init__(self, pipeline, n_bootstrap=200, random_state=None):
        self.pipeline = pipeline
        self.n_bootstrap = int(n_bootstrap)
        self.rng = np.random.default_rng(random_state)

    def fit(self, dataset, store_eigenvectors=False):
        n = len(dataset.subjects)

        eigvals = []
        eigvecs = []

        for _ in range(self.n_bootstrap):
            indices = self.rng.integers(0, n, size=n)
            sampled = dataset.subset(indices)
            model = deepcopy(self.pipeline)
            model.fit(sampled)
            eigvals.append(model.spectral_model.eigenvalues_)

            if store_eigenvectors:
                eigvecs.append(model.spectral_model.eigenvectors_)

        return BootstrapResult(
            eigenvalues=np.asarray(eigvals),
            eigenvectors=(
                np.asarray(eigvecs) if store_eigenvectors else None
            ),
        )
