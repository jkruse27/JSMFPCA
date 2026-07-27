from __future__ import annotations
from copy import deepcopy
import numpy as np


class BootstrapResampler:
    def __init__(self, estimator, n_bootstrap=200, random_state=None):
        self.estimator = estimator
        self.n_bootstrap = int(n_bootstrap)
        self.rng = np.random.default_rng(random_state)

    def fit(self, dataset):
        statistics = []
        n = len(dataset.subjects)

        for _ in range(self.n_bootstrap):
            indices = self.rng.integers(0, n, size=n)
            sample = dataset.subset(indices)
            estimator = deepcopy(self.estimator)
            estimator.fit(sample)
            statistics.append(estimator.bootstrap_statistics())

        return BootstrapResults(statistics)


class BootstrapResults:
    def __init__(self, statistics):
        self.statistics = {}
        keys = statistics[0].keys()

        for key in keys:
            self.statistics[key] = np.stack([s[key] for s in statistics])

    def __getitem__(self, key):
        return self.statistics[key]

    def mean(self, key):
        return self.statistics[key].mean(axis=0)

    def std(self, key):
        return self.statistics[key].std(axis=0, ddof=1)

    def percentile(self, key, q):
        return np.percentile(self.statistics[key], q, axis=0)

    def confidence_interval(self, key, alpha=0.05):
        return np.percentile(
            self.statistics[key],
            [100 * alpha / 2, 100 * (1 - alpha / 2)],
            axis=0
        )

    @property
    def keys(self):
        return tuple(self.statistics.keys())
