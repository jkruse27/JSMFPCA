from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class FourierBasis:
    period: float = 24.0
    n_harmonics: int = 12

    @property
    def n_basis(self) -> int:
        return 2 * self.n_harmonics

    @property
    def frequencies(self):
        return np.arange(1, self.n_harmonics + 1, dtype=float)

    @property
    def omega(self):
        return 2 * np.pi / self.period

    def design_matrix(self, x):
        x = np.asarray(x, dtype=float)
        X = np.empty((len(x), self.n_basis), dtype=float)

        for r in range(self.n_harmonics):
            angle = (r + 1) * self.omega * x
            X[:, 2 * r] = np.cos(angle)
            X[:, 2 * r + 1] = np.sin(angle)

        return X

    def fit(self, x, y, weights=None):
        X = self.design_matrix(x)

        if weights is not None:
            w = np.sqrt(np.asarray(weights))
            X = X * w[:, None]
            y = y * w[:, None]

        coef, *_ = np.linalg.lstsq(X, y, rcond=None)

        return coef

    def predict(self, x, coefficients):
        X = self.design_matrix(x)

        return X @ coefficients

    def split_coefficients(self, coefficients):
        coef = np.asarray(coefficients)
        cosine = coef[0::2]
        sine = coef[1::2]

        return cosine, sine

    def stack_coefficients(self, cosine, sine):
        cosine = np.asarray(cosine)
        sine = np.asarray(sine)
        coef = np.empty((self.n_basis,) + cosine.shape[1:], dtype=float)
        coef[0::2] = cosine
        coef[1::2] = sine

        return coef
