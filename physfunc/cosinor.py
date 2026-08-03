# File: jsmfpca/baselines/cosinor.py

from __future__ import annotations
import numpy as np


class ClassicalCosinor:
    def __init__(self, n_harmonics: int = 2, period: float = 24.0):
        self.n_harmonics = n_harmonics
        self.period = period
        self.is_fitted_ = False

    def fit(self, dataset):
        self.is_fitted_ = True
        return self

    def _design_matrix(self, hours: np.ndarray) -> np.ndarray:
        """Construct design matrix [1, cos(w*t), sin(w*t), ...]."""
        hours = np.asarray(hours, dtype=float)
        omega = 2.0 * np.pi / self.period
        cols = [np.ones_like(hours)]

        for r in range(1, self.n_harmonics + 1):
            cols.append(np.cos(r * omega * hours))
            cols.append(np.sin(r * omega * hours))

        return np.column_stack(cols)  # (N_obs, 2R + 1)

    def transform(self, dataset) -> np.ndarray:
        self._check_fitted()
        fingerprints = []

        for subj in dataset.subjects:
            X_design = self._design_matrix(subj.hours)
            # OLS fit per subject curves
            coef, *_ = np.linalg.lstsq(X_design, subj.curves, rcond=None)

            mesor = coef[0]  # (K,)
            features = [mesor]

            for r in range(self.n_harmonics):
                a_r = coef[1 + 2 * r]  # (K,)
                b_r = coef[2 + 2 * r]  # (K,)

                amp = np.sqrt(a_r**2 + b_r**2)
                phase = np.arctan2(-b_r, a_r)
                features.extend([amp, phase])

            fingerprints.append(np.hstack(features))

        return np.vstack(fingerprints)

    def fit_transform(self, dataset) -> np.ndarray:
        return self.fit(dataset).transform(dataset)

    def reconstruct(self, dataset) -> list[np.ndarray]:
        """Reconstruct subject curves using independent subject Cosinor fit."""
        self._check_fitted()
        reconstructed = []

        for subj in dataset.subjects:
            X_design = self._design_matrix(subj.hours)
            coef, *_ = np.linalg.lstsq(X_design, subj.curves, rcond=None)
            rec_curves = X_design @ coef
            reconstructed.append(rec_curves)

        return reconstructed

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("ClassicalCosinor is not fitted yet.")
