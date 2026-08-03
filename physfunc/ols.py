# File: jsmfpca/ols.py

from __future__ import annotations
import numpy as np
from scipy.linalg import eigh


class OLSHarmonicEstimator:
    def __init__(
        self,
        n_modes: int | str = 3,
        n_harmonics: int = 2,
        shrinkage: float = 0.25,
        rotate: bool = True,
        cv: int | None = 5,
        period: float = 24.0,
    ):
        self.n_modes = n_modes
        self.n_harmonics = n_harmonics
        self.shrinkage = shrinkage
        self.rotate = rotate
        self.cv = cv
        self.period = period

        # Fitted attributes
        self.mean_curve_ = None
        self.shape_basis_ = None
        self.spectral_eigenvectors_ = None
        self.is_fitted_ = False

    def fit(self, dataset):
        """Fit Shape FPCA and compute cross-spectral rotation eigenvectors."""
        # 1. Stack all curves for global Shape FPCA
        all_curves = np.vstack([subj.curves for subj in dataset.subjects])
        self.mean_curve_ = np.mean(all_curves, axis=0)
        centered = all_curves - self.mean_curve_

        cov = centered.T @ centered / len(all_curves)
        evals, evecs = eigh(cov)
        idx = np.argsort(evals)[::-1]

        if isinstance(self.n_modes, int):
            self.n_modes_ = self.n_modes
        else:
            explained = np.cumsum(evals[idx]) / np.sum(evals)
            self.n_modes_ = np.searchsorted(explained, 0.95) + 1

        self.shape_basis_ = evecs[:, idx[: self.n_modes_]].T

        # 2. Extract shape scores and fit subject OLS Fourier models
        dataset_scores = []
        for subj in dataset.subjects:
            scores = (subj.curves - self.mean_curve_) @ self.shape_basis_.T
            offsets = np.mean(scores, axis=0)
            dataset_scores.append(
                {"hours": subj.hours, "centered": scores - offsets}
            )

        # 3. Compute cross-spectral rotation eigenvectors U_r
        M = self.n_modes_
        R = self.n_harmonics
        Sigma = np.zeros((24, M, M), dtype=float)
        counts = np.zeros(24, dtype=float)

        for subj in dataset_scores:
            hours = subj["hours"]
            c = subj["centered"]
            for i in range(len(hours)):
                for j in range(len(hours)):
                    lag = int(round((hours[j] - hours[i]) % 24.0)) % 24
                    Sigma[lag] += np.outer(c[i], c[j])
                    counts[lag] += 1.0

        for lag in range(24):
            if counts[lag] > 0:
                Sigma[lag] /= counts[lag]

        # Two-sided Discrete Fourier Transform
        S_r = np.zeros((R, M, M), dtype=complex)
        for r in range(1, R + 1):
            for h in range(-11, 13):
                cov_h = Sigma[h % 24] if h >= 0 else Sigma[(-h) % 24].T
                weight = np.exp(-2j * np.pi * r * h / 24.0)
                S_r[r - 1] += cov_h * weight

        # Regularize and compute eigenvectors
        self.spectral_eigenvectors_ = np.zeros((R, M, M), dtype=complex)
        for r in range(R):
            S = (
                (1.0 - self.shrinkage) * S_r[r] +
                self.shrinkage * np.diag(np.diag(S_r[r]))
            )
            evals_s, evecs_s = eigh(S)
            idx_s = np.argsort(evals_s)[::-1]
            self.spectral_eigenvectors_[r] = evecs_s[:, idx_s]

        self.is_fitted_ = True
        return self

    def _fourier_design_matrix(self, hours: np.ndarray) -> np.ndarray:
        """Construct 24-hour Fourier design matrix."""
        omega = 2.0 * np.pi / self.period
        cols = []
        for r in range(1, self.n_harmonics + 1):
            cols.append(np.cos(r * omega * hours))
            cols.append(np.sin(r * omega * hours))
        return np.column_stack(cols)  # (N_obs, 2R)

    def transform(self, dataset) -> np.ndarray:
        self._check_fitted()
        fingerprints = []

        for subj in dataset.subjects:
            scores = (subj.curves - self.mean_curve_) @ self.shape_basis_.T
            offsets = np.mean(scores, axis=0)
            centered = scores - offsets

            # Direct OLS fit on shape scores
            X_design = self._fourier_design_matrix(subj.hours)
            coef, *_ = np.linalg.lstsq(X_design, centered, rcond=None)

            features = [offsets]
            for r in range(self.n_harmonics):
                a_r = coef[2 * r]
                b_r = coef[2 * r + 1]

                if self.rotate:
                    U_r = self.spectral_eigenvectors_[r]
                    proj_a = np.real(U_r.conj().T @ a_r)
                    proj_b = np.real(U_r.conj().T @ b_r)
                else:
                    proj_a, proj_b = a_r, b_r

                amp = np.sqrt(proj_a**2 + proj_b**2)
                phase = np.arctan2(-proj_b, proj_a)
                features.extend([amp, phase])

            fingerprints.append(np.hstack(features))

        return np.vstack(fingerprints)

    def fit_transform(self, dataset) -> np.ndarray:
        return self.fit(dataset).transform(dataset)

    def reconstruct(self, dataset) -> list[np.ndarray]:
        self._check_fitted()
        reconstructed = []

        for subj in dataset.subjects:
            scores = (subj.curves - self.mean_curve_) @ self.shape_basis_.T
            offsets = np.mean(scores, axis=0)
            centered = scores - offsets

            X_design = self._fourier_design_matrix(subj.hours)
            coef, *_ = np.linalg.lstsq(X_design, centered, rcond=None)

            rec_scores = offsets + X_design @ coef
            curves = rec_scores @ self.shape_basis_ + self.mean_curve_
            reconstructed.append(curves)

        return reconstructed

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("OLSHarmonicEstimator is not fitted yet.")
