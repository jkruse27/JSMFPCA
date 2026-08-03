# File: jsmfpca/baselines/diagonal_spectral.py

from __future__ import annotations
import numpy as np
from scipy.linalg import eigh


class DiagonalSpectralModel:
    def __init__(
        self, n_modes: int = 3, n_harmonics: int = 2,
        shrinkage: float = 0.25, period: float = 24.0
    ):
        self.n_modes = n_modes
        self.n_harmonics = n_harmonics
        self.shrinkage = shrinkage
        self.period = period

        self.mean_curve_ = None
        self.shape_basis_ = None
        self.diagonal_spectra_ = None
        self.noise_variance_ = 1e-4
        self.is_fitted_ = False

    def fit(self, dataset):
        # 1. Stage 0: Shape FPCA
        all_curves = np.vstack([subj.curves for subj in dataset.subjects])
        self.mean_curve_ = np.mean(all_curves, axis=0)
        centered = all_curves - self.mean_curve_

        cov = centered.T @ centered / len(all_curves)
        evals, evecs = eigh(cov)
        idx = np.argsort(evals)[::-1]
        self.shape_basis_ = evecs[:, idx[: self.n_modes]].T

        # 2. Extract shape scores X_i(d)
        dataset_scores = []
        for subj in dataset.subjects:
            scores = (subj.curves - self.mean_curve_) @ self.shape_basis_.T
            offsets = np.mean(scores, axis=0)
            dataset_scores.append(
                {"hours": subj.hours, "centered": scores - offsets}
            )

        # 3. Independent 1D lag covariance per shape component
        M = self.n_modes
        R = self.n_harmonics
        Sigma_diag = np.zeros((24, M))
        counts = np.zeros(24)

        for subj in dataset_scores:
            hours = subj["hours"]
            c = subj["centered"]
            for i in range(len(hours)):
                for j in range(len(hours)):
                    lag = int(round((hours[j] - hours[i]) % 24.0)) % 24
                    Sigma_diag[lag] += c[i] * c[j]
                    counts[lag] += 1.0

        for lag in range(24):
            if counts[lag] > 0:
                Sigma_diag[lag] /= counts[lag]

        # 4. Compute diagonal power spectral density S_{r, mm}
        S_diag = np.zeros((R, M))
        for r in range(1, R + 1):
            for h in range(-11, 13):
                weight = np.cos(2.0 * np.pi * r * h / 24.0)
                S_diag[r - 1] += Sigma_diag[h % 24] * weight

        # Regularize diagonal power spectrum
        self.diagonal_spectra_ = np.maximum(
            (1.0 - self.shrinkage) * S_diag + self.shrinkage *
            np.mean(S_diag, axis=1, keepdims=True), 1e-6
        )

        self.is_fitted_ = True
        return self

    def transform(self, dataset) -> np.ndarray:
        """Extract uncoupled circadian amplitude and phase fingerprints."""
        self._check_fitted()
        fingerprints = []

        for subj in dataset.subjects:
            scores = (subj.curves - self.mean_curve_) @ self.shape_basis_.T
            offsets = np.mean(scores, axis=0)
            centered = scores - offsets

            # 1D Fourier fit per mode
            omega = 2.0 * np.pi / self.period
            feature_vec = [offsets]

            for m in range(self.n_modes):
                y_m = centered[:, m]
                for r in range(1, self.n_harmonics + 1):
                    cos_t = np.cos(r * omega * subj.hours)
                    sin_t = np.sin(r * omega * subj.hours)
                    X_fourier = np.column_stack([cos_t, sin_t])

                    a, b = np.linalg.lstsq(X_fourier, y_m, rcond=None)[0]
                    amp = np.sqrt(a**2 + b**2)
                    phase = np.arctan2(-b, a)
                    feature_vec.extend([amp, phase])

            fingerprints.append(np.hstack(feature_vec))

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

            omega = 2.0 * np.pi / self.period
            rec_scores = np.zeros_like(scores)

            for m in range(self.n_modes):
                y_m = centered[:, m]
                cols = []
                for r in range(1, self.n_harmonics + 1):
                    cols.append(np.cos(r * omega * subj.hours))
                    cols.append(np.sin(r * omega * subj.hours))
                X_fourier = np.column_stack(cols)
                coef = np.linalg.lstsq(X_fourier, y_m, rcond=None)[0]
                rec_scores[:, m] = offsets[m] + X_fourier @ coef

            curves = rec_scores @ self.shape_basis_ + self.mean_curve_
            reconstructed.append(curves)

        return reconstructed

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("DiagonalSpectralModel is not fitted yet.")
