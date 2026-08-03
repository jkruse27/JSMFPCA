from __future__ import annotations
import numpy as np
from scipy.linalg import eigh, block_diag, solve


class JSMFPCA:
    def __init__(
        self,
        n_modes: int | str = 3,
        n_harmonics: int = 2,
        shrinkage: float = 0.25,
        cv: int | None = 5,
        period: float = 24.0,
        ridge: float = 1e-6,
    ):
        self.n_modes = n_modes
        self.n_harmonics = n_harmonics
        self.shrinkage = shrinkage
        self.cv = cv
        self.period = period
        self.ridge = ridge

        # Fitted attributes
        self.mean_curve_ = None
        self.shape_basis_ = None
        self.shape_eigenvalues_ = None
        self.cross_spectra_ = None
        self.shrunk_spectra_ = None
        self.spectral_eigenvalues_ = None
        self.spectral_eigenvectors_ = None
        self.noise_variance_ = None
        self.prior_covariance_ = None
        self.is_fitted_ = False

    # =========================================================================
    # Public Scikit-Learn Estimator API
    # =========================================================================

    def fit(self, dataset):
        if self.n_modes == "cv" or self.shrinkage == "cv":
            self._fit_with_cv(dataset)
        else:
            self._fit_single(dataset)

        self.is_fitted_ = True
        return self

    def transform(self, dataset) -> np.ndarray:
        self._check_fitted()
        posteriors = self._predict_posteriors(dataset)
        return self._build_fingerprints(posteriors)

    def fit_transform(self, dataset) -> np.ndarray:
        self.fit(dataset)
        return self.transform(dataset)

    def reconstruct(self, dataset) -> list[np.ndarray]:
        self._check_fitted()
        posteriors = self._predict_posteriors(dataset)
        reconstructed_list = []

        for subj, post in zip(dataset.subjects, posteriors):
            X_design = self._fourier_design_matrix(
                subj.hours, self.n_harmonics
            )
            b_hat = post["mean"].reshape(
                self.n_harmonics * 2, self.n_modes_, order="F"
            )
            circadian_reconstructed = X_design @ b_hat
            total_scores = post["offsets"] + circadian_reconstructed
            curves = total_scores @ self.shape_basis_ + self.mean_curve_
            reconstructed_list.append(curves)

        return reconstructed_list

    # =========================================================================
    # Stage 0: Shape FPCA
    # =========================================================================

    def _fit_shape_fpca(self, curves_stacked: np.ndarray, weights: np.ndarray):
        self.mean_curve_ = np.average(curves_stacked, axis=0, weights=weights)
        centered = curves_stacked - self.mean_curve_

        w_sqrt = np.sqrt(weights)
        X_weighted = centered * w_sqrt[:, None]
        cov = X_weighted.T @ X_weighted / np.sum(weights)

        evals, evecs = eigh(cov)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx].T

        if isinstance(self.n_modes, int):
            self.n_modes_ = self.n_modes
        else:
            explained = np.cumsum(evals) / np.sum(evals)
            self.n_modes_ = np.searchsorted(explained, 0.95) + 1

        self.shape_basis_ = evecs[: self.n_modes_]
        self.shape_eigenvalues_ = evals[: self.n_modes_]

    def _project_shape_scores(self, curves: np.ndarray) -> np.ndarray:
        centered = curves - self.mean_curve_
        return centered @ self.shape_basis_.T

    # =========================================================================
    # Stage 1 & 2: Circadian Fourier Fitting & Cross-Spectral Decomposition
    # =========================================================================

    def _fourier_design_matrix(
        self, hours: np.ndarray, n_harmonics: int
    ) -> np.ndarray:
        hours = np.asarray(hours, dtype=float)
        omega = 2.0 * np.pi / self.period
        cols = []

        for r in range(1, n_harmonics + 1):
            cols.append(np.cos(r * omega * hours))
            cols.append(np.sin(r * omega * hours))

        return np.column_stack(cols)

    def _compute_lag_covariance(
        self, dataset_scores: list[dict]
    ) -> np.ndarray:
        M = self.n_modes_
        Sigma = np.zeros((24, M, M), dtype=float)
        counts = np.zeros(24, dtype=float)

        for subj in dataset_scores:
            hours = subj["hours"]
            centered = subj["centered"]  # (N_obs, M)
            n_obs = len(hours)

            for i in range(n_obs):
                for j in range(n_obs):
                    lag = int(round((hours[j] - hours[i]) % 24.0)) % 24
                    Sigma[lag] += np.outer(centered[i], centered[j])
                    counts[lag] += 1.0

        for lag in range(24):
            if counts[lag] > 0:
                Sigma[lag] /= counts[lag]

        # Symmetrize ONLY lag 0
        Sigma[0] = (Sigma[0] + Sigma[0].T) / 2.0
        return Sigma

    def _compute_cross_spectra(self, Sigma: np.ndarray) -> np.ndarray:
        R = self.n_harmonics
        M = self.n_modes_
        S_r = np.zeros((R, M, M), dtype=complex)

        for r in range(1, R + 1):
            for h in range(-11, 13):
                cov_h = Sigma[h % 24] if h >= 0 else Sigma[(-h) % 24].T
                weight = np.exp(-2j * np.pi * r * h / 24.0)
                S_r[r - 1] += cov_h * weight

        return S_r

    def _shrink_and_decompose_spectra(self, S_r: np.ndarray):
        R, M, _ = S_r.shape
        self.shrunk_spectra_ = np.zeros_like(S_r)
        self.spectral_eigenvalues_ = np.zeros((R, M), dtype=float)
        self.spectral_eigenvectors_ = np.zeros((R, M, M), dtype=complex)

        discarded_variances = []

        for r in range(R):
            S = S_r[r]
            target = np.diag(np.diag(S))
            S_shrunk = (1.0 - self.shrinkage) * S + self.shrinkage * target
            self.shrunk_spectra_[r] = S_shrunk

            evals, evecs = eigh(S_shrunk)
            idx = np.argsort(evals)[::-1]
            evals = np.maximum(evals[idx], 1e-10)
            evecs = evecs[:, idx]

            self.spectral_eigenvalues_[r] = evals
            self.spectral_eigenvectors_[r] = evecs

            if M > 1:
                discarded_variances.append(np.mean(evals[1:]))

        self.noise_variance_ = (
            np.mean(discarded_variances) if discarded_variances else 1e-4
        )

    # =========================================================================
    # Stage 3: Real Gaussian BLUP Prior & Posterior Inference
    # =========================================================================

    def _build_real_prior_covariance(self) -> np.ndarray:
        R = self.n_harmonics
        blocks = []

        for r in range(R):
            evals = self.spectral_eigenvalues_[r]
            evecs = self.spectral_eigenvectors_[r]

            S = evecs @ np.diag(evals) @ evecs.conj().T
            A = np.real(S)
            B = np.imag(S)

            cov_real_2M = np.block([[A, -B], [B, A]])
            blocks.append(cov_real_2M)

        prior_cov = block_diag(*blocks)
        return prior_cov

    def _predict_posteriors(self, dataset) -> list[dict]:
        prior_cov = self._build_real_prior_covariance()
        dim_prior = prior_cov.shape[0]
        inv_prior = solve(
            prior_cov + self.ridge * np.eye(dim_prior), np.eye(dim_prior)
        )

        posteriors = []

        for subj in dataset.subjects:
            hours = subj.hours
            scores = self._project_shape_scores(subj.curves)

            offsets = np.mean(scores, axis=0)
            centered = scores - offsets

            X_design = self._fourier_design_matrix(hours, self.n_harmonics)
            H = np.kron(np.eye(self.n_modes_), X_design)  # (N_obs*M, 2R*M)

            y = centered.reshape(-1, order="F")
            n_obs_total = len(y)

            inv_noise = (1.0 / self.noise_variance_) * np.eye(n_obs_total)

            post_precision = inv_prior + H.T @ inv_noise @ H
            post_cov = solve(
                post_precision + self.ridge * np.eye(dim_prior),
                np.eye(dim_prior)
            )
            post_mean = post_cov @ (H.T @ inv_noise @ y)

            posteriors.append({
                "subject_id": subj.subject_id,
                "offsets": offsets,
                "mean": post_mean,
                "covariance": post_cov,
            })

        return posteriors

    # =========================================================================
    # Stage 4: Physiological Fingerprints
    # =========================================================================

    def _build_fingerprints(self, posteriors: list[dict]) -> np.ndarray:
        fingerprints = []

        for post in posteriors:
            b_offset = post["offsets"]

            b_fourier = post["mean"].reshape(
                self.n_harmonics * 2, self.n_modes_, order="F"
            )
            harmonic_features = []

            for r in range(self.n_harmonics):
                a_r = b_fourier[2 * r]
                b_r = b_fourier[2 * r + 1]

                U_r = self.spectral_eigenvectors_[r]
                proj_a = np.real(U_r.conj().T @ a_r)
                proj_b = np.real(U_r.conj().T @ b_r)

                amp = np.sqrt(proj_a**2 + proj_b**2)
                phase = np.arctan2(-proj_b, proj_a)
                harmonic_features.extend([amp, phase])

            z_i = np.hstack([b_offset] + harmonic_features)
            fingerprints.append(z_i)

        return np.vstack(fingerprints)

    # =========================================================================
    # Helper & Cross-Validation Methods
    # =========================================================================

    def _fit_single(self, dataset):
        all_curves = np.vstack([subj.curves for subj in dataset.subjects])
        weights = np.ones(all_curves.shape[0], dtype=float)

        self._fit_shape_fpca(all_curves, weights)

        dataset_scores = []
        for subj in dataset.subjects:
            scores = self._project_shape_scores(subj.curves)
            offsets = np.mean(scores, axis=0)
            centered = scores - offsets
            dataset_scores.append({
                "hours": subj.hours,
                "centered": centered
            })

        Sigma = self._compute_lag_covariance(dataset_scores)
        S_r = self._compute_cross_spectra(Sigma)
        self.cross_spectra_ = S_r
        self._shrink_and_decompose_spectra(S_r)

    def _fit_with_cv(self, dataset):
        n_subjects = dataset.n_subjects
        cv_folds = self.cv or 5
        indices = np.arange(n_subjects)
        fold_size = max(1, n_subjects // cv_folds)

        shrinkage_grid = (
            [0.0, 0.1, 0.25, 0.5]
            if self.shrinkage == "cv"
            else [self.shrinkage]
        )
        modes_grid = [2, 3, 5] if self.n_modes == "cv" else [self.n_modes]

        best_err = float("inf")
        best_params = (modes_grid[0], shrinkage_grid[0])

        for m in modes_grid:
            for lam in shrinkage_grid:
                fold_errors = []

                for fold in range(cv_folds):
                    val_idx = indices[
                        fold * fold_size: (fold + 1) * fold_size
                    ]
                    train_idx = np.setdiff1d(indices, val_idx)

                    train_data = dataset.subset(train_idx)
                    val_data = dataset.subset(val_idx)

                    # Instantiate temporary candidate model
                    cand = JSMFPCA(
                        n_modes=m,
                        n_harmonics=self.n_harmonics,
                        shrinkage=lam,
                        cv=None,
                        period=self.period,
                    )
                    cand._fit_single(train_data)

                    reconstructed = cand.reconstruct(val_data)
                    err = 0.0
                    for val_subj, rec_curves in zip(
                        val_data.subjects, reconstructed
                    ):
                        err += np.mean((val_subj.curves - rec_curves) ** 2)

                    fold_errors.append(err)

                mean_err = np.mean(fold_errors)
                if mean_err < best_err:
                    best_err = mean_err
                    best_params = (m, lam)

        self.n_modes = best_params[0]
        self.shrinkage = best_params[1]
        self._fit_single(dataset)

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError(
                "JSMFPCA instance is not fitted yet. Call 'fit' first."
            )
