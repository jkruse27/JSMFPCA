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
        self.dataset = dataset
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

    def _fit_shape_fpca(self, all_curves):
        self.mean_curve_ = np.mean(all_curves, axis=0)
        centered = all_curves - self.mean_curve_

        W = self.dataset.quadrature_weights.astype(float)
        sqrtW = np.sqrt(W)
        self.quadrature_weights_ = W

        Xw = centered * sqrtW
        U, S, Vt = np.linalg.svd(Xw, full_matrices=False)
        eigenvalues = (S ** 2) / (len(all_curves) - 1)

        phi = Vt / sqrtW
        norms = np.sqrt(np.sum(np.abs(phi)**2 * W, axis=1))
        phi /= norms[:, None]

        if isinstance(self.n_modes, int):
            self.n_modes_ = self.n_modes
        else:
            explained = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            self.n_modes_ = np.searchsorted(explained, 0.95) + 1

        self.shape_basis_ = phi[: self.n_modes_]
        self.shape_eigenvalues_ = eigenvalues[: self.n_modes_]

    def _project_shape_scores(self, curves: np.ndarray) -> np.ndarray:
        centered = curves - self.mean_curve_
        return (centered * self.quadrature_weights_) @ self.shape_basis_.T

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

    def _compute_lag_covariance(self, dataset_scores):
        M = self.n_modes_
        Sigma = np.zeros((24, M, M), dtype=float)
        lag_counts = np.zeros(24, dtype=float)

        for subj in dataset_scores:
            hours = np.asarray(subj["hours"], dtype=float)
            scores = subj["centered"]
            subject_cov = np.zeros((24, M, M), dtype=float)
            subject_counts = np.zeros(24, dtype=float)
            n = len(hours)

            for i in range(n):
                for j in range(n):
                    lag = int(round((hours[j] - hours[i]) % 24.0)) % 24
                    subject_cov[lag] += np.outer(scores[i], scores[j])
                    subject_counts[lag] += 1

            for lag in range(24):
                if subject_counts[lag] > 0:
                    subject_cov[lag] /= subject_counts[lag]
                    Sigma[lag] += subject_cov[lag]
                    lag_counts[lag] += 1

        for lag in range(24):
            if lag_counts[lag] > 0:
                Sigma[lag] /= lag_counts[lag]

        return Sigma

    def _compute_cross_spectra(self, lag_covariance):
        R = self.n_harmonics
        M = self.n_modes_
        spectra = np.zeros((R, M, M), dtype=complex)

        for r in range(1, R + 1):
            S = np.zeros((M, M), dtype=complex)
            for h in range(-11, 13):
                if h >= 0:
                    cov = lag_covariance[h]
                else:
                    cov = lag_covariance[-h].T

                weight = np.exp(-2j * np.pi * r * h / 24.0)
                S += cov * weight

            S /= 24.0
            S = 0.5 * (S + S.conj().T)
            spectra[r - 1] = S

        return spectra

    def _shrink_and_decompose_spectra(self, S_r: np.ndarray):
        R, M, _ = S_r.shape
        self.shrunk_spectra_ = np.zeros_like(S_r)
        self.spectral_eigenvalues_ = np.zeros((R, M), dtype=float)
        self.spectral_eigenvectors_ = np.zeros((R, M, M), dtype=complex)

        for r in range(R):
            S = S_r[r]
            target = np.diag(np.diag(S))
            S_shrunk = (1.0 - self.shrinkage) * S + self.shrinkage * target
            S_shrunk = 0.5 * (S_shrunk + S_shrunk.conj().T)
            self.shrunk_spectra_[r] = S_shrunk

            evals, evecs = eigh(S_shrunk)
            idx = np.argsort(evals)[::-1]
            evals = np.maximum(evals[idx], 1e-10)
            evecs = evecs[:, idx]

            self.spectral_eigenvalues_[r] = evals
            self.spectral_eigenvectors_[r] = evecs

    # =========================================================================
    # Stage 3: Real Gaussian BLUP Prior & Posterior Inference
    # =========================================================================

    def _build_real_prior_covariance(self) -> np.ndarray:
        R = self.n_harmonics
        blocks = []

        for r in range(R):
            S = self.shrunk_spectra_[r]
            A = S.real
            B = S.imag
            cov_real_2M = 0.5 * np.block([[A, -B], [B, A]])
            blocks.append(cov_real_2M)

        prior_cov = block_diag(*blocks)
        return prior_cov

    def _predict_posteriors(self, dataset) -> list[dict]:
        prior_cov = self.prior_covariance_
        dim_prior = prior_cov.shape[0]
        inv_prior = self.inv_prior_cov_

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
                z_r = a_r - 1j * b_r
                w_r = U_r.conj().T @ z_r

                proj_a = w_r.real
                proj_b = -w_r.imag

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
        self._fit_shape_fpca(all_curves)

        dataset_scores = []
        residual_variances = []

        for subj in dataset.subjects:
            scores = self._project_shape_scores(subj.curves)
            offsets = np.mean(scores, axis=0)
            centered = scores - offsets

            X_design = self._fourier_design_matrix(
                subj.hours, self.n_harmonics
            )

            coef, *_ = np.linalg.lstsq(
                X_design, centered, rcond=None
            )

            fitted = X_design @ coef
            residuals = centered - fitted

            residual_variances.append(
                np.var(residuals, axis=0, ddof=1)
            )

            dataset_scores.append(
                {"hours": subj.hours, "centered": centered}
            )

        residual_variances = np.vstack(residual_variances)
        self.mode_noise_variance_ = residual_variances.mean(axis=0)
        self.noise_variance_ = np.mean(self.mode_noise_variance_)
        Sigma = self._compute_lag_covariance(dataset_scores)
        self.lag_covariance_ = Sigma
        S_r = self._compute_cross_spectra(Sigma)
        self.cross_spectra_ = S_r
        self._shrink_and_decompose_spectra(S_r)
        self.prior_covariance_ = self._build_real_prior_covariance()
        dim_prior = self.prior_covariance_.shape[0]
        self.inv_prior_cov_ = solve(
            self.prior_covariance_ + self.ridge * np.eye(dim_prior),
            np.eye(dim_prior)
        )

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
