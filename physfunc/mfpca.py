# File: jsmfpca/baselines/mfpca.py

from __future__ import annotations
import numpy as np
from scipy.linalg import eigh


class TraditionalMFPCA:
    def __init__(self, explained_variance: float = 0.95):
        self.explained_variance = explained_variance

        # Fitted attributes
        self.mean_ = None
        self.visit_mean_ = None
        self.phi_ = None  # Level 1 eigenfunctions (between-subject)
        self.psi_ = None  # Level 2 eigenfunctions (within-subject)
        self.lambda_phi_ = None
        self.lambda_psi_ = None
        self.is_fitted_ = False

    def fit(self, dataset):
        """Fit Level 1 and Level 2 functional principal components."""
        # 1. Stack observations (N_subjects, N_visits, N_time)
        data_array = self._dataset_to_array(dataset)
        n_subjects, n_visits, n_time = data_array.shape

        # 2. Overall population mean trajectory mu(t)
        self.mean_ = np.nanmean(data_array, axis=(0, 1))

        # 3. Visit-specific mean shifts eta_j(t)
        self.visit_mean_ = np.nanmean(data_array - self.mean_, axis=0)

        # 4. Centered data
        centered = data_array - self.mean_ - self.visit_mean_[None, :, :]

        # 5. Subject mean curves (Level 1) & within-subject residuals (Level 2)
        subject_means = np.nanmean(centered, axis=1)  # (N_subjects, N_time)
        valid_subjects = ~np.isnan(subject_means).any(axis=1)
        residuals = centered - subject_means[:, None, :]
        residuals_flat = residuals.reshape(n_subjects * n_visits, n_time)

        # Remove padded visits
        valid_rows = ~np.isnan(residuals_flat).any(axis=1)
        residuals_flat = residuals_flat[valid_rows]

        # 6. Level 2 (within-subject) covariance Kw
        Kw = np.cov(residuals_flat, rowvar=False)

        # 7. Level 1 (between-subject) covariance Kb = K_subj - (1/J) * Kw
        K_subj = np.cov(subject_means[valid_subjects], rowvar=False)
        J_harmonic = n_visits
        Kb_raw = K_subj - (1.0 / J_harmonic) * Kw

        # Project Kb to nearest Positive Semidefinite matrix
        evals_b, vecs_b = eigh(Kb_raw)
        evals_b = np.maximum(evals_b, 0.0)
        idx_b = np.argsort(evals_b)[::-1]
        evals_b, vecs_b = evals_b[idx_b], vecs_b[:, idx_b].T

        evals_w, vecs_w = eigh(Kw)
        evals_w = np.maximum(evals_w, 0.0)
        idx_w = np.argsort(evals_w)[::-1]
        evals_w, vecs_w = evals_w[idx_w], vecs_w[:, idx_w].T

        # 8. Select retained components based on explained variance
        nb = self._choose_components(evals_b, self.explained_variance)
        nw = self._choose_components(evals_w, self.explained_variance)

        self.phi_ = vecs_b[:nb]
        self.psi_ = vecs_w[:nw]
        self.lambda_phi_ = evals_b[:nb]
        self.lambda_psi_ = evals_w[:nw]

        self.is_fitted_ = True
        return self

    def transform(self, dataset) -> np.ndarray:
        """Extract Level 1 subject principal component scores xi_i."""
        self._check_fitted()
        data_array = self._dataset_to_array(dataset)
        centered = data_array - self.mean_ - self.visit_mean_[None, :, :]

        # Ignore padded visits
        subject_means = np.nanmean(centered, axis=1)
        subject_means = np.nan_to_num(subject_means)

        # Project subject mean curves onto Level 1 eigenfunctions phi_k
        xi = subject_means @ self.phi_.T  # Shape: (N_subjects, nb)
        return xi

    def fit_transform(self, dataset) -> np.ndarray:
        return self.fit(dataset).transform(dataset)

    def reconstruct(self, dataset) -> list[np.ndarray]:
        self._check_fitted()
        data_array = self._dataset_to_array(dataset)
        n_subjects, n_visits, n_time = data_array.shape
        centered = data_array - self.mean_ - self.visit_mean_[None, :, :]
        subject_means = np.nanmean(centered, axis=1)

        # Level 1 and Level 2 score projections
        xi = subject_means @ self.phi_.T

        reconstructed_list = []
        for i, subj in enumerate(dataset.subjects):
            subj_reconstructed = []
            for j in range(len(subj.hours)):
                visit_idx = min(j, n_visits - 1)
                if np.isnan(centered[i, visit_idx]).all():
                    continue

                res_ij = centered[i, visit_idx] - subject_means[i]
                zeta_ij = res_ij @ self.psi_.T  # Level 2 scores

                curve_ij = (
                    self.mean_
                    + self.visit_mean_[visit_idx]
                    + xi[i] @ self.phi_
                    + zeta_ij @ self.psi_
                )
                subj_reconstructed.append(curve_ij)

            reconstructed_list.append(np.array(subj_reconstructed))

        return reconstructed_list

    @staticmethod
    def _choose_components(evals: np.ndarray, threshold: float) -> int:
        total = np.sum(evals)
        if total <= 0:
            return 1
        cumsum = np.cumsum(evals) / total
        return int(np.searchsorted(cumsum, threshold) + 1)

    @staticmethod
    def _dataset_to_array(dataset) -> np.ndarray:
        # Form 3D array (N_subjects, N_visits, N_time)
        curves_list = [subj.curves for subj in dataset.subjects]
        max_visits = max(len(c) for c in curves_list)
        n_time = curves_list[0].shape[1]

        data = np.full((len(curves_list), max_visits, n_time), np.nan)
        for i, c in enumerate(curves_list):
            data[i, : len(c), :] = c
        return data

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("TraditionalMFPCA is not fitted yet.")
