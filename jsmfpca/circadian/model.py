from __future__ import annotations
import numpy as np
from .harmonic import fit_harmonic, predict_harmonic
from .data import CircadianSubject, CircadianDataset
from ..data import ScoreDataset


class CircadianModel:
    def __init__(self, harmonic_order=2):
        self.harmonic_order = harmonic_order
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, scores: ScoreDataset):
        self.n_components_ = scores.n_components

        (
            self.population_fits_,
            self.population_coefficients_,
            self.population_curves_,
        ) = self._fit_population(scores)

        (
            self.between_variance_,
            self.within_variance_,
        ) = self._estimate_variances(scores)

        self._is_fitted = True

        return self

    def fit_transform(self, scores: ScoreDataset):
        self.fit(scores)

        return self.transform(scores)

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("CircadianModel has not been fitted.")

    # ------------------------------------------------------------------
    # Population harmonic models
    # ------------------------------------------------------------------

    def _fit_population(self, scores):
        fits = []
        coeffs = []
        curves = np.zeros((24, scores.n_components))
        hours_all = scores.stack_hours()

        for k in range(scores.n_components):
            values = scores.stack_scores()[:, k]
            fit = fit_harmonic(hours_all, values, order=self.harmonic_order)
            fits.append(fit)
            coeffs.append(fit.coefficients)
            curves[:, k] = predict_harmonic(fit, np.arange(24))

        coeffs = np.vstack(coeffs)

        return fits, coeffs, curves

    def inverse_transform(self, dataset):
        from jsmfpca.data import SubjectScores
        reconstructed_subjects = []

        for subject in dataset.subjects:
            reconstructed_scores = (
                subject.centered + subject.offsets + subject.fitted
            )

            reconstructed_subjects.append(
                SubjectScores(
                    subject_id=subject.subject_id,
                    hours=subject.hours,
                    scores=reconstructed_scores
                )
            )

        return ScoreDataset(subjects=reconstructed_subjects)

    # ------------------------------------------------------------------
    # Variance component estimation
    # ------------------------------------------------------------------

    def _estimate_variances(self, scores):
        between = np.zeros(scores.n_components)
        within = np.zeros(scores.n_components)

        for k in range(scores.n_components):
            subject_means = []
            pooled_within = []
            fit = self.population_fits_[k]

            for subject in scores.subjects:
                prediction = predict_harmonic(fit, subject.hours)
                residual = subject.scores[:, k] - prediction
                subject_means.append(residual.mean())
                pooled_within.extend(residual - residual.mean())

            between[k] = np.var(subject_means, ddof=1)
            within[k] = np.var(pooled_within, ddof=1)

        return between, within

    # ------------------------------------------------------------------
    # BLUP shrinkage
    # ------------------------------------------------------------------

    def _shrink_offset(self, raw_offset, sigma_between, sigma_within, n_obs):
        den = sigma_between + sigma_within / max(n_obs, 1)

        return raw_offset if den <= 0 else (sigma_between / den) * raw_offset

    @property
    def fitted(self):
        return self._is_fitted

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, scores: ScoreDataset):
        self._check_fitted()
        subjects = []

        for subject in scores.subjects:
            n_hours = subject.n_hours
            K = scores.n_components

            fitted = np.zeros((n_hours, K))
            residuals = np.zeros((n_hours, K))
            centered = np.zeros((n_hours, K))
            offsets = np.zeros(K)

            for k in range(K):
                fit = self.population_fits_[k]
                mu = predict_harmonic(fit, subject.hours)
                eta = subject.scores[:, k] - mu
                raw_offset = eta.mean()

                offset = self._shrink_offset(
                    raw_offset,
                    self.between_variance_[k],
                    self.within_variance_[k],
                    n_hours
                )

                fitted[:, k] = mu
                residuals[:, k] = eta
                centered[:, k] = subject.scores[:, k] - offset
                offsets[k] = offset

            subjects.append(
                CircadianSubject(
                    subject_id=subject.subject_id,
                    hours=subject.hours.copy(),
                    scores=subject.scores.copy(),
                    fitted=fitted,
                    offsets=offsets,
                    residuals=residuals,
                    centered=centered,
                )
            )

        return CircadianDataset(
            subjects=subjects,
            harmonic_order=self.harmonic_order,
            population_coefficients=self.population_coefficients_,
            population_curves=self.population_curves_,
            residual_variances=self.within_variance_
        )

    def reconstruct_population(self, hours=None):
        self._check_fitted()

        if hours is None:
            return self.population_curves_.copy()

        hours = np.asarray(hours)
        curves = np.zeros((len(hours), self.n_components_))

        for k, fit in enumerate(self.population_fits_):
            curves[:, k] = predict_harmonic(fit, hours)

        return curves

    def bootstrap_statistics(self):
        return {
            "harmonic_coefficients": self.coefficients_,
            "offset_variance": self.offset_variance_,
        }

    def predict_subject(self, subject):
        prediction = subject.fitted.copy()
        prediction += subject.offsets

        return prediction

    def residual_variance(self):
        self._check_fitted()

        return self.within_variance_.copy()

    def between_subject_variance(self):
        self._check_fitted()

        return self.between_variance_.copy()

    @property
    def n_components(self):
        self._check_fitted()
        return self.n_components_
