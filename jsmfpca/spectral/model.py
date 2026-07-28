from __future__ import annotations
import numpy as np
from ..circadian.data import CircadianDataset
from .covariance import estimate_lag_covariance
from .fourier import FourierBasis
from .shrinkage import shrink_all, decompose_all
from .coefficient_estimator import OLSCoefficientEstimator
from .data import SpectralDataset, SpectralSubject


class SpectralModel:
    def __init__(
        self, shrinkage=0.25, n_harmonics="cv", n_components=None,
        weighting="subject", estimator=None
    ):
        self.estimator = estimator or OLSCoefficientEstimator()
        self.shrinkage = shrinkage
        self.n_harmonics = n_harmonics
        self.n_components = n_components
        self.weight = weighting
        self._is_fitted = False
        self._fourier = self._build_fourier_matrix()
        self.fourier_basis = FourierBasis(period=24.0, n_harmonics=12)

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def fit(self, dataset: CircadianDataset):
        self.n_components_ = dataset.n_components
        self.Sigma_ = estimate_lag_covariance(dataset, weighting=self.weight)
        self.cross_spectra_ = self._compute_cross_spectra(self.Sigma_)
        self.shrunk_spectra_ = shrink_all(self.cross_spectra_, self.shrinkage)
        self.eigenvalues_, self.eigenvectors_ = decompose_all(
                                                        self.shrunk_spectra_
                                                    )
        self.basis_ = self.eigenvectors_
        noise_variances = []
        for i, evals in enumerate(self.eigenvalues_):
            if isinstance(self.n_components, int):
                keep = self.n_components
            else:
                idx = min(i, len(self.n_components) - 1)
                keep = self.n_components[idx]

            discarded = evals[keep:]
            if len(discarded) > 0 and np.sum(discarded) > 0:
                noise = np.mean(discarded)
            else:
                noise = 1e-6
            noise_variances.append(noise)

        self.noise_covariance_ = np.array(noise_variances)
        self._is_fitted = True
        return self

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("SpectralModel has not been fitted.")

    # ------------------------------------------------------------
    # Fourier matrix
    # ------------------------------------------------------------

    @staticmethod
    def _build_fourier_matrix():
        r = np.arange(13)[:, None]
        d = np.arange(24)[None, :]

        return np.exp(-2j * np.pi * r * d / 24)

    # ------------------------------------------------------------
    # Cross spectra
    # ------------------------------------------------------------

    def _compute_cross_spectra(self, Sigma):
        return np.einsum("rd,dij->rij", self._fourier, Sigma, optimize=True)

    def _process_cross_spectra(self, spectra):
        return spectra

    def bootstrap_statistics(self):
        return {
            "spectral_eigenvalues": self.eigenvalues_,
            "spectral_eigenvectors": self.eigenvectors_,
            "cross_spectra": self.shrunk_spectra_
        }

    @property
    def fitted(self):
        return self._is_fitted

    # ------------------------------------------------------------
    # Harmonic coefficient estimation
    # ------------------------------------------------------------

    def transform(self, dataset: CircadianDataset):
        self._check_fitted()

        subjects = [
            self.estimate_subject(subject) for subject in dataset.subjects
        ]

        return SpectralDataset(
            subjects=subjects,
            eigenvalues=self.eigenvalues_,
            eigenvectors=self.eigenvectors_
        )

    def estimate_subject(self, subject, observed_hours=None):
        self._check_fitted()

        if observed_hours is None:
            hours = subject.hours.astype(float)
            centered = subject.centered

        else:
            observed_hours = np.asarray(observed_hours)
            hours = subject.hours[observed_hours].astype(float)
            centered = subject.centered[observed_hours]

        coefficients = self._estimate_coefficients(hours, centered)
        rotated = self._rotate_coefficients(coefficients)

        return SpectralSubject(
            subject_id=subject.subject_id,
            hours=hours,
            offsets=subject.offsets.copy(),
            centered=centered.copy(),
            coefficients=coefficients,
            rotated_coefficients=rotated
        )

    def fit_transform(self, dataset):
        self.fit(dataset)

        return self.transform(dataset)

    # ------------------------------------------------------------
    # Subject harmonic regression
    # ------------------------------------------------------------

    def _estimate_coefficients(self, hours, centered):
        return self.estimator.fit(self.fourier_basis, hours, centered)

    def reconstruct_subject(
        self, subject, prediction_hours=None, rotated=False
    ):
        if prediction_hours is None:
            prediction_hours = np.arange(24)

        coef = (
            subject.rotated_coefficients if rotated else subject.coefficients
        )

        return self.fourier_basis.predict(prediction_hours, coef)

    # ------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------

    def _rotate_coefficients(self, coefficients):
        rotated = np.empty_like(coefficients)
        start_idx = 1 if len(coefficients) % 2 != 0 else 0

        if start_idx == 1:
            rotated[0] = coefficients[0]

        for r, U in enumerate(self.eigenvectors_):
            i = start_idx + 2 * r

            if i + 1 >= len(coefficients):
                break

            rotated[i] = (coefficients[i] @ U).real
            rotated[i + 1] = (coefficients[i + 1] @ U).real

        return rotated

    @property
    def coordinated_components(self):
        self._check_fitted()
        return self.eigenvectors_

    @property
    def spectral_variances(self):
        self._check_fitted()
        return self.eigenvalues_

    def get_params(self):
        return {
            "shrinkage": self.shrinkage,
            "weighting": self.weight,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
