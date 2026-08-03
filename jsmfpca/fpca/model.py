from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold
from data import JSMFPCAData, SubjectScores, ScoreDataset
from ..utils import trapezoidal_weights
from .base import (
    weighted_svd, project_scores, reconstruct_curves, explained_variance_ratio
)


class ShapeFPCA:
    def __init__(
            self, n_components=None, selection="fixed", max_components=20,
            n_splits=5, randomized=False, random_state=None, cv=None
    ):
        if n_components == "cv":
            self.n_components = None
            self.selection = "cv"
        else:
            self.n_components = n_components
            self.selection = selection

        self.max_components = max_components
        self.n_splits = cv if cv is not None else n_splits
        self.randomized = randomized
        self.random_state = random_state
        self._is_fitted = False

    @property
    def n_components(self):
        if hasattr(self, 'components_') and self.components_ is not None:
            return self.components_.shape[0]
        return getattr(self, '_n_components', None)

    @n_components.setter
    def n_components(self, value):
        self._n_components = value

    def _select_n_components(self, data):
        X = data.stack_curves()
        groups = data.stack_subject_ids()
        weights = trapezoidal_weights(data.scales)
        splitter = GroupKFold(self.n_splits)
        errors = np.zeros(self.max_components)

        for train_idx, test_idx in splitter.split(X, groups=groups):
            train = X[train_idx]
            test = X[test_idx]

            result = weighted_svd(
                train, weights, n_components=self.max_components,
                randomized=self.randomized, random_state=self.random_state
            )

            for k in range(1, self.max_components + 1):
                basis = result.basis[:k]
                scores = project_scores(test, result.mean, basis, weights)
                reconstructed = reconstruct_curves(scores, result.mean, basis)
                mse = np.mean((test - reconstructed) ** 2 * weights)
                errors[k - 1] += mse

        errors /= self.n_splits
        self.cv_errors_ = errors

        return np.argmin(errors) + 1

    def fit(self, data: JSMFPCAData):
        weights = trapezoidal_weights(data.scales)

        if self.selection == "cv":
            self.n_components = self._select_n_components(data)
        elif self.selection != "fixed":
            raise ValueError("selection must be 'fixed' or 'cv'.")

        X = data.stack_curves()

        result = weighted_svd(
            X, weights, n_components=self.n_components,
            randomized=self.randomized, random_state=self.random_state
        )

        self.mean_ = result.mean
        self.basis_ = result.basis
        self.weights_ = result.weights
        self.eigenvalues_ = result.eigenvalues
        self.singular_values_ = result.singular_values
        self.explained_variance_ratio_ = explained_variance_ratio(
                                            self.eigenvalues_
                                        )
        self.scales_ = data.scales.copy()
        self.selected_n_components_ = self.n_components
        norms = np.linalg.norm(self.basis_, axis=1, keepdims=True)
        self.components_ = self.basis_ / norms
        self.n_features_ = self.basis_.shape[1]
        self.explained_variance_ = self.eigenvalues_

        self._is_fitted = True

        return self

    def transform(self, data: JSMFPCAData):
        self._check_fitted()
        subjects = []

        for subject in data.subjects:
            scores = project_scores(
                subject.curves, self.mean_, self.basis_, self.weights_
            )

            subjects.append(
                SubjectScores(
                    subject_id=subject.subject_id,
                    hours=subject.hours.copy(),
                    scores=scores
                )
            )

        return ScoreDataset(subjects)

    def project_scores(self, data: JSMFPCAData):
        self._check_fitted()
        subjects = []

        for subject in data.subjects:
            scores = project_scores(
                subject.curves, self.mean_, self.basis_, self.weights_
            )

            subjects.append(
                SubjectScores(
                    subject_id=subject.subject_id,
                    hours=subject.hours.copy(),
                    scores=scores
                )
            )

        return ScoreDataset(subjects)

    def fit_transform(self, data):
        self.fit(data)

        return self.transform(data)

    def inverse_transform(self, scores):
        self._check_fitted()

        return reconstruct_curves(scores, self.mean_, self.basis_)

    def reconstruct_subject(self, subject_scores):
        curves = reconstruct_curves(
            subject_scores.scores, self.mean_, self.basis_
        )

        return curves

    def reconstruct_curves(self, scores_dataset):
        from jsmfpca.data import JSMFPCAData, SubjectCurves

        reconstructed_subjects = []
        for subject_scores in scores_dataset.subjects:
            reconstructed_curve_array = self.reconstruct_subject(
                subject_scores
            )

            reconstructed_subjects.append(
                SubjectCurves(
                    subject_id=subject_scores.subject_id,
                    hours=subject_scores.hours,
                    curves=reconstructed_curve_array,
                    metadata=getattr(subject_scores, 'metadata', {})
                )
            )

        return JSMFPCAData(
            subjects=reconstructed_subjects,
            scales=self.scales_
        )

    def reconstruction_error(self, data):
        transformed = self.transform(data)
        errors = []

        for subject, scores in zip(data.subjects, transformed):
            reconstructed = self.reconstruct_subject(scores)
            mse = np.mean((subject.curves - reconstructed) ** 2)
            errors.append(mse)

        return np.mean(errors)

    def bootstrap_statistics(self):
        return {
            "eigenvalues": self.eigenvalues_,
            "eigenfunctions": self.components_,
        }

    @property
    def fitted(self):
        return self._is_fitted

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("ShapeFPCA has not been fitted.")
