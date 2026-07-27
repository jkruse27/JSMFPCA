"""
Tests for lag covariance estimation.
"""

from __future__ import annotations

import numpy as np

from jsmfpca.stage0.model import ShapeFPCA
from jsmfpca.circadian.model import CircadianModel
from jsmfpca.spectral.covariance import estimate_lag_covariance


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _circadian_dataset(dataset):

    fpca = ShapeFPCA()
    fpca.fit(dataset)

    scores = fpca.project_scores(dataset)

    model = CircadianModel()
    model.fit(scores)

    return model.transform(scores)


# ---------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------

def test_covariance_shape(synthetic_dataset):

    dataset = _circadian_dataset(synthetic_dataset)

    Sigma = estimate_lag_covariance(dataset)

    K = dataset.n_components

    assert Sigma.shape == (24, K, K)


# ---------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------

def test_covariance_is_symmetric(synthetic_dataset):

    dataset = _circadian_dataset(synthetic_dataset)

    Sigma = estimate_lag_covariance(dataset)

    np.testing.assert_allclose(
        Sigma,
        Sigma.transpose(0, 2, 1),
        atol=1e-10,
    )


# ---------------------------------------------------------------------
# Zero-lag PSD
# ---------------------------------------------------------------------

def test_zero_lag_positive_semidefinite(synthetic_dataset):

    dataset = _circadian_dataset(synthetic_dataset)

    Sigma = estimate_lag_covariance(dataset)

    eig = np.linalg.eigvalsh(Sigma[0])

    assert np.all(eig >= -1e-10)


# ---------------------------------------------------------------------
# Finite values
# ---------------------------------------------------------------------

def test_covariance_is_finite(synthetic_dataset):

    dataset = _circadian_dataset(synthetic_dataset)

    Sigma = estimate_lag_covariance(dataset)

    assert np.isfinite(Sigma).all()


# ---------------------------------------------------------------------
# Subject ordering
# ---------------------------------------------------------------------

def test_subject_order_invariant(synthetic_dataset):

    dataset = _circadian_dataset(synthetic_dataset)

    Sigma1 = estimate_lag_covariance(dataset)

    rng = np.random.default_rng(123)

    shuffled = dataset.copy_with(
        subjects=[
            dataset.subjects[i]
            for i in rng.permutation(dataset.n_subjects)
        ]
    )

    Sigma2 = estimate_lag_covariance(shuffled)

    np.testing.assert_allclose(
        Sigma1,
        Sigma2,
        atol=1e-10,
    )


# ---------------------------------------------------------------------
# Weighting
# ---------------------------------------------------------------------

def test_subject_and_observation_weighting_agree_on_balanced_data(
    synthetic_dataset,
):

    dataset = _circadian_dataset(synthetic_dataset)

    Sigma_subject = estimate_lag_covariance(
        dataset,
        weighting="subject",
    )

    Sigma_observation = estimate_lag_covariance(
        dataset,
        weighting="observation",
    )

    np.testing.assert_allclose(
        Sigma_subject,
        Sigma_observation,
        atol=1e-10,
    )


# ---------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------

def test_missing_hours_runs(missing_dataset):

    dataset = _circadian_dataset(missing_dataset)

    Sigma = estimate_lag_covariance(dataset)

    assert np.isfinite(Sigma).all()


# ---------------------------------------------------------------------
# Sparse data
# ---------------------------------------------------------------------

def test_sparse_dataset_runs(sparse_dataset):

    dataset = _circadian_dataset(sparse_dataset)

    Sigma = estimate_lag_covariance(dataset)

    assert np.isfinite(Sigma).all()


# ---------------------------------------------------------------------
# Constant trajectories
# ---------------------------------------------------------------------

def test_constant_centered_scores_give_zero_covariance():

    from jsmfpca.circadian.data import (
        CircadianDataset,
        CircadianSubject,
    )

    subjects = []

    for subject_id in range(10):

        centered = np.zeros((24, 5))

        subjects.append(
            CircadianSubject(
                subject_id=subject_id,
                hours=np.arange(24),
                centered=centered,
                mean=np.zeros(5),
                offsets=np.zeros(5),
            )
        )

    dataset = CircadianDataset(subjects)

    Sigma = estimate_lag_covariance(dataset)

    np.testing.assert_allclose(
        Sigma,
        0.0,
        atol=1e-12,
    )
