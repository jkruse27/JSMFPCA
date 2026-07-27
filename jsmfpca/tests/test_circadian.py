"""
Tests for Stage 1 (Circadian model).
"""

from __future__ import annotations

import numpy as np

from jsmfpca.stage0.model import ShapeFPCA
from jsmfpca.circadian.model import CircadianModel


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _fit_stage1(dataset):

    fpca = ShapeFPCA()
    fpca.fit(dataset)

    scores = fpca.project_scores(dataset)

    model = CircadianModel()
    model.fit(scores)

    transformed = model.transform(scores)

    return fpca, model, scores, transformed


# ---------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------

def test_fit_runs(synthetic_dataset):

    _, model, _, _ = _fit_stage1(synthetic_dataset)

    assert model.fitted


# ---------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------

def test_transform_dimensions(synthetic_dataset):

    _, model, scores, transformed = _fit_stage1(
        synthetic_dataset
    )

    assert transformed.n_subjects == scores.n_subjects
    assert transformed.n_components == scores.n_components
    assert transformed.n_hours == scores.n_hours


# ---------------------------------------------------------------------
# Mean centering
# ---------------------------------------------------------------------

def test_subjects_are_centered(synthetic_dataset):

    _, _, _, transformed = _fit_stage1(
        synthetic_dataset
    )

    for subject in transformed.subjects:

        np.testing.assert_allclose(
            subject.centered.mean(axis=0),
            0.0,
            atol=1e-10,
        )


# ---------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------

def test_reconstruction_identity(synthetic_dataset):

    _, model, scores, transformed = _fit_stage1(
        synthetic_dataset
    )

    reconstructed = model.inverse_transform(
        transformed
    )

    np.testing.assert_allclose(
        reconstructed.as_array(),
        scores.as_array(),
        atol=1e-8,
    )


# ---------------------------------------------------------------------
# Mean preservation
# ---------------------------------------------------------------------

def test_mean_plus_centered_equals_scores(
    synthetic_dataset,
):

    _, _, scores, transformed = _fit_stage1(
        synthetic_dataset
    )

    for s0, s1 in zip(
        scores.subjects,
        transformed.subjects,
    ):

        reconstructed = (
            s1.mean
            + s1.centered
        )

        np.testing.assert_allclose(
            reconstructed,
            s0.scores,
            atol=1e-10,
        )


# ---------------------------------------------------------------------
# Hour preservation
# ---------------------------------------------------------------------

def test_hours_preserved(synthetic_dataset):

    _, _, scores, transformed = _fit_stage1(
        synthetic_dataset
    )

    for s0, s1 in zip(
        scores.subjects,
        transformed.subjects,
    ):

        np.testing.assert_array_equal(
            s0.hours,
            s1.hours,
        )


# ---------------------------------------------------------------------
# Subject ids
# ---------------------------------------------------------------------

def test_subject_ids_preserved(
    synthetic_dataset,
):

    _, _, scores, transformed = _fit_stage1(
        synthetic_dataset
    )

    ids0 = [
        s.subject_id
        for s in scores.subjects
    ]

    ids1 = [
        s.subject_id
        for s in transformed.subjects
    ]

    assert ids0 == ids1


# ---------------------------------------------------------------------
# Deterministic transform
# ---------------------------------------------------------------------

def test_transform_is_deterministic(
    synthetic_dataset,
):

    _, model, scores, _ = _fit_stage1(
        synthetic_dataset
    )

    t1 = model.transform(scores)

    t2 = model.transform(scores)

    np.testing.assert_allclose(
        t1.as_array(),
        t2.as_array(),
        atol=1e-12,
    )


def test_flat_subject_has_zero_centered():

    from jsmfpca.data import (
        ScoreDataset,
        ScoreSubject,
    )

    n_subjects = 5
    n_hours = 24
    n_components = 4

    subjects = []

    for subject_id in range(n_subjects):

        mean = np.random.randn(n_components)

        scores = np.tile(
            mean,
            (n_hours, 1),
        )

        subjects.append(
            ScoreSubject(
                subject_id=subject_id,
                hours=np.arange(n_hours),
                scores=scores,
            )
        )

    dataset = ScoreDataset(subjects)

    model = CircadianModel()

    model.fit(dataset)

    transformed = model.transform(dataset)

    for subject in transformed.subjects:

        np.testing.assert_allclose(
            subject.centered,
            0.0,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            subject.mean,
            subject.scores[0],
            atol=1e-12,
        )
