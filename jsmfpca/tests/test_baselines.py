from __future__ import annotations
import numpy as np
import pytest
from jsmfpca.pipeline import JSMFPCA
from jsmfpca.baselines.mfpca.pipeline import TraditionalMFPCAPipeline
from jsmfpca.baselines.diagonal import DiagonalSpectralPipeline
from jsmfpca.baselines.cosinor import CosinorModel


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

BASELINES = [
    JSMFPCA,
    TraditionalMFPCAPipeline,
    DiagonalSpectralPipeline,
    CosinorModel,
]


# ---------------------------------------------------------------------
# API
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Estimator", BASELINES)
def test_fit_runs(Estimator, synthetic_dataset):
    model = Estimator()
    model.fit(synthetic_dataset)

    assert model.result_ is not None


@pytest.mark.parametrize("Estimator", BASELINES)
def test_fit_transform_runs(Estimator, synthetic_dataset):
    model = Estimator()
    fps = model.fit_transform(synthetic_dataset)

    assert len(fps) == synthetic_dataset.n_subjects


@pytest.mark.parametrize("Estimator", BASELINES)
def test_transform_runs(Estimator, synthetic_dataset):
    model = Estimator()
    model.fit(synthetic_dataset)
    fps = model.transform(synthetic_dataset)

    assert len(fps) == synthetic_dataset.n_subjects


# ---------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Estimator", BASELINES)
def test_unique_subject_ids(Estimator, synthetic_dataset):
    model = Estimator()
    fps = model.fit_transform(synthetic_dataset)
    ids = [fp.subject_id for fp in fps]

    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("Estimator", BASELINES)
def test_fingerprint_vectors_are_finite(Estimator, synthetic_dataset):
    model = Estimator()
    fps = model.fit_transform(synthetic_dataset)

    for fp in fps:
        assert np.isfinite(fp.vector).all()


# ---------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Estimator", BASELINES)
def test_missing_dataset(Estimator, missing_dataset):
    model = Estimator()
    model.fit(missing_dataset)


@pytest.mark.parametrize("Estimator", BASELINES)
def test_sparse_dataset(Estimator, sparse_dataset):
    model = Estimator()
    model.fit(sparse_dataset)


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Estimator", BASELINES)
def test_deterministic(Estimator, synthetic_dataset):
    model1 = Estimator()
    fp1 = model1.fit_transform(synthetic_dataset)
    model2 = Estimator()
    fp2 = model2.fit_transform(synthetic_dataset)

    fp1 = sorted(fp1, key=lambda x: x.subject_id)
    fp2 = sorted(fp2, key=lambda x: x.subject_id)

    for a, b in zip(fp1, fp2):
        np.testing.assert_allclose(a.vector, b.vector, atol=1e-12)


# ---------------------------------------------------------------------
# Subject ordering
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Estimator", BASELINES)
def test_subject_order_invariant(Estimator, synthetic_dataset):
    rng = np.random.default_rng(123)
    order = rng.permutation(synthetic_dataset.n_subjects)
    shuffled = synthetic_dataset.copy_with(
        subjects=[synthetic_dataset.subjects[i] for i in order]
    )
    model1 = Estimator()

    fp1 = model1.fit_transform(synthetic_dataset)
    model2 = Estimator()
    fp2 = model2.fit_transform(shuffled)

    fp1 = sorted(fp1, key=lambda x: x.subject_id)
    fp2 = sorted(fp2, key=lambda x: x.subject_id)

    for a, b in zip(fp1, fp2):
        np.testing.assert_allclose(a.vector, b.vector, atol=1e-10)


# ---------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Estimator", BASELINES)
def test_returns_same_fingerprint_type(Estimator, synthetic_dataset):
    model = Estimator()
    fps = model.fit_transform(synthetic_dataset)
    reference = type(fps[0])

    assert all(isinstance(fp, reference) for fp in fps)


# ---------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------

@pytest.mark.parametrize("Estimator", BASELINES)
def test_reconstruction_runs(Estimator, synthetic_dataset):
    model = Estimator()
    model.fit(synthetic_dataset)
    reconstruction = model.reconstruct(synthetic_dataset)

    assert reconstruction is not None
