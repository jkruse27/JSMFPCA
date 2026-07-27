from __future__ import annotations
import numpy as np
from jsmfpca.pipeline import JSMFPCA


# ---------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------

def test_fit_runs(synthetic_dataset):
    model = JSMFPCA()
    model.fit(synthetic_dataset)

    assert model.result_ is not None


# ---------------------------------------------------------------------
# Fit-transform
# ---------------------------------------------------------------------

def test_fit_transform_runs(synthetic_dataset):
    model = JSMFPCA()
    fingerprints = model.fit_transform(synthetic_dataset)

    assert len(fingerprints) == synthetic_dataset.n_subjects


# ---------------------------------------------------------------------
# Transform after fit
# ---------------------------------------------------------------------

def test_transform_after_fit(synthetic_dataset):
    model = JSMFPCA()
    model.fit(synthetic_dataset)
    fingerprints = model.transform(synthetic_dataset)

    assert len(fingerprints) == synthetic_dataset.n_subjects


# ---------------------------------------------------------------------
# Stage consistency
# ---------------------------------------------------------------------

def test_all_stages_are_fitted(synthetic_dataset):
    model = JSMFPCA()
    model.fit(synthetic_dataset)

    assert model.result_.stage0.fitted
    assert model.result_.circadian.fitted
    assert model.result_.spectral.fitted


# ---------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------

def test_every_subject_has_fingerprint(synthetic_dataset):
    model = JSMFPCA()
    fingerprints = model.fit_transform(synthetic_dataset)

    assert len(fingerprints) == synthetic_dataset.n_subjects

    ids = {fp.subject_id for fp in fingerprints}

    assert len(ids) == synthetic_dataset.n_subjects


# ---------------------------------------------------------------------
# Deterministic
# ---------------------------------------------------------------------

def test_pipeline_is_deterministic(synthetic_dataset):
    model1 = JSMFPCA()
    fp1 = model1.fit_transform(synthetic_dataset)
    model2 = JSMFPCA()
    fp2 = model2.fit_transform(synthetic_dataset)

    for a, b in zip(fp1, fp2):
        np.testing.assert_allclose(a.vector, b.vector, atol=1e-12)


# ---------------------------------------------------------------------
# Missing observations
# ---------------------------------------------------------------------

def test_missing_dataset_runs(missing_dataset):
    model = JSMFPCA()
    model.fit(missing_dataset)


# ---------------------------------------------------------------------
# Sparse observations
# ---------------------------------------------------------------------

def test_sparse_dataset_runs(sparse_dataset):
    model = JSMFPCA()
    model.fit(sparse_dataset)


# ---------------------------------------------------------------------
# Subject ordering
# ---------------------------------------------------------------------

def test_subject_order_invariant(synthetic_dataset):
    rng = np.random.default_rng(42)
    model1 = JSMFPCA()
    fp1 = model1.fit_transform(synthetic_dataset)
    order = rng.permutation(synthetic_dataset.n_subjects)

    shuffled = synthetic_dataset.copy_with(subjects=[
        synthetic_dataset.subjects[i] for i in order
    ])

    model2 = JSMFPCA()
    fp2 = model2.fit_transform(shuffled)
    fp1 = sorted(fp1, key=lambda x: x.subject_id)
    fp2 = sorted(fp2, key=lambda x: x.subject_id)

    for a, b in zip(fp1, fp2):
        np.testing.assert_allclose(a.vector, b.vector, atol=1e-10)


# ---------------------------------------------------------------------
# Parameter API
# ---------------------------------------------------------------------

def test_get_set_params():
    model = JSMFPCA()
    params = model.get_params()
    model.set_params(**params)

    assert model.get_params() == params


# ---------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------

def test_result_contains_all_models(synthetic_dataset):
    model = JSMFPCA()
    model.fit(synthetic_dataset)
    result = model.result_

    assert result.stage0 is not None
    assert result.circadian is not None
    assert result.spectral is not None
    assert result.fingerprints is not None


# ---------------------------------------------------------------------
# Fingerprint uniqueness
# ---------------------------------------------------------------------

def test_fingerprints_not_identical(synthetic_dataset):
    model = JSMFPCA()
    fps = model.fit_transform(synthetic_dataset)
    vectors = np.vstack([fp.vector for fp in fps])

    assert np.var(vectors) > 0


# ---------------------------------------------------------------------
# Cross-validation execution
# ---------------------------------------------------------------------

def test_cv_pipeline_runs(synthetic_dataset):
    model = JSMFPCA(n_modes="cv", n_harmonics="cv", shrinkage="cv")
    model.fit(synthetic_dataset)

    assert model.result_ is not None
