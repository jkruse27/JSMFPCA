from __future__ import annotations
import numpy as np
from jsmfpca.fpca.model import ShapeFPCA


# ---------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------

def test_fit_runs(synthetic_dataset):
    model = ShapeFPCA()
    model.fit(synthetic_dataset)

    assert model.fitted


# ---------------------------------------------------------------------
# Projection dimensions
# ---------------------------------------------------------------------

def test_projection_dimensions(synthetic_dataset):
    model = ShapeFPCA()
    model.fit(synthetic_dataset)
    scores = model.project_scores(synthetic_dataset)

    assert scores.n_subjects == synthetic_dataset.n_subjects
    assert scores.n_components == model.n_components
    assert scores.n_hours == synthetic_dataset.n_hours


# ---------------------------------------------------------------------
# Reconstruction dimensions
# ---------------------------------------------------------------------

def test_reconstruction_shape(synthetic_dataset):
    model = ShapeFPCA()
    model.fit(synthetic_dataset)
    scores = model.project_scores(synthetic_dataset)
    reconstructed = model.reconstruct_curves(scores)

    assert reconstructed.n_subjects == synthetic_dataset.n_subjects
    assert reconstructed.n_hours == synthetic_dataset.n_hours
    assert reconstructed.n_scales == synthetic_dataset.n_scales


# ---------------------------------------------------------------------
# Exact reconstruction
# ---------------------------------------------------------------------

def test_exact_reconstruction(noiseless_dataset):
    model = ShapeFPCA(n_components=noiseless_dataset.n_scales)
    model.fit(noiseless_dataset)
    scores = model.project_scores(noiseless_dataset)
    reconstructed = model.reconstruct_curves(scores)
    observed = noiseless_dataset.as_array()
    predicted = reconstructed.as_array()
    np.testing.assert_allclose(observed, predicted, atol=1e-10)


# ---------------------------------------------------------------------
# Mean preservation
# ---------------------------------------------------------------------

def test_mean_curve_preserved(synthetic_dataset):
    model = ShapeFPCA()
    model.fit(synthetic_dataset)
    observed = synthetic_dataset.mean_curve()
    np.testing.assert_allclose(model.mean_, observed, atol=1e-12)


# ---------------------------------------------------------------------
# Orthonormal basis
# ---------------------------------------------------------------------

def test_basis_is_orthonormal(synthetic_dataset):
    model = ShapeFPCA()
    model.fit(synthetic_dataset)
    Phi = model.components_
    identity = Phi.T @ Phi
    np.testing.assert_allclose(identity, np.eye(identity.shape[0]), atol=1e-10)


# ---------------------------------------------------------------------
# Eigenvalues
# ---------------------------------------------------------------------

def test_eigenvalues_are_nonnegative(synthetic_dataset):
    model = ShapeFPCA()
    model.fit(synthetic_dataset)

    assert np.all(model.eigenvalues_ >= 0)


# ---------------------------------------------------------------------
# Explained variance
# ---------------------------------------------------------------------

def test_explained_variance_monotone(synthetic_dataset):
    model = ShapeFPCA()
    model.fit(synthetic_dataset)
    explained = model.explained_variance_

    assert np.all(np.diff(explained) <= 0)


# ---------------------------------------------------------------------
# Projection consistency
# ---------------------------------------------------------------------

def test_projection_reconstruction_projection(synthetic_dataset):
    model = ShapeFPCA()
    model.fit(synthetic_dataset)
    scores1 = model.project_scores(synthetic_dataset)
    reconstructed = model.reconstruct_curves(scores1)
    scores2 = model.project_scores(reconstructed)

    np.testing.assert_allclose(
        scores1.as_array(), scores2.as_array(), atol=1e-8
    )


# ---------------------------------------------------------------------
# Subject-order invariance
# ---------------------------------------------------------------------

def test_subject_order_invariant(synthetic_dataset):
    model1 = ShapeFPCA()
    model1.fit(synthetic_dataset)
    rng = np.random.default_rng(123)
    order = rng.permutation(synthetic_dataset.n_subjects)
    shuffled = synthetic_dataset.copy_with(subjects=[
        synthetic_dataset.subjects[i] for i in order
    ])

    model2 = ShapeFPCA()
    model2.fit(shuffled)

    np.testing.assert_allclose(
        model1.eigenvalues_, model2.eigenvalues_, atol=1e-10
    )

    Phi1 = model1.components_
    Phi2 = model2.components_

    for k in range(Phi1.shape[1]):
        corr = np.abs(np.dot(Phi1[:, k], Phi2[:, k]))

        assert corr > 0.999999


# ---------------------------------------------------------------------
# Hour-order invariance
# ---------------------------------------------------------------------

def test_hour_order_invariant(synthetic_dataset):
    rng = np.random.default_rng(456)
    permutation = rng.permutation(synthetic_dataset.n_hours)
    shuffled_subjects = []

    for subject in synthetic_dataset.subjects:
        shuffled_subjects.append(
            subject.copy_with(
                hours=subject.hours[permutation],
                curves=subject.curves[permutation]
            )
        )

    shuffled = synthetic_dataset.copy_with(subjects=shuffled_subjects)

    model1 = ShapeFPCA()
    model1.fit(synthetic_dataset)
    model2 = ShapeFPCA()
    model2.fit(shuffled)

    np.testing.assert_allclose(
        model1.eigenvalues_, model2.eigenvalues_, atol=1e-10
    )
