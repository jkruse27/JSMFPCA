from __future__ import annotations
import numpy as np
from jsmfpca.fpca.model import ShapeFPCA
from jsmfpca.circadian.model import CircadianModel
from jsmfpca.spectral.model import SpectralModel
from jsmfpca.spectral.diagonal import DiagonalSpectralModel


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _fit_spectral(dataset):
    fpca = ShapeFPCA()
    fpca.fit(dataset)
    scores = fpca.project_scores(dataset)

    circadian = CircadianModel()
    circadian.fit(scores)
    circadian_scores = circadian.transform(scores)

    spectral = SpectralModel()
    spectral.fit(circadian_scores)
    transformed = spectral.transform(circadian_scores)

    return fpca, circadian, spectral, circadian_scores, transformed


# ---------------------------------------------------------------------
# API
# ---------------------------------------------------------------------

def test_fit_runs(synthetic_dataset):
    _, _, model, _, _ = _fit_spectral(synthetic_dataset)

    assert model.fitted


def test_transform_runs(synthetic_dataset):
    _, _, model, dataset, transformed = _fit_spectral(synthetic_dataset)

    assert transformed.n_subjects == dataset.n_subjects


def test_fit_transform(synthetic_dataset):
    fpca = ShapeFPCA()
    fpca.fit(synthetic_dataset)
    scores = fpca.project_scores(synthetic_dataset)

    circadian = CircadianModel()
    circadian.fit(scores)
    circadian_scores = circadian.transform(scores)

    model = SpectralModel()
    t1 = model.fit_transform(circadian_scores)

    model.fit(circadian_scores)
    t2 = model.transform(circadian_scores)

    assert t1.n_subjects == t2.n_subjects


# ---------------------------------------------------------------------
# Cross spectra
# ---------------------------------------------------------------------

def test_cross_spectrum_shape(synthetic_dataset):
    _, _, model, dataset, _ = _fit_spectral(synthetic_dataset)

    assert model.cross_spectra_.shape == (
        model.n_harmonics, dataset.n_components, dataset.n_components
    )


def test_cross_spectra_are_hermitian(synthetic_dataset):
    _, _, model, _, _ = _fit_spectral(synthetic_dataset)

    np.testing.assert_allclose(
        model.cross_spectra_,
        model.cross_spectra_.conj().transpose(0, 2, 1),
        atol=1e-10
    )


def test_shrunk_spectra_are_hermitian(synthetic_dataset):
    _, _, model, _, _ = _fit_spectral(synthetic_dataset)

    np.testing.assert_allclose(
        model.shrunk_spectra_,
        model.shrunk_spectra_.conj().transpose(0, 2, 1),
        atol=1e-10
    )


# ---------------------------------------------------------------------
# Eigendecomposition
# ---------------------------------------------------------------------

def test_eigenvalues_nonnegative(synthetic_dataset):
    _, _, model, _, _ = _fit_spectral(synthetic_dataset)

    for eig in model.eigenvalues_:
        assert np.all(eig >= -1e-10)


def test_eigenvectors_orthonormal(synthetic_dataset):
    _, _, model, _, _ = _fit_spectral(synthetic_dataset)

    for U in model.eigenvectors_:
        np.testing.assert_allclose(
            U.conj().T @ U, np.eye(U.shape[1]), atol=1e-10
        )


# ---------------------------------------------------------------------
# Subject coefficients
# ---------------------------------------------------------------------

def test_subject_coefficients_shape(synthetic_dataset):
    _, _, model, dataset, transformed = _fit_spectral(synthetic_dataset)
    subject = transformed.subjects[0]

    assert subject.coefficients.shape == (
        2 * model.n_harmonics_, dataset.n_components
    )


def test_rotated_coefficients_shape(synthetic_dataset):
    _, _, model, dataset, transformed = _fit_spectral(synthetic_dataset)
    subject = transformed.subjects[0]

    assert subject.rotated_coefficients.shape == (
        2 * model.n_harmonics, dataset.n_components
    )


# ---------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------

def test_rotation_preserves_energy(synthetic_dataset):
    _, _, model, _, transformed = _fit_spectral(synthetic_dataset)

    for subject in transformed.subjects:
        original = np.linalg.norm(subject.coefficients)
        rotated = np.linalg.norm(subject.rotated_coefficients)
        np.testing.assert_allclose(original, rotated, atol=1e-10)


def test_inverse_rotation(synthetic_dataset):
    _, _, model, _, transformed = _fit_spectral(synthetic_dataset)
    subject = transformed.subjects[0]
    reconstructed = np.empty_like(subject.coefficients)

    for r, U in enumerate(model.eigenvectors_):
        i = 2 * r
        reconstructed[i] = subject.rotated_coefficients[i] @ U.conj().T
        reconstructed[i + 1] = subject.rotated_coefficients[i + 1] @ U.conj().T

    np.testing.assert_allclose(reconstructed, subject.coefficients, atol=1e-10)


# ---------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------

def test_reconstruction_shape(synthetic_dataset):
    _, _, model, _, transformed = _fit_spectral(synthetic_dataset)
    subject = transformed.subjects[0]
    reconstructed = model.reconstruct_subject(subject)

    assert reconstructed.shape == (24, model.n_components)


# ---------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------

def test_missing_hours(missing_dataset):
    _fit_spectral(missing_dataset)


def test_sparse_dataset(sparse_dataset):
    _fit_spectral(sparse_dataset)


def test_subject_order_invariant(synthetic_dataset):
    _, _, model1, dataset, _ = _fit_spectral(synthetic_dataset)
    rng = np.random.default_rng(123)

    shuffled = dataset.copy_with(subjects=[
        dataset.subjects[i] for i in rng.permutation(dataset.n_subjects)
    ])

    model2 = SpectralModel()
    model2.fit(shuffled)

    for e1, e2 in zip(model1.eigenvalues_, model2.eigenvalues_):
        np.testing.assert_allclose(e1, e2, atol=1e-10)


# ---------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------

def test_bootstrap_statistics(synthetic_dataset):
    _, _, model, _, _ = _fit_spectral(synthetic_dataset)
    stats = model.bootstrap_statistics()

    assert "spectral_eigenvalues" in stats
    assert "spectral_eigenvectors" in stats
    assert "cross_spectra" in stats


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

def test_get_set_params(synthetic_dataset):
    _, _, model, _, _ = _fit_spectral(synthetic_dataset)
    params = model.get_params()
    model.set_params(**params)

    assert model.get_params() == params


# ---------------------------------------------------------------------
# Independent baseline
# ---------------------------------------------------------------------

def test_diagonal_spectral_model(synthetic_dataset):
    fpca = ShapeFPCA()
    fpca.fit(synthetic_dataset)
    scores = fpca.project_scores(synthetic_dataset)

    circadian = CircadianModel()
    circadian.fit(scores)
    circadian_scores = circadian.transform(scores)

    model = DiagonalSpectralModel()
    model.fit(circadian_scores)

    for S in model.cross_spectra_:
        offdiag = S - np.diag(np.diag(S))
        np.testing.assert_allclose(offdiag, 0.0, atol=1e-12)
