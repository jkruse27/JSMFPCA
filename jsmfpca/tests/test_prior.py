from __future__ import annotations
import numpy as np
from jsmfpca.spectral.prior import HarmonicComponent, SpectralPrior


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

def _prior():
    rng = np.random.default_rng(123)

    K = 5
    R = 3

    components = []

    for harmonic in range(1, R + 1):
        Q, _ = np.linalg.qr(rng.standard_normal((K, K)))
        eig = np.linspace(2.0, 0.5, K)

        components.append(HarmonicComponent(
            harmonic=harmonic, eigenvectors=Q, eigenvalues=eig
        ))

    return SpectralPrior(components=components)


# ---------------------------------------------------------------------
# Harmonic component
# ---------------------------------------------------------------------

def test_component_covariance_shape():
    component = _prior().components[0]
    Sigma = component.covariance

    assert Sigma.shape == (component.n_modes, component.n_modes)


def test_component_precision_shape():
    component = _prior().components[0]
    P = component.precision

    assert P.shape == (component.n_modes, component.n_modes)


def test_component_precision_inverse():
    component = _prior().components[0]
    Sigma = component.covariance
    P = component.precision

    np.testing.assert_allclose(
        Sigma @ P, np.eye(component.n_modes), atol=1e-10
    )


# ---------------------------------------------------------------------
# Prior matrices
# ---------------------------------------------------------------------

def test_covariance_shape():
    prior = _prior()
    Sigma = prior.covariance()
    n = 2 * prior.n_harmonics * prior.n_modes

    assert Sigma.shape == (n, n)


def test_precision_shape():
    prior = _prior()
    P = prior.precision()
    n = 2 * prior.n_harmonics * prior.n_modes

    assert P.shape == (n, n)


def test_covariance_symmetric():
    prior = _prior()
    Sigma = prior.covariance()
    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-10)


def test_precision_symmetric():
    prior = _prior()
    P = prior.precision()
    np.testing.assert_allclose(P, P.T, atol=1e-10)


def test_precision_inverse_covariance():
    prior = _prior()
    Sigma = prior.covariance()
    P = prior.precision()
    np.testing.assert_allclose(Sigma @ P, np.eye(Sigma.shape[0]), atol=1e-10)


def test_covariance_positive_semidefinite():
    prior = _prior()
    eig = np.linalg.eigvalsh(prior.covariance())

    assert np.all(eig >= -1e-10)


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------

def test_sample_shape():
    prior = _prior()
    sample = prior.sample()

    assert sample.shape == (prior.n_basis, prior.n_modes)


def test_sample_mean():
    prior = _prior()
    samples = np.stack([prior.sample().reshape(-1) for _ in range(4000)])
    mean = samples.mean(axis=0)
    np.testing.assert_allclose(mean, 0.0, atol=0.08)


def test_sample_covariance():
    prior = _prior()
    samples = np.stack([prior.sample().reshape(-1) for _ in range(4000)])
    empirical = np.cov(samples, rowvar=False)
    theoretical = prior.covariance()
    np.testing.assert_allclose(empirical, theoretical, atol=0.15)


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

def test_variances():
    prior = _prior()
    variances = prior.variances()

    for component in prior.components:
        np.testing.assert_array_equal(
            variances[component.harmonic], component.eigenvalues
        )


def test_n_harmonics():
    prior = _prior()

    assert prior.n_harmonics == 3


def test_n_modes():
    prior = _prior()

    assert prior.n_modes == 5


def test_n_basis():
    prior = _prior()

    assert prior.n_basis == 6


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------

def test_empty_prior():
    prior = SpectralPrior()

    assert prior.n_harmonics == 0
    assert prior.n_modes == 0
    assert prior.n_basis == 0

    assert prior.covariance().shape == (0, 0)
    assert prior.precision().shape == (0, 0)


def test_single_harmonic():
    component = HarmonicComponent(
        harmonic=1, eigenvectors=np.eye(3), eigenvalues=np.ones(3)
    )
    prior = SpectralPrior(components=[component])

    assert prior.n_harmonics == 1
    assert prior.n_basis == 2

    Sigma = prior.covariance()

    assert Sigma.shape == (6, 6)


def test_single_mode():
    component = HarmonicComponent(
        harmonic=1, eigenvectors=np.ones((1, 1)), eigenvalues=np.ones(1)
    )

    prior = SpectralPrior(components=[component])
    Sigma = prior.covariance()
    np.testing.assert_allclose(Sigma, np.eye(2))
