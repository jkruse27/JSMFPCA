from __future__ import annotations
import numpy as np
from jsmfpca.spectral.blup import GaussianBLUP
from jsmfpca.spectral.prior import HarmonicComponent, SpectralPrior


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

def _prior():
    Q = np.eye(3)
    component = HarmonicComponent(
        harmonic=1, eigenvectors=Q, eigenvalues=np.array([4.0, 2.0, 1.0])
    )

    return SpectralPrior(components=[component])


def _problem():
    prior = _prior()
    H = np.eye(6)
    R = 0.1 * np.eye(6)
    y = np.array([1.2, -0.3, 0.5, 0.8, -1.0, 0.2])

    return prior, H, R, y


# ---------------------------------------------------------------------
# Posterior dimensions
# ---------------------------------------------------------------------

def test_posterior_shape():
    prior, H, R, y = _problem()
    blup = GaussianBLUP()
    posterior = blup.estimate(
        H=H, y=y, prior_covariance=prior.covariance(), noise_covariance=R
    )

    assert posterior.mean.shape == (6,)
    assert posterior.covariance.shape == (6, 6)


# ---------------------------------------------------------------------
# Posterior covariance
# ---------------------------------------------------------------------

def test_covariance_symmetric():
    prior, H, R, y = _problem()
    posterior = GaussianBLUP().estimate(H, y, prior.covariance(), R)

    np.testing.assert_allclose(
        posterior.covariance, posterior.covariance.T, atol=1e-10
    )


def test_covariance_positive_semidefinite():
    prior, H, R, y = _problem()
    posterior = GaussianBLUP().estimate(H, y, prior.covariance(), R)
    eig = np.linalg.eigvalsh(posterior.covariance)

    assert np.all(eig >= -1e-10)


# ---------------------------------------------------------------------
# Posterior variance
# ---------------------------------------------------------------------

def test_posterior_variance_smaller_than_prior():
    prior, H, R, y = _problem()
    posterior = GaussianBLUP().estimate(H, y, prior.covariance(), R)

    assert np.all(
        np.diag(posterior.covariance) <= np.diag(prior.covariance()) + 1e-10
    )


# ---------------------------------------------------------------------
# No-information limit
# ---------------------------------------------------------------------

def test_large_noise_returns_prior():
    prior, H, _, y = _problem()
    R = 1e12 * np.eye(6)
    posterior = GaussianBLUP().estimate(H, y, prior.covariance(), R)
    np.testing.assert_allclose(posterior.mean, 0, atol=1e-6)
    np.testing.assert_allclose(
        posterior.covariance, prior.covariance(), rtol=1e-6
    )


# ---------------------------------------------------------------------
# Perfect-information limit
# ---------------------------------------------------------------------

def test_small_noise_recovers_observations():
    prior, H, _, y = _problem()
    R = 1e-12 * np.eye(6)
    posterior = GaussianBLUP().estimate(H, y, prior.covariance(), R)
    np.testing.assert_allclose(posterior.mean, y, atol=1e-5)


# ---------------------------------------------------------------------
# Posterior prediction
# ---------------------------------------------------------------------

def test_prediction_shape():
    prior, H, R, y = _problem()
    blup = GaussianBLUP()
    posterior = blup.estimate(H, y, prior.covariance(), R)
    mean, covariance = blup.reconstruct(H, posterior)

    assert mean.shape == (6,)
    assert covariance.shape == (6, 6)


def test_prediction_covariance_symmetric():
    prior, H, R, y = _problem()
    blup = GaussianBLUP()
    posterior = blup.estimate(H, y, prior.covariance(), R)
    _, covariance = blup.reconstruct(H, posterior)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-10)


# ---------------------------------------------------------------------
# Linearity
# ---------------------------------------------------------------------

def test_linear_superposition():
    prior = _prior()
    H = np.eye(6)
    R = np.eye(6)
    y1 = np.ones(6)
    y2 = -np.ones(6)
    blup = GaussianBLUP()
    p1 = blup.estimate(H, y1, prior.covariance(), R)
    p2 = blup.estimate(H, y2, prior.covariance(), R)
    p12 = blup.estimate(H, y1 + y2, prior.covariance(), R)
    np.testing.assert_allclose(p12.mean, p1.mean + p2.mean, atol=1e-10)


# ---------------------------------------------------------------------
# Zero observations
# ---------------------------------------------------------------------

def test_zero_observation_returns_zero_mean():
    prior = _prior()
    H = np.eye(6)
    R = np.eye(6)
    y = np.zeros(6)
    posterior = GaussianBLUP().estimate(H, y, prior.covariance(), R)
    np.testing.assert_allclose(posterior.mean, 0, atol=1e-12)


# ---------------------------------------------------------------------
# Coefficient matrix
# ---------------------------------------------------------------------

def test_coefficient_matrix_shape():
    prior, H, R, y = _problem()
    posterior = GaussianBLUP().estimate(H, y, prior.covariance(), R)
    B = posterior.coefficient_matrix

    assert B.shape == (posterior.n_basis, posterior.n_modes)


# ---------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------

def test_finite_outputs():
    prior, H, R, y = _problem()
    posterior = GaussianBLUP().estimate(H, y, prior.covariance(), R)

    assert np.isfinite(posterior.mean).all()
    assert np.isfinite(posterior.covariance).all()
