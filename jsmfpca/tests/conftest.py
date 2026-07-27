from __future__ import annotations
import numpy as np
import pytest
from jsmfpca.data import CurveDataset, CurveSubject


# ---------------------------------------------------------------------
# Random generator
# ---------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(12345)


# ---------------------------------------------------------------------
# Common grids
# ---------------------------------------------------------------------

@pytest.fixture(scope="session")
def hours():
    return np.arange(24)


@pytest.fixture(scope="session")
def scales():
    return np.linspace(np.log10(5), np.log10(1000), 128)


# ---------------------------------------------------------------------
# Internal generator
# ---------------------------------------------------------------------

def _generate_dataset(rng, hours, scales, n_subjects=20, noise=0.0):
    subjects = []

    for subject_id in range(n_subjects):
        amplitude = rng.normal(1.0, 0.08)
        phase = rng.normal(0.0, 0.5)
        width = rng.normal(0.35, 0.02)
        curves = []

        for hour in hours:
            center = 1.45 + 0.18 * np.sin(2 * np.pi * (hour - phase) / 24)
            curve = amplitude * np.exp(-0.5 * ((scales - center) / width) ** 2)

            if noise > 0:
                curve += noise * rng.standard_normal(scales.size)

            curves.append(curve)

        subjects.append(
            CurveSubject(
                subject_id=subject_id,
                hours=hours.copy(),
                curves=np.asarray(curves)
            )
        )

    return CurveDataset(subjects=subjects, scales=scales)


# ---------------------------------------------------------------------
# Canonical datasets
# ---------------------------------------------------------------------

@pytest.fixture
def noiseless_dataset(rng, hours, scales):
    return _generate_dataset(rng, hours, scales, noise=0.0)


@pytest.fixture
def synthetic_dataset(rng, hours, scales):
    return _generate_dataset(rng, hours, scales, noise=0.03)


@pytest.fixture
def noisy_dataset(rng, hours, scales):
    return _generate_dataset(rng, hours, scales, noise=0.10)


# ---------------------------------------------------------------------
# Missing observations
# ---------------------------------------------------------------------

@pytest.fixture
def missing_dataset(synthetic_dataset):
    rng = np.random.default_rng(9876)
    subjects = []

    for subject in synthetic_dataset.subjects:
        keep = rng.random(subject.hours.size) > 0.30
        subjects.append(subject.copy_with(
            hours=subject.hours[keep], curves=subject.curves[keep]
        ))

    return synthetic_dataset.copy_with(subjects=subjects)


# ---------------------------------------------------------------------
# Sparse observations
# ---------------------------------------------------------------------

@pytest.fixture
def sparse_dataset(synthetic_dataset):
    rng = np.random.default_rng(4321)
    subjects = []

    for subject in synthetic_dataset.subjects:
        keep = np.sort(rng.choice(subject.hours.size, size=8, replace=False))
        subjects.append(subject.copy_with(
            hours=subject.hours[keep], curves=subject.curves[keep]
        ))

    return synthetic_dataset.copy_with(subjects=subjects)
