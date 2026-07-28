from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------
# One retained harmonic
# ---------------------------------------------------------------------

@dataclass(slots=True)
class HarmonicComponent:
    harmonic: int

    eigenvectors: np.ndarray
    eigenvalues: np.ndarray

    @property
    def n_modes(self):
        return self.eigenvectors.shape[0]

    @property
    def n_components(self):
        return self.eigenvectors.shape[1]

    @property
    def covariance(self):
        diag = np.diag(self.eigenvalues)
        return self.eigenvectors @ diag @ self.eigenvectors.T

    @property
    def precision(self):
        diag = np.diag(1.0 / self.eigenvalues)
        return self.eigenvectors @ diag @ self.eigenvectors.T


# ---------------------------------------------------------------------
# Complete prior
# ---------------------------------------------------------------------

@dataclass(slots=True)
class SpectralPrior:
    cross_quadrature_covariance: np.ndarray | None = None
    components: list[HarmonicComponent] = field(default_factory=list)
    ridge: float = 1e-8

    @property
    def n_harmonics(self):
        return len(self.components)

    @property
    def n_modes(self):
        return self.components[0].n_modes if self.components else 0

    @property
    def n_basis(self):
        return 2 * self.n_harmonics

    def covariance(self):
        blocks = []

        for component in self.components:
            Sigma = component.covariance
            blocks.append(Sigma)
            blocks.append(Sigma)

        return self._block_diag(blocks)

    def precision(self):
        blocks = []

        for component in self.components:
            P = component.precision
            blocks.append(P)
            blocks.append(P)

        return self._block_diag(blocks)

    def variances(self):
        return {
            component.harmonic: component.eigenvalues.copy()
            for component in self.components
        }

    def sample(self, random_state=None):
        rng = np.random.default_rng(random_state)
        Sigma = self.covariance()
        z = rng.multivariate_normal(np.zeros(Sigma.shape[0]), Sigma)

        return z.reshape(self.n_basis, self.n_modes, order="F")

    @staticmethod
    def _block_diag(blocks):
        if len(blocks) == 0:
            return np.empty((0, 0))

        sizes = [b.shape[0] for b in blocks]
        n = sum(sizes)
        out = np.zeros((n, n))
        i = 0

        for block in blocks:
            m = block.shape[0]
            out[i:i + m, i:i + m] = block
            i += m

        return out


class PriorBuilder:
    def build(self, model):
        components = []

        for harmonic, (evals, evecs) in enumerate(
            zip(model.eigenvalues_, model.eigenvectors_), start=1
        ):
            if isinstance(model.n_components, int):
                keep = model.n_components
            else:
                idx = min(harmonic - 1, len(model.n_components) - 1)
                keep = model.n_components[idx]

            components.append(
                HarmonicComponent(
                    harmonic=harmonic,
                    eigenvectors=evecs[:, :keep],
                    eigenvalues=evals[:keep]
                )
            )

        return SpectralPrior(components=components)
