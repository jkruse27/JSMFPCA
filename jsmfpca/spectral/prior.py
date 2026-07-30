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
        S = self.eigenvectors @ diag @ self.eigenvectors.conj().T
        A = np.real(S)
        B = np.imag(S)
        return np.block([[A, -B], [B, A]])

    @property
    def precision(self):
        diag = np.diag(1.0 / self.eigenvalues)
        S_inv = self.eigenvectors @ diag @ self.eigenvectors.conj().T
        A = np.real(S_inv)
        B = np.imag(S_inv)
        return np.block([[A, -B], [B, A]])


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

    def _align_prior(self, raw_matrix):
        H_harm = self.n_harmonics
        K_modes = self.n_modes

        if H_harm == 0 or K_modes == 0:
            return raw_matrix

        dim = 2 * H_harm * K_modes
        P = np.zeros((dim, dim))

        for r in range(H_harm):
            for part in (0, 1):
                for k in range(K_modes):
                    prior_idx = r * (2 * K_modes) + part * K_modes + k
                    h_idx = k * (2 * H_harm) + 2 * r + part
                    P[h_idx, prior_idx] = 1.0

        return P @ raw_matrix @ P.T

    def covariance(self):
        blocks = []
        for component in self.components:
            blocks.append(component.covariance)
        raw_cov = self._block_diag(blocks)
        return self._align_prior(raw_cov)

    def precision(self):
        blocks = []
        for component in self.components:
            blocks.append(component.precision)
        raw_prec = self._block_diag(blocks)
        return self._align_prior(raw_prec)

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
