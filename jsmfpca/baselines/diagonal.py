from __future__ import annotations
from dataclasses import dataclass
from jsmfpca.pipeline import JSMFPCA, JSMFPCAResult
from jsmfpca.spectral.model import SpectralModel
from jsmfpca.spectral.shrinkage import shrink_all
from jsmfpca.spectral.decomposition import decompose_all
import numpy as np


# ---------------------------------------------------------------------
# Spectral model
# ---------------------------------------------------------------------

class DiagonalSpectralModel(SpectralModel):
    def fit(self, dataset):
        self.n_components_ = dataset.n_components
        self.Sigma_ = self._estimate_covariance(dataset)
        self.cross_spectra_ = (self._compute_cross_spectra(self.Sigma_))
        self.shrunk_spectra_ = shrink_all(self.cross_spectra_, self.shrinkage)
        self.shrunk_spectra_ = [
            self._diagonalize(S) for S in self.shrunk_spectra_
        ]
        self.eigenvalues_, self.eigenvectors_ = decompose_all(
                                                        self.shrunk_spectra_
                                                    )
        self._is_fitted = True

        return self

    @staticmethod
    def _diagonalize(S):
        return np.diag(np.real(np.diag(S)))


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

@dataclass(slots=True)
class DiagonalSpectralResult(JSMFPCAResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DiagonalSpectralPipeline(JSMFPCA):
    def __init__(self, *args, **kwargs):

        kwargs["spectral"] = (
            DiagonalSpectralModel(shrinkage=kwargs.get("shrinkage", 0.25))
        )

        super().__init__(*args, **kwargs)
