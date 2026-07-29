from __future__ import annotations
from dataclasses import dataclass
from ..pipeline import JSMFPCA, JSMFPCAResult
from ..spectral.model import SpectralModel
import numpy as np


# ---------------------------------------------------------------------
# Spectral model
# ---------------------------------------------------------------------

class DiagonalSpectralModel(SpectralModel):
    def _process_cross_spectra(self, spectra):
        out = spectra.copy()
        for r in range(out.shape[0]):
            out[r] = np.diag(np.real(np.diag(out[r])))
        return out

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
