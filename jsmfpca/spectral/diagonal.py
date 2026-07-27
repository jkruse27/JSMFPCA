from __future__ import annotations
import numpy as np
from .model import SpectralModel


class DiagonalSpectralModel(SpectralModel):
    def _process_cross_spectra(self, spectra):
        out = spectra.copy()

        for r in range(out.shape[0]):
            out[r] = np.diag(np.diag(out[r]))

        return out
