from __future__ import annotations

import numpy as np
from .model import SpectralModel


class DiagonalSpectralModel(SpectralModel):
    """
    Independent-mode spectral model.

    Identical to SpectralModel except that each harmonic cross-spectrum is
    forced to be diagonal before shrinkage and eigendecomposition.
    """

    def _process_cross_spectra(
        self,
        spectra,
    ):

        out = spectra.copy()

        for r in range(out.shape[0]):
            out[r] = np.diag(np.diag(out[r]))

        return out
