from __future__ import annotations
from ..pipeline.estimator import JSMFPCA
from ..spectral.diagonal import DiagonalSpectralEstimator


class IndependentCircadian(JSMFPCA):
    def __init__(self, *args, **kwargs):
        kwargs["spectral"] = DiagonalSpectralEstimator()

        super().__init__(*args, **kwargs)
