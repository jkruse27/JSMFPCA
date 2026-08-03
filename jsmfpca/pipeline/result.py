from __future__ import annotations
from dataclasses import dataclass
from .model import JSMFPCAModel
from ..circadian.model import CircadianModel
from ..fpca.model import ShapeFPCA
from ..spectral.model import SpectralModel


@dataclass(slots=True)
class JSMFPCAResult:
    model: JSMFPCAModel
    fingerprints: list

    stage0: ShapeFPCA
    stage1: CircadianModel
    stage2: SpectralModel

    @property
    def fpca(self):
        return self.stage0

    @property
    def circadian(self):
        return self.stage1

    @property
    def spectral(self):
        return self.stage2
