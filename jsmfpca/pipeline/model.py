from __future__ import annotations
from dataclasses import dataclass
from ..fpca.model import ShapeFPCA
from ..circadian.model import CircadianModel
from ..spectral.model import SpectralModel
from ..spectral.prior import SpectralPrior
from ..spectral.inference import SpectralInference


@dataclass(slots=True)
class JSMFPCAModel:
    shape: ShapeFPCA
    circadian: CircadianModel
    spectral: SpectralModel
    prior: SpectralPrior
    inference: SpectralInference
