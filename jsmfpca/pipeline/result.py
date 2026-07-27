from __future__ import annotations
from dataclasses import dataclass
from .model import JSMFPCAModel


@dataclass(slots=True)
class JSMFPCAResult:
    model: JSMFPCAModel
    fingerprints: list
