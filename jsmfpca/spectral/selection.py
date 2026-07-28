from __future__ import annotations
import copy
from dataclasses import dataclass
from itertools import product


@dataclass(slots=True)
class SpectralSelectionResult:
    shrinkage: float
    n_harmonics: int
    n_components: tuple[int, ...]
    error: float


class SpectralSelector:
    def __init__(self, shrinkage_grid, harmonic_grid, component_grid, scoring):
        self.shrinkage_grid = tuple(shrinkage_grid)
        self.harmonic_grid = tuple(harmonic_grid)
        self.component_grid = tuple(component_grid)
        self.scoring = scoring

    def fit(self, estimator, dataset):
        best = None

        for alpha in self.shrinkage_grid:
            for n_harmonics in self.harmonic_grid:
                component_options = self._component_options(n_harmonics)

                for components in component_options:
                    model = copy.deepcopy(estimator)
                    model.set_params(
                        shrinkage=alpha,
                        n_harmonics=n_harmonics,
                        n_components=components
                    )

                    model.fit(dataset)
                    prediction = model.transform(dataset)
                    error = self.scoring(dataset, prediction)

                    if best is None or error < best.error:
                        best = SpectralSelectionResult(
                            shrinkage=alpha,
                            n_harmonics=n_harmonics,
                            n_components=components,
                            error=error
                        )

        return best

    def _component_options(self, n_harmonics):
        return product(self.component_grid, repeat=n_harmonics)
