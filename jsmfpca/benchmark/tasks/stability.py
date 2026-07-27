"""
Stability benchmark.

Measures robustness of learned representations under
data perturbations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmark.metrics import (
    cosine_similarity,
    relative_error,
)
from benchmark.task import (
    BenchmarkTask,
    TaskResult,
)


@dataclass(slots=True)
class StabilityTask(BenchmarkTask):
    """
    Evaluate fingerprint stability.

    Parameters
    ----------
    noise_level
        Standard deviation of additive Gaussian noise
        relative to signal standard deviation.

    n_repeats
        Number of perturbation repetitions.

    random_state
        Random seed.
    """

    noise_level: float = 0.05

    n_repeats: int = 20

    random_state: int = 42

    @property
    def name(self):

        return "stability"

    # -------------------------------------------------------------

    def evaluate(
        self,
        estimator,
        train_dataset,
        test_dataset,
    ):

        rng = np.random.default_rng(
            self.random_state
        )

        #
        # Original representation
        #

        original = estimator.transform(
            test_dataset
        )

        X_reference = self._stack(
            original
        )

        similarities = []

        errors = []

        #
        # Perturb and transform repeatedly
        #

        for _ in range(
            self.n_repeats
        ):

            perturbed = self._add_noise(
                test_dataset,
                rng,
            )

            transformed = estimator.transform(
                perturbed
            )

            X_perturbed = self._stack(
                transformed
            )

            similarities.append(

                np.mean(

                    [

                        cosine_similarity(
                            a,
                            b,
                        )

                        for a, b in zip(
                            X_reference,
                            X_perturbed,
                        )

                    ]

                )
            )

            errors.append(
                relative_error(
                    X_reference,
                    X_perturbed,
                )
            )

        metrics = {

            "mean_cosine_similarity":
                np.mean(similarities),

            "std_cosine_similarity":
                np.std(similarities),

            "mean_relative_error":
                np.mean(errors),

            "std_relative_error":
                np.std(errors),
        }

        return TaskResult(

            metrics=metrics,

            artifacts={

                "similarities": similarities,

                "relative_errors": errors,
            },
        )

    # -------------------------------------------------------------

    def _add_noise(
        self,
        dataset,
        rng,
    ):
        """
        Create noisy copy of dataset.

        Assumes CurveDataset implements copy().
        """

        noisy = dataset.copy()

        for subject in noisy.subjects:

            scale = np.nanstd(
                subject.curves
            )

            noise = rng.normal(
                loc=0.0,
                scale=self.noise_level * scale,
                size=subject.curves.shape,
            )

            subject.curves = (
                subject.curves
                +
                noise
            )

        return noisy

    # -------------------------------------------------------------

    @staticmethod
    def _stack(
        fingerprints,
    ):

        vectors = []

        for fp in fingerprints:

            if hasattr(
                fp,
                "vector",
            ):

                vectors.append(
                    fp.vector
                )

            else:

                vectors.append(
                    np.asarray(fp)
                )

        return np.vstack(
            vectors
        )
