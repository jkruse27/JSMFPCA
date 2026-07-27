from __future__ import annotations
import numpy as np
from benchmark.task import BenchmarkTask, TaskResult
from benchmark.metrics import (
    mae, mse, r2, reconstruction_error, relative_error
)


class ReconstructionTask(BenchmarkTask):
    @property
    def name(self):
        return "reconstruction"

    def evaluate(self, estimator, train_dataset, test_dataset):
        reconstruction = estimator.reconstruct(test_dataset)
        observed = self._stack_dataset(test_dataset)
        predicted = self._stack_reconstruction(reconstruction)

        mask = np.isfinite(observed) & np.isfinite(predicted)

        observed = observed[mask]
        predicted = predicted[mask]

        metrics = {
            "rmse": reconstruction_error(observed, predicted),
            "mse": mse(observed, predicted),
            "mae": mae(observed, predicted),
            "r2": r2(observed, predicted),
            "relative_error": relative_error(observed, predicted)
        }

        return TaskResult(
            metrics=metrics,
            artifacts={"observed": observed, "predicted": predicted}
        )

    @staticmethod
    def _stack_dataset(dataset):
        curves = []

        for subject in dataset.subjects:
            curves.append(np.asarray(subject.curves))

        return np.concatenate(curves, axis=0)

    @staticmethod
    def _stack_reconstruction(reconstruction):
        if isinstance(reconstruction, np.ndarray):
            if reconstruction.ndim == 3:
                return np.concatenate(reconstruction, axis=0)
            return reconstruction

        return np.concatenate(reconstruction, axis=0)
