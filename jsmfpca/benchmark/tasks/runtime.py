from __future__ import annotations
import tracemalloc
from time import perf_counter
from benchmark.task import BenchmarkTask, TaskResult


class RuntimeTask(BenchmarkTask):
    @property
    def name(self):
        return "runtime"

    def evaluate(self, estimator, train_dataset, test_dataset):
        tracemalloc.start()
        start = perf_counter()
        estimator.fit(train_dataset)
        fit_time = (perf_counter() - start)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        start = perf_counter()
        transformed = estimator.transform(test_dataset)
        transform_time = perf_counter() - start

        reconstruction_time = None

        if hasattr(estimator, "reconstruct"):
            start = perf_counter()
            estimator.reconstruct(test_dataset)
            reconstruction_time = perf_counter() - start

        metrics = {
            "fit_time": fit_time,
            "transform_time": transform_time,
            "peak_memory_mb": peak / 1024 / 1024
        }

        if reconstruction_time is not None:
            metrics["reconstruct_time"] = reconstruction_time

        return TaskResult(
            metrics=metrics, artifacts={"representation": transformed}
        )
