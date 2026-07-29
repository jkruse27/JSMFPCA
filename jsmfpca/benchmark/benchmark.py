from __future__ import annotations
from time import perf_counter
import numpy as np
from sklearn.model_selection import KFold
from .results import BenchmarkResult
from .task import BenchmarkTask


# ---------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------

class Benchmark:
    def __init__(
        self, estimators, tasks, cv=None,
        random_state=42, refit=True, verbose=True
    ):
        self.estimators = list(estimators)
        self.tasks = list(tasks)
        self.cv = cv
        self.random_state = random_state
        self.refit = refit
        self.verbose = verbose

    def evaluate(self, dataset) -> BenchmarkResult:
        results = BenchmarkResult()

        for estimator in self.estimators:
            estimator_name = estimator.__class__.__name__

            if self.verbose:
                print(f"Evaluating {estimator_name}")

            for fold, (train, test) in enumerate(self._splits(dataset)):
                if self.verbose and self.cv is not None:
                    print(f"  Fold {fold + 1}/{self.cv}")

                must_fit = (
                    self.cv is not None or
                    self.refit or
                    not getattr(estimator, "fitted", False)
                )

                if must_fit:
                    estimator.fit(train)

                for task in self.tasks:
                    if self.verbose:
                        print(f"    -> {task.name}")

                    start = perf_counter()
                    result = task.evaluate(estimator, train, test)
                    elapsed = perf_counter() - start

                    result.estimator = estimator_name
                    result.task = task.name
                    result.fold = fold
                    result.metrics.setdefault("evaluation_time", elapsed)
                    results.add(result)

        return results

    def evaluate_estimator(self, estimator, dataset) -> BenchmarkResult:
        return Benchmark(
            estimators=[estimator], tasks=self.tasks,
            refit=self.refit, verbose=self.verbose
        ).evaluate(dataset)

    def evaluate_task(self, task, dataset) -> BenchmarkResult:
        return Benchmark(
            estimators=self.estimators, tasks=[task],
            refit=self.refit, verbose=self.verbose
        ).evaluate(dataset)

    def add_task(self, task: BenchmarkTask):
        self.tasks.append(task)

    def add_estimator(self, estimator):
        self.estimators.append(estimator)

    def remove_task(self, name):
        self.tasks = [task for task in self.tasks if task.name != name]

    def remove_estimator(self, estimator_type):
        self.estimators = [
            estimator for estimator in self.estimators
            if not isinstance(estimator, estimator_type)
        ]

    @property
    def estimator_names(self):
        return [estimator.__class__.__name__ for estimator in self.estimators]

    @property
    def task_names(self):
        return [task.name for task in self.tasks]

    def __len__(self):
        return len(self.estimators) * len(self.tasks)

    def __repr__(self):
        return f"Benchmark({len(self.estimators)}, {len(self.tasks)})"

    def _splits(self, dataset):
        if self.cv is None:
            dataset.subject_indices = np.arange(dataset.n_subjects)
            yield dataset, dataset
            return

        kfold = KFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        for train_idx, test_idx in kfold.split(dataset.subjects):
            train = dataset.subset(train_idx)
            test = dataset.subset(test_idx)
            train.subject_indices = train_idx
            test.subject_indices = test_idx

            yield train, test
