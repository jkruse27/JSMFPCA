from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pickle
import pandas as pd
from .task import TaskResult


# ---------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------

@dataclass(slots=True)
class BenchmarkResult:
    results: list[TaskResult] = field(default_factory=list)

    # -------------------------------------------------------------
    # Basic API
    # -------------------------------------------------------------

    def add(self, result: TaskResult):
        self.results.append(result)

    @property
    def estimators(self):
        return sorted({r.estimator for r in self.results})

    @property
    def tasks(self):
        return sorted({r.task for r in self.results})

    # -------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------

    def filter(self, estimator=None, task=None):
        results = self.results

        if estimator is not None:
            results = [r for r in results if r.estimator == estimator]

        if task is not None:
            results = [r for r in results if r.task == task]

        return BenchmarkResult(results)

    # -------------------------------------------------------------
    # DataFrame conversion
    # -------------------------------------------------------------

    def dataframe(self):
        rows = []

        for result in self.results:
            row = dict(estimator=result.estimator, task=result.task)
            row.update(result.metrics)
            rows.append(row)

        return pd.DataFrame(rows)

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    def summary(self):
        df = self.dataframe()

        if df.empty:
            return df

        metric_columns = [
            c for c in df.columns if c not in ("estimator", "task")
        ]

        return df.groupby(["task", "estimator"])[metric_columns].mean()

    # -------------------------------------------------------------
    # Pivot table
    # -------------------------------------------------------------

    def pivot(self, metric):
        df = self.dataframe()

        return df.pivot(index="task", columns="estimator", values=metric)

    # -------------------------------------------------------------
    # Export
    # -------------------------------------------------------------

    def latex(self, float_format="%.3f"):
        return self.summary().to_latex(float_format=float_format)

    def csv(self, path):
        self.dataframe().to_csv(path, index=False)

    def save(self, path):
        path = Path(path)

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # -------------------------------------------------------------
    # Artifact access
    # -------------------------------------------------------------

    def artifacts(self, estimator, task) -> dict[str, Any]:
        for result in self.results:
            if (result.estimator == estimator and result.task == task):
                return result.artifacts

        raise KeyError(f"No result for {estimator=} {task=}")

    def __len__(self):
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, item):
        return self.results[item]
