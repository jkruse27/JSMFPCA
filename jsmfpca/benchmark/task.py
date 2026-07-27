from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------

@dataclass(slots=True)
class TaskResult:
    estimator: str = ""
    task: str = ""
    fold: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Base task
# ---------------------------------------------------------------------

class BenchmarkTask(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def evaluate(self, estimator, dataset) -> TaskResult:
        ...

    def __call__(self, estimator, dataset) -> TaskResult:
        return self.evaluate(estimator, dataset)
