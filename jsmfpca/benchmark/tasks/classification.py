from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from benchmark.metrics import accuracy, auc, confusion, f1, precision, recall
from benchmark.task import BenchmarkTask, TaskResult


@dataclass(slots=True)
class ClassificationTask(BenchmarkTask):
    labels: np.ndarray
    classifier: object = LogisticRegression(max_iter=1000)
    probability: bool = True

    @property
    def name(self):
        return "classification"

    def evaluate(self, estimator, train_dataset, test_dataset):
        train_fp = estimator.transform(train_dataset)
        test_fp = estimator.transform(test_dataset)

        X_train = self._stack(train_fp)
        X_test = self._stack(test_fp)

        y_train = self.labels[train_dataset.subject_indices]
        y_test = self.labels[test_dataset.subject_indices]

        clf = clone(self.classifier)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        metrics = {
            "accuracy": accuracy(y_test, y_pred),
            "precision": precision(y_test, y_pred),
            "recall": recall(y_test, y_pred),
            "f1": f1(y_test, y_pred)
        }

        artifacts = {
            "classifier": clf,
            "confusion_matrix": confusion(y_test, y_pred)
        }

        if self.probability and hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(X_test)
            if scores.shape[1] == 2:
                scores = scores[:, 1]
            metrics["auc"] = auc(y_test, scores)
            artifacts["scores"] = scores

        return TaskResult(metrics=metrics, artifacts=artifacts)

    @staticmethod
    def _stack(fingerprints):
        X = []

        for fp in fingerprints:
            if hasattr(fp, "vector"):
                X.append(fp.vector)
            else:
                X.append(np.asarray(fp))

        return np.vstack(X)
