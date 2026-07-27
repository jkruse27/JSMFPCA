from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.base import clone
from sklearn.cluster import KMeans
from benchmark.metrics import silhouette, ari, nmi
from benchmark.task import BenchmarkTask, TaskResult


@dataclass(slots=True)
class ClusteringTask(BenchmarkTask):
    n_clusters: int = 2
    clustering_model: object | None = None
    labels: np.ndarray | None = None

    @property
    def name(self):
        return "clustering"

    def evaluate(self, estimator, train_dataset, test_dataset):
        fingerprints = estimator.transform(test_dataset)
        X = self._stack(fingerprints)

        if self.clustering_model is None:
            clusterer = KMeans(
                n_clusters=self.n_clusters, random_state=0, n_init=20
            )

        else:
            clusterer = clone(self.clustering_model)

        cluster_labels = clusterer.fit_predict(X)
        metrics = {"silhouette": silhouette(X, cluster_labels)}

        if self.labels is not None:
            y_true = self.labels[test_dataset.subject_indices]
            metrics["ari"] = ari(y_true, cluster_labels)
            metrics["nmi"] = nmi(y_true, cluster_labels)

        return TaskResult(
            metrics=metrics,
            artifacts={
                "clusterer": clusterer,
                "cluster_labels": cluster_labels,
                "fingerprints": X
            }
        )

    @staticmethod
    def _stack(fingerprints):
        vectors = []

        for fp in fingerprints:
            if hasattr(fp, "vector"):
                vectors.append(fp.vector)
            else:
                vectors.append(np.asarray(fp))

        return np.vstack(vectors)
