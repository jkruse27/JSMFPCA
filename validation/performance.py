# File: validation/test.py

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    roc_auc_score,
    silhouette_score,
    adjusted_rand_score,
)
from sklearn.model_selection import KFold
from physfunc.data import JSMFPCAData
from physfunc.jsmfpca import JSMFPCA
from physfunc.mfpca import TraditionalMFPCA
from physfunc.diagonal import DiagonalSpectralModel
from physfunc.ols import OLSHarmonicEstimator
from physfunc.cosinor import ClassicalCosinor
from definitions import RESULTS, WINDOWS, load_jsmfpca_dataset


def evaluate_estimator(
    estimator, dataset: JSMFPCAData, labels: np.ndarray, cv: int = 5
) -> dict:
    # 1. Fit & Transform
    fingerprints = estimator.fit_transform(dataset)

    # 2. Reconstruction Error (MSE)
    reconstructed_curves = estimator.reconstruct(dataset)
    mse_list = []
    for orig_subj, rec_curves in zip(dataset.subjects, reconstructed_curves):
        mse = mean_squared_error(orig_subj.curves, rec_curves)
        mse_list.append(mse)
    mean_mse = float(np.mean(mse_list))

    # 3. Classification Performance (5-Fold CV Logistic Regression)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    acc_scores, auc_scores = [], []

    for train_idx, val_idx in kf.split(fingerprints):
        X_tr, X_val = fingerprints[train_idx], fingerprints[val_idx]
        y_tr, y_val = labels[train_idx], labels[val_idx]

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val)

        acc_scores.append(accuracy_score(y_val, preds))
        if len(np.unique(labels)) > 1:
            probs = clf.predict_proba(X_val)
            auc_scores.append(roc_auc_score(y_val, probs, multi_class="ovr"))

    mean_acc = float(np.mean(acc_scores))
    mean_auc = float(np.mean(auc_scores)) if auc_scores else np.nan

    # 4. Clustering Performance (K-Means)
    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(fingerprints)

    sil_score = float(silhouette_score(fingerprints, cluster_labels))
    ari_score = float(adjusted_rand_score(labels, cluster_labels))

    return {
        "Reconstruction_MSE": mean_mse,
        "Classification_Acc": mean_acc,
        "Classification_AUC": mean_auc,
        "Silhouette_Score": sil_score,
        "Adjusted_Rand_Index": ari_score,
    }


def run_benchmarks():
    dataset_name = "986"
    feat = "lmds"

    for window_size in WINDOWS:
        print(
            f"\n{'=' * 60}\nEvaluating Window Size: {window_size}h\n{'=' * 60}"
        )

        try:
            dataset, labels = load_jsmfpca_dataset(
                window_size, feat, dataset_name
            )
        except Exception as e:
            print(f"Skipping window {window_size} due to error: {e}")
            continue

        estimators = {
            "JS-MFPCA": JSMFPCA(n_modes=3, n_harmonics=2, shrinkage=0.25),
            "Traditional_MFPCA": TraditionalMFPCA(explained_variance=0.95),
            "Diagonal_Spectral": DiagonalSpectralModel(
                n_modes=3, n_harmonics=2, shrinkage=0.25
            ),
            "OLS_Harmonic": OLSHarmonicEstimator(
                n_modes=3, n_harmonics=2, shrinkage=0.25
            ),
            "Classical_Cosinor": ClassicalCosinor(n_harmonics=2),
        }

        results_dict = {}

        for name, est in estimators.items():
            print(f"Evaluating estimator: {name}...")
            try:
                metrics = evaluate_estimator(est, dataset, labels, cv=5)
                results_dict[name] = metrics
            except Exception as e:
                print(f"  Error evaluating {name}: {e}")

        summary_df = pd.DataFrame.from_dict(results_dict, orient="index")

        # Export Results
        out_dir = os.path.join(
            RESULTS, dataset_name, "benchmark", str(window_size)
        )
        os.makedirs(out_dir, exist_ok=True)
        summary_path = os.path.join(out_dir, f"{feat}_benchmark_summary.csv")
        summary_df.to_csv(summary_path)

        print(f"\n--- Benchmark Summary Table (Window: {window_size}h) ---")
        print(summary_df.to_string())
        print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    run_benchmarks()
