import os
import sys
import numpy as np
import pandas as pd

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "0_Commons",
    )
)
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from jsmfpca.data import JSMFPCAData, SubjectCurves  # noqa: E402
from jsmfpca.pipeline.estimator import JSMFPCA  # noqa: E402
from jsmfpca.baselines.mfpca.pipeline import (  # noqa: E402
    TraditionalMFPCAPipeline
)
from jsmfpca.baselines.ols import OLSHarmonicPipeline  # noqa: E402
from jsmfpca.baselines.diagonal import DiagonalSpectralPipeline  # noqa: E402
from jsmfpca.baselines.fpca import FPCA  # noqa: E402
from jsmfpca.benchmark.benchmark import Benchmark  # noqa: E402
from jsmfpca.benchmark.tasks.classification import (  # noqa: E402
    ClassificationTask
)
from jsmfpca.benchmark.tasks.reconstruction import (  # noqa: E402
    ReconstructionTask
)
from jsmfpca.benchmark.tasks.clustering import ClusteringTask  # noqa: E402
from constants import RESULTS, WINDOWS  # noqa: E402
from definitions import get_curves, get_features  # noqa: E402


def load_jsmfpca_dataset(
    window_size, feat, dataset_name, valid_hours=range(24)
):
    print(
        f"Loading data for {dataset_name} - {feat} (Window: {window_size}h)..."
    )

    df_features = get_features(window_size, dataset_name)
    if 'BNP' not in df_features.columns:
        raise ValueError(f"'BNP' column not found in dataset {dataset_name}.")

    df_bnp = df_features[['BNP']].groupby(df_features.index).first().dropna()
    df_bnp['bnp_tertile'] = pd.qcut(df_bnp['BNP'], q=3, labels=[0, 1, 2])

    subject_data = {}
    scales = None

    for hour in valid_hours:
        try:
            df_curves = get_curves(window_size, hour, feat, dataset_name)
        except Exception as e:
            print(f"  Warning: Could not load hour {hour} ({e})")
            continue

        if df_curves is None or df_curves.empty:
            continue

        if scales is None:
            scales = df_curves.columns.astype(float).values

        df_merged = df_curves.join(df_bnp['bnp_tertile'], how='inner')

        for subj_id, row in df_merged.iterrows():
            curve = row[df_curves.columns].values.astype(float)

            if np.isnan(curve).all():
                continue

            if subj_id not in subject_data:
                subject_data[subj_id] = {
                    'hours': [],
                    'curves': [],
                    'label': row['bnp_tertile']
                }

            subject_data[subj_id]['hours'].append(hour)
            subject_data[subj_id]['curves'].append(curve)

    subjects = []
    labels = []
    subject_ids = []

    for subj_id, data in subject_data.items():
        if len(data['hours']) < 4:
            continue

        subjects.append(
            SubjectCurves(
                subject_id=str(subj_id),
                hours=np.array(data['hours'], dtype=int),
                curves=np.vstack(data['curves'])
            )
        )
        labels.append(data['label'])
        subject_ids.append(str(subj_id))

    if not subjects:
        raise ValueError("No valid subjects found after aggregation.")

    print(f"  -> Successfully loaded {len(subjects)} subjects.")

    dataset = JSMFPCAData(subjects=subjects, scales=scales)
    return dataset, np.array(labels)


def run_benchmarks():
    dataset_name = '986'
    feat = 'lmds'  # or 'alphas'

    for window_size in WINDOWS:
        print(f"\n{'='*50}\nEvaluating Window Size: {window_size}\n{'='*50}")

        try:
            dataset, labels = load_jsmfpca_dataset(
                window_size, feat, dataset_name
            )
        except Exception as e:
            print(f"Skipping window {window_size} due to error: {e}")
            continue

        # Initialize Tasks
        tasks = [
            ReconstructionTask(),
            ClusteringTask(n_clusters=3, labels=labels),
            ClassificationTask(labels=labels, probability=True)
        ]

        # Initialize Estimators
        estimators = [
            JSMFPCA(n_modes="cv", n_harmonics=2, shrinkage="cv"),
            TraditionalMFPCAPipeline(explained_variance=0.95),
            DiagonalSpectralPipeline(
                n_modes="cv", n_harmonics=2, shrinkage="cv"
            ),
            FPCA(n_components="cv"),
            OLSHarmonicPipeline(n_modes="cv", n_harmonics=2, shrinkage="cv")
        ]

        # Run Benchmark Framework
        benchmark = Benchmark(
            estimators=estimators,
            tasks=tasks,
            cv=5,
            random_state=42,
            verbose=True
        )

        results = benchmark.evaluate(dataset)

        # Export Results
        out_dir = os.path.join(
            RESULTS, dataset_name, "benchmark", str(window_size)
        )
        os.makedirs(out_dir, exist_ok=True)

        # Save summary statistics as CSV
        summary_df = results.summary()
        summary_path = os.path.join(out_dir, f"{feat}_benchmark_summary.csv")
        summary_df.to_csv(summary_path)
        print(f"\nBenchmark results saved to: {summary_path}")

        # Print a quick terminal overview of the results
        print("\n--- Benchmark Summary ---")
        print(summary_df)


if __name__ == "__main__":
    run_benchmarks()
