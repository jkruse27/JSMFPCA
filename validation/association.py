from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
import os
import sys

# Path setup
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "0_Commons",
    )
)
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from jsmfpca.pipeline import JSMFPCA  # noqa: E402
from jsmfpca.data import JSMFPCAData, SubjectCurves  # noqa: E402
from jsmfpca.baselines.mfpca.pipeline import (  # noqa: E402
    TraditionalMFPCAPipeline
)
from jsmfpca.baselines.diagonal import DiagonalSpectralPipeline  # noqa: E402
from jsmfpca.baselines.ols import OLSHarmonicPipeline  # noqa: E402
from jsmfpca.baselines.fpca import FPCA  # noqa: E402
from definitions import get_curves, get_features  # noqa: E402

BASELINES = [
    JSMFPCA,
    TraditionalMFPCAPipeline,
    DiagonalSpectralPipeline,
    OLSHarmonicPipeline,
    FPCA,
]


def load_jsmfpca_dataset(
    window_size, feat, dataset_name, valid_hours=range(24)
):
    print(
        f"Loading data for {dataset_name} - {feat} (Window: {window_size}h)..."
    )

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

        for subj_id, row in df_curves.iterrows():
            curve = row[df_curves.columns].values.astype(float)

            if np.isnan(curve).all():
                continue

            if subj_id not in subject_data:
                subject_data[subj_id] = {
                    'hours': [],
                    'curves': []
                }

            subject_data[subj_id]['hours'].append(hour)
            subject_data[subj_id]['curves'].append(curve)

    subjects = []

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

    if not subjects:
        raise ValueError("No valid subjects found after aggregation.")

    print(f"  -> Successfully loaded {len(subjects)} subjects.")

    dataset = JSMFPCAData(subjects=subjects, scales=scales)
    return dataset


# ---------------------------------------------------------------------
# Helper function to extract X and y
# ---------------------------------------------------------------------
def _prepare_data(fingerprints, features_df, target_key):
    X = []
    y = []

    if 'subject_id' in features_df.columns:
        features_df = features_df.drop_duplicates(
            subset=['subject_id']
        ).set_index('subject_id')
    else:
        features_df = features_df[~features_df.index.duplicated(keep='first')]

    for fp in fingerprints:
        subj_id = fp.subject_id

        if subj_id in features_df.index:
            target_value = features_df.loc[subj_id, target_key]

            if isinstance(target_value, pd.Series):
                target_value = target_value.iloc[0]

            if not pd.isna(target_value):
                vector = fp.vector if hasattr(fp, "vector") else np.asarray(fp)
                X.append(vector)
                y.append(target_value)

    if len(X) == 0:
        raise ValueError(f"No valid samples found for target: {target_key}")

    return np.vstack(X), np.asarray(y)


# ---------------------------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------------------------
def evaluate_bnp_association(Estimator, dataset, features_df):
    model = Estimator()
    fingerprints = model.fit_transform(dataset)

    X, y_bnp = _prepare_data(fingerprints, features_df, target_key="BNP")

    regressor = Ridge(alpha=1.0)
    scores = cross_val_score(regressor, X, y_bnp, cv=3, scoring="r2")
    mean_r2 = np.max(scores)

    print(f"{Estimator.__name__} - BNP Association (R^2): {mean_r2:.4f}")
    assert np.isfinite(mean_r2), f"Non-finite R^2 for {Estimator.__name__}"


def evaluate_survival_state_association(Estimator, dataset, features_df):
    model = Estimator()
    fingerprints = model.fit_transform(dataset)

    X, y_state = _prepare_data(fingerprints, features_df, target_key="state")

    classifier = LogisticRegression(max_iter=1000)

    if len(np.unique(y_state)) > 1:
        scores = cross_val_score(
            classifier, X, y_state, cv=3, scoring="roc_auc"
        )
        mean_auc = np.max(scores)
    else:
        mean_auc = np.nan

    print(
        f"{Estimator.__name__} - Survival "
        f"State Association (AUC): {mean_auc:.4f}"
    )
    assert (
        np.isfinite(mean_auc) or np.isnan(mean_auc)
    ), f"Invalid AUC for {Estimator.__name__}"


# ---------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    dataset_name = "986"

    jsmfpca_data = load_jsmfpca_dataset(
        window_size=2, feat="lmds", dataset_name=dataset_name
    )
    features_dataframe = get_features(2, dataset_name)

    print("\nStarting Evaluations...\n" + "="*40)
    for EstimatorClass in BASELINES:
        print(f"\nEvaluating: {EstimatorClass.__name__}")
        print("-" * 40)

        evaluate_bnp_association(
            EstimatorClass, jsmfpca_data, features_dataframe
        )

        evaluate_survival_state_association(
            EstimatorClass, jsmfpca_data, features_dataframe
        )
