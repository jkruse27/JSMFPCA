# File: validation/association.py

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from physfunc.data import JSMFPCAData
from physfunc.jsmfpca import JSMFPCA
from physfunc.mfpca import TraditionalMFPCA
from physfunc.diagonal import DiagonalSpectralModel
from physfunc.ols import OLSHarmonicEstimator
from physfunc.cosinor import ClassicalCosinor
from definitions import load_jsmfpca_dataset, get_features

# All self-contained estimators
ESTIMATORS = [
    JSMFPCA,
    TraditionalMFPCA,
    DiagonalSpectralModel,
    OLSHarmonicEstimator,
    ClassicalCosinor,
]


def prepare_features_and_target(
    dataset: JSMFPCAData, fingerprints: np.ndarray,
    features_df: pd.DataFrame, target_key: str
) -> tuple[np.ndarray, np.ndarray]:
    if "subject_id" in features_df.columns:
        features_df = features_df.drop_duplicates(
            subset=["subject_id"]
        ).set_index("subject_id")
    else:
        features_df = features_df[~features_df.index.duplicated(keep="first")]

    X_list, y_list = [], []

    for i, subj in enumerate(dataset.subjects):
        subj_id = str(subj.subject_id)

        if subj_id in features_df.index:
            target_val = features_df.loc[subj_id, target_key]
            if isinstance(target_val, pd.Series):
                target_val = target_val.iloc[0]

            if not pd.isna(target_val):
                X_list.append(fingerprints[i])
                y_list.append(target_val)

    if not X_list:
        raise ValueError(
            f"No valid matched samples found for target '{target_key}'."
        )

    return np.vstack(X_list), np.asarray(y_list)


def evaluate_bnp_association(
    EstimatorClass, dataset: JSMFPCAData, features_df: pd.DataFrame
):
    model = EstimatorClass()
    fingerprints = model.fit_transform(dataset)

    X, y_bnp = prepare_features_and_target(
        dataset, fingerprints, features_df, target_key="BNP"
    )

    regressor = Ridge(alpha=1.0)
    scores = cross_val_score(regressor, X, y_bnp, cv=3, scoring="r2")
    mean_r2 = float(np.mean(scores))

    print(f"  {EstimatorClass.__name__:<22} | BNP R^2: {mean_r2:.4f}")
    assert np.isfinite(mean_r2), f"Infinite R^2 for {EstimatorClass.__name__}"


def evaluate_survival_association(
    EstimatorClass, dataset: JSMFPCAData, features_df: pd.DataFrame
):
    model = EstimatorClass()
    fingerprints = model.fit_transform(dataset)

    X, y_state = prepare_features_and_target(
        dataset, fingerprints, features_df, target_key="state"
    )

    if len(np.unique(y_state)) > 1:
        classifier = LogisticRegression(max_iter=1000)
        scores = cross_val_score(
            classifier, X, y_state, cv=3, scoring="roc_auc"
        )
        mean_auc = float(np.mean(scores))
    else:
        mean_auc = np.nan

    print(
        f"  {EstimatorClass.__name__:<22} | Survival ROC-AUC: {mean_auc:.4f}"
    )


def run_associations():
    dataset_name = "986"
    jsmfpca_data = load_jsmfpca_dataset(
        window_size=2, feat="lmds", dataset_name=dataset_name
    )
    features_dataframe = get_features(2, dataset_name)

    print("\n" + "=" * 55)
    print("Running Clinical Association Benchmark Suite")
    print("=" * 55)

    for EstimatorClass in ESTIMATORS:
        try:
            evaluate_bnp_association(
                EstimatorClass, jsmfpca_data, features_dataframe
            )
            evaluate_survival_association(
                EstimatorClass, jsmfpca_data, features_dataframe
            )
            print("-" * 55)
        except Exception as e:
            print(f" Error evaluating {EstimatorClass.__name__}: {e}")


if __name__ == "__main__":
    run_associations()
