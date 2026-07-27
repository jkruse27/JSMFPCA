from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------
# Lag covariance
# ---------------------------------------------------------------------

def estimate_lag_covariance(dataset, weighting="subject"):
    K = dataset.n_components
    Sigma = np.zeros((24, K, K))

    if weighting not in ("subject", "pair"):
        raise ValueError("weighting must be 'subject' or 'pair'.")

    # ----------------------------------------------------------
    # Subject-weighted estimator
    # ----------------------------------------------------------

    if weighting == "subject":
        subject_count = np.zeros(24)

        for lag in range(24):
            for subject in dataset.subjects:
                pairs = subject._lag_pairs[lag]

                if not pairs:
                    continue

                cov = np.zeros((K, K))

                for i, j in pairs:
                    cov += np.outer(subject.centered[i], subject.centered[j])

                Sigma[lag] += cov / len(pairs)
                subject_count[lag] += 1

        for lag in range(24):
            if subject_count[lag] > 0:
                Sigma[lag] /= subject_count[lag]

    # ----------------------------------------------------------
    # Pair-weighted estimator
    # ----------------------------------------------------------

    else:
        counts = np.zeros(24)

        for lag in range(24):
            for _, _, x, y in dataset.observed_pairs(lag):
                Sigma[lag] += np.outer(x, y)
                counts[lag] += 1

        for lag in range(24):
            if counts[lag] > 0:
                Sigma[lag] /= counts[lag]

    return Sigma


# ---------------------------------------------------------------------
# Lag counts
# ---------------------------------------------------------------------

def lag_counts(dataset):
    counts = np.zeros(24, dtype=int)

    for lag in range(24):
        for _ in dataset.observed_pairs(lag):
            counts[lag] += 1

    return counts


# ---------------------------------------------------------------------
# Hermitian projection
# ---------------------------------------------------------------------

def make_hermitian(S):
    return 0.5 * (S + S.conj().T)
