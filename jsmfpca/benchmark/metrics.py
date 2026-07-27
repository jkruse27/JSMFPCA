from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, normalized_mutual_info_score,
    precision_score, r2_score, recall_score, roc_auc_score, silhouette_score
)


# ---------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def relative_rmse(y_true, y_pred):
    den = np.sqrt(np.mean(np.square(y_true)))

    return np.nan if den == 0 else rmse(y_true, y_pred) / den


# ---------------------------------------------------------------------
# Matrix metrics
# ---------------------------------------------------------------------

def frobenius(A, B):
    return np.linalg.norm(A - B, ord="fro")


def spectral_norm(A, B):
    return np.linalg.norm(A - B, ord=2)


def principal_angle(U, V):
    s = np.clip(np.linalg.svd(U.T @ V, compute_uv=False), -1, 1)

    return np.arccos(s.min())  # Largest principal angle between two subspaces.


# ---------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)


def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)


def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)


def auc(y_true, scores):
    return roc_auc_score(y_true, scores)


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


# ---------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------

def silhouette(X, labels):
    return silhouette_score(X, labels)


def ari(labels_true, labels_pred):
    return adjusted_rand_score(labels_true, labels_pred)


def nmi(labels_true, labels_pred):
    return normalized_mutual_info_score(labels_true, labels_pred)


# ---------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------

def cosine_similarity(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)

    return np.nan if (nx == 0 or ny == 0) else (x @ y) / (nx * ny)


def relative_error(reference, estimate):
    reference = np.asarray(reference)
    estimate = np.asarray(estimate)

    den = np.linalg.norm(reference)

    return np.nan if den == 0 else np.linalg.norm(estimate - reference) / den


# ---------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------

def reconstruction_error(observed, reconstructed):
    observed = np.asarray(observed)
    reconstr = np.asarray(reconstructed)
    mask = np.isfinite(observed) & np.isfinite(reconstructed)

    return np.nan if mask.sum == 0 else rmse(observed[mask], reconstr[mask])
