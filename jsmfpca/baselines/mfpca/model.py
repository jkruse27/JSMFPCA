from __future__ import annotations
import numpy as np
from scipy.linalg import eigh
from ...utils import nearest_psd, sort_eigensystem
from .data import MFPCAResult


class TraditionalMFPCA:
    def __init__(self, explained_variance=0.99):
        self.explained_variance = explained_variance
        self.model_: MFPCAResult | None = None

    @staticmethod
    def _choose_components(explained, threshold):

        return np.searchsorted(np.cumsum(explained), threshold) + 1

    def fit(self, data):
        n_subjects, n_visits, n_time = data.shape

        mu = np.nanmean(data, axis=(0, 1))
        eta = np.nanmean(data - mu, axis=0)

        centered = data - mu - eta[None]
        subject_mean = np.nanmean(centered, axis=1)

        residual = centered - subject_mean[:, None, :]
        residual_flat = residual.reshape(n_subjects * n_visits, n_time)

        Kw = np.ma.cov(np.ma.masked_invalid(residual_flat), rowvar=False).data
        K_subj = np.ma.cov(
            np.ma.masked_invalid(subject_mean), rowvar=False
        ).data

        J_i = np.sum(~np.isnan(data[:, :, 0]), axis=1)
        J_harmonic = len(J_i) / np.sum(1.0 / np.maximum(J_i, 1))

        Kb_raw = K_subj - (1.0 / J_harmonic) * Kw
        Kb = nearest_psd(Kb_raw)

        vals_b, vecs_b = sort_eigensystem(*eigh(Kb))
        vals_w, vecs_w = sort_eigensystem(*eigh(Kw))

        nb = self._choose_components(
            vals_b / vals_b.sum(), self.explained_variance
        )
        nw = self._choose_components(
            vals_w / vals_w.sum(), self.explained_variance
        )

        self.model_ = MFPCAResult(
            mean=mu,
            visit_mean=eta,
            phi=vecs_b[:, :nb],
            psi=vecs_w[:, :nw],
            lambda_phi=vals_b[:nb],
            lambda_psi=vals_w[:nw],
            explained_between=(vals_b / vals_b.sum())[:nb],
            explained_within=(vals_w / vals_w.sum())[:nw],
        )
        return self

    @property
    def fitted(self):
        return self.model_ is not None

    @property
    def n_between(self):
        return self.model_.n_between

    @property
    def n_within(self):
        return self.model_.n_within

    @property
    def result(self):
        return self.model_
