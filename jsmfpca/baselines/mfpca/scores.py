from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from .data import MFPCAResult, SubjectScores, ScoreDataset


# ---------------------------------------------------------------------
# Projection matrices
# ---------------------------------------------------------------------

@dataclass(slots=True)
class ProjectionResult:
    centered: np.ndarray
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray


class ProjectionCalculator:
    def __init__(self, model: MFPCAResult, dt: float = 1.0):
        self.model = model
        self.dt = dt

    def transform(self, data: np.ndarray) -> ProjectionResult:
        mu = self.model.mean
        eta = self.model.visit_mean

        phi = self.model.phi
        psi = self.model.psi

        centered = data - mu[None, None] - eta[None]
        I, J, _ = centered.shape
        A = np.full((I, J, phi.shape[1]), np.nan)
        B = np.full((I, J, psi.shape[1]), np.nan)

        for i in range(I):
            for j in range(J):
                curve = centered[i, j]

                if np.isnan(curve).all():
                    continue

                A[i, j] = (curve @ phi) * self.dt
                B[i, j] = (curve @ psi) * self.dt

        C = (phi.T @ (psi * self.dt))

        return ProjectionResult(centered=centered, A=A, B=B, C=C)


# ---------------------------------------------------------------------
# Residual covariance
# ---------------------------------------------------------------------

class ResidualCovarianceEstimator:
    def __init__(self, ridge: float = 1e-8):
        self.ridge = ridge

    def fit(self, projection: ProjectionResult, model: MFPCAResult):
        A = projection.A
        B = projection.B
        C = projection.C

        n_between = model.n_between
        n_within = model.n_within

        rows = []

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isnan(A[i, j]).all():
                    continue
                rows.append(np.concatenate((A[i, j], B[i, j])))

        rows = np.asarray(rows)
        rows -= rows.mean(axis=0, keepdims=True)

        Sy = (rows.T @ rows) / rows.shape[0]
        G = block_diag(np.diag(model.lambda_phi), np.diag(model.lambda_psi))
        top = np.concatenate((np.eye(n_between), C), axis=1)
        bottom = np.concatenate((C.T, np.eye(n_within)), axis=1)
        Z = np.concatenate((top, bottom), axis=0)
        model_cov = Z @ G @ Z.T
        R = np.diag(np.maximum(np.diag(Sy - model_cov), self.ridge))

        return R


# ---------------------------------------------------------------------
# BLUP estimator
# ---------------------------------------------------------------------

class MFPCA_BLUP:
    def __init__(self, ridge: float = 1e-8):
        self.ridge = ridge

    def estimate(self, projection, model, subject_ids=None):
        A = projection.A
        B = projection.B
        C = projection.C

        R_single = ResidualCovarianceEstimator(self.ridge).fit(
                projection, model
            )

        phi_cov = np.diag(model.lambda_phi)
        psi_cov = np.diag(model.lambda_psi)

        scores = []

        I, J = A.shape[0], A.shape[1]

        for i in range(I):
            observed = np.where(~np.isnan(A[i, :, 0]))[0]

            if len(observed) == 0:
                continue

            Zi = self._build_design(
                len(observed), C, model.n_between, model.n_within
            )

            G = self._build_prior(len(observed), phi_cov, psi_cov)
            R = block_diag(*([R_single] * len(observed)))

            Sigma = Zi @ G @ Zi.T + R
            Sigma = (Sigma + Sigma.T) * 0.5

            c, lower = cho_factor(Sigma, check_finite=False)
            y = np.concatenate(
                [np.concatenate((A[i, j], B[i, j])) for j in observed]
            )

            alpha = cho_solve((c, lower), y, check_finite=False)
            u = G @ Zi.T @ alpha
            xi = u[: model.n_between]
            zeta = u[model.n_between:].reshape(len(observed), model.n_within)

            full_zeta = np.full((J, model.n_within, ), np.nan)
            full_zeta[observed] = zeta

            scores.append(
                SubjectScores(
                    subject_id=(i if subject_ids is None else subject_ids[i]),
                    xi=xi,
                    zeta=full_zeta,
                )
            )

        return ScoreDataset(scores)

    @staticmethod
    def _build_prior(J, phi_cov, psi_cov):
        blocks = [phi_cov]
        blocks.extend(psi_cov for _ in range(J))

        return block_diag(*blocks)

    # -------------------------------------------------------------

    @staticmethod
    def _build_design(J, C, n_between, n_within):
        top = np.concatenate((np.eye(n_between), C), axis=1)
        bottom = np.concatenate((C.T, np.eye(n_within)), axis=1)

        visit = np.concatenate((top, bottom), axis=0)
        obs = n_between + n_within

        rows = []
        left = visit[:, :n_between]
        right = visit[:, n_between:]

        for j in range(J):
            pieces = []
            for k in range(J):
                if j == k:
                    pieces.append(right)
                else:
                    pieces.append(np.zeros((obs, n_within)))

            rows.append(np.concatenate((left, *pieces), axis=1))

        return np.vstack(rows)


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------

def estimate_scores(model, data, subject_ids=None, dt: float = 1.0):
    projection = ProjectionCalculator(model, dt=dt).transform(data)

    return MFPCA_BLUP().estimate(projection, model, subject_ids=subject_ids)
