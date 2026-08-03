# File: jsmfpca/baselines/deep_learning.py

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


class FunctionalAutoencoder:
    """
    Circadian Functional Autoencoder Estimator.
    Compresses 24-hour scale profiles Y_i(h, s) into a low-dimensional
    bottleneck fingerprint z_i using a non-linear neural autoencoder

    Parameters
    ----------
    latent_dim : int, default=16
        Bottleneck fingerprint dimension (P).
    hidden_dim : int, default=32
        Hidden layer width for scale and temporal encoders.
    max_iter : int, default=200
        Maximum L-BFGS-B optimization iterations.
    l2_reg : float, default=1e-4
        L2 weight decay regularization.
    period : float, default=24.0
        Circadian cycle length in hours.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        max_iter: int = 200,
        l2_reg: float = 1e-4,
        period: float = 24.0,
    ):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.l2_reg = l2_reg
        self.period = period

        self.n_scales_ = None
        self.weights_ = None
        self.is_fitted_ = False

    # =========================================================================
    # Public Estimator API
    # =========================================================================

    def fit(self, dataset):
        """Fit Autoencoder weights on observed 24-hour subject profiles."""
        subjects_data, masks = self._prepare_3d_tensor(dataset)
        N, H, S = subjects_data.shape
        self.n_scales_ = S

        # Initialize network weights
        self.weights_ = self._init_weights(S, H)

        # Optimize autoencoder loss using L-BFGS-B
        w_vec = self._flatten_weights(self.weights_)
        res = minimize(
            fun=self._loss_and_grad,
            x0=w_vec,
            args=(subjects_data, masks, S, H),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "disp": False},
        )

        self.weights_ = self._unflatten_weights(res.x, S, H)
        self.is_fitted_ = True
        return self

    def transform(self, dataset) -> np.ndarray:
        """Encode subjects into bottleneck physiological fingerprints z_i."""
        self._check_fitted()
        subjects_data, masks = self._prepare_3d_tensor(dataset)
        z_list = []

        for i in range(len(dataset.subjects)):
            X_i = subjects_data[i]  # (24, S)
            mask_i = masks[i]        # (24,)
            z_i = self._encode_subject(X_i, mask_i, self.weights_)
            z_list.append(z_i)

        return np.vstack(z_list)

    def fit_transform(self, dataset) -> np.ndarray:
        return self.fit(dataset).transform(dataset)

    def reconstruct(self, dataset) -> list[np.ndarray]:
        """Reconstruct 24-hour scale profiles from bottleneck embeddings."""
        self._check_fitted()
        subjects_data, masks = self._prepare_3d_tensor(dataset)
        reconstructed_list = []

        for i, subj in enumerate(dataset.subjects):
            X_i = subjects_data[i]
            mask_i = masks[i]

            z_i = self._encode_subject(X_i, mask_i, self.weights_)
            X_rec = self._decode_subject(
                z_i, self.weights_, S=self.n_scales_, H=24
            )

            # Return reconstructed curves for observed hours
            obs_hours = subj.hours
            reconstructed_list.append(X_rec[obs_hours])

        return reconstructed_list

    # =========================================================================
    # Neural Network Forward Pass & Loss
    # =========================================================================

    def _encode_subject(
        self, X_i: np.ndarray, mask_i: np.ndarray, w: dict
    ) -> np.ndarray:
        """Forward Encoder: (24, S) -> (Latent_Dim,)."""
        # 1. Spatial Scale Encoder
        h_spatial = np.tanh(X_i @ w["W_spatial"] + w["b_spatial"])
        h_spatial = h_spatial * mask_i[:, None]

        # 2. Temporal Bottleneck Encoder
        h_flat = h_spatial.ravel()
        z_i = np.tanh(h_flat @ w["W_bottleneck"] + w["b_bottleneck"])
        return z_i

    def _decode_subject(
        self, z_i: np.ndarray, w: dict, S: int, H: int
    ) -> np.ndarray:
        """Forward Decoder: (Latent_Dim,) -> (24, S)."""
        # 1. Temporal Decoder
        h_flat_rec = np.tanh(z_i @ w["W_dec_temp"] + w["b_dec_temp"])
        h_spatial_rec = h_flat_rec.reshape(H, self.hidden_dim)

        # 2. Spatial Scale Decoder
        X_rec = h_spatial_rec @ w["W_dec_spatial"] + w["b_dec_spatial"]
        return X_rec

    def _loss_and_grad(
        self, w_vec: np.ndarray, data: np.ndarray,
        masks: np.ndarray, S: int, H: int
    ):
        """Compute MSE loss over observed hours + L2 regularization."""
        w = self._unflatten_weights(w_vec, S, H)
        N = data.shape[0]
        loss = 0.0

        for i in range(N):
            X_i = data[i]
            mask_i = masks[i]

            z_i = self._encode_subject(X_i, mask_i, w)
            X_rec = self._decode_subject(z_i, w, S, H)

            # MSE loss over observed hours only
            diff = (X_i - X_rec) * mask_i[:, None]
            loss += np.sum(diff ** 2) / (np.sum(mask_i) * S + 1e-8)

        loss /= N

        # L2 Regularization
        l2_penalty = 0.5 * self.l2_reg * np.sum(w_vec ** 2)
        total_loss = loss + l2_penalty

        # Approximate gradient via finite differences for stability
        eps = 1e-5
        grad = np.zeros_like(w_vec)
        for k in range(0, len(w_vec), max(1, len(w_vec) // 50)):
            w_vec_plus = w_vec.copy()
            w_vec_plus[k] += eps
            w_plus = self._unflatten_weights(w_vec_plus, S, H)

            l_plus = 0.0
            for i in range(N):
                z_i_p = self._encode_subject(data[i], masks[i], w_plus)
                rec_p = self._decode_subject(z_i_p, w_plus, S, H)
                l_plus += (
                    np.sum(((data[i] - rec_p) * masks[i][:, None]) ** 2)
                    / (np.sum(masks[i]) * S + 1e-8))
            l_plus = l_plus / N + 0.5 * self.l2_reg * np.sum(w_vec_plus ** 2)

            grad[k] = (l_plus - total_loss) / eps

        return total_loss, grad

    # =========================================================================
    # Helpers
    # =========================================================================

    def _prepare_3d_tensor(self, dataset) -> tuple[np.ndarray, np.ndarray]:
        N = len(dataset.subjects)
        S = dataset.subjects[0].n_scales
        tensor = np.zeros((N, 24, S), dtype=float)
        masks = np.zeros((N, 24), dtype=float)

        for i, subj in enumerate(dataset.subjects):
            for h, c in zip(subj.hours, subj.curves):
                tensor[i, h] = c
                masks[i, h] = 1.0

        return tensor, masks

    def _init_weights(self, S: int, H: int) -> dict:
        rng = np.random.default_rng(42)
        hd = self.hidden_dim
        ld = self.latent_dim

        return {
            "W_spatial": rng.standard_normal((S, hd)) * np.sqrt(2.0 / S),
            "b_spatial": np.zeros(hd),
            "W_bottleneck": (
                rng.standard_normal((H * hd, ld)) * np.sqrt(2.0 / (H * hd))
            ),
            "b_bottleneck": np.zeros(ld),
            "W_dec_temp": (
                rng.standard_normal((ld, H * hd)) * np.sqrt(2.0 / ld)
            ),
            "b_dec_temp": np.zeros(H * hd),
            "W_dec_spatial": rng.standard_normal((hd, S)) * np.sqrt(2.0 / hd),
            "b_dec_spatial": np.zeros(S),
        }

    def _flatten_weights(self, w: dict) -> np.ndarray:
        return np.hstack([arr.ravel() for arr in w.values()])

    def _unflatten_weights(self, w_vec: np.ndarray, S: int, H: int) -> dict:
        hd = self.hidden_dim
        ld = self.latent_dim
        shapes = [
            ("W_spatial", (S, hd)),
            ("b_spatial", (hd,)),
            ("W_bottleneck", (H * hd, ld)),
            ("b_bottleneck", (ld,)),
            ("W_dec_temp", (ld, H * hd)),
            ("b_dec_temp", (H * hd,)),
            ("W_dec_spatial", (hd, S)),
            ("b_dec_spatial", (S,)),
        ]

        unflattened = {}
        curr = 0
        for name, shape in shapes:
            size = int(np.prod(shape))
            unflattened[name] = w_vec[curr: curr + size].reshape(shape)
            curr += size

        return unflattened

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("FunctionalAutoencoder is not fitted yet.")
