import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# Add Commons to path as in original script
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "0_Commons",
    )
)
from constants import RESULTS, WINDOWS  # noqa: E402
from definitions import get_curves, get_features  # noqa: E402

warnings.filterwarnings("ignore")

# ==========================================
# 1. DATA AUGMENTATION & DATASET CLASSES
# ==========================================

class CurveAugmentation1D:
    """Data augmentation tailored for 1D scaling curves."""
    def __init__(self, shift_max=3, scale_range=(0.95, 1.05), noise_std=0.01, mask_ratio=0.1):
        self.shift_max = shift_max
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio

    def __call__(self, x):
        # x shape: (channels, seq_length)
        c, l = x.shape
        x_aug = x.clone()

        # 1. Amplitude scaling
        scale_factor = torch.empty(c, 1).uniform_(*self.scale_range)
        x_aug = x_aug * scale_factor

        # 2. Additive Gaussian noise
        noise = torch.randn_like(x_aug) * self.noise_std
        x_aug = x_aug + noise

        # 3. Scale shift (translation along scale axis)
        shift = np.random.randint(-self.shift_max, self.shift_max + 1)
        if shift != 0:
            x_aug = torch.roll(x_aug, shifts=shift, dims=-1)

        # 4. Masking / Cutout along scale axis
        mask_len = int(l * self.mask_ratio)
        if mask_len > 0:
            start_idx = np.random.randint(0, max(1, l - mask_len))
            x_aug[:, start_idx:start_idx + mask_len] = 0.0

        return x_aug


class ArrayDataset(Dataset):
    """Generic Dataset for PyTorch Tensors."""
    def __init__(self, x_data, y_data=None, transform=None):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.float32) if y_data is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        x_in = self.transform(x) if self.transform else x
        if self.y_data is not None:
            return x_in, self.y_data[idx]
        return x_in, x  # Return augmented and original for autoencoder reconstruction


# ==========================================
# 2. DEEP LEARNING ARCHITECTURES
# ==========================================

class ConvEncoder1D(nn.Module):
    """1D Convolutional Encoder for Scale Shape Extraction."""
    def __init__(self, in_channels=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        feat_map = F.leaky_relu(self.bn3(self.conv3(x)))
        return feat_map


class ConvDecoder1D(nn.Module):
    """1D Transposed Convolutional Decoder for Curve Reconstruction."""
    def __init__(self, out_channels=2, original_len=100):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(64, 32, kernel_size=5, padding=2, stride=2, output_padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.deconv2 = nn.ConvTranspose1d(32, 16, kernel_size=5, padding=2, stride=2, output_padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv_out = nn.Conv1d(16, out_channels, kernel_size=5, padding=2)
        self.original_len = original_len

    def forward(self, feat_map):
        x = F.leaky_relu(self.bn1(self.deconv1(feat_map)))
        x = F.leaky_relu(self.bn2(self.deconv2(x)))
        out = self.conv_out(x)
        return out[:, :, :self.original_len]


class ConvAutoencoder1D(nn.Module):
    """Self-Supervised Autoencoder."""
    def __init__(self, in_channels=2, original_len=100):
        super().__init__()
        self.encoder = ConvEncoder1D(in_channels=in_channels)
        self.decoder = ConvDecoder1D(out_channels=in_channels, original_len=original_len)

    def forward(self, x):
        feat_map = self.encoder(x)
        rec = self.decoder(feat_map)
        return rec


class ScaleAttention1D(nn.Module):
    """Attention Mechanism to highlight key scale regions."""
    def __init__(self, in_features=64):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv1d(in_features, 16, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        weights = self.att(x)  # (B, 1, Downsampled_L)
        context = torch.sum(x * weights, dim=-1) # (B, Channels)
        return context, weights


class BNPRegressor1D(nn.Module):
    """Downstream Supervised BNP Predictor."""
    def __init__(self, encoder, hidden_dim=32):
        super().__init__()
        self.encoder = encoder
        self.attention = ScaleAttention1D(in_features=64)
        self.head = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        feat_map = self.encoder(x)
        context, att_weights = self.attention(feat_map)
        out = self.head(context)
        return out.squeeze(-1), att_weights


# ==========================================
# 3. GRAD-CAM INTERPRETABILITY
# ==========================================

class GradCAM1D:
    """Grad-CAM for 1D Scaling Curves."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x_tensor):
        self.model.eval()
        self.model.zero_grad()
        
        pred, _ = self.model(x_tensor)
        pred.sum().backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations

        weights = torch.mean(grads, dim=-1, keepdim=True)
        cam = torch.sum(weights * acts, dim=1)
        cam = F.relu(cam)
        cam = cam - cam.min(dim=-1, keepdim=True)[0]
        cam = cam / (cam.max(dim=-1, keepdim=True)[0] + 1e-8)

        # Upsample back to original length
        cam_upsampled = F.interpolate(
            cam.unsqueeze(1),
            size=x_tensor.shape[-1],
            mode='linear',
            align_corners=False
        ).squeeze(1)

        return cam_upsampled.detach().cpu().numpy()


# ==========================================
# 4. DATA PROCESSING HELPERS
# ==========================================

def load_all_unsupervised_curves(window_size, datasets=['986', '108', 'Healthy']):
    """Combines curves across all datasets, hours, and features for Autoencoder training."""
    all_samples = []
    
    for ds in datasets:
        for hour in range(24):
            try:
                df_lmds = get_curves(window_size, hour, 'lmds', ds)
                df_alphas = get_curves(window_size, hour, 'alphas', ds)
            except Exception:
                continue

            if df_lmds.empty or df_alphas.empty:
                continue

            common_idx = df_lmds.index.intersection(df_alphas.index)
            if len(common_idx) == 0:
                continue

            lmds_arr = df_lmds.loc[common_idx].values.astype(np.float32)
            alphas_arr = df_alphas.loc[common_idx].values.astype(np.float32)

            # Filter rows with NaNs
            valid_mask = ~np.isnan(lmds_arr).any(axis=1) & ~np.isnan(alphas_arr).any(axis=1)
            if not np.any(valid_mask):
                continue

            sample_tensor = np.stack([lmds_arr[valid_mask], alphas_arr[valid_mask]], axis=1)
            all_samples.append(sample_tensor)

    if not all_samples:
        return None

    X_unsup = np.concatenate(all_samples, axis=0) # (N_total, 2, Scales)
    
    # Global standardization per channel
    mean = np.mean(X_unsup, axis=(0, 2), keepdims=True)
    std = np.std(X_unsup, axis=(0, 2), keepdims=True) + 1e-6
    X_unsup = (X_unsup - mean) / std

    return X_unsup, mean, std


def load_supervised_hour_data(window_size, hour, dataset='986'):
    """Loads paired curves and BNP for dataset '986' for a given hour."""
    df_features = get_features(window_size, dataset)

    if 'BNP' not in df_features.columns:
        return None, None, None, None

    df_bnp = df_features[['BNP']].groupby(df_features.index).first().dropna()
    df_bnp = df_bnp[df_bnp['BNP'] > 0] # Ensure valid positive BNP

    if len(df_bnp) < 10:
        return None, None, None, None

    df_lmds = get_curves(window_size, hour, 'lmds', dataset)
    df_alphas = get_curves(window_size, hour, 'alphas', dataset)

    if df_lmds.empty or df_alphas.empty:
        return None, None, None, None

    scales = df_lmds.columns.astype(float).values
    common_idx = df_lmds.index.intersection(df_alphas.index).intersection(df_bnp.index)

    if len(common_idx) < 10:
        return None, None, None, None

    lmds_arr = df_lmds.loc[common_idx].values.astype(np.float32)
    alphas_arr = df_alphas.loc[common_idx].values.astype(np.float32)
    bnp_arr = df_bnp.loc[common_idx, 'BNP'].values.astype(np.float32)

    valid_mask = ~np.isnan(lmds_arr).any(axis=1) & ~np.isnan(alphas_arr).any(axis=1)
    
    if np.sum(valid_mask) < 10:
        return None, None, None, None

    X = np.stack([lmds_arr[valid_mask], alphas_arr[valid_mask]], axis=1) # (N, 2, Scales)
    y_log_bnp = np.log10(bnp_arr[valid_mask]) # Continuous target: log10(BNP)
    subject_ids = common_idx[valid_mask].values

    return X, y_log_bnp, scales, subject_ids


# ==========================================
# 5. MAIN TRAINING & EVALUATION PIPELINE
# ==========================================

def run_experiment(window_sizes=WINDOWS, hours_to_eval=[0, 4, 8, 12, 16, 20]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    for ws in window_sizes:
        print(f"\n=======================================================")
        print(f"       PROCESSING WINDOW SIZE: {ws}")
        print(f"=======================================================")

        output_dir = f"{RESULTS}/986/deep_learning_results/{ws}"
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)
        os.makedirs(f"{output_dir}/metrics", exist_ok=True)

        # ---------------------------------------------------------
        # Step A: Unsupervised Pre-training across ALL datasets
        # ---------------------------------------------------------
        print("\n--> Step A: Pre-training Unsupervised Autoencoder across '986', '108', 'Healthy'...")
        X_unsup, unsup_mean, unsup_std = load_all_unsupervised_curves(ws, ['986', '108', 'Healthy'])

        if X_unsup is None:
            print(f"Skipping Window Size {ws}: Insufficient curve data.")
            continue

        seq_len = X_unsup.shape[-1]
        aug = CurveAugmentation1D()
        unsup_ds = ArrayDataset(X_unsup, transform=aug)
        unsup_loader = DataLoader(unsup_ds, batch_size=64, shuffle=True)

        autoencoder = ConvAutoencoder1D(in_channels=2, original_len=seq_len).to(device)
        optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion_ae = nn.MSELoss()

        autoencoder.train()
        for epoch in range(20):
            running_loss = 0.0
            for x_aug, x_orig in unsup_loader:
                x_aug, x_orig = x_aug.to(device), x_orig.to(device)
                optimizer_ae.zero_grad()
                rec = autoencoder(x_aug)
                loss = criterion_ae(rec, x_orig)
                loss.backward()
                optimizer_ae.step()
                running_loss += loss.item()

        print(f"    Autoencoder pre-training complete. Final MSE: {running_loss/len(unsup_loader):.4f}")
        torch.save(autoencoder.state_dict(), f"{output_dir}/models/autoencoder_pretrained.pt")

        # ---------------------------------------------------------
        # Step B: Supervised 5-Fold CV x 10 Repetitions on '986'
        # ---------------------------------------------------------
        for hour in hours_to_eval:
            print(f"\n--> Step B: Running 5-Fold CV x 10 Reps for Hour {hour}:00h on dataset '986'...")
            X_sup, y_sup, scales, subj_ids = load_supervised_hour_data(ws, hour, dataset='986')

            if X_sup is None:
                print(f"    Skipping Hour {hour}: Insufficient BNP paired data.")
                continue

            # Standardize supervised inputs using global unsupervised parameters
            X_sup_norm = (X_sup - unsup_mean) / unsup_std

            rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

            fold_metrics = []
            all_cams = []
            all_preds = []
            all_targets = []

            for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X_sup_norm)):
                X_train, y_train = X_sup_norm[train_idx], y_sup[train_idx]
                X_test, y_test = X_sup_norm[test_idx], y_sup[test_idx]

                train_ds = ArrayDataset(X_train, y_train, transform=CurveAugmentation1D())
                test_ds = ArrayDataset(X_test, y_test)

                train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
                test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

                # Initialize model with pre-trained encoder weights
                encoder_clone = ConvEncoder1D(in_channels=2)
                encoder_clone.load_state_dict(autoencoder.encoder.state_dict())

                model = BNPRegressor1D(encoder=encoder_clone).to(device)
                optimizer_bnp = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
                criterion_bnp = nn.MSELoss()

                # Fine-tuning
                model.train()
                for epoch in range(25):
                    for bx, by in train_loader:
                        bx, by = bx.to(device), by.to(device)
                        optimizer_bnp.zero_grad()
                        pred, _ = model(bx)
                        loss = criterion_bnp(pred, by)
                        loss.backward()
                        optimizer_bnp.step()

                # Evaluation
                model.eval()
                test_preds = []
                test_targets = []
                with torch.no_grad():
                    for bx, by in test_loader:
                        bx = bx.to(device)
                        pred, _ = model(bx)
                        test_preds.extend(pred.cpu().numpy())
                        test_targets.extend(by.numpy())

                test_preds = np.array(test_preds)
                test_targets = np.array(test_targets)

                mse = mean_squared_error(test_targets, test_preds)
                r2 = r2_score(test_targets, test_preds)
                r_val, _ = pearsonr(test_targets, test_preds) if len(test_preds) > 1 else (0, 0)
                rho_val, _ = spearmanr(test_targets, test_preds) if len(test_preds) > 1 else (0, 0)

                fold_metrics.append({
                    'rep_fold': fold_idx,
                    'mse': mse,
                    'r2': r2,
                    'pearson_r': r_val,
                    'spearman_rho': rho_val
                })

                all_preds.extend(test_preds)
                all_targets.extend(test_targets)

                # Compute Grad-CAM on test fold samples
                grad_cam = GradCAM1D(model, target_layer=model.encoder.conv3)
                test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                cams = grad_cam(test_tensor) # (N_test, Scales)
                all_cams.append(cams)

            # Aggregate Cross-Validation Results
            df_metrics = pd.DataFrame(fold_metrics)
            df_metrics.to_csv(f"{output_dir}/metrics/cv_metrics_hour_{hour}.csv", index=False)

            print(f"    Hour {hour}:00h Summary (50 Folds):")
            print(f"      MSE:         {df_metrics['mse'].mean():.4f} +/- {df_metrics['mse'].std():.4f}")
            print(f"      R2:          {df_metrics['r2'].mean():.4f} +/- {df_metrics['r2'].std():.4f}")
            print(f"      Pearson r:   {df_metrics['pearson_r'].mean():.4f} +/- {df_metrics['pearson_r'].std():.4f}")
            print(f"      Spearman rho:{df_metrics['spearman_rho'].mean():.4f} +/- {df_metrics['spearman_rho'].std():.4f}")

            # ---------------------------------------------------------
            # Step C: Plot & Save Interpretability Heatmap
            # ---------------------------------------------------------
            avg_cam = np.mean(np.concatenate(all_cams, axis=0), axis=0)
            avg_cam = (avg_cam - avg_cam.min()) / (avg_cam.max() - avg_cam.min() + 1e-8)

            mean_lmds = np.mean(X_sup[:, 0, :], axis=0)

            fig, ax1 = plt.subplots(figsize=(8, 4.5))

            # Plot Mean Curve Shape
            ax1.plot(scales, mean_lmds, color='black', lw=2, label=r"Mean $\lambda^2_s$ Profile")
            ax1.set_xlabel(r"Scale ($\log_{10}$)")
            ax1.set_ylabel(r"Mean $\lambda^2_s$", color='black')
            ax1.grid(True, linestyle='--', alpha=0.5)

            # Overlay Grad-CAM Importance Heatmap
            ax2 = ax1.twinx()
            ax2.fill_between(scales, 0, avg_cam, color='red', alpha=0.3, label="Scale Importance (Grad-CAM)")
            ax2.set_ylabel("Normalized Scale Importance", color='red')

            plt.title(f"BNP Association & Important Scale Regions - {hour}:00h (Window Size: {ws})")

            # Save figure
            fig_path = f"{output_dir}/figures/bnp_scale_importance_hour_{hour}.pdf"
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"    Saved interpretability plot to: {fig_path}")

if __name__ == "__main__":
    HOURS = [0, 4, 8, 12, 16, 20]
    run_experiment(window_sizes=WINDOWS, hours_to_eval=HOURS)
