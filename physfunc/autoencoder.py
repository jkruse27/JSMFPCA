import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ==========================================
# 1. DATA AUGMENTATIONS & DATASET
# ==========================================

class CurveAugmentation1D:
    def __init__(self, shift_max=3, scale_range=(0.95, 1.05),
                 noise_std=0.01, mask_ratio=0.1):
        self.shift_max = shift_max
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio

    def __call__(self, x):
        # x shape: (channels, sequence_length)
        c, length = x.shape
        x_aug = x.clone()

        # 1. Amplitude scaling
        scale_factor = torch.empty(c, 1).uniform_(*self.scale_range)
        x_aug = x_aug * scale_factor

        # 2. Additive Noise
        noise = torch.randn_like(x_aug) * self.noise_std
        x_aug = x_aug + noise

        # 3. Random Scale Shift (Translation along log-scale axis)
        shift = np.random.randint(-self.shift_max, self.shift_max + 1)
        if shift != 0:
            x_aug = torch.roll(x_aug, shifts=shift, dims=-1)

        # 4. Random Scale Masking / Cutout
        mask_len = int(length * self.mask_ratio)
        if mask_len > 0:
            start_idx = np.random.randint(0, max(1, length - mask_len))
            x_aug[:, start_idx:start_idx + mask_len] = 0.0

        return x_aug


class ScalingCurveDataset(Dataset):
    def __init__(self, df_curves_dict, df_features=None, target_col='BNP',
                 transform=None, is_unsupervised=True):
        """
        df_curves_dict: dict containing DataFrames for 'lmds' and 'alphas'
        df_features: DataFrame containing clinical/BNP targets
        """
        self.transform = transform
        self.is_unsupervised = is_unsupervised

        # Align subjects and extract arrays
        df_lmds = df_curves_dict['lmds']
        df_alphas = df_curves_dict.get('alphas', None)

        if not is_unsupervised and df_features is not None:
            # Filter for subjects with valid target (e.g., BNP)
            valid_targets = df_features[[target_col]].dropna()
            common_idx = df_lmds.index.intersection(valid_targets.index)
            df_lmds = df_lmds.loc[common_idx]
            if df_alphas is not None:
                df_alphas = df_alphas.loc[common_idx]
            self.targets = np.log10(
                valid_targets.loc[common_idx, target_col].values
            )
        else:
            self.targets = None

        self.scales = df_lmds.columns.astype(float).values
        self.lmds_data = df_lmds.values.astype(np.float32)

        if df_alphas is not None:
            self.alphas_data = df_alphas.values.astype(np.float32)
            # Combine channels: Shape (N, 2, Scales)
            self.data = np.stack([self.lmds_data, self.alphas_data], axis=1)
        else:
            # Single channel: Shape (N, 1, Scales)
            self.data = np.expand_dims(self.lmds_data, axis=1)

        # Normalize across curves (mean=0, std=1 per channel)
        self.mean = np.nanmean(self.data, axis=(0, 2), keepdims=True)
        self.std = np.nanstd(self.data, axis=(0, 2), keepdims=True) + 1e-6
        self.data = np.nan_to_num((self.data - self.mean) / self.std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])

        if self.transform:
            x_aug = self.transform(x)
        else:
            x_aug = x

        if self.is_unsupervised:
            return x_aug, x  # Input and reconstruction target
        else:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, y


# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================

class ConvEncoder1D(nn.Module):
    def __init__(self, in_channels=2, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        feat_map = F.leaky_relu(self.bn3(self.conv3(x)))
        return feat_map


class ConvDecoder1D(nn.Module):
    def __init__(self, out_channels=2, original_len=100):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(
            64, 32, kernel_size=5, padding=2, stride=2, output_padding=1
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.deconv2 = nn.ConvTranspose1d(
            32, 16, kernel_size=5, padding=2, stride=2, output_padding=1
        )
        self.bn2 = nn.BatchNorm1d(16)
        self.conv_out = nn.Conv1d(16, out_channels, kernel_size=5, padding=2)
        self.original_len = original_len

    def forward(self, feat_map):
        x = F.leaky_relu(self.bn1(self.deconv1(feat_map)))
        x = F.leaky_relu(self.bn2(self.deconv2(x)))
        out = self.conv_out(x)
        return out[:, :, :self.original_len]


class ConvAutoencoder1D(nn.Module):
    def __init__(self, in_channels=2, original_len=100):
        super().__init__()
        self.encoder = ConvEncoder1D(in_channels=in_channels)
        self.decoder = ConvDecoder1D(
            out_channels=in_channels, original_len=original_len
        )

    def forward(self, x):
        feat_map = self.encoder(x)
        rec = self.decoder(feat_map)
        return rec


class ScaleAttention1D(nn.Module):
    """ Attention block to weight scale importance """
    def __init__(self, in_features):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv1d(in_features, 16, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        weights = self.att(x)
        context = torch.sum(x * weights, dim=-1)
        return context, weights


class BNPRegressor1D(nn.Module):
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
# 3. INTERPRETABILITY & GRAD-CAM 1D
# ==========================================

class GradCAM1D:
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

    def __call__(self, x):
        self.model.zero_grad()
        output, _ = self.model(x)
        output.backward(torch.ones_like(output))

        grads = self.gradients[0]
        acts = self.activations[0]

        weights = torch.mean(grads, dim=-1, keepdim=True)
        cam = torch.sum(weights * acts, dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Interpolate CAM back to original curve length
        cam_upsampled = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=x.shape[-1],
            mode='linear',
            align_corners=False
        ).squeeze()

        return cam_upsampled.detach().cpu().numpy()


def train_pipeline(df_curves_lmds, df_curves_alphas, df_features, scales):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = df_curves_lmds.shape[1]

    curves_dict = {'lmds': df_curves_lmds, 'alphas': df_curves_alphas}

    # ----------------------------------------------------
    # Stage 1: Unsupervised Pre-training (Autoencoder)
    # ----------------------------------------------------
    aug = CurveAugmentation1D()
    unsup_dataset = ScalingCurveDataset(
        curves_dict, transform=aug, is_unsupervised=True
    )
    unsup_loader = DataLoader(unsup_dataset, batch_size=32, shuffle=True)

    autoencoder = ConvAutoencoder1D(
        in_channels=2, original_len=seq_len
    ).to(device)
    optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion_ae = nn.MSELoss()

    print("=== Stage 1: Training Unsupervised 1D Conv Autoencoder ===")
    autoencoder.train()
    for epoch in range(25):
        total_loss = 0
        for x_aug, x_orig in unsup_loader:
            x_aug, x_orig = x_aug.to(device), x_orig.to(device)
            optimizer_ae.zero_grad()
            rec = autoencoder(x_aug)
            loss = criterion_ae(rec, x_orig)
            loss.backward()
            optimizer_ae.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(
             f"Epoch [{epoch+1}/25], Loss: {total_loss/len(unsup_loader):.4f}"
            )

    # ----------------------------------------------------
    # Stage 2: Supervised Fine-Tuning on BNP
    # ----------------------------------------------------
    sup_dataset = ScalingCurveDataset(
        curves_dict, df_features=df_features,
        target_col='BNP', is_unsupervised=False
    )
    sup_loader = DataLoader(sup_dataset, batch_size=16, shuffle=True)

    bnp_model = BNPRegressor1D(encoder=autoencoder.encoder).to(device)
    optimizer_bnp = torch.optim.Adam(bnp_model.parameters(), lr=5e-4)
    criterion_bnp = nn.MSELoss()

    print("\n=== Stage 2: Fine-Tuning BNP Regressor ===")
    bnp_model.train()
    for epoch in range(30):
        total_loss = 0
        for x, y in sup_loader:
            x, y = x.to(device), y.to(device)
            optimizer_bnp.zero_grad()
            pred, _ = bnp_model(x)
            loss = criterion_bnp(pred, y)
            loss.backward()
            optimizer_bnp.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/30], MSE Loss: "
                  f"{total_loss/len(sup_loader):.4f}")

    # ----------------------------------------------------
    # Stage 3: Interpretability Visualization
    # ----------------------------------------------------
    print("\n=== Stage 3: Extracting Interpretability Heatmaps ===")
    bnp_model.eval()
    grad_cam = GradCAM1D(bnp_model, target_layer=bnp_model.encoder.conv3)

    sample_x, sample_y = sup_dataset[0]
    sample_x_tensor = sample_x.unsqueeze(0).to(device)

    # Generate Grad-CAM heatmaps across scales
    heatmap = grad_cam(sample_x_tensor)

    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Plot raw curve (Non-Gaussianity lambda^2_s)
    ax1.plot(scales, sample_x[0].numpy(),
             label=r"$\lambda^2_s$ Profile", color='black', lw=2)
    ax1.set_xlabel(r"Scale ($\log_{10}$)")
    ax1.set_ylabel(r"Normalized $\lambda^2_s$", color='black')

    # Overlay Grad-CAM scale importance
    ax2 = ax1.twinx()
    ax2.fill_between(scales, 0, heatmap, color='red',
                     alpha=0.3, label="Importance Heatmap")
    ax2.set_ylabel("Grad-CAM Importance", color='red')

    plt.title(f"Curve Region Importance (True Log BNP: {sample_y.item():.2f})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return bnp_model
