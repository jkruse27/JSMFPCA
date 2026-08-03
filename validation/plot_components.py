# File: validation/plot_components.py

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from physfunc.data import JSMFPCAData
from physfunc.jsmfpca import JSMFPCA
from physfunc.mfpca import TraditionalMFPCA
from definitions import load_jsmfpca_dataset

# Publication plot styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_inter_subject_components(dataset: JSMFPCAData, out_dir: str):
    scales = dataset.scales
    scales_plot = 10**scales if np.all(scales <= 5) else scales

    # Fit JS-MFPCA and Traditional MFPCA
    js_model = JSMFPCA(n_modes=3, n_harmonics=2, shrinkage=0.25).fit(dataset)
    mfpca_model = TraditionalMFPCA(explained_variance=0.95).fit(dataset)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- Panel A: Population Mean Curve & Shape Eigenfunctions ---
    ax = axes[0, 0]
    ax.plot(scales_plot, js_model.mean_curve_, color="black",
            lw=2.5, label=r"Population Mean $\mu(s)$")
    ax.set_xscale("log" if np.all(scales <= 5) else "linear")
    ax.set_xlabel("Scale s (seconds)")
    ax.set_ylabel(r"Value")
    ax.set_title("A. Population Mean Curve")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)

    # --- Panel B: Stage 0 Shape Modes (+/- 1.5 SD Effect) ---
    ax = axes[1, 0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for m in range(min(3, js_model.n_modes_)):
        ef = js_model.shape_basis_[m]
        ax.plot(scales_plot, ef, color=colors[m], lw=2,
                label=f"Mode {m+1} ({js_model.shape_eigenvalues_[m]:.2f})")

    ax.set_xscale("log" if np.all(scales <= 5) else "linear")
    ax.set_xlabel("Scale s (seconds)")
    ax.set_ylabel("Eigenfunction Value")
    ax.set_title(r"B. JS-MFPCA Shape Eigenfunctions $\phi_m(s)$")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)

    # --- Panel C: Traditional MFPCA Level 1 vs Level 2 Eigenfunctions ---
    ax = axes[0, 1]
    if mfpca_model.phi_ is not None and len(mfpca_model.phi_) > 0:
        ax.plot(scales_plot, mfpca_model.phi_[0], color="#d62728",
                lw=2, label=r"Level 1 (Between) $\phi_1(s)$")
    if mfpca_model.psi_ is not None and len(mfpca_model.psi_) > 0:
        ax.plot(scales_plot, mfpca_model.psi_[0], color="#9467bd",
                lw=2, linestyle="--", label=r"Level 2 (Within) $\psi_1(s)$")

    ax.set_xscale("log" if np.all(scales <= 5) else "linear")
    ax.set_xlabel("Scale s (seconds)")
    ax.set_ylabel("Eigenfunction Value")
    ax.set_title(
        r"C. MFPCA Level 1 ($\phi_k$) vs Level 2 ($\psi_l$) Eigenfunctions"
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)

    # --- Panel D: Inter-Subject Static Baseline Offsets ---
    ax = axes[1, 1]
    fingerprints = js_model.transform(dataset)
    for m in range(min(3, js_model.n_modes_)):
        offsets_m = fingerprints[:, m]
        ax.hist(offsets_m, bins=15, alpha=0.5,
                color=colors[m], label=f"Mode {m+1} Offset")

    ax.set_xlabel(r"Static Baseline Offset $\bar{\eta}_{i,m}$")
    ax.set_ylabel("Subject Count")
    ax.set_title(
        r"D. Inter-Subject Baseline Offset Distributions $\bar{\eta}_{i,m}$"
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)

    plt.tight_layout()
    file_path = os.path.join(out_dir, "inter_subject_components.pdf")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Inter-subject component plot saved to: {file_path}")


def plot_intra_subject_components(dataset: JSMFPCAData, out_dir: str):
    scales = dataset.scales
    scales_plot = 10**scales if np.all(scales <= 5) else scales

    js_model = JSMFPCA(n_modes=3, n_harmonics=2, shrinkage=0.25).fit(dataset)
    reconstructed = js_model.reconstruct(dataset)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- Panel A: 24-Hour Temporal Score Evolution ---
    ax = axes[0, 0]
    for subj in dataset.subjects[:10]:
        scores = (subj.curves - js_model.mean_curve_) @ js_model.shape_basis_.T
        ax.plot(subj.hours, scores[:, 0], marker="o", ms=4, alpha=0.6)

    ax.set_xlabel("Hour of Day (0–23h)")
    ax.set_ylabel(r"Shape Score $X_{i,1}(h)$")
    ax.set_title("A. 24-Hour Temporal Evolution of Shape Scores")
    ax.set_xticks(range(0, 25, 4))
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- Panel B: Reconstructed Functional Curves for Subject 0 ---
    ax = axes[1, 0]
    subj_0 = dataset.subjects[0]
    rec_0 = reconstructed[0]

    target_hours = [0, 4, 8, 12, 16, 20]
    cmap = plt.colormaps["plasma"].resampled(len(target_hours))

    for idx, h in enumerate(target_hours):
        if h in subj_0.hours:
            h_idx = np.where(subj_0.hours == h)[0][0]
            ax.plot(scales_plot, subj_0.curves[h_idx],
                    linestyle="--", alpha=0.5, color=cmap(idx))
            ax.plot(scales_plot, rec_0[h_idx], lw=2,
                    color=cmap(idx), label=f"{h}:00h")

    ax.set_xscale("log" if np.all(scales <= 5) else "linear")
    ax.set_xlabel("Scale s (seconds)")
    ax.set_ylabel("Value")
    ax.set_title("B. Reconstructed Curves at Representative Hours (Subj 0)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False, title="Hour")

    # --- Panel C: Coordinated Cross-Spectral Harmonic Modes U_r ---
    ax = axes[0, 1]
    U_1 = js_model.spectral_eigenvectors_[0]  # Harmonic 1 eigenvectors (M, M)

    # Plot Real and Imaginary components of top mode U_{1,1}
    modes_idx = np.arange(1, js_model.n_modes_ + 1)
    ax.bar(modes_idx - 0.15, np.real(U_1[:, 0]),
           width=0.3, label="Re(U_1)", color="#1f77b4")
    ax.bar(modes_idx + 0.15, np.imag(U_1[:, 0]),
           width=0.3, label="Im(U_1)", color="#ff7f0e")

    ax.set_xlabel("Shape Mode Dimension (m)")
    ax.set_ylabel("Coordinated Eigenvector Value")
    ax.set_title(r"C. Stage 2 Coordinated Harmonic Modes $U_1$ (Harmonic 1)")
    ax.set_xticks(modes_idx)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)

    # --- Panel D: Circadian Amplitude vs Acrophase Phase Plot ---
    ax = axes[1, 1]
    fingerprints = js_model.transform(dataset)

    # Fingerprint contains offsets (M) then [Amp, Phase] for each harmonic
    M = js_model.n_modes_
    amp_1 = fingerprints[:, M]       # Harmonic 1 Amplitude
    phase_1 = fingerprints[:, M + 1]   # Harmonic 1 Phase (radians)

    scatter = ax.scatter(
        phase_1, amp_1, c=amp_1, cmap="viridis", edgecolors="k", alpha=0.8
    )
    ax.set_xlabel(r"Circadian Phase / Acrophase $\phi_i$ (radians)")
    ax.set_ylabel(r"Circadian Amplitude $A_i$")
    ax.set_title("D. Circadian Amplitude vs. Phase Distribution (Harmonic 1)")
    fig.colorbar(scatter, ax=ax, label="Amplitude")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    file_path = os.path.join(out_dir, "intra_subject_components.pdf")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Intra-subject component plot saved to: {file_path}")


def run_component_analysis():
    dataset_name = "986"
    window_size = 2
    feat = "lmds"

    print("=" * 60)
    print("Starting Component Analysis & Visualization")
    print("=" * 60)

    # Load dataset
    dataset = load_jsmfpca_dataset(
        window_size=window_size, feat=feat, dataset_name=dataset_name
    )

    # Output figures directory
    out_dir = os.path.join(
        "data", "results", dataset_name, "figures"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Generate Inter-Subject and Intra-Subject component figures
    print("\n1. Generating Inter-Subject Component Figures...")
    plot_inter_subject_components(dataset, out_dir)

    print("\n2. Generating Intra-Subject Component Figures...")
    plot_intra_subject_components(dataset, out_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    run_component_analysis()
