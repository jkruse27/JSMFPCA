# Joint Spectral Multilevel Functional Principal Component Analysis (JS-MFPCA)

A hierarchical functional data analysis framework for discovering coordinated circadian dynamics from multivariate longitudinal physiological time series.

---

## Table of Contents

1. [Overview & Scientific Rationale](#overview--scientific-rationale)
2. [Software Architecture & Package Layout](#software-architecture--package-layout)
3. [Mathematical Foundations](#mathematical-foundations)
   - [Data Representation](#1-data-representation)
   - [Stage 0: Shape FPCA](#2-stage-0--shape-fpca)
   - [Stage 1: Circadian Baseline Centering](#3-stage-1--circadian-baseline-centering)
   - [Stage 2: Joint Cross-Spectral Covariance Modeling](#4-stage-2--joint-cross-spectral-covariance-modeling)
   - [Stage 3: Real Gaussian BLUP / PACE Estimation](#5-stage-3--real-gaussian-blup--pace-estimation)
   - [Stage 4: Physiological Fingerprints](#6-stage-4--physiological-fingerprints)
4. [Baseline Models](#baseline-models)
   - [Traditional MFPCA](#1-traditional-mfpca)
   - [Diagonal / Mode-Independent Spectral Model](#2-diagonal--mode-independent-spectral-model)
   - [OLS Harmonic Estimator](#3-ols-harmonic-estimator)
   - [Classical Cosinor Model](#4-classical-cosinor-model)
5. [Validation & Benchmark Framework](#validation--benchmark-framework)
6. [Quickstart & Usage Examples](#quickstart--usage-examples)

---

## Overview & Scientific Rationale

High-dimensional physiological signals—such as 24-hour Heart Rate Variability (HRV), ECG-derived Respiration (EDR), and multiscale non-Gaussian autonomic markers—exhibit complex circadian fluctuations. Traditional analytical approaches often treat these dynamics either as static average summary statistics or as isolated univariate sinusoids (cosinor analysis).

**JS-MFPCA** is built on the central hypothesis that:

> **Physiological phenotypes are defined not merely by average values or isolated circadian amplitudes, but by coordinated multiscale temporal organization across individuals.**

JS-MFPCA decomposes multivariate longitudinal functional data into hierarchical components:

* **Global Waveform Morphology**: Population-level curve shapes ($\phi_m(s)$).
* **Subject-Specific Baseline Offsets**: Static intensity shifts ($\bar{\eta}_{i,m}$).
* **Cross-Variable Harmonic Coordination**: Joint co-oscillations across different shape modes ($U_r, \Lambda_r$).
* **Physiological Fingerprints**: Compact, interpretable subject vectors ($z_i$) for phenotype discovery, classification, and prognosis.

---

## Software Architecture & Package Layout

The `physfunc` package uses a single-file, self-contained estimator design following the Scikit-Learn API guidelines (`fit`, `transform`, `fit_transform`, `reconstruct`).

```text
physfunc/
├── __init__.py                # Package exports (JSMFPCA, baselines, data containers)
├── jsmfpca.py                 # Core JS-MFPCA unified estimator
├── data.py                    # Data containers (JSMFPCAData, SubjectCurves)
├── mfpca.py                   # Traditional MFPCA (Di et al., 2009)
├── diagonal.py                # Diagonal / Mode-Independent Spectral Model
├── ols.py                     # OLS Harmonic Estimator
├── cosinor.py                 # Classical Cosinor Model

```

---

## Mathematical Foundations

```text
Raw Functional Observations {Y_i(h, s)}
                │
                ▼
   Stage 0: Shape FPCA (Extract Waveform Modes φ_m(s))
                │
                ▼
   Stage 1: Circadian Baseline Centering (Extract Offsets η̄_{i,m})
                │
                ▼
   Stage 2: Joint Cross-Spectral Modeling (Compute S_r, U_r, Λ_r)
                │
                ▼
   Stage 3: Real Gaussian BLUP / PACE (Estimate Posteriors b_i)
                │
                ▼
   Stage 4: Physiological Fingerprints (Construct z_i)

```

### 1. Data Representation

Let $Y_i(h, s)$ denote a functional observation for subject $i \in \{1, \dots, N\}$ recorded at hour-of-day $h \in \{0, 1, \dots, 23\}$, evaluated over a continuous scale/time grid $s \in [s_{\min}, s_{\max}]$:

$$Y_i(h, s) \in L^2([s_{\min}, s_{\max}])$$

Subjects may have incomplete recordings across the 24-hour cycle. The set of observed hours for subject $i$ is denoted by $\mathcal{H}_i \subseteq \{0, 1, \dots, 23\}$.

---

### 2. Stage 0 — Shape FPCA

Pooled functional observations across all subjects and hours are decomposed into an overall population mean trajectory $\mu(s)$ and orthogonal shape eigenfunctions $\phi_m(s)$:

$$Y_i(h, s) = \mu(s) + \sum_{m=1}^{M} \xi_{i,m}(h) \phi_m(s) + \epsilon_i(h, s)$$

#### Numerical Quadrature Integration

To preserve the continuous $L^2(w)$ inner product on non-uniform or logarithmic scale grids, a diagonal matrix of trapezoidal quadrature weights $W = \text{diag}(w_1, \dots, w_S)$ is applied during SVD:

$$X_c W^{1/2} = U \Sigma V^\top \implies \phi_m(s) = W^{-1/2} V_m$$

where $X_c$ is the row-centered matrix of all pooled functional curves.

The functional curve for subject $i$ at hour $h$ is projected into an $M$-dimensional score vector $X_i(h) = (\xi_{i,1}(h), \dots, \xi_{i,M}(h))^\top$:

$$\xi_{i,m}(h) = \int_{s_{\min}}^{s_{\max}} (Y_i(h, s) - \mu(s)) \phi_m(s) w(s) \, ds$$

---

### 3. Stage 1 — Circadian Baseline Centering

Subject shape scores $X_{i,m}(h)$ are decomposed into static baseline shifts $\bar{\eta}_{i,m}$ and sub-daily circadian deviations $\tilde{X}_{i,m}(h)$:

$$X_{i,m}(h) = \bar{\eta}_{i,m} + \tilde{X}_{i,m}(h)$$

where the static baseline offset for shape mode $m$ is computed as:

$$\bar{\eta}_{i,m} = \frac{1}{\vert{}\mathcal{H}_i\vert{}} \sum_{h \in \mathcal{H}_i} X_{i,m}(h)$$

---

### 4. Stage 2 — Joint Cross-Spectral Covariance Modeling

#### A. Two-Sided Lag Covariance

The $M \times M$ lag-covariance matrix $\Sigma(d)$ across shape mode deviations is calculated over lag differences $d \in \{-11, \dots, 12\}$:

$$\Sigma(d) = \text{Cov}(\tilde{X}_i(h), \tilde{X}_i(h+d))$$

For non-zero lags ($d \neq 0$), $\Sigma(d)$ is asymmetric ($\Sigma(-d) = \Sigma(d)^\top$). Preserving asymmetry is necessary to capture cross-channel phase leads and lags (the quadrature spectrum).

#### B. Two-Sided Cross-Spectral Density Matrix

Applying the Discrete Fourier Transform over centered lags $d \in \{-11, \dots, 12\}$ yields complex $M \times M$ Hermitian cross-spectral matrices $S_r$ for harmonic frequencies $r \in \{1, \dots, R\}$:

$$S_r = \sum_{d=-11}^{12} \Sigma(d) e^{-i \frac{2\pi r d}{24}} \in \mathbb{C}^{M \times M}$$

#### C. Ledoit-Wolf Shrinkage & Hermitian Decomposition

To prevent overfitting on noisy or small-sample cross-spectral matrices, shrinkage toward the diagonal target is applied:

$$\hat{S}_r = (1 - \lambda) S_r + \lambda \text{diag}(S_r)$$

where $\lambda \in$ controls regularization.

Hermitian eigendecomposition yields coordinated harmonic mode eigenvectors $U_r$ and variance eigenvalues $\Lambda_r$:

$$\hat{S}_r = U_r \Lambda_r U_r^*$$

* **Eigenvectors $U_r \in \mathbb{C}^{M \times M}$**: Define linear combinations of shape modes that co-oscillate at harmonic frequency $r$.
* **Eigenvalues $\Lambda_r$**: Quantify the variance explained by each coordinated harmonic mode.

---

### 5. Stage 3 — Real Gaussian BLUP / PACE Estimation

When subject recordings are incomplete ($\mathcal{H}_i \subset \{0, \dots, 23\}$), Fourier coefficients $b_i$ are estimated using Gaussian Best Linear Unbiased Prediction (BLUP / PACE).

#### A. Real Block Prior Construction

The complex spectral covariance $S_r = U_r \Lambda_r U_r^*$ is mapped into a real-valued $2M \times 2M$ block Gaussian prior covariance matrix $\Sigma_r$ for real Fourier sine/cosine coefficients $(a_{i,r}, b_{i,r})^\top$:

$$\Sigma_r = \begin{pmatrix} \text{Re}(\hat{S}_r) & -\text{Im}(\hat{S}_r) \\ \text{Im}(\hat{S}_r) & \text{Re}(\hat{S}_r) \end{pmatrix}$$

The complete prior covariance matrix across $R$ harmonics is:

$$\Sigma_{\text{prior}} = \text{block\_diag}(\Sigma_1, \dots, \Sigma_R) \in \mathbb{R}^{2MR \times 2MR}$$

#### B. Posterior Estimation

For subject $i$ with observed hours $\mathcal{H}_i$, the observation operator is $H_i = I_M \otimes X_{\text{fourier}}(\mathcal{H}_i)$, where $X_{\text{fourier}}$ is the $\vert{}\mathcal{H}_i\vert{} \times 2R$ Fourier design matrix. The posterior mean vector $\hat{b}_i$ is computed in closed form:

$$\Sigma_{\text{post}} = \left( \Sigma_{\text{prior}}^{-1} + H_i^\top R_{\text{noise}}^{-1} H_i \right)^{-1}$$

$$\hat{b}_i = \Sigma_{\text{post}} H_i^\top R_{\text{noise}}^{-1} y_i$$

where $R_{\text{noise}} = \sigma_e^2 I_{\vert{}\mathcal{H}_i\vert{} M}$.

---

### 6. Stage 4 — Physiological Fingerprints

Subject representation vectors $z_i$ are constructed by projecting estimated Fourier coefficients onto coordinated modes $U_r$:

$$z_i = \Big[ \underbrace{\bar{\eta}_{i,1}, \dots, \bar{\eta}_{i,M}}_{\text{Static Baseline Offsets}}, \quad \underbrace{\left\{ \text{Amp}\left(\text{Re}(U_r^* \hat{a}_{i,r})\right), \text{Phase}\left(\text{Im}(U_r^* \hat{b}_{i,r})\right) \right\}_{r=1}^R}_{\text{Coordinated Circadian Amplitudes \& Phases}} \Big]^\top$$

The resulting compact vectors $z_i \in \mathbb{R}^{M + 2MR}$ serve as individual physiological fingerprints for downstream classification, clustering, and risk stratification.

---

## Baseline Models

The repository provides four comparative baseline models:

### 1. Traditional MFPCA (`jsmfpca/baselines/mfpca.py`)

Classical Multilevel FPCA (Di et al., 2009):

$$Y_{ij}(t) = \mu(t) + \eta_j(t) + \xi_i(t) + \epsilon_{ij}(t)$$

Decomposes variability into Level 1 (between-subject $K_B$) and Level 2 (within-subject $K_W$) functional principal components without explicit circadian spectral structure.

### 2. Diagonal / Mode-Independent Spectral Model (`jsmfpca/baselines/diagonal.py`)

Constrains cross-spectral matrices $S_r$ to be strictly diagonal:

$$S_r = \text{diag}(S_{r, 11}, \dots, S_{r, MM})$$

Forces $\text{Cov}(X_k, X_l) = 0$ for $k \neq l$. Tests whether cross-channel coordination provides additional predictive value over uncoupled univariate models.

### 3. OLS Harmonic Estimator (`jsmfpca/baselines/ols.py`)

Combines Shape FPCA with Ordinary Least Squares (OLS) Fourier regression and cross-spectral mode rotation, omitting Bayesian prior shrinkage.

### 4. Classical Cosinor Model (`jsmfpca/baselines/cosinor.py`)

Fits independent, subject-level 24-hour harmonic OLS regression directly on raw functional time series without population shrinkage or joint modes:

$$Y_i(t) = M_i + \sum_{r=1}^{R} \left( A_{ir} \cos\left(\frac{2\pi rt}{24}\right) + B_{ir} \sin\left(\frac{2\pi rt}{24}\right) \right) + \epsilon_i(t)$$

---

## Validation & Benchmark Framework

### 1. Clinical Target Association Suite (`validation/association.py`)

Evaluates the association between extracted physiological fingerprints $z_i$ and clinical markers:

* **BNP Association**: 3-fold cross-validated Ridge regression ($R^2$).
* **Survival Outcome State**: 3-fold cross-validated Logistic Regression ROC-AUC.

```bash
python validation/association.py

```

### 2. End-to-End Benchmark Framework (`validation/test.py`)

Evaluates all estimators across three standardized tasks:

* **Reconstruction Accuracy**: Mean Squared Error (MSE) between $Y_i(h, s)$ and reconstructed curves $\hat{Y}_i(h, s)$.
* **Phenotype Classification**: 5-fold cross-validated ROC-AUC and Accuracy.
* **Intrinsic Clustering**: K-Means Silhouette score and Adjusted Rand Index (ARI).

```bash
python validation/test.py

```

---

## Quickstart & Usage Examples

### Basic Fitting & Feature Extraction

```python
from jsmfpca import JSMFPCA, JSMFPCAData, SubjectCurves
import numpy as np

# 1. Create synthetic subject curve data
subjects = []
for subj_id in range(20):
    hours = np.array()
    curves = np.random.randn(len(hours), 50)  # 50 scale/time points
    subjects.append(SubjectCurves(subject_id=str(subj_id), hours=hours, curves=curves))

dataset = JSMFPCAData(subjects=subjects, scales=np.linspace(0, 1, 50))

# 2. Instantiate and fit JS-MFPCA model
model = JSMFPCA(n_modes=3, n_harmonics=2, shrinkage=0.25)
model.fit(dataset)

# 3. Extract physiological fingerprints
fingerprints = model.transform(dataset)
print("Fingerprint Matrix Shape:", fingerprints.shape)  # Shape: (20, 15)

# 4. Reconstruct 24-hour functional curves
reconstructed = model.reconstruct(dataset)
print("Reconstructed Subject 0 Curves Shape:", reconstructed[0].shape)

```

---

## Citation & References

If you use JS-MFPCA in your research, please cite:

```bibtex
@article{jsmfpca2026,
  title={Joint Spectral Multilevel Functional Principal Component Analysis for Coordinated Circadian Dynamics},
  author={Kruse, Jo{\~a}o Gabriel Segato and Kiyono, Ken},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2026}
}

```

* **Di, C. Z., Crainiceanu, C. M., Caffo, B. S., & Punjabi, N. M. (2009)**. Multilevel functional principal component analysis. *Annals of Applied Statistics*, 3(1), 458-488.
* **Yao, F., Müller, H. G., & Wang, J. L. (2005)**. Functional data analysis for sparse longitudinal data. *Journal of the American Statistical Association*, 100(470), 577-590.

```