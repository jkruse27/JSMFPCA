# Joint Spectral Multilevel Functional Principal Component Analysis (JS-MFPCA)

A hierarchical functional data analysis framework for discovering coordinated circadian dynamics from multivariate longitudinal physiological signals.

---

## Overview

JS-MFPCA is a multilevel functional principal component framework designed to model complex circadian physiological signals while separating:

1. Global waveform morphology
2. Subject-specific temporal deviations
3. Shared circadian organization
4. Cross-variable harmonic coordination
5. Individual physiological fingerprints

The framework was developed for high-dimensional physiological time series such as:

* Heart rate variability (HRV)
* ECG-derived respiration (EDR)
* Multiscale autonomic markers
* Circadian functional phenotypes

The central hypothesis is:

> Physiological phenotypes are not defined only by average values or isolated circadian amplitudes, but by coordinated multiscale temporal organization across individuals.

JS-MFPCA extracts these coordinated structures through a hierarchical decomposition.

---

## Methodology

The model consists of four stages:

Raw functional observations
           |
           v
Stage 0: Shape FPCA
           |
           v
Stage 1: Circadian decomposition
           |
           v
Stage 2: Joint spectral covariance modeling
           |
           v
Stage 3: Physiological fingerprints

---

## Mathematical Formulation

### Data Representation

Each subject is represented by a multivariate functional observation:

$$Y_i(t)$$

Where:

* $i=1,\dots,N$ indexes subjects
* $t$ represents the observation time
* $K$ functional channels are measured

The observations are represented as:

$$Y_i(t) = \begin{bmatrix} Y_{i1}(t)\\ Y_{i2}(t)\\ \vdots\\ Y_{iK}(t) \end{bmatrix}$$

---

### Stage 0 — Shape FPCA

The first stage extracts dominant waveform morphology. The functional observation is decomposed as:

$$Y_i(t) = \mu(t) + \sum_{m=1}^{M} \xi_{im}\phi_m(t) + \epsilon_i(t)$$

Where:

* $\mu(t)$ is the population mean trajectory
* $\phi_m(t)$ are functional eigenfunctions
* $\xi_{im}$ are subject scores

The output is a low-dimensional score representation:

$$X_i = (\xi_{i1},\dots,\xi_{iM})$$

This removes unnecessary functional dimensionality.

---

### Stage 1 — Circadian Modeling

The FPCA scores are modeled as periodic processes. For each score dimension:

$$X_{im}(d)$$

Where $d$ is hour of day:

$$d \in [0,24)$$

A Fourier representation is used:

$$X_{im}(d) = \sum_{r=1}^{R} \left( a_{imr} \cos \left( \frac{2\pi rd}{24} \right) + b_{imr} \sin \left( \frac{2\pi rd}{24} \right) \right)$$

The coefficients are collected:

$$B_i = [a_{ir},b_{ir}]$$

The objective is not only to estimate individual rhythms, but to identify coordinated harmonic structures.

---

### Stage 2 — Joint Spectral Decomposition

The harmonic coefficient covariance is estimated. For lag $h$:

$$\Sigma(h) = \text{Cov}(X(t), X(t+h))$$

The discrete Fourier transform gives cross-spectral matrices:

$$S_r = \sum_h \Sigma(h) e^{-i2\pi rh/24}$$

Each harmonic has a cross-variable covariance structure:

$$S_r \in \mathbb{C}^{K\times K}$$

The decomposition:

$$S_r = U_r \Lambda_r U_r^*$$

Produces:

* Coordinated harmonic modes $U_r$
* Spectral variances $\Lambda_r$

The retained components define the physiological harmonic fingerprints.

---

### Spectral Shrinkage

Because cross-spectral matrices can be noisy, shrinkage is applied:

$$\hat S_r = (1-\lambda)S_r + \lambda D_r$$

Where:

* $D_r$ is a diagonal target
* $\lambda$ controls regularization

This improves stability in:

* Small cohorts
* High-dimensional settings
* Noisy physiological measurements

---

### Stage 3 — Physiological Fingerprints

For subject $i$, the final representation is:

$$z_i = f(B_i,U_r)$$

Where:

* Harmonic coefficients encode circadian timing
* Eigenvectors encode coordinated physiological modes
* Eigenvalues encode importance

The result is a compact vector describing the individual's circadian organization. These fingerprints can be used for:

* Clustering
* Classification
* Prognosis
* Phenotype discovery

---

### Bayesian BLUP / PACE Estimation

When observations are incomplete, coefficients are estimated using Gaussian posterior inference. Assume:

$$b_i \sim \mathcal{N}(0,\Sigma_b)$$

And:

$$y_i = Hb_i + \epsilon_i$$

With:

$$\epsilon_i \sim \mathcal{N}(0,R)$$

The posterior mean is:

$$E(b_i\vert{}y_i) = \Sigma_b H^T (H\Sigma_b H^T + R)^{-1}y_i$$

This provides:

* Missing-data robustness
* Uncertainty estimates
* Shrinkage toward population structure

---

## Software Architecture

```text
jsmfpca/
├── fpca/
│   └── ShapeFPCA
├── circadian/
│   └── CircadianModel
├── spectral/
│   ├── model.py
│   ├── covariance.py
│   ├── shrinkage.py
│   ├── estimator.py
│   └── selection.py
├── fingerprint/
│   └── FingerprintBuilder
├── baselines/
│   ├── mfpca.py
│   ├── diagonal_spectral.py
│   └── cosinor.py
├── benchmark/
│   ├── benchmark.py
│   ├── task.py
│   ├── results.py
│   ├── metrics.py
│   └── tasks/
│       ├── reconstruction.py
│       ├── classification.py
│       ├── clustering.py
│       ├── stability.py
│       └── runtime.py
└── tests/

```

---

## Main Estimator API

All estimators follow the same interface:

```python
model.fit(dataset)
representation = model.transform(dataset)
prediction = model.reconstruct(dataset)

```

This allows JS-MFPCA, MFPCA, diagonal spectral model, and cosinor to be evaluated identically.

---

## Baseline Models

### 1. Traditional MFPCA

Classical multilevel functional PCA:

$$Y_{ij}(t) = \mu(t) + \eta_j(t) + \xi_i(t) + \epsilon_{ij}(t)$$

Components:

* Between-subject variation
* Within-subject variation

*No explicit circadian spectral structure.*

### 2. Mode-Independent Spectral Model

A constrained JS-MFPCA:

$$S_r = \text{diag}(S_{r,11},\dots,S_{r,KK})$$

Forcing:

$$\text{Cov}(X_k,X_l) = 0$$

For:

$$k \neq l$$

*This tests whether cross-variable coordination provides additional information.*

### 3. Classical Cosinor Model

Independent subject-level harmonic regression:

$$y(t) = M + A \cos(\omega t-\phi) + \epsilon$$

No:

* Population shrinkage
* Covariance learning
* Joint modes

---

## Benchmark Framework

The benchmark module evaluates all estimators using identical protocols.

```python
benchmark = Benchmark(
    estimators=[
        JSMFPCA(),
        MFPCA(),
        DiagonalSpectral(),
        Cosinor(),
    ],
    tasks=[
        ReconstructionTask(),
        ClassificationTask(),
        ClusteringTask(),
        StabilityTask(),
        RuntimeTask(),
    ],
    cv=5,
)

results = benchmark.evaluate(dataset)

```

---

## Benchmark Tasks

### Reconstruction

Evaluates: 

$$\vert{}\vert{}Y-\hat Y\vert{}\vert{}$$


Metrics:

* RMSE
* MAE
* MSE
* $R^2$

### Classification

Uses learned fingerprints:

```text
fingerprint -> classifier -> clinical label

```

Metrics:

* Accuracy
* AUC
* Precision
* Recall
* F1

### Clustering

Evaluates intrinsic phenotype structure.
Metrics:

* Silhouette score
* Adjusted Rand Index
* Normalized Mutual Information

### Stability

Measures robustness under perturbation: 

$$\text{Similarity}(z,z')$$


Metrics:

* Cosine similarity
* Relative fingerprint error

### Runtime

Measures:

* Fitting time
* Transformation time
* Reconstruction time
* Memory consumption

---

## Model Selection

### Number of FPCA Modes

Selected by:

* Explained variance
* Cross-validation

### Harmonic Order

Controls temporal complexity.

* Typical values: $R=1,2,3$

### Spectral Shrinkage

$\lambda$ is selected by:

* Reconstruction error
* Validation likelihood
* Stability

---

## Scientific Questions Addressed

The framework enables testing:

**Does coordinated circadian structure improve phenotype prediction?**
Compare: **JS-MFPCA > MFPCA > Cosinor**

**Does cross-variable organization contain biological information?**
Compare: Full $S_r$ against Diagonal $S_r$

**Are physiological fingerprints stable?**
Evaluate: 

$$z_i \rightarrow z_i+\epsilon$$

 and quantify degradation.

**Does hierarchical modeling improve missing-data robustness?**
Evaluate performance under:

* Incomplete sampling
* Noisy measurements
* Sparse recordings

---

## Design Principles

The implementation follows:

1. Separation of modeling and evaluation
2. Common estimator API
3. Reproducible benchmarking
4. Explicit mathematical assumptions
5. Minimal task-specific logic

The goal is not only a statistical model, but a complete framework for discovering interpretable physiological phenotypes from longitudinal functional data.
