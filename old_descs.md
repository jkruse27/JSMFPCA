# Joint Spectral Multilevel FPCA (JS-MFPCA) or Hierarchical Spectral Functional Decomposition (HSFD)
### A cross-modally-coupled, identifiable, circadian-spectral model for 24-hour nonlinear HRV curves

---

## 1. Data structure and modeling goal

For subject $i \in \{1,\dots,N\}$, at hour-of-day $h$, a window starting at hour $h$ yields a within-window nonlinear-HRV curve $X_i(h,s)$:

- $h \in \mathbb{Z}_{24} = \{0,1,\dots,23\}$: a **cyclic** index (hour $23$ is adjacent to hour $0$);
- $s \in [s_{\min},s_{\max}]$: the within-window curve coordinate (a scale/complexity axis, continuous or a fixed grid);
- $\mathcal{H}_i \subseteq \mathbb{Z}_{24}$: the hours actually observed for subject $i$ (missing hours are the norm).

**Goal.** From $\{X_i(h,\cdot)\}_{h\in\mathcal H_i}$ across all $i$, produce:

1. a population-level description of *what curve shapes exist* (Stage 0);
2. the population circadian pattern of each shape and each subject's typical offset from it (Stage 1);
3. how those shapes' circadian modulation is **coupled across shape-modes** — not just within each mode separately — with an explicit, checkable statistical assumption enabling this to be estimated cheaply (Stage 2);
4. a compact, interpretable, per-subject fingerprint (Stage 3);
5. optional nonlinear cross-subject geometry on that fingerprint (Stage 4);
6. optional linkage to a clinical outcome, without contaminating anything estimated in Stages 0–3 (Stage 5).

---

## 2. Statistical assumptions (stated explicitly)

- **A1 (square integrability).** $X_i(h,\cdot)\in L^2([s_{\min},s_{\max}])$ for every observed $(i,h)$, finite second moments, so the Karhunen–Loève / Mercer expansion in Stage 0 exists.
- **A2 (shared shape vocabulary).** The pooled covariance $C_s(s,s')=\mathrm{Cov}(X_i(h,s),X_i(h,s'))$, marginalized over $i$ and $h$, has a well-defined countable eigenbasis with distinct eigenvalues among the retained $K$ modes — needed for $\phi_k(s)$ identifiability up to sign. This implicitly assumes the *set of possible curve shapes* does not itself change qualitatively over the day — only the *amount* of each shape present does. (Checkable: compare pooled $\phi_k$ against separately-pooled night-only / day-only eigenbases; material disagreement flags a shape vocabulary that isn't shared, which this model does not capture.)
- **A3 (missingness).** Hours go missing at random given observed data/covariates (MAR). If missingness is itself informative (e.g. sensor removed during a symptomatic episode), this needs explicit handling — at minimum, test whether the *number or identity* of missing hours correlates with the outcome.
- **A4 (joint circular stationarity — the key assumption).** Let $\dot{\boldsymbol\eta}_i(h)=(\dot\eta_{i,1}(h),\dots,\dot\eta_{i,K}(h))^\top$ be the subject's circadian-shape deviation vector (Section 5). Assume $\{\dot{\boldsymbol\eta}_i(h)\}_h$ is mean-zero and **jointly** circularly wide-sense stationary: $\mathrm{Cov}(\dot{\boldsymbol\eta}_i(h),\dot{\boldsymbol\eta}_i(h')) = \Sigma\big((h-h')\bmod 24\big)$ for a common (population-level), matrix-valued function $\Sigma:\mathbb Z_{24}\to\mathbb R^{K\times K}$ — the *same* law for every subject, only the realized deviation differing. This is what licenses the spectral reduction in Stage 2; it is a natural (not much stronger) joint extension of assuming each mode is marginally stationary, and it is directly checkable (Section 5, diagnostic).
- **A5 (conditional independence across levels).** Between-subject offsets and within-subject circadian deviations are independent given subject identity — the standard multilevel/mixed-model assumption inherited from MFPCA.
- **A6 (shrinkage validity).** BLUP/PACE shrinkage is exact under joint Gaussianity of scores and noise; otherwise it is interpreted as the best *linear* (not necessarily fully Bayes-optimal) predictor, as is standard practice for MFPCA/PACE.
- **A7 (identifiability of any optional nonlinear stage).** If an auxiliary-variable-conditioned nonlinear factor model is used in Stage 4, the auxiliary variable must be observed and must **not** be the eventual outcome of interest, or the identifiability guarantee it relies on no longer applies cleanly to held-out prediction of that outcome.

---

## 3. Stage 0 — Pooled linear shape basis over $s$

Pool every observed $(i,h)$ curve (deliberately mixing all sources of variation at this stage—the only question being asked here is *"what curve shapes are physiologically possible?"*):

$$
\mu(s)=\mathbb E[X_i(h,s)],\qquad
C_s(s,s')
=
\mathrm{Cov}\!\left(X_i(h,s),X_i(h,s')\right).
$$

By Mercer's theorem,

$$
C_s(s,s')
=
\sum_{k=1}^{\infty}
\lambda_k
\phi_k(s)
\phi_k(s'),
$$

where the eigenfunctions $\{\phi_k\}$ form an orthonormal basis of
$L^2([s_{\min},s_{\max}])$. We retain the first $K$ components by minimizing
cross-validated reconstruction error rather than using a fixed explained-variance
threshold.

For every observed curve,

$$
\xi_{i,k}(h)
=
\int
\left(
X_i(h,s)-\mu(s)
\right)
\phi_k(s)\,ds.
$$

### Numerical implementation

After discretizing the scale axis into $S$ points $\{s_j\}_{j=1}^S$, let $$X_c \in \mathbb R^{M\times S}$$ denote the pooled centered data matrix containing all $M=\sum_i|\mathcal H_i|$ observed curves, one curve per row. To preserve the continuous $L^2$ inner product on a generally non-uniform (logarithmic) scale grid, let

$$W = \operatorname{diag}(w_1,\ldots,w_S)$$

be the diagonal matrix of trapezoidal quadrature weights over the scale axis. Rather than explicitly forming the empirical covariance matrix, we compute the weighted thin singular value decomposition $$X_cW^{1/2} = U\Sigma V^\top.$$ This is algebraically equivalent to eigendecomposing the weighted empirical covariance operator,

$$C_s = \frac{1}{M-1} W^{-1/2} V \Sigma^2 V^\top W^{-1/2},$$

so that

$$\phi_k=W^{-1/2}V_k,\qquad\lambda_k=\frac{\sigma_k^2}{M-1},$$

where $\sigma_k$ is the $k$th singular value. The resulting eigenfunctions are
renormalized with respect to the weighted inner product,

$$\langle f,g\rangle_W=\sum_{j=1}^{S}w_jf(s_j)g(s_j),$$

ensuring convergence to the continuous FPCA solution as the discretization is refined. For a new curve, the corresponding scores are computed using the weighted projection

$$\xi_{i,k}(h)=\sum_{j=1}^{S}w_j\left(X_i(h,s_j)-\mu(s_j)\right)\phi_k(s_j),$$

which is the discrete approximation of the continuous projection integral above. If the scale axis itself is sparsely or irregularly observed rather than densely sampled, the projection step may instead be replaced by a PACE-type conditional expectation estimator (Yao, Müller & Wang, 2005), using the same shrinkage principle later employed for the circadian domain in Stage 2.

---

## 4. Stage 1 — Population circadian pattern and between-subject offset

For each mode $k$, fit the population circadian pattern by low-order harmonic regression on pooled $\xi_{i,k}(h)$:

$$
\mu_k(h)=a_{k,0}+\sum_{r=1}^{R_0}\Big[a_{k,r}\cos\Big(\tfrac{2\pi rh}{24}\Big)+b_{k,r}\sin\Big(\tfrac{2\pi rh}{24}\Big)\Big]
$$

Residual $\eta_{i,k}(h)=\xi_{i,k}(h)-\mu_k(h)$. Subject/mode offset $\bar\eta_{i,k}=\mathbb E_h[\eta_{i,k}(h)]$ (MFPCA's between-subject score, recovered as a special case), shrunk via

$$
\hat{\bar\eta}_{i,k}=\frac{\sigma^2_{B,k}}{\sigma^2_{B,k}+\sigma_k^2/|\mathcal H_i|}\;\bar\eta_{i,k}^{\text{raw}}
$$

with $\sigma^2_{B,k}=\mathrm{Var}_i(\bar\eta_{i,k})$, $\sigma_k^2$ the residual variance. **This stage is left mode-by-mode deliberately** — the population mean pattern for mode $k$ already pools over the full dataset (large effective $N$), so there is little to gain from cross-mode borrowing here; the real gain from joint modeling is in Stage 2, for subject-level shrinkage from sparse individual data.

---

## 5. Stage 2 — Joint cross-spectral circadian deviation model (the corrected core)

### 5.1 Setup

Define the $K$-vector circadian-shape deviation $\dot{\boldsymbol\eta}_i(h)=\big(\eta_{i,1}(h)-\hat{\bar\eta}_{i,1},\ \dots,\ \eta_{i,K}(h)-\hat{\bar\eta}_{i,K}\big)^\top$. Under **A4**, its covariance is the matrix-valued lag function $\Sigma(d)\in\mathbb R^{K\times K}$, $d\in\mathbb Z_{24}$, satisfying $\Sigma\big((-d)\bmod24\big)=\Sigma(d)^\top$ (from $\mathrm{Cov}(\cdot,\cdot)$ symmetry).

### 5.2 Theorem (joint spectral decomposition)

Let $\mathbf\Sigma\in\mathbb R^{24K\times24K}$ be the block matrix with block $(a,b)=\Sigma\big((a-b)\bmod24\big)$. Define the cross-spectral matrix at harmonic $r$:

$$
S_r=\sum_{d=0}^{23}\Sigma(d)\,e^{-2\pi i rd/24}\ \in\ \mathbb C^{K\times K},\qquad r=0,1,\dots,12
$$

Then $S_r$ is Hermitian for every $r$, and the eigenvalues of $\mathbf\Sigma$ are **exactly** the union (with multiplicity) of the eigenvalues of $S_0$ (once), $S_{12}$ (once), and $S_r$ for $r=1,\dots,11$ (each **twice**). Diagonalizing $S_r=U_r\Lambda_rU_r^\top$ gives, at harmonic $r$, the linear combinations of the $K$ shape-modes that co-oscillate together ("coordinated circadian rhythms"), with $\Lambda_r$ their variances.

### 5.3 Proof

$S_r$ Hermitian: using $\Sigma(-d\bmod24)=\Sigma(d)^\top$ and reindexing $d'=(-d)\bmod24$,
$$
S_r^\dagger=\sum_d\Sigma(d)^\top e^{2\pi i rd/24}=\sum_{d'}\Sigma(d')e^{2\pi i r(24-d')/24}=\sum_{d'}\Sigma(d')e^{-2\pi i rd'/24}=S_r.
$$
Eigenvalue identity: let $e_r[h]=e^{2\pi i rh/24}$ and consider a length-$24K$ vector built from blocks $e_r[h]\,v$ for $v\in\mathbb C^K$. The $a$-th block of $\mathbf\Sigma(e_r\otimes v)$ is
$$
\sum_{b=0}^{23}\Sigma\big((a-b)\bmod24\big)e_r[b]\,v
=\sum_{d=0}^{23}\Sigma(d)\,e_r[a-d]\,v
=e_r[a]\underbrace{\Big(\sum_{d=0}^{23}\Sigma(d)e^{-2\pi i rd/24}\Big)}_{=S_r}v.
$$
So if $S_rv=\lambda v$, then $\mathbf\Sigma(e_r\otimes v)=\lambda(e_r\otimes v)$: $e_r\otimes v$ is an eigenvector of $\mathbf\Sigma$ with the same eigenvalue $\lambda$. Since $\Sigma(d)$ is real, $S_{24-r}=\overline{S_r}$, so the harmonic-$r$ and harmonic-$(24-r)$ eigenpairs are complex conjugates of one another and combine (standard real/complex-DFT symmetry, exactly as in reconstructing a real signal from a Hermitian-symmetric spectrum) into real eigenvectors of $\mathbf\Sigma$ spanning a $2$-real-dimensional eigenspace at eigenvalue $\lambda$, for $r=1,\dots,11$; $r=0,12$ each contribute one real eigenvector ($S_0,S_{12}$ are real symmetric). $\blacksquare$

**Verified numerically.** A 3-mode process with a chosen cross-mode-correlated harmonic amplitude structure gave, from a direct numerical `eigh` of the full $72\times72$ matrix versus the union of eigenvalues of each estimated $3\times3$ cross-spectral matrix:
```
direct eigh (top 12):     [46.322 46.322 29.937 29.937 18.643 18.643 17.633 17.633 9.575 9.575 5.131 5.131]
theory, union of S_r eigs: [46.321 46.321 29.936 29.936 18.644 18.644 17.633 17.633 9.575 9.575 5.131 5.131]
max abs difference across all 72 eigenvalues: 0.0026   (sampling noise only)
```

### 5.4 Estimation

1. Pool cross-products across subjects: $\hat\Sigma(d)=\frac1{\sum_i|\mathcal H_i(d)|}\sum_i\sum_{h\in\mathcal H_i(d)}\dot{\boldsymbol\eta}_i(h)\dot{\boldsymbol\eta}_i(h+d)^\top$, where $\mathcal H_i(d)$ is the set of hours for which both $h$ and $h+d\ (\mathrm{mod}\ 24)$ were observed for subject $i$.
2. $\hat S_r=\sum_d\hat\Sigma(d)e^{-2\pi i rd/24}$.
3. **Shrink** $\hat S_r$ toward its diagonal (Ledoit–Wolf-type target), shrinkage intensity chosen by cross-validated reconstruction error. This is necessary because $\hat S_r$ has $O(K^2)$ free parameters per harmonic versus $O(K)$ in a mode-independent model — with realistic $N$, unregularized off-diagonals are noisy. The selected shrinkage intensity is itself diagnostic: if cross-validation pushes it to (near-)full diagonal shrinkage, the data do not support cross-mode coupling and Stage 2 collapses back to the mode-independent model — this supersedes needing a separate, upfront hypothesis test for separability.
4. Diagonalize the shrunk $\hat S_r=\hat U_r\hat\Lambda_r\hat U_r^\top$, retain harmonics/components by cross-validated reconstruction.
5. **Diagnostic for A4.** Separately estimate a fully nonparametric joint covariance surface (periodic-kernel smoothing of $\dot{\boldsymbol\eta}_i(h)\dot{\boldsymbol\eta}_i(h')^\top$ without imposing lag-only dependence) and compare its eigenstructure to the harmonic prediction above (eigenvalue agreement, principal angles between eigenspaces) — material disagreement is a precise, checkable signature that circadian coupling is not well described by joint stationarity (e.g. a genuinely non-classical, localized, or subject-heterogeneous timing pattern), exactly generalizing the scalar diagnostic used earlier for a single mode.
6. **Subject-level scores.** Multivariate BLUP/PACE: stack subject $i$'s observed values across modes and hours into one vector; using the (shrunk) $\hat\Sigma$-implied prior covariance and the estimated measurement-noise covariance, form the standard multivariate Gaussian conditional-expectation estimator for the subject's full (unobserved-hours-included) harmonic coefficients — the direct multivariate generalization of univariate PACE, reconstructing missing hours **jointly across modes and time** rather than mode-by-mode.

---

## 6. Stage 3 — Interpretable, rotated fingerprint

$$
Z_i=\Big(\underbrace{\hat{\bar\eta}_{i,1},\dots,\hat{\bar\eta}_{i,K}}_{\text{typical level, per mode (Stage 1)}},\ \underbrace{\big\{\hat U_r^\top(\hat{\mathbf a}_{i,r},\hat{\mathbf b}_{i,r})\big\}_{r=1}^R}_{\text{coordinated-rhythm amplitudes (Stage 2)}}\Big)
$$

Project onto $\hat U_r$'s columns — the coordinated-rhythm axes discovered by diagonalizing $S_r$ — rather than reporting the raw per-mode harmonic coefficients $(\hat{\mathbf a}_{i,r},\hat{\mathbf b}_{i,r})$; only the rotated version actually uses the fact that $S_r$'s eigendecomposition tells you which modes move together. With $K$ mesor components and, say, $R{=}3$ retained harmonics each contributing $K'\le K$ non-negligible coordinated components (cos+sin), $d=K+2\sum_rK'_r$ — typically on the order of $15$–$40$.

Every coordinate has a physiological reading: "typical level of shape-mode $k$" (mesor); "how much of coordinated-rhythm component $c$ at harmonic $r$ this subject shows" (an amplitude on a specific, data-discovered, cross-modal circadian pattern).

---

## 7. Stage 4 — Nonlinear cross-subject geometry (optional, on $Z_i$ only)

In increasing order of complexity/data requirement:

1. **Principal curves/surfaces** (Hastie & Stuetzle, 1989) through $\{Z_i\}$ — minimal data requirement, fully visualizable, each subject's arc-length position is one interpretable nonlinear score.
2. **Kernel PCA / Isomap / diffusion maps** if structure isn't one-dimensional.
3. **Auxiliary-variable-identified nonlinear factor model** (iVAE-style; Khemakhem, Kingma, Monti & Hyvärinen, 2020) only with enough subjects (order of a few hundred) and a specific reason to expect genuinely latent (not just curved) structure — applied to $Z_i$, where it is well-posed, never to the raw curve.

The network's role, if used at all, is strictly to model geometry **among already-identified variables** — it never defines the latent space itself, which is what keeps this stage free of the reparameterization/ordering ambiguities that affect free-form neural latent models or naively-trained neural eigenfunction learners.

---

## 8. Stage 5 — Outcome linkage (optional, fully separated)

Fit the representation (Stages 0–4) with **no** outcome $Y_i$ involved. Only afterward, fit
$$
Y_i = g(Z_i)
$$
via a sign-preserving, regularized method (ridge/elastic-net regression, or the cross-covariance/PLS construction derived earlier in this line of work — **never** squared-outcome weighting of a covariance operator), with model/regularization choice by **nested** cross-validation, and always benchmarked against the unsupervised fingerprint alone. If representation should be nudged toward outcome-relevance, do so as a separate fine-tuning stage on $Z_i$, with the nudge strength itself chosen by nested CV — never by modifying the covariance/spectral estimation in Stages 0–2.

---

## 9. Full algorithm

```
INPUT: for i = 1..N, hourly curves {X_i(h, ·)}_{h in H_i}, H_i ⊆ {0,...,23}

STAGE 0  (pooled shape basis over s)
  pool all observed (i,h) curves → mean μ(s), covariance C_s(s,s')
  eigendecompose → {φ_k(s), λ_k}; choose K by CV reconstruction error
  ξ_{i,k}(h) = <X_i(h,·) − μ(·), φ_k>   for every observed (i,h)   (PACE if s sparse)

STAGE 1  (population circadian pattern + shrunk subject offset, per mode k)
  μ_k(h): harmonic regression of pooled ξ_{·,k}(h) over h
  η_{i,k}(h) = ξ_{i,k}(h) − μ_k(h)
  \barη_{i,k}^raw = mean_{h in H_i} η_{i,k}(h);  BLUP-shrink → \hatη̄_{i,k}   (per mode, independently — Section 4)

STAGE 2  (joint cross-spectral circadian deviation model)
  \dotη_i(h) = ( η_{i,1}(h) − \hatη̄_{i,1}, ..., η_{i,K}(h) − \hatη̄_{i,K} )        [K-vector]
  Σ̂(d) = pooled cross-products, over co-observed (h, h+d) pairs, across subjects
  Ŝ_r  = DFT_d[ Σ̂(d) ]                                                          [K x K, r=0..12]
  shrink each Ŝ_r toward diagonal (Ledoit-Wolf; intensity by CV)
  diagonalize: Ŝ_r = Û_r Λ̂_r Û_r^T  → coordinated-rhythm axes + variances
  [diagnostic] nonparametric joint covariance surface vs. harmonic prediction → check A4
  multivariate BLUP/PACE: shrink subject-level harmonic coefficients (â_{i,r}, b̂_{i,r})
    jointly across modes AND time, using only H_i

STAGE 3  (rotated, interpretable fingerprint)
  Z_i = ( \hatη̄_{i,1..K},  { Û_r^T (â_{i,r}, b̂_{i,r}) }_{r=1..R} )     [dimension d, modest]

STAGE 4  (optional nonlinear geometry on Z_i)
  principal curve / kernel-PCA / diffusion map / (if N large) identified nonlinear factor model

STAGE 5  (optional outcome linkage, fully separated)
  nested-CV-validated g: Z_i -> Y_i   (never applied to raw curves; never modifies Stages 0-2)

OUTPUT: μ(s), {φ_k(s)}, {μ_k(h)}, {Ŝ_r, Û_r, Λ̂_r} with A4 diagnostic, and Z_i per subject.
```

---

## 10. Why this is identifiable

Every object above is one of exactly two things: (i) a fixed linear projection onto $\phi_k(s)$, determined once by an ordinary eigendecomposition of $C_s$; or (ii) a covariance/cross-spectral matrix of quantities *already fully determined* once (i) is fixed, followed by its eigendecomposition. At no stage is a flexible function jointly optimized together with free per-subject parameters — the mechanism that makes free-form neural latent models (and naively-trained neural-eigenfunction learners) non-identifiable without extra machinery. Estimating $S_r$ and diagonalizing it is exactly as identified (up to sign, and up to arbitrary rotation only within a repeated-eigenvalue subspace — the same caveat ordinary PCA always carries) as Stage 0's $\phi_k(s)$ is. This also avoids the CP/tensor non-uniqueness that a general "sum of $M$ separable covariance operators" model would need to resolve separately: the $s$-side basis here is fixed and singular (one $\phi_k$ each), so all of the added cross-structure lives in the small, well-posed $K\times K$ spectral objects, never in a joint tensor decomposition.

---

## 11. Novelties

- A closed-form, verified theorem that a jointly circularly-stationary $K$-vector deviation process is block-diagonalized by the DFT into $K\times K$ cross-spectral matrices — the exact multivariate generalization of the scalar cosinor/harmonic result, giving "coordinated circadian rhythms across physiological modes" a precise spectral definition rather than a descriptive one.
- A precise, falsifiable operational test for both (a) whether circadian modulation is classical/stationary at all (Section 5.4, item 5) and (b) whether it is coupled across shape-modes (the CV-selected shrinkage intensity toward diagonal, Section 5.4 item 3) — replacing two separate motivational claims ("physiology is nonlinear," "mechanisms interact across scale and time") with two concrete, reportable statistics.
- A nested, cost-aware architecture that keeps every hard "raw curve → numbers" step inside classical, identifiable, low-sample-complexity machinery, and reserves genuinely nonlinear tools for an already low-dimensional, already-denoised fingerprint.
- Native, principled missing-hour handling (multivariate BLUP/PACE) rather than imputation bolted on afterward.

## 12. Where it builds from

| Component | Source |
|---|---|
| Multilevel between/within decomposition, BLUP shrinkage | Di, Crainiceanu, Caffo & Punjabi (2009), *Multilevel functional principal component analysis*, Ann. Appl. Stat. |
| Sparse/irregular FPCA via conditional expectation | Yao, Müller & Wang (2005), *Functional Data Analysis for Sparse Longitudinal Data*, JASA |
| Spectral theory for periodically-correlated functional processes | Kidziński, Kokoszka & Mohammadi Jouzdani, *PCA of periodically correlated functional time series* |
| FPCA respecting circular/manifold topology | Recent work on FPCA for manifold-indexed (circular) data |
| Classical cosinor / harmonic regression | Halberg-tradition chronobiology cosinor methodology |
| Separable / product-sum covariance models in spatio-temporal statistics | Genton (2007), *Separable approximations of space-time covariance matrices*; Cressie & Huang (1999); Gneiting (2002) product-sum constructions |
| Shrinkage estimation of covariance/spectral matrices | Ledoit & Wolf-type shrinkage-toward-diagonal estimators |
| Nonlinear-but-interpretable low-dimensional structure | Hastie & Stuetzle (1989), *Principal Curves*, JASA |
| Identifiable nonlinear latent variable models | Khemakhem, Kingma, Monti & Hyvärinen (2020), *iVAE*, AISTATS |
| Neural eigenfunction learning (context for why Stage 4's network is scoped as it is) | Pfau, Petersen, Agarwal, Barrett & Stachenfeld (2019), *Spectral Inference Networks*; Deng et al. (2022), *NeuralEF* |

## 13. Where it innovates

- **Vs. plain MFPCA**: replaces the exchangeable-visit assumption with a derived joint spectral structure over hour-of-day; recovers MFPCA's between/within split as the degenerate case where circadian structure and cross-mode coupling are both ignored.
- **Vs. SMFPCA**: outcome supervision, if used, touches only the low-dimensional $Z_i$, sign-preserving and nested-CV-validated — none of the bounded-amplification, sign-blindness, or binary-outcome degeneracies apply.
- **Vs. HIPF/CHPF**: identifiable by construction (no free-form decoder jointly optimized with per-subject codes); no train/test inconsistency (a new subject's fingerprint uses the same closed-form shrinkage as training).
- **Vs. a single separable covariance $C_h\otimes C_s$**: already captures mode-specific and now cross-mode temporal dynamics, not a single shared temporal law for every shape mode.
- **Vs. a general sum of $M$ separable operators**: gets the cross-structure benefit without the CP/tensor identifiability question, by keeping the $s$-basis fixed/singular and pushing all richness into a small $K\times K$ spectral object.
- **Vs. a neural spectral operator**: gets a genuinely richer, coupled spectral structure without importing the ordering/identifiability machinery (auxiliary-variable conditioning, careful SpIN/NeuralEF-style loss design) that a neural eigenfunction learner would need.

## 14. Expected regimes of improvement — and when this is overkill

- **Helps most** when distinct physiological mechanisms plausibly act on different shape-modes at different, possibly-linked circadian times (e.g. a shared disease process producing coordinated night/morning effects across scales) — exactly what the cross-spectral matrices are built to reveal.
- **Buys little over Stage-1-only mode-independent modeling** if cross-validated shrinkage of $\hat S_r$ collapses to near-diagonal — report the simpler model in that case and say so.
- **Buys little over classical single-mode cosinor** if the Section 5.4 diagnostic shows the data-driven and harmonic-predicted eigenbases coincide — the joint/spectral machinery isn't earning its keep if circadian modulation really is close to a low-order sinusoid per mode with no cross-mode coupling.
- **Needs enough subjects** for the $K\times K$ cross-spectral matrices to be stable (check via bootstrap) before trusting coordinated-rhythm interpretations, and a sample size compounding across two nested estimation levels — larger than what plain MFPCA would need.

## 15. Implementation approach

- **Computational cost is small, correctly understood.** $K$ is a handful (3–10), so the working matrices are at most a few hundred-dimensional — trivial relative to the $\sim$12,000-dimensional joint $(h,s)$ covariance a fully unstructured approach would require. The earlier "computational obstacle" is better understood as a **statistical** one (estimating that many covariance parameters from realistic $N$ is ill-posed) — which is exactly why this factored construction, not a bigger computer, is the right fix.
- **Model selection, all by cross-validated reconstruction (not fixed variance thresholds):** $K$ (Stage 0), number of retained harmonics $R$ and coordinated components per harmonic (Stage 2), shrinkage intensity for $\hat S_r$ (Stage 2), and — separately, nested — any regularization touching $Y$ (Stage 5).
- **Diagnostics to run before trusting the model:** (a) compare pooled vs. daypart-specific $\phi_k(s)$ (checks A2); (b) compare harmonic-predicted vs. nonparametric joint eigenbasis (checks A4); (c) check whether CV pushes $\hat S_r$ shrinkage toward full diagonality (checks whether cross-mode coupling is actually supported); (d) bootstrap stability of $\mu_k(h)$ and $\hat S_r$.
- **Missingness mechanism.** Confirm approximate MAR; at minimum check whether the number/identity of missing hours correlates with the outcome, and if so treat missingness as an explicit feature rather than a nuisance to shrink away.
- **Minimum data guidance.** Population-level curves ($\mu(s)$, $\mu_k(h)$) can be estimated even from subjects with only 1–2 observed hours each, provided *aggregate* hour-of-day coverage across the cohort is reasonable; individual coordinated-rhythm fingerprints need a handful of well-spread hours per subject to be informative; trusting Stage 4 nonlinear geometry needs on the order of a few hundred subjects with reasonably complete circadian coverage.
- **Baseline to beat.** Before adopting the full pipeline, benchmark against (a) plain MFPCA ignoring hour structure, (b) mode-independent circadian modeling (Stage 2 with $\hat S_r$ forced diagonal), and (c) classical single-subject cosinor fits with no cross-subject shrinkage. If neither reconstruction accuracy nor the diagnostics above show a material gain over (b), the cross-modal machinery isn't justified for that dataset and the simpler model should be reported.






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
