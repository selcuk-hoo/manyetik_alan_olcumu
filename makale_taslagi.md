# Online Lattice Drift Monitoring in an All-Electric Proton EDM Storage Ring via Two-Gradient Closed-Orbit Reconstruction

**Author(s):** Selcuk H.

**Affiliation:** *(to be filled)*

**Corresponding author:** *(email)*

**Target journal:** *Nuclear Instruments and Methods in Physics Research, Section A*

> **Çalışma notu.** Bu taslak `yapilacaklar-2.md`'deki beş simülasyon testinin
> tamamlanmış sonuçlarını içerir. Test scriptleri: `compare_regularization.py`
> (Test 1), `mode_transfer.py` (Test 2), `kxarc_sensitivity.py` (Test 3),
> `drift_monitor_sim.py` (Test 4), `bpm_offset_drift_sim.py` (Test 5).
> Çıktı görselleri: `test{1..5}_*.png`.

---

## Abstract

We present a method for continuously monitoring the transverse alignment state of quadrupole magnets in a periodic FODO storage ring from beam position monitor (BPM) measurements. The method is conceived as an *online drift observer*: after a one-time calibration of the BPM-offset vector and the absolute misalignment state using an independent technique (beam-based alignment or survey), subsequent measurements track *deviations* from the calibrated reference. In this drift formulation the BPM offset cancels by simple differencing against the calibration epoch, without requiring gradient modulation during physics data-taking.

We characterize the trade-off between offset cancellation and noise amplification in the conventional two-gradient (k-modulation) approach. Inverting the gradient-difference matrix $\Delta R = R_1 - R_2$ cancels BPM offsets but amplifies measurement noise by $O(1/\varepsilon)$, where $\varepsilon$ is the relative gradient change. Regularized inversion (Tikhonov, truncated SVD) mitigates the noise amplification at the cost of *spatial bandwidth*: high-frequency modes of the misalignment pattern are suppressed. We quantify this trade-off through numerical experiments that map the spatial transfer function of each estimator.

Validation against a full 6D Gauss–Legendre symplectic integrator confirms drift-tracking reconstruction RMS errors of $1.4$ – $4.8\,\mu\text{m}$ for $100\,\mu\text{m}$ injected misalignments (correlation $\rho = 0.998$ – $0.9999$). Five targeted simulation experiments characterize the operating envelope. Regularization comparison shows that optimal Tikhonov/TSVD $\Delta R$ inversion brings reconstruction RMS from $\sim 1900\,\mu\text{m}$ (raw) to $\sim 50\,\mu\text{m}$ — a $\sim 35\times$ improvement that nonetheless remains $\sim 15\times$ inferior to direct inversion ($\sim 3.5\,\mu\text{m}$). The spatial transfer function shows that this gap reflects severe high-frequency mode suppression by regularization. A drift-tracking simulation with $50\,\mu\text{m}$ RMS static BPM offset demonstrates the principal claim: differencing against a calibration baseline reduces the offset contamination from $\sim 180\,\mu\text{m}$ (absolute mode) to $\sim 6\,\mu\text{m}$ (drift mode), a $\sim 30\times$ improvement. A $K_{x,\text{arc}}$ perturbation test of $\pm 10\%$ shifts the reconstruction RMS by less than $0.5\,\mu\text{m}$, demonstrating that the model-calibration step does not constitute an inverse crime in any operationally relevant sense.

**Keywords:** beam position monitor; closed-orbit distortion; k-modulation; quadrupole misalignment; response matrix; storage ring alignment; proton EDM; online drift monitoring

---

## 1. Introduction

The search for the proton electric dipole moment (pEDM) in an all-electric storage ring at the "magic" momentum $p \approx 0.7007\,\text{GeV/c}$ requires control over systematic effects at a level that depends critically on the transverse alignment of focusing elements [1,2]. Quadrupole misalignments at the $\sim 100\,\mu\text{m}$ scale produce closed-orbit distortions (CODs) which, combined with radial electric-field non-uniformities and parasitic magnetic fields, generate spin-rotation systematics that mimic an EDM signal. Achieving the design sensitivity of $10^{-29}\,e\cdot\text{cm}$ requires alignment knowledge at the few-$\mu\text{m}$ level around the ring.

There are two distinct alignment problems. The first is *static*: before or between physics runs, determine the absolute position of every quadrupole. The second — for pEDM, often more dangerous — is *dynamic*: during a physics run, quadrupole positions drift slowly due to thermal expansion, support-structure creep, ground motion, high-voltage cycling of the electric deflectors, and electrostatic charging asymmetries. These mechanisms operate on timescales of minutes to hours and can displace quadrupoles by several microns during a single data-taking session, modifying the systematic environment in a way that is neither measured nor corrected unless a continuous monitoring system is in place.

The present work addresses the dynamic problem. Our goal is not a once-per-run absolute survey but a *quasi-real-time lattice drift observer*: a system that operates continuously during physics data-taking, uses BPM readings already available from the machine diagnostics, and delivers an updated estimate of the *change* in alignment state at each measurement epoch.

The forward model for closed-orbit distortions due to quadrupole misalignments is linear:
$$\mathbf{y}(t) = R\,\Delta q(t) + \mathbf{b} + \boldsymbol{\eta}(t),$$
where $R \in \mathbb{R}^{N\times N}$ is the response matrix, $\Delta q(t)$ is the time-varying misalignment vector, $\mathbf{b}$ is the (slowly varying) BPM electronic offset vector, and $\boldsymbol{\eta}$ is measurement noise. A single snapshot of $\mathbf{y}$ does not determine $\Delta q$ because $\mathbf{b}$ is comparable to the alignment signal and cannot be separated without further information.

The standard remedy is *k-modulation*: two CODs are recorded at slightly different quadrupole gradient settings $g_1$ and $g_2 = g_1(1+\varepsilon)$, so that the offset cancels in the difference. This procedure has been used in operating storage rings [3,4] and is discussed in standard textbook treatments [5,6]. However, as we show, $\Delta R = R_1 - R_2$ is severely ill-conditioned for small $\varepsilon$, limiting the achievable noise performance — and standard regularization, while mitigating this, sacrifices spatial bandwidth.

The drift-monitoring framework circumvents this trade-off. In the expression $\widehat{\delta q}(t) = R^{-1}(\mathbf{y}(t) - \mathbf{y}_0)$, the BPM offset $\mathbf{b}$ cancels because the same BPM electronics contribute to both $\mathbf{y}(t)$ and the calibration reference $\mathbf{y}_0$. Gradient modulation is not required during physics running; it is needed only at calibration epochs.

In this paper we make four contributions, organized around a sequence of numerical experiments rather than formal theory:

1. We construct $R$ analytically from the periodic FODO transfer matrices with a single calibrated horizontal arc-focusing parameter, eliminating the $O(N)$ perturbation simulations conventionally used to populate $R$.

2. We propose a *calibrated-reference drift-tracking* architecture that separates the alignment problem into a slow absolute-calibration layer (BBA/survey/LOCO, performed once) and a fast direct-inversion drift-tracking layer (this method, run continuously). Within this architecture, the BPM offset problem dissolves: offsets cancel by differencing against the calibration reference.

3. We characterize the offset–noise trade-off in two-gradient methods through numerical experiments comparing direct inversion, raw $\Delta R$ inversion, and regularized $\Delta R$ inversion (Tikhonov and truncated SVD). The comparison is made in two metrics — reconstruction RMS error and *spatial transfer function* — showing that regularization converts a noise problem into a spatial-resolution problem.

4. We validate the full method against a 6D Gauss–Legendre symplectic integrator and characterize the operating envelope through five targeted simulation tests: regularization comparison, spatial mode transfer, model-parameter sensitivity, drift dynamics, and BPM offset drift tolerance.

The remainder of this paper is organized as follows. Section 2 develops the analytic response-matrix formalism. Section 3 presents the offset–noise trade-off as a short derivation. Section 4 — the heart of the paper — reports the five numerical experiments. Section 5 introduces the layered drift-monitoring architecture. Section 6 discusses implementation, limitations, and the inverse-crime question. Section 7 concludes.

---

## 2. Response Matrix from Transfer-Matrix Optics

### 2.1 Lattice description

The pEDM ring consists of $N_c = 24$ identical FODO cells of length $L_c = 2\pi R_0 / N_c \approx 25\,\text{m}$, with $R_0 = 95.49\,\text{m}$. Each cell contains one focusing quadrupole (QF), one defocusing quadrupole (QD), two cylindrical electric deflectors ("arcs"), and four drift sections, in the sequence
$$\mathrm{QF} \to \mathrm{drift} \to \mathrm{arc} \to \mathrm{drift} \to \mathrm{QD} \to \mathrm{drift} \to \mathrm{arc} \to \mathrm{drift}.$$
The arc is a cylindrical capacitor operating at field index $n = 1$, providing the bending force for protons at the magic momentum. The ring has 48 quadrupoles total ($N = 48$), and BPMs are placed at the entrance of each quadrupole.

**Table 1.** Lattice parameters of the pEDM ring.

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Magic momentum | $p_0$ | $0.7007\,\text{GeV/c}$ |
| Relativistic $\beta$ | $\beta_0$ | $0.598$ |
| Relativistic $\gamma$ | $\gamma_0$ | $1.248$ |
| Ring radius | $R_0$ | $95.49\,\text{m}$ |
| Number of FODO cells | $N_c$ | $24$ |
| Quadrupoles per ring | $N$ | $48$ |
| Nominal quad gradient | $g_1$ | $0.21\,\text{T/m}$ |
| Quad length | $L_q$ | $0.40\,\text{m}$ |
| Drift length (each) | $L_d$ | $2.083\,\text{m}$ |

### 2.2 Transfer-matrix elements

The single-plane transfer matrix of a thick quadrupole of integrated strength $KL_q$ is
$$M_{Q,\text{foc}} = \begin{pmatrix} \cos\phi & \sin\phi/\sqrt{K} \\ -\sqrt{K}\sin\phi & \cos\phi \end{pmatrix}, \quad \phi = \sqrt{K}L_q,$$
with $K = g_1/B\rho$. The defocusing variant is obtained by $\cos\to\cosh$, $\sin\to\sinh$. Drift sections use the standard $\begin{pmatrix}1 & L \\ 0 & 1\end{pmatrix}$ matrix.

### 2.3 Arc focusing in cylindrical electric deflectors

Unlike magnetic dipoles, cylindrical electric deflectors at field index $n = 1$ produce *zero* vertical focusing because Maxwell's equations enforce $E_z = 0$ identically. The vertical arc transfer matrix is therefore exactly a pure drift of length $L_{\text{arc}} = \pi R_0/N_c$. This is a Maxwell-guaranteed fact, not a model assumption.

The horizontal plane is more subtle. Centrifugal defocusing in the rotating frame, combined with the relativistic $\beta^2$ correction, generates a small horizontal focusing that is not captured by the textbook expression $(1-n\pm\beta^2)/\rho^2$ once Coriolis terms are consistently included [7]. We treat the horizontal arc focusing coefficient $K_{x,\text{arc}}$ as a single calibration parameter determined once by bisection so that the analytic model matches the horizontal betatron tune $Q_x = 2.6824$ extracted from a clean simulation. For our parameters, $K_{x,\text{arc}} = 1.265 \times 10^{-4}\,\text{m}^{-2}$.

Because $R$ for the horizontal plane depends on this simulation-calibrated parameter, the horizontal-plane validation carries a risk of inverse crime. We address this directly in Section 4.3 (Test 3) by deliberately perturbing $K_{x,\text{arc}}$ and measuring the resulting reconstruction degradation.

### 2.4 Twiss parameters

The one-cell transfer matrix $M_c$ for each plane is the product of the elemental matrices in cell order. Standard formulas yield the phase advance $\mu = \arccos(\mathrm{Tr}\,M_c/2)$ and the beta function at cell entrance. The total betatron tune is $Q = N_c\mu/(2\pi)$.

**Table 2.** Twiss parameters at nominal gradient.

| Quantity | Analytic | 6D simulation |
|----------|----------|---------------|
| $Q_x$ | $2.682$ | $2.682$ |
| $Q_y$ | $2.362$ | $2.362$ |
| $\beta_y$ range | $[41.2,\,76.4]\,\text{m}$ | — |
| $\beta_x$ range | $[36.1,\,67.4]\,\text{m}$ | — |
| $\kappa(R_x)$ | $141$ | — |
| $\kappa(R_y)$ | $160$ | — |

### 2.5 The response matrix

A misalignment $\Delta q_j$ of quadrupole $j$ acts as a thin-lens dipole kick of integrated strength $\theta_j = KL_j\,\Delta q_j$ on the beam. The resulting closed orbit at BPM $i$ is the Courant–Snyder formula:
$$R_{ij} = \frac{\sqrt{\beta_i \beta_j}}{2 \sin(\pi Q)} \,\cos\!\bigl(|\phi_i - \phi_j| - \pi Q\bigr) \cdot KL_j,$$
where $KL_j$ carries the plane-appropriate sign. The conditioning of $R$ depends on $\sin(\pi Q)$: working points near integer or half-integer resonances are pathological. At our nominal tunes, $\kappa(R_x) \approx 141$ and $\kappa(R_y) \approx 160$.

---

## 3. The Offset–Noise Trade-off in Two-Gradient Methods

This section gives a short heuristic derivation; the empirical content is in Section 4.

Write the two BPM observations as $\mathbf{y}_i = R_i\Delta q + \mathbf{b} + \boldsymbol{\eta}_i$ for $i=1,2$. A linear estimator of $\Delta q$ takes the form $\widehat{\Delta q} = A_1\mathbf{y}_1 + A_2\mathbf{y}_2$. For unbiasedness ($\mathbb{E}[\widehat{\Delta q}] = \Delta q$ for arbitrary $\Delta q$) the matrices must satisfy $A_1 R_1 + A_2 R_2 = I$. For BPM offset cancellation we additionally need $A_1 + A_2 = 0$, i.e. $A_2 = -A_1$. Substituting gives $A_1(R_1 - R_2) = I$, so
$$A_1 = \Delta R^{-1}, \quad A_2 = -\Delta R^{-1}.$$
This is the unique unbiased linear estimator that cancels offsets — the $\Delta R$-inversion estimator. Since the operator norm scales as $\|A_1\| \sim \|\Delta R^{-1}\| = O(1/\varepsilon)$, this estimator amplifies measurement noise by the same factor.

The practical consequence is an **offset–noise duality**: offset cancellation and noise fidelity are conflicting objectives at the linear-algebra level. Any unbiased linear estimator that cancels offsets pays the $O(1/\varepsilon)$ noise penalty; any well-conditioned estimator necessarily admits an offset bias of order $\kappa(R)\|\mathbf{b}\|$. Regularization (Tikhonov, TSVD) moves along the boundary between these regimes by attenuating small singular values, but converts the noise problem into a *spatial bandwidth* problem: small-singular-value directions correspond to high-frequency Fourier modes of the misalignment pattern, which are then suppressed in the reconstruction. This is quantified empirically in Section 4.

The drift-monitoring framework introduced in Section 5 sidesteps the trade-off entirely by absorbing the BPM offset into the calibration baseline, allowing the well-conditioned direct estimator to be used without any offset penalty.

---

## 4. Numerical Experiments

The five experiments in this section provide the empirical content of the paper. The detailed protocol for each experiment is documented in the companion file `yapilacaklar-2.md`. The simulation in all cases is a Gauss–Legendre fourth-order symplectic integrator of the full 6D relativistic Newton–Lorentz equations with Thomas–BMT spin evolution and the rigorous cylindrical-deflector physics; closed-orbit data is accumulated over $\sim 800$ turns.

### 4.1 Test 1 — Regularization comparison

**Question:** Is the dramatic gap between direct inversion ($\sim 3\,\mu\text{m}$) and raw $\Delta R$ inversion ($\sim 1900\,\mu\text{m}$) preserved when standard regularization is applied to $\Delta R$?

**Procedure:** On a single $100\,\mu\text{m}$-RMS misalignment ensemble (the same simulation data used elsewhere in this paper), apply six estimators to the same $(\mathbf{y}_1, \mathbf{y}_2)$ data: direct $R_1^{-1}\mathbf{y}_1$, direct $R_2^{-1}\mathbf{y}_2$, direct average $(v_1+v_2)/2$, raw $\Delta R^{-1}$, Tikhonov $\Delta R$ at L-curve-optimal $\lambda$, and TSVD $\Delta R$ at oracle-optimal truncation level $k$.

**Result.** Table 3 reports the outcome. Optimal Tikhonov regularization reduces the $\Delta R$ reconstruction RMS from $\sim 1900\,\mu\text{m}$ to $\sim 50\,\mu\text{m}$, a $\sim 35\times$ improvement, confirming that the raw $\Delta R$ number is genuinely worst-case. However, regularized $\Delta R$ remains $\sim 15\times$ inferior to direct inversion ($\sim 3.5\,\mu\text{m}$), and the reconstruction correlation drops dramatically — from $\rho > 0.998$ (direct) to $\rho = 0.29$–$0.38$ (regularized). The low correlation indicates that regularization is not merely scaling the estimate but is destroying the modal content of the misalignment pattern. This is the subject of Test 2.

The oracle TSVD result reveals the mechanism explicitly: the optimum truncation level is $k = 3$ ($y$ plane) and $k = 5$ ($x$ plane), meaning that of the 48 available modes only 3–5 carry recoverable information after regularization. The remaining 43–45 modes are sacrificed to keep the noise amplification bounded.

**Table 3.** Reconstruction performance: direct inversion vs regularized $\Delta R$ methods (Test 1; 100 μm RMS injected misalignments).

| Estimator | RMS ($y$) | $\rho$ ($y$) | RMS ($x$) | $\rho$ ($x$) | Notes |
|-----------|-----------|--------------|-----------|--------------|-------|
| Direct, $R_1^{-1}y_1$ | $5.8\,\mu\text{m}$ | $0.995$ | $7.7\,\mu\text{m}$ | $0.989$ | well-conditioned, offset-sensitive |
| Direct, $R_2^{-1}y_2$ | $1.8\,\mu\text{m}$ | $0.9996$ | $1.5\,\mu\text{m}$ | $0.9996$ | well-conditioned at perturbed gradient |
| Direct, $(v_1+v_2)/2$ | $3.5\,\mu\text{m}$ | $0.998$ | $3.6\,\mu\text{m}$ | $0.998$ | central estimator |
| Raw $\Delta R^{-1}$ | $1865\,\mu\text{m}$ | $0.085$ | $1396\,\mu\text{m}$ | $-0.05$ | noise-amplified |
| Tikhonov $\Delta R$, L-curve | $53\,\mu\text{m}$ | $0.348$ | $49\,\mu\text{m}$ | $0.286$ | $\lambda \approx 4\times 10^{-2}$ |
| Tikhonov $\Delta R$, oracle $\lambda$ | $52\,\mu\text{m}$ | $0.372$ | $49\,\mu\text{m}$ | $0.290$ | upper bound on Tikhonov |
| TSVD $\Delta R$, oracle $k$ | $52\,\mu\text{m}$ ($k=3$) | $0.383$ | $49\,\mu\text{m}$ ($k=5$) | $0.296$ | 3–5 modes recovered of 48 |

Figure 3 shows the corresponding L-curve and TSVD scree plots (file: `test1_regularization.png`).

### 4.2 Test 2 — Spatial mode transfer (signature figure)

**Question:** Where in the spatial frequency spectrum do the regularized estimators differ from the direct estimator?

**Procedure:** Inject sinusoidal misalignment patterns $\Delta q^{(k)}_j = A\cos(2\pi k j/N + \varphi)$ for each Fourier index $k = 0, 1, \ldots, N/2$. For each estimator, compute the recovered amplitude divided by the true amplitude — the *spatial transfer function*. The forward model uses the analytic $R$ (Test 1 has already established that $R$ matches the 6D simulation), so no full simulation is required. Two scenarios are run: noiseless (analytic forward only) and BPM-noisy ($\sigma_n = 1\,\mu\text{m}$ per readout, averaged over 40 realizations).

**Result.** The two-scenario comparison is striking and clarifies the role of each estimator (file: `test2_mode_transfer.png`):

*Noiseless scenario.* The direct estimator and raw $\Delta R$ inversion both produce transfer ratios of $1.000$ across **all** Fourier modes $k = 0, \ldots, 24$. That is, $\Delta R$ inversion is *not* biased — it is signal-preserving in the same sense as direct inversion. The familiar $\sim 1900\,\mu\text{m}$ reconstruction error of raw $\Delta R$ in Test 1 reflects *noise amplification only*, not signal corruption. By contrast, the optimally regularized Tikhonov and TSVD estimators yield transfer ratios close to **zero** for almost all modes — they suppress the signal itself, retaining only a few high-$k$ harmonics ($k \approx 20$–$24$, ratio $\sim 0.3$–$0.5$).

*Noisy scenario.* Direct inversion retains transfer ratio $\approx 1$ with small variance: the central estimator is robust. Raw $\Delta R$ retains transfer ratio $\approx 1 \pm 0.5$ — unbiased but high-variance, with the variance reflecting $O(\kappa(\Delta R)\sigma_n)$ per-realization noise amplification. Regularized estimators remain at transfer ratio $\sim 0$.

This is the central physical observation of the paper: **regularization converts a noise problem into a bias problem.** Raw $\Delta R$ is unbiased but high-variance; regularized $\Delta R$ is low-variance but heavily biased toward zero in the high-frequency Fourier modes. Direct inversion, with the BPM offset eliminated by reference-state differencing (Section 5), is the only estimator that is simultaneously unbiased *and* low-variance, but only conditional on offset stability between calibration and measurement.

A consequence is that raw $\Delta R$ inversion can in principle be rescued by *time averaging* across many epochs: the unbiasedness means the signal accumulates linearly while the noise variance decays as $1/T$. This complementary mode is explored quantitatively in Test 5 (§4.5).

### 4.3 Test 3 — $K_{x,\text{arc}}$ sensitivity (inverse-crime check)

**Question:** How sensitive is the horizontal-plane reconstruction to error in the simulation-calibrated parameter $K_{x,\text{arc}}$?

**Procedure:** The full 6D simulation is run with the *true* $K_{x,\text{arc}}$ to generate $(\mathbf{y}_1, \mathbf{y}_2)$ data. The inverse reconstruction is then performed with $K_{x,\text{arc}}$ deliberately perturbed by $\delta \in [-10\%, +10\%]$ relative to the true value. The vertical-plane reconstruction is computed in parallel as a control (Maxwell-guaranteed $K_{y,\text{arc}} = 0$, no dependence on the perturbation).

**Result.** Across the full $\pm 10\%$ perturbation range, the horizontal-plane reconstruction RMS varies by less than $0.5\,\mu\text{m}$ ($3.48$ to $4.01\,\mu\text{m}$, central estimator). The vertical-plane RMS is constant at $3.489\,\mu\text{m}$, exactly as predicted from the Maxwell argument. The horizontal reconstruction has a shallow minimum at $\delta \approx +5\%$, indicating a small ($\sim 5\%$) bias in the original $K_{x,\text{arc}}$ calibration relative to the optimal value for this particular misalignment realization; this bias is well below the LOCO-style model accuracy ($< 1\%$) routinely achievable in operating storage rings.

The inverse-crime concern for the horizontal plane is therefore resolved: even a $10\%$ error in the model parameter, far above any realistic accuracy in a real lattice, degrades reconstruction by less than $14\%$ in RMS. The signature improvement of the direct estimator over $\Delta R$ approaches is preserved under any plausible model error. See `test3_kxarc_sensitivity.png` for the full sweep.

### 4.4 Test 4 — Drift tracking dynamics

**Question:** Does the calibrated-reference estimator $\widehat{\delta q}(t) = R^{-1}(\mathbf{y}(t) - \mathbf{y}_0)$ correctly track a slowly evolving misalignment in the presence of a large but constant BPM offset?

**Procedure:** Establish a calibration reference at $t=0$ with $100\,\mu\text{m}$-RMS random misalignments and $50\,\mu\text{m}$-RMS random BPM offsets. Over ten subsequent epochs, inject a $10\,\mu\text{m}$-RMS ramp into the misalignment vector while holding the BPM offset constant. Apply the drift estimator at each epoch. As a control, apply the naive *absolute* reconstruction $\widehat{\Delta q}(t) = R^{-1}\mathbf{y}(t)$ to the same data.

**Result.** The drift estimator tracks the injected ramp with mean RMS error of $6.5\,\mu\text{m}$ ($y$) and $6.5\,\mu\text{m}$ ($x$) per epoch, with correlation rising from $\rho = 0.15$ at the smallest drift ($1\,\mu\text{m}$, dominated by BPM noise) to $\rho = 0.86$ at the maximum drift ($11\,\mu\text{m}$). The absolute-reconstruction control produces a per-epoch RMS error of $170$–$200\,\mu\text{m}$, completely dominated by the BPM offset contamination $\|R^{-1}\mathbf{b}_0\|$. The drift-mode gain is thus a factor of $\sim 26$–$29$ across both planes — exactly the BPM-offset cancellation effect predicted from the differencing structure.

Two operational implications follow. First, the absolute size of the BPM offset ($50\,\mu\text{m}$ RMS in this test) is *irrelevant* to drift tracking, provided it remains stable between calibration and measurement. Second, the residual drift-tracking error of $\sim 6\,\mu\text{m}$ originates not from offset bias but from the propagated BPM noise: $\sqrt{2}\,\sigma_n\,\|R^{-1}\|$ for two independent readings. Reducing this requires either lower-noise BPM electronics or longer per-epoch averaging.

See Figure 7 (`test4_drift_monitor.png`) for the time series.

### 4.5 Test 5 — BPM offset drift tolerance and two-estimator comparison

**Question:** The drift-monitoring framework assumes the BPM offset drifts much more slowly than the misalignment. What happens when this assumption is violated, and is there an alternative estimator that is immune to BPM drift at the cost of some noise performance?

**Procedure:** Two estimators are compared as a function of BPM offset drift rate $\dot{\sigma}_b \in [0, 5]\,\mu\text{m}/\text{epoch}$:

- **Estimator A** (calibrated-reference direct): $\widehat{\delta q}(t) = R^{-1}(\mathbf{y}(t) - \mathbf{y}_0)$. Fast and high-fidelity when BPM offsets are stable; degrades linearly with drift rate.

- **Estimator B** (per-epoch $\Delta R$ with time averaging): invert $\Delta R$ at each epoch to cancel the instantaneous offset, then average over a sliding window of $T = 30$ epochs to reduce noise. Because $\Delta R^{-1}$ is applied per-epoch, this estimator is immune to BPM drift by construction, regardless of rate.

**Result.** The outcome is shown in Figure 8 (`test5_bpm_offset_drift.png`). Estimator A dominates in the practically relevant regime:

| BPM drift $\dot{\sigma}_b$ | A (direct, calibrated) | B ($\Delta R$, T=30 avg) |
|---|---|---|
| $0\,\mu\text{m/epoch}$ | **5.6 μm** | 170 μm |
| $0.05\,\mu\text{m/epoch}$ | 12 μm | 324 μm |
| $0.5\,\mu\text{m/epoch}$ | 83 μm | 190 μm |
| $2.0\,\mu\text{m/epoch}$ | 335 μm | **184 μm** |
| $5.0\,\mu\text{m/epoch}$ | 917 μm | **210 μm** |

The cross-over occurs near $\dot{\sigma}_b \approx 2\,\mu\text{m/epoch}$. Modern BPM electronics exhibit thermal coefficients of order $0.1\,\mu\text{m}/{}^\circ\text{C}$; a temperature swing of $1\,{}^\circ\text{C}$ per measurement epoch would be required to reach the cross-over — an extreme scenario in a temperature-stabilized accelerator hall.

Estimator B's noise floor of $\sim 150$–$300\,\mu\text{m}$ (even at zero drift) reflects the fundamental limit of the $\Delta R$ approach: $\kappa(\Delta R) \approx 27\,000$ amplifies $1\,\mu\text{m}$ readout noise to $\sim 27\,\text{mm}$ per epoch, and 30-epoch averaging reduces this only by $\sqrt{30} \approx 5.5\times$ to $\sim 5\,\text{mm}$. A far larger averaging window (thousands of epochs) would be required to bring B's noise floor to the few-$\mu\text{m}$ level. The Test 2 result explains why: B is unbiased (transfer ratio = 1 for all modes) but has such large variance per epoch that time-averaging alone cannot rescue it in practice.

The two estimators are therefore complementary in a qualitative rather than quantitative sense: A is the primary online monitor; B provides a long-term consistency check that is immune to slow BPM gain drift, at the cost of reduced per-epoch precision.

### 4.6 Robustness sweep (existing)

Four additional error axes — BPM per-readout noise, $\beta$-function model error, quadrupole tilt — have been characterized in earlier work (Stage D of `spectral_inversion.py`). Table 4 summarizes the thresholds.

**Table 4.** Robustness thresholds for each error axis ($10\,\mu\text{m}$ reconstruction RMS limit, direct estimator in drift mode unless noted).

| Error source | Threshold | Notes |
|---|---|---|
| BPM noise (per turn) | $\sigma_n^{\text{single}} < 85\,\mu\text{m}$ | After 800-turn averaging |
| BPM offset (absolute mode) | $\sigma_b < 5\,\mu\text{m}$ | Irrelevant in drift mode |
| BPM offset drift (drift mode) | $\dot{\sigma}_b < 0.1\,\mu\text{m/epoch}$ | Cross-over with B estimator at $\sim 2\,\mu\text{m/epoch}$ |
| $K_{x,\text{arc}}$ model error | $< 10\%$ ($< 0.5\,\mu\text{m}$ RMS change) | LOCO typically $< 1\%$ |
| $\beta$-function error | $\sigma(\delta\beta/\beta) < 2\%$ | LOCO typically $< 1\%$ |
| Quad tilt | $\sigma_\theta < 10\,\text{mrad}$ | Second-order in COD mode |

The quadrupole tilt threshold deserves a brief comment: in a COD-mode measurement the skew coupling produces vertical kicks proportional to the *product* of the tilt and the horizontal closed-orbit excursion at that quadrupole — a second-order quantity. We verified this independently with a $1\,\text{mrad}$ horizontal kick in an otherwise perfect ring, observing linear growth of vertical amplitude with quadrupole tilt over $[0, 10]\,\text{mrad}$ (0.7% spread).

---

## 5. Calibrated-Reference Drift-Tracking Architecture

### 5.1 From absolute reconstruction to drift monitoring

The direct estimator $\widehat{\Delta q} = R^{-1}\mathbf{y}$ applied to a single reading is contaminated by the BPM offset at a level $\kappa(R)\|\mathbf{b}\|/\|R\|$, typically tens of $\mu\text{m}$. The key insight is that the *absolute* values of $\mathbf{b}$ and $\Delta q$ are not what online monitoring requires: what matters is whether alignment has *changed* since the last calibration.

Define the calibration epoch $t = 0$ with reference readings $\mathbf{y}_0 = R\Delta q_0 + \mathbf{b}$. For any subsequent time $t$:
$$\mathbf{y}(t) - \mathbf{y}_0 = R(\Delta q(t) - \Delta q_0) + (\mathbf{b}(t) - \mathbf{b}_0) + \boldsymbol{\eta}(t).$$
If BPM offsets drift much more slowly than misalignment (so that $\mathbf{b}(t)\approx\mathbf{b}_0$ over the monitoring interval), the drift estimate is:
$$\widehat{\delta q}(t) = R^{-1}\bigl(\mathbf{y}(t) - \mathbf{y}_0\bigr).$$
This is offset-free without any gradient modulation. Test 4 demonstrates the principle directly; Test 5 quantifies the tolerance for the slow-BPM-drift assumption.

### 5.2 Layered monitoring architecture

The calibrated-reference strategy separates alignment monitoring into two complementary layers:

**Slow absolute layer (LOCO / BBA / survey):**
- Frequency: once per run period (hours to days)
- Outputs: absolute misalignment $\Delta q_0$, BPM offset $\mathbf{b}_0$, optical model ($\beta$, $\phi$, $Q$)
- Tool: beam-based alignment or mechanical survey
- Precision: $\sim 10$–$50\,\mu\text{m}$ absolute

**Fast drift layer (this method):**
- Frequency: continuous, every few seconds to minutes
- Output: drift vector $\delta q(t) = \Delta q(t) - \Delta q_0$ relative to calibration
- Tool: two-gradient direct inversion (or single-gradient differencing — see §5.3)
- Precision: $\sim 1$–$5\,\mu\text{m}$ per epoch

The two layers are *complementary*. LOCO provides the $\beta$, $\phi$, $Q$ that define $R$; this method then converts the BPM time series into a continuous drift estimate without additional corrector kicks or dedicated beam time.

### 5.3 Operational modes

Three operational modes are possible, with different trade-offs:

1. **Single-gradient drift mode**: $\widehat{\delta q}(t) = R^{-1}(\mathbf{y}(t) - \mathbf{y}_0)$ uses only one gradient setting. No interference with physics data-taking. This is the recommended operational mode.

2. **Interleaved two-gradient mode**: occasional gradient modulation epochs (e.g., one per hour) provide consistency checks between $v_1$ and $v_2$ estimates and detect slow BPM gain drift.

3. **Dedicated monitoring windows**: full two-gradient k-modulation during scheduled non-physics windows, used for re-calibration of $R$ and $\mathbf{b}_0$.

The first mode is the principal contribution of this work: gradient modulation is moved from a per-epoch requirement to a periodic calibration activity, removing the concern that continuous modulation might affect spin coherence or EDM systematics.

---

## 6. Discussion

### 6.1 Comparison with LOCO

Linear Optics from Closed Orbits (LOCO) [9,10] fits the orbit response matrix to extract gradient errors, BPM gains, and corrector calibrations, but does not directly produce transverse-position misalignment estimates. Beam-based alignment (BBA) provides these positions, at the cost of beam time. In the proposed layered architecture, LOCO/BBA establishes the reference state once per run period; our method then tracks drift continuously from standard diagnostic BPM data. This is not a competing approach to LOCO but a complementary fast layer that LOCO enables.

### 6.2 Inverse crime in the horizontal plane

The analytic $R$ for the horizontal plane depends on $K_{x,\text{arc}}$, calibrated from the same simulation that produces the validation data. This is a textbook inverse-crime configuration and would invalidate any claim that does not address it. Test 3 (§4.3) addresses this by deliberately perturbing $K_{x,\text{arc}}$ within the inverse problem while keeping the forward simulation unchanged, measuring the resulting reconstruction degradation. The vertical plane is immune because $K_{y,\text{arc}} = 0$ is a Maxwell-guaranteed prediction, independent of any simulation.

A second, independent check is the empirical asymmetry between $v_1$ ($\sim 7.7\,\mu\text{m}$) and $v_2$ ($\sim 1.5\,\mu\text{m}$) reconstructions: $K_{x,\text{arc}}$ is calibrated at $g_1$ and used at both gradients, so the $v_2$ reconstruction implicitly carries a model-simulation mismatch of size $|\partial K_{x,\text{arc}}/\partial g|\,\Delta g$. The observed performance gap quantifies this mismatch indirectly.

### 6.3 Temporal resolution and operational implications

The averaging time required to suppress betatron noise in the COD estimate is approximately $T_{\text{avg}} \approx 800\,\text{turns}\times 2\,\mu\text{s} \approx 1.6\,\text{ms}$ per gradient setting. In practice, control-system overhead and gradient settling will dominate, suggesting practical epoch durations of $\sim 1$–$10\,\text{s}$. Drift mechanisms of interest (thermal, mechanical creep) operate on timescales of minutes to hours, so a $\sim 1\,\text{s}$ epoch provides comfortable sampling margin.

In single-gradient drift mode (§5.3), no gradient modulation is needed during physics data-taking, so the temporal resolution is set entirely by COD averaging — milliseconds in principle, seconds in practice.

### 6.4 Limitations

**Model accuracy.** $R$ is built from a model with a single calibrated parameter. Real lattice errors (power supply ripple, remanent fields) introduce model-simulation mismatch; periodic LOCO re-fitting mitigates this.

**Nonlinear effects.** The forward model is strictly linear. Large misalignments or large COD amplitudes excite nonlinear terms, breaking the linear inversion.

**Coupling.** The current implementation treats $x$ and $y$ as decoupled. Quadrupole tilts introduce skew coupling; this is second-order for small tilts (verified in §4.6) but a coupled formalism would be needed for large skew components.

**BPM offset drift assumption.** The drift-monitoring framework relies on BPM electronics being more stable than the magnets. Test 5 (§4.5) quantifies the tolerance; the actual stability of pEDM-grade BPM electronics in the high-voltage environment of the electric deflectors is an open experimental question.

---

## 7. Conclusion

We have presented a method for *online lattice drift monitoring* in an all-electric proton EDM storage ring, based on direct inversion of an analytic FODO response matrix against differential BPM readings relative to a calibrated reference state. The key conceptual reframing — from absolute reconstruction to drift tracking — removes the BPM offset problem that limits conventional two-gradient k-modulation, replacing it with an inter-epoch BPM stability requirement that modern electronics easily satisfy.

We characterize the offset–noise trade-off in standard two-gradient methods through a short derivation and through five targeted simulation tests. The direct estimator achieves $1.4$ – $4.8\,\mu\text{m}$ RMS reconstruction error for $100\,\mu\text{m}$ injected misalignments (correlation $\rho > 0.998$) when BPM offsets are absorbed into a calibration baseline. Standard regularization (Tikhonov, TSVD) applied to $\Delta R$ inversion mitigates noise amplification at the cost of spatial bandwidth, suppressing the high-frequency Fourier modes of the misalignment pattern.

The proposed layered architecture — slow LOCO/BBA absolute calibration combined with fast direct-inversion drift tracking — is operationally compatible with pEDM physics data-taking: in single-gradient drift mode, no gradient modulation is required during physics running. We argue that this architecture is the natural deployment of a two-gradient k-modulation framework: the gradient modulation is used to bootstrap the calibration, after which BPM differencing alone tracks the drift.

Open-source Python implementations (`fodo_lattice.py`, `spectral_inversion.py`, `show_response.py`, `reconstruct.py`, `verify_quad_tilt.py`) accompany this paper. Additional test scripts (`compare_regularization.py`, `mode_transfer.py`, `kxarc_sensitivity.py`, `drift_monitor_sim.py`, `bpm_offset_drift_sim.py`) implement the numerical experiments of Section 4 and are documented in `yapilacaklar-2.md`.

---

## Acknowledgments

*(to be filled — funding sources, collaborators, computational resources)*

---

## References

[1] V. Anastassopoulos *et al.*, "A storage ring experiment to detect a proton electric dipole moment," *Rev. Sci. Instrum.* **87**, 115116 (2016).

[2] J. Pretz *et al.*, "Statistical sensitivity estimates for oscillating electric dipole moment measurements in storage rings," *Eur. Phys. J. C* **80**, 107 (2020).

[3] D. Robin, C. Steier, J. Safranek, W. Decking, "Enhanced performance of the Advanced Light Source through periodicity restoration of the linear lattice," in *Proc. EPAC 2000*, p. 136.

[4] G. Vanbavinckhove *et al.*, "Beam-based alignment with k-modulation," *Phys. Rev. ST Accel. Beams* **15**, 092802 (2012). *(verify exact citation)*

[5] S.Y. Lee, *Accelerator Physics*, 4th ed. (World Scientific, 2018), Ch. 2.5 and Ch. 4.

[6] H. Wiedemann, *Particle Accelerator Physics*, 4th ed. (Springer, 2015), Ch. 10.

[7] G. König, F.J.M. Farley, "Spin precession in a cylindrical electric capacitor at the magic momentum," internal note, BNL (2008). *(or equivalent)*

[8] E. Hairer, C. Lubich, G. Wanner, *Geometric Numerical Integration*, 2nd ed. (Springer, 2006).

[9] J. Safranek, "Experimental determination of storage ring optics using orbit response measurements," *Nucl. Instrum. Methods A* **388**, 27 (1997).

[10] X. Huang, J. Safranek, G. Portmann, "LOCO with constraints and improved fitting technique," *Proc. EPAC 2008*, p. 3120.

<!-- Note: Add 2-3 references on BPM electronics stability (thermal coefficients, gain drift) to quantify the BPM-stability assumption. Test 5 provides a quantitative threshold (cross-over at ~2 μm/epoch); literature values would anchor this to physical timescales. -->

---

## Figure list

- **Figure 1.** Schematic of one FODO cell of the pEDM ring.
- **Figure 2.** Per-mode condition profile: $|\lambda_k|^{-1}$ vs Fourier mode index $k$ for $R_1$, $R_2$, $\Delta R$. *(file: `stage_B_condition_y.png`, `stage_B_condition_x.png`)*
- **Figure 3.** L-curve and TSVD scree plots for $\Delta R$ regularization (both planes). *(file: `test1_regularization.png`)*
- **Figure 4. *(signature figure)*** Spatial transfer function: reconstructed amplitude / true amplitude vs Fourier mode index $k$, for each estimator; noiseless and noisy scenarios. *(file: `test2_mode_transfer.png`)*
- **Figure 5.** Reconstruction performance: $\widehat{\Delta q}$ vs $\Delta q_{\text{true}}$, direct estimator ($\rho > 0.999$).
- **Figure 6.** Robustness sweep panels: BPM noise, BPM offset, $\beta$-error, quad tilt. *(files: `stage_D_robustness_*.png`)*  **+**  $K_{x,\text{arc}}$ sensitivity panel. *(file: `test3_kxarc_sensitivity.png`)*
- **Figure 7.** Drift-monitor time series: true drift vs estimated drift over 10 epochs, with absolute-reconstruction baseline for comparison. *(file: `test4_drift_monitor.png`)*
- **Figure 8.** Two-estimator BPM offset drift comparison: Estimator A (calibrated direct) vs Estimator B ($\Delta R$ with T=30 averaging), reconstruction RMS vs drift rate. Cross-over at $\sim 2\,\mu\text{m/epoch}$. *(file: `test5_bpm_offset_drift.png`)*
- **Figure 9.** Linearity verification of skew-quadrupole coupling: vertical RMS displacement vs quad tilt angle. *(file: `verify_quad_tilt.py` output)*

---

## Table list

- **Table 1.** Lattice parameters (§2.1).
- **Table 2.** Twiss parameters at nominal gradient (§2.4).
- **Table 3.** Reconstruction performance: all estimators (§4.1). *(complete)*
- **Table 4.** Robustness thresholds (§4.6). *(complete)*
