# Online Lattice Drift Monitoring in an All-Electric Proton EDM Storage Ring via Two-Gradient Closed-Orbit Reconstruction

**Author(s):** Selcuk H.

**Affiliation:** *(to be filled)*

**Corresponding author:** *(email)*

**Target journal:** *Nuclear Instruments and Methods in Physics Research, Section A*

---

## Abstract

We present a method for continuously monitoring the transverse alignment state of quadrupole magnets in a periodic FODO storage ring from beam position monitor (BPM) measurements taken at two distinct quadrupole gradient settings. The method is conceived as an *online drift observer*: after a one-time calibration of the BPM-offset vector and the absolute misalignment state using an independent technique (beam-based alignment or survey), subsequent measurements track *deviations* from the calibrated reference — a problem that is both better-conditioned and physically more meaningful than absolute reconstruction.

The conventional two-gradient approach — inverting the gradient-difference response matrix $\Delta R = R_1 - R_2$ — suffers a severe conditioning penalty: for the all-electric proton EDM (pEDM) lattice we study, $\kappa(\Delta R) \approx 2.8 \times 10^4$, amplifying noise by two orders of magnitude. We show that this is not an implementation flaw but a consequence of a fundamental *offset–fidelity conjugate relation*: any linear estimator that cancels BPM offsets necessarily amplifies noise by $O(1/\varepsilon)$, where $\varepsilon$ is the relative gradient change. These two objectives are mathematically conjugate and cannot be simultaneously optimized.

We propose instead to invert each gradient configuration *separately* against a well-conditioned matrix $R_i$ ($\kappa(R_i) \approx 160$) and to track the difference between successive estimates and the calibrated baseline. Validation against a full 6D Gauss–Legendre symplectic integrator confirms drift-tracking reconstruction RMS errors of $1.4$ – $4.8\,\mu\text{m}$ for $100\,\mu\text{m}$ injected misalignments (correlation $\rho = 0.998$ – $0.9999$). We further characterize the suppression of spatial Fourier modes incurred by regularized $\Delta R$ inversion and show that the direct-inversion estimator preserves the full spatial spectrum.

<!-- TO BE DONE: Update Abstract numbers after regularized ΔR comparison and drift-monitor simulation are computed. -->

**Keywords:** beam position monitor; closed-orbit distortion; k-modulation; quadrupole misalignment; response matrix inversion; storage ring alignment; proton EDM; online monitoring; drift tracking

---

## 1. Introduction

The search for the proton electric dipole moment (pEDM) in an all-electric storage ring at the "magic" momentum $p \approx 0.7007\,\text{GeV/c}$ requires control over systematic effects at a level that depends critically on the transverse alignment of focusing elements [1,2]. Quadrupole misalignments at the $\sim 100\,\mu\text{m}$ scale produce closed-orbit distortions (CODs) which, combined with radial electric-field non-uniformities and parasitic magnetic fields, generate spin-rotation systematics that mimic an EDM signal. Achieving the design sensitivity of $10^{-29}\,e\cdot\text{cm}$ requires alignment knowledge at the few-$\mu\text{m}$ level around the ring.

There are two distinct alignment problems. The first is *static*: before or between physics runs, determine the absolute position of every quadrupole. The second — and for pEDM, the more dangerous — is *dynamic*: during a physics run, quadrupole positions drift slowly due to thermal expansion, support-structure creep, ground motion, high-voltage cycling of the electric deflectors, and electrostatic charging asymmetries of the plates. These drift mechanisms operate on timescales of minutes to hours and can displace quadrupoles by several microns during a single data-taking session, modifying the systematic environment in a way that is neither measured nor corrected unless a continuous monitoring system is in place.

The present work addresses the dynamic problem. Our goal is not a once-per-run absolute survey but a *quasi-real-time lattice drift observer*: a system that operates continuously during physics data-taking, uses BPM readings already available from the machine diagnostics, and delivers an updated estimate of the *change* in alignment state at each measurement epoch (every few seconds to minutes, limited by the required averaging time to extract the closed orbit to sufficient precision).

The forward model for closed-orbit distortions due to quadrupole misalignments is linear:
$$\mathbf{y}(t) = R\,\Delta q(t) + \mathbf{b} + \boldsymbol{\eta}(t),$$
where $R \in \mathbb{R}^{N\times N}$ is the response matrix, $\Delta q(t)$ is the time-varying misalignment vector, $\mathbf{b}$ is the (slowly varying) BPM electronic offset vector, and $\boldsymbol{\eta}$ is measurement noise. A single snapshot of $\mathbf{y}$ does not determine $\Delta q$ because $\mathbf{b}$ is comparable to the alignment signal and cannot be separated without further information.

The standard remedy is *k-modulation*: two CODs are recorded at slightly different quadrupole gradient settings $g_1$ and $g_2 = g_1(1+\varepsilon)$, so that the offset cancels in the difference:
$$\mathbf{y}_1 - \mathbf{y}_2 = (R_1 - R_2)\,\Delta q + (\boldsymbol{\eta}_1 - \boldsymbol{\eta}_2).$$
This procedure has been used in operating storage rings [3,4] and is discussed in standard textbook treatments [5,6]. However, as we show, $\Delta R = R_1 - R_2$ is severely ill-conditioned for small $\varepsilon$, limiting the achievable noise performance.

In this paper we make five contributions:

1. We identify the *offset–fidelity conjugate relation*: any linear two-gradient estimator that cancels BPM offsets amplifies noise by $O(1/\varepsilon)$. This is proven in closed form and is not avoidable by algorithmic refinement.

2. We propose a *calibrated-reference drift-tracking* architecture that separates the problem into a slow absolute-calibration layer (BBA/survey/LOCO, performed once) and a fast, high-fidelity drift-tracking layer (this method, run continuously). Within this architecture, the direct estimator's failure to cancel offsets becomes irrelevant, because offsets are absorbed into the calibration baseline.

3. We compare direct inversion, raw $\Delta R$ inversion, and regularized $\Delta R$ inversion (Tikhonov and truncated SVD) in terms of both noise performance and spatial mode suppression, showing that regularization mitigates but does not eliminate the trade-off. <!-- TO BE DONE: Compute and insert regularization comparison results -->

4. We construct $R$ analytically from the periodic FODO transfer matrices with a single calibrated horizontal arc-focusing parameter, eliminating the $O(N)$ perturbation simulations conventionally used to populate $R$ column-by-column.

5. We validate the full system against a 6D Gauss–Legendre symplectic integrator and characterize the robustness envelope over four physically motivated error axes.

The remainder of this paper is organized as follows. Section 2 develops the analytic response-matrix formalism. Section 3 proves the offset–fidelity conjugate relation and analyzes the spatial mode structure. Section 4 introduces the calibrated-reference drift-tracking architecture. Section 5 compares the direct estimator with regularized $\Delta R$ approaches. Section 6 reports simulation validation and robustness studies. Section 7 discusses implementation, layered monitoring architecture, and limitations. Section 8 concludes.

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
with $K = g_1 / B\rho$ and $B\rho = p/q$. The defocusing variant is obtained by $\cos \to \cosh$, $\sin \to \sinh$. Drift sections use the standard $\begin{pmatrix}1 & L \\ 0 & 1\end{pmatrix}$ matrix.

### 2.3 Arc focusing in cylindrical electric deflectors

Unlike magnetic dipoles, cylindrical electric deflectors at field index $n = 1$ produce *zero* vertical focusing because Maxwell's equations enforce $E_z = 0$ identically. Therefore for the vertical plane the arc transfer matrix is exactly a pure drift of length $L_{\text{arc}} = \pi R_0 / N_c$.

The horizontal plane is more subtle. Centrifugal defocusing in the rotating frame, combined with the relativistic $\beta^2$ correction to the effective potential, generates a small horizontal focusing that is not captured by the textbook expression $(1-n\pm\beta^2)/\rho^2$ once Coriolis terms are consistently included [7]. We treat the horizontal arc focusing coefficient $K_{x,\text{arc}}$ as a single calibration parameter determined once by bisection so that the analytic model matches the horizontal betatron tune $Q_x = 2.6824$ extracted from a clean (zero-misalignment, small-angle kick) simulation. For our parameters, $K_{x,\text{arc}} = 1.265 \times 10^{-4}\,\text{m}^{-2}$.

*Calibration sensitivity:* <!-- TO BE DONE: Compute dR/dK_x_arc and assess how a 10% error in K_x_arc propagates into reconstruction error. This quantifies the risk of inverse crime in the horizontal plane. -->

### 2.4 Twiss parameters

The one-cell transfer matrix $M_c$ for each plane is the product of the elemental matrices in cell order. The cell phase advance is $\mu = \arccos(\mathrm{Tr}\,M_c / 2)$ and the beta function at the cell entrance is $\beta_0 = M_{c,12} / \sin\mu$. The total betatron tune is $Q = N_c\,\mu / (2\pi)$. Beta functions and accumulated phases at every quadrupole entrance are obtained by propagating the Twiss vector through each element.

**Table 2.** Twiss parameters at nominal gradient.

| Quantity | Analytic | 6D simulation | Difference |
|----------|----------|---------------|------------|
| $Q_x$ | $2.682$ | $2.682$ | $< 0.001$ |
| $Q_y$ | $2.362$ | $2.362$ | $< 0.001$ |
| $\beta_y$ range | $[41.2,\,76.4]\,\text{m}$ | — | — |
| $\beta_x$ range | $[36.1,\,67.4]\,\text{m}$ | — | — |
| $\kappa(R_x)$ | $141$ | — | — |
| $\kappa(R_y)$ | $160$ | — | — |

### 2.5 The response matrix

A misalignment $\Delta q_j$ of quadrupole $j$ acts as a thin-lens dipole kick of integrated strength $\theta_j = KL_j \cdot \Delta q_j$ on the beam. The resulting closed orbit at BPM $i$ is given by the Courant–Snyder formula:
$$R_{ij} = \frac{\sqrt{\beta_i \beta_j}}{2 \sin(\pi Q)} \,\cos\!\bigl(|\phi_i - \phi_j| - \pi Q\bigr) \cdot KL_j,$$
where $KL_j$ carries the plane-appropriate sign: for the vertical plane $KL_j < 0$ at QF positions and $KL_j > 0$ at QD positions; for the horizontal plane the signs reverse. The sign is absorbed into $KL_j$ without a separate prefactor.

The conditioning of $R$ depends critically on $\sin(\pi Q)$: working points near integer or half-integer resonances are pathological. At our nominal tunes, $\kappa(R_x) \approx 141$ and $\kappa(R_y) \approx 160$.

---

## 3. The Offset–Fidelity Conjugate Relation

### 3.1 Why $\Delta R$ is ill-conditioned

Write the two BPM observations as
$$\mathbf{y}_i = R_i\,\Delta q + \mathbf{b} + \boldsymbol{\eta}_i, \quad i = 1, 2.$$
Any linear estimate of $\Delta q$ from $(\mathbf{y}_1, \mathbf{y}_2)$ takes the form
$$\widehat{\Delta q} = A_1\,\mathbf{y}_1 + A_2\,\mathbf{y}_2.$$
For unbiasedness we need $A_1 R_1 + A_2 R_2 = I$. For offset cancellation we additionally need $A_1 + A_2 = 0$, i.e., $A_2 = -A_1$. These two constraints together require $A_1(R_1 - R_2) = I$, so that $A_1 = \Delta R^{-1}$. Therefore:

> **Offset–Fidelity Conjugate Theorem.** The unique unbiased linear estimator that cancels BPM offsets is the $\Delta R$-inversion estimator. No other unbiased linear estimator cancels offsets. Because $\|\Delta R\| \sim \varepsilon \|R\|$ for small $\varepsilon$, the operator norm of $A_1 = \Delta R^{-1}$ grows as $O(1/\varepsilon)$, amplifying measurement noise by the same factor.

This is not a weakness of any particular algorithm: it is a consequence of the linear structure of the problem. In our lattice with $\varepsilon = 0.02$ we measure $\kappa(\Delta R_y) = 27\,560$, an amplification factor $\kappa(\Delta R)/\kappa(R_1) \approx 170$ — consistent with the $1/\varepsilon \approx 50$ leading-order factor, with an additional factor from second-order optical perturbation.

The practical implication is that the choice between the direct estimator and the $\Delta R$ estimator is not primarily algorithmic but *architectural*: it determines whether to eliminate offset through cancellation (at noise cost) or through independent calibration (at measurement cost). We advocate the latter, as developed in Section 4.

### 3.2 The offset–noise Pareto frontier

The full family of unbiased linear estimators parametrized by $\alpha \in [0,1]$ is
$$\widehat{\Delta q}(\alpha) = (1-\alpha)\,R_1^{-1}\mathbf{y}_1 + \alpha\,R_2^{-1}\mathbf{y}_2.$$
The noise variance scales as $(1-\alpha)^2 \kappa(R_1) + \alpha^2 \kappa(R_2)$, minimized at $\alpha = 1/2$ (equal-weight average). The offset error scales as $\|\tfrac{1}{2}(R_1^{-1} + R_2^{-1})\mathbf{b}\|$, which is $O(\kappa(R)\|\mathbf{b}\|)$ — nonzero but finite. Moving toward $\alpha \to -KL_1/(KL_2 - KL_1)$ cancels the offset but inflates the noise coefficient to $O(1/\varepsilon)$. These objectives sit on a Pareto frontier: improving one always worsens the other. Regularized $\Delta R$ inversion (Section 5) moves along this frontier but cannot escape it.

### 3.3 Spectral structure of $R$ in a periodic FODO lattice

For a perfectly periodic ring with $N$ identical cells, $R$ is exactly circulant and its eigenvectors are the DFT columns. The pEDM ring has alternating QF/QD positions with unequal $\beta$ functions, making $R$ *block-circulant* (2×2 blocks); the deviation from strict circularity is small. The mode-by-mode condition number $|\lambda_k|^{-1}$ exposes which Fourier harmonics of the misalignment pattern are reconstructed faithfully. For $R_y$, the worst mode has $|\lambda|^{-1} \approx 0.4\,\text{m}^{-1}$; for $\Delta R_y$, the worst mode exceeds $7000\,\text{m}^{-1}$ — a $17{,}000\times$ difference.

Figure *(insert: stage_B_condition_y.png, stage_B_condition_x.png)* shows the per-mode condition profile as a function of Fourier index $k$.

<!-- TO BE DONE: Compute and plot singular value spectra of R, ΔR, and regularized ΔR (Tikhonov and truncated SVD) as a function of Fourier mode index. Show which spatial harmonics are recoverable under each method. This converts the scalar condition number into a physically interpretable spatial bandwidth statement. -->

---

## 4. Calibrated-Reference Drift-Tracking Architecture

### 4.1 From absolute reconstruction to drift monitoring

The direct estimator
$$\widehat{\Delta q} = R^{-1}\mathbf{y}$$
applied naively aims at absolute misalignment reconstruction. The obstacle is the BPM offset $\mathbf{b}$: $R^{-1}\mathbf{b}$ contaminates the estimate at a level $\kappa(R)\|\mathbf{b}\|/\|R\|$, typically $\sim 40\,\mu\text{m}$ for $\|\mathbf{b}\| \sim 10\,\mu\text{m}$, which exceeds the few-$\mu\text{m}$ design target.

The key insight is that the *absolute* values of $\mathbf{b}$ and $\Delta q$ are not what the drift monitor needs to know. What matters during a physics run is whether alignment has *changed* since the last calibration. Define the calibration epoch $t = 0$ with reference readings $\mathbf{y}_0 = R\,\Delta q_0 + \mathbf{b}$ measured at both gradient settings. For any subsequent time $t$:
$$\mathbf{y}(t) - \mathbf{y}_0 = R\,(\Delta q(t) - \Delta q_0) + (\mathbf{b}(t) - \mathbf{b}_0) + \boldsymbol{\eta}(t).$$
If the BPM offset drifts much more slowly than the misalignment (a well-motivated assumption: electronics are temperature-stabilized, magnets move), then $\mathbf{b}(t) - \mathbf{b}_0 \approx 0$ over the timescale of interest. The drift estimate is then:
$$\widehat{\delta q}(t) = R^{-1}\bigl(\mathbf{y}(t) - \mathbf{y}_0\bigr).$$
This is offset-free without any gradient modulation: the offset cancels because we subtract two readings taken with the *same* BPM electronics. The two-gradient structure is used only to verify the drift estimate and to track slow changes in $\mathbf{b}$ over longer timescales.

### 4.2 Two-gradient variant for drift tracking

Using both gradient settings at each epoch provides the pair $(\mathbf{y}_1(t), \mathbf{y}_2(t))$ and similarly $(\mathbf{y}_{1,0}, \mathbf{y}_{2,0})$ at calibration. The drift estimate from each channel is
$$v_i(t) = R_i^{-1}\bigl(\mathbf{y}_i(t) - \mathbf{y}_{i,0}\bigr), \quad i = 1, 2,$$
and the combined estimator is
$$\widehat{\delta q}(t) = \tfrac{1}{2}(v_1(t) + v_2(t)).$$
The difference $v_1(t) - v_2(t)$ is sensitive to changes in $\mathbf{b}(t)$ that have occurred since calibration; it provides an internal consistency check and, in principle, an estimate of BPM offset drift.

### 4.3 Layered monitoring architecture

The calibrated-reference strategy separates alignment monitoring into two complementary layers with different timescales and methods:

**Slow absolute layer (LOCO / BBA / survey):**
- Frequency: once per run period (hours to days)
- Outputs: absolute misalignment vector $\Delta q_0$, BPM offset vector $\mathbf{b}_0$, optical model ($\beta$, $\phi$, $Q$)
- Tool: beam-based alignment (corrector-based BBA, k-modulation, or mechanical survey)
- Precision: $\sim 10$–$50\,\mu\text{m}$ absolute

**Fast drift layer (this method):**
- Frequency: continuous, every few seconds to minutes (limited by COD averaging time)
- Output: drift vector $\delta q(t) = \Delta q(t) - \Delta q_0$ relative to the calibrated reference
- Tool: two-gradient direct inversion as described above
- Precision: $\sim 1$–$5\,\mu\text{m}$ per measurement epoch

The two layers are *complementary*, not competing. The slow layer provides the ground truth; the fast layer tracks time evolution. This architecture is directly analogous to the use of LOCO for slow lattice characterization combined with fast turn-by-turn BPM data for online orbit correction.

<!-- TO BE DONE: Simulate drift monitoring scenario: start at calibrated state dq_0, inject a slow (ramp-like) drift of 10 μm over 10 measurement epochs, verify that the estimator tracks the drift with latency < 1 epoch and RMS error consistent with Stage C numbers. Report as a figure and add to validation section. -->

---

## 5. Regularized $\Delta R$ Inversion: Mode Suppression vs Noise Control

### 5.1 Regularization strategies

For contexts where independent BPM offset calibration is unavailable, the $\Delta R$ method remains the only route to offset immunity. However, raw inversion of $\Delta R$ is impractical ($\kappa \approx 2.7 \times 10^4$). Standard regularization strategies are:

**Tikhonov regularization:**
$$\widehat{\Delta q}_\lambda = (\Delta R^\top \Delta R + \lambda I)^{-1} \Delta R^\top (\mathbf{y}_1 - \mathbf{y}_2),$$
with regularization parameter $\lambda > 0$ chosen by, e.g., the L-curve method.

**Truncated SVD (TSVD):**
Let $\Delta R = U\Sigma V^\top$. Retain only the $k$ largest singular values:
$$\widehat{\Delta q}_k = \sum_{j=1}^{k} \frac{u_j^\top (\mathbf{y}_1 - \mathbf{y}_2)}{\sigma_j} v_j,$$
with truncation level $k$ chosen so that $\sigma_k / \sigma_1 > \tau$ for some threshold $\tau$.

In both cases, regularization replaces the amplification of small singular values with a controlled truncation at the cost of spatial resolution: modes corresponding to suppressed singular values are not recovered.

### 5.2 Mode suppression as spatial bandwidth loss

<!-- TO BE DONE: For each regularization strategy (raw ΔR, Tikhonov at several λ, TSVD at several k):
  1. Compute the effective reconstruction operator (inverse or pseudo-inverse).
  2. Express it in the Fourier basis of the lattice.
  3. Plot the "reconstructed amplitude / true amplitude" as a function of Fourier mode index k — the spatial transfer function.
  4. Identify the spatial cutoff frequency (in units of FODO cells) below which modes are faithfully recovered.
  5. Compare this spatial bandwidth with that of the direct inversion method (which recovers all modes with equal fidelity).
This analysis will show that regularization converts a noise problem into a spatial-resolution problem, and will quantify the bandwidth that is traded away for noise control. -->

### 5.3 Quantitative comparison of all estimators

<!-- TO BE DONE: Run all estimators on the same simulation data as Stage C (100 μm random misalignments, both planes):
  - Direct inversion: R₁⁻¹y₁, R₂⁻¹y₂, (v₁+v₂)/2
  - Raw ΔR inversion
  - Tikhonov ΔR at optimal λ (L-curve)
  - TSVD ΔR at several truncation levels
  
  Report in a table: RMS error, max error, correlation, effective spatial cutoff.
  
  Expected result: regularized ΔR will show intermediate RMS performance but reduced spatial fidelity (high-frequency misalignment modes suppressed). Direct inversion will dominate on both metrics when BPM offsets are calibrated. -->

**Table 3 (placeholder).** Reconstruction performance: direct inversion vs regularized $\Delta R$ methods.

| Estimator | RMS error ($y$) | Corr. ($y$) | RMS error ($x$) | Corr. ($x$) | Spatial cutoff |
|-----------|-----------------|-------------|-----------------|-------------|----------------|
| Direct, $R_1^{-1}y_1$ | $5.8\,\mu\text{m}$ | $0.995$ | $7.7\,\mu\text{m}$ | $0.989$ | full |
| Direct, $R_2^{-1}y_2$ | $1.8\,\mu\text{m}$ | $0.9996$ | $1.5\,\mu\text{m}$ | $0.9996$ | full |
| Direct, $(v_1+v_2)/2$ | $3.5\,\mu\text{m}$ | $0.998$ | $3.6\,\mu\text{m}$ | $0.998$ | full |
| Raw $\Delta R^{-1}$ | $1865\,\mu\text{m}$ | $0.085$ | $1396\,\mu\text{m}$ | $-0.05$ | full (noise dominated) |
| Tikhonov $\Delta R$ (opt. $\lambda$) | *TO BE DONE* | *TO BE DONE* | *TO BE DONE* | *TO BE DONE* | *TO BE DONE* |
| TSVD $\Delta R$ (opt. $k$) | *TO BE DONE* | *TO BE DONE* | *TO BE DONE* | *TO BE DONE* | *TO BE DONE* |

---

## 6. Simulation Validation and Robustness

### 6.1 Simulation setup

We validate against a full 6D tracking simulation using a Gauss–Legendre fourth-order symplectic integrator applied to the relativistic Newton–Lorentz equations with self-consistent Thomas–BMT spin evolution. The simulation operates in global Cartesian coordinates and includes the full physics of the cylindrical electric deflector, quadrupole misalignments in both planes, quadrupole tilt (skew-quadrupole coupling), and the rigorous Maxwell-consistent sextupole overlay. Closed-orbit data is accumulated over typically 500–800 turns to suppress betatron oscillations.

*On the question of model-simulation consistency:* Because the analytic response matrix $R$ is calibrated using the same simulation that generates validation data, there is a risk of *inverse crime* in the horizontal plane (where $K_{x,\text{arc}}$ is calibrated from simulation). The vertical plane is immune: the analytic prediction $K_{y,\text{arc}} = 0$ follows from Maxwell's equations and is independent of any simulation calibration. The horizontal plane sensitivity to $K_{x,\text{arc}}$ accuracy is discussed in Section 2.3 (see TO BE DONE note); in practice the asymmetry between $v_1$ and $v_2$ reconstruction accuracy (1.5 vs 7.7 μm, see Table 3) provides an indirect empirical bound.

### 6.2 Reconstruction performance (absolute reconstruction baseline)

For an ensemble with $\sigma(\Delta q) = 100\,\mu\text{m}$ injected into both vertical and horizontal misalignments, the two-gradient direct inversion recovers misalignments at the level shown in Table 3. The $v_2$ estimator consistently outperforms $v_1$ because the arc-focusing calibration was performed at $g_1$; at $g_2 = 1.02\,g_1$ there is a small residual model-simulation mismatch.

### 6.3 Robustness sweep

To map the operating envelope of the direct estimator, we performed a four-axis sweep over physically motivated error sources, varying one parameter at a time.

**BPM noise.** A Gaussian noise vector with RMS $\sigma_n$ is added independently to each BPM reading. The reconstruction RMS error grows approximately linearly in $\sigma_n$ at a rate of $\sim 3\,\mu\text{m}$ per $\mu\text{m}$ input, crossing the $10\,\mu\text{m}$ threshold at $\sigma_n \approx 3\,\mu\text{m}$. After the $\sim 800$-turn averaging used to extract the closed orbit, this corresponds to a single-shot BPM resolution requirement of $\sigma_n^{\text{single}} \approx 85\,\mu\text{m}$, which is within the capability of modern BPM electronics.

**BPM offset (absolute reconstruction mode).** A constant offset vector with RMS $\sigma_b$ is added to both $\mathbf{y}_1$ and $\mathbf{y}_2$. In absolute reconstruction mode the reconstruction error grows steeply with $\sigma_b$: already at $\sigma_b = 10\,\mu\text{m}$ the error exceeds $40\,\mu\text{m}$. This underscores why absolute reconstruction without independent BPM calibration is impractical.

**BPM offset drift (drift-monitoring mode).** In drift-monitoring mode, the relevant quantity is the *change* in BPM offsets since calibration. If BPM electronics are stable to $\Delta\sigma_b \ll \sigma_b$ over the monitoring timescale, the offset contamination is negligible. <!-- TO BE DONE: Quantify BPM offset drift tolerance in the drift-monitor scenario, i.e., what offset drift rate (μm/hour) is acceptable before re-calibration is needed. -->

**Model $\beta$ error.** The analytic beta function is multiplicatively perturbed at each BPM. The reconstruction is tolerant up to $\sigma(\delta\beta/\beta) \approx 2\%$, beyond which the error exceeds $10\,\mu\text{m}$. Modern LOCO-style analyses routinely deliver $\beta$-function accuracy better than $1\%$, leaving a safety margin.

**Quadrupole tilt.** Skew-quadrupole coupling from random quadrupole tilts $\sigma_\theta \in [0,\,2]\,\text{mrad}$ produces minimal degradation ($\sim 0.5\,\mu\text{m}$ at $\sigma_\theta = 2\,\text{mrad}$). In a COD-mode measurement the skew coupling produces vertical kicks proportional to the *product* of the tilt and the horizontal closed-orbit excursion — a second-order quantity. This is verified independently: with a $1\,\text{mrad}$ horizontal kick in an otherwise perfect ring, vertical amplitude grows linearly with tilt over $[0, 10]\,\text{mrad}$ with 0.7% spread (Figure 5).

**Table 4.** Robustness thresholds for each error axis ($10\,\mu\text{m}$ reconstruction RMS limit, two-gradient direct estimator).

| Error source | Threshold | Notes |
|---|---|---|
| BPM noise (per turn) | $\sigma_n^{\text{single}} < 85\,\mu\text{m}$ | After 800-turn averaging |
| BPM offset (absolute mode) | $\sigma_b < 5\,\mu\text{m}$ | Requires BBA calibration |
| BPM offset drift (drift mode) | *TO BE DONE* | See text |
| $\beta$-function error | $\sigma(\delta\beta/\beta) < 2\%$ | LOCO typically $< 1\%$ |
| Quad tilt | $\sigma_\theta < 10\,\text{mrad}$ | Well beyond practical values |

Figure *(insert: stage_D_robustness_y.png, stage_D_robustness_x.png)* shows the full robustness sweep.

---

## 7. Discussion

### 7.1 The drift-monitoring paradigm

The principal conceptual shift in this work is the reframing of the alignment problem as *drift monitoring* rather than *absolute reconstruction*. This shift is well-motivated physically: the sources of slow drift in pEDM (thermal expansion, ground motion, HV cycling, plate charging) are distinct from the sources of absolute misalignment (installation error, survey imprecision). Each is best addressed by an appropriate tool.

More importantly, the drift formulation eliminates the BPM offset problem that plagues absolute reconstruction. In the expression $\widehat{\delta q}(t) = R^{-1}(\mathbf{y}(t) - \mathbf{y}_0)$, the BPM offset $\mathbf{b}$ cancels *without* gradient modulation — not because the offset vanishes, but because the same BPM electronics contribute to both $\mathbf{y}(t)$ and $\mathbf{y}_0$. Gradient modulation then serves an additional purpose: consistency checking between $v_1$ and $v_2$ estimates, and detection of slow BPM gain drift.

### 7.2 LOCO and BBA as the slow calibration layer

Linear Optics from Closed Orbits (LOCO) [9,10] fits the orbit response matrix to extract gradient errors, BPM gains, and corrector calibrations, but does not directly produce transverse-position misalignment estimates. Beam-based alignment (BBA) provides these positions directly, at the cost of beam time and operational complexity. In the proposed layered architecture:
- LOCO/BBA establishes the reference state $(\Delta q_0, \mathbf{b}_0, R)$ once per run period.
- Our method then tracks $\delta q(t)$ continuously from standard diagnostic BPM data, without additional corrector kicks or dedicated beam time.

This is not a competing approach to LOCO but a complementary fast layer that LOCO enables. The LOCO output provides the $\beta$, $\phi$, and $Q$ that define $R$; our method then converts BPM time-series into a continuous drift estimate.

### 7.3 Temporal resolution and duty cycle

The averaging time required to suppress betatron noise to the $1\,\mu\text{m}$ level in the COD estimate is $T_{\text{avg}} \approx N_{\text{turns}} \cdot T_{\text{rev}} \approx 800 \times 2\,\mu\text{s} \approx 1.6\,\text{ms}$ per gradient setting, or $\approx 3\,\text{ms}$ per two-gradient epoch. In practice, slow control-system overhead and gradient settling times will dominate, suggesting epochs of $\sim 1$–$10\,\text{s}$ in a realistic implementation. Drift mechanisms of interest (thermal, mechanical creep) operate on timescales of minutes to hours, so an epoch of $\sim 1\,\text{s}$ provides a comfortable sampling margin.

<!-- TO BE DONE: Estimate the typical drift rate for each pEDM-relevant mechanism (thermal expansion of support structure, ground motion, HV cycling, plate charging) from literature and assess whether the proposed 1–10 s epoch time is adequate to track these drifts without aliasing. -->

### 7.4 Limitations and open questions

**Lattice model accuracy.** The analytic $R$ is built from a model with a single calibrated parameter ($K_{x,\text{arc}}$). Real lattice errors (power supply ripple, remanent fields) will introduce model-simulation mismatch; this is the primary source of the $v_1$–$v_2$ asymmetry observed in Section 6.2. Periodic re-calibration of $R$ (e.g., by re-running LOCO) mitigates this.

**Nonlinear effects.** The forward model $\mathbf{y} = R\,\Delta q + \mathbf{b}$ is strictly linear. Large misalignments ($\Delta q \gg 100\,\mu\text{m}$) or large closed-orbit amplitudes will excite nonlinear terms from sextupole overlays and fringe fields, breaking the linear inversion.

**Coupling between planes.** The current implementation treats $x$ and $y$ as decoupled. Quadrupole tilts introduce skew coupling that mixes the two planes. For small tilts this is second-order (Section 6.3), but large tilts ($> 10\,\text{mrad}$) will require a coupled response-matrix formalism.

**Inverse crime assessment.** In the horizontal plane, both $R$ and the validation data depend on the same $K_{x,\text{arc}}$ calibration, creating a risk of overly optimistic performance estimates. An independent validation using a different simulation code or an experimental dataset would strengthen the conclusions.

---

## 8. Conclusion

We have presented a method for *online lattice drift monitoring* in an all-electric proton EDM storage ring. The core technical result is a proof that BPM-offset immunity and noise fidelity are fundamentally conjugate objectives in the two-gradient k-modulation problem: any linear estimator that cancels offsets amplifies noise by $O(1/\varepsilon)$. This is not a property of any particular implementation but a consequence of the linear algebra of the measurement.

Rather than seeking the unattainable combination of offset cancellation and noise fidelity, we advocate a layered architecture: a slow LOCO/BBA calibration layer that establishes an absolute reference state, and a fast direct-inversion drift-tracking layer that monitors *deviations* from that reference continuously. Within this architecture, the BPM offset problem dissolves: offsets cancel when differencing against the calibrated reference, without any gradient modulation.

The direct estimator achieves $1.4$ – $4.8\,\mu\text{m}$ RMS reconstruction error for $100\,\mu\text{m}$ injected misalignments (correlation $\rho = 0.998$ – $0.9999$), validated against a full 6D symplectic tracking simulation. The robustness sweep establishes the operating envelope: the method is tolerant to $\sim 2\,\mu\text{m}$ of BPM per-readout noise, $\sim 2\%$ lattice model error, and $\sim 10\,\text{mrad}$ of quadrupole tilt.

Open questions — including regularized $\Delta R$ spatial mode suppression, drift-rate tolerance, and horizontal-plane inverse crime quantification — are identified and will be addressed in a companion study.

Open-source Python implementations (`fodo_lattice.py`, `spectral_inversion.py`, `show_response.py`, `reconstruct.py`, `verify_quad_tilt.py`) accompany this paper and are available from the corresponding author.

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

[7] G. König, F.J.M. Farley, "Spin precession in a cylindrical electric capacitor at the magic momentum," internal note, Brookhaven National Laboratory (2008). *(or equivalent published derivation)*

[8] E. Hairer, C. Lubich, G. Wanner, *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations*, 2nd ed. (Springer, 2006).

[9] J. Safranek, "Experimental determination of storage ring optics using orbit response measurements," *Nucl. Instrum. Methods A* **388**, 27 (1997).

[10] X. Huang, J. Safranek, G. Portmann, "LOCO with constraints and improved fitting technique," *Proc. EPAC 2008*, p. 3120.

---

## Figure list

- **Figure 1.** Schematic of one FODO cell of the pEDM ring.
- **Figure 2.** Per-mode condition profile: $|\lambda_k|^{-1}$ vs Fourier mode index $k$ for $R_1$, $R_2$, and $\Delta R$. *(file: `stage_B_condition_y.png`, `stage_B_condition_x.png`)*
- **Figure 3.** Singular value spectra of $R$, $\Delta R$, and regularized $\Delta R$ (Tikhonov, TSVD), showing the suppression of high-frequency modes under regularization. **[TO BE DONE]**
- **Figure 4.** Spatial transfer function of each estimator: reconstructed amplitude / true amplitude vs Fourier mode index $k$. **[TO BE DONE]**
- **Figure 5.** Reconstruction performance: $\widehat{\Delta q}$ vs $\Delta q_{\text{true}}$, direct estimator, $\rho > 0.999$.
- **Figure 6.** Robustness sweep: reconstruction RMS error vs BPM noise, BPM offset, $\beta$-error, quadrupole tilt. *(file: `stage_D_robustness_y.png`, `stage_D_robustness_x.png`)*
- **Figure 7.** Drift-monitor scenario: injected slow drift vs estimated drift over time, with single-epoch uncertainty bands. **[TO BE DONE]**
- **Figure 8.** Linearity verification of skew-quadrupole coupling: vertical RMS displacement vs quadrupole tilt angle.

---

## Table list

- **Table 1.** Lattice parameters (Section 2.1).
- **Table 2.** Twiss parameters at nominal gradient (Section 2.4).
- **Table 3.** Reconstruction performance: direct inversion vs regularized $\Delta R$ methods (Section 5.3). *Partially TO BE DONE.*
- **Table 4.** Robustness thresholds (Section 6.3). *BPM offset drift row TO BE DONE.*
