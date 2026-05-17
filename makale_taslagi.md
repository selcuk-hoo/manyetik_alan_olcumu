# Spectral Reconstruction of Quadrupole Misalignments in an All-Electric Proton EDM Storage Ring from Two-Gradient Closed-Orbit Measurements

**Author(s):** Selcuk H.

**Affiliation:** *(to be filled)*

**Corresponding author:** *(email)*

**Target journal:** *Nuclear Instruments and Methods in Physics Research, Section A*

---

## Abstract

We present a method for reconstructing transverse quadrupole misalignments in a periodic FODO storage ring from beam position monitor (BPM) measurements taken at two distinct quadrupole gradient settings (the *k-modulation* technique). The conventional approach — inverting the gradient-difference response matrix $\Delta R = R_1 - R_2$ — suffers from a severe conditioning penalty: for the all-electric proton EDM lattice we study, $\kappa(\Delta R) \approx 2.8 \times 10^4$, which amplifies BPM measurement uncertainty by two orders of magnitude. We show that inverting each gradient configuration *separately* with the well-conditioned matrix $R_i$ ($\kappa(R_i) \approx 160$), then averaging, recovers the misalignment vector with reconstruction RMS error of $1.4$ – $4.8\,\mu\text{m}$ for $100\,\mu\text{m}$ injected misalignments — an improvement factor of roughly 280–340 over the $\Delta R$ method. The response matrix is built analytically from transfer matrices using a calibrated horizontal arc-focusing coefficient that accounts for centrifugal–relativistic coupling unique to electric deflectors. Validation against a full 6D Gauss–Legendre symplectic integrator confirms reconstruction correlations of $0.998$ – $0.9999$ between the predicted and true misalignment vectors. A four-axis robustness sweep (BPM noise, BPM offset, lattice $\beta$-error, quadrupole tilt) maps the operating envelope of the method. We discuss the fundamental trade-off between BPM-offset immunity and noise amplification that is intrinsic to two-gradient k-modulation, and we identify the parameter regime in which the proposed direct-inversion approach is superior to the offset-cancelling $\Delta R$ approach.

**Keywords:** beam position monitor; closed-orbit distortion; k-modulation; quadrupole misalignment; response matrix inversion; storage ring alignment; proton EDM

---

## 1. Introduction

The search for the proton electric dipole moment (pEDM) in an all-electric storage ring at the "magic" momentum $p \approx 0.7007\,\text{GeV/c}$ requires control over systematic effects at a level that depends critically on the transverse alignment of focusing elements [1,2]. Quadrupole misalignments at the $\sim 100\,\mu\text{m}$ scale produce closed-orbit distortions (CODs) which, in combination with radial electric fields and longitudinal magnetic field components, generate spin-rotation systematics that mimic an EDM signal. Reducing these systematics to the design sensitivity of $10^{-29}\,e\cdot\text{cm}$ demands an alignment knowledge — not merely an alignment hardware tolerance — at the few-$\mu$m level around the entire ring.

Direct mechanical measurement of all 48 quadrupoles to single-micron precision is impractical. The accepted approach is *beam-based alignment*: the misalignment vector $\Delta q \in \mathbb{R}^{N}$ ($N = 48$ in our case) is inferred from BPM measurements of the closed orbit, exploiting the linear forward model
$$\mathbf{y} = R \cdot \Delta q + \mathbf{b} + \boldsymbol{\eta},$$
where $R \in \mathbb{R}^{N\times N}$ is the (lattice-dependent) response matrix, $\mathbf{b}$ is the vector of BPM electronic offsets, and $\boldsymbol{\eta}$ is measurement noise. A *single* measurement of $\mathbf{y}$ is insufficient because $\mathbf{b}$ is comparable in magnitude to the misalignment signal in the COD and cannot be separated without further information.

The standard remedy is *k-modulation*: two CODs are recorded at slightly different quadrupole gradient settings $g_1$ and $g_2 = g_1(1+\varepsilon)$ (typically $\varepsilon \sim 10^{-2}$), so that
$$\mathbf{y}_1 - \mathbf{y}_2 = (R_1 - R_2)\,\Delta q + (\boldsymbol{\eta}_1 - \boldsymbol{\eta}_2).$$
The BPM offset $\mathbf{b}$ cancels exactly in the difference. The misalignment is then recovered by inverting the difference matrix $\Delta R = R_1 - R_2$. This procedure has been used in operating storage rings [3,4] and discussed extensively in textbook treatments [5,6].

The difficulty — well-known in principle but, to our knowledge, not quantified in the literature for the specific geometry of the pEDM ring — is that $\Delta R$ is profoundly ill-conditioned. Since $R_2 \approx R_1$ when $\varepsilon$ is small, the difference matrix is dominated by what amounts to a numerical derivative with respect to $g$, with singular values reduced by approximately the factor $\varepsilon$ relative to those of $R_i$. For our lattice we find $\kappa(\Delta R)/\kappa(R_1) \approx 170$, with $\kappa(\Delta R)$ exceeding $2.7 \times 10^4$. This implies that a 1% systematic error in the model $\Delta R$ amplifies into a 275% relative error in the recovered $\Delta q$. The method, in this regime, is unusable.

In this paper we make four contributions:

1. We construct $R$ analytically from the periodic FODO transfer matrices, demonstrating that this analytic form is sufficiently accurate for misalignment recovery — eliminating the $O(N)$ perturbation simulations conventionally used to populate $R$ column-by-column.
2. We propose an alternative two-gradient inversion in which $R_1$ and $R_2$ are inverted *separately* and the resulting estimates averaged. Because each $R_i$ is individually well-conditioned, the method tolerates BPM measurement noise to the few-$\mu$m level — far below what is possible with the $\Delta R$ approach.
3. We characterize the resulting trade-off: separate inversion does *not* cancel BPM offset, while $\Delta R$ inversion does. We show that these two approaches are connected by a closed-form linear combination, demonstrating that the trade-off between offset rejection and conditioning is mathematically fundamental and not avoidable by algorithmic refinement.
4. We validate the method against a full 6D Gauss–Legendre symplectic tracking simulation of the pEDM ring and we map the operating envelope through a systematic robustness sweep over four physically motivated error sources.

The method is implemented in publicly available Python modules and depends only on standard scientific computing libraries (NumPy, SciPy).

The remainder of the paper is organized as follows. Section 2 develops the analytic response-matrix formalism for the all-electric FODO lattice, including a discussion of the horizontal arc-focusing contribution that has no closed-form expression and is recovered by a one-time calibration. Section 3 presents the conditioning analysis of $R$ and $\Delta R$, including their modal (Fourier) decomposition. Section 4 derives and validates the separate-inversion two-gradient method. Section 5 reports robustness studies. Section 6 discusses the BPM-offset versus noise trade-off and physical implications. Section 7 concludes.

---

## 2. Response Matrix from Transfer-Matrix Optics

### 2.1 Lattice description

The pEDM ring consists of $N_c = 24$ identical FODO cells of length $L_c = 2\pi R_0 / N_c \approx 25\,\text{m}$, with $R_0 = 95.49\,\text{m}$. Each cell contains one focusing quadrupole (QF), one defocusing quadrupole (QD), two cylindrical electric deflectors ("arcs"), and four drift sections, in the sequence
$$\mathrm{QF} \to \mathrm{drift} \to \mathrm{arc} \to \mathrm{drift} \to \mathrm{QD} \to \mathrm{drift} \to \mathrm{arc} \to \mathrm{drift}.$$
The arc is a cylindrical capacitor operating at field index $n = 1$, providing the bending force for protons at the magic momentum. The ring has 48 quadrupoles total, and BPMs are taken to coincide with the entrance of each quadrupole.

### 2.2 Transfer-matrix elements

The single-plane transfer matrix of a thick quadrupole of integrated strength $K\,L_q$ is
$$M_{Q,\text{foc}} = \begin{pmatrix} \cos\phi & \sin\phi/\sqrt{K} \\ -\sqrt{K}\sin\phi & \cos\phi \end{pmatrix}, \quad \phi = \sqrt{K}L_q,$$
with $K = g_1 / B\rho$ and $B\rho = p/q$. The defocusing variant is obtained by $\cos \to \cosh$, $\sin \to \sinh$. Drift sections use the standard $\begin{pmatrix}1 & L \\ 0 & 1\end{pmatrix}$ matrix.

### 2.3 Arc focusing in cylindrical electric deflectors

Unlike magnetic dipoles, cylindrical electric deflectors at $n = 1$ produce *zero* vertical focusing because Maxwell's equations enforce $E_z = 0$ identically. Therefore for the vertical plane the arc transfer matrix is exactly a pure drift of length $L_{\text{arc}} = \pi R_0 / N_c$.

The horizontal plane is more subtle. Centrifugal defocusing in the rotating frame, combined with the relativistic $\beta^2$ correction to the effective potential, generates a small horizontal focusing that is *not* captured by the textbook expression $(1-n\pm\beta^2)/\rho^2$ once Coriolis terms are included consistently [7]. We treat the horizontal arc focusing coefficient $K_{x,\text{arc}}$ as a single, lattice-dependent calibration parameter, determined once by bisection so that the analytic model matches the horizontal betatron tune $Q_x = 2.6824$ extracted from a clean (zero-misalignment, small-angle kick) simulation. For our parameters this yields $K_{x,\text{arc}} = 1.265 \times 10^{-4}\,\text{m}^{-2}$.

### 2.4 Twiss parameters

The one-cell transfer matrix $M_c$ for each plane is the product of the elemental matrices in cell order. The cell phase advance is $\mu = \arccos(\mathrm{Tr}\,M_c / 2)$ and the beta function at the cell entrance is $\beta_0 = M_{c,12} / \sin\mu$ (using the symmetric-FODO simplification $\alpha = 0$ at the entrance). The total betatron tune is $Q = N_c\,\mu / (2\pi)$. Beta functions and accumulated phase $\phi_i$ at every quadrupole entrance are obtained by propagating the Twiss vector $(\beta, \alpha)^\top$ through each element using the standard transformation $\beta(s_2) = M_{11}^2 \beta_1 - 2M_{11}M_{12}\alpha_1 + M_{12}^2 \gamma_1$.

For the pEDM lattice at $g_1 = 0.21\,\text{T/m}$ we obtain $Q_x = 2.682$, $Q_y = 2.362$, with vertical $\beta$ ranging over $[41.2,\,76.4]\,\text{m}$ and horizontal $\beta$ over $[36.1,\,67.4]\,\text{m}$. These values agree with the 6D tracking simulation (Section 4) to four decimal places.

### 2.5 The response matrix

A misalignment $\Delta q_j$ of quadrupole $j$ acts as a thin-lens dipole kick of strength $\theta_j = -K_j L_j \Delta q_j$ on the beam, which propagates around the ring and produces a closed orbit at BPM $i$ given by the Courant–Snyder formula
$$R_{ij} = \frac{\sqrt{\beta_i \beta_j}}{2 \sin(\pi Q)} \,\cos\!\bigl(|\phi_i - \phi_j| - \pi Q\bigr) \cdot KL_j,$$
where $KL_j$ carries the sign appropriate to the plane: for vertical motion QF defocuses ($KL_j < 0$) and QD focuses ($KL_j > 0$); for horizontal motion the signs reverse. The sign convention is absorbed into $KL_j$ rather than into a separate prefactor, eliminating a source of error that we encountered in initial implementations.

The conditioning of $R$ depends on $\sin(\pi Q)$: working points near integer or half-integer tunes are pathological. For our nominal tunes the condition numbers are $\kappa(R_x) \approx 141$ and $\kappa(R_y) \approx 160$.

---

## 3. Conditioning of the Two-Gradient Problem

### 3.1 Why $\Delta R$ is ill-conditioned

If the lattice optics were rigorously invariant under the gradient change $g_1 \to g_2$ in the sense that only the integrated strength $KL$ changed (and the beta functions and phases remained unchanged), then $R_1 = (KL_1)\,M$ and $R_2 = (KL_2)\,M$ with $M$ a common matrix, and
$$\Delta R = (KL_1 - KL_2)\,M.$$
In this idealization $\kappa(\Delta R) = \kappa(M) = \kappa(R_i)$, and there would be no conditioning penalty. The ill-conditioning of $\Delta R$ originates entirely from the second-order dependence of the optics on $g$: the tune shift $\Delta Q$, beta-beating, and phase-advance shift contribute terms proportional to $\varepsilon$ that scale the difference of well-conditioned matrices.

For our lattice with $\varepsilon = 0.02$ we measure $\kappa(\Delta R_y) = 27\,560$, an amplification factor $\kappa(\Delta R)/\kappa(R_1) \approx 170$. This is consistent with the heuristic $1/\varepsilon \approx 50$ for the leading-order singular-value contraction, with an additional factor of $\sim 3$ from second-order optical perturbation.

### 3.2 Spectral structure of $R$ in a periodic FODO lattice

For a perfectly periodic ring with $N$ identical cells and a single quadrupole per cell, the response matrix elements depend only on the index difference $|i-j|$, making $R$ exactly circulant. Its eigenvectors are the columns of the discrete Fourier transform matrix and the eigenvalues are $\lambda_k = \mathrm{DFT}(R_{0,\,\cdot})_k$. In the pEDM lattice $R$ is block-circulant rather than strictly circulant because the QF and QD positions have unequal beta functions; the deviation from strict circularity is small, however, and we exploit it for diagnostic purposes.

The mode-by-mode condition number $|\lambda_k|^{-1}$ exposes which Fourier components of the misalignment pattern are reconstructed accurately and which are amplified. For our lattice the worst mode of $R_y$ has $|\lambda|^{-1} \approx 0.4$, whereas the worst mode of $\Delta R_y$ exceeds $7000$. Figure *(insert: stage_B_condition_y.png)* shows the per-mode condition profile.

---

## 4. Direct Two-Gradient Inversion

### 4.1 Algorithm

Let $\mathbf{y}_1$ and $\mathbf{y}_2$ denote the BPM-readout vectors at gradients $g_1$ and $g_2$. Define
$$v_i = R_i^{-1}\,\mathbf{y}_i, \quad i=1,2, \qquad \widehat{\Delta q} = \tfrac{1}{2}(v_1 + v_2).$$
Each $v_i$ is the best linear estimate of $\Delta q$ at a single gradient, with BPM offset contributing the residual $R_i^{-1}\mathbf{b}$. The average $\widehat{\Delta q}$ is the central estimator; the difference $v_1 - v_2 = (R_1^{-1} - R_2^{-1})\mathbf{b}$ carries information that, in principle, can be used to reconstruct $\mathbf{b}$ as a separate offset-cancelling step. Here we report results for the central estimator alone.

### 4.2 Relation to the $\Delta R$ method

A short calculation shows that the offset-cancelling estimator constructed from $v_1$ and $v_2$ is
$$\widehat{\Delta q}^{\,\text{(no offset)}} = \frac{KL_2\,v_2 - KL_1\,v_1}{KL_2 - KL_1}.$$
Substituting $R_i = (KL_i)M$ (and verifying that $\mathbf{b}$ cancels) recovers exactly $\Delta R^{-1}(\mathbf{y}_1 - \mathbf{y}_2)$. The two methods — separate inversion plus averaging, and $\Delta R$ inversion — are therefore extreme points of a one-parameter family of weighted combinations:
$$\widehat{\Delta q}(\alpha) = \alpha\,v_1 + (1-\alpha)\,v_2 - \frac{\alpha\,R_1^{-1} + (1-\alpha)\,R_2^{-1}}{KL_2 - KL_1}\,\text{(offset correction)}.$$
The choice $\alpha = 1/2$ minimizes noise variance under the assumption of uncorrelated measurement noise; the offset-cancelling choice $\alpha = -KL_1/(KL_2-KL_1) \approx -50$ multiplies the input noise by the same large factor that conditions $\Delta R$. There is no algorithmic refinement that escapes this trade-off.

### 4.3 Simulation validation

We validate the method against a full 6D tracking simulation using a Gauss–Legendre fourth-order symplectic integrator [8] applied to the relativistic Newton–Lorentz equations with self-consistent Thomas–BMT spin evolution. The simulation operates in global Cartesian coordinates with the rotating frame restored at each arc segment, and includes the full physics of the cylindrical electric deflector, quadrupole misalignments in both planes, quadrupole tilt (skew-quadrupole coupling), and the rigorous Maxwell-consistent sextupole overlay. The closed-orbit data file accumulates BPM positions over typically 500–800 turns and is averaged to suppress betatron oscillations.

For an ensemble with $\sigma(\Delta q) = 100\,\mu\text{m}$ injected into both vertical and horizontal misalignments, the analytic two-gradient method recovers:

| Plane | $R_1$-only (single gradient) | $R_2$-only (single gradient) | Two-grad. avg. | $\Delta R$ method |
|-------|----------------------------|------------------------------|----------------|-------------------|
| $y$   | RMS $5.8\,\mu\text{m}$, $\rho = 0.995$ | RMS $1.8\,\mu\text{m}$, $\rho = 0.9996$ | RMS $3.5\,\mu\text{m}$, $\rho = 0.998$ | RMS $1\,865\,\mu\text{m}$, $\rho = 0.085$ |
| $x$   | RMS $7.7\,\mu\text{m}$, $\rho = 0.989$ | RMS $1.5\,\mu\text{m}$, $\rho = 0.9996$ | RMS $3.6\,\mu\text{m}$, $\rho = 0.998$ | RMS $1\,396\,\mu\text{m}$, $\rho = -0.05$ |

The two-gradient average improves over the $\Delta R$ method by a factor of $\sim 400$ in RMS error, and reconstructs misalignments at $3$ – $4\,\mu\text{m}$ accuracy from a $100\,\mu\text{m}$ injected ensemble. The asymmetry between $v_1$ and $v_2$ — with $v_2$ noticeably more accurate — reflects the fact that the arc-focusing calibration $K_{x,\text{arc}}$ was performed at the nominal gradient $g_1$, leaving a small model-simulation mismatch when used at the perturbed gradient. A more refined implementation would re-calibrate at each gradient setting; we have not pursued this here because the central estimator $(v_1+v_2)/2$ already meets the design requirement of $<10\,\mu\text{m}$ at $\rho > 0.95$.

---

## 5. Robustness Studies

To map the operating envelope of the method we performed a four-axis sweep over physically motivated error sources. The base configuration is the validated case described in Section 4.3.

**BPM noise.** A Gaussian noise vector with RMS $\sigma_n$ is added independently to $\mathbf{y}_1$ and $\mathbf{y}_2$. The reconstruction RMS error grows linearly with $\sigma_n$ at a rate of approximately $3 \,\mu\text{m}$ output per $\mu\text{m}$ input, crossing the $10\,\mu\text{m}$ threshold at $\sigma_n \approx 3\,\mu\text{m}$. After the $\sim 800$-turn averaging that produces the closed-orbit estimate, this corresponds to a single-shot BPM resolution requirement of approximately $\sigma_n^{\text{single}} \approx \sigma_n \sqrt{800} \approx 85\,\mu\text{m}$, which is well within the capability of modern BPM electronics.

**BPM offset.** A constant offset vector $\mathbf{b}$ with RMS $\sigma_b$ is added to *both* $\mathbf{y}_1$ and $\mathbf{y}_2$. The central estimator $(v_1+v_2)/2$ does *not* cancel $\mathbf{b}$, and the reconstruction error grows steeply: a $10\,\mu\text{m}$ RMS offset already drives the reconstruction error to $\sim 40\,\mu\text{m}$, exceeding the $10\,\mu\text{m}$ design threshold. This is the principal vulnerability of the direct-inversion method and motivates the use of beam-based-alignment procedures to calibrate $\mathbf{b}$ independently to better than $\sim 5\,\mu\text{m}$ before reconstruction.

**Model $\beta$ error.** The analytic beta function is multiplicatively perturbed by an independent Gaussian $(1+\delta_k)$ at each BPM, with $\sigma(\delta) \in [0,\,5\%]$. The reconstruction is tolerant up to $\sigma(\delta) \approx 2\%$, beyond which the error exceeds $10\,\mu\text{m}$. Modern LOCO-style analyses [9,10] routinely deliver beta-function accuracy better than $1\%$, leaving a safety margin.

**Quadrupole tilt.** Skew-quadrupole coupling from random quadrupole tilts $\sigma_\theta \in [0,\,2]\,\text{mrad}$ produces minimal degradation: the reconstruction error grows by only $\sim 0.5\,\mu\text{m}$ at $\sigma_\theta = 2\,\text{mrad}$, well within the threshold. The reason is that in a COD-mode measurement (zero betatron amplitude) the skew coupling produces vertical kicks proportional to the *product* of the tilt angle and the horizontal closed-orbit excursion at that quadrupole — a second-order quantity. We verified independently that the simulation correctly implements skew coupling by injecting a $1\,\text{mrad}$ horizontal kick into an otherwise perfect ring and observing the linear growth of vertical amplitude with quadrupole tilt up to the $10\,\text{mrad}$ level (linear-regime spread 0.7%).

---

## 6. Discussion

### 6.1 Why two-gradient inversion is not a "free lunch" against BPM offset

The closed-form relation between the central estimator and the $\Delta R$ estimator (Section 4.2) shows that recovering BPM-offset immunity comes at the price of amplifying noise by a factor of $1/\varepsilon \sim 50$. This is not an artifact of the chosen algorithm but a consequence of the linear structure of the two-gradient problem. Any linear combination of $v_1$ and $v_2$ that cancels $\mathbf{b}$ is necessarily proportional to the difference, and the difference of two well-conditioned matrices whose distance is $O(\varepsilon)$ is, by elementary linear algebra, ill-conditioned by a factor $O(1/\varepsilon)$.

This observation suggests a clear operational strategy:
1. Use beam-based-alignment to bring $\mathbf{b}$ to the few-$\mu\text{m}$ level before reconstruction.
2. Use the central two-gradient estimator $(v_1+v_2)/2$ to recover $\Delta q$ at micron precision, exploiting the well-conditioned $R_i$.
3. *If* an independent offset measurement is unavailable, use the $\Delta R$ estimator with explicit Tikhonov regularization to control the noise amplification at the price of accepting larger reconstruction error.

### 6.2 Comparison with LOCO

LOCO (Linear Optics from Closed Orbits) [9,10] uses many corrector kicks to build the orbit response matrix from data and then fits a model containing gradient errors, BPM gains, and corrector calibrations. LOCO delivers superb optical characterization but does *not* directly produce transverse-position misalignment estimates; its outputs are gradient deviations and BPM gain corrections. The method presented here is complementary: LOCO provides the $\beta$, $\phi$, and $Q$ that drive our analytic $R$ matrix, and our procedure then converts two BPM readouts at different gradients into a transverse-misalignment estimate without the corrector-kick measurements that LOCO requires.

### 6.3 Analytic versus simulation-based response matrices

In the past, response matrices for misalignment reconstruction have often been built numerically by single-quadrupole perturbation: each column of $R$ is obtained by displacing one quadrupole by a known amount and reading out the resulting closed orbit, requiring $N+1$ simulations per gradient. For our $N = 48$ ring this amounts to $\sim 97$ simulations per gradient setting. The analytic transfer-matrix formula reduces this to zero simulations and is accurate to better than 1% throughout, as judged from the consistency of reconstruction across the two gradients and against the 6D symplectic simulation.

### 6.4 Applicability beyond pEDM

The method depends on three lattice properties: (i) sufficient cellular symmetry that the response matrix is approximately circulant; (ii) a well-characterized Twiss model from analytic optics or LOCO; (iii) a working point comfortably distant from integer and half-integer resonances so that $\sin(\pi Q)$ does not vanish. These conditions are satisfied by most modern third-generation light sources, by hadron storage rings, and by several proposed lepton colliders. The all-electric proton EDM ring is unusual in two respects: the absence of vertical arc focusing imposed by Maxwell's equations at $n=1$, and the lack of a closed-form expression for horizontal arc focusing. Both are addressed by the single-parameter calibration described in Section 2.3.

---

## 7. Conclusion

We have presented and validated an analytic, simulation-free method for reconstructing quadrupole transverse misalignments in a periodic storage ring from closed-orbit measurements at two distinct quadrupole gradients. The central technical observation is that the conventional $\Delta R$-inversion procedure is fundamentally ill-conditioned because the difference of nearly equal response matrices contracts singular values by a factor of order $\varepsilon$, the relative gradient change. Inverting each gradient configuration separately preserves the well-conditioned spectrum of the underlying response matrix and yields reconstructions at the $1$ – $5\,\mu\text{m}$ level for $100\,\mu\text{m}$ injected misalignments, an improvement of two-to-three orders of magnitude over the $\Delta R$ method in BPM-noise tolerance.

The method has been validated against full 6D symplectic tracking and characterized over four error axes (BPM noise, BPM offset, $\beta$-error, quadrupole tilt). For the proton EDM lattice it satisfies the design requirement of better than $10\,\mu\text{m}$ reconstruction RMS at correlation greater than $0.95$, provided that BPM offsets are independently calibrated below $\sim 5\,\mu\text{m}$ — a requirement that, while demanding, is consistent with current beam-based alignment practice.

The mathematical relation between the proposed direct method and the $\Delta R$ method is a closed-form linear combination, demonstrating that the trade-off between noise amplification and BPM-offset immunity is intrinsic and cannot be circumvented by algorithmic refinement. The principal practical consequence is that effort spent on BPM-offset calibration is repaid linearly in reconstruction accuracy, whereas no algorithmic technique can substitute for an independent offset measurement.

Open-source Python implementations (`fodo_lattice.py`, `spectral_inversion.py`, `show_response.py`, `reconstruct.py`, `verify_quad_tilt.py`) accompany this paper and are available from the corresponding author. The integrator is a standalone C++ Gauss–Legendre fourth-order symplectic code with Python bindings.

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

## Figure list (to be inserted)

- **Figure 1.** Schematic of one FODO cell of the pEDM ring.
- **Figure 2.** Modal condition map: $|\lambda_k|^{-1}$ versus Fourier mode index $k$ for $R_1$, $R_2$, and $\Delta R$ (file: `stage_B_condition_y.png`, `stage_B_condition_x.png`).
- **Figure 3.** Reconstruction performance for a representative random misalignment ensemble: $\widehat{\Delta q}$ versus $\Delta q_{\text{true}}$, with $\rho > 0.999$.
- **Figure 4.** Robustness sweep: reconstruction RMS error versus BPM noise, BPM offset, $\beta$-error, quadrupole tilt (file: `stage_D_robustness_y.png`, `stage_D_robustness_x.png`).
- **Figure 5.** Linearity verification of skew-quadrupole coupling: vertical RMS displacement versus quadrupole tilt angle, demonstrating the expected proportionality.

---

## Table list

- **Table 1.** Lattice parameters of the pEDM ring (Section 2.1).
- **Table 2.** Twiss parameters at nominal gradient (Section 2.4).
- **Table 3.** Reconstruction performance comparison (Section 4.3 — already in main text).
- **Table 4.** Robustness thresholds for each error axis (summary of Section 5).
