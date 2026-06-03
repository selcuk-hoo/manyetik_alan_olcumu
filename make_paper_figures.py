#!/usr/bin/env python3
"""make_paper_figures.py — figures and tables for paper_draft.tex.

Output files:
  fig1_falseedm_scan.png   – false-EDM rate vs k, bar chart (Table 1)
  fig2_orbit_gain.png      – ||RF_k|| orbit gain vs k (Table 2)
  fig3_mode_patterns.png   – misalignment pattern F_k and orbit response
                             M_k = R@F_k along the ring, for k=1,2,3,4
                             (4 rows × 2 columns)
  fig4_sigma_model.png     – k=2 error budget: model uncertainty vs BPM offset
  fig5_offset_whiteness.png– white BPM offset → broadband recovered spectrum
                             (no spurious peak; lowest at k=2)
  table2_gain.txt          – Table 2 raw numbers
  table3_orbit.txt         – Table 3 raw numbers

Projection estimator (correct formulation):
    dq = a_c * F_k_cos + a_s * F_k_sin   (unnormalized F_k)
    y  = R @ dq + b
    [a_c, a_s] = lstsq([M_c, M_s], y),  M_c = R@F_c,  M_s = R@F_s
    A_fit = sqrt(a_c² + a_s²)   [same units as misalignment: metres]
  Theory: |δA| ≤ ||b|| / ||M_k||,  ||M_k=2|| ≈ 167

Usage:
  python3 make_paper_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

# ── style ───────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE   = "#2166ac"
RED    = "#d6604d"
ORANGE = "#f4a742"
GRAY   = "#888888"
GREEN  = "#4dac26"
PURPLE = "#762a83"

# ── Fourier basis (unnormalized) ─────────────────────────────────────────────

def Fcos(k, n_q=48):
    """FODO-antisymmetric cosine mode (unnormalized)."""
    j = np.arange(n_q)
    if k == 0:
        return (-1.0) ** j
    return (-1.0) ** j * np.cos(2 * np.pi * k * (j // 2) / (n_q // 2))

def Fsin(k, n_q=48):
    """FODO-antisymmetric sine mode (unnormalized)."""
    j = np.arange(n_q)
    return (-1.0) ** j * np.sin(2 * np.pi * k * (j // 2) / (n_q // 2))

def RF_unit_norm(R, k):
    """||RF_k|| with unit-normalized F_k (Table 2 values)."""
    Fc = Fcos(k, R.shape[0]);  Fc = Fc / np.linalg.norm(Fc)
    return np.linalg.norm(R @ Fc)

def M_col_norm(R, k):
    """||M_k|| = sqrt(N_cells) * ||RF_k||  ≈ 167 for k=2."""
    return np.sqrt(R.shape[0] // 2) * RF_unit_norm(R, k)

# ── projection estimator ─────────────────────────────────────────────────────

def project_amplitude(y, R, k):
    """Estimate amplitude of harmonic k from orbit measurement y (metres).
    [a_c, a_s] = lstsq([M_c, M_s], y),  A = sqrt(a_c²+a_s²)
    """
    Mc = R @ Fcos(k, R.shape[0])
    Ms = R @ Fsin(k, R.shape[0])
    M2 = np.column_stack([Mc, Ms])
    a2, _, _, _ = np.linalg.lstsq(M2, y, rcond=None)
    return float(np.sqrt(a2[0]**2 + a2[1]**2)), float(np.arctan2(a2[1], a2[0]))

# ── test orbit generator ─────────────────────────────────────────────────────

def make_orbit(R, k_true=2, A_true=10e-6, phi_true=0.3,
               contaminants=None, b_sigma=0.0, rng=None):
    """Generate test orbit with known harmonic pattern + BPM offset.
    dq = A_true*(cos(phi)*Fc + sin(phi)*Fs) + contaminants
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if contaminants is None:
        contaminants = {4: (300e-6, 0.7), 6: (300e-6, 1.2), 8: (200e-6, 2.1)}
    n_q = R.shape[0]
    dq = A_true * (np.cos(phi_true) * Fcos(k_true, n_q)
                   + np.sin(phi_true) * Fsin(k_true, n_q))
    for k_c, (A_c, phi_c) in contaminants.items():
        dq += A_c * (np.cos(phi_c) * Fcos(k_c, n_q)
                     + np.sin(phi_c) * Fsin(k_c, n_q))
    b = rng.normal(0, b_sigma, n_q) if b_sigma > 0 else np.zeros(n_q)
    return R @ dq + b

# ── response matrix ──────────────────────────────────────────────────────────

R = np.load("R_dy_1.npy")   # 48×48, units: m/m
N_Q = R.shape[0]

svs = np.linalg.svd(R, compute_uv=False)
kappa_R = svs[0] / svs[-1]
Mk2_norm = M_col_norm(R, 2)

print(f"R: {R.shape},  κ(R) = {kappa_R:.1f},  ||M_k=2|| = {Mk2_norm:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — False-EDM rate vs k  (stroboscopic spin-tracking results)
# ══════════════════════════════════════════════════════════════════════════════

# false_edm_mode_scan.py — 10 ms, A=10 μm, CO-launch + stroboscopic
dsydt = {0: +1.80e-10, 1: +4.93e-10, 2: +1.44e-9,
         3: -6.64e-10, 4: -1.98e-10, 5: -9.92e-11}
co_amp = {0: 0.058, 1: 0.070, 2: 0.198, 3: 0.088, 4: 0.028, 5: 0.014}

ks = sorted(dsydt)
bar_col = [RED if k == 2 else BLUE for k in ks]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

ax = axes[0]
ax.bar(ks, [abs(dsydt[k]) * 1e9 for k in ks],
       color=bar_col, width=0.6, edgecolor="white", linewidth=0.8)
ax.bar(2, abs(dsydt[2]) * 1e9, color=RED, width=0.6,
       edgecolor="black", linewidth=1.3, label="$k=2$ (dominant)")
for k in ks:
    sign = "+" if dsydt[k] > 0 else u"−"
    ax.text(k, abs(dsydt[k]) * 1e9 + 0.02, sign,
            ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Fourier mode $k$")
ax.set_ylabel(r"$|dS_y/dt|$ [$10^{-9}$ rad/s]")
ax.set_title("(a) False-EDM rate")
ax.set_xticks(ks)
ax.legend(frameon=False)

ax = axes[1]
ax.bar(ks, [co_amp[k] for k in ks],
       color=bar_col, width=0.6, edgecolor="white", linewidth=0.8)
ax.bar(2, co_amp[2], color=RED, width=0.6, edgecolor="black", linewidth=1.3)
ax.set_xlabel("Fourier mode $k$")
ax.set_ylabel("Closed-orbit amplitude [mm]")
ax.set_title("(b) Closed-orbit amplitude")
ax.set_xticks(ks)

fig.suptitle(r"$A=10\,\mu$m single-harmonic misalignment, true EDM off",
             fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig("fig1_falseedm_scan.png", bbox_inches="tight")
plt.close(fig)
print("fig1_falseedm_scan.png  ✓")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Fourier mode orbit gain
# ══════════════════════════════════════════════════════════════════════════════

k_list = list(range(1, 13))
rf_norms = [RF_unit_norm(R, k) for k in k_list]
m_norms  = [M_col_norm(R, k)   for k in k_list]
bar_col2 = [RED if k == 2 else BLUE for k in k_list]

fig, ax = plt.subplots(figsize=(7, 3.8))
xpos = np.arange(len(k_list))
ax.bar(xpos, rf_norms, color=bar_col2, width=0.6,
       edgecolor="white", linewidth=0.8)
ax.bar(xpos[1], rf_norms[1], color=RED, width=0.6,
       edgecolor="black", linewidth=1.3,
       label=fr"$k=2$: $\|RF_2\|={rf_norms[1]:.1f}$")
ax.axhline(np.mean(rf_norms), color=GRAY, linestyle="--", linewidth=0.8,
           label=f"Mean = {np.mean(rf_norms):.1f}")
ax.set_xticks(xpos)
ax.set_xticklabels([str(k) for k in k_list])
ax.set_xlabel("Fourier mode $k$")
ax.set_ylabel(r"$\|RF_k\|$ (unit-normalized $F_k$)")
ax.set_title(r"Fourier mode orbit gain  ($Q_y \approx 2.68$)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("fig2_orbit_gain.png", bbox_inches="tight")
plt.close(fig)
print("fig2_orbit_gain.png  ✓")

# table output
with open("table2_gain.txt", "w") as f:
    f.write("% Table 2: Fourier mode orbit gain\n")
    f.write(f"% {'k':>3}  {'||RF_k||':>10}  {'||M_k||':>10}\n")
    for k, rf, m in zip(k_list, rf_norms, m_norms):
        tag = "  % <-- dominant" if k == 2 else ""
        f.write(f"  {k:3d}  {rf:10.3f}  {m:10.1f}{tag}\n")
print("table2_gain.txt  ✓")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Mode-k quad misalignment: horizontal-field drive and orbit response
#   Left  : coherent horizontal field B_x ∝ G·Δy that drives the orbit.
#           The F/D gradient sign (-1)^j and the (-1)^j displacement sign of the
#           FODO-antisymmetric mode cancel, leaving a smooth cos(2πk·m/24) drive.
#   Right : closed-orbit response M_k = R·F_k (the measured orbit), vs s.
#   Rows k = 1,2,3,4; k=2 sits closest to Q_y≈2.68 and is resonantly amplified.
# ══════════════════════════════════════════════════════════════════════════════

A_demo  = 10e-6                                       # 10 μm misalignment amplitude
C_ring  = 2 * np.pi * 95.49                           # ring circumference ≈ 600 m
s_bpm   = np.linspace(0, C_ring, N_Q, endpoint=False) # quad/BPM s-positions [m]
k_vals3 = [1, 2, 3, 4]

def drive_field(k, n_q=N_Q):
    """Coherent horizontal field B_x ∝ G·Δy for a mode-k vertical misalignment.
    Gradient sign (-1)^j and displacement sign (-1)^j cancel → smooth cosine."""
    m = np.arange(n_q) // 2
    if k == 0:
        return np.ones(n_q)
    return np.cos(2 * np.pi * k * m / (n_q // 2))

fig3, axes3 = plt.subplots(4, 2, figsize=(9.5, 10))
fig3.subplots_adjust(hspace=0.55, wspace=0.38)

for row, k in enumerate(k_vals3):
    drive   = drive_field(k)                       # normalized B_x pattern (±1)
    orbit   = A_demo * (R @ Fcos(k, N_Q)) * 1e6    # orbit response [μm]
    col_k   = RED if k == 2 else BLUE
    mk_norm = M_col_norm(R, k)

    # ── left: coherent horizontal-field drive ────────────────────────────────
    ax_L = axes3[row, 0]
    ax_L.plot(s_bpm, drive, "o-", color=col_k, ms=4, lw=1.2)
    ax_L.axhline(0, color="k", linewidth=0.4)
    ax_L.set_ylim(-1.35, 1.35)
    ax_L.set_ylabel(r"$B_x$ drive [arb.]", fontsize=9)
    ax_L.set_title(fr"$k={k}$   Horizontal-field drive (coherent)", fontsize=9)
    ax_L.tick_params(labelsize=8)

    # ── right: closed-orbit response ──────────────────────────────────────────
    ax_R = axes3[row, 1]
    ax_R.plot(s_bpm, orbit, "o-", color=col_k, ms=4, lw=1.2)
    ax_R.axhline(0, color="k", linewidth=0.4)
    ax_maxR = np.max(np.abs(orbit)) * 1.35
    ax_R.set_ylim(-ax_maxR, ax_maxR)
    ax_R.set_ylabel(r"Orbit $[\mu\mathrm{m}]$", fontsize=9)
    ax_R.set_title(fr"$k={k}$   Orbit response  ($\|M_k\|={mk_norm:.0f}$)",
                   fontsize=9)
    ax_R.tick_params(labelsize=8)

# x-labels on the bottom row only
for col in range(2):
    axes3[3, col].set_xlabel(r"Ring position $s$ [m]", fontsize=10)

fig3.suptitle(
    r"Mode-$k$ quad misalignment ($A=10\,\mu$m): coherent horizontal-field "
    r"drive (left) and closed-orbit response (right)"   "\n"
    r"$k=2$ yields the largest orbit ($\|M_{k=2}\|=167$, "
    r"$\sim\!4\times$ the neighbouring modes)",
    fontsize=10, y=1.005)
fig3.savefig("fig3_mode_patterns.png", bbox_inches="tight")
plt.close(fig3)
print("fig3_mode_patterns.png  ✓")
for k in k_vals3:
    print(f"  k={k}: ||M_k|| = {M_col_norm(R, k):.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Error budget: gradient model error vs BPM offset noise
# ══════════════════════════════════════════════════════════════════════════════

sigma_m_vals = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]) / 100.0
A_true   = 10e-6
phi_true = 0.3
N_MC     = 300

# Three BPM-offset scenarios: no offset, moderate, severe
b_scenarios = [
    (0.0,    r"No BPM offset ($\sigma_b = 0$)",     BLUE),
    (100e-6, r"$\sigma_b = 100\,\mu$m",             GREEN),
    (300e-6, r"$\sigma_b = 300\,\mu$m",             RED),
]

fig4, ax4 = plt.subplots(figsize=(7, 4.5))

tol_300 = None
for b_sig, label, color in b_scenarios:
    rng4 = np.random.default_rng(99)
    means4, stds4 = [], []
    for sig_m in sigma_m_vals:
        errs = []
        for _ in range(N_MC):
            y = make_orbit(R, A_true=A_true, phi_true=phi_true,
                           b_sigma=b_sig, rng=rng4)
            # Estimator uses a perturbed response matrix (gradient errors)
            eps     = rng4.normal(0, sig_m, N_Q)
            R_model = R * (1 + eps)[np.newaxis, :]
            A_fit, _ = project_amplitude(y, R_model, 2)
            errs.append(abs(A_fit - A_true) * 1e6)
        means4.append(np.mean(errs))
        stds4.append(np.std(errs))
    means4 = np.array(means4)
    stds4  = np.array(stds4)
    ax4.errorbar(sigma_m_vals * 100, means4, yerr=stds4,
                 fmt="o-", color=color, capsize=3, markersize=4,
                 linewidth=1.5, label=label)
    if b_sig == 300e-6:
        # tolerance where the 300-μm curve crosses the 10-μm target
        tol_300 = float(np.interp(10.0, means4, sigma_m_vals * 100))

ax4.axhline(10, color=ORANGE, linestyle=":", linewidth=1.2,
            label=r"$10\,\mu$m target")
if tol_300 is not None and tol_300 < 10.0:
    ax4.axvline(tol_300, color=GRAY, linestyle="--", linewidth=0.9,
                label=fr"Tolerance ($\sigma_b=300\,\mu$m) $\approx{tol_300:.1f}$%")

ax4.set_xlabel(r"Gradient model error $\sigma_\mathrm{model} = \delta K/K$ [%]")
ax4.set_ylabel(r"$k=2$ amplitude error $[\mu\mathrm{m}]$")
ax4.set_title("Error budget: gradient model uncertainty vs BPM offset noise")
ax4.legend(frameon=False, loc="upper left")
ax4.set_xlim(-0.2, 10.5)
ax4.set_ylim(bottom=-0.1)
fig4.tight_layout()
fig4.savefig("fig4_sigma_model.png", bbox_inches="tight")
plt.close(fig4)
print("fig4_sigma_model.png  ✓")
if tol_300 is not None:
    print(f"  Tolerance (sigma_b=300 μm, 10 μm target): ~{tol_300:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — BPM-offset whiteness / robustness
#   (a) one realization of a white BPM offset b(s) vs s.
#   (b) harmonic amplitude spectrum recovered by the estimator â = M⁺ y:
#       a white offset gives a broadband floor with NO spurious peak — and it
#       is LOWEST exactly at k=2 (largest ||M_k||), whereas a true k=2
#       misalignment recovers as a clean line standing above the floor.
# ══════════════════════════════════════════════════════════════════════════════

KMAX = 12
cols5 = [Fcos(0, N_Q)]
for k in range(1, KMAX + 1):
    cols5.append(Fcos(k, N_Q))
    cols5.append(Fsin(k, N_Q))
F_full = np.column_stack(cols5)
M_full = R @ F_full
M_pinv = np.linalg.pinv(M_full, rcond=1e-3)

def amp_spectrum(ahat):
    """Collapse [a0, c1, s1, c2, s2, ...] into amplitude per harmonic k=0..KMAX."""
    out = [abs(ahat[0])]
    for k in range(1, KMAX + 1):
        out.append(np.hypot(ahat[1 + 2 * (k - 1)], ahat[2 + 2 * (k - 1)]))
    return np.array(out)

sigma_b5 = 300e-6
A_sig5   = 10e-6
rng5     = np.random.default_rng(1)

# true k=2 signal (Δq = 10 μm in mode 2) → recovered spectrum
a_sig = amp_spectrum(M_pinv @ (R @ (A_sig5 * Fcos(2, N_Q)))) * 1e6

# pure white offset (Δq = 0), many realizations → recovered spectrum
N_TR5 = 400
specs = np.array([amp_spectrum(M_pinv @ rng5.normal(0, sigma_b5, N_Q)) * 1e6
                  for _ in range(N_TR5)])
off_mean, off_std = specs.mean(0), specs.std(0)
b_example = np.random.default_rng(7).normal(0, sigma_b5, N_Q) * 1e6

theory_floor = np.array([np.nan] + [sigma_b5 * 1e6 / M_col_norm(R, k)
                                    for k in range(1, KMAX + 1)])
kk = np.arange(0, KMAX + 1)

fig5, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) example white-offset realization vs s
axA.plot(s_bpm, b_example, "o-", color=GRAY, ms=3, lw=0.8)
axA.axhline(0, color="k", lw=0.4)
axA.set_xlabel(r"Ring position $s$ [m]")
axA.set_ylabel(r"BPM offset $b$ [$\mu$m]")
axA.set_title(fr"(a) White BPM offset ($\sigma_b={sigma_b5*1e6:.0f}\,\mu$m)")

# (b) recovered harmonic spectrum: offset floor vs true k=2 signal
ks = kk[1:9]
axB.errorbar(ks, off_mean[1:9], yerr=off_std[1:9], fmt="o", color=BLUE,
             capsize=3, ms=5, label=r"White offset (recovered, mean $\pm1\sigma$)")
axB.plot(ks, theory_floor[1:9], "--", color=GRAY, lw=1.0,
         label=r"Theory floor $\sigma_b/\|M_k\|$")
axB.plot([2], [a_sig[2]], "*", color=RED, ms=16,
         label=fr"True $k=2$ signal (${a_sig[2]:.0f}\,\mu$m)")
axB.axhline(A_sig5 * 1e6, color=RED, ls=":", lw=0.8, alpha=0.5)
axB.set_yscale("log")
axB.set_xlabel(r"Fourier mode $k$")
axB.set_ylabel(r"Recovered amplitude $|\hat a_k|$ [$\mu$m]")
axB.set_title(r"(b) Offset is broadband; $k=2$ signal stands clear")
axB.set_xticks(ks)
axB.legend(frameon=False, fontsize=8, loc="upper left")

fig5.suptitle(
    r"BPM-offset robustness: a white offset produces no spurious harmonic peak "
    r"and is lowest at $k=2$ (largest $\|M_k\|$)",
    fontsize=10, y=1.02)
fig5.tight_layout()
fig5.savefig("fig5_offset_whiteness.png", bbox_inches="tight")
plt.close(fig5)
print("fig5_offset_whiteness.png  ✓")
print(f"  k=2: signal {a_sig[2]:.1f} μm  vs  offset floor "
      f"{off_mean[2]:.2f} ± {off_std[2]:.2f} μm")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3 — Misalignment amplitude vs orbit norm
# ══════════════════════════════════════════════════════════════════════════════

truth = {2: (10e-6, 0.3), 4: (300e-6, 0.7), 6: (300e-6, 1.2), 8: (200e-6, 2.1)}
with open("table3_orbit.txt", "w") as f:
    f.write("% Table 3: Misalignment amplitude vs orbit norm contribution\n")
    f.write(f"  {'k':>3}  {'Misalignment [um]':>18}  {'Orbit norm [um]':>16}  {'Gain':>6}\n")
    for k, (A, _) in sorted(truth.items()):
        dq_k   = A * Fcos(k, N_Q)
        orb_nm = np.linalg.norm(R @ dq_k) * 1e6
        gain   = orb_nm / (A * 1e6)
        f.write(f"  {k:3d}  {A*1e6:>18.0f}  {orb_nm:>16.0f}  {gain:>6.1f}\n")
print("table3_orbit.txt  ✓")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("  SUMMARY (comparison with paper values)")
print("=" * 60)
print(f"  ||M_k=2|| = {Mk2_norm:.1f}  (paper: 167)")
print(f"  κ(R) = {kappa_R:.1f}  (paper: 248.7)")
print()
print("All output files generated.")
