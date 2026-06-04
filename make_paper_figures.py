#!/usr/bin/env python3
"""make_paper_figures.py — figures and tables for paper_draft.tex.

Output files:
  fig1_falseedm_scan.png   – false-EDM rate vs k, bar chart (Table 1)
  fig2_orbit_gain.png      – ||RF_k|| orbit gain vs k (Table 2)
  fig3_mode_patterns.png   – white BPM offset: flat FODO-antisymmetric Fourier
                             spectrum (k=0..5), showing no k=2 enhancement
  fig4_sigma_model.png     – k=2 error budget: model uncertainty vs BPM offset
  fig5_offset_whiteness.png– white BPM offset → broadband recovered spectrum
                             (no spurious peak; lowest at k=2)
  fig6_matched_filter.png  – why offset is invisible: element-wise product
                             y_j × m̂_{k=2,j} cancels for offset, sums for signal
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

# Gömülü referans verisi: build_response_matrix.py çıktısı (gzip+base64).
# Lokal R_dy_1.npy yoksa veya sıfır matris içeriyorsa bu veri kullanılır.
_R_FALLBACK_B64 = (
    "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"
)


def _load_R():
    """Load R_dy_1.npy if present and valid; otherwise use the embedded reference."""
    import base64, gzip, io as _io
    try:
        Rmat = np.load("R_dy_1.npy")
        if np.max(np.abs(Rmat)) > 1e-10:
            print("R loaded from R_dy_1.npy")
            return Rmat
        raise ValueError("R_dy_1.npy contains a zero matrix")
    except (FileNotFoundError, ValueError) as exc:
        print(f"  Note: {exc}")
        print("  Using embedded reference R (run build_response_matrix.py to regenerate locally)")
        raw = gzip.decompress(base64.b64decode(_R_FALLBACK_B64))
        return np.load(_io.BytesIO(raw))


R = _load_R()
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
# kappa_spin: empirical dSy/dt / orbit_amplitude ratio from false_edm_mode_scan (k=1..5)
KAPPA_SPIN = 7e-6   # rad/s / m
with open("table2_gain.txt", "w") as f:
    f.write("% Table 2: Fourier mode orbit gain and false-EDM rate per 1 μm misalignment\n")
    f.write(f"% {'k':>3}  {'||RF_k||':>10}  {'||M_k||':>10}  {'dSy/dt [rad/s]':>16}\n")
    f.write(f"% {'':>3}  {'':>10}  {'':>10}  {'per 1 um ampl.':>16}\n")
    for k, rf, m in zip(k_list, rf_norms, m_norms):
        dsydt = KAPPA_SPIN * m * 1e-6   # 1 μm = 1e-6 m amplitude
        tag = "  % <-- dominant" if k == 2 else ""
        f.write(f"  {k:3d}  {rf:10.3f}  {m:10.1f}  {dsydt:16.3e}{tag}\n")
print("table2_gain.txt  ✓")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — White BPM offset: flat FODO-antisymmetric Fourier spectrum
#
# BPM electronic offsets enter the measurement directly (y = R·Δq + b),
# NOT through the response matrix R.  Because b is uncorrelated white noise,
# its projection onto any FODO-antisymmetric Fourier mode F_k is equally
# small — there is no k=2 enhancement.  This is why the method is robust:
# the response-matrix path amplifies the genuine k=2 signal by ||M_2||≈167,
# while b bypasses R entirely and contributes only its (unenhanced) projection.
#
# Left : one realisation of b(s) — random, structureless
# Right: Fourier amplitude |a_k| = sqrt(a_c²+a_s²) for k=0..5
#        over N_MC realisations.  All modes get ≈ σ_b√(π/48) ≈ 77 μm —
#        no k=2 peak, confirming b is white in the Fourier basis.
# ══════════════════════════════════════════════════════════════════════════════

C_ring = 2 * np.pi * 95.49                           # circumference ≈ 600 m
s_bpm  = np.linspace(0, C_ring, N_Q, endpoint=False) # BPM s-positions [m]

A_demo3  = 10e-6          # 10 μm misalignment amplitude (for signal orbit reference)
sigma_b3 = 300e-6
N_MC3    = 2000
rng3     = np.random.default_rng(42)
k_show   = [0, 1, 2, 3, 4, 5]

amp_mc = {k: [] for k in k_show}
for _ in range(N_MC3):
    b = rng3.normal(0, sigma_b3, N_Q)
    for k in k_show:
        Fc = Fcos(k, N_Q);  ac = Fc @ b / (Fc @ Fc)
        if k == 0:
            A_k = abs(ac)
        else:
            Fs = Fsin(k, N_Q);  as_ = Fs @ b / (Fs @ Fs)
            A_k = np.hypot(ac, as_)
        amp_mc[k].append(A_k * 1e6)

means3 = np.array([np.mean(amp_mc[k]) for k in k_show])
stds3  = np.array([np.std(amp_mc[k])  for k in k_show])
theory3 = sigma_b3 * 1e6 * np.sqrt(np.pi / 48)   # Rayleigh mean for k≥1 [μm]

b_ex3 = np.random.default_rng(7).normal(0, sigma_b3, N_Q) * 1e6  # example [μm]

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 4.2))

# ── left: example BPM offset realisation ──────────────────────────────────────
ax3a.plot(s_bpm, b_ex3, "o-", color=GRAY, ms=3, lw=0.8)
ax3a.axhline(0, color="k", lw=0.4)
ax3a.set_xlabel(r"Ring position $s$ [m]")
ax3a.set_ylabel(r"BPM offset $b\;[\mu\mathrm{m}]$")
ax3a.set_title(fr"(a) White BPM offset ($\sigma_b = {sigma_b3*1e6:.0f}\,\mu$m)")

# ── right: Fourier amplitude spectrum ─────────────────────────────────────────
# ── right: Fourier amplitude spectrum — log scale, three reference levels ─────
xpos3  = np.arange(len(k_show))
colors3 = [RED if k == 2 else BLUE for k in k_show]
ax3b.bar(xpos3, means3, color=colors3, width=0.6, alpha=0.75,
         edgecolor="white", linewidth=0.8)
ax3b.errorbar(xpos3, means3, yerr=stds3,
              fmt="none", color="k", capsize=4, linewidth=1.0)

# — k=2 signal orbit amplitude: A * ||M_k2||  (what BPMs actually read)
signal_orbit_um = A_demo3 * 1e6 * Mk2_norm       # 10 μm × 167 = 1670 μm
ax3b.axhline(signal_orbit_um, color=RED, ls="-", lw=1.8,
             label=fr"$k=2$ signal orbit: $A\|M_{{k=2}}\|={signal_orbit_um:.0f}\,\mu$m")

# — estimator residual: σ_b / ||M_k2||
est_floor = sigma_b3 * 1e6 / Mk2_norm       # ≈ 1.8 μm
ax3b.axhline(est_floor, color=GREEN, ls="--", lw=1.4,
             label=fr"Estimator floor: $\sigma_b/\|M_{{k=2}}\|={est_floor:.1f}\,\mu$m")

# — theoretical BPM offset Fourier level
ax3b.axhline(theory3, color=GRAY, ls=":", lw=1.2,
             label=fr"BPM offset $F_k$ level: $\approx{theory3:.0f}\,\mu$m")

ax3b.set_yscale("log")
ax3b.set_ylim(0.5, signal_orbit_um * 4)
ax3b.set_xticks(xpos3)
ax3b.set_xticklabels([f"$k={k}$" for k in k_show])
ax3b.set_ylabel(r"Amplitude [$\mu$m]  (log scale)")
ax3b.set_title("(b) Three scales: orbit signal ≫ offset $F_k$ level ≫ estimator floor")
ax3b.legend(frameon=False, fontsize=8.5)

fig3.suptitle(
    r"BPM offset $\mathbf{b}$ is white in the Fourier basis ($\approx77\,\mu$m/mode), "
    r"but the $k=2$ signal orbit is $A\|M_{k=2}\|=1670\,\mu$m — "
    r"$22\times$ larger"  "\n"
    r"The estimator further divides the offset by $\|M_{k=2}\|=167$, "
    r"leaving only $\approx1.8\,\mu$m contamination",
    fontsize=9.5, y=1.03)
fig3.tight_layout()
fig3.savefig("fig3_mode_patterns.png", bbox_inches="tight")
plt.close(fig3)
print("fig3_mode_patterns.png  ✓")
k2_idx3 = k_show.index(2)
print(f"  k=2 Fourier amplitude of white offset: "
      f"{means3[k2_idx3]:.1f} ± {stds3[k2_idx3]:.1f} μm  "
      f"(theory: {theory3:.1f} μm)")
print(f"  k=2 signal orbit: {signal_orbit_um:.0f} μm  "
      f"({signal_orbit_um/means3[k2_idx3]:.0f}× larger than offset F_k level)")

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
    f.write("% Table 3: Misalignment amplitude vs orbit norm and false-EDM contribution\n")
    f.write(f"  {'k':>3}  {'Misalignment':>14}  {'Orbit norm':>12}  "
            f"{'Gain':>6}  {'dSy/dt':>14}\n")
    f.write(f"  {'':>3}  {'[um]':>14}  {'[um]':>12}  "
            f"{'':>6}  {'[rad/s]':>14}\n")
    for k, (A, _) in sorted(truth.items()):
        dq_k   = A * Fcos(k, N_Q)
        orb_um = np.linalg.norm(R @ dq_k) * 1e6
        gain   = orb_um / (A * 1e6)
        dsydt  = KAPPA_SPIN * orb_um * 1e-6
        f.write(f"  {k:3d}  {A*1e6:>14.0f}  {orb_um:>12.0f}  "
                f"{gain:>6.1f}  {dsydt:>14.3e}\n")
print("table3_orbit.txt  ✓")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Matched-filter intuition: why BPM offset is nearly invisible
#
# The estimator does NOT measure how large the orbit is.
# It measures how closely the orbit RESEMBLES the k=2 fingerprint.
#
#   â_{k=2} = (y · m̂_{k=2}) / ||M_{k=2}||
#
# Left:   orbit measurement y(s) — signal orbit or random offset
# Middle: k=2 template m̂_{k=2}(s) = M_{k=2}/||M_{k=2}||  (same for both rows)
# Right:  element-wise product y_j × m̂_j — fills area above/below zero
#         → all positive for signal (large sum) → alternates for offset (cancels)
# ══════════════════════════════════════════════════════════════════════════════

A6      = 10e-6
Mc_raw  = R @ Fcos(2, N_Q)
Mc_norm = np.linalg.norm(Mc_raw)
m_hat6  = Mc_raw / Mc_norm                                  # unit-norm template

# True k=2 signal orbit (all products positive)
y_sig_um = A6 * Mc_raw * 1e6                                # μm
prod_sig  = y_sig_um * m_hat6                               # μm (all ≥ 0)
est_sig   = np.sum(prod_sig) / Mc_norm                      # μm ≈ 10

# Random white BPM offset (products alternate → cancel)
b6_um    = np.random.default_rng(7).normal(0, 300e-6, N_Q) * 1e6  # μm
prod_off  = b6_um * m_hat6                                  # μm (mixed sign)
est_off   = np.sum(prod_off) / Mc_norm                      # μm ≈ small

cases = [
    (y_sig_um,  prod_sig, est_sig, RED,
     fr"True $k=2$ signal orbit  (peak $\approx{np.max(np.abs(y_sig_um)):.0f}\,\mu$m)"),
    (b6_um,     prod_off, est_off, GRAY,
     fr"Random BPM offset  ($\sigma_b = 300\,\mu$m)"),
]

fig6, axes6 = plt.subplots(2, 3, figsize=(12, 6))
fig6.subplots_adjust(hspace=0.52, wspace=0.38)

for row, (y_in, prod, est, col, input_title) in enumerate(cases):

    # ── left: input y(s) ────────────────────────────────────────────────────
    ax = axes6[row, 0]
    ax.fill_between(s_bpm, y_in, alpha=0.25, color=col)
    ax.plot(s_bpm, y_in, "-", color=col, lw=1.0)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel(r"[$\mu$m]", fontsize=9)
    ax.set_title(input_title, fontsize=9)
    ax.tick_params(labelsize=8)

    # ── middle: template m̂_{k=2} (same for both rows) ───────────────────────
    ax = axes6[row, 1]
    ax.fill_between(s_bpm, m_hat6, alpha=0.15, color=BLUE)
    ax.plot(s_bpm, m_hat6, "-", color=BLUE, lw=1.3)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel(r"$\hat{m}_{k=2}$ [a.u.]", fontsize=9)
    ax.tick_params(labelsize=8)
    if row == 0:
        ax.set_title(r"$k=2$ fingerprint  $\hat{m}_{k=2} = M_{k=2}/\|M_{k=2}\|$",
                     fontsize=9)
    else:
        ax.set_title("Same fingerprint", fontsize=9)

    # ── right: element-wise product y_j × m̂_j ───────────────────────────────
    ax = axes6[row, 2]
    ax.fill_between(s_bpm, prod, 0,
                    where=(prod >= 0), color=GREEN, alpha=0.70, label="positive")
    ax.fill_between(s_bpm, prod, 0,
                    where=(prod < 0),  color=RED,   alpha=0.70, label="negative")
    ax.plot(s_bpm, prod, "k-", lw=0.4)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(r"$y_j\,\hat{m}_j\;[\mu$m]", fontsize=9)
    ax.tick_params(labelsize=8)
    if row == 0:
        ax.set_title("Products: all positive → large sum", fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    else:
        ax.set_title("Products: alternating → nearly cancel", fontsize=9)
    ax.text(0.97, 0.97, fr"$\hat{{a}}_{{k=2}} = {est:.1f}\,\mu$m",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            color=GREEN if abs(est) > 5 else "0.3",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.92))

# x-labels on bottom row only
for c in range(3):
    axes6[1, c].set_xlabel(r"Ring position $s$ [m]", fontsize=9)

fig6.suptitle(
    r"The estimator does not ask \textit{how large} is the orbit — "
    r"it asks \textit{how closely does the orbit resemble the $k=2$ fingerprint?}"  "\n"
    r"A genuine $k=2$ signal matches perfectly (products all positive). "
    r"A random offset does not match (products cancel). "
    r"That is why BPM offset is nearly invisible.",
    fontsize=9.5, y=1.02)

fig6.savefig("fig6_matched_filter.png", bbox_inches="tight")
plt.close(fig6)
print("fig6_matched_filter.png  ✓")
print(f"  Signal estimate : {est_sig:.2f} μm  (true: {A6*1e6:.1f} μm)")
print(f"  Offset estimate : {est_off:.3f} μm  (floor: {300e-6*1e6/Mc_norm:.3f} μm)")

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
