#!/usr/bin/env python3
"""test_quad_tilt.py — Quad rulosu (s-ekseni etrafında θ açısı) etkisi

Quad bir θ açısıyla yuvarlandığında (s-ekseni etrafında) skew-quad terimi
oluşur. Yatay orbit x_co,j varsa dikey sapmaya katkı:

  Δy'_skew,j = 2·G·L·θ_j·x_co,j / p_magic

Bu, birim δy_j'nin verdiği normal tepmesiyle aynı yayılım kanalından geçer:

  COD_tilt ≈ R_dy · (2·θ · x_co)      [element-wise çarpım]

Yatay orbit ise yatay hizalama hatalarından (dx ~ N(0, σ_dx)) kaynaklanır:

  x_co = R_dx · dx_truth,   σ_dx = 100 μm (gerçekçi yatay hata)

θ_rms taraması: 0.1, 0.5, 1, 2, 5 mrad  (N=50 MC her seviye)
İki bileşen taranır: θ ve ayrıca σ_dx.

Çıktı: quad_tilt_scan.png
"""
import json, math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fourier_reconstruct import fodo_basis, amp_phase_from_coeffs, clean_reconstruct

with open("params.json") as f:
    cfg = json.load(f)
N_Q     = 2 * int(cfg["nFODO"])
ANTISYM = cfg.get("smooth_antisym_fodo", True)

R_dy = np.load("R_dy_1.npy")
R_dx = np.load("R_dx_1.npy")

RNG   = np.random.default_rng(42)
TRUTH = {
    1: (30e-6,  0.80),
    2: (10e-6,  1.50),
    3: (25e-6,  0.30),
}
for k in range(4, 11):
    TRUTH[k] = (float(RNG.uniform(100e-6, 300e-6)), float(RNG.uniform(0, 2*math.pi)))


def build_dy(truth=TRUTH):
    dy = np.zeros(N_Q)
    for k, (A, phi) in truth.items():
        F, _ = fodo_basis(N_Q, [k], ANTISYM)
        dy += A*math.cos(phi)*F[:, 0] + A*math.sin(phi)*F[:, 1]
    return dy


def clean_fit(y, candidate_ks=None):
    if candidate_ks is None:
        candidate_ks = list(range(1, 11))
    accum, _, F_cache = clean_reconstruct([R_dy], [y], candidate_ks, ANTISYM)
    out = {}
    for k in candidate_ks:
        F_k, meta_k = F_cache[k]
        a_k = accum[k]
        d = {kind: a_k[i] for i, (_, kind) in enumerate(meta_k)}
        ac, as_ = d.get('cos', 0.0), d.get('sin', 0.0)
        out[k] = (math.sqrt(ac**2 + as_**2), math.atan2(as_, ac))
    return out


K_TARGETS = [1, 2, 3]
THETA_RMS = [0.1e-3, 0.5e-3, 1e-3, 2e-3, 5e-3]  # [rad]
SIGMA_DX  = 100e-6    # yatay hizalama hatası RMS [m]
N_MC      = 50

dy_truth = build_dy()
y_bpm    = R_dy @ dy_truth
mc_rng   = np.random.default_rng(13)

# ── Monte Carlo ──────────────────────────────────────────────────────────
res = {k: {"mean": [], "std": []} for k in K_TARGETS}

for theta_rms in THETA_RMS:
    dA_by_k = {k: [] for k in K_TARGETS}
    for _ in range(N_MC):
        # Rastgele yatay hizalama hatası → yatay orbit
        dx   = mc_rng.normal(0, SIGMA_DX, N_Q)
        x_co = R_dx @ dx          # yatay kapalı yörünge [m]

        # Rastgele quad rulosu
        theta = mc_rng.normal(0, theta_rms, N_Q)

        # Sahteki dikey COD katkısı (skew-quad çapraz bağlaşım)
        dy_eff_tilt = 2.0 * theta * x_co            # [m], element-wise
        y_tilt      = R_dy @ dy_eff_tilt             # [m] BPM tepkisi

        y_meas = y_bpm + y_tilt
        fit    = clean_fit(y_meas)
        for k in K_TARGETS:
            A_f = fit.get(k, (0.0,))[0]
            A_t = TRUTH[k][0]
            dA_by_k[k].append((A_f - A_t) / A_t * 100)

    for k in K_TARGETS:
        a = np.array(dA_by_k[k])
        res[k]["mean"].append(np.mean(a))
        res[k]["std"].append(np.std(a))

# ── Tablo ─────────────────────────────────────────────────────────────────
print("=" * 65)
print(f"  Quad Rulosu Taraması  (σ_dx = {SIGMA_DX*1e6:.0f} μm, N={N_MC} MC)")
print(f"  {'θ_rms [mrad]':>14}  " +
      "  ".join(f"k={k} ort±σ [%]" for k in K_TARGETS))
print("  " + "-"*60)
for i, theta_rms in enumerate(THETA_RMS):
    row = f"  {theta_rms*1e3:>14.1f}  "
    for k in K_TARGETS:
        m, s = res[k]["mean"][i], res[k]["std"][i]
        row += f"{m:>+6.2f}±{s:>5.2f}%  "
    print(row)
print("=" * 65)
print(f"\n  Model: COD_tilt ≈ R_dy · (2·θ · x_co),  x_co = R_dx · dx")
print(f"  x_co RMS  ≈ {np.linalg.norm(R_dx @ mc_rng.normal(0,SIGMA_DX,N_Q)) / math.sqrt(N_Q) * 1e6:.1f} μm")

# ── Grafik ────────────────────────────────────────────────────────────────
thetas_mrad = [t*1e3 for t in THETA_RMS]
colors = {1: "tab:blue", 2: "tab:red", 3: "tab:green"}

fig, ax = plt.subplots(figsize=(7, 5))
for k in K_TARGETS:
    ax.errorbar(thetas_mrad, res[k]["mean"], yerr=res[k]["std"],
                marker="o", label=f"k={k}  (A={TRUTH[k][0]*1e6:.0f}μm)",
                color=colors[k], lw=2, capsize=4)
ax.axhline(0, color="k", lw=0.7, ls="--")
ax.axvline(1.0, color="gray", lw=1, ls=":", label="1 mrad referans")
ax.set_xlabel("Quad rulosu $\\theta_{\\mathrm{rms}}$ [mrad]", fontsize=11)
ax.set_ylabel("Genlik hatası ΔA/A [%]  (ort ± 1σ)", fontsize=11)
ax.set_title(f"Quad rulosu etkisi  (σ_dx = {SIGMA_DX*1e6:.0f} μm yatay hata, N={N_MC} MC)\n"
             r"Model: $\delta\mathrm{COD} = R_{dy}\cdot(2\theta\cdot x_{\mathrm{co}})$",
             fontsize=10)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig("quad_tilt_scan.png", dpi=140)
print("→ quad_tilt_scan.png kaydedildi")
