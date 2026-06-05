#!/usr/bin/env python3
"""test_combined_systematics.py — Sistematik hataların kombine etkisi (CLEAN geri çatım)

Kabul edilebilir limitlerde tüm hatalar aynı anda uygulanır (tek ölçüm):
  • BPM gürültüsü    σ_noise   = 5 μm
  • BPM ofseti       σ_offset  = 50 μm
  • Quad rulosu      θ_rms     = 1 mrad  (σ_dx = 100 μm yatay hata ile)
  • Gradyan hatası   σ_G       = 0.5%

Beş senaryo: her hata kaynağı ayrı ayrı + tümünün kombinasyonu.
Geri çatım: CLEAN algoritması (aday k=1..10).

N_MC = 200 Monte Carlo. Çıktı: combined_systematics.png
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


def build_dy():
    dy = np.zeros(N_Q)
    for k, (A, phi) in TRUTH.items():
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
N_MC      = 200

LEVELS = {
    "σ_noise":  5e-6,
    "σ_offset": 50e-6,
    "θ_rms":    1e-3,
    "σ_dx":     100e-6,
    "σ_G":      0.005,
}

dy    = build_dy()
y_nom = R_dy @ dy
mc    = np.random.default_rng(17)

# Hangi katkıların bireysel etkisini göstereceğiz
CONTRIB_NAMES = ["Gürültü", "Ofset", "Rulo", "Gradyan", "Kombine"]

# dA_mc[contrib_name][k] = list of signed % errors
dA_mc = {cn: {k: [] for k in K_TARGETS} for cn in CONTRIB_NAMES}

for _ in range(N_MC):
    noise  = mc.normal(0, LEVELS["σ_noise"],  N_Q)
    offset = mc.normal(0, LEVELS["σ_offset"], N_Q)
    dx     = mc.normal(0, LEVELS["σ_dx"],     N_Q)
    theta  = mc.normal(0, LEVELS["θ_rms"],    N_Q)
    eps_g  = mc.normal(0, LEVELS["σ_G"],      N_Q)

    x_co   = R_dx @ dx
    y_tilt = R_dy @ (2.0 * theta * x_co)
    y_grad = R_dy @ (eps_g * dy)

    scenarios = {
        "Gürültü":  y_nom + noise,
        "Ofset":    y_nom + offset,
        "Rulo":     y_nom + y_tilt,
        "Gradyan":  y_nom + y_grad,
        "Kombine":  y_nom + noise + offset + y_tilt + y_grad,
    }
    for cn, y_s in scenarios.items():
        fit = clean_fit(y_s)
        for k in K_TARGETS:
            A_t = TRUTH[k][0]
            A_f = fit.get(k, (0.0,))[0]
            dA_mc[cn][k].append((A_f - A_t) / A_t * 100)

def stats(lst):
    a = np.array(lst)
    return np.mean(a), np.std(a)

# ── Tablo ─────────────────────────────────────────────────────────────────
print("=" * 75)
print(f"  Kombine Sistematikler  (N={N_MC} MC)")
print(f"  Seviyeler: σ_noise={LEVELS['σ_noise']*1e6:.0f}μm  "
      f"σ_offset={LEVELS['σ_offset']*1e6:.0f}μm  "
      f"θ_rms={LEVELS['θ_rms']*1e3:.1f}mrad  σ_G={LEVELS['σ_G']*100:.1f}%")
print()
print(f"  {'Senaryo':28s}  " + "  ".join(f"k={k} ort±σ [%]" for k in K_TARGETS))
print("  " + "-"*70)
for cn in CONTRIB_NAMES:
    row = f"  {cn.replace(chr(10),' '):28s}"
    for k in K_TARGETS:
        m, s = stats(dA_mc[cn][k])
        row += f"  {m:>+6.2f}±{s:>5.2f}%"
    print(row)
print("=" * 75)

# ── Grafik: çubuk + hata barları ──────────────────────────────────────────
colors = {1: "tab:blue", 2: "tab:red", 3: "tab:green"}
cn_short = {
    "Gürültü":  f"BPM\ngürültü\n{LEVELS['σ_noise']*1e6:.0f}μm",
    "Ofset":    f"BPM\nofset\n{LEVELS['σ_offset']*1e6:.0f}μm",
    "Rulo":     f"Rulo\n{LEVELS['θ_rms']*1e3:.0f}mrad",
    "Gradyan":  f"Grad.\n{LEVELS['σ_G']*100:.1f}%",
    "Kombine":  "Kombine",
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax_i, k in enumerate(K_TARGETS):
    ax = axes[ax_i]
    labels = [cn_short[cn] for cn in CONTRIB_NAMES]
    means  = [stats(dA_mc[cn][k])[0] for cn in CONTRIB_NAMES]
    stds   = [stats(dA_mc[cn][k])[1] for cn in CONTRIB_NAMES]
    bar_colors = ["steelblue", "tab:orange", "purple", "olive", "tab:red"]
    xi = np.arange(len(CONTRIB_NAMES))
    ax.bar(xi, means, color=bar_colors, alpha=0.85, edgecolor="k", lw=0.5,
           yerr=stds, capsize=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(3.5, color="k", lw=0.5, ls=":")  # bireysel / kombine sınırı
    ax.set_xticks(xi)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ΔA/A [%] (ort ± 1σ)", fontsize=10)
    ax.set_title(f"k={k}  (A={TRUTH[k][0]*1e6:.0f}μm)", fontsize=11)

fig.suptitle(f"Sistematik hata katkıları — k=1,2,3 geri çatım  (N={N_MC} MC)  —  CLEAN\n"
             "Bireysel (4 çubuk) | Kombine",
             fontsize=11)
fig.tight_layout()
fig.savefig("combined_systematics.png", dpi=140)
print("\n→ combined_systematics.png kaydedildi")
