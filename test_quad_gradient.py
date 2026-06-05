#!/usr/bin/env python3
"""test_quad_gradient.py — Quad gradyan hatası etkisi

Her quad'ın nominal gradyanına bağımsız rastgele hata eklenir:
  G_j_gerçek = G_j_nominal · (1 + ε_j),   ε_j ~ N(0, σ_G²)

Tepki matrisi modeli (sütun ölçekleme):
  R_perturbed[:, j] ≈ R_nominal[:, j] · (1 + ε_j)

Bozulmuş COD:
  y_BPM_bozulmuş = R_nominal @ ((1 + ε) · Δy_truth)
                 = y_BPM_nominal + R_nominal @ (ε · Δy_truth)

İkinci terim, yalnızca mevcut R_dy_1.npy ile hesaplanır; ek tracking gerekmez.
Nominal R ile geri çatımdaki sapma bu fazladan katkıdan kaynaklanır.

σ_G taraması: 0.1%, 0.5%, 2%, 5%  (N=200 MC)

Çıktı: quad_gradient_scan.png
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

R = np.load("R_dy_1.npy")

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
    accum, _, F_cache = clean_reconstruct([R], [y], candidate_ks, ANTISYM)
    out = {}
    for k in candidate_ks:
        F_k, meta_k = F_cache[k]
        a_k = accum[k]
        d = {kind: a_k[i] for i, (_, kind) in enumerate(meta_k)}
        ac, as_ = d.get('cos', 0.0), d.get('sin', 0.0)
        out[k] = (math.sqrt(ac**2 + as_**2), math.atan2(as_, ac))
    return out


K_TARGETS = [1, 2, 3]
SIGMA_G   = [0.001, 0.005, 0.02, 0.05]   # fraksiyonel gradyan hatası
N_MC      = 200

dy      = build_dy()
y_nom   = R @ dy        # gürültüsüz nominal COD
mc_rng  = np.random.default_rng(55)

res = {k: {"mean": [], "std": []} for k in K_TARGETS}

for sigma_g in SIGMA_G:
    dA_by_k = {k: [] for k in K_TARGETS}
    for _ in range(N_MC):
        eps        = mc_rng.normal(0, sigma_g, N_Q)   # per-quad fractional error
        # Sütun ölçekleme: y_bozulmuş = R@dy + R@(eps*dy)
        delta_y    = R @ (eps * dy)
        y_meas     = y_nom + delta_y
        fit        = clean_fit(y_meas)
        for k in K_TARGETS:
            A_f = fit.get(k, (0.0,))[0]
            A_t = TRUTH[k][0]
            dA_by_k[k].append((A_f - A_t) / A_t * 100)

    for k in K_TARGETS:
        a = np.array(dA_by_k[k])
        res[k]["mean"].append(np.mean(a))
        res[k]["std"].append(np.std(a))

print("=" * 65)
print(f"  Quad Gradyan Hatası Taraması  (N={N_MC} MC)  —  CLEAN geri çatım")
print(f"  {'σ_G [%]':>10}  " + "  ".join(f"k={k} ort±σ [%]" for k in K_TARGETS))
print("  " + "-"*60)
for i, sg in enumerate(SIGMA_G):
    row = f"  {sg*100:>8.1f}%  "
    for k in K_TARGETS:
        m, s = res[k]["mean"][i], res[k]["std"][i]
        row += f"{m:>+6.2f}±{s:>5.2f}%  "
    print(row)
print("=" * 65)

# ── Grafik ────────────────────────────────────────────────────────────────
sg_pct  = [s*100 for s in SIGMA_G]
colors  = {1: "tab:blue", 2: "tab:red", 3: "tab:green"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for k in K_TARGETS:
    ax.errorbar(sg_pct, res[k]["mean"], yerr=res[k]["std"],
                marker="o", label=f"k={k}  (A={TRUTH[k][0]*1e6:.0f}μm)",
                color=colors[k], lw=2, capsize=4)
ax.axhline(0, color="k", lw=0.7, ls="--")
ax.set_xlabel("Gradyan hatası σ_G [%]", fontsize=11)
ax.set_ylabel("Genlik hatası ΔA/A [%]  (ort ± 1σ)", fontsize=11)
ax.set_title("Quad gradyan hatası — ortalama sapma", fontsize=11)
ax.legend(fontsize=10)

ax = axes[1]
for k in K_TARGETS:
    ax.plot(sg_pct, res[k]["std"], "o-", color=colors[k],
            label=f"k={k}  (A={TRUTH[k][0]*1e6:.0f}μm)", lw=2)
ax.set_xlabel("Gradyan hatası σ_G [%]", fontsize=11)
ax.set_ylabel("Genlik belirsizliği σ(ΔA/A) [%]", fontsize=11)
ax.set_title("Quad gradyan hatası — RMS belirsizlik  (CLEAN)", fontsize=11)
ax.legend(fontsize=9)

fig.suptitle("Quad gradyan hatası etkisi  (sütun ölçekleme modeli, N=200 MC)  —  CLEAN geri çatım", fontsize=11)
fig.tight_layout()
fig.savefig("quad_gradient_scan.png", dpi=140)
print("→ quad_gradient_scan.png kaydedildi")
