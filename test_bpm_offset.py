#!/usr/bin/env python3
"""test_bpm_offset.py — BPM sistematik ofseti etkisi (CLEAN geri çatım)

BPM ofseti: her BPM'e sabit ama rastgele bir önyargı eklenir:
  y_meas_i = y_true_i + b_i,   b_i ~ N(0, σ_offset²)  [ölçümden ölçüme değişmez]

Ölçüm: tek orbit ölçümü, CLEAN algoritması (k=1..10 aday seti).

σ_offset taraması: 10, 50, 200, 500 μm  (N=100 MC her seviye)
Referans: sec:whiteness analitik tahmin — δA/A ≈ σ_b / ‖M_{k=2}‖ = σ_b / 167

Çıktı: bpm_offset_scan.png
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
SIGMAS    = [10e-6, 50e-6, 200e-6, 500e-6]
N_MC      = 100

dy     = build_dy()
y_true = R @ dy
mc_rng = np.random.default_rng(77)

# ── Analitik referans: ‖M_{k=2}‖ = 167 → δA ≈ σ_b / 167 ────────────────
F2, _ = fodo_basis(N_Q, [2], ANTISYM)
M2    = R @ F2
norm_M2 = np.linalg.norm(M2[:, 0])   # k=2 sütun normu

res = {k: {"mean": [], "std": [], "analytic": []} for k in K_TARGETS}

for sigma in SIGMAS:
    dA_by_k = {k: [] for k in K_TARGETS}
    for _ in range(N_MC):
        b = mc_rng.normal(0, sigma, N_Q)   # sabit ofset
        fit = clean_fit(y_true + b)
        for k in K_TARGETS:
            A_t = TRUTH[k][0]
            A_f = fit.get(k, (0.0,))[0]
            dA_by_k[k].append((A_f - A_t) / A_t * 100)

    for k in K_TARGETS:
        a = np.array(dA_by_k[k])
        # Analitik k=2 önyargı tahmini: σ_b / ‖M_{k=2}‖ (beyaz ofset için σ)
        Fk, _ = fodo_basis(N_Q, [k], ANTISYM)
        norm_Mk = np.linalg.norm((R @ Fk)[:, 0])
        analytic_std = sigma / norm_Mk / TRUTH[k][0] * 100
        res[k]["mean"].append(np.mean(a))
        res[k]["std"].append(np.std(a))
        res[k]["analytic"].append(analytic_std)

# ── Tablo ─────────────────────────────────────────────────────────────────
print("=" * 75)
print(f"  BPM Ofseti Taraması  (N={N_MC} MC)  —  CLEAN geri çatım")
print(f"  Analitik tahmin: σ(ΔA/A) ≈ σ_b / ‖M_k‖;  ‖M_{{k=2}}‖ = {norm_M2:.1f}")
for sigma in SIGMAS:
    print(f"\n  σ_offset = {sigma*1e6:.0f} μm")
    print(f"    {'k':>2}  {'MC ort±σ [%]':>18}  {'Analitik σ [%]':>15}")
    i = SIGMAS.index(sigma)
    for k in K_TARGETS:
        m, s = res[k]["mean"][i], res[k]["std"][i]
        an   = res[k]["analytic"][i]
        print(f"    {k:>2}  {m:>+7.2f} ± {s:>6.2f}%  {an:>13.2f}%")
print("=" * 75)

# ── Grafik ────────────────────────────────────────────────────────────────
sigmas_um = [s*1e6 for s in SIGMAS]
colors    = {1: "tab:blue", 2: "tab:red", 3: "tab:green"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for k in K_TARGETS:
    ax.errorbar(sigmas_um, res[k]["mean"], yerr=res[k]["std"],
                marker="o", label=f"k={k}  MC",
                color=colors[k], lw=2, capsize=4)
ax.axhline(0, color="k", lw=0.7, ls="--")
ax.set_xscale("log")
ax.set_xlabel("BPM ofset σ [μm]", fontsize=11)
ax.set_ylabel("Genlik hatası ΔA/A [%]  (ort ± 1σ)", fontsize=11)
ax.set_title("BPM ofseti — ortalama sapma (CLEAN)", fontsize=11)
ax.legend(fontsize=10)

ax = axes[1]
for k in K_TARGETS:
    ax.plot(sigmas_um, res[k]["std"], "o-", color=colors[k], lw=2, label=f"k={k} MC σ")
    ax.plot(sigmas_um, res[k]["analytic"], "--", color=colors[k], alpha=0.6, label=f"k={k} analitik")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("BPM ofset σ [μm]", fontsize=11)
ax.set_ylabel("RMS genlik hatası |ΔA/A| [%]", fontsize=11)
ax.set_title("BPM ofseti — RMS hata vs. analitik σ_b/‖M_k‖", fontsize=11)
ax.legend(fontsize=8, ncol=2)

fig.suptitle("BPM ofseti etkisi — CLEAN geri çatım  (N=100 MC)\n"
             r"Analitik tahmin: $\sigma(\Delta A/A)\approx\sigma_b/\|M_k\|$",
             fontsize=11)
fig.tight_layout()
fig.savefig("bpm_offset_scan.png", dpi=140)
print("→ bpm_offset_scan.png kaydedildi")
