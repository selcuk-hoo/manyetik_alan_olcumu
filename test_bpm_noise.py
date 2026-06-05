#!/usr/bin/env python3
"""test_bpm_noise.py — BPM gürültüsünün k=1,2,3 ölçümüne etkisi

Her ölçümde bağımsız Gaussian BPM gürültüsü eklenir:
  y_meas_i = y_true_i + ε_i,   ε_i ~ N(0, σ²)

σ taraması: 1, 5, 20, 50 μm  (N=200 Monte Carlo her seviye için)
Gerçek misalignment: k=1 (30μm) + k=2 (10μm) + k=3 (25μm) + k=4..10 kirleticiler.
Geri çatım: CLEAN algoritması (aday k=1..10).

Çıktı: bpm_noise_scan.png
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
SIGMAS    = [1e-6, 5e-6, 20e-6, 50e-6]
N_MC      = 200

dy      = build_dy()
y_true  = R @ dy
mc_rng  = np.random.default_rng(99)

# ── Monte Carlo ──────────────────────────────────────────────────────────
# dA_mc[sigma_idx][k] = list of signed relative amplitude errors [%]
results = {k: {"mean": [], "std": [], "rms_abs": []} for k in K_TARGETS}

for sigma in SIGMAS:
    dA_by_k = {k: [] for k in K_TARGETS}
    for _ in range(N_MC):
        y_noisy = y_true + mc_rng.normal(0, sigma, N_Q)
        fit = clean_fit(y_noisy)
        for k in K_TARGETS:
            A_f, _ = fit.get(k, (0.0, 0.0))
            A_t    = TRUTH[k][0]
            dA_by_k[k].append((A_f - A_t) / A_t * 100)

    for k in K_TARGETS:
        arr = np.array(dA_by_k[k])
        results[k]["mean"].append(np.mean(arr))
        results[k]["std"].append(np.std(arr))
        results[k]["rms_abs"].append(np.sqrt(np.mean(arr**2)))

# ── Tablo ─────────────────────────────────────────────────────────────────
sig_labels = [f"{s*1e6:.0f} μm" for s in SIGMAS]
print("=" * 70)
print("  BPM Gürültüsü Taraması — Genlik Hatası ΔA/A [%]  (N=200 MC)")
print(f"  {'σ_BPM':>8}  " + "  ".join(f"k={k} ort±σ" for k in K_TARGETS))
print("  " + "-"*65)
for i, sigma in enumerate(SIGMAS):
    row = f"  {sigma*1e6:>5.0f} μm  "
    for k in K_TARGETS:
        m, s = results[k]["mean"][i], results[k]["std"][i]
        row += f"{m:>+6.2f}±{s:>5.2f}%  "
    print(row)
print("=" * 70)

# ── Grafik ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = {1: "tab:blue", 2: "tab:red", 3: "tab:green"}
sigmas_um = [s*1e6 for s in SIGMAS]

ax = axes[0]
for k in K_TARGETS:
    ax.errorbar(sigmas_um, results[k]["mean"], yerr=results[k]["std"],
                marker="o", label=f"k={k}  (A={TRUTH[k][0]*1e6:.0f}μm)",
                color=colors[k], lw=2, capsize=4)
ax.axhline(0, color="k", lw=0.7, ls="--")
ax.set_xscale("log")
ax.set_xlabel("BPM gürültü σ [μm]", fontsize=11)
ax.set_ylabel("Genlik hatası ΔA/A [%]  (ort ± 1σ)", fontsize=11)
ax.set_title("BPM gürültüsü — ortalama sapma", fontsize=11)
ax.legend(fontsize=10)

ax = axes[1]
for k in K_TARGETS:
    A_t_um = TRUTH[k][0]*1e6
    rms_abs = results[k]["rms_abs"]
    ax.plot(sigmas_um, rms_abs, "o-",
            label=f"k={k}  (A={A_t_um:.0f}μm)",
            color=colors[k], lw=2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("BPM gürültü σ [μm]", fontsize=11)
ax.set_ylabel("RMS genlik hatası |ΔA/A| [%]", fontsize=11)
ax.set_title("BPM gürültüsü — RMS hata (log-log)", fontsize=11)
ax.legend(fontsize=10)
# Teorik eğim 1 için referans çizgisi
sigma_ref = np.array([1, 50])
ax.plot(sigma_ref, sigma_ref * rms_abs[0] / sigmas_um[0],
        "k--", lw=0.8, label="∝ σ eğim-1")
ax.legend(fontsize=9)

fig.suptitle("k=1,2,3 geri çatım kalitesi — BPM gürültüsü taraması\n"
             "(tam baz k=1..10, N=200 MC)", fontsize=11)
fig.tight_layout()
fig.savefig("bpm_noise_scan.png", dpi=140)
print("→ bpm_noise_scan.png kaydedildi")
