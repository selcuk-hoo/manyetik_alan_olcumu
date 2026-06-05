#!/usr/bin/env python3
"""test_reconstruction_quality.py — k=1,2,3 geri çatım kalitesi

Senaryo: k=1 (30μm) + k=2 (10μm) + k=3 (25μm) hedef harmonikleri;
         k=4..10 her biri 100–300μm büyük kirleticiler.

Üç yöntem karşılaştırılır:
  A) Eksik baz (yalnız k=1,2,3): kirletici sızıntısı varsa ortaya çıkar
  B) Tam baz (k=1..10): kirleticiler açıkça modellenir → temiz ayrıştırma
  C) CLEAN algoritması: dominant harmonikler sırayla soyulur

Gürültüsüz ve σ_noise = 1 μm BPM gürültülü durumlar için 100-iter MC.

Çıktı: reconstruction_quality.png
"""
import json, math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fourier_reconstruct import (fodo_basis, amp_phase_from_coeffs,
                                  clean_reconstruct)

with open("params.json") as f:
    cfg = json.load(f)
N_Q     = 2 * int(cfg["nFODO"])
ANTISYM = cfg.get("smooth_antisym_fodo", True)

R = np.load("R_dy_1.npy")

# ── gerçek misalignment ───────────────────────────────────────────────────
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


def fit(R_mat, y, k_fit):
    F, meta = fodo_basis(N_Q, k_fit, ANTISYM)
    M = R_mat @ F
    a, _, _, _ = np.linalg.lstsq(M, y, rcond=None)
    return amp_phase_from_coeffs(a, meta)


def errors(fit_res, k_targets=(1, 2, 3)):
    out = {}
    for k in k_targets:
        A_f, p_f = fit_res.get(k, (0.0, 0.0))
        A_t, p_t = TRUTH.get(k, (0.0, 0.0))
        dA = (A_f - A_t) / A_t * 100 if A_t > 0 else float('nan')
        dp = abs(((p_f - p_t + math.pi) % (2*math.pi)) - math.pi)
        out[k] = (dA, dp, A_f)
    return out


K_TARGETS   = [1, 2, 3]
K_FULL      = list(range(1, 11))
N_MC        = 100
SIGMA_NOISE = 1e-6          # BPM gürültü σ [m]

dy_truth  = build_dy()
y_bpm_clean = R @ dy_truth  # gürültüsüz COD

# ── CLEAN için yardımcı ──────────────────────────────────────────────────
def clean_fit(y, candidate_ks=K_FULL):
    accum, _, F_cache = clean_reconstruct([R], [y], candidate_ks, ANTISYM)
    result = {}
    for k in candidate_ks:
        F_k, meta_k = F_cache[k]
        a_k = accum[k]
        d = {kind: a_k[i] for i, (_, kind) in enumerate(meta_k)}
        ac, as_ = d.get('cos', 0.0), d.get('sin', 0.0)
        result[k] = (math.sqrt(ac**2 + as_**2), math.atan2(as_, ac))
    return result


# ── Gürültüsüz durum ──────────────────────────────────────────────────────
res_A = errors(fit(R, y_bpm_clean, K_TARGETS))
res_B = errors(fit(R, y_bpm_clean, K_FULL))
res_C = errors(clean_fit(y_bpm_clean))

# ── Monte Carlo: σ_noise = 1 μm ──────────────────────────────────────────
mc_rng = np.random.default_rng(7)
dA_mc = {m: {k: [] for k in K_TARGETS} for m in ("A", "B", "C")}
for _ in range(N_MC):
    noise = mc_rng.normal(0, SIGMA_NOISE, N_Q)
    y_noisy = y_bpm_clean + noise
    for m, k_fit, fn in [("A", K_TARGETS, lambda y: fit(R, y, K_TARGETS)),
                          ("B", K_FULL,    lambda y: fit(R, y, K_FULL)),
                          ("C", None,      clean_fit)]:
        e = errors(fn(y_noisy))
        for k in K_TARGETS:
            dA_mc[m][k].append(e[k][0])

mc_mean = {m: {k: np.mean(v) for k, v in dA_mc[m].items()} for m in ("A","B","C")}
mc_std  = {m: {k: np.std(v)  for k, v in dA_mc[m].items()} for m in ("A","B","C")}

# ── Tablo ─────────────────────────────────────────────────────────────────
print("=" * 72)
print("  Geri Çatım Kalitesi — gürültüsüz")
print(f"  {'k':>2}  {'Gerçek [μm]':>11}  {'A: k=1,2,3 baz':>16}  "
      f"{'B: tam baz':>10}  {'C: CLEAN':>8}")
print("  " + "-"*68)
for k in K_TARGETS:
    A_true = TRUTH[k][0]*1e6
    dA_A, _, A_A = res_A[k]
    dA_B, _, A_B = res_B[k]
    dA_C, _, A_C = res_C[k]
    print(f"  {k:>2}  {A_true:>8.1f} μm  "
          f"{A_A*1e6:>7.2f} μm ({dA_A:>+6.2f}%)  "
          f"{A_B*1e6:>5.2f} μm ({dA_B:>+6.2f}%)  "
          f"{A_C*1e6:>5.2f} μm ({dA_C:>+6.2f}%)")
print()
print(f"  Monte Carlo (σ_noise = {SIGMA_NOISE*1e6:.0f} μm, N={N_MC}):")
print(f"  {'k':>2}  {'A: ort±σ [%]':>16}  {'B: ort±σ [%]':>16}  {'C: ort±σ [%]':>16}")
for k in K_TARGETS:
    print(f"  {k:>2}  {mc_mean['A'][k]:>+7.2f}±{mc_std['A'][k]:>5.2f}%  "
          f"{mc_mean['B'][k]:>+7.2f}±{mc_std['B'][k]:>5.2f}%  "
          f"{mc_mean['C'][k]:>+7.2f}±{mc_std['C'][k]:>5.2f}%")
print("=" * 72)

# ── Grafik ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
k_labels = [f"k={k}" for k in K_TARGETS]
x = np.arange(len(K_TARGETS))
w = 0.25

for ax, title, data_clean, mc_m, noise_label in [
    (axes[0], "Gürültüsüz", (res_A, res_B, res_C), None, ""),
    (axes[1], f"σ_noise = {SIGMA_NOISE*1e6:.0f} μm  (N={N_MC})", None, mc_mean, True),
]:
    if data_clean is not None:
        for i, (m, label, c) in enumerate([
            ("A", "Eksik baz\n(k=1,2,3)", "tab:blue"),
            ("B", "Tam baz\n(k=1..10)",   "tab:green"),
            ("C", "CLEAN",                 "tab:orange"),
        ]):
            vals = [data_clean[i][k][0] for k in K_TARGETS]
            ax.bar(x + (i-1)*w, vals, w*0.9, label=label, color=c, alpha=0.85)
    else:
        colors = ["tab:blue", "tab:green", "tab:orange"]
        labels = ["Eksik baz\n(k=1,2,3)", "Tam baz\n(k=1..10)", "CLEAN"]
        for i, m_key in enumerate(["A", "B", "C"]):
            means = [mc_mean[m_key][k] for k in K_TARGETS]
            stds  = [mc_std[m_key][k]  for k in K_TARGETS]
            ax.bar(x + (i-1)*w, means, w*0.9, label=labels[i],
                   color=colors[i], alpha=0.85, yerr=stds, capsize=4)

    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(k_labels)
    ax.set_ylabel("Genlik hatası ΔA/A [%]", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(min(-5, ax.get_ylim()[0]*1.1), max(5, ax.get_ylim()[1]*1.1))

fig.suptitle("k=1,2,3 geri çatım kalitesi  (k=4..10 kirletici 100–300μm)", fontsize=12)
fig.tight_layout()
fig.savefig("reconstruction_quality.png", dpi=140)
print("\n→ reconstruction_quality.png kaydedildi")
