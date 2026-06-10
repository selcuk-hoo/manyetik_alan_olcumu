#!/usr/bin/env python3
"""fig_8_rls_suppression.py — ŞEKİL 8: R-LS ile sahte-EDM bastırımı.

Makale senaryosu (test_combined_systematics ile aynı yapı):
  k=2 hedef: A=10 μm, φ=1.50 rad  (sabit gerçek değer)
  k=1: 30 μm, k=3: 25 μm  (diğer hedef modlar)
  k=4..10: 100–300 μm, rastgele genlik ve faz  ← kontaminantlar
  BPM ofseti: σ_b = 50 μm, rastgele gerçekleme

5 tohum (seed) için:
  1. Gerçek hizalama vektörünü oluştur.
  2. Yörünge y = R_dy @ dy; 50 μm BPM ofseti ekle.
  3. CLEAN (aday k=1..10) ile k=2 bileşenini geri çat.
  4. Sahte-EDM oranları karşılaştır:
       • Düzeltme öncesi  = C_false × A_{k=2,gerçek}   (sabit, 10 μm)
       • Düzeltme sonrası = C_false × ||a_true – â||    (artık k=2)
       • Bastırım oranı   = önceki / sonraki

Çıktı: fig_8_rls_suppression.png
"""
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from fourier_reconstruct import fodo_basis, clean_reconstruct

# ── Parametreler ─────────────────────────────────────────────────────────────

with open("params.json") as f:
    cfg = json.load(f)

N_Q     = 2 * int(cfg["nFODO"])        # 48 kuadrupol
ANTISYM = cfg.get("smooth_antisym_fodo", True)

R_dy = np.load("R_dy_1.npy")           # 48×48 yörünge tepki matrisi

# Sahte-EDM ölçek faktörü: k=2, A=10 μm → dS_y/dt = 1.42×10⁻⁹ rad/s  (mod tarama)
C_FALSE = 1.42e-9 / 10e-6              # (rad/s) / m

SIGMA_B = 50e-6                         # BPM ofseti σ (BBA sonrası) [m]
CANDIDATE_KS = list(range(1, 11))       # CLEAN aday modları k=1..10

# Sabit gerçek değerler (makalede olduğu gibi)
TRUTH_FIXED = {
    1: (30e-6,  0.80),
    2: (10e-6,  1.50),   # ← hedef k=2
    3: (25e-6,  0.30),
}

N_SEEDS = 5
SEEDS   = [101, 202, 303, 404, 505]

# ── Yardımcılar ──────────────────────────────────────────────────────────────

def build_dy(truth_dict):
    """truth {k:(A,phi)} → 48-elem. hizalama vektörü."""
    dy = np.zeros(N_Q)
    for k, (A, phi) in truth_dict.items():
        F, _ = fodo_basis(N_Q, [k], ANTISYM)
        dy += A * math.cos(phi) * F[:, 0] + A * math.sin(phi) * F[:, 1]
    return dy


def k2_amplitude_from_dy(dy):
    """dy'nin k=2 FODO-antisim Fourier genliği + katsayı vektörü."""
    F2, _ = fodo_basis(N_Q, [2], ANTISYM)
    a, _, _, _ = np.linalg.lstsq(F2, dy, rcond=None)
    return float(np.linalg.norm(a)), a


def clean_k2_estimate(y_meas):
    """CLEAN (k=1..10) ile y_meas'den â_{k=2} vektörünü tahmin et."""
    accum, _, F_cache = clean_reconstruct(
        [R_dy], [y_meas], CANDIDATE_KS, ANTISYM,
        gain=0.2, max_iter=400, tol=2e-4)
    a_k2 = accum[2]           # [â_cos, â_sin]
    return float(np.linalg.norm(a_k2)), a_k2


# ── 5 tohum için simülasyon ──────────────────────────────────────────────────

print("=" * 68)
print("  R-LS (CLEAN) SAHTE-EDM BASTIRIMI — 5 RASTGELE TOHUM")
print(f"  k=2 gerçek = 10 μm,  k=4..10 kontaminant: 100–300 μm (rastgele)")
print(f"  σ_b = {SIGMA_B*1e6:.0f} μm BPM ofseti  |  CLEAN aday k=1..10")
print("=" * 68)
print(f"  {'Tohum':>5}  {'A_k2_gerçek':>13}  {'A_k2_tahmin':>13}  "
      f"{'A_k2_artık':>12}  {'Bastırım':>10}")
print(f"  {'-'*5}  {'-'*13}  {'-'*13}  {'-'*12}  {'-'*10}")

rows = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)

    # Kontaminant modları (k=4..10) bu tohuma özgü rastgele
    truth = dict(TRUTH_FIXED)
    for k in range(4, 11):
        A_c   = float(rng.uniform(100e-6, 300e-6))
        phi_c = float(rng.uniform(0, 2 * math.pi))
        truth[k] = (A_c, phi_c)

    # Gerçek hizalama ve k=2 genliği
    dy = build_dy(truth)
    A_true, a_true = k2_amplitude_from_dy(dy)

    # Yörünge + BPM ofseti
    y_true = R_dy @ dy
    bpm    = rng.standard_normal(N_Q) * SIGMA_B
    y_meas = y_true + bpm

    # CLEAN ile k=2 tahmini
    A_hat, a_hat = clean_k2_estimate(y_meas)

    # Artık k=2 (düzeltme sonrası hata)
    a_residual  = a_true - a_hat
    A_residual  = float(np.linalg.norm(a_residual))

    false_before = A_true     * C_FALSE
    false_after  = A_residual * C_FALSE
    suppression  = false_before / false_after if false_after > 1e-20 else np.inf

    rows.append(dict(
        seed=seed,
        contaminant_rms=float(np.std([truth[k][0] for k in range(4, 11)])),
        A_true=A_true, A_hat=A_hat, A_res=A_residual,
        edm_before=false_before, edm_after=false_after,
        suppression=suppression))

    print(f"  {seed:>5}  {A_true*1e6:10.2f} μm  {A_hat*1e6:10.2f} μm  "
          f"{A_residual*1e6:9.2f} μm  {suppression:9.1f}×")

print(f"  {'─'*5}  {'─'*13}  {'─'*13}  {'─'*12}  {'─'*10}")
suppressions = [r['suppression'] for r in rows]
errors_pct   = [abs(r['A_hat'] - r['A_true']) / r['A_true'] * 100 for r in rows]
print(f"  {'ORT':>5}  {'':>13}  {'':>13}  {'':>12}  "
      f"{np.mean(suppressions):9.1f}×")
print(f"  k=2 tahmin hatası: {np.mean(errors_pct):.1f}% ± {np.std(errors_pct):.1f}%")
print()

# ── Çizim ─────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

labels = [f"#{r['seed']}" for r in rows]
x = np.arange(N_SEEDS)

# ── Sol panel: k=2 genlikleri ─────────────────────────────────────────────────
A_true_um = [r['A_true'] * 1e6 for r in rows]
A_hat_um  = [r['A_hat']  * 1e6 for r in rows]
A_res_um  = [r['A_res']  * 1e6 for r in rows]

w = 0.25
ax1.bar(x - w, A_true_um, w, color="steelblue",  label="Gerçek $k=2$  (10 μm sabit)", zorder=3)
ax1.bar(x,     A_hat_um,  w, color="darkorange", label="R-LS (CLEAN) tahmini",        zorder=3)
ax1.bar(x + w, A_res_um,  w, color="seagreen",   label="Artık  $|$gerçek – tahmin$|$", zorder=3)
ax1.set_yscale("log")
ax1.set_ylim(1e-2, 30)
ax1.set_ylabel("$k=2$ genliği  [μm]", fontsize=11)
ax1.set_xlabel("Rastgele gerçekleme (tohum)", fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.legend(fontsize=8.5, loc="lower right")
ax1.set_title("$k=2$ harmonik: gerçek / CLEAN tahmini / artık", fontsize=11)
ax1.yaxis.grid(True, which="both", ls="--", alpha=0.4)
ax1.set_axisbelow(True)

# Tahmin hatası yüzdesi üstüne yaz
for i, r in enumerate(rows):
    pct = abs(r['A_hat'] - r['A_true']) / r['A_true'] * 100
    ax1.annotate(f"{pct:.0f}%", xy=(x[i], max(r['A_hat'], r['A_true']) * 1e6 * 1.18),
                 ha="center", va="bottom", fontsize=7.5, color="darkorange")

# ── Sağ panel: Sahte-EDM ─────────────────────────────────────────────────────
edm_before_n = [r['edm_before'] * 1e9 for r in rows]
edm_after_n  = [r['edm_after']  * 1e9 for r in rows]

ax2.bar(x - 0.2, edm_before_n, 0.4, color="tomato",      label="Düzeltme öncesi", zorder=3)
ax2.bar(x + 0.2, edm_after_n,  0.4, color="forestgreen", label="Düzeltme sonrası", zorder=3)

for i, r in enumerate(rows):
    top = max(r['edm_before'], r['edm_after']) * 1e9 * 1.55
    ax2.annotate(f"{r['suppression']:.0f}×",
                 xy=(x[i], top), ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color="#333333")

ax2.set_yscale("log")
ax2.set_ylim(4e-12, 6e-9)
ax2.set_ylabel(r"$|dS_y/dt|$  [rad/s]", fontsize=11)
ax2.set_xlabel("Rastgele gerçekleme (tohum)", fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=10)
ax2.legend(fontsize=9, loc="upper right")
ax2.set_title("Sahte-EDM bastırımı: R-LS $k=2$ düzeltmesi", fontsize=11)
ax2.yaxis.grid(True, which="both", ls="--", alpha=0.4)
ax2.set_axisbelow(True)

mean_supp = np.mean(suppressions)
fig.suptitle(
    r"R-LS (CLEAN) ile $k=2$ sahte-EDM bastırımı — "
    rf"$A_{{k=2}}^{{\mathrm{{gerçek}}}}=10\,\mu$m, "
    rf"$k=4\!...\!10$ kontaminant 100–300$\,\mu$m, "
    rf"$\sigma_b=50\,\mu$m"
    f"  (ort. bastırım $\\approx{mean_supp:.0f}\\times$)",
    fontsize=10.5)
fig.tight_layout()
fig.savefig("fig_8_rls_suppression.png", dpi=140)
print("→ fig_8_rls_suppression.png kaydedildi")
