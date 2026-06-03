#!/usr/bin/env python3
"""make_paper_figures.py — paper_draft.tex için tüm grafik ve tablo çıktıları.

Üretilen dosyalar:
  fig1_falseedm_scan.png   – k=0..5 yanlış-EDM hızı (çubuk grafik, Tablo 1)
  fig2_orbit_gain.png      – ||RF_k|| Fourier modu kazanımları (Tablo 2)
  fig3_offset_scaling.png  – BPM ofset ölçeklenmesi: k=2 hatası vs σ_b
  fig4_sigma_model.png     – Model hatası toleransı: k=2 hatası vs σ_R
  table2_gain.txt          – Tablo 2 LaTeX satırları
  table3_orbit.txt         – Tablo 3 ham sayılar

Yöntem notu:
  Projeksiyon tahmincisi DOğRU formülasyon:
    dq = A_cos * F_k_cos + A_sin * F_k_sin   (normalize EDİLMEMİŞ F_k)
    y  = R @ dq + b
    [A_cos, A_sin] = lstsq([M_c, M_s], y)   M_c=R@F_c, M_s=R@F_s
    A_fit = sqrt(A_cos²+A_sin²)   [metre cinsinden, dq ile aynı birim]
  Teori: |δA| ≤ ||b||/||M_k||  burada ||M_k||=sqrt(24)*||RF_k_unit|| ≈167

Kullanım:
  python3 make_paper_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

# ── stil ────────────────────────────────────────────────────────────────────

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

# ── Fourier baz (normalize EDİLMEMİŞ — birim: [1]) ──────────────────────────

def Fcos(k, n_q=48):
    """FODO-antisimetrik cosinus bileşeni (normalize edilmemiş)."""
    j = np.arange(n_q)
    if k == 0:
        return (-1.0) ** j
    return (-1.0) ** j * np.cos(2 * np.pi * k * (j // 2) / (n_q // 2))

def Fsin(k, n_q=48):
    """FODO-antisimetrik sinüs bileşeni (normalize edilmemiş)."""
    j = np.arange(n_q)
    return (-1.0) ** j * np.sin(2 * np.pi * k * (j // 2) / (n_q // 2))

def RF_unit_norm(R, k):
    """||RF_k|| – birim-normalize F_k ile (makale Table 2 değerleri)."""
    Fc = Fcos(k, R.shape[0])
    Fc = Fc / np.linalg.norm(Fc)
    return np.linalg.norm(R @ Fc)

def M_col_norm(R, k):
    """||M_k|| = sqrt(N_cells) * ||RF_k|| ≈ 167 için k=2."""
    return np.sqrt(R.shape[0] // 2) * RF_unit_norm(R, k)

# ── projeksiyon tahmincisi ───────────────────────────────────────────────────

def project_amplitude(y, R, k):
    """k. harmoniğin genliğini yörünge ölçümünden kestir (A metreler).

    Formül: [a_c, a_s] = lstsq([M_c, M_s], y)
            A = sqrt(a_c² + a_s²)
    M_c = R @ F_c_unnorm,  M_s = R @ F_s_unnorm
    """
    Mc = R @ Fcos(k, R.shape[0])
    Ms = R @ Fsin(k, R.shape[0])
    M2 = np.column_stack([Mc, Ms])
    a2, _, _, _ = np.linalg.lstsq(M2, y, rcond=None)
    return float(np.sqrt(a2[0]**2 + a2[1]**2)), float(np.arctan2(a2[1], a2[0]))

# ── test yörüngesi üretici ───────────────────────────────────────────────────

def make_orbit(R, k_true=2, A_true=10e-6, phi_true=0.3,
               contaminants=None, b_sigma=0.0, rng=None):
    """Bilinen harmonik düzeni + BPM ofseti içeren test yörüngesi.

    dq = A_true * (cos(phi)*Fcos(k) + sin(phi)*Fsin(k))
       + Σ_c A_c*(cos(φ_c)*Fcos(k_c) + sin(φ_c)*Fsin(k_c))
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if contaminants is None:
        # Tipik test durumu: k=4,6,8 büyük kontaminantlar (makale Sec 4.2)
        contaminants = {4: (300e-6, 0.7), 6: (300e-6, 1.2), 8: (200e-6, 2.1)}

    n_q = R.shape[0]
    dq = A_true * (np.cos(phi_true) * Fcos(k_true, n_q)
                   + np.sin(phi_true) * Fsin(k_true, n_q))
    for k_c, (A_c, phi_c) in contaminants.items():
        dq += A_c * (np.cos(phi_c) * Fcos(k_c, n_q)
                     + np.sin(phi_c) * Fsin(k_c, n_q))

    b = rng.normal(0, b_sigma, n_q) if b_sigma > 0 else np.zeros(n_q)
    return R @ dq + b

# ── tepki matrisi ─────────────────────────────────────────────────────────────

R = np.load("R_dy_1.npy")   # 48×48, birim: m/m (düzeltilmiş)
N_Q = R.shape[0]

svs = np.linalg.svd(R, compute_uv=False)
kappa_R = svs[0] / svs[-1]
Mk2_norm = M_col_norm(R, 2)

print(f"R: {R.shape},  κ(R) = {kappa_R:.1f},  ||M_k=2|| = {Mk2_norm:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# ŞEKİL 1 — Yanlış-EDM hızı vs k  (stroboskopik spin takibi sonuçları)
# ══════════════════════════════════════════════════════════════════════════════

# false_edm_mode_scan.py — 10 ms, A=10 μm, CO-launch + stroboskopik
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
       edgecolor="black", linewidth=1.3, label="$k=2$ (baskın)")
for k in ks:
    sign = "+" if dsydt[k] > 0 else "−"
    ax.text(k, abs(dsydt[k]) * 1e9 + 0.02, sign,
            ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Fourier modu $k$")
ax.set_ylabel(r"$|dS_y/dt|$ [$10^{-9}$ rad/s]")
ax.set_title("(a) Yanlış-EDM hızı")
ax.set_xticks(ks)
ax.legend(frameon=False)

ax = axes[1]
ax.bar(ks, [co_amp[k] for k in ks],
       color=bar_col, width=0.6, edgecolor="white", linewidth=0.8)
ax.bar(2, co_amp[2], color=RED, width=0.6, edgecolor="black", linewidth=1.3)
ax.set_xlabel("Fourier modu $k$")
ax.set_ylabel("CO genliği [mm]")
ax.set_title("(b) Kapalı yörünge genliği")
ax.set_xticks(ks)

fig.suptitle(r"$A=10\,\mu$m tek harmonik bozunumu, gerçek EDM kapalı",
             fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig("fig1_falseedm_scan.png", bbox_inches="tight")
plt.close(fig)
print("fig1_falseedm_scan.png  ✓")

# ══════════════════════════════════════════════════════════════════════════════
# ŞEKİL 2 — Fourier modu kazanımları
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
           label=f"Ortalama = {np.mean(rf_norms):.1f}")
ax.set_xticks(xpos)
ax.set_xticklabels([str(k) for k in k_list])
ax.set_xlabel("Fourier modu $k$")
ax.set_ylabel(r"$\|RF_k\|$ (birim-norm $F_k$)")
ax.set_title(r"Fourier modu yörünge kazanımı  ($Q_y \approx 2.68$)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("fig2_orbit_gain.png", bbox_inches="tight")
plt.close(fig)
print("fig2_orbit_gain.png  ✓")

# tablo çıktısı
with open("table2_gain.txt", "w") as f:
    f.write("% Tablo 2: Fourier modu kazanımları\n")
    f.write(f"% {'k':>3}  {'||RF_k||':>10}  {'||M_k||':>10}\n")
    for k, rf, m in zip(k_list, rf_norms, m_norms):
        tag = "  % <-- baskın" if k == 2 else ""
        f.write(f"  {k:3d}  {rf:10.3f}  {m:10.1f}{tag}\n")
print("table2_gain.txt  ✓")

# ══════════════════════════════════════════════════════════════════════════════
# ŞEKİL 3 — BPM ofset ölçeklenmesi
# ══════════════════════════════════════════════════════════════════════════════

sigma_b_vals = np.array([0, 50, 100, 150, 200, 300, 400, 500]) * 1e-6
N_MC = 300
A_true = 10e-6
phi_true = 0.3
rng = np.random.default_rng(42)

means3, stds3 = [], []
for sig in sigma_b_vals:
    errs = []
    for _ in range(N_MC):
        y = make_orbit(R, A_true=A_true, phi_true=phi_true,
                       b_sigma=sig, rng=rng)
        A_fit, _ = project_amplitude(y, R, 2)
        errs.append(abs(A_fit - A_true) * 1e6)
    means3.append(np.mean(errs))
    stds3.append(np.std(errs))

means3 = np.array(means3)
stds3  = np.array(stds3)
theory_bias = sigma_b_vals * 1e6 / Mk2_norm   # δA = σ_b / ||M_k||

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(sigma_b_vals * 1e6, means3, yerr=stds3,
            fmt="o", color=BLUE, capsize=3, markersize=5,
            label="Simülasyon (MC ortalama ± 1σ)")
ax.plot(sigma_b_vals * 1e6, theory_bias,
        color=RED, linestyle="--", linewidth=1.5,
        label=fr"Teori: $\sigma_b/\|M_{{k=2}}\|$ = $\sigma_b/{Mk2_norm:.0f}$")
ax.axhline(10, color=ORANGE, linestyle=":", linewidth=1.2, label="10 μm hedef")
ax.set_xlabel(r"BPM ofset r.m.s.  $\sigma_b$ [μm]")
ax.set_ylabel(r"$k=2$ genlik hatası [μm]")
ax.set_title("BPM ofset ölçeklenmesi")
ax.legend(frameon=False, loc="upper left")
ax.set_xlim(-10, 520)
ax.set_ylim(bottom=-0.05)
fig.tight_layout()
fig.savefig("fig3_offset_scaling.png", bbox_inches="tight")
plt.close(fig)
print("fig3_offset_scaling.png  ✓")
print(f"  σ_b=100 μm → hata: {means3[2]:.3f} ± {stds3[2]:.3f} μm  "
      f"(teori: {100/Mk2_norm:.3f} μm)")

# ══════════════════════════════════════════════════════════════════════════════
# ŞEKİL 4 — Model hatası toleransı  σ_model = δK/K
# ══════════════════════════════════════════════════════════════════════════════

sigma_m_vals = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]) / 100.0
rng2 = np.random.default_rng(99)

means4, stds4 = [], []
for sig_m in sigma_m_vals:
    errs = []
    for _ in range(N_MC):
        # Gerçek yörünge: R doğru, b=100 μm
        y = make_orbit(R, A_true=A_true, phi_true=phi_true,
                       b_sigma=100e-6, rng=rng2)
        # Tahminci: pertürbe R_model (sütun ölçekleme = gradyan hatası)
        eps = rng2.normal(0, sig_m, N_Q)
        R_model = R * (1 + eps)[np.newaxis, :]
        A_fit, _ = project_amplitude(y, R_model, 2)
        errs.append(abs(A_fit - A_true) * 1e6)
    means4.append(np.mean(errs))
    stds4.append(np.std(errs))

means4 = np.array(means4)
stds4  = np.array(stds4)

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(sigma_m_vals * 100, means4, yerr=stds4,
            fmt="s", color=BLUE, capsize=3, markersize=5,
            label="Simülasyon (MC ortalama ± 1σ)")
ax.axhline(10, color=ORANGE, linestyle=":", linewidth=1.2, label="10 μm hedef")

# tolerans sınırını veri üzerinden bul
tol_val = np.interp(10.0, means4, sigma_m_vals * 100)
ax.axvline(tol_val, color=GREEN, linestyle="--", linewidth=1.0,
           label=fr"Tolerans $\approx${tol_val:.1f}%")
ax.set_xlabel(r"Gradyan model hatası  $\sigma_\mathrm{model} = \delta K/K$ [%]")
ax.set_ylabel(r"$k=2$ genlik hatası [μm]")
ax.set_title("Model hatası toleransı")
ax.legend(frameon=False)
ax.set_xlim(-0.2, 10.5)
ax.set_ylim(bottom=-0.1)
fig.tight_layout()
fig.savefig("fig4_sigma_model.png", bbox_inches="tight")
plt.close(fig)
print("fig4_sigma_model.png  ✓")
print(f"  σ_model=3% → hata: {means4[4]:.2f} ± {stds4[4]:.2f} μm")
print(f"  σ_model=4% → hata: {means4[5]:.2f} ± {stds4[5]:.2f} μm")
print(f"  Tolerans sınırı (10 μm kesişim): ~{tol_val:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# TABLO 3 — Hizalama genliği vs yörünge normu
# ══════════════════════════════════════════════════════════════════════════════

truth = {2: (10e-6, 0.3), 4: (300e-6, 0.7), 6: (300e-6, 1.2), 8: (200e-6, 2.1)}
with open("table3_orbit.txt", "w") as f:
    f.write("% Tablo 3: Hizalama genliği vs yörünge normu katkısı\n")
    f.write(f"  {'k':>3}  {'Hizalama [μm]':>15}  {'Yörünge normu [μm]':>20}  {'Kazanım':>8}\n")
    for k, (A, _) in sorted(truth.items()):
        dq_k = A * Fcos(k, N_Q)
        orb_nm = np.linalg.norm(R @ dq_k) * 1e6
        gain = orb_nm / (A * 1e6)
        f.write(f"  {k:3d}  {A*1e6:>15.0f}  {orb_nm:>20.0f}  {gain:>8.1f}\n")
print("table3_orbit.txt  ✓")

# ══════════════════════════════════════════════════════════════════════════════
# Özet
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("  ÖZET (paper değerleriyle karşılaştırma)")
print("=" * 60)
print(f"  ||M_k=2|| = {Mk2_norm:.1f}  (paper: 167)")
print(f"  κ(R) = {kappa_R:.1f}  (paper: 248.7)")
print(f"  σ_b=100μm → hata = {means3[2]:.3f} μm  (paper: 0.6%×10μm = 0.06 μm)")
print(f"  Model toleransı = {tol_val:.1f}%  (paper: 3–4%)")
print()
print("Tüm dosyalar oluşturuldu.")
