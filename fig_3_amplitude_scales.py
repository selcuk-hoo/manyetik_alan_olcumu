#!/usr/bin/env python3
"""fig_3_amplitude_scales.py — ŞEKİL 3: BPM ofseti bastırımının üç genlik ölçeği.

Sol panel : üç genlik ölçeğinin logaritmik çubuk grafiği
            (sinyal yörüngesi, ham Fourier ofseti, tahminleyici kirletme tabanı).
Sağ panel : BPM okumaları — k=2 sinyali (10 μm) ile sinyale eklenen
            100 μm ofset; eşleşmiş-filtre tahmin değerleri gösterilir.

Çıktı: fig_3_amplitude_scales.png
"""
import json, math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fourier_reconstruct import fodo_basis

with open("params.json") as f:
    cfg = json.load(f)
N_Q     = 2 * int(cfg["nFODO"])
ANTISYM = cfg.get("smooth_antisym_fodo", True)
R0      = float(cfg.get("R0", 95.49))

R = np.load("R_dy_1.npy")

# ── Temel büyüklükler ─────────────────────────────────────────────────────────
F2, _ = fodo_basis(N_Q, [2], ANTISYM)
M2c   = R @ F2[:, 0]          # k=2 cos sütunu
M2s   = R @ F2[:, 1]          # k=2 sin sütunu
norm_M2 = np.linalg.norm(M2c)  # ‖M_{k=2}‖ ≈ 167

A_misalign = 10e-6             # gerçek k=2 kaçıklık genliği [m]
phi_true   = 1.50              # faz [rad]
sigma_b    = 100e-6            # BPM ofset RMS [m]

# Sinyal misalignment vektörü
dy_k2 = (A_misalign * math.cos(phi_true) * F2[:, 0]
        + A_misalign * math.sin(phi_true) * F2[:, 1])
y_signal = R @ dy_k2           # BPM uzayında sinyal yörüngesi

# Üç genlik ölçeği [μm]
scale_orbit   = np.linalg.norm(y_signal) * 1e6         # 1669 μm
scale_fourier = sigma_b * math.sqrt(math.pi / N_Q) * 1e6  # ~26 μm
scale_floor   = sigma_b / norm_M2 * 1e6                # ~0.6 μm

# Rastgele beyaz ofset
rng = np.random.default_rng(7)
b   = rng.normal(0, sigma_b, N_Q)
y_meas = y_signal + b          # sinyal + ofset

# Eşleşmiş-filtre tahminleri
m_hat = M2c / norm_M2
est_signal = np.dot(y_signal, m_hat) / norm_M2 * 1e6   # μm
est_noisy  = np.dot(y_meas,   m_hat) / norm_M2 * 1e6   # μm
est_offset = np.dot(b,        m_hat) / norm_M2 * 1e6   # μm (ofset katkısı)

s_pos = np.arange(N_Q) * (2 * math.pi * R0 / N_Q)

# ── Grafik ───────────────────────────────────────────────────────────────────
fig, (ax_bar, ax_bpm) = plt.subplots(1, 2, figsize=(13, 5))

# Sol: üç genlik ölçeği (log ölçek çubuk grafiği)
labels = [
    "Sinyal\nyörüngesi\n" + r"$A\|M_{k=2}\|$",
    "BPM ofseti\nFourier seviyesi\n" + r"$\sigma_b\sqrt{\pi/N}$",
    "Tahminleyici\nkirletme tabanı\n" + r"$\sigma_b/\|M_{k=2}\|$",
]
values = [scale_orbit, scale_fourier, scale_floor]
colors = ["tab:blue", "tab:orange", "tab:red"]
xi = np.arange(3)
bars = ax_bar.bar(xi, values, color=colors, alpha=0.85, edgecolor="k", lw=0.7,
                  width=0.55)
ax_bar.set_yscale("log")
ax_bar.set_xticks(xi)
ax_bar.set_xticklabels(labels, fontsize=9)
ax_bar.set_ylabel("Genlik [μm]", fontsize=11)
ax_bar.set_title("Üç genlik ölçeği hiyerarşisi\n"
                 r"$1669\,\mu$m $\gg$ $26\,\mu$m $\gg$ $0{,}6\,\mu$m",
                 fontsize=10)
for bar, val in zip(bars, values):
    ax_bar.text(bar.get_x() + bar.get_width()/2, val * 1.5,
                f"{val:.1f} μm", ha="center", va="bottom", fontsize=9,
                fontweight="bold")
# Referans: 10 μm hedef
ax_bar.axhline(10, color="gray", lw=1, ls=":", label="10 μm hedef")
ax_bar.legend(fontsize=9)

# Sağ: BPM okumaları
ax_bpm.plot(s_pos, y_signal * 1e6, "-", color="tab:blue", lw=2,
            label=fr"k=2 sinyal  ($A=10\,\mu$m;  $\hat{{a}}_{{k=2}}={est_signal:.1f}\,\mu$m)")
ax_bpm.plot(s_pos, y_meas * 1e6, "-", color="tab:orange", lw=1, alpha=0.8,
            label=fr"Sinyal + $\sigma_b=100\,\mu$m ofset  "
                  fr"($\hat{{a}}_{{k=2}}={est_noisy:.1f}\,\mu$m)")
ax_bpm.axhline(0, color="k", lw=0.5)
ax_bpm.set_xlabel("Halka konumu s [m]", fontsize=11)
ax_bpm.set_ylabel("BPM okuması [μm]", fontsize=11)
ax_bpm.set_title("BPM okumaları: sinyal vs sinyal+ofset\n"
                 f"Eşleşmiş-filtre SNR = "
                 fr"$A\|M_{{k=2}}\|/\sigma_b = {A_misalign*1e6:.0f}"
                 fr"\times{norm_M2:.0f}/{sigma_b*1e6:.0f} \approx {A_misalign*norm_M2/sigma_b:.1f}$",
                 fontsize=10)
ax_bpm.legend(fontsize=9)
ax_bpm.text(0.02, 0.04,
            f"Ofset katkısı: {est_offset:+.2f} μm  "
            f"(beklenen: ≲{sigma_b*1e6/norm_M2:.1f} μm)",
            transform=ax_bpm.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="lightyellow", ec="0.6"))

fig.suptitle("BPM ofseti bastırımı — tepki matrisi kuvvetlenmesi çift yönlü çalışır\n"
             r"Sinyali $\|M_{k=2}\|=167$ kat büyütür; ofseti $\|M_{k=2}\|$ kat böler",
             fontsize=11)
fig.tight_layout()
fig.savefig("fig_3_amplitude_scales.png", dpi=140)
print("→ fig_3_amplitude_scales.png kaydedildi")
print(f"  ‖M_{{k=2}}‖ = {norm_M2:.1f}")
print(f"  Sinyal yörüngesi normu   = {scale_orbit:.1f} μm")
print(f"  Ofset Fourier seviyesi   = {scale_fourier:.1f} μm")
print(f"  Tahminleyici kirletme    = {scale_floor:.3f} μm")
print(f"  Eşleşmiş-filtre SNR      = {A_misalign * norm_M2 / sigma_b:.2f}")
