#!/usr/bin/env python3
"""fig_2_svd.py — ŞEKİL 2: R tepki matrisinin SVD analizi.

Sol panel : tekil değer spektrumu (ilk iki ayrışmış gösterilir).
Sağ paneller: en büyük iki sağ tekil vektör v1, v2 ile
              k=2 FODO-antisimetrik cos/sin modlarının karşılaştırması.

Çıktı: fig_2_svd.png
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

R = np.load("R_dy_1.npy")

# ── SVD ──────────────────────────────────────────────────────────────────────
U, S, Vt = np.linalg.svd(R)
V = Vt.T   # sütunlar = sağ tekil vektörler

# k=2 FODO-antisimetrik cos/sin modları
F2, meta2 = fodo_basis(N_Q, [2], ANTISYM)
f2_cos = F2[:, 0] / np.linalg.norm(F2[:, 0])
f2_sin = F2[:, 1] / np.linalg.norm(F2[:, 1])

# İşaret belirsizliğini düzelt: iç çarpım negatifse çevir
v1 = V[:, 0] * np.sign(np.dot(V[:, 0], f2_cos))
v2 = V[:, 1] * np.sign(np.dot(V[:, 1], f2_sin))

# Korelasyon katsayıları
r1 = np.dot(v1, f2_cos)
r2 = np.dot(v2, f2_sin)

j = np.arange(N_Q)
s_pos = j * (2 * math.pi * cfg.get("R0", 95.49) / N_Q)

# ── Grafik ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 4.5))
gs  = fig.add_gridspec(1, 3, wspace=0.38)

# Panel 1: Tekil değer spektrumu
ax0 = fig.add_subplot(gs[0])
idx = np.arange(1, N_Q + 1)
ax0.semilogy(idx[2:], S[2:], "o", color="0.65", ms=4, label="diğer")
ax0.semilogy(idx[:2], S[:2], "o", color="tab:red", ms=7, zorder=5,
             label=r"$\sigma_1,\,\sigma_2$  (k=2 modu)")
ax0.axhline(S[0], color="tab:red", lw=0.6, ls="--", alpha=0.4)
ax0.set_xlabel("Tekil değer sırası", fontsize=11)
ax0.set_ylabel(r"Tekil değer $\sigma_i$", fontsize=11)
ax0.set_title(f"R tekil değer spektrumu\n"
              f"$\\kappa(R)={S[0]/S[-1]:.0f}$,  "
              f"$\\sigma_1/\\sigma_3={S[0]/S[2]:.1f}$", fontsize=10)
ax0.legend(fontsize=9)
ax0.set_xlim(0.5, N_Q + 0.5)

# Panel 2: v1 vs F_{k=2,cos}
ax1 = fig.add_subplot(gs[1])
ax1.plot(s_pos, f2_cos, "--", color="tab:blue", lw=1.8,
         label=r"$F_{k=2,\cos}$ (FODO antisim.)")
ax1.plot(s_pos, v1,     "-",  color="tab:red",  lw=1.4,
         label=fr"$v_1$  ($r={r1:.4f}$)")
ax1.axhline(0, color="k", lw=0.5)
ax1.set_xlabel("Halka konumu s [m]", fontsize=10)
ax1.set_ylabel("Normalize genlik", fontsize=10)
ax1.set_title(r"En büyük tekil vektör $v_1$" "\n"
              r"vs $F_{k=2,\cos}$", fontsize=10)
ax1.legend(fontsize=8)

# Panel 3: v2 vs F_{k=2,sin}
ax2 = fig.add_subplot(gs[2])
ax2.plot(s_pos, f2_sin, "--", color="tab:blue", lw=1.8,
         label=r"$F_{k=2,\sin}$ (FODO antisim.)")
ax2.plot(s_pos, v2,     "-",  color="tab:orange", lw=1.4,
         label=fr"$v_2$  ($r={r2:.4f}$)")
ax2.axhline(0, color="k", lw=0.5)
ax2.set_xlabel("Halka konumu s [m]", fontsize=10)
ax2.set_title(r"İkinci tekil vektör $v_2$" "\n"
              r"vs $F_{k=2,\sin}$", fontsize=10)
ax2.legend(fontsize=8)

fig.suptitle("Tepki matrisi SVD analizi — $R$'nin doğal modu $k=2$'dir\n"
             r"$\kappa(R)=\sigma_1/\sigma_{48}\approx249$  "
             r"(koşullanma sayısı; yörünge kazancı değil)",
             fontsize=11)
fig.tight_layout()
fig.savefig("fig_2_svd.png", dpi=140)
print(f"→ fig_2_svd.png kaydedildi")
print(f"  σ₁={S[0]:.2f}  σ₂={S[1]:.2f}  σ₃={S[2]:.2f}  κ={S[0]/S[-1]:.1f}")
print(f"  v₁·F_cos korelasyonu: {r1:.5f}")
print(f"  v₂·F_sin korelasyonu: {r2:.5f}")
