#!/usr/bin/env python3
"""plot_misalignment_pattern.py — Gerçekçi quad hizalama hatası örüntüsü

Senaryo: k=1 (30 μm) + k=2 (10 μm) + k=3 (25 μm) hedef harmonikleri
         k=4..10 her biri 100–300 μm rastgele genlik+faz kirleticiler

Çıktı: misalignment_pattern.png
"""
import json, math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fourier_reconstruct import fodo_basis

with open("params.json") as f:
    cfg = json.load(f)
N_Q      = 2 * int(cfg["nFODO"])
ANTISYM  = cfg.get("smooth_antisym_fodo", True)
circ     = (2*math.pi*cfg["R0"] + 4*cfg["nFODO"]*cfg["driftLen"]
            + 2*cfg["nFODO"]*cfg["quadLen"])
cell_len = circ / cfg["nFODO"]

# ── gerçek misalignment (tüm betikler için aynı tohum/değerler) ──────────
RNG   = np.random.default_rng(42)
TRUTH = {
    1: (30e-6,  0.80),
    2: (10e-6,  1.50),
    3: (25e-6,  0.30),
}
for k in range(4, 11):
    TRUTH[k] = (float(RNG.uniform(100e-6, 300e-6)), float(RNG.uniform(0, 2*math.pi)))


def build_dy(truth, n_q=N_Q, antisym=ANTISYM):
    dy = np.zeros(n_q)
    for k, (A, phi) in truth.items():
        F, _ = fodo_basis(n_q, [k], antisym)
        dy += A*math.cos(phi)*F[:, 0] + A*math.sin(phi)*F[:, 1]
    return dy


def fit_amp_phase(dy, k, n_q=N_Q, antisym=ANTISYM):
    F, _ = fodo_basis(n_q, [k], antisym)
    a, _, _, _ = np.linalg.lstsq(F, dy, rcond=None)
    return math.sqrt(a[0]**2 + a[1]**2), math.atan2(a[1], a[0])


dy = build_dy(TRUTH)

# s pozisyonları: QF hücrenin 1/4'ünde, QD 3/4'ünde
s_pos = np.array([(j//2)*cell_len + (0.25 if j%2==0 else 0.75)*cell_len
                  for j in range(N_Q)])

k_all   = list(range(1, 12))
amps    = [fit_amp_phase(dy, k)[0]*1e6 for k in k_all]
colors  = ["tab:blue" if k in {1, 2, 3} else "tab:orange" for k in k_all]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Sol: ring boyunca profil
bw = cell_len * 0.20
ax1.bar(s_pos[0::2], dy[0::2]*1e6, width=bw, color="tab:blue",   alpha=0.8, label="QF")
ax1.bar(s_pos[1::2], dy[1::2]*1e6, width=bw, color="tab:orange", alpha=0.8, label="QD")
ax1.axhline(0, color="k", lw=0.5)
ax1.set_xlabel("Azimuthal konum $s$ [m]", fontsize=11)
ax1.set_ylabel("Dikey hizalama hatası $\\Delta y$ [μm]", fontsize=11)
ax1.set_title("Rastgele quad hizalama hatası profili\n"
              "$k=1$(30μm)+$k=2$(10μm)+$k=3$(25μm) + kirleticiler $k=4$–10 (100–300μm)",
              fontsize=10)
ax1.legend(fontsize=10)

# Sağ: Fourier genlik spektrumu
bars = ax2.bar(k_all, amps, color=colors, alpha=0.85, edgecolor="k", lw=0.5)
for k, A in zip(k_all, amps):
    ax2.text(k, A + 4, f"{A:.0f}", ha="center", va="bottom", fontsize=8)
ax2.set_xlabel("Fourier modu $k$", fontsize=11)
ax2.set_ylabel("Genlik [μm]", fontsize=11)
ax2.set_title("Hizalama hatasının Fourier spektrumu\n"
              "(mavi = hedefler $k=1,2,3$; turuncu = kirleticiler)", fontsize=10)
ax2.set_xticks(k_all)
handles = [Patch(color="tab:blue",   alpha=0.85, label="Hedef $k=1,2,3$"),
           Patch(color="tab:orange", alpha=0.85, label="Kirletici $k=4$–10")]
ax2.legend(handles=handles, fontsize=10)

fig.tight_layout()
fig.savefig("misalignment_pattern.png", dpi=140)
print("→ misalignment_pattern.png kaydedildi")

print("\nGerçek Fourier bileşenleri vs. doğrudan fit:")
print(f"  {'k':>3}  {'A_gerçek [μm]':>14}  {'φ_gerçek [rad]':>14}  {'A_fit [μm]':>10}  {'hata %':>7}")
for k in k_all:
    A_fit, _ = fit_amp_phase(dy, k)
    A_true, phi_true = TRUTH.get(k, (0.0, 0.0))
    err = abs(A_fit - A_true) / A_true * 100 if A_true > 0 else float('nan')
    print(f"  {k:>3}  {A_true*1e6:>14.1f}  {phi_true:>14.2f}  {A_fit*1e6:>10.1f}  {err:>7.2f}")
