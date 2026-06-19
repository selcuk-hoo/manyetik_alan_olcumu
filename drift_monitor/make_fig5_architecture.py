#!/usr/bin/env python3
"""
make_fig5_architecture.py — Şekil 5: iki-katmanlı işletme mimarisi şeması (§5).

PRD tek-sütun (3.375 in genişlik) dikey akış şeması: yukarıdan aşağı
  yavaş mutlak katman (LOCO/BBA) → çıktılar (Δq₀, b₀, R)
  → hızlı drift katmanı R⁻¹(y−y₀) ← BPM y(t)
  → antisimetrik drift izleme; altta simetrik kör-nokta uyarısı.

Veriden bağımsız kavram şeması. Kullanım:
    python3 make_fig5_architecture.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_DIR = os.path.dirname(os.path.abspath(__file__))
COL = 3.375          # PRD tek-sütun genişliği [inç]

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def _box(ax, xy, w, h, text, fc, fs=6.8, ec="black"):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.03",
        linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3)


def _arrow(ax, p0, p1, text="", color="black"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=11,
        linewidth=1.1, color=color, zorder=4))
    if text:
        ax.text((p0[0] + p1[0]) / 2 + 0.03, (p0[1] + p1[1]) / 2, text,
                ha="left", va="center", fontsize=6, color=color, zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(COL, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 1) Yavaş mutlak katman (üst)
    _box(ax, (0.05, 0.80), 0.90, 0.13,
         "YAVAŞ MUTLAK KATMAN  (saatlik–günlük)\n"
         r"LOCO + BBA + survey  $\Rightarrow$  $\Delta q_0,\ \mathbf{b}_0,\ R$",
         fc="#cfe3f7", fs=6.6)
    ax.text(0.5, 0.95, "Mutlak referans", ha="center", fontsize=7.5,
            fontweight="bold")

    # 2) Hızlı drift katmanı (orta sol)
    _box(ax, (0.05, 0.45), 0.56, 0.16,
         "HIZLI DRIFT KATMANI\n(sürekli, sn–dk)\n"
         r"$\widehat{\delta q}(t)=R^{-1}(\mathbf{y}(t)-\mathbf{y}_0)$",
         fc="#d6f5d6", fs=6.6)
    ax.text(0.33, 0.625, "Bu çalışmanın katkısı", ha="center", fontsize=7,
            fontweight="bold", color="#1a7f1a")

    # 3) BPM ölçümü (orta sağ)
    _box(ax, (0.66, 0.47), 0.29, 0.12,
         r"BPM $\mathbf{y}(t)$" "\n(fizik run'ı)", fc="#fff0cc", fs=6.6)

    # 4) Sonuç (alt)
    _box(ax, (0.13, 0.16), 0.74, 0.12,
         "Antisimetrik hizalama driftini izle\n"
         r"(6–7 $\mu$m RMS, ofsetten bağımsız)", fc="#f7d6d6", fs=6.8)

    # Oklar
    _arrow(ax, (0.40, 0.80), (0.33, 0.61), r"$R,\ \mathbf{y}_0$")
    _arrow(ax, (0.66, 0.53), (0.61, 0.53), r"$\mathbf{y}(t)$")
    _arrow(ax, (0.33, 0.45), (0.45, 0.28))

    # Kör-nokta uyarısı (alt)
    ax.text(0.5, 0.045,
            "Sınır: simetrik (sahte-EDM-kritik) alt-uzay her iki katmanda da\n"
            "gürültü sınırında → dış gözlemlenebilir (demet ayrımı/spin) gerekir",
            ha="center", va="center", fontsize=5.8, color="#a11", style="italic")

    out = os.path.join(_DIR, "fig5_mimari.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"kaydedildi: {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
