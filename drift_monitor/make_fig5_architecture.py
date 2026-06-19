#!/usr/bin/env python3
"""
make_fig5_architecture.py — Şekil 5: iki-katmanlı işletme mimarisi şeması (§5).

Veriden bağımsız bir kavram şeması: yavaş mutlak katman (LOCO/BBA) + hızlı
drift katmanı (R⁻¹(y−y₀)) ve bunların besleme ilişkisi.

Kullanım:
    python3 make_fig5_architecture.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_DIR = os.path.dirname(os.path.abspath(__file__))


def _box(ax, xy, w, h, text, fc, ec="black"):
    """Yuvarlatılmış köşeli kutu + ortalanmış metin."""
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=9.5, zorder=3)


def _arrow(ax, p0, p1, text="", color="black", style="-|>"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=18,
        linewidth=1.6, color=color, zorder=4))
    if text:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + 0.02, text, ha="center", va="bottom",
                fontsize=8.5, color=color, zorder=3)


def main():
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.08, 1)
    ax.axis("off")

    # Yavaş mutlak katman (üst)
    _box(ax, (0.06, 0.72), 0.40, 0.18,
         "YAVAŞ MUTLAK KATMAN\n(saatlik–günlük)\nLOCO + BBA + survey",
         fc="#cfe3f7")
    ax.text(0.26, 0.93, "Mutlak referans", ha="center", fontsize=10,
            fontweight="bold")

    # Çıktılar kutusu
    _box(ax, (0.56, 0.72), 0.38, 0.18,
         r"$\Delta q_0,\ \mathbf{b}_0,\ R$" "\n(örgü modeli:\nβ, φ, Q)",
         fc="#e8e8e8")

    # Hızlı drift katmanı (alt)
    _box(ax, (0.06, 0.30), 0.40, 0.20,
         "HIZLI DRIFT KATMANI\n(sürekli, saniye–dakika)\n"
         r"$\widehat{\delta q}(t)=R^{-1}(\mathbf{y}(t)-\mathbf{y}_0)$",
         fc="#d6f5d6")
    ax.text(0.26, 0.525, "Bu çalışmanın katkısı", ha="center", fontsize=10,
            fontweight="bold", color="#1a7f1a")

    # BPM ölçümü
    _box(ax, (0.56, 0.32), 0.38, 0.16,
         r"BPM okuması $\mathbf{y}(t)$" "\n(fizik run'ı boyunca)",
         fc="#fff0cc")

    # Sonuç / kullanım
    _box(ax, (0.27, 0.06), 0.46, 0.13,
         "Antisimetrik hizalama driftini izle\n(6–7 μm RMS, ofsetten bağımsız)",
         fc="#f7d6d6")

    # Oklar
    _arrow(ax, (0.46, 0.81), (0.56, 0.81))                       # mutlak → çıktılar
    _arrow(ax, (0.75, 0.72), (0.40, 0.50), r"$R,\ \mathbf{y}_0$") # R, y0 → drift
    _arrow(ax, (0.75, 0.48), (0.46, 0.42), r"$\mathbf{y}(t)$")    # BPM → drift
    _arrow(ax, (0.30, 0.30), (0.44, 0.19))                       # drift → sonuç

    # Kör nokta uyarısı (kesikli, kırmızı)
    ax.text(0.50, -0.05,
            "Sınır: simetrik (sahte-EDM-kritik) alt-uzay her iki katmanda da\n"
            "gürültü sınırında → dış gözlemlenebilir (demet ayrımı/spin) gerekir",
            ha="center", va="center", fontsize=8, color="#a11", style="italic")

    ax.set_title("Şekil 5 — İki-katmanlı hizalama izleme mimarisi", fontsize=12)
    out = os.path.join(_DIR, "fig5_mimari.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"kaydedildi: {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
