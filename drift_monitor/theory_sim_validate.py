#!/usr/bin/env python3
"""
theory_sim_validate.py — Teori (analitik Courant-Snyder R) ile gerçek C++
demet izleyicisinden (integrator.cpp) kurulan tepki matrisinin karşılaştırması.

Amaç: drift_makalesi'nde kullanılan analitik R'nin (fodo_lattice.py) tam
parçacık takibiyle uyumunu göstermek — yöntemin "teori↔simülasyon" temeli.
Karşılaştırılan: (a) eleman-eleman R, (b) SVD spektrumu, (c) κ, (d) en kötü
modların simetrik içeriği.

Çıktı: fig7_theory_sim.png + konsolda uyum metrikleri.
Süre: ~2 dk (97 izleyici koşumu, paralel).

Kullanım:
    python3 theory_sim_validate.py [--workers N]
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_DIR)
sys.path.insert(0, _DIR)
sys.path.insert(0, _BASE)

import fodo_lattice as fl
from build_response_matrix import build_matrices

# PRD tek-sütun stili (make_figures.py ile aynı)
COL = 3.375
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.5, "legend.frameon": False,
    "lines.linewidth": 1.1, "lines.markersize": 2.6, "axes.linewidth": 0.6,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def sym_frac(v):
    N = len(v); s = v.copy()
    for c in range(N // 2):
        m = 0.5 * (v[2 * c] + v[2 * c + 1]); s[2 * c] = m; s[2 * c + 1] = m
    return np.sum(s ** 2) / np.sum(v ** 2)


def analytic_R(cfg, plane):
    Kx = fl.calibrate_K_x_arc(cfg) if plane == "x" else None
    beta, phi, Q = fl.compute_twiss_at_quads(cfg, plane, K_x_arc=Kx)
    KL = fl.signed_KL(cfg, plane)
    return fl.build_response_matrix(beta, phi, Q, KL)


def worst_sym(R, n=8):
    _, _, Vt = np.linalg.svd(R)
    return np.mean([sym_frac(Vt[-i - 1]) for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", "-w", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    os.chdir(_BASE)
    with open("params.json") as f:
        cfg = json.load(f)

    print("=" * 60)
    print("Teori (analitik) ↔ Simülasyon (C++ izleyici) R karşılaştırması")
    print("=" * 60)

    # 1) Simülasyon R (gerçek izleyici)
    print(f"\n[1/2] İzleyiciden R kuruluyor (paralel, w={args.workers})...")
    R_dy_sim, R_dx_sim = build_matrices(cfg, g1_override=cfg["g1"],
                                        delta_q=1e-4, label="sim",
                                        n_workers=args.workers)

    # 2) Analitik R (teori)
    print("\n[2/2] Analitik R hesaplanıyor...")
    R_dy_an = analytic_R(cfg, "y")
    R_dx_an = analytic_R(cfg, "x")

    fig, axes = plt.subplots(2, 1, figsize=(COL, 4.6))

    for ax, plane, Rs, Ra, lab in (
            (axes[0], "y", R_dy_sim, R_dy_an, "(a)"),
            (axes[1], "x", R_dx_sim, R_dx_an, "(b)")):
        # Olası global işaret uyumu (analitik konvansiyon)
        if np.sum(Rs * Ra) < 0:
            Ra = -Ra
        rel = np.linalg.norm(Rs - Ra) / np.linalg.norm(Ra)
        corr = np.corrcoef(Rs.ravel(), Ra.ravel())[0, 1]
        ks, ka = np.linalg.cond(Rs), np.linalg.cond(Ra)
        ws, wa = worst_sym(Rs), worst_sym(Ra)
        print(f"\n--- Düzlem {plane} ---")
        print(f"  ‖R_sim−R_an‖/‖R_an‖ = {rel:.3f}")
        print(f"  korelasyon (eleman)  = {corr:.4f}")
        print(f"  κ(R_sim)={ks:.0f}   κ(R_an)={ka:.0f}")
        print(f"  en kötü 8 mod simetrik güç: sim %{100*ws:.0f}  an %{100*wa:.0f}")

        # Eleman-eleman saçılım
        ax.plot(Ra.ravel(), Rs.ravel(), "o", color="C0", ms=1.6, alpha=0.4)
        lim = np.max(np.abs(Ra)) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "-", color="C3", lw=0.8)
        ax.set_xlabel(r"$R_{ij}$ analitik (teori)")
        ax.set_ylabel(r"$R_{ij}$ izleyici (sim.)")
        ax.text(0.04, 0.90,
                f"düzlem {plane}: korelasyon={corr:.4f}\n"
                rf"$\kappa$: sim {ks:.0f} / an {ka:.0f}",
                transform=ax.transAxes, fontsize=6.3, va="top")
        ax.text(-0.22, 1.02, lab, transform=ax.transAxes,
                fontsize=8, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.tight_layout(h_pad=0.6)
    out = os.path.join(_DIR, "fig7_theory_sim.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"\nKaydedildi: {os.path.relpath(out, _BASE)}")


if __name__ == "__main__":
    main()
