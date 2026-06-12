#!/usr/bin/env python3
"""test_antisymmetry.py — σ¹ (doğrusal) ile σ² (geometrik faz) bileşenlerini ayırt eder.

YÖNTEM:
  +δy ve −δy konfigürasyonlarını koşturup eğimleri topla/çıkar:
    antisim = (rate[+δy] − rate[−δy]) / 2  →  σ¹ bileşeni (B_x doğrudan)
    sim     = (rate[+δy] + rate[−δy]) / 2  →  σ² bileşeni (geometrik faz)

  Gerekçe:
    dSy/dt = C₁(δy) + C₂(δy, δy)
    C₁ δy → −δy altında işaret değiştirir   → toplam sıfır, fark 2×σ¹
    C₂ (quadratik)  işaret değiştirmez       → toplam 2×σ², fark sıfır

  CO=True ile hem +δy hem −δy için ayrı kapalı yörünge aranır.
  Hem dikey (quad_dy) hem yatay (quad_dx) hatalar eklenir.

ÇIKTI: test_antisymmetry.json, test_antisymmetry.png
"""

import json, os, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)

from integrator import integrate_particle
from false_edm_mode_scan import setup_fields, find_closed_orbit, _make_state

C_LIGHT = 299792458.0

N_SEEDS    = 2
SIGMA_LIST = [2e-6, 5e-6, 10e-6, 20e-6, 80e-6]  # [m]
T2         = 1e-3    # [s]
DT         = 1e-11
CO_TURNS   = 40
CO_ITER    = 2
RET_STEP   = 10000  # sadece Poincaré istiyoruz, sürekli iz gerekmez


def run_one(fields, p_mag, direction, quad_dy, quad_dx, circ, beta0, sign=+1):
    """sign=+1: +δy,  sign=−1: −δy konfigürasyonu."""
    dy = sign * quad_dy
    dx = sign * quad_dx
    n_q = len(dy)
    T_rev = circ / (beta0 * C_LIGHT)

    v_co, resid = find_closed_orbit(
        fields, p_mag, direction, dy, DT, T_rev,
        n_turns=CO_TURNS, n_iter=CO_ITER)
    y_co = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])

    fields.poincare_quad_index = 0.0
    _, poin, poin_t = integrate_particle(
        y_co, 0.0, T2, DT, fields=fields,
        return_steps=RET_STEP, quad_dy=dy, quad_dx=dx)

    ts = np.asarray(poin_t, float)
    sy = np.asarray(poin[:, 7], float)
    slope = float(np.polyfit(ts, sy, 1)[0])
    return slope, float(np.hypot(v_co[0], v_co[1]) * 1e3)


def main():
    t_wall = time.time()
    with open("params.json") as fh:
        cfg = json.load(fh)

    results = []

    print(f"{'σ [μm]':>8}  {'seed':>4}  {'rate+ [rad/s]':>16}  "
          f"{'rate- [rad/s]':>16}  {'antisim [σ¹]':>16}  {'sim [σ²]':>16}")

    for sigma_m in SIGMA_LIST:
        antisym_vals = []
        sym_vals     = []

        for seed in range(N_SEEDS):
            fields, y0, beta0, R0, p_mag, direction = setup_fields(cfg)
            n_q   = 2 * int(fields.nFODO)
            circ  = (2 * np.pi * R0
                     + 4 * fields.nFODO * fields.driftLen
                     + 2 * fields.nFODO * fields.quadLen)

            rng      = np.random.default_rng(seed)
            quad_dy  = rng.normal(0.0, sigma_m, n_q)
            quad_dx  = rng.normal(0.0, sigma_m, n_q)

            r_plus,  co_plus  = run_one(fields, p_mag, direction,
                                        quad_dy, quad_dx, circ, beta0, sign=+1)
            r_minus, co_minus = run_one(fields, p_mag, direction,
                                        quad_dy, quad_dx, circ, beta0, sign=-1)

            antisym = (r_plus - r_minus) / 2.0  # σ¹ bileşeni
            sym     = (r_plus + r_minus) / 2.0  # σ² bileşeni

            antisym_vals.append(antisym)
            sym_vals.append(sym)

            print(f"{sigma_m*1e6:8.1f}  {seed:4d}  "
                  f"{r_plus:16.4e}  {r_minus:16.4e}  "
                  f"{antisym:16.4e}  {sym:16.4e}")

        results.append({
            "sigma_um":  sigma_m * 1e6,
            "antisym_vals": antisym_vals,   # σ¹ (her seed için)
            "sym_vals":     sym_vals,        # σ² (her seed için)
            "antisym_rms": float(np.std(antisym_vals)),
            "sym_rms":     float(np.std(sym_vals)),
            "sym_mean":    float(np.mean(sym_vals)),
            "|antisym|_mean": float(np.mean(np.abs(antisym_vals))),
            "|sym|_mean":     float(np.mean(np.abs(sym_vals))),
        })

    with open("test_antisymmetry.json", "w") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    # ── Log-log fit ve grafik ───────────────────────────────────────────────
    sigmas = np.array([r["sigma_um"] for r in results]) * 1e-6
    antisym_means = np.array([r["|antisym|_mean"] for r in results])
    sym_means     = np.array([r["|sym|_mean"]     for r in results])
    sym_abs_mean  = np.array([abs(r["sym_mean"])  for r in results])

    def fit_slope(x, y, label=""):
        mask = (y > 0)
        if mask.sum() < 2:
            return float("nan")
        p = np.polyfit(np.log10(x[mask]), np.log10(y[mask]), 1)
        print(f"  {label}: α = {p[0]:.3f}")
        return p[0]

    print("\n--- Log-log eğim analizi ---")
    a1 = fit_slope(sigmas, antisym_means, "|antisim| (σ¹ bileşeni)")
    a2 = fit_slope(sigmas, sym_means,     "|sim| (σ² bileşeni)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.loglog(sigmas * 1e6, antisym_means, "b-o", label=f"|antisim| α={a1:.2f}")
    ax.loglog(sigmas * 1e6, sym_means,     "r-s", label=f"|sim| α={a2:.2f}")
    # referans çizgiler
    s0 = sigmas[len(sigmas)//2]
    ax.loglog(sigmas * 1e6,
              antisym_means[len(sigmas)//2] * (sigmas/s0)**1,
              "b--", lw=0.7, alpha=0.5, label="∝σ¹")
    ax.loglog(sigmas * 1e6,
              max(sym_means[sym_means > 0]) * (sigmas/s0)**2,
              "r--", lw=0.7, alpha=0.5, label="∝σ²")
    ax.set_xlabel("σ [μm]"); ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("+δy / −δy ayrıştırması: σ¹ vs σ²")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for i, r in enumerate(results):
        sigma_um = r["sigma_um"]
        av = r["antisym_vals"]
        sv = r["sym_vals"]
        ax.scatter([sigma_um]*len(av), av, c="blue", s=15, alpha=0.7,
                   label="antisim (σ¹)" if i == 0 else "")
        ax.scatter([sigma_um]*len(sv), sv, c="red",  s=15, alpha=0.7,
                   label="sim (σ²)" if i == 0 else "")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("σ [μm]"); ax.set_ylabel("dSy/dt [rad/s]")
    ax.set_title("Seed dağılımı: antisim işaret tutarlılığı?")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("test_antisymmetry.png", dpi=150)
    print(f"\nKaydedildi: test_antisymmetry.json, test_antisymmetry.png")
    print(f"Toplam süre: {(time.time()-t_wall)/60:.1f} dk")


if __name__ == "__main__":
    main()
