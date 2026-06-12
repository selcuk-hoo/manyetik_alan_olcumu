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
  Tüm (σ, seed) çiftleri N_WORKERS işlemciyle paralel koşar.

ÇIKTI: test_antisymmetry.json, test_antisymmetry.png
"""

import json, os, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)

C_LIGHT = 299792458.0

N_SEEDS    = 5
SIGMA_LIST = [2e-6, 5e-6, 10e-6, 20e-6, 40e-6, 80e-6]  # [m]
T2         = 2e-3    # [s]
DT         = 1e-11
CO_TURNS   = 60
CO_ITER    = 3
RET_STEP   = 10000  # sadece Poincaré istiyoruz, sürekli iz gerekmez
N_WORKERS  = 10


def run_pair(args):
    """Her (sigma_m, seed) çifti için bağımsız worker — kendi fields nesnesi oluşturur."""
    sigma_m, seed, cfg = args
    import os, sys
    os.chdir(BASE)
    sys.path.insert(0, BASE)

    from integrator import integrate_particle
    from false_edm_mode_scan import setup_fields, find_closed_orbit, _make_state

    fields, y0, beta0, R0, p_mag, direction = setup_fields(cfg)
    n_q  = 2 * int(fields.nFODO)
    circ = (2 * np.pi * R0
            + 4 * fields.nFODO * fields.driftLen
            + 2 * fields.nFODO * fields.quadLen)
    T_rev = circ / (beta0 * C_LIGHT)

    rng     = np.random.default_rng(seed)
    quad_dy = rng.normal(0.0, sigma_m, n_q)
    quad_dx = rng.normal(0.0, sigma_m, n_q)

    def _run_one(sign):
        dy = sign * quad_dy
        dx = sign * quad_dx
        v_co, _ = find_closed_orbit(
            fields, p_mag, direction, dy, DT, T_rev,
            n_turns=CO_TURNS, n_iter=CO_ITER)
        y_co = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
        fields.poincare_quad_index = 0.0
        _, poin, poin_t = integrate_particle(
            y_co, 0.0, T2, DT, fields=fields,
            return_steps=RET_STEP, quad_dy=dy, quad_dx=dx)
        ts = np.asarray(poin_t, float)
        sy = np.asarray(poin[:, 7], float)
        return float(np.polyfit(ts, sy, 1)[0]), float(np.hypot(v_co[0], v_co[1]) * 1e3)

    r_plus,  co_plus  = _run_one(+1)
    r_minus, co_minus = _run_one(-1)
    return sigma_m, seed, r_plus, r_minus, co_plus, co_minus


def main():
    t_wall = time.time()
    with open("params.json") as fh:
        cfg = json.load(fh)

    tasks = [(sigma_m, seed, cfg)
             for sigma_m in SIGMA_LIST
             for seed in range(N_SEEDS)]

    print(f"Toplam {len(tasks)} görev, {N_WORKERS} işlemci ile paralel koşuyor...")
    print(f"{'σ [μm]':>8}  {'seed':>4}  {'rate+ [rad/s]':>16}  "
          f"{'rate- [rad/s]':>16}  {'antisim [σ¹]':>16}  {'sim [σ²]':>16}")

    # Ham sonuçları topla
    raw = {}  # (sigma_m, seed) -> (r_plus, r_minus)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(run_pair, t): t for t in tasks}
        for fut in as_completed(futures):
            sigma_m, seed, r_plus, r_minus, co_plus, co_minus = fut.result()
            antisym = (r_plus - r_minus) / 2.0
            sym     = (r_plus + r_minus) / 2.0
            raw[(sigma_m, seed)] = (r_plus, r_minus)
            print(f"{sigma_m*1e6:8.1f}  {seed:4d}  "
                  f"{r_plus:16.4e}  {r_minus:16.4e}  "
                  f"{antisym:16.4e}  {sym:16.4e}", flush=True)

    # σ bazında topla
    results = []
    for sigma_m in SIGMA_LIST:
        antisym_vals = []
        sym_vals     = []
        for seed in range(N_SEEDS):
            r_plus, r_minus = raw[(sigma_m, seed)]
            antisym_vals.append((r_plus - r_minus) / 2.0)
            sym_vals.append((r_plus + r_minus) / 2.0)
        results.append({
            "sigma_um":       sigma_m * 1e6,
            "antisym_vals":   antisym_vals,
            "sym_vals":       sym_vals,
            "antisym_rms":    float(np.std(antisym_vals)),
            "sym_rms":        float(np.std(sym_vals)),
            "sym_mean":       float(np.mean(sym_vals)),
            "|antisym|_mean": float(np.mean(np.abs(antisym_vals))),
            "|sym|_mean":     float(np.mean(np.abs(sym_vals))),
        })

    with open("test_antisymmetry.json", "w") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nKaydedildi: test_antisymmetry.json")

    # ── Log-log fit ve grafik ───────────────────────────────────────────────
    sigmas        = np.array([r["sigma_um"] for r in results]) * 1e-6
    antisym_means = np.array([r["|antisym|_mean"] for r in results])
    sym_means     = np.array([r["|sym|_mean"]     for r in results])

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

    try:
        for backend in ["Agg", "svg", "pdf"]:
            try:
                matplotlib.use(backend)
                plt.switch_backend(backend)
                break
            except Exception:
                continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.loglog(sigmas * 1e6, antisym_means, "b-o", label=f"|antisim| α={a1:.2f}")
        ax.loglog(sigmas * 1e6, sym_means,     "r-s", label=f"|sim| α={a2:.2f}")
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
        print("Figür: test_antisymmetry.png")
    except Exception as e:
        print(f"Grafik kaydedilemedi ({e}), JSON yeterli.")

    print(f"Toplam süre: {(time.time()-t_wall)/60:.1f} dk")


if __name__ == "__main__":
    main()
