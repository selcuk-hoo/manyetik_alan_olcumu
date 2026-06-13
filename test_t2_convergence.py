#!/usr/bin/env python3
"""test_t2_convergence.py — CO=False t2 yakınsama testi.

SORU:
  CO=False modunda rastgele 10 μm rms hizalama hatası için dSy/dt eğimi
  t2 arttıkça nereye yakınsar?
    (a) ~1e-5 rad/s  → Omarov'un geometrik faz (ikinci mertebe, σ²)
    (b) ~1e-9 rad/s  → CO=True seküler (birinci mertebe sıfırlanır)
    (c) Başka bir değer

YÖNTEM:
  N_SEEDS tohum × 4 t2 değeri = 20 paralel koşum
  t2 ∈ {1, 3, 10, 20} ms, σ=10 μm rms, CO=False, rastgele desen
  Stroboskopik ölçüm (tur-başına S_y örneklemesi)

ÇIKTI: test_t2_convergence.json, test_t2_convergence.png, konsol tablosu
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)

from false_edm_mode_scan import setup_fields

C_LIGHT = 299792458.0

N_SEEDS   = 5
SIGMA_M   = 1e-5        # 10 μm rms
T2_LIST   = [1e-3, 3e-3, 10e-3, 20e-3]   # [s]
RET_STEP  = 5000
DT        = 1e-11
DO_CO     = False


def _run_one_random(task):
    """Rastgele quad_dy deseni için tek simülasyon (multiprocessing worker).

    task = (seed, sigma_m, t2, return_steps, dt, do_co)
    """
    seed, sigma_m, t2, return_steps, dt, do_co = task
    import os, json, time
    import numpy as np
    from integrator import integrate_particle
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open("params.json") as fh:
        config = json.load(fh)
    fields, y0, beta0, R0, p_mag, direction = setup_fields(config)
    n_q = 2 * int(fields.nFODO)

    # Rastgele misalignment deseni
    rng = np.random.default_rng(seed)
    quad_dy = rng.normal(0.0, sigma_m, n_q)

    circ  = (2 * np.pi * R0
             + 4 * fields.nFODO * fields.driftLen
             + 2 * fields.nFODO * fields.quadLen)
    T_rev = circ / (beta0 * C_LIGHT)

    t_start = time.time()

    # CO=False: eksenden fırlat (betatron salınımı var)
    y_launch = y0

    # Stroboskopik ölçüm: tur-başına S_y örneklemesi
    fields.poincare_quad_index = 0.0
    hist, poin, poin_t = integrate_particle(
        y_launch, 0.0, t2, dt, fields=fields,
        return_steps=return_steps, quad_dy=quad_dy)

    slope = float("nan")
    slope_err = float("nan")
    n_poin = 0
    if poin is not None and len(poin) > 5:
        ts = np.asarray(poin_t, float)
        ss = np.asarray(poin[:, 7], float)
        n_poin = len(ts)
        slope = float(np.polyfit(ts, ss, 1)[0])
        # Yakınsama hatası: dt=2 koşumla Richardson farkı
        _, poin2, poin2_t = integrate_particle(
            y_launch, 0.0, t2, 2 * dt, fields=fields,
            return_steps=return_steps, quad_dy=quad_dy)
        if poin2 is not None and len(poin2) > 5:
            slope2 = float(np.polyfit(
                np.asarray(poin2_t, float),
                np.asarray(poin2[:, 7], float), 1)[0])
            slope_err = abs(slope - slope2)

    n_turns = t2 / T_rev
    elapsed = time.time() - t_start
    return {
        "seed": seed,
        "sigma_um": sigma_m * 1e6,
        "t2": t2,
        "n_turns": n_turns,
        "n_poin": n_poin,
        "dSy_dt": slope,
        "dSy_dt_err": slope_err,
        "runtime_s": elapsed,
        # S_y zaman serisi (stroboskopik) — görselleştirme için
        "sy_strobe": (poin[:, 7].tolist() if poin is not None and len(poin) > 5
                      else []),
        "t_strobe": (poin_t if isinstance(poin_t, list)
                     else poin_t.tolist() if poin_t is not None else []),
    }


def main():
    t_wall = time.time()
    ctx = mp.get_context("spawn")

    tasks = []
    meta  = []
    for seed in range(N_SEEDS):
        for t2 in T2_LIST:
            tasks.append((seed, SIGMA_M, t2, RET_STEP, DT, DO_CO))
            meta.append((seed, t2))

    print("=" * 72)
    print(f"  T2 YAKINSAMA TESTİ — CO=False, σ={SIGMA_M*1e6:.0f} μm rms")
    print(f"  {N_SEEDS} tohum × {len(T2_LIST)} t2 değeri = {len(tasks)} koşum")
    print(f"  t2 ∈ {[t*1e3 for t in T2_LIST]} ms")
    print("=" * 72)

    nw = min(mp.cpu_count(), len(tasks))
    with ctx.Pool(processes=nw) as pool:
        results = pool.map(_run_one_random, tasks)

    # ── Konsol tablosu ──────────────────────────────────────────────────────
    print(f"\n{'tohum':>6} {'t2[ms]':>7} {'tur':>6} "
          f"{'dSy/dt[rad/s]':>15} {'hata':>12} {'süre[s]':>8}")
    print("-" * 62)
    for r in results:
        print(f"{r['seed']:>6} {r['t2']*1e3:>7.1f} {r['n_turns']:>6.0f} "
              f"{r['dSy_dt']:>15.3e} {r['dSy_dt_err']:>12.2e} "
              f"{r['runtime_s']:>8.1f}")

    # ── İstatistik özeti (her t2 için) ──────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"{'t2[ms]':>7} {'tur':>6} {'medyan|f|':>12} {'min|f|':>12} "
          f"{'maks|f|':>12} {'std':>12}")
    print("-" * 68)
    t2_vals = sorted(set(r["t2"] for r in results))
    summary_rows = []
    for t2 in t2_vals:
        grp = [r for r in results if r["t2"] == t2]
        f_abs = np.array([abs(r["dSy_dt"]) for r in grp])
        med = float(np.median(f_abs))
        mn  = float(f_abs.min())
        mx  = float(f_abs.max())
        std = float(f_abs.std())
        n_turns = grp[0]["n_turns"]
        print(f"{t2*1e3:>7.1f} {n_turns:>6.0f} {med:>12.3e} {mn:>12.3e} "
              f"{mx:>12.3e} {std:>12.3e}")
        summary_rows.append({"t2_ms": t2*1e3, "n_turns": n_turns,
                              "median": med, "min": mn, "max": mx, "std": std})

    # Yakınsama yorumu
    medians = [r["median"] for r in summary_rows]
    ratio_1_20 = medians[0] / max(medians[-1], 1e-30)
    print(f"\nYAKINSAMA: t2=1ms / t2=20ms oranı = {ratio_1_20:.1f}×")
    if medians[-1] < 1e-7:
        print("  → 20 ms'de 1e-7'nin altında: Omarov'un 1e-5 tabanına ulaşılamadı.")
        print("    Kaynakları: betatron sızması tamamen kaybolmamış VEYA")
        print("    gerçek seküler sinyal CO=True seviyesine yakınsıyor.")
    elif medians[-1] < 5e-5:
        print("  → ~1e-5 band'ında: Omarov'un geometrik faz bölgesiyle uyumlu.")
    else:
        print("  → Hâlâ yüksek; daha uzun t2 gerekiyor olabilir.")

    # ── Kayıt ───────────────────────────────────────────────────────────────
    out = {
        "_aciklama": "CO=False t2 yakınsama testi, σ=10 μm rms rastgele desen",
        "sigma_um": SIGMA_M * 1e6,
        "N_SEEDS": N_SEEDS,
        "t2_list_ms": [t*1e3 for t in T2_LIST],
        "DO_CO": DO_CO,
        "ozet": summary_rows,
        "satirlar": [{k: v for k, v in r.items()
                      if k not in ("sy_strobe", "t_strobe")}
                     for r in results],
    }
    with open("test_t2_convergence.json", "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nKaydedildi: test_t2_convergence.json")

    # ── Figür ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    t2_arr = np.array([r["t2"] for r in results]) * 1e3
    f_abs  = np.array([abs(r["dSy_dt"]) for r in results])
    # Her tohum için çizgi
    for seed in range(N_SEEDS):
        mask = [r["seed"] == seed for r in results]
        t2_s = np.array([r["t2"] for r, m in zip(results, mask) if m]) * 1e3
        f_s  = np.array([abs(r["dSy_dt"]) for r, m in zip(results, mask) if m])
        idx  = np.argsort(t2_s)
        ax.plot(t2_s[idx], f_s[idx], "o-", alpha=0.5, ms=5,
                label=f"tohum {seed}")

    # Medyan
    t2_med = np.array([r["t2_ms"] for r in summary_rows])
    f_med  = np.array([r["median"] for r in summary_rows])
    ax.plot(t2_med, f_med, "k^-", ms=9, lw=2, label="medyan")

    # Referans çizgileri
    ax.axhline(1e-5, color="green", ls="--", lw=1.2,
               label="Omarov geometrik faz ~1e-5")
    ax.axhline(2e-9, color="purple", ls=":", lw=1.2,
               label="CO=True seküler ~2e-9")

    # t2^{-2} ölçek kılavuzu: 1 ms noktasına sabitle
    if not np.isnan(f_med[0]) and f_med[0] > 0:
        t2_guide = np.logspace(np.log10(t2_med[0]), np.log10(t2_med[-1]), 50)
        guide = f_med[0] * (t2_guide / t2_med[0])**(-2)
        ax.plot(t2_guide, guide, "r--", lw=1, alpha=0.6, label="∝ t2⁻²")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("izleme süresi t2 [ms]", fontsize=11)
    ax.set_ylabel("|dSy/dt| [rad/s]", fontsize=11)
    ax.set_title(f"CO=False t2 yakınsama — σ={SIGMA_M*1e6:.0f} μm rms, "
                 f"{N_SEEDS} tohum", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")

    # Sağ panel: en uzun t2 (20 ms) için stroboskopik S_y serisi örneği
    ax2 = axes[1]
    long_t2 = max(T2_LIST)
    for seed in range(min(N_SEEDS, 3)):
        r = next((r for r in results
                  if r["seed"] == seed and abs(r["t2"] - long_t2) < 1e-6), None)
        if r is None or not r["sy_strobe"]:
            continue
        ts = np.array(r["t_strobe"]) * 1e3
        ss = np.array(r["sy_strobe"])
        fit = np.polyfit(ts / 1e3, ss, 1)
        ax2.plot(ts, (ss - fit[1]) * 1e9, ".", ms=1.8, alpha=0.7,
                 label=f"tohum {seed}: {r['dSy_dt']:.2e} rad/s")
    ax2.set_xlabel("t [ms]", fontsize=11)
    ax2.set_ylabel(r"$\Delta S_y$ (stroboskopik, DC çıkarıldı) [$\times 10^{-9}$]",
                   fontsize=10)
    ax2.set_title(f"t2={long_t2*1e3:.0f} ms stroboskopik S_y", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("test_t2_convergence.png", dpi=150)
    print("Figür: test_t2_convergence.png")
    print(f"Toplam süre: {(time.time()-t_wall)/60:.1f} dk")


if __name__ == "__main__":
    main()
