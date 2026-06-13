#!/usr/bin/env python3
"""test_sigma_scan.py — Sahte EDM'nin hizalama σ'sına bağımlılığı.

SORU:
  f ≡ |dSy/dt| ~ σ^α   nerede  α = ?
    α=1  → birinci mertebe doğrusal (deterministic first-order secular)
    α=2  → ikinci mertebe kuadratik (Omarov geometrik faz, rastgele desen)

YÖNTEM:
  CO=False, t2=20 ms (t2 yakınsama testinden seçildi)
  σ ∈ {2, 5, 10, 20, 40} μm rms, N_SEEDS tohum her nokta için
  Log-log eğim fit → α belirlenir

ÇIKTI: test_sigma_scan.json, test_sigma_scan.png, konsol tablosu
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
SIGMA_LIST = [2e-6, 5e-6, 10e-6, 20e-6, 40e-6]   # [m]
T2        = 1e-3     # 1 ms — CO=True modunda t2=0.5ms'de bile kararlı
RET_STEP  = 5000
DT        = 1e-11
DO_CO     = True     # kapalı yörünge üzerinde fırlat → betatron yok → temiz σ eğimi


CO_TURNS = 60   # kapalı yörünge arama probu için tur sayısı


def _run_one_sigma(task):
    """Tek (seed, sigma) çifti için simülasyon (multiprocessing worker).

    task = (seed, sigma_m, t2, return_steps, dt, do_co)
    """
    seed, sigma_m, t2, return_steps, dt, do_co = task
    import os, json, time
    import numpy as np
    from integrator import integrate_particle
    from false_edm_mode_scan import (setup_fields, find_closed_orbit,
                                      _make_state)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open("params.json") as fh:
        config = json.load(fh)
    fields, y0, beta0, R0, p_mag, direction = setup_fields(config)
    n_q = 2 * int(fields.nFODO)

    rng = np.random.default_rng(seed)
    quad_dy = rng.normal(0.0, sigma_m, n_q)

    circ  = (2 * np.pi * R0
             + 4 * fields.nFODO * fields.driftLen
             + 2 * fields.nFODO * fields.quadLen)
    T_rev = circ / (beta0 * C_LIGHT)

    t_start = time.time()

    if do_co:
        v_co, resid_rms = find_closed_orbit(
            fields, p_mag, direction, quad_dy, dt, T_rev,
            n_turns=CO_TURNS, n_iter=2)
        y_launch = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
        co_off_mm = float(np.hypot(v_co[0], v_co[1]) * 1e3)
        resid_beta_mm = float(resid_rms * 1e3)
    else:
        y_launch = y0
        co_off_mm = 0.0
        resid_beta_mm = float("nan")

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
        # Richardson yakınsama hatası
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
        "co_off_mm": co_off_mm,
        "resid_beta_mm": resid_beta_mm,
        "dSy_dt": slope,
        "dSy_dt_err": slope_err,
        "runtime_s": elapsed,
    }


def main():
    t_wall = time.time()
    ctx = mp.get_context("spawn")

    tasks = []
    for seed in range(N_SEEDS):
        for sigma in SIGMA_LIST:
            tasks.append((seed, sigma, T2, RET_STEP, DT, DO_CO))

    co_str = "True (kapalı yörünge)" if DO_CO else "False (eksenden)"
    print("=" * 72)
    print(f"  σ TARAMA TESTİ — CO={co_str}, t2={T2*1e3:.0f} ms")
    print(f"  {N_SEEDS} tohum × {len(SIGMA_LIST)} σ değeri = {len(tasks)} koşum")
    print(f"  σ ∈ {[s*1e6 for s in SIGMA_LIST]} μm rms")
    print("=" * 72)
    print(f"{'tohum':>6} {'σ[μm]':>8} {'tur':>6} "
          f"{'dSy/dt[rad/s]':>15} {'hata':>12} {'süre[s]':>8}")
    print("-" * 64)
    sys.stdout.flush()

    # imap_unordered: her iş biter bitmez yaz + kaydet
    results = []
    json_path = "test_sigma_scan.json"
    nw = min(mp.cpu_count(), len(tasks))
    with ctx.Pool(processes=nw) as pool:
        for r in pool.imap_unordered(_run_one_sigma, tasks):
            results.append(r)
            print(f"{r['seed']:>6} {r['sigma_um']:>8.1f} {r['n_turns']:>6.0f} "
                  f"{r['dSy_dt']:>15.3e} {r['dSy_dt_err']:>12.2e} "
                  f"{r['runtime_s']:>8.1f}")
            sys.stdout.flush()
            # her adımda JSON'a yaz — kill'e karşı güvenli
            with open(json_path, "w") as fh:
                json.dump({"_partial": True, "satirlar": results}, fh,
                          indent=2, ensure_ascii=False)

    # ── İstatistik özeti (her σ için) ────────────────────────────────────────
    results_sorted = sorted(results, key=lambda r: (r["sigma_um"], r["seed"]))
    print(f"\n{'─'*72}")
    print(f"{'σ[μm]':>8} {'medyan|f|':>12} {'std':>12} {'min':>12} {'maks':>12}")
    print("-" * 60)
    sigma_vals = sorted(set(r["sigma_um"] for r in results))
    summary_rows = []
    for sv in sigma_vals:
        grp = [r for r in results if r["sigma_um"] == sv]
        f_abs = np.array([abs(r["dSy_dt"]) for r in grp])
        med = float(np.median(f_abs))
        std = float(f_abs.std())
        mn  = float(f_abs.min())
        mx  = float(f_abs.max())
        print(f"{sv:>8.1f} {med:>12.3e} {std:>12.3e} {mn:>12.3e} {mx:>12.3e}")
        summary_rows.append({
            "sigma_um": sv,
            "median": med, "std": std, "min": mn, "max": mx,
            "n_seeds": len(grp),
        })

    # ── Log-log eğim fit ─────────────────────────────────────────────────────
    log_s = np.log10([r["sigma_um"] for r in summary_rows])
    log_f = np.log10([r["median"] for r in summary_rows
                      if r["median"] > 0])
    log_s_clean = log_s[[i for i, r in enumerate(summary_rows) if r["median"] > 0]]
    if len(log_s_clean) >= 2:
        alpha, logb = np.polyfit(log_s_clean, log_f, 1)
        b = 10**logb
        print(f"\nLog-log eğim fit: |f| = {b:.3e} × σ^{alpha:.2f}")
        print(f"  α = {alpha:.2f}  "
              f"({'doğrusal (1. mertebe)' if abs(alpha-1) < 0.3 else 'kuadratik (2. mertebe)' if abs(alpha-2) < 0.3 else 'belirsiz'})")
    else:
        alpha = float("nan"); b = float("nan")
        print("\nYetersiz veri noktası için fit yapılamadı.")

    # ── Kayıt ───────────────────────────────────────────────────────────────
    out = {
        "_aciklama": "σ tarama testi: |dSy/dt| vs hizalama hatası σ, CO=False",
        "t2_ms": T2 * 1e3,
        "N_SEEDS": N_SEEDS,
        "sigma_list_um": [s * 1e6 for s in SIGMA_LIST],
        "DO_CO": DO_CO,
        "fit_alpha": float(alpha),
        "fit_b": float(b),
        "ozet": summary_rows,
        "satirlar": results_sorted,
    }
    with open("test_sigma_scan.json", "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nKaydedildi: test_sigma_scan.json")

    # ── Figür ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Sol: σ vs |f| log-log, her tohum + medyan + fit
    ax = axes[0]
    sigma_vals_arr = sorted(set(r["sigma_um"] for r in results))
    for seed in range(N_SEEDS):
        sigs = []
        fabs = []
        for sv in sigma_vals_arr:
            r = next((r for r in results
                      if r["seed"] == seed and r["sigma_um"] == sv), None)
            if r:
                sigs.append(sv)
                fabs.append(abs(r["dSy_dt"]))
        if sigs:
            ax.plot(sigs, fabs, "o", alpha=0.4, ms=5)

    # Medyan
    med_sigs = [r["sigma_um"] for r in summary_rows]
    med_fabs = [r["median"] for r in summary_rows]
    ax.plot(med_sigs, med_fabs, "k^-", ms=9, lw=2, label="medyan")

    # Fit eğrileri
    sg = np.logspace(np.log10(min(sigma_vals_arr)*0.8),
                     np.log10(max(sigma_vals_arr)*1.2), 100)
    if not np.isnan(alpha):
        ax.plot(sg, b * (sg**alpha), "r-", lw=1.8,
                label=f"fit: σ^{alpha:.2f}")
    # Referans: σ¹ ve σ²
    med10 = next((r["median"] for r in summary_rows if r["sigma_um"] == 10.0),
                 None)
    if med10 and med10 > 0:
        ax.plot(sg, med10 * (sg / 10)**1, "g--", lw=1, alpha=0.7,
                label="∝ σ¹ (1. mertebe)")
        ax.plot(sg, med10 * (sg / 10)**2, "b--", lw=1, alpha=0.7,
                label="∝ σ² (2. mertebe)")

    ax.axhline(1e-5, color="green", ls=":", lw=1,
               label="Omarov 10 μm ~1e-5")
    ax.axhline(2e-9, color="purple", ls=":", lw=1,
               label="CO=True ~2e-9")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("σ [μm rms]", fontsize=11)
    ax.set_ylabel("|dSy/dt| [rad/s]", fontsize=11)
    ax.set_title(f"Sahte EDM vs σ — CO=False, t2={T2*1e3:.0f} ms", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # Sağ: tek-tohum σ saçılımı (box-plot benzeri)
    ax2 = axes[1]
    positions = list(range(len(sigma_vals_arr)))
    for i, sv in enumerate(sigma_vals_arr):
        grp = [abs(r["dSy_dt"]) for r in results if r["sigma_um"] == sv
               and not np.isnan(r["dSy_dt"])]
        if grp:
            ax2.scatter([sv] * len(grp), grp, alpha=0.6, s=30, zorder=3)
            ax2.errorbar([sv], [np.median(grp)],
                         yerr=[[np.median(grp) - min(grp)],
                               [max(grp) - np.median(grp)]],
                         fmt="k^", ms=8, capsize=5, lw=2, zorder=4)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("σ [μm rms]", fontsize=11)
    ax2.set_ylabel("|dSy/dt| [rad/s]", fontsize=11)
    ax2.set_title(f"Tohum saçılımı — {N_SEEDS} tohum/nokta", fontsize=10)
    ax2.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("test_sigma_scan.png", dpi=150)
    print("Figür: test_sigma_scan.png")
    print(f"Toplam süre: {(time.time()-t_wall)/60:.1f} dk")


if __name__ == "__main__":
    main()
