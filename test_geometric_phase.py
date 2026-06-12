#!/usr/bin/env python3
"""test_geometric_phase.py — Omarov Fig 9(a) kopyası: dx+dy birlikte → σ² mü?

HİPOTEZ:
  Geometrik (Berry) faz = sırası değiştirilemeyen ardışık dönmelerin çarpımı.
    dy hatası → B_r = g·(y−dy) → radyal eksen (x̂) etrafında dönme
    dx hatası → B_y = g·(x−dx) → dikey eksen (ŷ) etrafında dönme
  İki FARKLI eksen ancak dx+dy birlikte varsa oluşur → dSy/dt ∝ σ_x·σ_y ∝ σ².
  dy-only kurulumda (önceki testler) komütatör sıfır; birinci mertebe de
  yörünge kapanmasıyla iptal → Omarov'un düz-kafes bulgusuyla tutarlı.

KURULUM (Omarov Fig 9a ile aynı):
  Tüm quadlar hem x hem y yönünde bağımsız N(0, σ) ile kaydırılır.
  A) Omarov: CO=False, t2'yi N_WIN pencereye böl, pencere ortalamalarına fit
  B) CO=True: dikey kapalı yörünge üzerinde fırlat, tüm noktalara fit
     (dx dikey dinamiği ideal quadda etkilemez → dikey CO araması dy ile yeterli)

σ TARAMASI: σ ∈ {2, 5, 10, 20, 40} μm, 5 tohum → α_omarov vs α_cotrue

ÇIKTI: test_geometric_phase.json, test_geometric_phase.png, konsol tablosu
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

N_SEEDS    = 5
SIGMA_LIST = [2e-6, 5e-6, 10e-6, 20e-6, 40e-6]   # [m]
T2         = 1e-3    # toplam süre
N_WIN      = 5       # Omarov pencere sayısı
DT         = 1e-11
RET_STEP   = 5000
CO_TURNS   = 60


def _run_one(task):
    """Tek (seed, sigma) için hem Omarov hem CO=True eğimini hesapla."""
    seed, sigma_m = task
    import os, json, time
    import numpy as np
    from integrator import integrate_particle
    from false_edm_mode_scan import (setup_fields, find_closed_orbit,
                                      _make_state)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open("params.json") as fh:
        cfg = json.load(fh)
    fields, y0, beta0, R0, p_mag, direction = setup_fields(cfg)
    n_q = 2 * int(fields.nFODO)

    # Omarov Fig 9a: HEM x HEM y yönünde bağımsız rastgele kaydırma
    rng = np.random.default_rng(seed)
    quad_dy = rng.normal(0.0, sigma_m, n_q)
    quad_dx = rng.normal(0.0, sigma_m, n_q)

    circ  = (2 * np.pi * R0
             + 4 * fields.nFODO * fields.driftLen
             + 2 * fields.nFODO * fields.quadLen)
    T_rev = circ / (beta0 * C_LIGHT)

    fields.poincare_quad_index = 0.0

    t0 = time.time()

    # ── Yöntem A: Omarov (CO=False, pencereleme) ───────────────────────────
    hist_a, poin_a, pt_a = integrate_particle(
        y0, 0.0, T2, DT, fields=fields,
        return_steps=RET_STEP, quad_dy=quad_dy, quad_dx=quad_dx)

    slope_omarov = float("nan")
    if poin_a is not None and len(poin_a) > N_WIN * 2:
        ts_a = np.asarray(pt_a, float)
        ss_a = np.asarray(poin_a[:, 7], float)
        win_size = T2 / N_WIN
        t_mids, sy_means = [], []
        for i in range(N_WIN):
            t_lo = i * win_size
            t_hi = (i + 1) * win_size
            mask = (ts_a >= t_lo) & (ts_a < t_hi)
            if mask.sum() > 1:
                t_mids.append(0.5 * (t_lo + t_hi))
                sy_means.append(ss_a[mask].mean())
        if len(t_mids) >= 2:
            slope_omarov = float(np.polyfit(t_mids, sy_means, 1)[0])

    # Richardson hatası için 2dt koşumu
    hist_a2, poin_a2, pt_a2 = integrate_particle(
        y0, 0.0, T2, 2 * DT, fields=fields,
        return_steps=RET_STEP, quad_dy=quad_dy, quad_dx=quad_dx)
    slope_omarov2 = float("nan")
    if poin_a2 is not None and len(poin_a2) > N_WIN * 2:
        ts_a2 = np.asarray(pt_a2, float)
        ss_a2 = np.asarray(poin_a2[:, 7], float)
        win_size = T2 / N_WIN
        t_mids2, sy_means2 = [], []
        for i in range(N_WIN):
            t_lo = i * win_size
            t_hi = (i + 1) * win_size
            mask = (ts_a2 >= t_lo) & (ts_a2 < t_hi)
            if mask.sum() > 1:
                t_mids2.append(0.5 * (t_lo + t_hi))
                sy_means2.append(ss_a2[mask].mean())
        if len(t_mids2) >= 2:
            slope_omarov2 = float(np.polyfit(t_mids2, sy_means2, 1)[0])
    err_omarov = abs(slope_omarov - slope_omarov2) if not (
        np.isnan(slope_omarov) or np.isnan(slope_omarov2)) else float("nan")

    # ── Yöntem B: CO=True ──────────────────────────────────────────────────
    v_co, resid = find_closed_orbit(
        fields, p_mag, direction, quad_dy, DT, T_rev,
        n_turns=CO_TURNS, n_iter=2)
    y_co = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])

    hist_b, poin_b, pt_b = integrate_particle(
        y_co, 0.0, T2, DT, fields=fields,
        return_steps=RET_STEP, quad_dy=quad_dy, quad_dx=quad_dx)

    slope_cotrue = float("nan")
    if poin_b is not None and len(poin_b) > 5:
        ts_b = np.asarray(pt_b, float)
        ss_b = np.asarray(poin_b[:, 7], float)
        slope_cotrue = float(np.polyfit(ts_b, ss_b, 1)[0])

    _, poin_b2, pt_b2 = integrate_particle(
        y_co, 0.0, T2, 2 * DT, fields=fields,
        return_steps=RET_STEP, quad_dy=quad_dy, quad_dx=quad_dx)
    err_cotrue = float("nan")
    if poin_b2 is not None and len(poin_b2) > 5:
        slope_b2 = float(np.polyfit(
            np.asarray(pt_b2, float), np.asarray(poin_b2[:, 7], float), 1)[0])
        err_cotrue = abs(slope_cotrue - slope_b2)

    elapsed = time.time() - t0
    n_turns = T2 / T_rev
    return {
        "seed": seed,
        "sigma_um": sigma_m * 1e6,
        "n_turns": n_turns,
        "slope_omarov": slope_omarov,
        "err_omarov": err_omarov,
        "slope_cotrue": slope_cotrue,
        "err_cotrue": err_cotrue,
        "co_off_mm": float(np.hypot(v_co[0], v_co[1]) * 1e3),
        "runtime_s": elapsed,
    }


def fit_alpha(sigma_list, medians):
    """Log-log eğim — NaN ve sıfırları atla."""
    valid = [(s, m) for s, m in zip(sigma_list, medians)
             if m > 0 and not np.isnan(m)]
    if len(valid) < 2:
        return float("nan"), float("nan")
    ls = np.log10([v[0] for v in valid])
    lf = np.log10([v[1] for v in valid])
    alpha, logb = np.polyfit(ls, lf, 1)
    return float(alpha), float(10**logb)


def main():
    t_wall = time.time()
    ctx = mp.get_context("spawn")

    tasks = [(seed, sigma)
             for seed in range(N_SEEDS)
             for sigma in SIGMA_LIST]

    print("=" * 72)
    print(f"  GEOMETRİK FAZ TESTİ — Omarov Fig 9a: dx+dy birlikte, σ² mü?")
    print(f"  {N_SEEDS} tohum × {len(SIGMA_LIST)} σ = {len(tasks)} koşum")
    print(f"  t2={T2*1e3:.0f} ms, pencere={T2/N_WIN*1e3:.1f} ms")
    print("=" * 72)
    header = (f"{'tohum':>5} {'σ[μm]':>7} "
              f"{'Omarov[rad/s]':>14} {'hata_O':>10} "
              f"{'CO-True[rad/s]':>14} {'hata_C':>10}  {'süre[s]':>7}")
    print(header)
    print("-" * len(header))
    sys.stdout.flush()

    results = []
    json_path = "test_geometric_phase.json"
    nw = min(mp.cpu_count(), len(tasks))
    with ctx.Pool(processes=nw) as pool:
        for r in pool.imap_unordered(_run_one, tasks):
            results.append(r)
            print(f"{r['seed']:>5} {r['sigma_um']:>7.1f} "
                  f"{r['slope_omarov']:>14.3e} {r['err_omarov']:>10.2e} "
                  f"{r['slope_cotrue']:>14.3e} {r['err_cotrue']:>10.2e}  "
                  f"{r['runtime_s']:>7.1f}")
            sys.stdout.flush()
            with open(json_path, "w") as fh:
                json.dump({"_partial": True, "satirlar": results}, fh,
                          indent=2, ensure_ascii=False)

    # ── İstatistik özeti ────────────────────────────────────────────────────
    results_sorted = sorted(results, key=lambda r: (r["sigma_um"], r["seed"]))
    sigma_vals = sorted(set(r["sigma_um"] for r in results))

    print(f"\n{'─'*72}")
    print(f"{'σ[μm]':>7}  {'medyan|Omarov|':>14}  {'medyan|CO-True|':>15}")
    print("-" * 42)
    summary = []
    for sv in sigma_vals:
        grp = [r for r in results if r["sigma_um"] == sv]
        med_o = float(np.median([abs(r["slope_omarov"]) for r in grp
                                 if not np.isnan(r["slope_omarov"])]))
        med_c = float(np.median([abs(r["slope_cotrue"]) for r in grp
                                 if not np.isnan(r["slope_cotrue"])]))
        print(f"{sv:>7.1f}  {med_o:>14.3e}  {med_c:>15.3e}")
        summary.append({"sigma_um": sv, "med_omarov": med_o, "med_cotrue": med_c})

    alpha_o, b_o = fit_alpha(
        [r["sigma_um"] for r in summary],
        [r["med_omarov"] for r in summary])
    alpha_c, b_c = fit_alpha(
        [r["sigma_um"] for r in summary],
        [r["med_cotrue"] for r in summary])

    print(f"\nOmarov  eğim fit: α = {alpha_o:.2f}  (|f| = {b_o:.3e} × σ^α)")
    print(f"CO-True eğim fit: α = {alpha_c:.2f}  (|f| = {b_c:.3e} × σ^α)")

    label_o = ("doğrusal (σ¹)" if abs(alpha_o - 1) < 0.3
               else "kuadratik (σ²)" if abs(alpha_o - 2) < 0.3
               else f"belirsiz (α={alpha_o:.2f})")
    label_c = ("doğrusal (σ¹)" if abs(alpha_c - 1) < 0.3
               else "kuadratik (σ²)" if abs(alpha_c - 2) < 0.3
               else f"belirsiz (α={alpha_c:.2f})")
    print(f"  Omarov → {label_o}")
    print(f"  CO-True → {label_c}")

    # ── Kayıt ───────────────────────────────────────────────────────────────
    out = {
        "t2_ms": T2 * 1e3, "N_WIN": N_WIN, "N_SEEDS": N_SEEDS,
        "sigma_list_um": [s * 1e6 for s in SIGMA_LIST],
        "alpha_omarov": alpha_o, "b_omarov": b_o,
        "alpha_cotrue": alpha_c, "b_cotrue": b_c,
        "ozet": summary,
        "satirlar": results_sorted,
    }
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nKaydedildi:", json_path)

    # ── Figür ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sg = np.logspace(np.log10(min(SIGMA_LIST)*1e6*0.8),
                     np.log10(max(SIGMA_LIST)*1e6*1.2), 100)

    for ax_idx, (key, label, alpha, b, color) in enumerate([
        ("slope_omarov", "Omarov (pencereleme)", alpha_o, b_o, "tab:orange"),
        ("slope_cotrue", "CO=True",             alpha_c, b_c, "tab:blue"),
    ]):
        ax = axes[ax_idx]
        for seed in range(N_SEEDS):
            sigs = sorted(set(r["sigma_um"] for r in results))
            pts = [(r["sigma_um"], abs(r[key])) for r in results
                   if r["seed"] == seed and not np.isnan(r[key])]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        "o", alpha=0.4, ms=5, color=color)

        med_pts = [(r["sigma_um"],
                    np.median([abs(rr[key]) for rr in results
                               if rr["sigma_um"] == r["sigma_um"]
                               and not np.isnan(rr[key])]))
                   for r in summary]
        ax.plot([p[0] for p in med_pts], [p[1] for p in med_pts],
                "^-", ms=9, lw=2, color=color, label="medyan")

        if not np.isnan(alpha):
            ax.plot(sg, b * sg**alpha, "-", lw=2, color="black",
                    label=f"fit: σ^{alpha:.2f}")

        # Referans çizgileri
        med10 = next((r["med_omarov" if key == "slope_omarov" else "med_cotrue"]
                      for r in summary if r["sigma_um"] == 10.0), None)
        if med10 and med10 > 0:
            ax.plot(sg, med10 * (sg / 10)**1, "g--", lw=1, alpha=0.6,
                    label="∝ σ¹")
            ax.plot(sg, med10 * (sg / 10)**2, "b--", lw=1, alpha=0.6,
                    label="∝ σ²")

        ax.axhline(1e-5, color="green", ls=":", lw=1, label="Omarov 10μm~1e-5")
        ax.axhline(2e-9, color="purple", ls=":", lw=1, label="CO=True ~2e-9")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("σ [μm rms]", fontsize=11)
        ax.set_ylabel("|dSy/dt| [rad/s]", fontsize=11)
        ax.set_title(f"{label}  —  α = {alpha:.2f}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

    plt.suptitle(f"Geometrik faz (dx+dy)  |  t2={T2*1e3:.0f} ms, "
                 f"N_win={N_WIN}", fontsize=12)
    plt.tight_layout()
    plt.savefig("test_geometric_phase.png", dpi=150)
    print("Figür: test_geometric_phase.png")
    print(f"Toplam süre: {(time.time()-t_wall)/60:.1f} dk")


if __name__ == "__main__":
    main()
