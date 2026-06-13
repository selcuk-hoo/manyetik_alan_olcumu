#!/usr/bin/env python3
"""test_sy_diagnostics.py — σ¹ sinyali gerçek mi, artefakt mı? Üç ayrıştırıcı test.

TEORİK BEKLENTİ:
  Dikey kapalı yörüngede ∮B_x ds = 0 (kapanma koşulu, E_z=0) →
  birinci mertebe spin birikimi ΔS_y/tur ∝ (G+1/γ)·∮B_x ds = 0,
  HER örgüde, simetri gerektirmeden. Ölçtüğümüz σ¹ bu yüzden şüpheli.

ÜÇ TEST (hepsi CO=True, seed 0):
  1) STROBOSKOPİK: tur-başına Poincaré S_y serisinin ilk ~30 turdaki
     birikim hızı. Teorik ~0; σ ile orantılı çıkarsa integratör
     tutarsızlığı (spin kick ≠ yörünge kuvveti örneklemesi).
  2) PENCERE BAĞIMLILIĞI: aynı 3 ms koşumda fit penceresi
     [0,0.5] [0,1] [0,2] [0,3] ms → eğim pencereyle düşüyorsa S_y
     yavaş salınımdır (sinkrotron?), lineer fit artefaktı.
  3) İZ + FFT: S_y(t) sürekli kaydı, lineer fit artığının baskın
     frekansı → betatron mu (≈ Q_y/T_rev ~ yüzlerce kHz) yoksa
     sinkrotron mu (~kHz) ayırt edilir.

ÇIKTI: test_sy_diagnostics.json, test_sy_diagnostics.png, konsol
"""

import json
import os
import sys
import time

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

SEED       = 0
SIGMA_LIST = [10e-6, 40e-6]     # [m] — iki nokta: kalan etkinin σ ölçeği için
T2         = 3e-3               # [s] uzun koşum (pencere testi için)
DT         = 1e-11
RET_STEP   = 2000               # sürekli iz örnekleme: 20 ns
CO_TURNS   = 60
CO_ITER    = 3                  # CO kalitesi: 2 → 3 (kalıntı şüphesine karşı)
WINDOWS_MS = [0.5, 1.0, 2.0, 3.0]


def main():
    t_wall = time.time()
    with open("params.json") as fh:
        cfg = json.load(fh)

    out = {"seed": SEED, "t2_ms": T2 * 1e3, "co_iter": CO_ITER, "vakalar": []}
    fig, axes = plt.subplots(len(SIGMA_LIST), 3, figsize=(16, 4.5 * len(SIGMA_LIST)))
    if len(SIGMA_LIST) == 1:
        axes = axes[None, :]

    for row, sigma_m in enumerate(SIGMA_LIST):
        fields, y0, beta0, R0, p_mag, direction = setup_fields(cfg)
        n_q = 2 * int(fields.nFODO)

        rng = np.random.default_rng(SEED)
        quad_dy = rng.normal(0.0, sigma_m, n_q)
        quad_dx = rng.normal(0.0, sigma_m, n_q)

        circ  = (2 * np.pi * R0
                 + 4 * fields.nFODO * fields.driftLen
                 + 2 * fields.nFODO * fields.quadLen)
        T_rev = circ / (beta0 * C_LIGHT)

        print(f"\n{'='*72}")
        print(f"  σ = {sigma_m*1e6:.0f} μm  |  CO arama (n_iter={CO_ITER})...")
        v_co, resid = find_closed_orbit(
            fields, p_mag, direction, quad_dy, DT, T_rev,
            n_turns=CO_TURNS, n_iter=CO_ITER)
        y_co = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
        print(f"  CO ofseti = {np.hypot(v_co[0], v_co[1])*1e3:.4f} mm, "
              f"betatron kalıntısı = {resid*1e3:.3e} mm")

        fields.poincare_quad_index = 0.0
        hist, poin, poin_t = integrate_particle(
            y_co, 0.0, T2, DT, fields=fields,
            return_steps=RET_STEP, quad_dy=quad_dy, quad_dx=quad_dx)

        ts_p = np.asarray(poin_t, float)
        sy_p = np.asarray(poin[:, 7], float)

        # ── Test 1: stroboskopik erken birikim ──────────────────────────────
        n_early = min(30, len(sy_p))
        rate_strobo = float(np.polyfit(ts_p[:n_early], sy_p[:n_early], 1)[0])
        per_turn = rate_strobo * T_rev
        print(f"  [1] Stroboskopik (ilk {n_early} tur): "
              f"dSy/dt = {rate_strobo:.3e} rad/s  "
              f"(= {per_turn:.3e} rad/tur)")

        # ── Test 2: fit penceresi bağımlılığı ──────────────────────────────
        win_slopes = []
        for w_ms in WINDOWS_MS:
            mask = ts_p <= w_ms * 1e-3
            if mask.sum() > 5:
                s = float(np.polyfit(ts_p[mask], sy_p[mask], 1)[0])
            else:
                s = float("nan")
            win_slopes.append(s)
            print(f"  [2] pencere [0, {w_ms:.1f}] ms → eğim = {s:.3e} rad/s")

        # ── Test 3: sürekli iz + artık FFT ─────────────────────────────────
        ts_h = np.arange(len(hist)) * (RET_STEP * DT)
        sy_h = np.asarray(hist[:, 7], float)
        coef = np.polyfit(ts_h, sy_h, 1)
        resid_h = sy_h - np.polyval(coef, ts_h)
        # FFT — baskın frekans
        fft = np.fft.rfft(resid_h * np.hanning(len(resid_h)))
        freqs = np.fft.rfftfreq(len(resid_h), RET_STEP * DT)
        k_peak = int(np.argmax(np.abs(fft[1:])) + 1)
        f_peak = float(freqs[k_peak])
        f_rev = 1.0 / T_rev
        print(f"  [3] artık salınım: genlik(rms) = {resid_h.std():.3e}, "
              f"baskın f = {f_peak/1e3:.2f} kHz "
              f"(f_rev = {f_rev/1e3:.1f} kHz, oran = {f_peak/f_rev:.3f})")

        out["vakalar"].append({
            "sigma_um": sigma_m * 1e6,
            "co_off_mm": float(np.hypot(v_co[0], v_co[1]) * 1e3),
            "resid_beta_mm": float(resid * 1e3),
            "rate_strobo": rate_strobo,
            "per_turn_rad": per_turn,
            "windows_ms": WINDOWS_MS,
            "win_slopes": win_slopes,
            "resid_rms": float(resid_h.std()),
            "f_peak_hz": f_peak,
            "f_rev_hz": f_rev,
        })

        # ── Paneller ────────────────────────────────────────────────────────
        ax = axes[row, 0]
        ax.plot(ts_p * 1e3, sy_p, ".", ms=2)
        ax.plot(ts_p * 1e3, np.polyval(np.polyfit(ts_p, sy_p, 1), ts_p),
                "r-", lw=1, label=f"fit {np.polyfit(ts_p, sy_p, 1)[0]:.2e} rad/s")
        ax.set_xlabel("t [ms]"); ax.set_ylabel("S_y (Poincaré)")
        ax.set_title(f"σ={sigma_m*1e6:.0f} μm — stroboskopik S_y")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[row, 1]
        ax.plot(WINDOWS_MS, np.abs(win_slopes), "o-")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("fit penceresi [ms]"); ax.set_ylabel("|eğim| [rad/s]")
        ax.set_title("pencere bağımlılığı (düşüş → artefakt)")
        ax.grid(alpha=0.3, which="both")

        ax = axes[row, 2]
        m = freqs < 5 * f_rev
        ax.semilogy(freqs[m] / 1e3, np.abs(fft[m]) + 1e-30)
        ax.axvline(f_rev / 1e3, color="r", ls="--", lw=1, label="f_rev")
        ax.set_xlabel("f [kHz]"); ax.set_ylabel("|FFT(artık)|")
        ax.set_title(f"artık spektrum (tepe {f_peak/1e3:.1f} kHz)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    with open("test_sy_diagnostics.json", "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nKaydedildi: test_sy_diagnostics.json")

    plt.tight_layout()
    plt.savefig("test_sy_diagnostics.png", dpi=150)
    print("Figür: test_sy_diagnostics.png")
    print(f"Toplam süre: {(time.time()-t_wall)/60:.1f} dk")


if __name__ == "__main__":
    main()
