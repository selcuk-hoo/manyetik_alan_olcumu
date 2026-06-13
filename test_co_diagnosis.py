#!/usr/bin/env python3
"""test_co_diagnosis.py — Kapalı yörünge (CO) tanı testi.

SORU (Omarov PRD 105, 032001 ile çelişki):
  Omarov simetrik-hibrit halkada quad kaçıklığının BİRİNCİ MERTEBE sahte
  EDM üretmediğini gösterir: düz-kafes testiyle "random misalignments of
  quads alone do not cause vertical spin build-up", görülen etki σ²
  (geometrik faz). 10 μm rms → tek ışın ~1e-5 rad/s.
  Bizim test_b / test_orbit serisi ise 10 μm → ~1e-3 rad/s ve c_k DOĞRUSAL.

HİPOTEZ:
  Bizim büyük doğrusal sinyal, parçacığın kapalı yörünge ÜZERİNDE değil
  EKSENDEN (CO=False) fırlatılmasından kaynaklanan betatron sızmasıdır.
  measure_dSy_dt yorumu sızma tabanını ~5e-9 rad/tur verir; T_rev≈4.4μs
  ile bu ~1.1e-3 rad/s eder — gözlenen değerle aynı mertebede.

YÖNTEM:
  Tek desen (k=2 modu, domine eden mod), sabit 10 μm genlik. İki koşul:
    (A) CO=False — eksenden fırlatma (test_b/test_orbit yöntemi)
    (B) CO=True  — find_closed_orbit ile kaymış yörüngeye oturtma
  Her ikisinde t2 ∈ {0.5, 1, 2, 4} ms taraması.

  Tanı imzası:
    CO=False → |f| büyük (~1e-3), t2'ye duyarlı  → ARTEFAKT
    CO=True  → |f| küçük, t2'den bağımsız         → gerçek seküler

Çıktı: test_co_diagnosis.json, test_co_diagnosis.png, konsol tablosu
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

from false_edm_mode_scan import _run_one_k

K_MODE   = 2          # domine eden mod (kazanç G_2≈24)
AMP      = 1e-5       # 10 μm cos katsayısı
T2_LIST  = [5e-4, 1e-3, 2e-3, 4e-3]   # izleme süreleri [s]
RET_STEP = 5000
DT       = 1e-11
CO_TURNS = 60


def main():
    t0 = time.time()
    ctx = mp.get_context("spawn")

    # Görev listesi: (k, amp, t2, return_steps, dt, do_co, co_turns)
    tasks = []
    meta  = []
    for do_co in (False, True):
        for t2 in T2_LIST:
            tasks.append((K_MODE, AMP, t2, RET_STEP, DT, do_co, CO_TURNS))
            meta.append((do_co, t2))

    print("=" * 72)
    print("  KAPALI YÖRÜNGE TANI TESTİ — k=2 modu, 10 μm")
    print(f"  {len(tasks)} koşum (2 CO durumu × {len(T2_LIST)} t2)")
    print("=" * 72)

    nw = min(mp.cpu_count(), len(tasks))
    with ctx.Pool(processes=nw) as pool:
        results = pool.map(_run_one_k, tasks)

    # Sonuçları düzenle
    rows = []
    for (do_co, t2), r in zip(meta, results):
        rows.append({
            "do_co": do_co,
            "t2": t2,
            "n_turns": t2 / (4.4e-6),     # yaklaşık tur sayısı
            "dSy_dt": r["dSy_dt"],        # stroboskopik eğim
            "dSy_dt_sg": r["dSy_dt_sg"],  # eski sürekli-SG eğim
            "co_off_mm": r["co_off_mm"],
            "resid_beta_mm": r["resid_beta_mm"],
        })

    # ── Konsol tablosu ──────────────────────────────────────────────────────
    print(f"\n{'CO':>6} {'t2[ms]':>7} {'dSy/dt(strobe)':>16} "
          f"{'dSy/dt(SG)':>14} {'CO_ofset[mm]':>13} {'kalan_beta[mm]':>15}")
    print("-" * 76)
    for row in rows:
        co = "True" if row["do_co"] else "False"
        print(f"{co:>6} {row['t2']*1e3:>7.1f} "
              f"{row['dSy_dt']:>16.3e} {row['dSy_dt_sg']:>14.3e} "
              f"{row['co_off_mm']:>13.4f} {row['resid_beta_mm']:>15.4e}")

    # ── Tanı analizi ────────────────────────────────────────────────────────
    cof = [r for r in rows if not r["do_co"]]
    cot = [r for r in rows if r["do_co"]]
    f_cof = np.array([abs(r["dSy_dt"]) for r in cof])
    f_cot = np.array([abs(r["dSy_dt"]) for r in cot])

    # t2 duyarlılığı: maks/min oranı (1'e yakın → t2'den bağımsız → seküler)
    span_cof = float(f_cof.max() / max(f_cof.min(), 1e-30))
    span_cot = float(f_cot.max() / max(f_cot.min(), 1e-30))
    ratio    = float(np.median(f_cof) / max(np.median(f_cot), 1e-30))

    print(f"\n{'─'*72}")
    print("TANI:")
    print(f"  CO=False medyan |f| = {np.median(f_cof):.3e} rad/s  "
          f"(t2 değişim aralığı {span_cof:.1f}×)")
    print(f"  CO=True  medyan |f| = {np.median(f_cot):.3e} rad/s  "
          f"(t2 değişim aralığı {span_cot:.1f}×)")
    print(f"  CO=False / CO=True oranı = {ratio:.1f}×")
    if ratio > 10 and span_cot < 3:
        print("  → CO=False sinyali ARTEFAKT (betatron sızması).")
        print("    Gerçek seküler false-EDM CO=True değeri kadar küçük.")
    elif ratio < 3:
        print("  → İki yöntem uyumlu; büyük sinyal CO artefaktı DEĞİL.")
    else:
        print("  → Belirsiz; t2 aralığını veya tur sayısını artır.")

    out = {"_aciklama": "Kapalı yörünge tanı testi: CO=False vs CO=True, "
                        "k=2 modu 10 μm, t2 taraması",
           "K_MODE": K_MODE, "AMP_um": AMP * 1e6,
           "satirlar": rows,
           "tani": {"co_false_medyan": float(np.median(f_cof)),
                    "co_true_medyan": float(np.median(f_cot)),
                    "oran": ratio,
                    "co_false_t2_span": span_cof,
                    "co_true_t2_span": span_cot}}
    with open("test_co_diagnosis.json", "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nKaydedildi: test_co_diagnosis.json")

    # ── Figür ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    t2ms = np.array(T2_LIST) * 1e3

    ax[0].loglog(t2ms, f_cof, "o-", label="CO=False (eksenden)", ms=8)
    ax[0].loglog(t2ms, f_cot, "s-", label="CO=True (kapalı yörünge)", ms=8)
    ax[0].axhline(1e-5, color="r", ls="--", lw=1,
                  label="Omarov 10 μm tek-ışın ~1e-5")
    ax[0].set_xlabel("izleme süresi t2 [ms]")
    ax[0].set_ylabel("|dSy/dt| [rad/s]")
    ax[0].set_title(f"k={K_MODE} modu, 10 μm — CO durumu ve t2 bağımlılığı")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    # Sağ panel: stroboskopik vs SG (yöntem tutarlılığı)
    ax[1].loglog(f_cof, [abs(r["dSy_dt_sg"]) for r in cof], "o",
                 label="CO=False", ms=8)
    ax[1].loglog(f_cot, [abs(r["dSy_dt_sg"]) for r in cot], "s",
                 label="CO=True", ms=8)
    lims = [min(f_cof.min(), f_cot.min()), max(f_cof.max(), f_cot.max())]
    ax[1].plot(lims, lims, "k:", lw=1, label="y=x")
    ax[1].set_xlabel("stroboskopik dSy/dt [rad/s]")
    ax[1].set_ylabel("sürekli-SG dSy/dt [rad/s]")
    ax[1].set_title("Eğim ölçüm yöntemi tutarlılığı")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("test_co_diagnosis.png", dpi=150)
    print("Figür: test_co_diagnosis.png")
    print(f"Toplam süre: {(time.time()-t0)/60:.1f} dk")


if __name__ == "__main__":
    main()
