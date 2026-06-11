#!/usr/bin/env python3
"""test_realistic_amplitude.py — Test 7: Gerçekçi (büyük) betatron'da CW/CCW+flip iptali.

Sorular (kullanıcıdan):
  1. "Mutlak kicker değeri önemsizse, kicker=0'da (demetler COD'den çok uzakta,
     eksende) bile (CW,n)−(CCW,f) iptal etmeli." → mantıksal kontrol.
  2. Gerçekte betatron salınımı ~1 cm mertebesinde, her parçacığın BÜYÜK injection
     açısı var. Kicker sadece küçük bir bias ekliyor. Bu kadar büyük genlikte
     iptal hâlâ çalışır mı?

Fizik:
  (CW normal) ve (CCW flip) tam simetri eşi: CW (y, +y') ile CCW flip (y, −y')
  aynı p_y → birebir aynı sahte EDM → farkta iptal. Bu simetri PERTÜRBATİF
  DEĞİL (Hamiltonyen'in kesin simetrisi), lineer lattis → genlikten BAĞIMSIZ
  geçerli olmalı. Yani 1 cm betatron'da bile iptal beklenir. Pratik sınır:
  iki demet arasındaki DİFERANSİYEL uyumsuzluk (mirror'dan sapma).

Yöntem (üniform 0.2 lattice, aynı misalignment):
  Mirror açı taraması Δ: CW (0, +Δ), CCW flip (0, −Δ). Δ=0 → kicker yok
  (eksende). Δ büyüdükçe betatron ~β·Δ ≈ 1 cm'e çıkar. Tek tek sahte EDM ve
  FARK ölçülür. Ayrıca büyük genlikte küçük diferansiyel uyumsuzluk eklenir.

Çıktı:
  test_realistic_amplitude.png
  Terminal tablo
"""

import json, os, sys, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REF_SEED   = 7
A_MISALIGN = 1e-5
T2         = 5e-4
RSTEPS     = 3000
# Mirror açı taraması [μrad]. β~30m → β·Δ: 300μrad ≈ 9 mm ≈ 1 cm betatron.
DELTA_URAD = [0.0, 1.0, 10.0, 30.0, 100.0, 300.0]
DIFF_TEST  = (100.0, 1.0)   # (büyük mirror Δ, küçük diferansiyel ε) [μrad]


def _suppress():
    fd = os.dup(1); n = os.open(os.devnull, os.O_WRONLY); os.dup2(n, 1); os.close(n); return fd
def _restore(fd):
    os.dup2(fd, 1); os.close(fd)


def _worker(task):
    label, direction, flip, yv, ypv, dy_list, t2, rsteps = task
    import os, sys, json, numpy as np
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from false_edm_mode_scan import setup_fields, _make_state
    from integrator import integrate_particle
    with open("params.json") as fh:
        config = json.load(fh)
    fields, _, beta0, R0, p_mag, _ = setup_fields(config)
    if flip:
        fields.quadG1 = -fields.quadG1; fields.quadG0 = -fields.quadG0
    dt = float(config.get("dt", 1e-11))
    dy = np.asarray(dy_list, float)
    y_launch = _make_state([0.0, float(yv), 0.0, float(ypv)],
                           p_mag, float(direction), [0.0, 0.0, float(direction)])
    saved = _suppress()
    try:
        fields.poincare_quad_index = 0.0
        _, poin, poin_t = integrate_particle(y_launch, 0.0, t2, dt, fields=fields,
                                              return_steps=rsteps, quad_dy=dy)
    finally:
        _restore(saved)
    return label, float(np.polyfit(np.asarray(poin_t, float),
                                   np.asarray(poin[:, 7], float), 1)[0])


def main():
    t0 = time.time()
    with open("params.json") as fh:
        config = json.load(fh)
    from false_edm_mode_scan import setup_fields
    fields, *_ = setup_fields(config)
    n_q = 2*int(fields.nFODO); direction = fields.direction
    beta_est = 30.0  # kabaca dikey beta [m] (β·Δ ile betatron genliği tahmini)
    rng = np.random.default_rng(REF_SEED)
    dy_ref = (rng.standard_normal(n_q) * A_MISALIGN).tolist()

    print(f"Lattice üniform {fields.quadG1} T/m, aynı misalignment (RMS=10μm)")
    print(f"Mirror açı Δ → betatron genliği ≈ β·Δ (β≈{beta_est} m)")

    tasks = []
    for d_ur in DELTA_URAD:
        d = d_ur * 1e-6
        tasks.append((f"cw_{d_ur}",  direction,  False, 0.0, +d, dy_ref, T2, RSTEPS))
        tasks.append((f"ccw_{d_ur}", -direction, True,  0.0, -d, dy_ref, T2, RSTEPS))
    # Büyük genlikte küçük diferansiyel uyumsuzluk
    D, eps = DIFF_TEST
    tasks.append(("dcw",  direction,  False, 0.0, +D*1e-6,        dy_ref, T2, RSTEPS))
    tasks.append(("dccw", -direction, True,  0.0, -(D-eps)*1e-6,  dy_ref, T2, RSTEPS))

    nw = min(mp.cpu_count(), len(tasks))
    print(f"\n{len(tasks)} simülasyon ({nw} işçi)...")
    with mp.get_context("spawn").Pool(processes=nw) as pool:
        res = dict(pool.map(_worker, tasks))

    print(f"\n{'─'*82}")
    print("Mirror açı Δ (kicker=0 → Δ=0): tek tek sahte EDM ve FARK")
    print(f"{'─'*82}")
    hdr = (f"{'Δ [μrad]':>9}  {'betatron≈':>10}  {'CW normal':>13}  {'CCW flip':>13}  "
           f"{'FARK':>13}")
    print(hdr); print('─'*len(hdr))
    cw_a, ccw_a, diff_a = [], [], []
    for d_ur in DELTA_URAD:
        scw = res[f"cw_{d_ur}"]; sccw = res[f"ccw_{d_ur}"]; df = scw - sccw
        cw_a.append(scw); ccw_a.append(sccw); diff_a.append(df)
        amp_mm = beta_est * d_ur*1e-6 * 1e3
        print(f"{d_ur:>9.1f}  {amp_mm:>8.2f}mm  {scw:>13.3e}  {sccw:>13.3e}  {df:>13.3e}")

    df_big = res["dcw"] - res["dccw"]
    print(f"\n  Büyük genlik (Δ={DIFF_TEST[0]}μrad ≈ {beta_est*DIFF_TEST[0]*1e-6*1e3:.0f}mm) + "
          f"diferansiyel ε={DIFF_TEST[1]}μrad uyumsuzluk:")
    print(f"    CW={res['dcw']:.3e}, CCW flip={res['dccw']:.3e}, FARK={df_big:.3e}")
    print(f"    → mükemmel mirror'da fark ~1e-15; ε=1μrad uyumsuzluk farkı "
          f"{abs(df_big):.1e}'e çıkarıyor")

    elapsed = time.time() - t0
    print(f"\nToplam süre: {elapsed:.1f} s")
    print("\nYORUM:")
    print("  Δ=0 (kicker yok): fark iptal ediyorsa → mantık doğru, mutlak değer önemsiz.")
    print("  Δ büyük (1 cm betatron): fark hâlâ küçükse → simetri genlikten bağımsız.")
    print("  Pratik sınır: iki demet arası DİFERANSİYEL uyumsuzluk (mirror'dan sapma).")

    # Figür
    fig, ax = plt.subplots(figsize=(8.5, 6))
    dd = np.array(DELTA_URAD)
    ax.semilogy(dd, np.abs(cw_a)+1e-18,  'o-', color='tab:blue',  ms=7, label='CW normal (tek tek)')
    ax.semilogy(dd, np.abs(ccw_a)+1e-18, 's-', color='tab:orange',ms=7, label='CCW flip (tek tek)')
    ax.semilogy(dd, np.abs(diff_a)+1e-18,'D-', color='tab:green', ms=8, lw=2,
                label='FARK (mükemmel mirror)')
    ax.scatter([DIFF_TEST[0]], [abs(df_big)], color='red', s=120, marker='*', zorder=5,
               label=f'FARK ({DIFF_TEST[1]}μrad diferansiyel uyumsuzluk)')
    ax.axhline(1e-9, color='gray', ls='--', alpha=0.7, label='1e-9 (Omarov hedef)')
    ax.set_xlabel("Mirror injection açısı Δ [μrad]  (β·Δ ≈ betatron genliği)")
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("Test 7: Gerçekçi büyük betatron'da CW/CCW+flip iptali\n"
                 "(mirror simetride iptal genlikten bağımsız; sınır = diferansiyel)")
    ax.legend(fontsize=8.5, loc='center right'); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    out = "test_realistic_amplitude.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()
