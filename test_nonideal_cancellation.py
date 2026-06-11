#!/usr/bin/env python3
"""test_nonideal_cancellation.py — Test 5: İdeal olmayan parçacık için
CW/CCW+flip iptali hâlâ çalışıyor mu?

Soru:
  (CW normal) − (CCW flip) farkı, parçacık kapalı yörünge ÜZERİNDEYKEN sahte
  EDM'yi 1e-16'ya iptal ediyordu. Peki parçacık COD'nin üzerinde DEĞİLSE
  (injection hatası δy → betatron salınımı, tek tek sahte EDM ~1e-5..1e-6)
  bu fark hâlâ iptal ediyor mu, yoksa betatron kirlenmesi farkta kalıyor mu?

  Bu, Omarov'un 1e-9 tabanının "mükemmel oturtma" gerektirip gerektirmediğini
  ve yöntemin gerçekçi parçacıkta ne kadar sağlam olduğunu söyler.

Fizik:
  (CW normal) ve (CCW flip) her fiziksel noktada AYNI kick'i görür → kapalı
  yörünge konumu aynı, dinamik birbirinin aynası. Eğer betatron-kaynaklı sahte
  EDM de bu simetri altında AYNI işaretliyse, fark onu da iptal eder (yöntem
  sağlam). Zıt işaretliyse fark onu iki katına çıkarır (yöntem oturtmaya bağlı).

Yöntem (üniform 0.2 lattice):
  1. CW normal ve CCW flip için kapalı yörünge bulunur (konumları çakışır).
  2. δy ∈ {0, 0.5, 1, 2, 5, 10} μm: her iki beam KENDİ COD'sinden δy kadar
     sapacak şekilde fırlatılır (aynı betatron genliği).
  3. Tek tek sahte EDM'ler ve FARK ölçülür.
  4. Fark δy ile büyüyor mu (iptal bozuluyor) yoksa küçük mü kalıyor (sağlam)?

Çıktı:
  test_nonideal_cancellation.png
  Terminal tablo
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

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Parametreler ─────────────────────────────────────────────────────────────
REF_SEED     = 7
A_MISALIGN   = 1e-5
T2           = 5e-4
CO_TURNS     = 24
CO_ITER      = 2
RETURN_STEPS = 3000
DELTA_UM     = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]   # COD'den injection sapması [μm]


def _suppress_stdout():
    fd = os.dup(1); null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1); os.close(null); return fd


def _restore_stdout(fd):
    os.dup2(fd, 1); os.close(fd)


def _worker(task):
    """Verilen yön/flip/fırlatma için sahte EDM ölç (CO ARAMAZ — sabit launch).

    Görev: (label, direction, flip, yv, ypv, dy_list, t2, rsteps)
    Dönüş: (label, slope)
    """
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
        fields.quadG1 = -fields.quadG1
        fields.quadG0 = -fields.quadG0
    dt = float(config.get("dt", 1e-11))
    dy = np.asarray(dy_list, dtype=float)

    y_launch = _make_state([0.0, float(yv), 0.0, float(ypv)],
                           p_mag, float(direction), [0.0, 0.0, float(direction)])
    saved = _suppress_stdout()
    try:
        fields.poincare_quad_index = 0.0
        _, poin, poin_t = integrate_particle(
            y_launch, 0.0, t2, dt, fields=fields, return_steps=rsteps, quad_dy=dy)
    finally:
        _restore_stdout(saved)
    slope = float(np.polyfit(np.asarray(poin_t, float),
                             np.asarray(poin[:, 7], float), 1)[0])
    return label, slope


def _find_co(direction, flip, dy_ref, config):
    """Verilen yön/flip için kapalı yörünge fırlatma noktası."""
    from false_edm_mode_scan import setup_fields, find_closed_orbit, C
    fields, _, beta0, R0, p_mag, _ = setup_fields(config)
    if flip:
        fields.quadG1 = -fields.quadG1
        fields.quadG0 = -fields.quadG0
    dt    = float(config.get("dt", 1e-11))
    circ  = (2*np.pi*R0 + 4*fields.nFODO*fields.driftLen
             + 2*fields.nFODO*fields.quadLen)
    T_rev = circ / (beta0 * C)
    saved = _suppress_stdout()
    try:
        v_co, _ = find_closed_orbit(fields, p_mag, float(direction), dy_ref, dt,
                                     T_rev, n_turns=CO_TURNS, n_iter=CO_ITER)
    finally:
        _restore_stdout(saved)
    return float(v_co[1]), float(v_co[3])


def main():
    t0 = time.time()
    with open("params.json") as fh:
        config = json.load(fh)
    from false_edm_mode_scan import setup_fields
    fields, *_ = setup_fields(config)
    n_q = 2 * int(fields.nFODO)
    direction = fields.direction

    rng = np.random.default_rng(REF_SEED)
    dy_ref = rng.standard_normal(n_q) * A_MISALIGN

    print(f"Lattice: üniform {fields.quadG1} T/m")
    print("CW normal ve CCW flip kapalı yörüngeleri bulunuyor...")
    yc_cwn,  ypc_cwn  = _find_co(direction,  False, dy_ref, config)
    yc_ccwf, ypc_ccwf = _find_co(-direction, True,  dy_ref, config)
    print(f"  CW normal : y_co={yc_cwn*1e6:+.3f} μm, y'={ypc_cwn*1e6:+.3f} μrad")
    print(f"  CCW flip  : y_co={yc_ccwf*1e6:+.3f} μm, y'={ypc_ccwf*1e6:+.3f} μrad")

    # Görevler: her δy için her iki beam, KENDİ COD'sinden δy sapma
    tasks = []
    for d_um in DELTA_UM:
        d = d_um * 1e-6
        tasks.append((f"cwn_{d_um}",  direction,  False,
                      yc_cwn + d,  ypc_cwn,  dy_ref.tolist(), T2, RETURN_STEPS))
        tasks.append((f"ccwf_{d_um}", -direction, True,
                      yc_ccwf + d, ypc_ccwf, dy_ref.tolist(), T2, RETURN_STEPS))

    nw = min(mp.cpu_count(), len(tasks))
    print(f"\n{len(tasks)} simülasyon ({nw} işçi)...")
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=nw) as pool:
        res = dict(pool.map(_worker, tasks))

    # Sonuçlar
    print(f"\n{'─'*78}")
    print("İdeal olmayan parçacık: COD'den δy sapma → tek tek sahte EDM ve FARK")
    print(f"{'─'*78}")
    hdr = (f"{'δy [μm]':>9}  {'CW normal':>14}  {'CCW flip':>14}  "
           f"{'FARK (CW−CCWf)':>16}  {'iptal':>9}")
    print(hdr); print('─'*len(hdr))
    cwn_arr, ccwf_arr, diff_arr = [], [], []
    for d_um in DELTA_UM:
        scwn  = res[f"cwn_{d_um}"]
        sccwf = res[f"ccwf_{d_um}"]
        diff  = scwn - sccwf
        ratio = abs(scwn / diff) if diff != 0 else float('inf')
        cwn_arr.append(scwn); ccwf_arr.append(sccwf); diff_arr.append(diff)
        print(f"{d_um:>9.1f}  {scwn:>14.3e}  {sccwf:>14.3e}  {diff:>16.3e}  "
              f"{ratio:>8.0f}×")

    elapsed = time.time() - t0
    print(f"\nToplam süre: {elapsed:.1f} s")
    print("\nYORUM:")
    print("  FARK küçük kalıyorsa → simetri betatron'u da iptal eder (yöntem sağlam).")
    print("  FARK tek tek değerlerle birlikte büyüyorsa → iptal oturtmaya bağlı.")

    # Figür
    fig, ax = plt.subplots(figsize=(8.5, 6))
    dd = np.array(DELTA_UM)
    ax.semilogy(dd, np.abs(cwn_arr),  'o-', color='tab:blue',
                label='CW normal (tek tek)', markersize=7)
    ax.semilogy(dd, np.abs(ccwf_arr), 's-', color='tab:orange',
                label='CCW flip (tek tek)', markersize=7)
    ax.semilogy(dd, np.abs(diff_arr), 'D-', color='tab:green', linewidth=2,
                label='FARK (CW − CCW flip)', markersize=8)
    ax.set_xlabel("COD'den injection sapması δy [μm]")
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("Test 5: İdeal olmayan parçacıkta CW/CCW+flip iptali\n"
                 "(fark küçük kalırsa simetri betatron'u da temizler)")
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    out = "test_nonideal_cancellation.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()
