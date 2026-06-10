#!/usr/bin/env python3
"""test_kicker_precision.py — Test 6: Injection kicker hassasiyeti (CW/CCW ortak alan).

Soru:
  Injection kick'i fiziksel bir mıknatıs (kicker) ile verilir; her iki karşı-dönen
  demet AYNI alanı görür. Kicker'ın ne kadar hassas ayarlanması gerekir? Ve kicker
  hatası (CW,normal) − (CCW,flip) farkında iptal olur mu, sızar mı?

Fizik:
  Kicker açıyı (p_y) değiştirir. Kapalı yörüngeye oturmak için iki demetin de aynı
  p_y'si gerekir → y'_CW=+y'*, y'_CCW=−y'* (p_z işaret değiştirdiği için mirror).
  Bir kicker hatası δ:
   - Kicker quad-flip ile TERS çevrilirse → mirror hata (CW +δ, CCW −δ) →
     ikisinin p_y'si AYNI kayar → ORTAK MOD → farkta iptal.
   - Kicker ters çevrilmezse → aynı δy' (CW +δ, CCW +δ) → p_y'ler ZIT kayar →
     DİFERANSİYEL → farkta sızar.

Yöntem (üniform 0.2 lattice, aynı misalignment):
  Açı sapması δ taranır (μrad). İki senaryo:
   A) Kicker ters çevrilmiş (mirror δ): CW y'+δ, CCW y'−δ.
   B) Kicker ters çevrilmemiş (ortak δ): CW y'+δ, CCW y'+δ.
  Her senaryo için tek tek sahte EDM ve FARK ölçülür.

Çıktı:
  test_kicker_precision.png
  Terminal tablo + hassasiyet (farkı 1e-9 altında tutan δ)
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
CO_TURNS   = 24
CO_ITER    = 2
RSTEPS     = 3000
DELTA_URAD = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]   # injection açı sapması [μrad]


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


def _find_co(direction, flip, dy_ref, config):
    from false_edm_mode_scan import setup_fields, find_closed_orbit, C
    fields, _, beta0, R0, p_mag, _ = setup_fields(config)
    if flip:
        fields.quadG1 = -fields.quadG1; fields.quadG0 = -fields.quadG0
    dt = float(config.get("dt", 1e-11))
    circ = (2*np.pi*R0 + 4*fields.nFODO*fields.driftLen + 2*fields.nFODO*fields.quadLen)
    T_rev = circ / (beta0 * C)
    saved = _suppress()
    try:
        v, _ = find_closed_orbit(fields, p_mag, float(direction), dy_ref, dt, T_rev,
                                 n_turns=CO_TURNS, n_iter=CO_ITER)
    finally:
        _restore(saved)
    return float(v[1]), float(v[3])


def main():
    t0 = time.time()
    with open("params.json") as fh:
        config = json.load(fh)
    from false_edm_mode_scan import setup_fields
    fields, *_ = setup_fields(config)
    n_q = 2*int(fields.nFODO); direction = fields.direction
    rng = np.random.default_rng(REF_SEED)
    dy_ref = rng.standard_normal(n_q) * A_MISALIGN

    print(f"Lattice üniform {fields.quadG1} T/m. Kapalı yörüngeler bulunuyor...")
    yc_cwn,  ypc_cwn  = _find_co(direction,  False, dy_ref, config)
    yc_ccwf, ypc_ccwf = _find_co(-direction, True,  dy_ref, config)
    print(f"  CW normal : y_co={yc_cwn*1e6:+.3f}μm  y'={ypc_cwn*1e6:+.3f}μrad")
    print(f"  CCW flip  : y_co={yc_ccwf*1e6:+.3f}μm  y'={ypc_ccwf*1e6:+.3f}μrad")

    tasks = []
    for d_ur in DELTA_URAD:
        d = d_ur * 1e-6
        # A) kicker TERS çevrilmiş (mirror): CW +δ, CCW −δ
        tasks.append((f"A_cw_{d_ur}",  direction,  False, yc_cwn,  ypc_cwn + d,
                      dy_ref.tolist(), T2, RSTEPS))
        tasks.append((f"A_ccw_{d_ur}", -direction, True,  yc_ccwf, ypc_ccwf - d,
                      dy_ref.tolist(), T2, RSTEPS))
        # B) kicker çevrilmemiş (ortak δy'): CW +δ, CCW +δ
        tasks.append((f"B_cw_{d_ur}",  direction,  False, yc_cwn,  ypc_cwn + d,
                      dy_ref.tolist(), T2, RSTEPS))
        tasks.append((f"B_ccw_{d_ur}", -direction, True,  yc_ccwf, ypc_ccwf + d,
                      dy_ref.tolist(), T2, RSTEPS))

    nw = min(mp.cpu_count(), len(tasks))
    print(f"\n{len(tasks)} simülasyon ({nw} işçi)...")
    with mp.get_context("spawn").Pool(processes=nw) as pool:
        res = dict(pool.map(_worker, tasks))

    print(f"\n{'─'*78}")
    print("Injection açı sapması δ → (CW,n)−(CCW,f) farkı")
    print(f"{'─'*78}")
    hdr = (f"{'δ [μrad]':>9}  {'A: kicker ters (mirror)':>26}  {'B: kicker düz (ortak)':>26}")
    print(hdr); print('─'*len(hdr))
    print(f"{'':>9}  {'fark [rad/s]':>26}  {'fark [rad/s]':>26}")
    print('─'*len(hdr))
    diffA, diffB = [], []
    for d_ur in DELTA_URAD:
        dA = res[f"A_cw_{d_ur}"] - res[f"A_ccw_{d_ur}"]
        dB = res[f"B_cw_{d_ur}"] - res[f"B_ccw_{d_ur}"]
        diffA.append(dA); diffB.append(dB)
        print(f"{d_ur:>9.2f}  {dA:>26.3e}  {dB:>26.3e}")

    elapsed = time.time() - t0
    print(f"\nToplam süre: {elapsed:.1f} s")
    print("\nYORUM:")
    print("  A (kicker quad ile ters çevrilir): hata ortak mod → fark ~0 kalır → sağlam.")
    print("  B (kicker çevrilmez): hata diferansiyel → fark δ ile büyür → hassasiyet kritik.")
    # B için 1e-9 toleransı (lineer interpolasyon)
    dB_abs = np.abs(diffB)
    if np.any(dB_abs > 1e-9):
        idx = np.argmax(dB_abs > 1e-9)
        if idx > 0:
            d0, d1 = DELTA_URAD[idx-1], DELTA_URAD[idx]
            f0, f1 = dB_abs[idx-1], dB_abs[idx]
            tol = d0 + (1e-9 - f0)*(d1-d0)/(f1-f0) if f1 != f0 else d1
            print(f"  B senaryosunda farkı 1e-9 altında tutmak için δ < ~{tol:.3f} μrad gerekir.")

    # Figür
    fig, ax = plt.subplots(figsize=(8.5, 6))
    dd = np.array(DELTA_URAD)
    ax.semilogy(dd, np.abs(diffA)+1e-18, 'o-', color='tab:green', markersize=8,
                label='A: kicker quad ile TERS çevrilmiş (ortak hata)')
    ax.semilogy(dd, np.abs(diffB)+1e-18, 's-', color='tab:red', markersize=8,
                label='B: kicker çevrilmemiş (diferansiyel hata)')
    ax.axhline(1e-9, color='gray', ls='--', alpha=0.7, label='1e-9 (Omarov hedef)')
    ax.set_xlabel("Injection açı sapması δ [μrad]")
    ax.set_ylabel("|(CW,n) − (CCW,f)| fark [rad/s]")
    ax.set_title("Test 6: Injection kicker hassasiyeti\n"
                 "kicker quad ile ters çevrilirse hata ortak mod → iptal")
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    out = "test_kicker_precision.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()
