#!/usr/bin/env python3
"""test_quad_dx_effect.py — Test 9: Yatay kuadrupol hizalama hatası (quad_dx).

Sorular:
  1. Yatay hizalama hatası (dx) dikey sahte EDM'ye katkı verir mi?
     Lineer kafeste B_Z = G1*(X−R0−dx) → yatay COD kayar; dikey Sy'ye
     doğrudan katkı beklenmez. Pratik sonuç ne?
  2. dx+dy birlikte uygulandığında (CW,n)−(CCW,f) iptali hâlâ çalışır mı?
  3. dx amplitüdü büyüdükçe ne olur?

Fizik:
  Lineer kafes: dx → yatay COD; Sy'ye lineer katkı yok (x-y kupajsız).
  Dikey kapalı yörünge (y_co, y'_co) yalnızca dy'e bağlı, dx'ten bağımsız.
  dx+dy birlikte: iptal şeması dx'e bağlı değil, kombinasyonda da çalışmalı.

Bölümler:
  A) Saf dx taraması — dy=0, dx değişiyor → dSy/dt ölçülür.
  B) dx+dy kombinasyonu — sabit dy=10μm, dx değişiyor.
  C) CW/CCW+flip iptali — hem dx hem dy ile iptal çalışıyor mu?

Not: find_closed_orbit sadece quad_dy alır. Lineer kafeste dikey CO sadece
     dy'e bağlı → dx sıfır kabul edilerek CO bulunur, simülasyon dx ile koşulur.

Çıktı:
  test_quad_dx_effect.png
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
A_DY       = 1e-5       # sabit dy RMS = 10 μm
T2         = 5e-4
RSTEPS     = 3000
# dx tarama genlikleri (RMS) [μm]
DX_SCAN_UM = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0]


def _suppress():
    fd = os.dup(1); n = os.open(os.devnull, os.O_WRONLY); os.dup2(n, 1); os.close(n); return fd
def _restore(fd):
    os.dup2(fd, 1); os.close(fd)


def _worker(task):
    """quad_dx ve quad_dy ile sahte EDM ölç."""
    label, direction, flip, yv, ypv, dy_list, dx_list, t2, rsteps = task
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
    dx = np.asarray(dx_list, float)
    y_launch = _make_state([0.0, float(yv), 0.0, float(ypv)],
                           p_mag, float(direction), [0.0, 0.0, float(direction)])
    saved = _suppress()
    try:
        fields.poincare_quad_index = 0.0
        _, poin, poin_t = integrate_particle(y_launch, 0.0, t2, dt, fields=fields,
                                              return_steps=rsteps, quad_dy=dy,
                                              quad_dx=dx)
    finally:
        _restore(saved)
    return label, float(np.polyfit(np.asarray(poin_t, float),
                                   np.asarray(poin[:, 7], float), 1)[0])


def _find_co_dy_only(direction, flip, dy, config):
    """Dikey kapalı yörünge: dx lineer kafeste dikey CO'yu etkilemez."""
    from false_edm_mode_scan import setup_fields, find_closed_orbit, C
    fields, _, beta0, R0, p_mag, _ = setup_fields(config)
    if flip:
        fields.quadG1 = -fields.quadG1; fields.quadG0 = -fields.quadG0
    dt = float(config.get("dt", 1e-11))
    circ = (2*np.pi*R0 + 4*fields.nFODO*fields.driftLen + 2*fields.nFODO*fields.quadLen)
    T_rev = circ / (beta0 * C)
    saved = _suppress()
    try:
        v, _ = find_closed_orbit(fields, p_mag, float(direction), dy, dt, T_rev,
                                 n_turns=24, n_iter=2)
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
    dy_ref = rng.standard_normal(n_q) * A_DY          # sabit dy (10μm RMS)

    # Dikey CO sadece dy'e bağlı → dy ile bir kez bul, tüm dx senaryolarında kullan
    print(f"Lattice üniform {fields.quadG1} T/m")
    print("Dikey kapalı yörüngeler bulunuyor (sadece dy ile)...")
    yc_cwn,   ypc_cwn   = _find_co_dy_only(direction,  False, dy_ref, config)
    yc_ccwf,  ypc_ccwf  = _find_co_dy_only(-direction, True,  dy_ref, config)
    yc_cwn0,  ypc_cwn0  = _find_co_dy_only(direction,  False, np.zeros(n_q), config)
    print(f"  CW normal  (dy): y_co={yc_cwn*1e6:+.3f}μm  y'={ypc_cwn*1e6:+.3f}μrad")
    print(f"  CCW flip   (dy): y_co={yc_ccwf*1e6:+.3f}μm  y'={ypc_ccwf*1e6:+.3f}μrad")
    print(f"  CW normal (dy=0): y_co={yc_cwn0*1e6:+.3f}μm  y'={ypc_cwn0*1e6:+.3f}μrad")

    # dx dizileri: A için farklı rastgele seed (dy_ref ile bağımsız)
    rng2 = np.random.default_rng(REF_SEED + 100)
    dx_arrays = {}
    for dx_um in DX_SCAN_UM:
        rng_i = np.random.default_rng(REF_SEED + 100 + int(dx_um*10))
        dx_arrays[dx_um] = rng_i.standard_normal(n_q) * dx_um * 1e-6
    dx_ref = dx_arrays[10.0]    # 10μm dx referansı (C bölümü için)
    dx_zero = np.zeros(n_q)

    tasks = []

    # ── BÖLÜM A: Saf dx (dy=0) ────────────────────────────────────────────
    for dx_um in DX_SCAN_UM:
        tasks.append((f"A_{dx_um}", direction, False,
                      yc_cwn0, ypc_cwn0,
                      np.zeros(n_q).tolist(), dx_arrays[dx_um].tolist(), T2, RSTEPS))

    # ── BÖLÜM B: dy_ref + dx taraması ─────────────────────────────────────
    for dx_um in DX_SCAN_UM:
        tasks.append((f"B_{dx_um}", direction, False,
                      yc_cwn, ypc_cwn,
                      dy_ref.tolist(), dx_arrays[dx_um].tolist(), T2, RSTEPS))

    # ── BÖLÜM C: CW/CCW+flip iptali — dx=0 ve dx=10μm karşılaştırması ────
    # C1: sadece dy
    tasks.append(("C1_cwn",  direction,  False, yc_cwn,  ypc_cwn,
                  dy_ref.tolist(), dx_zero.tolist(), T2, RSTEPS))
    tasks.append(("C1_ccwf", -direction, True,  yc_ccwf, ypc_ccwf,
                  dy_ref.tolist(), dx_zero.tolist(), T2, RSTEPS))
    # C2: dy + dx birlikte
    tasks.append(("C2_cwn",  direction,  False, yc_cwn,  ypc_cwn,
                  dy_ref.tolist(), dx_ref.tolist(), T2, RSTEPS))
    tasks.append(("C2_ccwf", -direction, True,  yc_ccwf, ypc_ccwf,
                  dy_ref.tolist(), dx_ref.tolist(), T2, RSTEPS))

    nw = mp.cpu_count()
    print(f"\n{len(tasks)} simülasyon ({nw} işçi)...")
    with mp.get_context("spawn").Pool(processes=nw) as pool:
        res = dict(pool.map(_worker, tasks))

    # ── Sonuçlar A ────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("BÖLÜM A: Saf dx (dy=0) → dikey sahte EDM")
    print(f"{'─'*62}")
    print(f"{'dx RMS [μm]':>12}  {'dSy/dt [rad/s]':>16}")
    print('─'*32)
    sA = []
    for dx_um in DX_SCAN_UM:
        s = res[f"A_{dx_um}"]
        sA.append(s)
        print(f"{dx_um:>12.1f}  {s:>16.3e}")

    # ── Sonuçlar B ────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("BÖLÜM B: dy_ref(10μm) + dx taraması")
    print(f"{'─'*62}")
    print(f"{'dx RMS [um]':>12}  {'dSy/dt [rad/s]':>16}  {'dx=0dan fark':>14}")
    print('─'*46)
    s_dy_only = res["B_0.0"]
    sB = []
    for dx_um in DX_SCAN_UM:
        s = res[f"B_{dx_um}"]
        sB.append(s)
        print(f"{dx_um:>12.1f}  {s:>16.3e}  {s-s_dy_only:>14.3e}")

    # ── Sonuçlar C ────────────────────────────────────────────────────────
    s1_cwn  = res["C1_cwn"];  s1_ccwf = res["C1_ccwf"]
    s2_cwn  = res["C2_cwn"];  s2_ccwf = res["C2_ccwf"]
    diff1 = s1_cwn - s1_ccwf
    diff2 = s2_cwn - s2_ccwf
    sup1 = abs(s1_cwn / diff1) if diff1 != 0 else float('inf')
    sup2 = abs(s2_cwn / diff2) if diff2 != 0 else float('inf')

    print(f"\n{'─'*68}")
    print("BÖLÜM C: (CW,n) − (CCW,f) iptali")
    print(f"{'─'*68}")
    print(f"  Sadece dy (dx=0):")
    print(f"    CW_n={s1_cwn:+.3e}  CCW_f={s1_ccwf:+.3e}  FARK={diff1:+.3e}  → {sup1:.0f}× bastırma")
    print(f"\n  dx+dy birlikte (dx_RMS=dy_RMS=10μm):")
    print(f"    CW_n={s2_cwn:+.3e}  CCW_f={s2_ccwf:+.3e}  FARK={diff2:+.3e}  → {sup2:.0f}× bastırma")

    elapsed = time.time() - t0
    print(f"\nToplam süre: {elapsed:.1f} s")
    print("\nYORUM:")
    if max(np.abs(sA[1:])) < 1e-12:
        print("  Bölüm A: dx saf dikey sahte EDM üretmiyor → lineer kafes x-y bağımsız (beklenen).")
    else:
        print(f"  Bölüm A: dx sahte EDM üretiyor ({max(np.abs(sA[1:])):.1e}) → lineer olmayan etki var.")
    print("  Bölüm B: dx eklenmesi dy bazlı sahte EDM'yi değiştirmiyorsa → bağımsız kanallar.")
    print("  Bölüm C: iptal oranı dx eklenince değişmiyorsa → şema dx'e de sağlam.")

    # Figür
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    dx_arr_plot = np.array(DX_SCAN_UM)
    dy_ref_val  = abs(s_dy_only)

    ax = axes[0]
    ax.semilogy(dx_arr_plot, np.abs(sA)+1e-18, 'o-', color='tab:purple', ms=7,
                label='saf dx (dy=0)')
    ax.axhline(dy_ref_val+1e-18, color='tab:blue', ls='--', alpha=0.7,
               label=f'dy=10μm referansı\n({dy_ref_val:.1e})')
    ax.set_xlabel("dx RMS [μm]")
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("Bölüm A: Saf dx\n(dy=0 — yatay misalignment dikey EDM'ye katkısı?)")
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    ax.semilogy(dx_arr_plot, np.abs(sB)+1e-18, 's-', color='tab:orange', ms=7,
                label='dy_ref(10μm) + dx')
    ax.axhline(dy_ref_val+1e-18, color='tab:blue', ls='--', alpha=0.7,
               label=f'dx=0 referansı ({dy_ref_val:.1e})')
    ax.set_xlabel("dx RMS [μm]")
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("Bölüm B: dy_ref + dx taraması\n(dx eklenmesi sahte EDM'yi değiştirir mi?)")
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

    ax = axes[2]
    labels = ['CW_n\n(dy)', 'CCW_f\n(dy)', 'FARK\n(dy)',
              'CW_n\n(dx+dy)', 'CCW_f\n(dx+dy)', 'FARK\n(dx+dy)']
    vals   = [abs(s1_cwn), abs(s1_ccwf), abs(diff1)+1e-18,
              abs(s2_cwn), abs(s2_ccwf), abs(diff2)+1e-18]
    colors = ['tab:blue', 'tab:orange', 'tab:green',
              'tab:cyan',  'tab:red',    'tab:purple']
    ax.bar(range(6), vals, color=colors, alpha=0.85)
    ax.set_yscale('log')
    ax.set_xticks(range(6)); ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(1e-9, color='gray', ls='--', alpha=0.7, label='1e-9 hedef')
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("Bölüm C: (CW,n)−(CCW,f) iptali\ndx+dy birlikte")
    ax.legend(fontsize=8); ax.grid(True, axis='y', which='both', alpha=0.3)

    plt.tight_layout()
    out = "test_quad_dx_effect.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()
