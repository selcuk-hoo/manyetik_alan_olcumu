#!/usr/bin/env python3
"""test_quad_flip_symmetry.py — Test 4: CCW + quad-flip simetrisi (Omarov şeması).

Soru:
  Quad-flip (tüm quad gradyanlarının işaretini çevirme) uygulanınca kapalı
  yörüngeye oturtma işi CW ve CCW için aynı anda çalışır mı? Ve bu kombinasyon
  sahte EDM'yi iptal edip gerçek EDM'yi korur mu?

Fizik (Omarov karşı-dönen beam + quad-flip):
  - Sahte EDM: ışın yönünde TEK (odd), quad-flip'te TEK (odd).
    → misalignment × quad gradyanına bağlı.
  - Gerçek EDM: ışın yönünde TEK (odd), quad-flip'te ÇİFT (even).
    → E alanına bağlı, quad polaritesine değil.

  Quad-flip manyetik kick işaretini çevirir; ışın ters dönünce de kick işareti
  çevriliyordu. İkisi birlikte → kick'ler AYNI → kapalı yörünge KONUMU aynı
  (açı ters, çünkü traversal yönü döner). Dolayısıyla (CW,normal) ve (CCW,flip)
  aynı injection konumunu paylaşır.

  İşaret tablosu (F=sahte, D=gerçek):
    (CW, normal): +F, +D
    (CCW, flip) : +F, -D   [odd×odd=+ sahte; odd×even=- gerçek]
  → (CW,normal) - (CCW,flip): sahte EDM İPTAL (F-F=0), gerçek EDM 2D KORUNUR.

Yöntem (lattice üniform 0.2 T/m, tüm quad'lar aynı büyüklük):
  Bölüm A: yeni lattice'te mod başına işaret yapısı (k=1..4) korunuyor mu?
  Bölüm B: 4 konfig (CW/CCW × normal/flip) kapalı yörünge + sahte EDM.
           Oturtma çakışması ve (CW,n)-(CCW,f) iptali gösterilir.

Çıktı:
  test_quad_flip_symmetry.png
  Terminal tablolar
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

from fourier_reconstruct import fodo_basis

# ── Test parametreleri ───────────────────────────────────────────────────────
REF_SEED     = 7        # önceki testlerle aynı referans hizalama
A_MISALIGN   = 1e-5     # hizalama RMS [m] = 10 μm
A_MODE       = 1e-5     # tek-mod genliği (Bölüm A) [m]
T2           = 5e-4
CO_TURNS     = 24
CO_ITER      = 2
RETURN_STEPS = 3000
K_TABLE      = [1, 2, 3, 4]


def _suppress_stdout():
    fd = os.dup(1); null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1); os.close(null); return fd


def _restore_stdout(fd):
    os.dup2(fd, 1); os.close(fd)


def _worker(task):
    """Kapalı yörüngeye otur + sahte EDM (dSy/dt) ölç. Quad-flip opsiyonel.

    Görev: (label, direction, flip, dy_list, t2, co_turns, co_iter, rsteps)
    Dönüş: (label, slope, y_co, yp_co)
    """
    label, direction, flip, dy_list, t2, co_turns, co_iter, rsteps = task

    import os, sys, json, numpy as np
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from false_edm_mode_scan import setup_fields, find_closed_orbit, _make_state, C
    from integrator import integrate_particle

    with open("params.json") as fh:
        config = json.load(fh)
    fields, _, beta0, R0, p_mag, _ = setup_fields(config)

    # QUAD-FLIP: tüm quad gradyanlarının işaretini çevir (g1 ve g0)
    if flip:
        fields.quadG1 = -fields.quadG1
        fields.quadG0 = -fields.quadG0

    dt    = float(config.get("dt", 1e-11))
    circ  = (2*np.pi*R0 + 4*fields.nFODO*fields.driftLen
             + 2*fields.nFODO*fields.quadLen)
    T_rev = circ / (beta0 * C)
    dy    = np.asarray(dy_list, dtype=float)

    saved = _suppress_stdout()
    try:
        v_co, _ = find_closed_orbit(fields, p_mag, float(direction), dy, dt, T_rev,
                                     n_turns=co_turns, n_iter=co_iter)
        y_launch = _make_state(v_co, p_mag, float(direction), [0.0, 0.0, float(direction)])
        fields.poincare_quad_index = 0.0
        _, poin, poin_t = integrate_particle(
            y_launch, 0.0, t2, dt, fields=fields, return_steps=rsteps, quad_dy=dy)
    finally:
        _restore_stdout(saved)

    slope = float(np.polyfit(np.asarray(poin_t, float),
                             np.asarray(poin[:, 7], float), 1)[0])
    return label, slope, float(v_co[1]), float(v_co[3])


def main():
    t0 = time.time()

    with open("params.json") as fh:
        config = json.load(fh)
    from false_edm_mode_scan import setup_fields
    fields, *_ = setup_fields(config)
    n_q     = 2 * int(fields.nFODO)
    antisym = config.get("smooth_antisym_fodo", True)
    direction = fields.direction
    ctx = mp.get_context("spawn"); nw = mp.cpu_count()

    print(f"Lattice: tüm quad'lar {fields.quadG1} T/m (üniform), "
          f"QF +g / QD −g")

    # ══ BÖLÜM A: yeni lattice'te işaret yapısı (k=1..4, CW normal) ════════
    def mode_dy(k):
        Fk, _ = fodo_basis(n_q, [k], antisym)
        return (A_MODE * Fk[:, 0]).tolist()

    tasksA = [(f"k{k}", direction, False, mode_dy(k), T2, CO_TURNS, CO_ITER, RETURN_STEPS)
              for k in K_TABLE]
    print(f"\nBÖLÜM A: işaret yapısı — {len(tasksA)} simülasyon...")
    with ctx.Pool(processes=min(nw, len(tasksA))) as pool:
        resA = {r[0]: r for r in pool.map(_worker, tasksA)}

    print(f"{'─'*52}")
    print("Mod başına sahte EDM (üniform 0.2 lattice, CW normal)")
    print(f"{'─'*52}")
    print(f"{'k':>3}  {'dSy/dt [rad/s]':>16}  {'işaret':>7}")
    print('─'*52)
    for k in K_TABLE:
        sl = resA[f"k{k}"][1]
        print(f"{k:>3}  {sl:>16.3e}  {'+' if sl>0 else '−':>7}")
    k2pos = resA["k2"][1] > 0
    k3neg = resA["k3"][1] < 0
    print(f"\n  k=2 pozitif: {k2pos},  k=3 negatif: {k3neg}  "
          f"→ rezonans işaret yapısı {'KORUNDU' if (k2pos and k3neg) else 'DEĞİŞTİ'}")

    # ══ BÖLÜM B: 4 konfig (CW/CCW × normal/flip) ═════════════════════════
    rng = np.random.default_rng(REF_SEED)
    dy_ref = (rng.standard_normal(n_q) * A_MISALIGN).tolist()

    configs = [
        ("CW_normal",  direction,  False),
        ("CCW_normal", -direction, False),
        ("CW_flip",    direction,  True),
        ("CCW_flip",   -direction, True),
    ]
    tasksB = [(name, d, f, dy_ref, T2, CO_TURNS, CO_ITER, RETURN_STEPS)
              for name, d, f in configs]
    print(f"\nBÖLÜM B: 4 konfig — {len(tasksB)} simülasyon...")
    with ctx.Pool(processes=min(nw, len(tasksB))) as pool:
        resB = {r[0]: r for r in pool.map(_worker, tasksB)}

    print(f"\n{'─'*72}")
    print("4 konfig: kapalı yörünge + sahte EDM (aynı misalignment)")
    print(f"{'─'*72}")
    print(f"{'konfig':>12}  {'y_co [μm]':>11}  {'y_co [μrad]':>12}  {'dSy/dt [rad/s]':>16}")
    print('─'*72)
    for name, _, _ in configs:
        _, sl, yc, ypc = resB[name]
        print(f"{name:>12}  {yc*1e6:>11.3f}  {ypc*1e6:>12.3f}  {sl:>16.3e}")

    # Oturtma çakışması: (CW,normal) ↔ (CCW,flip) konum eşleşmesi
    yc_cwn  = resB["CW_normal"][2];  yc_ccwf = resB["CCW_flip"][2]
    print(f"\n  Oturtma çakışması (konum):")
    print(f"    CW_normal  y_co = {yc_cwn*1e6:+.3f} μm")
    print(f"    CCW_flip   y_co = {yc_ccwf*1e6:+.3f} μm  "
          f"→ fark {abs(yc_cwn-yc_ccwf)*1e9:.1f} nm (≈0 → AYNI injection)")

    # İptal şeması: (CW,n)-(CCW,f) sahte EDM'yi iptal eder, gerçek EDM'yi 2D korur
    s_cwn  = resB["CW_normal"][1]
    s_ccwf = resB["CCW_flip"][1]
    print(f"\n  Sahte EDM iptali (aynı injection orbitini paylaşan çift):")
    print(f"    CW_normal : {s_cwn:+.3e} rad/s  (= +F)")
    print(f"    CCW_flip  : {s_ccwf:+.3e} rad/s  (= +F, işaret "
          f"{'AYNI' if np.sign(s_cwn)==np.sign(s_ccwf) else 'ZIT'})")
    diff = s_cwn - s_ccwf
    supp = abs(s_cwn / diff) if diff != 0 else float('inf')
    print(f"    FARK (CW,n) − (CCW,f) = {diff:+.3e} rad/s  → sahte EDM {supp:.0f}× iptal")
    print(f"    (EDM açık olsaydı bu fark = 2D gerçek EDM'yi verirdi)")

    elapsed = time.time() - t0
    print(f"\nToplam süre: {elapsed:.1f} s")

    # ══ Figür ═════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Test 4: CCW + Quad-Flip Simetrisi (üniform 0.2 lattice)", fontsize=13)

    # Panel 1: kapalı yörünge konumları (oturtma çakışması)
    ax = axes[0]
    names = [c[0] for c in configs]
    ycs   = [resB[n][2]*1e6 for n in names]
    colors = ['tab:blue', 'tab:red', 'tab:cyan', 'tab:orange']
    ax.bar(range(4), ycs, color=colors, alpha=0.85)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(names, rotation=20, fontsize=9)
    ax.set_ylabel("kapalı yörünge y_co [μm]")
    ax.set_title("Oturtma konumu\n(CW,normal)≡(CCW,flip), (CCW,normal)≡(CW,flip)")
    ax.grid(True, axis='y', alpha=0.3)

    # Panel 2: sahte EDM işaretleri
    ax = axes[1]
    sls = [resB[n][1] for n in names]
    ax.bar(range(4), sls, color=colors, alpha=0.85)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(names, rotation=20, fontsize=9)
    ax.set_ylabel("dSy/dt [rad/s]")
    ax.set_title("Sahte EDM işareti\n(CW,n)≈(CCW,f) → farkları sahte EDM'yi iptal eder")
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    out = "test_quad_flip_symmetry.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()
