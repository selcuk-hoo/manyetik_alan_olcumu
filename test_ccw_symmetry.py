#!/usr/bin/env python3
"""test_ccw_symmetry.py — CW/CCW injection kick çakışması (Omarov simetrisi).

Soru:
  Hizalama hatalı halkada CW (saat yönü) parçacığı kapalı yörüngeye oturtan
  injection kick'i, aynı manyetik alandaki CCW (ters yön) parçacığını da
  oturtuyor mu, yoksa uzaklaştırıyor mu?

Fizik (Omarov CCW + quad-flip simetrisi):
  Quad misalignment'tan doğan dikey kick MANYETİKTİR (F = qv×B). Işın yönü
  tersine dönünce v işaret değiştirir → kick işaret değiştirir → kapalı
  yörünge CW ve CCW için FARKLIDIR. Dolayısıyla tek bir injection ayarı
  iki beam'i aynı anda yörüngeye oturtamaz; birini oturturken diğerini
  betatron salınımına sokar. Bu, sahte EDM'nin CW/CCW altında zıt davranıp
  CW+CCW ortalamasında iptal olmasının temelidir.

  KRİTİK: alanlar (E0, quad gradyanı, MİSALIGNMENT) iki yön için BİRE BİR
  AYNI. Sadece başlangıç momentumunun işareti (ışın yönü) çevrilir.
  (C++'ta field_params[11]=direction yalnızca RF'te kullanılır; ışın yönü
  gerçekte _make_state'teki p_mag·direction işaretiyle belirlenir.)

Yöntem:
  1. Aynı dy_ref için CW ve CCW kapalı yörüngeleri bulunur (kick_cw, kick_ccw).
  2. İki kick arasındaki doğru taranır: launch(t) = (1−t)·kick_cw + t·kick_ccw.
     Her t için HEM CW HEM CCW beam'in sahte EDM'si ölçülür.
     t=0 → CW oturur (taban), CCW uzakta;  t=1 → tam tersi.
  3. Tabanlardaki dSy/dt işaretleri karşılaştırılır (CW+CCW iptali için).

Çıktı:
  test_ccw_symmetry.png — sahte EDM vs injection ayarı (CW & CCW eğrileri)
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

# ── Test parametreleri ───────────────────────────────────────────────────────
REF_SEED     = 7        # CW testiyle AYNI referans hizalama deseni
A_MISALIGN   = 1e-5     # hizalama RMS [m] = 10 μm
T2           = 5e-4     # simülasyon süresi [s]
CO_TURNS     = 24       # kapalı yörünge bulma tur sayısı
CO_ITER      = 2        # Newton yinelemesi (orbit bulma, hassas)
RETURN_STEPS = 3000
N_TSCAN      = 7        # kick_cw → kick_ccw doğrusu üzerinde tarama noktası


# ── stdout bastırma ──────────────────────────────────────────────────────────

def _suppress_stdout():
    fd = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.close(null)
    return fd


def _restore_stdout(fd):
    os.dup2(fd, 1)
    os.close(fd)


# ── Worker: sabit fırlatma noktasından spin takibi ───────────────────────────

def _worker(task):
    """Verilen yön + fırlatma noktasından dSy/dt ölç. Alan = dy_ref (sabit).

    Görev: (label, direction, y_launch, yp_launch, dy_list, t2, return_steps)
    Dönüş: (label, slope [rad/s])
    """
    label, direction, yv, ypv, dy_list, t2, return_steps = task

    import os, sys, json, numpy as np
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from false_edm_mode_scan import setup_fields, _make_state
    from integrator import integrate_particle

    with open("params.json") as fh:
        config = json.load(fh)

    fields, _, beta0, R0, p_mag, _ = setup_fields(config)
    dt = float(config.get("dt", 1e-11))
    dy = np.asarray(dy_list, dtype=float)

    # Işın yönü: başlangıç momentumu + spin işareti (alan AYNI kalır)
    y_launch = _make_state([0.0, float(yv), 0.0, float(ypv)],
                           p_mag, float(direction),
                           [0.0, 0.0, float(direction)])

    saved = _suppress_stdout()
    try:
        fields.poincare_quad_index = 0.0
        _, poin, poin_t = integrate_particle(
            y_launch, 0.0, t2, dt,
            fields=fields, return_steps=return_steps, quad_dy=dy,
        )
    finally:
        _restore_stdout(saved)

    slope = float(np.polyfit(np.asarray(poin_t, float),
                             np.asarray(poin[:, 7], float), 1)[0])
    return label, slope


# ── Ana rutin ────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    from false_edm_mode_scan import setup_fields, find_closed_orbit, C

    with open("params.json") as fh:
        config = json.load(fh)
    fields, _, beta0, R0, p_mag, direction = setup_fields(config)
    n_q   = 2 * int(fields.nFODO)
    dt    = float(config.get("dt", 1e-11))
    circ  = (2*np.pi*R0 + 4*fields.nFODO*fields.driftLen
             + 2*fields.nFODO*fields.quadLen)
    T_rev = circ / (beta0 * C)

    dir_cw  = direction        # mevcut yön (CW kabul)
    dir_ccw = -direction       # ters yön (CCW)

    # AYNI misalignment her iki yön için (CW testiyle aynı tohum)
    rng    = np.random.default_rng(REF_SEED)
    dy_ref = rng.standard_normal(n_q) * A_MISALIGN

    # ── ADIM 1: CW ve CCW kapalı yörüngeleri (aynı alan) ──────────────────
    print("Aynı misalignment (RMS=10μm) için CW ve CCW kapalı yörüngeleri:")
    saved = _suppress_stdout()
    try:
        vco_cw, res_cw = find_closed_orbit(fields, p_mag, dir_cw, dy_ref,
                                           dt, T_rev, n_turns=CO_TURNS, n_iter=CO_ITER)
        vco_ccw, res_ccw = find_closed_orbit(fields, p_mag, dir_ccw, dy_ref,
                                             dt, T_rev, n_turns=CO_TURNS, n_iter=CO_ITER)
    finally:
        _restore_stdout(saved)

    kick_cw  = np.array([vco_cw[1],  vco_cw[3]])    # (y, y')
    kick_ccw = np.array([vco_ccw[1], vco_ccw[3]])
    print(f"  kick_cw  (CW oturtan) : y={kick_cw[0]*1e6:+7.2f} μm, "
          f"y'={kick_cw[1]*1e6:+6.2f} μrad")
    print(f"  kick_ccw (CCW oturtan): y={kick_ccw[0]*1e6:+7.2f} μm, "
          f"y'={kick_ccw[1]*1e6:+6.2f} μrad")
    sep_um = np.hypot((kick_cw[0]-kick_ccw[0]),
                      (kick_cw[1]-kick_ccw[1])*0) * 1e6
    print(f"  → iki kick arası konum farkı: {sep_um:.2f} μm")

    # ── ADIM 2: kick_cw → kick_ccw doğrusu taraması ───────────────────────
    ts    = np.linspace(0.0, 1.0, N_TSCAN)
    tasks = []
    for i, t in enumerate(ts):
        launch = (1 - t) * kick_cw + t * kick_ccw
        tasks.append((f"cw_{i}",  dir_cw,  launch[0], launch[1],
                      dy_ref.tolist(), T2, RETURN_STEPS))
        tasks.append((f"ccw_{i}", dir_ccw, launch[0], launch[1],
                      dy_ref.tolist(), T2, RETURN_STEPS))

    n_workers = min(mp.cpu_count(), len(tasks))
    print(f"\nADIM 2: {len(tasks)} simülasyon ({n_workers} işçi)...")
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        res = dict(pool.map(_worker, tasks))

    slope_cw  = np.array([res[f"cw_{i}"]  for i in range(N_TSCAN)])
    slope_ccw = np.array([res[f"ccw_{i}"] for i in range(N_TSCAN)])

    # ── Tablo ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("Injection ayarı taraması (t=0: kick_cw,  t=1: kick_ccw)")
    print(f"{'─'*72}")
    hdr = f"{'t':>5}  {'launch y [μm]':>14}  {'CW |dSy/dt|':>14}  {'CCW |dSy/dt|':>14}"
    print(hdr); print('─'*len(hdr))
    for i, t in enumerate(ts):
        ly = ((1-t)*kick_cw[0] + t*kick_ccw[0]) * 1e6
        print(f"{t:>5.2f}  {ly:>14.2f}  {abs(slope_cw[i]):>14.3e}  "
              f"{abs(slope_ccw[i]):>14.3e}")

    # Tabanlar ve işaretler
    s_cw_floor  = slope_cw[0]     # t=0, CW kendi kick'inde
    s_ccw_floor = slope_ccw[-1]   # t=1, CCW kendi kick'inde
    s_cw_displaced  = slope_cw[-1]   # CW, CCW'nin kick'inde (uzakta)
    s_ccw_displaced = slope_ccw[0]   # CCW, CW'nin kick'inde (uzakta)

    print(f"\n{'─'*72}")
    print("Çakışma: birini oturturken diğeri uzaklaşıyor mu?")
    print(f"{'─'*72}")
    print(f"  CW  @ kick_cw  (oturmuş)  : {s_cw_floor:+.3e} rad/s")
    print(f"  CCW @ kick_cw  (uzakta)   : {s_ccw_displaced:+.3e} rad/s  "
          f"→ {abs(s_ccw_displaced/s_ccw_floor):.0f}× taban")
    print(f"  CCW @ kick_ccw (oturmuş)  : {s_ccw_floor:+.3e} rad/s")
    print(f"  CW  @ kick_ccw (uzakta)   : {s_cw_displaced:+.3e} rad/s  "
          f"→ {abs(s_cw_displaced/s_cw_floor):.0f}× taban")

    print(f"\n  Taban işaretleri (CW+CCW iptali için):")
    print(f"    CW  tabanı: {s_cw_floor:+.3e}")
    print(f"    CCW tabanı: {s_ccw_floor:+.3e}")
    same = (np.sign(s_cw_floor) == np.sign(s_ccw_floor))
    avg = 0.5*(s_cw_floor + s_ccw_floor)
    print(f"    işaretler {'AYNI' if same else 'ZIT'} → "
          f"CW+CCW ortalaması = {avg:+.3e} rad/s")

    elapsed = time.time() - t0
    print(f"\nToplam süre: {elapsed:.1f} s")

    # ── Figür ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    launch_y_um = np.array([((1-t)*kick_cw[0] + t*kick_ccw[0])*1e6 for t in ts])

    ax.semilogy(launch_y_um, np.abs(slope_cw), 'o-', color='tab:blue',
                markersize=8, label='CW beam')
    ax.semilogy(launch_y_um, np.abs(slope_ccw), 's-', color='tab:red',
                markersize=8, label='CCW beam')

    ax.axvline(kick_cw[0]*1e6, color='tab:blue', ls=':', alpha=0.7,
               label=f"kick_cw  (y={kick_cw[0]*1e6:+.1f}μm)")
    ax.axvline(kick_ccw[0]*1e6, color='tab:red', ls=':', alpha=0.7,
               label=f"kick_ccw (y={kick_ccw[0]*1e6:+.1f}μm)")

    ax.set_xlabel("Injection fırlatma konumu y [μm]")
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("CW/CCW injection kick çakışması\n"
                 "(aynı misalignment; bir beam'i oturtan ayar diğerini uzaklaştırır)")
    ax.legend(fontsize=9, loc='upper center')
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    out = "test_ccw_symmetry.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()
