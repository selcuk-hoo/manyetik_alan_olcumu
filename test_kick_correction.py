#!/usr/bin/env python3
"""test_kick_correction.py — Test 1: Injection kick'i ve tolerans taraması.

Soru:
  Bir parçacığı kapalı yörüngeye oturtan injection "kick"ini (fırlatma
  koşulunu) belirle. Ardından parçacık bu ideal kick'ten saptığında
  ("ideal olmayan parçacıklar") false EDM nasıl bozulur?

Fizik:
  Hizalama hatalı halkada kapalı yörünge y=0'dan kayar (COD). Parçacık
  ideal y=0'dan fırlatılırsa kapalı yörünge etrafında betatron salınımı
  yapar → S_y ölçümü kirlenir (CO=False rejimi, ~10⁻⁴ rad/s). Parçacık
  tam kapalı yörünge ÜZERİNDE fırlatılırsa betatron yok → gerçek (küçük)
  false EDM ölçülür (CO=True rejimi, ~10⁻⁹ rad/s).

  "İdeal kick" = parçacığı kapalı yörüngeye oturtan injection koşulu
  (y_co, y'_co). Bu, injection noktasında konum + açı olmak üzere 2 serbestlik
  derecesi gerektirir → tek noktada 2-DOF "kick".

Test akışı:
  1. Referans hizalama deseni için kapalı yörünge fırlatma koşulu bulunur
     (find_closed_orbit, Newton). Bu "ideal kick"tir → nerede/nasıl.
  2. İdeal parçacık (tam kick) → dSy/dt taban değeri ölçülür.
  3. CO=False (kick yok, y=0 fırlatma) → kontrast (~10⁵× büyük).
  4. İdeal olmayan parçacıklar: kick'ten δy kadar sapan fırlatma
     (injection jitter). Sapma RMS'i taranır, false EDM dağılımı ölçülür.
  5. Sonuç: injection toleransı — false EDM'yi kabul edilebilir tutmak için
     gereken fırlatma hassasiyeti.

Çıktı:
  test_kick_correction.png — |dSy/dt| vs injection sapması
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

# ── Test parametreleri ───────────────────────────────────────────────────────
REF_SEED       = 7        # referans hizalama deseni tohumu (tekrarlanabilir)
A_MISALIGN     = 1e-5     # referans hizalama RMS [m] = 10 μm
T2             = 5e-4     # simülasyon süresi [s]
CO_TURNS       = 24       # referans kapalı yörünge bulma tur sayısı
CO_ITER        = 2        # Newton yinelemesi (referans, bir kez → daha hassas)
RETURN_STEPS   = 3000     # Poincaré kayıt kapasitesi
JITTER_UM      = [1, 3, 10, 30, 100]   # injection sapma seviyeleri [μm]
M_JITTER       = 6        # her seviye için rastgele realizasyon sayısı
JITTER_SEED    = 123      # jitter realizasyon tohumu


# ── Paralel worker ───────────────────────────────────────────────────────────

def _suppress_stdout():
    """C++ verbose çıktısını /dev/null'a yönlendirir."""
    fd = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.close(null)
    return fd


def _restore_stdout(fd):
    os.dup2(fd, 1)
    os.close(fd)


def _worker(task):
    """Verilen fırlatma koşulundan spin takibi → dSy/dt ölçümü.

    Görev demeti: (label, y_launch_phase, dy_ref_list, t2, return_steps)
      y_launch_phase : [y0, y'0] dikey faz-uzayı fırlatma noktası [m, rad]
      dy_ref_list    : referans halka quad dikey hataları [m], boy n_q
    Dönüş: (label, slope [rad/s])

    NOT: kapalı yörünge BU worker'da aranmaz — fırlatma noktası dışarıdan
    (ideal kick ± jitter) verilir. Betatron kirlenmesi kasıtlı olarak ölçülür.
    """
    label, y_launch_phase, dy_ref_list, t2, return_steps = task

    import os, sys, json, numpy as np
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from false_edm_mode_scan import setup_fields, _make_state, C
    from integrator import integrate_particle

    with open("params.json") as fh:
        config = json.load(fh)

    fields, _, beta0, R0, p_mag, direction = setup_fields(config)
    dt = float(config.get("dt", 1e-11))
    dy = np.asarray(dy_ref_list, dtype=float)

    yv, ypv  = float(y_launch_phase[0]), float(y_launch_phase[1])
    y_launch = _make_state([0.0, yv, 0.0, ypv], p_mag, direction,
                           [0.0, 0.0, direction])

    saved = _suppress_stdout()
    try:
        fields.poincare_quad_index = 0.0   # tur-başına stroboskopik örnekleme
        _, poin, poin_t = integrate_particle(
            y_launch, 0.0, t2, dt,
            fields=fields,
            return_steps=return_steps,
            quad_dy=dy,
        )
    finally:
        _restore_stdout(saved)

    sy    = np.asarray(poin[:, 7], float)
    ts    = np.asarray(poin_t,     float)
    slope = float(np.polyfit(ts, sy, 1)[0])
    return label, slope


# ── Ana rutin ────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from false_edm_mode_scan import (setup_fields, find_closed_orbit,
                                       _make_state, C)

    with open("params.json") as fh:
        config = json.load(fh)

    fields, _, beta0, R0, p_mag, direction = setup_fields(config)
    n_q   = 2 * int(fields.nFODO)
    dt    = float(config.get("dt", 1e-11))
    circ  = (2*np.pi*R0 + 4*fields.nFODO*fields.driftLen
             + 2*fields.nFODO*fields.quadLen)
    T_rev = circ / (beta0 * C)

    # ── Referans hizalama deseni (gerçekçi: rastgele, RMS=10μm) ───────────
    rng_ref = np.random.default_rng(REF_SEED)
    dy_ref  = rng_ref.standard_normal(n_q) * A_MISALIGN

    # ── ADIM 1: İdeal kick — kapalı yörüngeye oturtan fırlatma koşulu ──────
    print("Referans halka: rastgele hizalama, RMS = 10 μm")
    print("Kapalı yörünge fırlatma koşulu (ideal kick) hesaplanıyor...")
    saved = _suppress_stdout()
    try:
        v_co, resid = find_closed_orbit(fields, p_mag, direction, dy_ref,
                                        dt, T_rev, n_turns=CO_TURNS, n_iter=CO_ITER)
    finally:
        _restore_stdout(saved)

    y_co, yp_co = float(v_co[1]), float(v_co[3])
    co_off_um   = np.hypot(y_co, yp_co * 0) * 1e6   # konum bileşeni [μm]
    print(f"  İdeal kick → injection noktasında:")
    print(f"    konum  y_co  = {y_co*1e6:+.2f} μm")
    print(f"    açı    y'_co = {yp_co*1e6:+.2f} μrad")
    print(f"    kalan betatron RMS = {resid*1e6:.2f} μm  (≈0 → oturtma başarılı)")
    print(f"  → 'nerede': injection azimutunda;  'nasıl': 2-DOF (konum+açı)")

    # ── Görev listesi ─────────────────────────────────────────────────────
    tasks = []
    # İdeal parçacık (tam kick)
    tasks.append(("ideal", [y_co, yp_co], dy_ref.tolist(), T2, RETURN_STEPS))
    # CO=False (kick yok): tasarım y=0'dan fırlatma
    tasks.append(("cofalse", [0.0, 0.0], dy_ref.tolist(), T2, RETURN_STEPS))

    # İdeal olmayan parçacıklar: ideal kick ± jitter (konum sapması)
    rng_jit = np.random.default_rng(JITTER_SEED)
    for lvl_um in JITTER_UM:
        for m in range(M_JITTER):
            dyj = rng_jit.standard_normal() * lvl_um * 1e-6   # [m]
            tasks.append((
                f"jit_{lvl_um}_{m}",
                [y_co + dyj, yp_co],          # konum sapması, açı korunur
                dy_ref.tolist(), T2, RETURN_STEPS,
            ))

    n_total   = len(tasks)
    n_workers = min(mp.cpu_count(), n_total)
    print(f"\nToplam {n_total} simülasyon, {n_workers} işçi ile başlatılıyor...")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_worker, tasks)

    elapsed = time.time() - t0
    res_map = {lbl: sl for lbl, sl in results}

    # ── Sonuçlar ──────────────────────────────────────────────────────────
    s_ideal   = res_map["ideal"]
    s_cofalse = res_map["cofalse"]

    print(f"\n{'─'*64}")
    print("ADIM 2-3: İdeal kick vs kick yok (kontrast)")
    print(f"{'─'*64}")
    print(f"  İdeal parçacık (tam kick) : |dSy/dt| = {abs(s_ideal):.3e} rad/s")
    print(f"  Kick yok (CO=False)       : |dSy/dt| = {abs(s_cofalse):.3e} rad/s")
    ratio = abs(s_cofalse / s_ideal) if s_ideal != 0 else float('nan')
    print(f"  Bastırma oranı            : {ratio:.1f}×")

    print(f"\n{'─'*64}")
    print(f"ADIM 4: İdeal olmayan parçacıklar — injection jitter taraması")
    print(f"{'─'*64}")
    hdr = f"{'jitter [μm]':>12}  {'medyan |dSy/dt|':>18}  {'maks |dSy/dt|':>16}"
    print(hdr)
    print('─' * len(hdr))
    jit_by_lvl = {}
    for lvl_um in JITTER_UM:
        arr = np.abs([res_map[f"jit_{lvl_um}_{m}"] for m in range(M_JITTER)])
        jit_by_lvl[lvl_um] = arr
        print(f"{lvl_um:>12d}  {np.median(arr):>18.3e}  {np.max(arr):>16.3e}")

    print(f"\nToplam süre: {elapsed:.1f} s")

    # ── Figür: |dSy/dt| vs injection sapması ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    # jitter dağılımı (her seviye için noktalar + medyan çizgisi)
    med = [np.median(jit_by_lvl[l]) for l in JITTER_UM]
    for lvl_um in JITTER_UM:
        ys = jit_by_lvl[lvl_um]
        ax.scatter([lvl_um]*len(ys), ys, color='steelblue', alpha=0.5, s=30,
                   zorder=3)
    ax.plot(JITTER_UM, med, 'b-o', linewidth=1.5, markersize=8,
            label='medyan |dSy/dt| (ideal olmayan parçacık)', zorder=4)

    # taban (ideal kick) ve tavan (CO=False)
    ax.axhline(abs(s_ideal), color='green', ls='--', alpha=0.8,
               label=f'ideal kick (taban) = {abs(s_ideal):.1e}')
    ax.axhline(abs(s_cofalse), color='red', ls='--', alpha=0.8,
               label=f'kick yok / CO=False = {abs(s_cofalse):.1e}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("İdeal kick'ten injection sapması [μm]")
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("Test 1: Injection kick toleransı\n"
                 "(referans: rastgele hizalama RMS=10 μm)")
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    out = "test_kick_correction.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()
