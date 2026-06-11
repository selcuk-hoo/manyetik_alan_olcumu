#!/usr/bin/env python3
"""test_b_trim_launch_dep.py — c_k'nin fırlatma koşuluna bağımlılığı.

Soru:
  Trim kalibrasyonu eksen fırlatmasında (y=py=0) yapılıyor.
  Gerçek demetteki parçacıklar farklı başlangıç koşullarında (y0, py0).
  c_k bu fırlatma koşuluna bağlı mı?
  Eksen-kalibre trim, gerçek demet ortalamasında artık bırakır mı?

Fizik beklentisi (lineer lattisin öngörüsü):
  Lineer lattiste betatron hareketi misalignment-kaynaklı kapalı yörünge
  etrafında periyodik salınım yapar. Poincaré ortalama ⟨Δy⟩ = kapalı yörünge,
  betatron genliğinden bağımsız. False-EDM = c_k·a_k yalnızca kapalı yörüngeden
  gelir → c_k fırlatma koşulundan BAĞIMSIZ olmalı.

  Eğer bu doğruysa: demet ortalaması dSy/dt ≡ eksen dSy/dt → trim tam çalışır.

Plan:
  Bölüm 1: Başlangıç konumu taraması
    k=2, A=10μm; y0 = 0..2mm taranır; c_k(y0)/c_k(0) grafiği.
  Bölüm 2: Başlangıç açısı taraması
    k=2, A=10μm; py0 (açısal eşdeğeri) taranır.
  Bölüm 3: Demet ortalaması
    N=30 parçacık Gauss(0, σ_beam) dağılımından çekilir;
    eksen-kalibre trimle bastırma ölçülür; demet ortalamasında artık hesaplanır.

Tüm koşullar CO=False — kapalı yörünge aranmaz.
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

# ── Parametreler ─────────────────────────────────────────────────────────────
A_MODE       = 1e-5      # referans mod genliği [m] = 10 μm
T2           = 1e-3      # entegrasyon süresi [s]
RETURN_STEPS = 6000
K_CALIB      = 2         # kalibrasyon modu

# Konum taraması: y0 = 0 .. 2 mm
Y_OFFSETS    = [0.0, 0.1e-3, 0.2e-3, 0.5e-3, 1.0e-3, 2.0e-3]   # [m]
# Açı taraması: eşdeğer py0/pz0 = 0 .. 1 mrad
PY_FRACS     = [0.0, 0.1e-3, 0.2e-3, 0.5e-3, 1.0e-3]            # boyutsuz α=py/pz

# Demet benzetimi
N_BEAM       = 30        # parçacık sayısı
SIGMA_Y      = 0.5e-3    # konum sigma [m] = 0.5 mm
SIGMA_PY     = 0.2e-3    # açısal sigma [rad]
BEAM_SEED    = 77


def _suppress_stdout():
    fd = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.close(null)
    return fd


def _restore_stdout(fd):
    os.dup2(fd, 1)
    os.close(fd)


def _worker(task):
    """CO=False parçacık izleme → dSy/dt eğim katsayısı.

    Görev: (label, dy_list, y_offset, py_frac, t2, return_steps)
      y_offset [m]: başlangıç dikey konumu (ideal = 0)
      py_frac [-]:  başlangıç açısı py0/pz0 (= sinα ≈ α [rad])
    """
    label, dy_list, y_offset, py_frac, t2, return_steps = task

    import os, sys, json, numpy as np
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from false_edm_mode_scan import setup_fields
    from integrator import integrate_particle

    with open("params.json") as fh:
        config = json.load(fh)
    fields, y0_base, beta0, R0, p_mag, direction = setup_fields(config)
    dt = float(config.get("dt", 1e-11))
    dy = np.asarray(dy_list, dtype=float)

    # Başlangıç koşulunu değiştir: dikey konum ve açı
    y0 = list(y0_base)
    y0[2] = y_offset                      # dikey konum [m]
    y0[3] = py_frac * abs(p_mag)          # dikey momentum = α * pz0

    saved = _suppress_stdout()
    try:
        fields.poincare_quad_index = 0.0
        _, poin, poin_t = integrate_particle(
            y0, 0.0, t2, dt,
            fields=fields, return_steps=return_steps, quad_dy=dy)
    finally:
        _restore_stdout(saved)

    slope = float(np.polyfit(np.asarray(poin_t, float),
                             np.asarray(poin[:, 7], float), 1)[0])
    return label, slope


def mode_vec(n_q, k, amp, antisym):
    """Tek FODO Fourier modu (cos fazı) → quad dy vektörü [m]."""
    Fk, _ = fodo_basis(n_q, [k], antisym)
    return Fk[:, 0] * amp


def run_pool(ctx, tasks, nw):
    with ctx.Pool(processes=min(nw, len(tasks))) as pool:
        return dict(pool.map(_worker, tasks))


def main():
    t0 = time.time()
    with open("params.json") as fh:
        config = json.load(fh)
    from false_edm_mode_scan import setup_fields
    fields, *_ = setup_fields(config)
    n_q     = 2 * int(fields.nFODO)
    antisym = config.get("smooth_antisym_fodo", True)
    ctx     = mp.get_context("spawn")
    nw      = mp.cpu_count()

    # Kalibrasyon modu vektörü (eksen fırlatması için referans)
    v_k2 = mode_vec(n_q, K_CALIB, A_MODE, antisym)

    # ══ BÖLÜM 1: Konum taraması ═══════════════════════════════════════════════
    print(f"\n{'═'*65}")
    print("BÖLÜM 1: Başlangıç konumu taraması (k=2, A=10μm, py0=0)")
    print(f"{'═'*65}")

    tasks1 = []
    for y0 in Y_OFFSETS:
        tasks1.append((f"y_{y0:.1e}", v_k2.tolist(), y0, 0.0, T2, RETURN_STEPS))
        tasks1.append((f"y0_{y0:.1e}", np.zeros(n_q).tolist(), y0, 0.0, T2, RETURN_STEPS))

    print(f"  {len(tasks1)} simülasyon ({nw} işçi)...")
    res1 = run_pool(ctx, tasks1, nw)

    ck_y = {}
    f_bos_y = {}
    for y0 in Y_OFFSETS:
        f_sig = res1[f"y_{y0:.1e}"]
        f_bos = res1[f"y0_{y0:.1e}"]
        ck_y[y0] = (f_sig - f_bos) / A_MODE   # taban çıkar
        f_bos_y[y0] = f_bos

    ck_ref = ck_y[Y_OFFSETS[0]]   # eksen kalibrasyonu

    print(f"\n  {'y0 [μm]':>10}  {'c_k [rad/s/m]':>15}  {'c_k/c_k(0)':>12}  {'taban f_bos':>14}")
    print(f"  {'─'*60}")
    for y0 in Y_OFFSETS:
        ratio = ck_y[y0] / ck_ref if ck_ref != 0 else float('nan')
        print(f"  {y0*1e6:>10.0f}  {ck_y[y0]:>15.4e}  {ratio:>12.5f}  {f_bos_y[y0]:>14.3e}")

    spread_y = max(abs(ck_y[y0]/ck_ref - 1) for y0 in Y_OFFSETS) * 100
    print(f"\n  → Maks sapma: %{spread_y:.3f}  (lineer limit: 0)")

    # ══ BÖLÜM 2: Açı taraması ═════════════════════════════════════════════════
    print(f"\n{'═'*65}")
    print("BÖLÜM 2: Başlangıç açısı taraması (k=2, A=10μm, y0=0)")
    print(f"{'═'*65}")

    tasks2 = []
    for py in PY_FRACS:
        tasks2.append((f"py_{py:.1e}", v_k2.tolist(), 0.0, py, T2, RETURN_STEPS))
        tasks2.append((f"py0_{py:.1e}", np.zeros(n_q).tolist(), 0.0, py, T2, RETURN_STEPS))

    print(f"  {len(tasks2)} simülasyon ({nw} işçi)...")
    res2 = run_pool(ctx, tasks2, nw)

    ck_py = {}
    for py in PY_FRACS:
        f_sig = res2[f"py_{py:.1e}"]
        f_bos = res2[f"py0_{py:.1e}"]
        ck_py[py] = (f_sig - f_bos) / A_MODE

    print(f"\n  {'α=py/pz [mrad]':>14}  {'c_k [rad/s/m]':>15}  {'c_k/c_k(0)':>12}")
    print(f"  {'─'*50}")
    for py in PY_FRACS:
        ratio = ck_py[py] / ck_ref if ck_ref != 0 else float('nan')
        print(f"  {py*1e3:>14.1f}  {ck_py[py]:>15.4e}  {ratio:>12.5f}")

    spread_py = max(abs(ck_py[py]/ck_ref - 1) for py in PY_FRACS) * 100
    print(f"\n  → Maks sapma: %{spread_py:.3f}")

    # ══ BÖLÜM 3: Demet ortalaması benzetimi ══════════════════════════════════
    print(f"\n{'═'*65}")
    print(f"BÖLÜM 3: Demet ortalaması (N={N_BEAM} parçacık, "
          f"σ_y={SIGMA_Y*1e3:.1f}mm, σ_α={SIGMA_PY*1e3:.1f}mrad)")
    print(f"{'═'*65}")

    rng = np.random.default_rng(BEAM_SEED)
    beam_y  = rng.normal(0, SIGMA_Y,  N_BEAM)
    beam_py = rng.normal(0, SIGMA_PY, N_BEAM)

    tasks3 = []
    # Moda sahip lattis: k=2, A=10μm
    for i, (yb, pyb) in enumerate(zip(beam_y, beam_py)):
        tasks3.append((f"bm_{i}", v_k2.tolist(), yb, pyb, T2, RETURN_STEPS))
    # Eksen fırlatması referansı (trim kalibrasyonu)
    tasks3.append(("bm_eksen", v_k2.tolist(), 0.0, 0.0, T2, RETURN_STEPS))

    print(f"  {len(tasks3)} simülasyon ({nw} işçi)...")
    res3 = run_pool(ctx, tasks3, nw)

    f_eksen = res3["bm_eksen"]
    c_eksen = f_eksen / A_MODE   # eksen-kalibre c_k

    beam_f = np.array([res3[f"bm_{i}"] for i in range(N_BEAM)])
    beam_mean_f = np.mean(beam_f)
    beam_std_f  = np.std(beam_f)

    # Trim miktarı (eksen kalibrasyonundan)
    A_trim = -f_eksen / c_eksen   # = -A_MODE (tautoloji ama konsept açık)

    # Trim sonrası artık: eksen ölçümünü sıfırladık, demet ortalamasında artık?
    # Gerçekte: trim k=2 moduyla f_eksen'i sıfırlıyor → eşdeğer olarak
    # lattis misalignment = v_k2*A_trim eklendi.
    # Trim sonrası her parçacığın f'i = f_i - f_eksen (çünkü trim lineer)
    beam_f_posttrim = beam_f - f_eksen     # eksen-kalibre trim sonrası
    beam_mean_posttrim = np.mean(beam_f_posttrim)
    beam_rms_posttrim  = np.sqrt(np.mean(beam_f_posttrim**2))

    # Bastırma: |demet ortalaması ön| / |demet ortalaması son|
    suppression = abs(beam_mean_f) / abs(beam_mean_posttrim) if beam_mean_posttrim != 0 else float('inf')

    print(f"\n  Eksen fırlatması dSy/dt     : {f_eksen:+.4e} rad/s")
    print(f"  Demet ortalaması dSy/dt (ön): {beam_mean_f:+.4e} rad/s")
    print(f"  Demet std dSy/dt (ön)        : {beam_std_f:.4e} rad/s")
    print(f"  Eksen-kalibre trim miktarı  : {A_trim*1e6:.3f} μm (k=2)")
    print(f"\n  Trim sonrası (eksen ↦ sıfır):")
    print(f"  Demet ortalaması artık      : {beam_mean_posttrim:+.4e} rad/s")
    print(f"  Demet RMS artık             : {beam_rms_posttrim:.4e} rad/s")
    print(f"  Bastırma oranı (ortalama)   : {suppression:.1f}×")
    print(f"  Artık / eksen sinyali       : {abs(beam_mean_posttrim/f_eksen)*100:.4f}%")

    # Parçacık başına c_k dağılımı
    ck_beam = beam_f / A_MODE
    ck_mean = np.mean(ck_beam)
    ck_std  = np.std(ck_beam)
    ck_rel  = ck_std / abs(ck_ref) * 100

    print(f"\n  Parçacık başına c_k dağılımı:")
    print(f"  c_k ortalaması : {ck_mean:+.4e} rad/s/m")
    print(f"  c_k std        : {ck_std:.4e} rad/s/m")
    print(f"  c_k std / c_k  : %{ck_rel:.3f}")
    print(f"  Eksen c_k      : {ck_ref:+.4e} rad/s/m")

    # JSON özeti
    sonuc = {
        "_aciklama": "c_k fırlatma koşulu bağımlılık testi (CO=False)",
        "ck_ref_axis": float(ck_ref),
        "konum_taramasi": {str(int(y0*1e6)): float(ck_y[y0]) for y0 in Y_OFFSETS},
        "aci_taramasi":   {str(py*1e3): float(ck_py[py]) for py in PY_FRACS},
        "sapma_konum_max_yuzde": float(spread_y),
        "sapma_aci_max_yuzde":   float(spread_py),
        "demet_N": N_BEAM,
        "sigma_y_m": SIGMA_Y,
        "sigma_py_rad": SIGMA_PY,
        "demet_ck_std_rel_yuzde": float(ck_rel),
        "demet_ortalama_artik_oran": float(abs(beam_mean_posttrim/f_eksen)*100),
        "bastirma": float(suppression),
    }
    with open("test_b_launch_dep.json", "w") as fh:
        json.dump(sonuc, fh, indent=2)
    print("\nKaydedildi: test_b_launch_dep.json")

    elapsed = time.time() - t0
    print(f"Toplam süre: {elapsed:.1f} s")

    # ══ Figür ═════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("c_k fırlatma koşuluna bağımlılığı (CO=False, k=2, A=10μm)",
                 fontsize=12)

    # Panel 1: Konum taraması
    ax = axes[0]
    ratios_y = [ck_y[y0]/ck_ref for y0 in Y_OFFSETS]
    ax.plot([y0*1e3 for y0 in Y_OFFSETS], ratios_y,
            'o-', color='tab:blue', ms=9, lw=2)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.7, label='ideal (1.000)')
    ax.set_xlabel("Başlangıç konumu y₀ [mm]")
    ax.set_ylabel("c_k(y₀) / c_k(0)")
    ax.set_title(f"Konum taraması\nmaks sapma: %{spread_y:.3f}")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    ax.set_ylim(0.98, 1.02)

    # Panel 2: Açı taraması
    ax = axes[1]
    ratios_py = [ck_py[py]/ck_ref for py in PY_FRACS]
    ax.plot([py*1e3 for py in PY_FRACS], ratios_py,
            's-', color='tab:orange', ms=9, lw=2)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.7, label='ideal (1.000)')
    ax.set_xlabel("Başlangıç açısı α = py/pz [mrad]")
    ax.set_ylabel("c_k(α) / c_k(0)")
    ax.set_title(f"Açı taraması\nmaks sapma: %{spread_py:.3f}")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    ax.set_ylim(0.98, 1.02)

    # Panel 3: Demet c_k dağılımı
    ax = axes[2]
    sorted_y = sorted(zip(beam_y, ck_beam))
    ys = [s[0]*1e3 for s in sorted_y]
    cks = [s[1] for s in sorted_y]
    ax.scatter(ys, cks, s=60, alpha=0.7, color='tab:purple', label='demet parçacıkları')
    ax.axhline(ck_ref, color='tab:red', lw=2, ls='-', label=f'eksen c_k ({ck_ref:.3e})')
    ax.axhline(ck_mean, color='tab:blue', lw=1.5, ls='--',
               label=f'demet ort. ({ck_mean:.3e})')
    ax.set_xlabel("Başlangıç konumu y₀ [mm]")
    ax.set_ylabel("c_k [rad/s/m]")
    ax.set_title(f"Demet c_k dağılımı (N={N_BEAM})\n"
                 f"std/c_k = %{ck_rel:.3f},  artık/sinyal = %{abs(beam_mean_posttrim/f_eksen)*100:.3f}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "test_b_launch_dep.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()
