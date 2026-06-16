#!/usr/bin/env python3
"""false_edm_correction_test.py — Rastgele kaçıklık → R-LS düzeltme → false EDM.

Amaç (makalenin yeni omurgası):
  "R-LS ile k=1,2,3 harmoniklerini kestirip çıkarmak, false EDM'i bastırıyor mu?"

Akış (her seed için):
  1. Rastgele 48-quad dikey kaçıklık (σ=100 μm).  → false EDM (baseline) ölç
  2. R-LS ile k=1,2,3 FODO-antisim. harmoniklerini orbit'ten kestir.
  3. dy_corrected = dy - Σ_{k=1,2,3} F_k â_k     (ideal düzeltme)
     → false EDM tekrar ölç
  4. baseline vs düzeltme: dS_y/dt düşüşünü raporla.

False EDM metriği baz-bağımsız (spin takibi), dolayısıyla adil.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from false_edm_mode_scan import (setup_fields, find_closed_orbit, _make_state,
                                  measure_dSy_dt_model, C)
from fourier_reconstruct import fodo_basis

R1 = np.load("R_dy_1.npy")


def measure_false_edm(quad_dy, t2=5e-4, return_steps=5000, dt=None, co_turns=60):
    """Verilen kaçıklık vektörü için seküler dS_y/dt [rad/s].

    Birincil ölçüm madde 2 model fitidir (salınım sızıntısı çıkarılmış seküler
    eğim). Düz polyfit eğimi yapay olarak ∝A doğrusal şişer; model fit gerçek
    ∝A² seküler terimi açığa çıkarır."""
    from integrator import integrate_particle
    with open("params.json") as f:
        config = json.load(f)
    fields, y0, beta0, R0, p_mag, direction = setup_fields(config)
    if dt is None:
        dt = float(config.get("dt", 1e-11))

    circ = (2*np.pi*R0 + 4*fields.nFODO*fields.driftLen
            + 2*fields.nFODO*fields.quadLen)
    T_rev = circ / (beta0 * C)

    # Kapalı yörünge fırlatması (betatron salınımını yok et)
    v_co, resid_rms = find_closed_orbit(fields, p_mag, direction, quad_dy,
                                        dt, T_rev, n_turns=co_turns)
    y_launch = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
    co_off_mm = float(np.hypot(v_co[0], v_co[1]) * 1e3)

    # Stroboskopik spin takibi
    fields.poincare_quad_index = 0.0
    hist, poin, poin_t = integrate_particle(
        y_launch, 0.0, t2, dt, fields=fields, return_steps=return_steps,
        quad_dy=quad_dy)
    sy_strobe = np.asarray(poin[:, 7], float)
    ts = np.asarray(poin_t, float)
    slope = float(measure_dSy_dt_model(sy_strobe, ts))   # madde 2: seküler eğim
    return slope, co_off_mm


def _worker(task):
    """Paralel worker: (etiket, quad_dy) → (etiket, dSy_dt, co_off)."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    label, quad_dy, t2, return_steps = task
    t0 = time.time()
    slope, co_off = measure_false_edm(np.asarray(quad_dy), t2=t2,
                                      return_steps=return_steps)
    return label, slope, co_off, time.time() - t0


def rls_reconstruct(y, ks):
    """R-LS: R·F_k·a = y, verilen k'ler için katsayılar (dict k -> [ac,as])."""
    n_q = R1.shape[0]
    F = np.column_stack([fodo_basis(n_q, [k])[0] for k in ks])
    M = R1 @ F
    a, *_ = np.linalg.lstsq(M, y, rcond=None)
    out = {}
    idx = 0
    for k in ks:
        out[k] = a[idx:idx+2]; idx += 2
    return out


def build_correction(coeffs):
    """Kestirilen harmoniklerden kaçıklık katkısı Σ F_k â_k."""
    n_q = R1.shape[0]
    dy = np.zeros(n_q)
    for k, a in coeffs.items():
        Fk, _ = fodo_basis(n_q, [k])
        dy += Fk @ a
    return dy


def main():
    import multiprocessing as mp
    SIGMA_DY = 100e-6
    SIG_KS = [1, 2, 3]
    N_SEEDS = int(os.environ.get("N_SEEDS", "1"))
    T2 = float(os.environ.get("T2", "5e-4"))
    RETURN_STEPS = int(os.environ.get("RETURN_STEPS", "5000"))

    print(f"False-EDM düzeltme testi:  σ_dy={SIGMA_DY*1e6:.0f}μm, "
          f"R-LS k={SIG_KS}, {N_SEEDS} seed, t2={T2*1e3:.1f}ms")

    tasks = []
    meta = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        dy = rng.normal(0, SIGMA_DY, R1.shape[0])
        y = R1 @ dy                                  # gürültüsüz orbit
        coeffs = rls_reconstruct(y, SIG_KS)
        dy_corr = dy - build_correction(coeffs)

        # Kestirilen harmonik genlikleri (rapor için)
        amps = {k: float(np.hypot(*coeffs[k]) * 1e6) for k in SIG_KS}
        meta.append((seed, dy, dy_corr, amps))
        tasks.append((f"s{seed}_base", dy.tolist(), T2, RETURN_STEPS))
        tasks.append((f"s{seed}_corr", dy_corr.tolist(), T2, RETURN_STEPS))

    nproc = min(len(tasks), max(1, mp.cpu_count()))
    print(f"  {len(tasks)} spin-takip koşusu, {nproc} paralel süreç...\n")
    t0 = time.time()
    with mp.Pool(nproc) as pool:
        results = pool.map(_worker, tasks)
    res = {label: (slope, co_off, rt) for label, slope, co_off, rt in results}

    print(f"{'='*70}")
    print(f"  {'seed':>4}  {'k=1,2,3 [μm]':>16}  {'baseline':>12}  "
          f"{'düzeltme':>12}  {'bastırım':>9}")
    print(f"{'='*70}")
    base_all, corr_all = [], []
    for seed, dy, dy_corr, amps in meta:
        sb, _, _ = res[f"s{seed}_base"]
        sc, _, _ = res[f"s{seed}_corr"]
        amp_str = "/".join(f"{amps[k]:.0f}" for k in SIG_KS)
        ratio = abs(sb) / abs(sc) if abs(sc) > 0 else float('inf')
        print(f"  {seed:>4}  {amp_str:>16}  {sb:>+12.3e}  {sc:>+12.3e}  "
              f"{ratio:>7.1f}×")
        base_all.append(abs(sb)); corr_all.append(abs(sc))
    print(f"{'='*70}")
    print(f"  Ortalama |dS_y/dt|:  baseline={np.mean(base_all):.3e}  "
          f"düzeltme={np.mean(corr_all):.3e}  "
          f"bastırım={np.mean(base_all)/np.mean(corr_all):.1f}×")
    print(f"  Toplam duvar-saati: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
