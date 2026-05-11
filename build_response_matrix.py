#!/usr/bin/env python3
"""
build_response_matrix.py

K-modülasyon için iki optik konfigürasyonda tepki matrislerini hesaplar:

  R_dy [48×48] : quad_dy (dikey hizalama hatası) → y_COD [m/m]
  R_dx [48×48] : quad_dx (radyal hizalama hatası) → x_COD [m/m]

Dipol tilt ve quad tilt modelde YOK — bunlar ölçümde gürültü olarak kalır,
tepki matrisi tarafından görülmez.

İki konfigürasyon (g_nom, g_pert = g_nom × 1.02) matrislerinin farkı:
  ΔR_dy = R_dy_2 - R_dy_1   →  test_kmod_reconstruction.py bunu kullanır
  ΔR_dx = R_dx_2 - R_dx_1

• dx ve dy AYRI koşumlarla pertürbe edilir.
• Tüm koşumlar ProcessPoolExecutor ile paralelleştirilir.
  Her worker kendi geçici dizinine chdir eder (cod_data.txt çakışmasını önler).

Komut satırı:
  python build_response_matrix.py --workers 7
"""
import argparse
import atexit
import json
import numpy as np
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from integrator import integrate_particle, FieldParams

BASE = os.path.dirname(os.path.abspath(__file__))


def setup_fields(config, g1_override=None, g0_override=None):
    """FieldParams ve başlangıç koşullarını oluşturur."""
    M2  = 0.938272046
    AMU = 1.792847356
    C   = 299792458.0
    M1  = 1.672621777e-27

    p_magic = M2 / np.sqrt(AMU)
    beta0   = p_magic / np.sqrt(p_magic**2 + M2**2)
    gamma0  = 1.0 / np.sqrt(1.0 - beta0**2)
    R0      = config["R0"]
    E0_V_m  = -(p_magic * beta0 / R0) * 1e9

    direction = float(config.get("direction", -1))
    p_mag = gamma0 * M1 * C * beta0

    g1 = g1_override if g1_override is not None else config.get("g1", 0.0)

    alanlar = FieldParams()
    alanlar.R0          = R0
    alanlar.E0          = E0_V_m
    alanlar.E0_power    = config.get("E0_power", 1.0)
    alanlar.B0ver       = config.get("B0ver", 0.0)
    alanlar.B0rad       = config.get("B0rad", 0.0)
    alanlar.B0long      = config.get("B0long", 0.0)
    g0 = g0_override if g0_override is not None else config.get("g0", g1)
    alanlar.quadG1      = g1
    alanlar.quadG0      = g0
    alanlar.sextK1      = config.get("sextK1", 0.0)
    alanlar.quadSwitch  = float(config.get("quadSwitch", 1))
    alanlar.sextSwitch  = float(config.get("sextSwitch", 0))
    alanlar.EDMSwitch   = 0.0
    alanlar.direction   = direction
    alanlar.nFODO       = float(config.get("nFODO", 24))
    alanlar.quadLen     = float(config.get("quadLen", 0.4))
    alanlar.driftLen    = float(config.get("driftLen", 2.0))
    alanlar.poincare_quad_index = 999.0
    alanlar.rfSwitch    = 0.0
    alanlar.rfVoltage   = 0.0
    alanlar.h           = float(config.get("h", 1.0))
    alanlar.quadModA    = 0.0
    alanlar.quadModF    = 0.0

    state0 = [
        0.0, 0.0, 0.0,
        0.0, 0.0, p_mag * direction,
        0.0, 0.0, direction,
    ]
    return alanlar, state0


def read_cod_quads(nFODO):
    """QF/QD giriş noktalarında x ve y COD. Dosya yolu CWD-relative."""
    cd = np.loadtxt("cod_data.txt", skiprows=1)
    n = int(nFODO)
    x_bpm = np.empty(2 * n)
    y_bpm = np.empty(2 * n)
    for k in range(n):
        qf = k * 8 + 2
        qd = k * 8 + 6
        x_bpm[2*k]     = cd[qf, 1]
        y_bpm[2*k]     = cd[qf, 2]
        x_bpm[2*k + 1] = cd[qd, 1]
        y_bpm[2*k + 1] = cd[qd, 2]
    return x_bpm, y_bpm


def run_sim(alanlar, state0, config, quad_dy, quad_dx,
            dipole_tilt=None, quad_tilt=None):
    """Tek koşum, BPM COD'larını döndürür. Çıktı dosyaları CWD'ye yazılır."""
    for fname in ("cod_data.txt", "rf.txt"):
        if os.path.exists(fname):
            os.remove(fname)
    n_q = 2 * int(alanlar.nFODO)
    if dipole_tilt is None:
        dipole_tilt = np.zeros(n_q)
    if quad_tilt is None:
        quad_tilt = np.zeros(n_q)
    integrate_particle(
        state0,
        t0=0.0,
        t_end=config.get("t2", 1e-3),
        h=config.get("dt", 1e-11),
        fields=alanlar,
        return_steps=10,
        quad_dy=quad_dy,
        quad_dx=quad_dx,
        dipole_tilt=dipole_tilt,
        quad_tilt=quad_tilt,
    )
    return read_cod_quads(int(alanlar.nFODO))


# ── Paralelleştirme yardımcıları ─────────────────────────────────────────
def _worker_init():
    """Her worker süreci kendi geçici dizinine chdir eder."""
    tmp = tempfile.mkdtemp(prefix=f"kmod_w{os.getpid()}_")
    os.chdir(tmp)
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)


def _run_one(task):
    """Worker giriş noktası: setup + run_sim → COD."""
    config, g1_override, g0_override, kind, idx, dy_arr, dx_arr, tilt_arr = task
    alanlar, state0 = setup_fields(config, g1_override=g1_override,
                                   g0_override=g0_override)
    x_cod, y_cod = run_sim(alanlar, state0, config, dy_arr, dx_arr,
                           dipole_tilt=tilt_arr)
    return kind, idx, x_cod, y_cod


def build_matrices(config, g1_override=None, g0_override=None,
                   delta_q=1e-4, label="",
                   sigma_noise=0.0, noise_seed=77,
                   n_workers=1):
    """R_dy ve R_dx matrislerini hesaplar (paralel).

    Tepki matrisi yalnızca quad hizalama hatalarını modeller.
    Dipol tilt tepki matrisi kasıtlı olarak hesaplanmaz.
    """
    n_q = 2 * int(config.get("nFODO", 24))
    zeros = np.zeros(n_q)

    # 1 referans + n_q dy + n_q dx = 2*n_q + 1 koşum
    tasks = []
    tasks.append((config, g1_override, g0_override,
                  'ref', 0, zeros, zeros, zeros))
    for i in range(n_q):
        dy = zeros.copy(); dy[i] = delta_q
        tasks.append((config, g1_override, g0_override,
                      'dy', i, dy, zeros, zeros))
    for i in range(n_q):
        dx = zeros.copy(); dx[i] = delta_q
        tasks.append((config, g1_override, g0_override,
                      'dx', i, zeros, dx, zeros))

    n_total = len(tasks)
    if label:
        print(f"  [{label}] {n_total} kosum, n_workers={n_workers}")
    t0 = time.time()

    results = {}
    progress_step = max(1, n_total // 10)

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_worker_init) as pool:
            futures = [pool.submit(_run_one, t) for t in tasks]
            for j, fut in enumerate(as_completed(futures), 1):
                kind, idx, x_cod, y_cod = fut.result()
                results[(kind, idx)] = (x_cod, y_cod)
                if label and (j % progress_step == 0 or j == n_total):
                    el = time.time() - t0
                    rem = el / j * (n_total - j)
                    print(f"  [{label}] {j}/{n_total}  "
                          f"({el:.0f}s gecti, ~{rem:.0f}s kaldi)")
    else:
        for j, t in enumerate(tasks, 1):
            kind, idx, x_cod, y_cod = _run_one(t)
            results[(kind, idx)] = (x_cod, y_cod)
            if label and (j % progress_step == 0 or j == n_total):
                el = time.time() - t0
                rem = el / j * (n_total - j)
                print(f"  [{label}] {j}/{n_total}  "
                      f"({el:.0f}s gecti, ~{rem:.0f}s kaldi)")

    x0_clean, y0_clean = results[('ref', 0)]
    rng = np.random.default_rng(noise_seed) if sigma_noise > 0 else None

    def _noisy(arr):
        return arr + rng.normal(0, sigma_noise, len(arr)) if rng is not None else arr

    x0 = _noisy(x0_clean)
    y0 = _noisy(y0_clean)

    R_dy = np.zeros((n_q, n_q))
    R_dx = np.zeros((n_q, n_q))

    for i in range(n_q):
        _, y_cod = results[('dy', i)]
        R_dy[:, i] = (_noisy(y_cod) - y0) / delta_q
    for i in range(n_q):
        x_cod, _ = results[('dx', i)]
        R_dx[:, i] = (_noisy(x_cod) - x0) / delta_q

    if label:
        print(f"  [{label}] tamamlandi ({time.time()-t0:.1f}s)")
    return R_dy, R_dx


def main():
    default_workers = max(1, (os.cpu_count() or 2) - 1)
    parser = argparse.ArgumentParser(description="Tepki matrisi insasi - paralel.")
    parser.add_argument("--workers", "-w", type=int, default=default_workers,
                        help=f"Paralel worker sayisi (default: cekirdek-1 = {default_workers})")
    args = parser.parse_args()

    os.chdir(BASE)
    with open("params.json") as f:
        config = json.load(f)

    n_q     = 2 * int(config.get("nFODO", 24))
    delta_q = 1e-4
    g1_nom  = config.get("g1", 0.21)
    eps     = 0.02
    g1_pert = g1_nom * (1.0 + eps)

    sigma_noise = config.get("bpm_noise_sigma", 0.0)

    print("=" * 60)
    print(f"Konfigurasyon 1: nominal optik  (g1={g1_nom}, n_workers={args.workers})")
    print("=" * 60)
    print(f"  n_quad={n_q},  delta_q={delta_q*1e3:.2f} mm")
    if sigma_noise > 0:
        print(f"  BPM gurultusu (R insasinda): sigma={sigma_noise*1e6:.1f} um")
    print()

    t_total = time.time()

    R_dy_1, R_dx_1 = build_matrices(
        config, g1_override=g1_nom,
        delta_q=delta_q, label="nom",
        sigma_noise=sigma_noise, n_workers=args.workers,
    )

    np.save("R_dy_1.npy", R_dy_1)
    np.save("R_dx_1.npy", R_dx_1)

    print(f"\n  [nom] kappa(R_dy) = {np.linalg.cond(R_dy_1):.3e}")
    print(f"  [nom] kappa(R_dx) = {np.linalg.cond(R_dx_1):.3e}")

    print()
    print("=" * 60)
    print(f"Konfigurasyon 2: perturbe optik  (g1={g1_pert:.4f}, +{eps*100:.0f}%)")
    print("=" * 60)
    print()

    R_dy_2, R_dx_2 = build_matrices(
        config, g1_override=g1_pert,
        delta_q=delta_q, label="pert",
        sigma_noise=sigma_noise, n_workers=args.workers,
    )

    np.save("R_dy_2.npy", R_dy_2)
    np.save("R_dx_2.npy", R_dx_2)

    dR_dy = R_dy_2 - R_dy_1
    dR_dx = R_dx_2 - R_dx_1
    np.save("dR_dy.npy", dR_dy)
    np.save("dR_dx.npy", dR_dx)

    print(f"\n  [pert] kappa(R_dy) = {np.linalg.cond(R_dy_2):.3e}")
    print(f"  [pert] kappa(R_dx) = {np.linalg.cond(R_dx_2):.3e}")
    print(f"\n  kappa(dR_dy) = {np.linalg.cond(dR_dy):.3e}")
    print(f"  kappa(dR_dx) = {np.linalg.cond(dR_dx):.3e}")

    total_elapsed = time.time() - t_total
    print(f"\nToplam sure: {total_elapsed:.0f}s  (n_workers={args.workers})")


if __name__ == "__main__":
    main()
