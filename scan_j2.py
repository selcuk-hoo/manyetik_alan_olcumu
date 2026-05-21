#!/usr/bin/env python3
"""
scan_j2.py — Adım 6: j1 sabit, j2 taraması

j1 = params.json["kmod_quad1_index"]
j2 ∈ 0..47 (j1 hariç), sadece dy koşumları (κ(dR_dy) için yeterli)

Tüm j2'lerin pertürbasyon koşumları TEK havuzda paralel.
Her j2 için: 1 ref + 48 dy = 49 koşum.

Süre tahmini (10 worker, ~78s CPU/koşum):
  Tam tarama (47 j2): ~2303 koşum → ~5 saat
  Kaba tarama  (12 j2, --step 4): ~588 koşum → ~76 dakika

Ön koşul: R_dy_1.npy mevcut (build_response_matrix.py ile üretilir).

Kullanım:
  python scan_j2.py                          # tüm j2
  python scan_j2.py --step 4                 # her 4'üncü j2 (kaba tarama)
  python scan_j2.py --j2-list 4 8 12 16 20  # belirli j2'ler
"""
import argparse
import json
import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from build_response_matrix import _run_one, _worker_init

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    default_w = max(1, (os.cpu_count() or 2) - 1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", "-w", type=int, default=default_w)
    parser.add_argument("--step", type=int, default=1,
                        help="j2 tarama adımı (1=tam, 2=her ikinci, 4=kaba)")
    parser.add_argument("--j2-list", nargs="+", type=int, default=None,
                        help="Taranacak j2 değerleri (--step'i ezer)")
    args = parser.parse_args()

    os.chdir(BASE)
    with open("params.json") as f:
        config = json.load(f)

    n_q    = 2 * int(config.get("nFODO", 24))
    g0     = config.get("g0", 0.2)
    g1     = config.get("g1", g0)
    g2     = config.get("g2", g0)
    j1     = config.get("kmod_quad1_index", 2)
    delta_q = 1e-4

    R_dy_1 = np.load("R_dy_1.npy")
    print(f"Baseline yüklendi: R_dy_1 {R_dy_1.shape}")
    print(f"j1={j1}  g0={g0}  g1={g1}  g2={g2}  workers={args.workers}")

    if args.j2_list is not None:
        j2_list = [j for j in args.j2_list if j != j1 and 0 <= j < n_q]
    else:
        j2_list = [j for j in range(0, n_q, args.step) if j != j1]

    zeros = np.zeros(n_q)

    # Tüm j2'ler için task listesi (dy only: ref + n_q sütun)
    all_tasks = []
    for j2 in j2_list:
        dG = np.zeros(n_q)
        dG[j1] = (g1 - g0) / g0
        dG[j2] = (g2 - g0) / g0

        all_tasks.append((config, g0, None, (j2, 'ref'), 0,
                          zeros.copy(), zeros.copy(), zeros.copy(), dG.copy()))
        for i in range(n_q):
            dy = zeros.copy()
            dy[i] = delta_q
            all_tasks.append((config, g0, None, (j2, 'dy'), i,
                               dy, zeros.copy(), zeros.copy(), dG.copy()))

    n_total = len(all_tasks)
    est_min = n_total / args.workers * 78 / 60
    print(f"\n{len(j2_list)} j2 × {n_q + 1} koşum = {n_total} toplam")
    print(f"Tahmini süre: ~{est_min:.0f} dakika  ({args.workers} worker)\n")

    # TEK havuz — tüm j2 paralel
    raw = {}  # (j2, 'ref'/'dy', col_idx) → y_cod
    t0 = time.time()
    progress_step = max(1, n_total // 20)

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        futures = [pool.submit(_run_one, t) for t in all_tasks]
        for done, fut in enumerate(as_completed(futures), 1):
            (j2_tag, tag), col_idx, x_cod, y_cod = fut.result()
            raw[(j2_tag, tag, col_idx)] = y_cod
            if done % progress_step == 0 or done == n_total:
                el = time.time() - t0
                rem = el / done * (n_total - done)
                print(f"  {done}/{n_total}  ({el:.0f}s, ~{rem:.0f}s kalan)")

    # Her j2 için κ(dR_dy) hesapla
    print()
    scan_results = []
    for j2 in j2_list:
        y0     = raw[(j2, 'ref', 0)]
        R_dy_2 = np.zeros((n_q, n_q))
        for i in range(n_q):
            R_dy_2[:, i] = (raw[(j2, 'dy', i)] - y0) / delta_q
        dR    = R_dy_2 - R_dy_1
        kappa = np.linalg.cond(dR)
        scan_results.append((j2, kappa))

    scan_results.sort(key=lambda x: x[1])

    print("=" * 55)
    print(f"{'j2':>4}  {'κ(dR_dy)':>12}  {'not':}")
    print("=" * 55)
    for j2, k in scan_results:
        flag = "  ← EN İYİ" if j2 == scan_results[0][0] else ""
        print(f"  {j2:3d}  {k:12.3e}{flag}")

    best_j2, best_k = scan_results[0]
    print(f"\nEn iyi j2={best_j2}  κ(dR_dy)={best_k:.3e}")
    print(f"(Referans: uniform κ≈160, tek-quad κ≈10⁸)")

    np.save("scan_j2_results.npy", np.array([[j, k] for j, k in scan_results]))
    print("Sonuçlar 'scan_j2_results.npy' kaydedildi.")
    print(f"Toplam süre: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
