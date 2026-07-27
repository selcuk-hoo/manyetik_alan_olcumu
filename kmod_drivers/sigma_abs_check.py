#!/usr/bin/env python3
"""σ=10 μm'de CW-tek sahte EDM'in mutlak seviyesi: çok-seed, PARALEL.
JSON ile AYNI ayar (fast_measure_g varsayılan: n_turns=14, t2=3e-4).
Amaç: 10-15 seed'de ~1e-5 mi ~1e-6 mı görülüyor? (Omarov fit: 1.5e-5 @ σ=10.)

Kullanım (lokal):
    python3 kmod_drivers/sigma_abs_check.py [nseed] [nworker]
    # ör. 15 seed, 8 çekirdek:  python3 kmod_drivers/sigma_abs_check.py 15 8

Not: C++ integratörünün ilerleme çıktısı (fd 1) işçilerde susturulur, yoksa
ekranı boğar. Paralelliği çıktının zaman damgalarından görürsün: paralelse ilk
`nworker` seed ~aynı t'de biter, seri ise t=40,80,120... diye artar.
"""
import os
# numpy/BLAS aşırı-aboneliğini önle (işçi başına tek iş parçacığı) — numpy'den ÖNCE
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, numpy as np, time
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (BASE, os.path.join(BASE, "kmod_drivers"), os.path.join(BASE, "berry_data")):
    sys.path.insert(0, p)
os.chdir(BASE)

SG = 10e-6


def measure_seed(seed):
    """Tek seed, bağımsız süreç. C++ printf'leri (fd 1) susturulur."""
    from paper_runs import fast_measure_g, NQ
    try:  # numpy/OpenBLAS import'u affinity'yi tek çekirdeğe kilitleyebilir — aç
        os.sched_setaffinity(0, range(os.cpu_count()))
    except (AttributeError, OSError):
        pass
    rng = np.random.default_rng(3000 + seed)
    dx = rng.normal(0, SG, NQ)
    dy = rng.normal(0, SG, NQ)
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    os.dup2(devnull, 1)
    try:
        f, _ = fast_measure_g(dx, dy, g_scale=1.0)
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)
    return seed, float(f)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="σ=10μm CW-tek sahte-EDM mutlak seviye (paralel)")
    ap.add_argument("-n", "--seeds", type=int, default=15)
    ap.add_argument("-w", "--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("pos", nargs="*", type=int, help="geriye dönük: [nseed] [nworker]")
    a = ap.parse_args()
    nseed = a.pos[0] if len(a.pos) >= 1 else a.seeds
    nworker = a.pos[1] if len(a.pos) >= 2 else a.workers
    nworker = max(1, min(nworker, nseed))
    try:  # ana süreçte de affinity kilidini aç (çocuklar miras alsın)
        os.sched_setaffinity(0, range(os.cpu_count()))
    except (AttributeError, OSError):
        pass
    naff = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    print(f"σ=10μm, {nseed} seed, {nworker} çekirdek | cpu_count={os.cpu_count()} "
          f"affinity={naff}. Omarov fit: 1.5e-5 @ σ=10.\n", flush=True)

    vals = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=nworker) as ex:
        futs = {ex.submit(measure_seed, s): s for s in range(nseed)}
        done = 0
        for fut in as_completed(futs):
            seed, f = fut.result()
            vals[seed] = f
            done += 1
            v = np.array(list(vals.values()))
            rms = np.sqrt((v ** 2).mean())
            print(f"[{done:2d}/{nseed}] seed {seed:2d}: f={f:.3e}  | "
                  f"koşu RMS={rms:.3e} ({rms/1e-9:.0f}× hedef)  "
                  f"[Omarov/RMS = {1.5e-5/rms:.1f}×]  t={time.time()-t0:.0f}s",
                  flush=True)

    v = np.array([vals[s] for s in sorted(vals)])
    rms = np.sqrt((v ** 2).mean())
    print(f"\n=== {len(v)} seed, σ=10μm ({time.time()-t0:.0f}s) ===")
    print(f"mean={v.mean():.3e}  RMS={rms:.3e}  medyan={np.median(v):.3e}  max={v.max():.3e}")
    print(f"RMS → {rms/1e-9:.0f}× hedef ; Omarov 1.5e-5 ile oran {1.5e-5/rms:.1f}×")


if __name__ == "__main__":
    main()
