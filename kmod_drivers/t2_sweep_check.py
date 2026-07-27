#!/usr/bin/env python3
"""t2-TARAMASI: seküler sahte-EDM eğiminin ölçüm penceresi (t2) ile değişimi.

Hipotez: fast_measure_g'nin varsayılan t2=3e-4 (~90 tur) penceresi, seküler
geometrik fazı yavaş spin-tune dalgasından ayırmaya yetmiyor → b ~10× düşük
çıkıyor. Docstring (measure_dSy_dt_model): "periyodu t2'den uzun yavaş dalga
doğru ile dejenere; yalnız daha uzun t2 ile çözülür."

Bu betik tek σ=10μm seed'i giderek uzayan t2 ile ölçer. b, uzun t2'de Omarov'un
~1.5e-5'ine tırmanıyorsa → suçlu pencere/estimator; sabit ~1.6e-6 kalıyorsa değil.

Kullanım (lokal):
    python3 kmod_drivers/t2_sweep_check.py [--seed 0] [--t2 3e-4,1e-3,3e-3,1e-2]
UYARI: t2 ∝ integrasyon adımı. t2=1e-2 ~30× ağırdır (tek parçacık, birkaç dk).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, numpy as np, time, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (BASE, os.path.join(BASE, "kmod_drivers"), os.path.join(BASE, "berry_data")):
    sys.path.insert(0, p)
os.chdir(BASE)

SG = 10e-6


def measure_at_t2(args):
    seed, t2 = args
    from paper_runs import fast_measure_g, NQ
    try:
        os.sched_setaffinity(0, range(os.cpu_count()))
    except (AttributeError, OSError):
        pass
    rng = np.random.default_rng(3000 + seed)
    dx = rng.normal(0, SG, NQ)
    dy = rng.normal(0, SG, NQ)
    devnull = os.open(os.devnull, os.O_WRONLY); saved = os.dup(1); os.dup2(devnull, 1)
    try:
        f, _ = fast_measure_g(dx, dy, g_scale=1.0, t2=t2)
    finally:
        os.dup2(saved, 1); os.close(saved); os.close(devnull)
    return t2, float(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t2", type=str, default="3e-4,1e-3,3e-3,1e-2")
    a = ap.parse_args()
    t2list = [float(x) for x in a.t2.split(",")]
    try:
        os.sched_setaffinity(0, range(os.cpu_count()))
    except (AttributeError, OSError):
        pass
    # T_rev (yaklaşık, tur sayısını göstermek için)
    import json as _j
    cfg = _j.load(open("params.json"))
    R0 = float(cfg["R0"]); beta = 0.598
    T_rev = 2 * np.pi * R0 / (beta * 2.99792458e8)
    print(f"seed={a.seed}, σ=10μm.  T_rev≈{T_rev:.2e}s.  Omarov fit: 1.5e-5 @ σ=10.\n",
          flush=True)

    out = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(len(t2list), os.cpu_count() or 4)) as ex:
        futs = [ex.submit(measure_at_t2, (a.seed, t2)) for t2 in t2list]
        for fut in as_completed(futs):
            t2, f = fut.result()
            out[t2] = f
            print(f"t2={t2:.1e}s (~{t2/T_rev:5.0f} tur): b={f:.3e} rad/s "
                  f"({f/1e-9:6.0f}× hedef)  [Omarov/b = {1.5e-5/f:5.1f}×]  "
                  f"t={time.time()-t0:.0f}s", flush=True)

    print("\n=== özet (artan t2) ===")
    for t2 in t2list:
        if t2 in out:
            print(f"  t2={t2:.1e}: b={out[t2]:.3e}  ({out[t2]/1e-9:.0f}× hedef)")
    print("b uzun t2'de ~1.5e-5'e tırmanıyorsa → pencere/estimator suçlu; "
          "sabit kalıyorsa değil.")


if __name__ == "__main__":
    main()
