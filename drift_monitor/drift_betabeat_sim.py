#!/usr/bin/env python3
"""
drift_betabeat_sim.py — Test 8'in GERÇEK izleyiciyle doğrulaması.

Test 8 (analitik) β-beating'i β,φ'yi elle bozarak modelliyordu. Burada
β-beating gerçek kaynağından — kuadrupol gradyan hataları (quad_dG) —
C++ izleyicide indüklenir; bozulmuş tepki matrisi R_true tam parçacık
takibiyle kurulur. Drift monitör nominal R_nom (gradyan hatasız, izleyici)
ile çalışır:  δq̂(t) = R_nom⁻¹(y_true(t) − y0).

Soru: gradyan-hatası kaynaklı model uyumsuzluğu drift kurtarımını ne kadar
bozar? (Test 8'in izleyici karşılığı.)

Süre: her ε için 1 R_true inşası (~2.5 dk, paralel). ε listesi kısadır.

Kullanım:
    python3 drift_betabeat_sim.py --workers 7
"""
import os
import sys
import json
import argparse
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_DIR)
sys.path.insert(0, _BASE)
from build_response_matrix import build_matrices

OFF, NOISE, RAMP, DQ0, NEP, NSEED = 50e-6, 1e-6, 10e-6, 100e-6, 10, 15


def drift_recovery(R_true, R_nom, seed):
    """δq̂ = R_nom⁻¹(y_true−y0) ile drift takip hatası RMS [m]."""
    N = R_true.shape[0]
    Rinv = np.linalg.inv(R_nom)
    rng = np.random.default_rng(1000 + seed)
    dq0 = rng.normal(0, DQ0, N)
    b0 = rng.normal(0, OFF, N)
    ramp = rng.normal(0, RAMP, N)
    y0 = R_true @ dq0 + b0 + rng.normal(0, NOISE, N)
    errs = []
    for t in range(1, NEP + 1):
        dqt = dq0 + ramp * (t / NEP)
        yt = R_true @ dqt + b0 + rng.normal(0, NOISE, N)
        dqhat = Rinv @ (yt - y0)
        errs.append(np.sqrt(np.mean((dqhat - (dqt - dq0)) ** 2)))
    return np.mean(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", "-w", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--eps", type=float, nargs="+", default=[0.0, 0.02, 0.05])
    args = ap.parse_args()
    os.chdir(_BASE)
    with open("params.json") as f:
        cfg = json.load(f)
    nq = 2 * int(cfg["nFODO"])

    # Nominal R (gradyan hatasız) — izleyici
    print(f"[nominal] R_nom izleyiciden... (w={args.workers})")
    R_nom_dy, _ = build_matrices(cfg, g1_override=cfg["g1"], delta_q=1e-4,
                                 label="nom", n_workers=args.workers)

    print(f"\n{'ε(grad)':>8} {'κ(R_true)':>10} {'y-track[μm]':>12}")
    rows = []
    for eps in args.eps:
        if eps == 0.0:
            R_true = R_nom_dy
        else:
            rng = np.random.default_rng(7000 + int(eps * 1e4))
            dG = rng.normal(0, eps, nq)
            R_true, _ = build_matrices(cfg, g1_override=cfg["g1"], delta_q=1e-4,
                                       label=f"eps{eps}", n_workers=args.workers,
                                       quad_dG_pert=dG)
        errs = [drift_recovery(R_true, R_nom_dy, s) * 1e6 for s in range(NSEED)]
        med = np.median(errs)
        kap = np.linalg.cond(R_true)
        rows.append((eps, kap, med))
        print(f"{eps:>8.3f} {kap:>10.0f} {med:>12.2f}")

    np.save(os.path.join(_DIR, "betabeat_tracker.npy"), np.array(rows))
    print("\nKaydedildi: drift_monitor/betabeat_tracker.npy")
    print("Not: ε = kuadrupol fraksiyonel gradyan hatası RMS; indüklenen "
          "β-beating mertebesi benzer. Test 8 (analitik) ile karşılaştır.")


if __name__ == "__main__":
    main()
