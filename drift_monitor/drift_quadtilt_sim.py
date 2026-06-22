#!/usr/bin/env python3
"""
drift_quadtilt_sim.py — Quad tilt kaynaklı x-y kuplajının drift monitöre etkisi.

Test 6 (ve makalenin lineer modeli) x ve y düzlemlerini AYRIK kabul eder. Quad
tilt (kuadrupol dönmesi) gerçek bir skew bileşeni → x-y kuplajı yaratır. Bu
betik, gerçek izleyiciyle kuplajlı tam tepki matrisini (4 blok: R_yy, R_yx,
R_xy, R_xx) kurar; sonra DÜZLEM-AYRIK monitör (R_yy⁻¹, R_xx⁻¹) ile drift
kurtarımı yapıp, ihmal edilen kuplajın takip hatasına etkisini ölçer.

Soru: gerçekçi quad tilt (~0.2 mrad) altında düzlem-ayrık varsayım kırılır mı?

Çıktı: her tilt için (i) kuplaj oranı ‖R_yx‖/‖R_xx‖, (ii) decoupled drift
takip hatası (y, x). Tablo 6 yanına eklenir.

Kullanım: python3 drift_quadtilt_sim.py --workers 7
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_DIR)
sys.path.insert(0, _BASE)
import build_response_matrix as brm

OFF, NOISE, RAMP, DQ0, NEP, NSEED = 50e-6, 1e-6, 10e-6, 100e-6, 10, 15


def _worker(task):
    cfg, kind, idx, dq, qtilt = task
    nq = 2 * int(cfg["nFODO"])
    dy = np.zeros(nq); dx = np.zeros(nq)
    if kind == "dy":
        dy[idx] = dq
    elif kind == "dx":
        dx[idx] = dq
    f, s = brm.setup_fields(cfg, g1_override=cfg["g1"])
    x_cod, y_cod = brm.run_sim(f, s, cfg, dy, dx, quad_tilt=np.asarray(qtilt))
    return kind, idx, x_cod, y_cod


def build_coupled(cfg, qtilt, dq, nworkers):
    """Kuplajlı 4-blok tepki matrisi (R_yy, R_yx, R_xy, R_xx)."""
    nq = 2 * int(cfg["nFODO"])
    ql = qtilt.tolist()
    tasks = [(cfg, "ref", 0, dq, ql)]
    tasks += [(cfg, "dy", j, dq, ql) for j in range(nq)]
    tasks += [(cfg, "dx", j, dq, ql) for j in range(nq)]
    res = {}
    with ProcessPoolExecutor(nworkers, initializer=brm._worker_init) as pool:
        for fut in as_completed([pool.submit(_worker, t) for t in tasks]):
            k, i, xc, yc = fut.result()
            res[(k, i)] = (xc, yc)
    x0, y0 = res[("ref", 0)]
    Ryy = np.zeros((nq, nq)); Rxy = np.zeros((nq, nq))
    Ryx = np.zeros((nq, nq)); Rxx = np.zeros((nq, nq))
    for j in range(nq):
        xc, yc = res[("dy", j)]
        Ryy[:, j] = (yc - y0) / dq; Rxy[:, j] = (xc - x0) / dq
        xc, yc = res[("dx", j)]
        Ryx[:, j] = (yc - y0) / dq; Rxx[:, j] = (xc - x0) / dq
    return Ryy, Ryx, Rxy, Rxx


def decoupled_recovery(blocks, seed):
    """Düzlem-ayrık monitör (R_yy⁻¹, R_xx⁻¹) ile kuplajlı veriden kurtarım."""
    Ryy, Ryx, Rxy, Rxx = blocks
    nq = Ryy.shape[0]
    Ryy_i = np.linalg.inv(Ryy); Rxx_i = np.linalg.inv(Rxx)
    rng = np.random.default_rng(2000 + seed)
    dy0 = rng.normal(0, DQ0, nq); dx0 = rng.normal(0, DQ0, nq)
    by = rng.normal(0, OFF, nq); bx = rng.normal(0, OFF, nq)
    ry = rng.normal(0, RAMP, nq); rx = rng.normal(0, RAMP, nq)

    def orbit(dy, dx):
        return (Ryy @ dy + Ryx @ dx + by + rng.normal(0, NOISE, nq),
                Rxy @ dy + Rxx @ dx + bx + rng.normal(0, NOISE, nq))
    y0, x0 = orbit(dy0, dx0)
    ey, ex = [], []
    for t in range(1, NEP + 1):
        dyt = dy0 + ry * (t / NEP); dxt = dx0 + rx * (t / NEP)
        yt, xt = orbit(dyt, dxt)
        dyhat = Ryy_i @ (yt - y0); dxhat = Rxx_i @ (xt - x0)
        ey.append(np.sqrt(np.mean((dyhat - (dyt - dy0)) ** 2)))
        ex.append(np.sqrt(np.mean((dxhat - (dxt - dx0)) ** 2)))
    return np.mean(ey), np.mean(ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", "-w", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--tilts", type=float, nargs="+",
                    default=[0.0, 2e-4, 1e-3])   # rad RMS (0, 0.2, 1 mrad)
    args = ap.parse_args()
    os.chdir(_BASE)
    with open("params.json") as f:
        cfg = json.load(f)
    nq = 2 * int(cfg["nFODO"])

    print(f"{'tilt[mrad]':>10} {'‖Ryx‖/‖Rxx‖':>13} {'y-track[μm]':>12} {'x-track[μm]':>12}")
    rows = []
    for tilt in args.tilts:
        rng = np.random.default_rng(8000 + int(tilt * 1e5))
        qtilt = rng.normal(0, tilt, nq) if tilt > 0 else np.zeros(nq)
        t0 = time.time()
        blocks = build_coupled(cfg, qtilt, 1e-4, args.workers)
        Ryy, Ryx, Rxy, Rxx = blocks
        coup = np.linalg.norm(Ryx) / np.linalg.norm(Rxx)
        errs = [decoupled_recovery(blocks, s) for s in range(NSEED)]
        ey = np.median([e[0] for e in errs]) * 1e6
        ex = np.median([e[1] for e in errs]) * 1e6
        rows.append((tilt, coup, ey, ex))
        print(f"{tilt*1e3:>10.2f} {coup*100:>12.3f}% {ey:>12.2f} {ex:>12.2f}  "
              f"({time.time()-t0:.0f}s)")

    np.save(os.path.join(_DIR, "quadtilt_coupling.npy"), np.array(rows))
    print("\nKaydedildi: drift_monitor/quadtilt_coupling.npy")


if __name__ == "__main__":
    main()
