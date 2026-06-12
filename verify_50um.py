#!/usr/bin/env python3
"""verify_50um.py — σ=50 μm RMS'in 2e-5 rad/s sahte EDM verdiğini doğrular.

sim ∝ σ² (geometrik faz) olduğundan, σ=10μm'deki |sim|_mean=7.89e-7'den
σ=50μm için 7.89e-7 × 25 ≈ 1.97e-5 rad/s beklenir. 10 seed ile paralel ölçer.
"""

import json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)

C_LIGHT  = 299792458.0
SIGMA    = 50e-6   # [m]
N_SEEDS  = 10
T2       = 2e-3
DT       = 1e-11
CO_TURNS = 60
CO_ITER  = 3
RET_STEP = 10000
N_WORKERS = 10


def run_pair(args):
    sigma_m, seed, cfg = args
    import os, sys
    os.chdir(BASE); sys.path.insert(0, BASE)
    from integrator import integrate_particle
    from false_edm_mode_scan import setup_fields, find_closed_orbit, _make_state

    fields, y0, beta0, R0, p_mag, direction = setup_fields(cfg)
    n_q  = 2 * int(fields.nFODO)
    circ = (2 * np.pi * R0 + 4 * fields.nFODO * fields.driftLen
            + 2 * fields.nFODO * fields.quadLen)
    T_rev = circ / (beta0 * C_LIGHT)

    rng     = np.random.default_rng(seed)
    quad_dy = rng.normal(0.0, sigma_m, n_q)
    quad_dx = rng.normal(0.0, sigma_m, n_q)

    def _run_one(sign):
        dy = sign * quad_dy; dx = sign * quad_dx
        v_co, _ = find_closed_orbit(fields, p_mag, direction, dy, DT, T_rev,
                                    n_turns=CO_TURNS, n_iter=CO_ITER)
        y_co = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
        fields.poincare_quad_index = 0.0
        _, poin, poin_t = integrate_particle(y_co, 0.0, T2, DT, fields=fields,
                                             return_steps=RET_STEP, quad_dy=dy, quad_dx=dx)
        ts = np.asarray(poin_t, float); sy = np.asarray(poin[:, 7], float)
        return float(np.polyfit(ts, sy, 1)[0])

    r_plus  = _run_one(+1)
    r_minus = _run_one(-1)
    return seed, r_plus, r_minus


def main():
    t0 = time.time()
    with open("params.json") as fh:
        cfg = json.load(fh)
    tasks = [(SIGMA, s, cfg) for s in range(N_SEEDS)]

    print(f"σ = {SIGMA*1e6:.0f} μm, {N_SEEDS} seed, {N_WORKERS} işlemci")
    print(f"{'seed':>4}  {'rate+ [rad/s]':>16}  {'rate- [rad/s]':>16}  "
          f"{'sim [σ²]':>16}  {'antisim [σ¹]':>16}")

    sims, antis = [], []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_pair, t): t for t in tasks}
        for fut in as_completed(futs):
            seed, rp, rm = fut.result()
            sym = (rp + rm) / 2.0; anti = (rp - rm) / 2.0
            sims.append(sym); antis.append(anti)
            print(f"{seed:4d}  {rp:16.4e}  {rm:16.4e}  {sym:16.4e}  {anti:16.4e}", flush=True)

    sims = np.array(sims); antis = np.array(antis)
    print(f"\n  |sim|_mean   = {np.mean(np.abs(sims)):.4e} rad/s   (σ² geometrik faz)")
    print(f"  |sim|_max    = {np.max(np.abs(sims)):.4e} rad/s")
    print(f"  sim_rms      = {np.std(sims):.4e} rad/s")
    print(f"  |antisim|_mn = {np.mean(np.abs(antis)):.4e} rad/s   (σ³ artığı)")
    print(f"\n  Beklenen (10μm×25): {7.89e-7*25:.4e} rad/s")
    print(f"  Süre: {(time.time()-t0)/60:.1f} dk")


if __name__ == "__main__":
    main()
