#!/usr/bin/env python3
"""make_fig5.py — Figure 5: BPM-offset whiteness (Multiprocessing, Scatter plot)"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp

from paper_plot_utils import apply_paper_style, load_R, Fcos, Fsin, M_col_norm, BLUE, RED, GRAY

_M_pinv = None
_N_Q = None
_KMAX = 12

def init_worker(M_pinv, N_Q):
    global _M_pinv, _N_Q
    _M_pinv = M_pinv
    _N_Q = N_Q

def amp_spectrum(ahat):
    out = [abs(ahat[0])]
    for k in range(1, _KMAX + 1):
        out.append(np.hypot(ahat[1 + 2 * (k - 1)], ahat[2 + 2 * (k - 1)]))
    return np.array(out)

def compute_mc_iteration(args):
    sigma_b, seed = args
    rng = np.random.default_rng(seed)
    b = rng.normal(0, sigma_b, _N_Q)
    ahat = _M_pinv @ b
    return amp_spectrum(ahat) * 1e6

def make_fig5():
    apply_paper_style()
    R = load_R()
    N_Q = R.shape[0]
    C_ring = 2 * np.pi * 95.49
    s_bpm  = np.linspace(0, C_ring, N_Q, endpoint=False)
    
    # Prepare pseudo-inverse matrix for estimator
    cols5 = [Fcos(0, N_Q)]
    for k in range(1, _KMAX + 1):
        cols5.append(Fcos(k, N_Q))
        cols5.append(Fsin(k, N_Q))
    F_full = np.column_stack(cols5)
    M_full = R @ F_full
    M_pinv = np.linalg.pinv(M_full, rcond=1e-3)
    
    sigma_b5 = 100e-6
    A_sig5   = 10e-6
    
    # True k=2 signal
    a_sig = amp_spectrum(M_pinv @ (R @ (A_sig5 * Fcos(2, N_Q)))) * 1e6
    
    # Multiprocessing MC Loop
    N_TR5 = 400
    print(f"Running {N_TR5} MC iterations using 12 cores...")
    seeds = np.random.SeedSequence(5).generate_state(N_TR5)
    tasks = [(sigma_b5, seeds[i]) for i in range(N_TR5)]
    
    with mp.Pool(processes=12, initializer=init_worker, initargs=(M_pinv, N_Q)) as pool:
        specs = np.array(pool.map(compute_mc_iteration, tasks))
        
    off_mean = specs.mean(0)
    off_std  = specs.std(0)
    
    b_example = np.random.default_rng(7).normal(0, sigma_b5, N_Q) * 1e6
    
    theory_floor = np.array([np.nan] + [sigma_b5 * 1e6 / M_col_norm(R, k)
                                        for k in range(1, _KMAX + 1)])
    
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4))
    
    # (a) example white-offset
    axA.plot(s_bpm, b_example, "o-", color=GRAY, ms=3, lw=0.8)
    axA.axhline(0, color="k", lw=0.5)
    axA.set_xlabel(r"Ring position $s$ [m]")
    axA.set_ylabel(r"BPM offset $b$ [$\mu$m]")
    axA.set_title(fr"(a) White BPM offset ($\sigma_b={sigma_b5*1e6:.0f}\,\mu$m)")
    
    # (b) recovered harmonic spectrum
    kk = np.arange(0, _KMAX + 1)
    ks = kk[1:12]
    axB.errorbar(ks, off_mean[1:12], yerr=off_std[1:12], fmt="o", color=BLUE,
                 capsize=4, markersize=7)
    
    axB.set_yscale("log")
    axB.set_xlabel("Fourier mode $k$")
    axB.set_ylabel(r"Recovered amplitude $|\hat{a}_k|$ [$\mu$m]")
    axB.set_title("(b) Offset is broadband")
    axB.set_xticks(ks)
    axB.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig5_offset_whiteness.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"fig5_offset_whiteness.png generated at {out_path}")

if __name__ == "__main__":
    make_fig5()
