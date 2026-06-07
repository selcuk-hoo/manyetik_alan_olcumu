#!/usr/bin/env python3
"""make_fig3.py — Figure 3: White BPM offset mode patterns (Scatter plot with Multiprocessing)"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp

from paper_plot_utils import apply_paper_style, load_R, Fcos, Fsin, M_col_norm, BLUE, RED, GREEN, GRAY

# Global variables for multiprocessing worker
_N_Q = None
_k_show = None
_sigma_b = None

def init_worker(n_q, k_show, sigma_b):
    global _N_Q, _k_show, _sigma_b
    _N_Q = n_q
    _k_show = k_show
    _sigma_b = sigma_b

def compute_mc_iteration(seed):
    rng = np.random.default_rng(seed)
    b = rng.normal(0, _sigma_b, _N_Q)
    amps = {}
    for k in _k_show:
        Fc = Fcos(k, _N_Q)
        ac = Fc @ b / (Fc @ Fc)
        if k == 0:
            A_k = abs(ac)
        else:
            Fs = Fsin(k, _N_Q)
            as_ = Fs @ b / (Fs @ Fs)
            A_k = np.hypot(ac, as_)
        amps[k] = A_k * 1e6
    return amps

def make_fig3():
    apply_paper_style()
    R = load_R()
    
    C_ring = 2 * np.pi * 95.49
    N_Q = R.shape[0]
    s_bpm  = np.linspace(0, C_ring, N_Q, endpoint=False)
    
    A_demo3  = 10e-6
    sigma_b3 = 100e-6
    N_MC3    = 2000
    k_show   = list(range(0, 12))
    
    # Multiprocessing MC Loop
    print(f"Running {N_MC3} MC iterations using 12 cores...")
    seeds = np.random.SeedSequence(42).generate_state(N_MC3)
    
    with mp.Pool(processes=12, initializer=init_worker, initargs=(N_Q, k_show, sigma_b3)) as pool:
        results = pool.map(compute_mc_iteration, seeds)
        
    amp_mc = {k: [] for k in k_show}
    for res in results:
        for k in k_show:
            amp_mc[k].append(res[k])
            
    means3 = np.array([np.mean(amp_mc[k]) for k in k_show])
    stds3  = np.array([np.std(amp_mc[k])  for k in k_show])
    theory3 = sigma_b3 * 1e6 * np.sqrt(np.pi / 48)
    
    b_ex3 = np.random.default_rng(7).normal(0, sigma_b3, N_Q) * 1e6
    
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 4))
    
    # (a) White BPM offset realisation
    ax3a.plot(s_bpm, b_ex3, "o-", color=GRAY, ms=3, lw=0.8)
    ax3a.axhline(0, color="k", lw=0.4)
    ax3a.set_xlabel(r"Ring position $s$ [m]")
    ax3a.set_ylabel(r"BPM offset $b\;[\mu\mathrm{m}]$")
    ax3a.set_title(fr"(a) White BPM offset ($\sigma_b = {sigma_b3*1e6:.0f}\,\mu$m)")
    
    # (b) Fourier amplitude spectrum (Scatter with error bars)
    xpos3  = np.array(k_show)
    
    for i, k in enumerate(k_show):
        ax3b.errorbar(xpos3[i], means3[i], yerr=stds3[i], fmt='o', color=BLUE,
                      capsize=4, markersize=7)

    ax3b.set_yscale("log")
    ax3b.set_ylim(bottom=0.5)
    ax3b.set_xticks(xpos3)
    ax3b.set_xticklabels([f"{k}" for k in k_show])
    ax3b.set_xlabel("Fourier mode $k$")
    ax3b.set_ylabel(r"Amplitude [$\mu$m]  (log scale)")
    ax3b.set_title("(b) Offset $F_k$ level vs Signal Orbit")
    ax3b.legend(frameon=False, fontsize=8.5, loc='center left', bbox_to_anchor=(1, 0.5))
    ax3b.grid(True, axis='y', linestyle='--', alpha=0.2)
    
    fig3.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig3_mode_patterns.png")
    fig3.savefig(out_path, bbox_inches="tight")
    plt.close(fig3)
    print(f"fig3_mode_patterns.png generated at {out_path}")

if __name__ == "__main__":
    make_fig3()
