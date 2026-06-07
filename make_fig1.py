#!/usr/bin/env python3
"""make_fig1.py — Figure 1: False-EDM rate vs k (Scatter with Error bars, Full Spin Tracking)"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_plot_utils import apply_paper_style, BLUE
from false_edm_mode_scan import run_scan

def make_fig1():
    apply_paper_style()
    
    A_true = 10e-6     # 10 um
    k_list = list(range(0, 12))
    
    cache_file = "make_fig1_cache.json"
    
    if os.path.exists(cache_file):
        print(f"Loading cached spin tracking data from {cache_file}...")
        import json
        with open(cache_file, 'r') as f:
            results = json.load(f)
    else:
        print(f"Running full spin tracking for k=0..11 using 12 cores...")
        results, _ = run_scan(k_list, amp_coef=A_true, t2=5e-4, nproc=12)
        
        # Save to cache
        import json
        cache_data = []
        for r in results:
            cache_data.append({
                "k": r["k"],
                "dSy_dt": r["dSy_dt"],
                "dSy_dt_err": r.get("dSy_dt_err", 0.0) if not np.isnan(r.get("dSy_dt_err", 0.0)) else 0.0,
                "co_off_mm": r["co_off_mm"]
            })
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    dsydt = {}
    dsydt_err = {}
    co_amp = {}
    
    for r in results:
        k = r["k"]
        dsydt[k] = r["dSy_dt"]
        dsydt_err[k] = r.get("dSy_dt_err", 0.0)
        co_amp[k] = r["co_off_mm"]
        
    ks = sorted(dsydt.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    
    # Plot (a) False-EDM rate
    ax = axes[0]
    for k in ks:
        val = abs(dsydt[k]) * 1e9
        err = dsydt_err[k] * 1e9 if not np.isnan(dsydt_err[k]) else val * 0.05
        color = BLUE
        
        # Scatter with error bars (no plus signs)
        ax.errorbar(k, val, yerr=err, fmt='o', color=color, capsize=4, markersize=7)

    ax.set_xlabel("Fourier mode $k$")
    ax.set_ylabel(r"$|dS_y/dt|$ [$10^{-9}$ rad/s]")
    ax.set_title("(a) False-EDM rate")
    ax.set_xticks(ks)
    ax.set_ylim(bottom=0, top=max(abs(v)*1e9 + dsydt_err.get(k, 0)*1e9 for k, v in dsydt.items()) * 1.3)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Plot (b) Closed-orbit amplitude
    ax = axes[1]
    for k in ks:
        val = co_amp[k]
        color = BLUE
        
        ax.plot(k, val, 'o', color=color, markersize=7)

    ax.set_xlabel("Fourier mode $k$")
    ax.set_ylabel("Closed-orbit amplitude [mm]")
    ax.set_title("(b) Closed-orbit amplitude")
    ax.set_xticks(ks)
    ax.set_ylim(bottom=0, top=max(co_amp.values()) * 1.2)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    fig.suptitle(r"$A=10\,\mu$m single-harmonic misalignment, true EDM off", fontsize=11, y=1.02)
    fig.tight_layout()
    
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig1_falseedm_scan.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"fig1_falseedm_scan.png generated at {out_path}")

if __name__ == "__main__":
    make_fig1()
