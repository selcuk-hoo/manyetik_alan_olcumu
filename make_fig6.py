#!/usr/bin/env python3
"""make_fig6.py — Figure 6: Combined Systematics Error Budget via Beam Dynamics Simulation

Maksat: Hata kaynaklarının (Gürültü, Ofset, Rulo, Gradyan) bireysel ve kombine etkisini 
doğrudan tam C++ demet dinamiği simülasyonu ile üretilen kapalı yörünge (orbit) 
üzerinden CLEAN geri çatım algoritmasıyla test etmek. Doğrusal R matrisi varsayımları 
yerine gerçek simülasyon çıktısı kullanılarak makale için güvenilir veri sağlanır.
"""

import os
import json
import math
import tempfile
import shutil
import numpy as np
import multiprocessing as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_plot_utils import apply_paper_style, BLUE, RED, GREEN, GRAY
from fourier_reconstruct import fodo_basis, clean_reconstruct
from build_response_matrix import setup_fields, run_sim, read_cod_quads

# Global variables for multiprocessing workers
_config = None
_R_dy = None
_n_q = None
_antisym = None

def _worker_init(config, R_dy, n_q, antisym):
    global _config, _R_dy, _n_q, _antisym
    _config = config
    _R_dy = R_dy
    _n_q = n_q
    _antisym = antisym

def _simulate_mc_iteration(task):
    """
    Her MC döngüsünde rastgele hata profilleri oluşturulur ve doğrudan
    integrator.cpp ile simüle edilir. Çıkan orbit CLEAN algoritmasına verilir.
    """
    mc_idx, LEVELS, TRUTH, K_TARGETS, cn_cases = task
    
    rng = np.random.default_rng(mc_idx)
    
    # 1. Base Truth Orbit Generation (Background Modes)
    dy_truth = np.zeros(_n_q)
    for k, (A, phi) in TRUTH.items():
        F, _ = fodo_basis(_n_q, [k], _antisym)
        dy_truth += A * math.cos(phi) * F[:, 0] + A * math.sin(phi) * F[:, 1]
        
    # 2. Random Errors for this MC iteration
    noise  = rng.normal(0, LEVELS["σ_noise"],  _n_q)
    offset = rng.normal(0, LEVELS["σ_offset"], _n_q)
    dx     = rng.normal(0, LEVELS["σ_dx"],     _n_q)
    theta  = rng.normal(0, LEVELS["θ_rms"],    _n_q)
    eps_g  = rng.normal(0, LEVELS["σ_G"],      _n_q)
    
    # We will simulate 5 scenarios by running the C++ integrator.
    # To do this safely in parallel, use a temporary directory.
    old_dir = os.getcwd()
    tmp_dir = tempfile.mkdtemp(prefix=f"sim_mc_{mc_idx}_")
    os.chdir(tmp_dir)
    
    results = {cn: {} for cn in cn_cases}
    try:
        alanlar, state0 = setup_fields(_config)
        
        # Helper to run simulation and return measured orbit
        def get_simulated_orbit(q_dy, q_dx=None, q_tilt=None, q_dG=None, q_noise=None, q_offset=None):
            if q_dx is None: q_dx = np.zeros(_n_q)
            if q_tilt is None: q_tilt = np.zeros(_n_q)
            if q_dG is None: q_dG = np.zeros(_n_q)
            
            run_sim(alanlar, state0, _config, quad_dy=q_dy, quad_dx=q_dx, 
                    quad_tilt=q_tilt, quad_dG=q_dG)
            _, y_true = read_cod_quads(alanlar.nFODO)
            
            if q_offset is not None:
                y_true += q_offset
            if q_noise is not None:
                y_true += q_noise
            return y_true
            
        # 1. Base Machine (No errors, only truth background)
        y_base = get_simulated_orbit(q_dy=dy_truth)
        accum_base, _, F_base = clean_reconstruct([_R_dy], [y_base], list(range(1, 11)), _antisym)
        A_base = {}
        for k in K_TARGETS:
            db = {kind: accum_base[k][i] for i, (_, kind) in enumerate(F_base[k][1])}
            A_base[k] = math.sqrt(db.get('cos',0)**2 + db.get('sin',0)**2)
            
        # 2. Systematics Scenarios
        g1_nom = _config.get("g1", 0.21)
        dg = g1_nom * eps_g
        scenarios = {
            "Gürültü":  dict(q_dy=dy_truth, q_noise=noise),
            "Ofset":    dict(q_dy=dy_truth, q_offset=offset),
            "Rulo":     dict(q_dy=dy_truth, q_dx=dx, q_tilt=theta),
            "Gradyan":  dict(q_dy=dy_truth, q_dG=dg),
            "Kombine":  dict(q_dy=dy_truth, q_dx=dx, q_tilt=theta, q_dG=dg, q_noise=noise, q_offset=offset)
        }
        
        results = {cn: {} for cn in cn_cases}
        
        for cn in cn_cases:
            if cn not in scenarios: continue
            
            y_s = get_simulated_orbit(**scenarios[cn])
            accum, _, F_cache = clean_reconstruct([_R_dy], [y_s], list(range(1, 11)), _antisym)
            
            for k in K_TARGETS:
                a_k = accum[k]
                dp = {kind: a_k[i] for i, (_, kind) in enumerate(F_cache[k][1])}
                A_f = math.sqrt(dp.get('cos',0)**2 + dp.get('sin',0)**2)
                
                results[cn][k] = (A_f - A_base[k]) / A_base[k] * 100
                
        return results
        
    finally:
        os.chdir(old_dir)
        shutil.rmtree(tmp_dir)

def make_fig6():
    apply_paper_style()
    
    with open("params.json") as f:
        config = json.load(f)
        
    config["t2"] = 1e-3  # Short tracking for orbit
    n_q = 2 * int(config["nFODO"])
    antisym = config.get("smooth_antisym_fodo", True)
    
    try:
        R_dy = np.load("R_dy_1.npy")
    except:
        from paper_plot_utils import load_R
        R_dy = load_R()

    LEVELS = {
        "σ_noise":  5e-6,
        "σ_offset": 50e-6,
        "θ_rms":    1e-3,
        "σ_dx":     100e-6,
        "σ_G":      0.005,
    }
    
    TRUTH = {
        1: (30e-6,  0.80),
        2: (10e-6,  1.50),
        3: (25e-6,  0.30),
    }
    rng_t = np.random.default_rng(42)
    for k in range(4, 11):
        TRUTH[k] = (float(rng_t.uniform(100e-6, 300e-6)), float(rng_t.uniform(0, 2*math.pi)))

    K_TARGETS = [1, 2, 3]
    N_MC = 50  # 50 is reasonable for 12 cores, takes ~1-2 mins total
    
    CONTRIB_NAMES = ["Gürültü", "Ofset", "Rulo", "Gradyan", "Kombine"]
    
    tasks = [(i, LEVELS, TRUTH, K_TARGETS, CONTRIB_NAMES) for i in range(N_MC)]
    
    cache_file = "make_fig6_cache.json"
    if os.path.exists(cache_file):
        print(f"Loading cached spin tracking data from {cache_file}...")
        with open(cache_file, 'r') as f:
            mc_results = json.load(f)
    else:
        print(f"Running full beam dynamics MC ({N_MC} iterations) for Combined Systematics...")
        ctx = mp.get_context("spawn")
        with ctx.Pool(12, initializer=_worker_init, initargs=(config, R_dy, n_q, antisym)) as pool:
            mc_results = pool.map(_simulate_mc_iteration, tasks)
            
        # Convert dictionary keys from string/int to standard types for JSON (int keys become string in json)
        # But here mc_results is a list of dicts: {"Gürültü": {1: val, 2: val, 3: val}, ...}
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(mc_results, f, indent=2)
        
    # Aggregate results
    dA_mc = {cn: {k: [] for k in K_TARGETS} for cn in CONTRIB_NAMES}
    for res in mc_results:
        for cn in CONTRIB_NAMES:
            for k in K_TARGETS:
                # json keys might be strings if loaded from cache
                dA_mc[cn][k].append(res[cn][str(k)] if str(k) in res[cn] else res[cn][k])
                
    def stats(lst):
        a = np.array(lst)
        return np.mean(a), np.std(a)

    # Simplified Scatter / Errorbar Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    
    cn_short = [
        f"BPM Noise\n{LEVELS['σ_noise']*1e6:.0f}μm",
        f"BPM Offset\n{LEVELS['σ_offset']*1e6:.0f}μm",
        f"Quad Tilt\n{LEVELS['θ_rms']*1e3:.0f}mrad",
        f"Gradient\n{LEVELS['σ_G']*100:.1f}%",
        "Combined"
    ]
    
    colors = ["steelblue", "tab:orange", "purple", "olive", "tab:red"]
    x_pos = np.arange(len(CONTRIB_NAMES))
    
    for ax_i, k in enumerate(K_TARGETS):
        ax = axes[ax_i]
        means = [stats(dA_mc[cn][k])[0] for cn in CONTRIB_NAMES]
        stds  = [stats(dA_mc[cn][k])[1] for cn in CONTRIB_NAMES]
        
        for i, (m, s, c) in enumerate(zip(means, stds, colors)):
            ax.errorbar(x_pos[i], m, yerr=s, fmt='o', color=c, capsize=5, markersize=8, markeredgecolor='black')
            
        ax.axhline(0, color="k", lw=0.8, linestyle='--')
        ax.axvline(3.5, color="k", lw=0.5, ls=":")
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cn_short, fontsize=9, rotation=20, ha='right')
        if ax_i == 0:
            ax.set_ylabel("Amplitude Error $\Delta A/A$ [%]")
        
        # Add empirical false EDM rate (using KAPPA=7e-6 as estimate to answer user's question visually)
        # dSy/dt = KAPPA * A_k=2 * error%
        if k == 2:
            base_rate = 7e-6 * (TRUTH[k][0] * 1e6) * 1e-9 # nominal ~ 7e-14 rad/s
            # This is just a note, we stick to Delta A / A as it's cleaner.
            
        ax.set_title(f"Mode $k={k}$ (True $A={TRUTH[k][0]*1e6:.0f}\\mu$m)")
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    fig.suptitle(f"Systematics Error Budget via Beam Dynamics Simulation (N={N_MC} MC, CLEAN algorithm)", fontsize=12)
    fig.tight_layout()
    
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig6_combined_systematics.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"fig6_combined_systematics.png generated at {out_path}")

if __name__ == "__main__":
    make_fig6()
