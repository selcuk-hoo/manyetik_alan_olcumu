#!/usr/bin/env python3
"""make_fig8.py — Figure 8: Pure BPM Noise Systematic Analysis

Yöntem: 
- K=1, 2, 3 modlarında A=10 um dikey quad_dy kayması uygulanır.
- Makine idealdir (gradyan hatası, rulo hatası vs. yoktur).
- Farklı BPM gürültü seviyeleri (RMS = 0, 1, 10, 50 um) uygulanır.
- CLEAN kullanılarak genlik geri çatılır.
- Noise=0 duruma göre genlikteki % artış (hata) çizilir.
"""

import os
import json
import math
import numpy as np
import multiprocessing as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_plot_utils import apply_paper_style, load_R, BLUE
from fourier_reconstruct import fodo_basis, clean_reconstruct
from build_response_matrix import setup_fields, run_sim, read_cod_quads

_config = None
_R_dy = None
_n_q = None
_antisym = None

def _worker_init(config, R_mat, n_q_val, antisym):
    global _config, _R_dy, _n_q, _antisym
    _config = config
    _R_dy = R_mat
    _n_q = n_q_val
    _antisym = antisym

def _simulate_noise_error(task):
    b_sig, n_mc, target_k = task
    
    A_true = 10e-6
    F_target, _ = fodo_basis(_n_q, [target_k], _antisym)
    quad_dy = A_true * F_target[:, 0]  # cos phase
    
    alanlar, state0 = setup_fields(_config)
    
    # We don't need a temporary directory because we don't need to run 
    # run_sim repeatedly in the MC loop! The true physical orbit is perfectly static!
    # We can just get the baseline orbit once.
    # Actually, because workers are in parallel, each worker will run one sim for its task.
    # To be safe from file collision, let's use tempdir.
    import tempfile, shutil
    old_dir = os.getcwd()
    tmp_dir = tempfile.mkdtemp(prefix=f"sim_fig8_k{target_k}_b{b_sig}_")
    os.chdir(tmp_dir)
    
    try:
        run_sim(alanlar, state0, _config, quad_dy=quad_dy, quad_dx=np.zeros(_n_q), quad_tilt=np.zeros(_n_q))
        _, y_base = read_cod_quads(alanlar.nFODO)
    finally:
        os.chdir(old_dir)
        shutil.rmtree(tmp_dir)
        
    # Consistent noise generator based on target_k
    rng_noise = np.random.default_rng(int(b_sig * 1e6) + target_k * 1337)
    errors = []
    
    # Base reconstruct (noise=0)
    accum_b, _, F_cache_b = clean_reconstruct([_R_dy], [y_base], list(range(1, 11)), _antisym)
    a_kb = accum_b[target_k]
    db = {kind: a_kb[i] for i, (_, kind) in enumerate(F_cache_b[target_k][1])}
    A_fit_base = math.sqrt(db.get('cos',0)**2 + db.get('sin',0)**2)
    
    for _ in range(n_mc):
        noise = rng_noise.normal(0, b_sig, _n_q) if b_sig > 0 else np.zeros(_n_q)
        y_meas = y_base + noise
        
        accum_p, _, F_cache_p = clean_reconstruct([_R_dy], [y_meas], list(range(1, 11)), _antisym)
        a_kp = accum_p[target_k]
        dp = {kind: a_kp[i] for i, (_, kind) in enumerate(F_cache_p[target_k][1])}
        A_fit_pert = math.sqrt(dp.get('cos',0)**2 + dp.get('sin',0)**2)
        
        err_pct = (A_fit_pert - A_fit_base) / A_fit_base * 100
        errors.append(err_pct)
        
    return target_k, b_sig, np.mean(errors), np.std(errors)

def make_fig8():
    apply_paper_style()
    
    with open("params.json") as f:
        config = json.load(f)
    config["t2"] = 1.0e-3
    n_q = 2 * int(config.get("nFODO", 24))
    antisym = config.get("smooth_antisym_fodo", True)
    
    try:
        R_dy = np.load("R_dy_1.npy")
    except:
        from paper_plot_utils import load_R
        R_dy = load_R()
    
    b_sigs = [0.0, 1e-6, 10e-6, 50e-6]
    n_mc_per_point = 100 # We can do 100 since it's very fast
    k_targets = [1, 2, 3]
    
    tasks = []
    for target_k in k_targets:
        for b_sig in b_sigs:
            n_mc = n_mc_per_point if b_sig > 0 else 1
            tasks.append((b_sig, n_mc, target_k))
                
    cache_file = "make_fig8_cache.json"
    if os.path.exists(cache_file):
        print(f"Loading cached spin tracking data from {cache_file}...")
        with open(cache_file, 'r') as f:
            results = json.load(f)
    else:
        print(f"Running pure BPM Noise simulations using 12 cores...")
        ctx = mp.get_context("spawn")
        with ctx.Pool(12, initializer=_worker_init, initargs=(config, R_dy, n_q, antisym)) as pool:
            raw_results = pool.map(_simulate_noise_error, tasks)
        
        # Save to cache
        results = [{"k": r[0], "b_sig": r[1], "mean": r[2], "std": r[3]} for r in raw_results]
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2)
            
    # Group results
    data = {k: {"x": [], "mean": [], "std": []} for k in k_targets}
    for r in results:
        k = int(r["k"])
        b_sig = r["b_sig"]
        data[k]["x"].append(b_sig * 1e6) # plot in um
        data[k]["mean"].append(r["mean"])
        data[k]["std"].append(r["std"])
        
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {1: BLUE, 2: "red", 3: "gray"}
    markers = {1: "o", 2: "s", 3: "D"}
    
    for k in k_targets:
        x = data[k]["x"]
        y = data[k]["mean"]
        yerr = data[k]["std"]
        
        ax.errorbar(x, y, yerr=yerr, fmt=markers[k]+"-", color=colors[k], 
                    capsize=4, markersize=6, label=f"Mode $k={k}$")
            
    ax.set_xlabel(r"BPM Noise $\sigma_b$ [$\mu$m]")
    ax.set_ylabel(r"Relative Amplitude Error $\Delta A / A_0$ [%]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
            
    fig.suptitle("Error budget: Pure BPM Noise", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig8_noise_model.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"fig8_noise_model.png generated at {out_path}")

if __name__ == "__main__":
    make_fig8()
