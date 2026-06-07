#!/usr/bin/env python3
"""make_fig4.py — Figure 4: Systematic Gradient Error Analysis via Full Spin Tracking

Yöntem: 
- Bütün quad'lar %0.1, %0.3, %1.0, %3.0 oranında SİSTEMATİK olarak artırılır.
- K=1, 2, 3 modlarında A=10 um dikey quad_dy kayması uygulanır.
- Full C++ integrator ile kapalı yörünge elde edilir.
- Nominal R matrisi ve CLEAN kullanılarak genlik geri çatılır.
- İdeal non-linear taban (G=0) duruma göre genlikteki % artış çizilir.
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

from paper_plot_utils import apply_paper_style, load_R, BLUE, RED, GRAY
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

def _simulate_gradient_error(task):
    """
    1. K hedefini oluştur.
    2. G=0 (base) ve G=sig_m için simülasyonları koş.
    3. Gürültü ekle, CLEAN ile A_fit bul.
    4. Yüzde değişimi döndür: (A_G - A_base) / A_base * 100
    """
    sig_m, b_sig, n_mc, target_k = task
    
    A_true = 10e-6
    F_target, _ = fodo_basis(_n_q, [target_k], _antisym)
    quad_dy = A_true * F_target[:, 0]  # cos phase
    
    alanlar, state0 = setup_fields(_config)
    g1_nom = _config.get("g1", 0.21)
    
    old_dir = os.getcwd()
    tmp_dir = tempfile.mkdtemp(prefix=f"sim_fig4_k{target_k}_g{sig_m}_")
    os.chdir(tmp_dir)
    
    try:
        # Base orbit (G=0)
        run_sim(alanlar, state0, _config, quad_dy=quad_dy, quad_dx=np.zeros(_n_q), quad_dG=np.zeros(_n_q))
        _, y_base = read_cod_quads(alanlar.nFODO)
        
        # Perturbed orbit
        quad_dG = np.ones(_n_q) * (g1_nom * sig_m)
        run_sim(alanlar, state0, _config, quad_dy=quad_dy, quad_dx=np.zeros(_n_q), quad_dG=quad_dG)
        _, y_pert = read_cod_quads(alanlar.nFODO)
        
    finally:
        os.chdir(old_dir)
        shutil.rmtree(tmp_dir)
        
    # Consistent noise generator based on target_k and b_sig
    rng_noise = np.random.default_rng(int(b_sig * 1e6) + target_k * 1337)
    errors = []
    
    for _ in range(n_mc):
        noise = rng_noise.normal(0, b_sig, _n_q) if b_sig > 0 else np.zeros(_n_q)
        
        y_b = y_base + noise
        y_p = y_pert + noise
        
        # Base reconstruct
        accum_b, _, F_cache_b = clean_reconstruct([_R_dy], [y_b], list(range(1, 11)), _antisym)
        a_kb = accum_b[target_k]
        db = {kind: a_kb[i] for i, (_, kind) in enumerate(F_cache_b[target_k][1])}
        A_fit_base = math.sqrt(db.get('cos',0)**2 + db.get('sin',0)**2)
        
        # Perturbed reconstruct
        accum_p, _, F_cache_p = clean_reconstruct([_R_dy], [y_p], list(range(1, 11)), _antisym)
        a_kp = accum_p[target_k]
        dp = {kind: a_kp[i] for i, (_, kind) in enumerate(F_cache_p[target_k][1])}
        A_fit_pert = math.sqrt(dp.get('cos',0)**2 + dp.get('sin',0)**2)
        
        err_pct = (A_fit_pert - A_fit_base) / A_fit_base * 100
        errors.append(err_pct)
        
    return target_k, sig_m, b_sig, np.mean(errors), np.std(errors)

def make_fig4():
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
    
    sig_m_vals = [0.001, 0.003, 0.01, 0.03]
    b_sigs = [0.0]
    n_mc_per_point = 50
    k_targets = [1, 2, 3]
    
    tasks = []
    for target_k in k_targets:
        for b_sig in b_sigs:
            for sig_m in sig_m_vals:
                n_mc = n_mc_per_point if b_sig > 0 else 1
                tasks.append((sig_m, b_sig, n_mc, target_k))
                
    cache_file = "make_fig4_cache.json"
    if os.path.exists(cache_file):
        print(f"Loading cached spin tracking data from {cache_file}...")
        with open(cache_file, 'r') as f:
            results = json.load(f)
    else:
        print(f"Running systematic gradient simulations using 12 cores...")
        ctx = mp.get_context("spawn")
        with ctx.Pool(12, initializer=_worker_init, initargs=(config, R_dy, n_q, antisym)) as pool:
            raw_results = pool.map(_simulate_gradient_error, tasks)
        
        # Save to cache
        results = [{"k": r[0], "sig_m": r[1], "b_sig": r[2], "mean": r[3], "std": r[4]} for r in raw_results]
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2)
            
    # Group results
    data = {k: {b: {"x": [], "mean": [], "std": []} for b in b_sigs} for k in k_targets}
    for r in results:
        k = int(r["k"])
        sig_m, b_sig = r["sig_m"], r["b_sig"]
        if b_sig not in b_sigs: continue
        data[k][b_sig]["x"].append(sig_m * 100)
        data[k][b_sig]["mean"].append(r["mean"])
        data[k][b_sig]["std"].append(r["std"])
        
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {1: BLUE, 2: RED, 3: GRAY}
    markers = {1: "o", 2: "s", 3: "D"}
    
    for k in k_targets:
        x = data[k][0.0]["x"]
        y = data[k][0.0]["mean"]
        yerr = data[k][0.0]["std"]
        
        ax.errorbar(x, y, yerr=yerr, fmt=markers[k]+"-", color=colors[k], 
                    capsize=4, markersize=6, label=f"Mode $k={k}$")
            
    ax.set_xlabel(r"Systematic Gradient Increase $\delta K/K$ [%]")
    ax.set_ylabel(r"Relative Amplitude Error $\Delta A / A_0$ [%]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
            
    fig.suptitle("Error budget: Systematic Gradient Shift", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig4_sigma_model.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"fig4_sigma_model.png generated at {out_path}")

if __name__ == "__main__":
    make_fig4()
