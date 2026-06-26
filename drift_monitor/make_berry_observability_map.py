import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

_D = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_D)
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.join(_PROJ, "berry_data"))
os.chdir(_PROJ)

from drift_monitor.fodo_lattice import compute_twiss_at_quads, signed_KL, build_response_matrix
from false_edm_4d import find_co_4d, setup_fields, _make_state, _T_rev
from integrator import integrate_particle

def get_berry_weights(N=48, lam=10):
    d1 = np.load(os.path.join(_PROJ, "berry_data", "run1_data.npz"))
    d2 = np.load(os.path.join(_PROJ, "berry_data", "run2_data.npz"))
    f_val = np.concatenate([d1['f'], d2['f']])
    xo = np.vstack([d1['xo'], d2['xo']])
    yo = np.vstack([d1['yo'], d2['yo']])
    M, L = xo.shape
    idx = np.linspace(0, L - 1, N).astype(int)
    X = xo[:, idx] * yo[:, idx]
    mu = X.mean(0)
    sd = X.std(0) + 1e-30
    Xs = (X - mu) / sd
    w_norm = np.linalg.solve(Xs.T @ Xs + lam * np.eye(N), Xs.T @ (f_val - f_val.mean()))
    w_raw = w_norm / sd
    bias = f_val.mean() - mu @ w_raw
    return w_raw, bias, idx

def s_hat(arr):
    qf = arr[0::2]
    qd = arr[1::2]
    s = 0.5 * (qf + qd)
    d = 0.5 * (qf - qd)
    num = np.sum(s ** 2) - np.sum(d ** 2)
    den = np.sum(s ** 2) + np.sum(d ** 2)
    return num / den if den > 0 else 0.0

def evaluate_mode(task):
    i, mode, sigma_i, w_raw, bias, sample_idx = task
    
    # Her worker için ayrı setup
    import os, json, sys, numpy as np
    sys.path.insert(0, _PROJ)
    sys.path.insert(0, os.path.join(_PROJ, "berry_data"))
    os.chdir(_PROJ)
    from false_edm_4d import find_co_4d, setup_fields, _make_state, _T_rev
    from integrator import integrate_particle
    
    with open("params.json", "r") as f:
        config = json.load(f)
        
    fields, y0, beta0, R0, p_mag, direction = setup_fields(config)
    T_rev = _T_rev(fields, beta0, R0)
    DT = config["dt"]
    NQ = len(mode)
    
    # Her iki düzlemde de kaçıklık olsun (gerçek sinyal için şart)
    dy = mode * 10e-6
    dx = mode * 10e-6
    
    # n_iter=1 ile yaklaşık kapalı yörüngeyi bul (daha hızlı)
    v_co, resid = find_co_4d(fields, p_mag, direction, dx, dy, np.zeros(NQ), T_rev, n_iter=1)
    launch = _make_state(v_co, p_mag, direction, [0, 0, direction])
    
    fields.poincare_quad_index = -1.0
    hist, _, _ = integrate_particle(launch, 0.0, T_rev, DT, fields=fields, return_steps=480, quad_dx=dx, quad_dy=dy)
    
    xo = hist[:, 0]
    yo = hist[:, 1]
    
    X_mode = xo[sample_idx] * yo[sample_idx]
    f_pred = np.dot(X_mode, w_raw) + bias
    sh = s_hat(dy)
    
    return i, sigma_i, abs(f_pred), sh

def main():
    print("Berry ağırlıkları öğreniliyor...")
    N_points = 48
    w_raw, bias, sample_idx = get_berry_weights(N=N_points, lam=10)
    
    with open("params.json", "r") as f:
        config = json.load(f)
        
    print("Analitik R matrisi ve SVD hesaplanıyor...")
    beta, phi, Q = compute_twiss_at_quads(config, 'y')
    KL = signed_KL(config, 'y')
    R = build_response_matrix(beta, phi, Q, KL)
    U, S, Vt = np.linalg.svd(R)
    NQ = len(beta)
    
    tasks = [(i, Vt[i, :], S[i], w_raw, bias, sample_idx) for i in range(NQ)]
    
    observability = np.zeros(NQ)
    false_edm = np.zeros(NQ)
    s_hats = np.zeros(NQ)
    
    print("Her mod için tracker koşturuluyor (Paralel 10 worker)...")
    done = 0
    with ProcessPoolExecutor(10) as pool:
        for i, sig, f_pred, sh in pool.map(evaluate_mode, tasks):
            observability[i] = sig
            false_edm[i] = f_pred
            s_hats[i] = sh
            done += 1
            print(f"Mod {i:2d} tamamlandı ({done}/{NQ}) | Sigma={sig:.2e} | |f_pred|={f_pred:.2e} | S_hat={sh:+.2f}")
    
    plt.figure(figsize=(8, 6))
    
    mask_sym = s_hats > 0.0
    mask_anti = ~mask_sym
    
    plt.scatter(observability[mask_anti], false_edm[mask_anti], c='red', label='Antisimetrik (Gözlenebilir Sürücüler)', alpha=0.7, edgecolors='k')
    plt.scatter(observability[mask_sym], false_edm[mask_sym], c='blue', marker='s', label='Simetrik (Kör Artıklar)', alpha=0.7, edgecolors='k')
    
    plt.xscale('log')
    plt.yscale('log')
    
    med_obs = np.median(observability)
    med_edm = np.median(false_edm)
    plt.axvline(med_obs, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(med_edm, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel(r'Gözlenebilirlik $\sigma_i$ (SVD Özdeğeri)')
    plt.ylabel(r'Sahte-EDM Katkısı $|f_{pred}|$ [rad/s] (Berry Öngörüsü)')
    plt.title(r'Mod-Bazlı Gözlenebilirlik vs Sahte-EDM Haritası\n(10 $\mu$m RMS quad hizalama hatası)')
    
    plt.text(observability.max(), false_edm.max(), 'İZLE\n(Yüksek-EDM, Gözlenebilir)', ha='right', va='top', fontsize=9, alpha=0.7, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(observability.min(), false_edm.max(), 'İRREDÜSİBL ARTIK\n(Yüksek-EDM, Kör)', ha='left', va='top', fontsize=9, alpha=0.7, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(_D, 'berry_observability_map.png'), dpi=150)
    print("Grafik kaydedildi: berry_observability_map.png")

if __name__ == "__main__":
    main()
