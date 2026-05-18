"""
compare_regularization.py — Test 1 (yapilacaklar-2.md §Test 1)

Aynı simülasyon verisi üzerinde altı estimator'ı karşılaştırır:
  1. Direct R1⁻¹·y1
  2. Direct R2⁻¹·y2
  3. Direct ortalama (v1+v2)/2
  4. Ham ΔR⁻¹·(y1-y2)
  5. Tikhonov ΔR (lambda taraması + L-curve optimumu)
  6. TSVD ΔR (truncation k taraması)

Çıktı:
  • Konsol tablosu: her estimator için RMS, max, korelasyon (y ve x düzlemi)
  • PNG 1: Tikhonov L-curve (her düzlem)
  • PNG 2: TSVD scree (RMS vs k, her düzlem)

reconstruct.py'deki simülasyon kurulumu yeniden kullanılır.

Kullanım:
    python compare_regularization.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fodo_lattice import (
    compute_twiss_at_quads,
    signed_KL,
    build_response_matrix,
    calibrate_K_x_arc,
    direct_invert,
)
from reconstruct import generate_misalignments, run_simulation, EPS, T_END

with open("test_params.json", "r") as _f:
    _tp = json.load(_f)
_t1 = _tp["test1"]
LAM_LOG_MIN = float(_t1["lambda_grid_log_min"])
LAM_LOG_MAX = float(_t1["lambda_grid_log_max"])
LAM_GRID_N  = int(_t1["lambda_grid_n"])


# =============================================================================
# Estimator'lar
# =============================================================================
def tikhonov(R, y, lam):
    """(RᵀR + λI)⁻¹ Rᵀ y."""
    N = R.shape[0]
    A = R.T @ R + lam * np.eye(N)
    return np.linalg.solve(A, R.T @ y)


def tsvd(R, y, k):
    """En büyük k singüler değeri tut, geri kalanını sıfırla."""
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    s_inv = np.zeros_like(s)
    s_inv[:k] = 1.0 / s[:k]
    return Vt.T @ (s_inv * (U.T @ y))


def metrics(dq_hat, dq_true):
    err = dq_hat - dq_true
    rms = np.sqrt(np.mean(err ** 2))
    mx = np.max(np.abs(err))
    if np.std(dq_hat) > 0 and np.std(dq_true) > 0:
        cr = np.corrcoef(dq_hat, dq_true)[0, 1]
    else:
        cr = np.nan
    return rms, mx, cr


# =============================================================================
# Yanıt matrisleri
# =============================================================================
def build_R(config, g, plane, K_x_arc=None):
    cfg = dict(config); cfg['g1'] = g
    beta, phi, Q = compute_twiss_at_quads(cfg, plane, K_x_arc=K_x_arc)
    KL = signed_KL(cfg, plane)
    return build_response_matrix(beta, phi, Q, KL)


# =============================================================================
# Tikhonov optimum (L-curve köşesi: en yüksek eğrilik)
# =============================================================================
def lcurve_optimal_lambda(R, y, lam_grid):
    """L-curve köşe noktası: log-log eğride maksimum eğrilik."""
    eta = np.empty_like(lam_grid)   # ‖dq_hat‖
    rho = np.empty_like(lam_grid)   # ‖R·dq_hat - y‖
    for i, lam in enumerate(lam_grid):
        dq_hat = tikhonov(R, y, lam)
        eta[i] = np.linalg.norm(dq_hat)
        rho[i] = np.linalg.norm(R @ dq_hat - y)
    # log uzayında köşe: ikinci türev maksimumu
    x = np.log(rho + 1e-30)
    z = np.log(eta + 1e-30)
    dx = np.gradient(x); dz = np.gradient(z)
    ddx = np.gradient(dx); ddz = np.gradient(dz)
    curv = (dx * ddz - dz * ddx) / (dx**2 + dz**2 + 1e-30) ** 1.5
    i_opt = int(np.argmax(curv[2:-2])) + 2
    return lam_grid[i_opt], i_opt, rho, eta


def tsvd_optimal_k(R, y, dq_true):
    """Gerçek dq_true ile karşılaştırarak optimum k (oracle).
    Gerçek deneyde dq_true bilinmediği için bu üst-sınır referansıdır.
    """
    N = R.shape[0]
    best_rms = np.inf; best_k = N
    rms_list = []
    for k in range(1, N + 1):
        dq_hat = tsvd(R, y, k)
        rms = np.sqrt(np.mean((dq_hat - dq_true) ** 2))
        rms_list.append(rms)
        if rms < best_rms:
            best_rms = rms; best_k = k
    return best_k, best_rms, np.array(rms_list)


# =============================================================================
# Düzlem başına analiz
# =============================================================================
def analyze_plane(plane, dq_true, y1, y2, config, axL, axS):
    K_x_arc = calibrate_K_x_arc(config) if plane == 'x' else None
    g_nom = config['g1']
    g_pert = g_nom * (1.0 + EPS)
    R1 = build_R(config, g_nom,  plane, K_x_arc)
    R2 = build_R(config, g_pert, plane, K_x_arc)
    dR = R1 - R2
    dy = y1 - y2

    print(f"\n========== Düzlem: {plane} ==========")
    print(f"  κ(R1)={np.linalg.cond(R1):.1f}  κ(R2)={np.linalg.cond(R2):.1f}  "
          f"κ(ΔR)={np.linalg.cond(dR):.1f}")

    results = []

    # 1-3. Direct estimator'lar
    v1 = direct_invert(R1, y1)
    v2 = direct_invert(R2, y2)
    avg = 0.5 * (v1 + v2)
    results.append(("Direct R1⁻¹·y1",   v1))
    results.append(("Direct R2⁻¹·y2",   v2))
    results.append(("Direct (v1+v2)/2", avg))

    # 4. Ham ΔR
    dq_raw = direct_invert(dR, dy)
    results.append(("Ham ΔR⁻¹",         dq_raw))

    # 5. Tikhonov ΔR — L-curve
    lam_grid = np.logspace(LAM_LOG_MIN, LAM_LOG_MAX, LAM_GRID_N)
    lam_opt, i_opt, rho, eta = lcurve_optimal_lambda(dR, dy, lam_grid)
    dq_tik = tikhonov(dR, dy, lam_opt)
    results.append((f"Tikhonov ΔR (λ={lam_opt:.2e})", dq_tik))

    # Oracle Tikhonov (gerçek dq_true ile en iyi λ)
    rms_lam = [np.sqrt(np.mean((tikhonov(dR, dy, lam) - dq_true)**2))
               for lam in lam_grid]
    lam_oracle = lam_grid[int(np.argmin(rms_lam))]
    dq_tik_oracle = tikhonov(dR, dy, lam_oracle)
    results.append((f"Tikhonov ΔR oracle (λ={lam_oracle:.2e})", dq_tik_oracle))

    # 6. TSVD ΔR
    k_opt, _, rms_k = tsvd_optimal_k(dR, dy, dq_true)
    dq_tsvd = tsvd(dR, dy, k_opt)
    results.append((f"TSVD ΔR oracle (k={k_opt})", dq_tsvd))

    # Konsol tablosu
    print()
    print(f"  {'Estimator':40s}  {'RMS [μm]':>10s}  {'max [μm]':>10s}  {'corr':>8s}")
    print("  " + "-" * 75)
    for name, dq_hat in results:
        rms, mx, cr = metrics(dq_hat, dq_true)
        print(f"  {name:40s}  {rms*1e6:10.3f}  {mx*1e6:10.3f}  {cr:8.4f}")

    # L-curve grafiği
    axL.loglog(rho, eta, '-o', markersize=2)
    axL.loglog(rho[i_opt], eta[i_opt], 'rs', markersize=10,
               label=f'L-curve corner\nλ={lam_opt:.2e}')
    axL.set_xlabel(r"$\|\Delta R \cdot \widehat{\Delta q} - (y_1 - y_2)\|$")
    axL.set_ylabel(r"$\|\widehat{\Delta q}\|$")
    axL.set_title(f"L-curve, plane = {plane}")
    axL.legend(fontsize=8); axL.grid(True, alpha=0.3)

    # TSVD scree
    axS.semilogy(np.arange(1, len(rms_k) + 1), rms_k * 1e6, '-o', markersize=3)
    axS.axvline(k_opt, color='r', ls='--', label=f'optimum k={k_opt}')
    axS.set_xlabel("Truncation level k")
    axS.set_ylabel("RMS reconstruction error [μm]")
    axS.set_title(f"TSVD scree, plane = {plane}")
    axS.legend(fontsize=8); axS.grid(True, alpha=0.3)

    return results


# =============================================================================
# Main
# =============================================================================
def main():
    with open("params.json", "r") as f:
        config = json.load(f)

    print("=" * 72)
    print("compare_regularization.py — Test 1")
    print("=" * 72)
    print(f"EPS={EPS:.4f}  T_END={T_END:.2e}  λ-grid=[10^{LAM_LOG_MIN:.0f}..10^{LAM_LOG_MAX:.0f}, "
          f"n={LAM_GRID_N}]  (test_params.json)")

    # Misalignments ve simülasyonlar (reconstruct.py ile aynı)
    dy_q, dx_q, dt_q, dip_t = generate_misalignments(config)
    g_nom = config['g1']; g_pert = g_nom * (1.0 + EPS)

    print(f"\n[1/2] Sim g_nom = {g_nom:.5f} T/m ...")
    x1, y1 = run_simulation(config, g_nom,  dy_q, dx_q, dt_q, dip_t)
    print(f"[2/2] Sim g_pert = {g_pert:.5f} T/m ...")
    x2, y2 = run_simulation(config, g_pert, dy_q, dx_q, dt_q, dip_t)

    # 2x2 figür: y satırı (L-curve, TSVD), x satırı (L-curve, TSVD)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    analyze_plane('y', dy_q, y1, y2, config, axes[0, 0], axes[0, 1])
    analyze_plane('x', dx_q, x1, x2, config, axes[1, 0], axes[1, 1])
    fig.tight_layout()
    fig.savefig("test1_regularization.png", dpi=130)
    print("\nKaydedildi: test1_regularization.png")


if __name__ == "__main__":
    main()
