"""
test6_fair_comparison.py — Test 6 (adil karşılaştırma)

Üç estimator AYNI gerçekçi senaryoda drift δq'yu kestirmeye çalışır:
  A: Analitik R + ΔR⁻¹ per-epoch absolute, sonra epoch farkı
  B: Analitik R + tek-gradient drift mode   R⁻¹(y(t) - y₀)
  C: Sayısal  R + ΔR⁻¹ per-epoch absolute, sonra epoch farkı (v2.7 yaklaşımı)

Senaryo (test_params.json test6):
  Statik misalignment: params.json'dan (~58 μm RMS uniform)
  Drift δq: DRIFT_RMS μm RMS rastgele
  BPM ofset: BPM_OFFSET μm RMS sabit
  Quad tilt: QUAD_TILT_MAX (modelde YOK)
  Dipol tilt: DIPOLE_TILT_MAX (modelde YOK)
  BPM gürültü: BPM_NOISE μm RMS her okumada

Sayısal R bir kez inşa edilir, R_num_*.npy dosyalarına cache'lenir.
Bir sonraki çalıştırmada cache yüklenir.

Kullanım:
    python test6_fair_comparison.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fodo_lattice import (
    compute_twiss_at_quads, signed_KL, build_response_matrix,
    calibrate_K_x_arc, direct_invert,
)
from reconstruct import generate_misalignments, run_simulation, EPS

with open("test_params.json", "r") as _f:
    _tp = json.load(_f)
_t6 = _tp["test6"]
BPM_NOISE        = float(_t6["BPM_NOISE"])
BPM_OFFSET       = float(_t6["BPM_OFFSET"])
DRIFT_RMS        = float(_t6["DRIFT_RMS"])
QUAD_TILT_MAX    = float(_t6["QUAD_TILT_MAX"])
DIPOLE_TILT_MAX  = float(_t6["DIPOLE_TILT_MAX"])
DRIFT_SEED       = int(_t6["DRIFT_SEED"])
OFFSET_SEED      = int(_t6["OFFSET_SEED"])
NOISE_SEED       = int(_t6["NOISE_SEED"])
QUAD_TILT_SEED   = int(_t6["QUAD_TILT_SEED"])
DIPOLE_TILT_SEED = int(_t6["DIPOLE_TILT_SEED"])
PERTURB_DELTA    = float(_t6["R_perturbation_delta"])


# =============================================================================
# Yanıt matrisleri
# =============================================================================
def build_R_analytic(config, g, plane, K_x_arc=None):
    cfg = dict(config); cfg['g1'] = g
    beta, phi, Q = compute_twiss_at_quads(cfg, plane, K_x_arc=K_x_arc)
    KL = signed_KL(cfg, plane)
    return build_response_matrix(beta, phi, Q, KL)


def build_R_numerical(config, g, plane, delta, cache_path):
    """48 quadı tek tek δ kadar oynat, finite-difference R inşa et.
    cache_path varsa onu yükler/kaydeder. Hata kaynakları (tilt, ofset,
    gürültü) BU İNŞADA YOKTUR — gerçek deneydeki LOCO benzeri kalibrasyon
    modelini taklit eder."""
    if cache_path and os.path.exists(cache_path):
        print(f"  [cache] yüklendi: {cache_path}")
        return np.load(cache_path)

    n_q = 2 * int(config['nFODO'])
    zero = np.zeros(n_q)
    print(f"  baseline (plane={plane}, g={g:.5f}) ...")
    x_b, y_b = run_simulation(config, g, zero, zero, zero, zero)
    y_b_pl = y_b if plane == 'y' else x_b

    R = np.zeros((n_q, n_q))
    for j in range(n_q):
        if j % 8 == 0:
            print(f"  quad {j}/{n_q} ...")
        d = np.zeros(n_q); d[j] = delta
        if plane == 'y':
            x_p, y_p = run_simulation(config, g, d, zero, zero, zero)
            y_p_pl = y_p
        else:
            x_p, y_p = run_simulation(config, g, zero, d, zero, zero)
            y_p_pl = x_p
        R[:, j] = (y_p_pl - y_b_pl) / delta

    if cache_path:
        np.save(cache_path, R)
        print(f"  [cache] kaydedildi: {cache_path}")
    return R


# =============================================================================
# Estimator değerlendirme
# =============================================================================
def evaluate_plane(plane_name, y_cal_n, y_cal_p, y_now_n, y_now_p,
                   R_an, dR_an, dR_num, true_drift):
    """Üç estimator: hepsi drift δq tahmini için."""
    # A: analitik ΔR, per-epoch absolute → epoch farkı
    q_now_A = direct_invert(dR_an, y_now_n - y_now_p)
    q_cal_A = direct_invert(dR_an, y_cal_n - y_cal_p)
    drift_A = q_now_A - q_cal_A

    # B: analitik R, tek-gradient drift mode
    drift_B = direct_invert(R_an, y_now_n - y_cal_n)

    # C: sayısal ΔR, per-epoch absolute → epoch farkı
    q_now_C = direct_invert(dR_num, y_now_n - y_now_p)
    q_cal_C = direct_invert(dR_num, y_cal_n - y_cal_p)
    drift_C = q_now_C - q_cal_C

    def stats(est):
        err = est - true_drift
        rms = float(np.sqrt(np.mean(err ** 2)))
        if np.std(est) > 1e-15 and np.std(true_drift) > 1e-15:
            corr = float(np.corrcoef(est, true_drift)[0, 1])
        else:
            corr = float('nan')
        return rms, corr

    return [
        ('A: Analitik ΔR (epoch farkı)',  drift_A, stats(drift_A)),
        ('B: Analitik R, drift mode',     drift_B, stats(drift_B)),
        ('C: Sayısal ΔR (v2.7 tarzı)',    drift_C, stats(drift_C)),
    ]


# =============================================================================
# Main
# =============================================================================
def main():
    with open("params.json", "r") as f:
        config = json.load(f)

    print("=" * 72)
    print("test6_fair_comparison.py — Test 6 (adil karşılaştırma)")
    print("=" * 72)
    print(f"BPM gürültüsü    : {BPM_NOISE*1e6:.1f} μm")
    print(f"BPM ofseti       : {BPM_OFFSET*1e6:.1f} μm (sabit)")
    print(f"Drift RMS        : {DRIFT_RMS*1e6:.1f} μm")
    print(f"Quad tilt max    : {QUAD_TILT_MAX*1e3:.2f} mrad")
    print(f"Dipol tilt max   : {DIPOLE_TILT_MAX*1e3:.2f} mrad")
    print(f"ε = Δg/g         : {EPS:.4f}")
    print(f"R perturbation δ : {PERTURB_DELTA*1e6:.1f} μm (sayısal R)")

    g_nom  = config['g1']
    g_pert = g_nom * (1.0 + EPS)
    n_q    = 2 * int(config['nFODO'])

    # 1. Hataları üret
    dy_static, dx_static, _, _ = generate_misalignments(config)

    rng_drift = np.random.default_rng(DRIFT_SEED)
    dy_drift = rng_drift.normal(0, DRIFT_RMS, n_q)
    dx_drift = rng_drift.normal(0, DRIFT_RMS, n_q)

    rng_off = np.random.default_rng(OFFSET_SEED)
    b_off_y = rng_off.normal(0, BPM_OFFSET, n_q)
    b_off_x = rng_off.normal(0, BPM_OFFSET, n_q)

    rng_qt = np.random.default_rng(QUAD_TILT_SEED)
    quad_tilt = rng_qt.uniform(-QUAD_TILT_MAX, QUAD_TILT_MAX, n_q)
    rng_dt = np.random.default_rng(DIPOLE_TILT_SEED)
    dip_tilt  = rng_dt.uniform(-DIPOLE_TILT_MAX, DIPOLE_TILT_MAX, n_q)

    rng_n = np.random.default_rng(NOISE_SEED)

    print(f"\nGerçek hata büyüklükleri:")
    print(f"  Statik dy/dx RMS : {np.std(dy_static)*1e6:.2f} / "
          f"{np.std(dx_static)*1e6:.2f} μm")
    print(f"  Drift  dy/dx RMS : {np.std(dy_drift)*1e6:.2f} / "
          f"{np.std(dx_drift)*1e6:.2f} μm")
    print(f"  BPM ofset RMS    : {np.std(b_off_y)*1e6:.2f} μm")
    print(f"  Quad tilt RMS    : {np.std(quad_tilt)*1e3:.3f} mrad")
    print(f"  Dipol tilt RMS   : {np.std(dip_tilt)*1e3:.3f} mrad")

    # 2. 4 simülasyon: cal/now × g_nom/g_pert
    print(f"\n--- Veri simülasyonları (4 koşum) ---")
    print(f"[1/4] cal  g_nom  (Δq = static, tilt'ler dahil)")
    x_cal_n, y_cal_n = run_simulation(config, g_nom,  dy_static, dx_static,
                                       quad_tilt, dip_tilt)
    print(f"[2/4] cal  g_pert")
    x_cal_p, y_cal_p = run_simulation(config, g_pert, dy_static, dx_static,
                                       quad_tilt, dip_tilt)
    print(f"[3/4] now  g_nom  (Δq = static + drift)")
    x_now_n, y_now_n = run_simulation(config, g_nom,
                                       dy_static+dy_drift, dx_static+dx_drift,
                                       quad_tilt, dip_tilt)
    print(f"[4/4] now  g_pert")
    x_now_p, y_now_p = run_simulation(config, g_pert,
                                       dy_static+dy_drift, dx_static+dx_drift,
                                       quad_tilt, dip_tilt)

    # 3. BPM ofset + gürültü uygula
    def add_errors(y, b):
        return y + b + rng_n.normal(0, BPM_NOISE, len(y))

    y_cal_n_m = add_errors(y_cal_n, b_off_y)
    y_cal_p_m = add_errors(y_cal_p, b_off_y)
    y_now_n_m = add_errors(y_now_n, b_off_y)
    y_now_p_m = add_errors(y_now_p, b_off_y)
    x_cal_n_m = add_errors(x_cal_n, b_off_x)
    x_cal_p_m = add_errors(x_cal_p, b_off_x)
    x_now_n_m = add_errors(x_now_n, b_off_x)
    x_now_p_m = add_errors(x_now_p, b_off_x)

    # 4. Analitik R'ler
    print(f"\n--- Analitik R'ler ---")
    K_x_arc = calibrate_K_x_arc(config)
    R_y_an_n = build_R_analytic(config, g_nom,  'y')
    R_y_an_p = build_R_analytic(config, g_pert, 'y')
    R_x_an_n = build_R_analytic(config, g_nom,  'x', K_x_arc=K_x_arc)
    R_x_an_p = build_R_analytic(config, g_pert, 'x', K_x_arc=K_x_arc)
    dR_y_an = R_y_an_n - R_y_an_p
    dR_x_an = R_x_an_n - R_x_an_p
    print(f"  κ(R_y) = {np.linalg.cond(R_y_an_n):.1f}  "
          f"κ(ΔR_y_an) = {np.linalg.cond(dR_y_an):.1f}")
    print(f"  κ(R_x) = {np.linalg.cond(R_x_an_n):.1f}  "
          f"κ(ΔR_x_an) = {np.linalg.cond(dR_x_an):.1f}")

    # 5. Sayısal R'ler (cache'li)
    print(f"\n--- Sayısal R'ler (LOCO benzeri, cache'li) ---")
    R_y_num_n = build_R_numerical(config, g_nom,  'y', PERTURB_DELTA,
                                   "R_num_y_nom.npy")
    R_y_num_p = build_R_numerical(config, g_pert, 'y', PERTURB_DELTA,
                                   "R_num_y_pert.npy")
    R_x_num_n = build_R_numerical(config, g_nom,  'x', PERTURB_DELTA,
                                   "R_num_x_nom.npy")
    R_x_num_p = build_R_numerical(config, g_pert, 'x', PERTURB_DELTA,
                                   "R_num_x_pert.npy")
    dR_y_num = R_y_num_n - R_y_num_p
    dR_x_num = R_x_num_n - R_x_num_p
    print(f"  κ(ΔR_y_num) = {np.linalg.cond(dR_y_num):.1f}")
    print(f"  κ(ΔR_x_num) = {np.linalg.cond(dR_x_num):.1f}")
    rel_diff_y = np.linalg.norm(R_y_an_n - R_y_num_n) / np.linalg.norm(R_y_num_n)
    rel_diff_x = np.linalg.norm(R_x_an_n - R_x_num_n) / np.linalg.norm(R_x_num_n)
    print(f"  ‖R_an - R_num‖/‖R_num‖ : y={rel_diff_y:.3%}  x={rel_diff_x:.3%}")

    # 6. Değerlendirme
    res_y = evaluate_plane('y', y_cal_n_m, y_cal_p_m, y_now_n_m, y_now_p_m,
                           R_y_an_n, dR_y_an, dR_y_num, dy_drift)
    res_x = evaluate_plane('x', x_cal_n_m, x_cal_p_m, x_now_n_m, x_now_p_m,
                           R_x_an_n, dR_x_an, dR_x_num, dx_drift)

    print(f"\n" + "=" * 72)
    print(f"DRİFT TAHMİNİ — gerçek: dy={np.std(dy_drift)*1e6:.2f} μm, "
          f"dx={np.std(dx_drift)*1e6:.2f} μm")
    print("=" * 72)
    for plane_name, results in [('y', res_y), ('x', res_x)]:
        print(f"\n========== Düzlem: {plane_name} ==========")
        print(f"  {'Estimator':35s}  {'RMS [μm]':>10s}  {'corr':>8s}")
        print("  " + "-" * 60)
        for name, _, (rms, corr) in results:
            print(f"  {name:35s}  {rms*1e6:10.2f}  {corr:8.4f}")

    # 7. Grafik
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, (plane_name, results, true_drift) in enumerate([
            ('y', res_y, dy_drift), ('x', res_x, dx_drift)]):
        for col, (name, est, (rms, corr)) in enumerate(results):
            ax = axes[row, col]
            ax.plot(true_drift*1e6, est*1e6, 'o', alpha=0.6, markersize=5)
            lim = max(np.max(np.abs(true_drift)),
                      np.max(np.abs(est))) * 1.2 * 1e6
            ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.4, lw=0.8)
            ax.set_xlabel(f"Gerçek δq_{plane_name} [μm]")
            ax.set_ylabel("Tahmin [μm]")
            ax.set_title(f"{name}\nRMS={rms*1e6:.2f}μm, corr={corr:.3f}",
                         fontsize=9)
            ax.grid(True, alpha=0.3)
    fig.suptitle("Test 6: A (analitik ΔR) vs B (drift mode) vs C (sayısal ΔR)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("test6_fair_comparison.png", dpi=140)
    print("\nKaydedildi: test6_fair_comparison.png")


if __name__ == "__main__":
    main()
