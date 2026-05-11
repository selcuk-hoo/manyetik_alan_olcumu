#!/usr/bin/env python3
"""
scan_qtilt_contamination.py

quad_random_tilt_max'i tarayarak k-mod geri catiminin (dy, dx) ne kadar
bozuldugunu nicel olarak olcer.

Tepki matrisi quad_tilt'i MODELLEMEZ. Bu yuzden gercek makinede tilt
varsa Δy ve Δx sinyalinde modellenmemis bir kirlilik bulunur; geri catim
RMS hatasi ve korelasyon bunu acik bir sekilde ortaya koyar.

On kosul: build_response_matrix.py calistirilmis (R_dy_1/2, R_dx_1/2 mevcut).

Cikti:
  scan_qtilt_contamination.csv
  scan_qtilt_contamination.png
"""
import json
import os
import time
import numpy as np

from build_response_matrix import setup_fields, run_sim

BASE = os.path.dirname(os.path.abspath(__file__))


def reconstruct_one(config, theta_max, dy_true, dx_true,
                    g1_nom, g1_pert, dR_dy, dR_dx, seed=44):
    """Tek theta_max icin iki konfigurasyon kosumu + geri catim."""
    n_q = len(dy_true)

    if theta_max > 0:
        rng = np.random.default_rng(seed)
        qtilt = rng.uniform(-theta_max, theta_max, n_q)
    else:
        qtilt = np.zeros(n_q)

    # Nominal optik
    a1, s1 = setup_fields(config, g1_override=g1_nom)
    x1, y1 = run_sim(a1, s1, config, dy_true, dx_true,
                     dipole_tilt=None, quad_tilt=qtilt)
    # Perturbe optik
    a2, s2 = setup_fields(config, g1_override=g1_pert)
    x2, y2 = run_sim(a2, s2, config, dy_true, dx_true,
                     dipole_tilt=None, quad_tilt=qtilt)

    dy_meas = y2 - y1
    dx_meas = x2 - x1

    dy_rec = np.linalg.solve(dR_dy, dy_meas)
    dx_rec = np.linalg.solve(dR_dx, dx_meas)

    err_dy = dy_rec - dy_true
    err_dx = dx_rec - dx_true
    rms_dy = float(np.std(err_dy))
    rms_dx = float(np.std(err_dx))
    corr_dy = float(np.corrcoef(dy_true, dy_rec)[0, 1])
    corr_dx = float(np.corrcoef(dx_true, dx_rec)[0, 1])
    return rms_dy, rms_dx, corr_dy, corr_dx


def main():
    os.chdir(BASE)
    with open("params.json") as f:
        config = json.load(f)

    for fn in ("R_dy_1.npy", "R_dy_2.npy", "R_dx_1.npy", "R_dx_2.npy"):
        if not os.path.exists(fn):
            raise FileNotFoundError(
                f"{fn} bulunamadi. Once build_response_matrix.py calistir.")

    R_dy_1 = np.load("R_dy_1.npy")
    R_dy_2 = np.load("R_dy_2.npy")
    R_dx_1 = np.load("R_dx_1.npy")
    R_dx_2 = np.load("R_dx_2.npy")
    dR_dy = R_dy_2 - R_dy_1
    dR_dx = R_dx_2 - R_dx_1

    n_q     = R_dy_1.shape[0]
    g1_nom  = config.get("g1", 0.21)
    eps     = 0.02
    g1_pert = g1_nom * (1.0 + eps)

    # Sabit gercek dy/dx hatalar
    dy_max    = config.get("quad_random_dy_max", 0.3e-3)
    dx_max    = config.get("quad_random_dx_max", 0.3e-3)
    quad_seed = config.get("quad_random_seed", 13)
    rng = np.random.default_rng(seed=quad_seed)
    dy_true = rng.uniform(-dy_max, dy_max, n_q)
    dx_true = rng.uniform(-dx_max, dx_max, n_q)

    print("=" * 70)
    print("quad_tilt kontaminasyon taramasi")
    print("=" * 70)
    print(f"  Sabit gercek hatalar: dy_RMS={np.std(dy_true)*1e3:.3f} mm, "
          f"dx_RMS={np.std(dx_true)*1e3:.3f} mm")
    print(f"  g_nom={g1_nom}, g_pert={g1_pert:.4f} (eps={eps*100:.0f}%)")
    print(f"  dR_dy kappa={np.linalg.cond(dR_dy):.2e}, "
          f"dR_dx kappa={np.linalg.cond(dR_dx):.2e}")

    thetas = np.array([0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3])

    print()
    print(f"{'theta_max[mrad]':>16s}  {'dy err[um]':>12s}  {'dx err[um]':>12s}  "
          f"{'corr_dy':>10s}  {'corr_dx':>10s}")
    print("-" * 70)

    results = []
    t0 = time.time()
    for th in thetas:
        rms_dy, rms_dx, c_dy, c_dx = reconstruct_one(
            config, th, dy_true, dx_true,
            g1_nom, g1_pert, dR_dy, dR_dx)
        results.append((th, rms_dy, rms_dx, c_dy, c_dx))
        print(f"{th*1e3:16.4f}  {rms_dy*1e6:12.3f}  {rms_dx*1e6:12.3f}  "
              f"{c_dy:10.6f}  {c_dx:10.6f}")
    print(f"\nToplam sure: {time.time()-t0:.1f}s")

    arr = np.array(results)
    np.savetxt("scan_qtilt_contamination.csv", arr,
               header="theta_max_rad rms_dy_m rms_dx_m corr_dy corr_dx",
               fmt="%.6e")

    # Olcekleme analizi: kontaminasyon hatasi (baseline cikarmali)
    th = arr[1:, 0]
    base_dy = arr[0, 1]
    base_dx = arr[0, 2]
    d_err_dy = np.sqrt(np.maximum(arr[1:, 1]**2 - base_dy**2, 1e-30))
    d_err_dx = np.sqrt(np.maximum(arr[1:, 2]**2 - base_dx**2, 1e-30))

    def _slope(y):
        m = y > 0
        if m.sum() < 3:
            return float('nan')
        return np.polyfit(np.log(th[m]), np.log(y[m]), 1)[0]

    sdy = _slope(d_err_dy)
    sdx = _slope(d_err_dx)

    print("\n" + "=" * 70)
    print("Kontaminasyon olceklemesi (sqrt(err^2 - base^2) vs theta)")
    print("=" * 70)
    print(f"  Baseline hatalar (theta=0): dy={base_dy*1e6:.2f} um, "
          f"dx={base_dx*1e6:.2f} um")
    print(f"  d log(quad_tilt katkisi dy) / d log(theta) = {sdy:.3f}")
    print(f"  d log(quad_tilt katkisi dx) / d log(theta) = {sdx:.3f}")
    print()
    if abs(sdy - 1.0) < 0.3 and abs(sdx - 1.0) < 0.3:
        print("  -> Tilt katkisi LINEER (kuplaj ∝ theta), tepki matrisine")
        print("     yansiyan sinyal genligi de lineer artıyor.")
    else:
        print("  -> Egim 1'den uzak: nu_x~nu_y rezonansi veya dogrusal")
        print("     olmayan rejim olabilir.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

        ax[0].loglog(arr[:, 0]*1e3 + 1e-6, arr[:, 1]*1e6, 'o-', label="dy hata RMS")
        ax[0].loglog(arr[:, 0]*1e3 + 1e-6, arr[:, 2]*1e6, 's-', label="dx hata RMS")
        ax[0].axhline(dy_max*1e6, color='r', ls=':', alpha=0.5,
                      label=f"gercek hata seviyesi ({dy_max*1e6:.0f} um)")
        ax[0].set_xlabel("theta_max [mrad]  (+1e-6 offset, log)")
        ax[0].set_ylabel("Geri catim hatasi RMS [um]")
        ax[0].set_title("k-mod geri catim hatasi vs quad_tilt")
        ax[0].grid(True, which='both', alpha=0.3)
        ax[0].legend()

        ax[1].semilogx(arr[:, 0]*1e3 + 1e-6, arr[:, 3], 'o-', label="corr(dy)")
        ax[1].semilogx(arr[:, 0]*1e3 + 1e-6, arr[:, 4], 's-', label="corr(dx)")
        ax[1].axhline(0.99, color='g', ls=':', alpha=0.5, label="0.99 esik")
        ax[1].axhline(0.3, color='r', ls=':', alpha=0.5, label="0.3 (anlamsiz)")
        ax[1].set_xlabel("theta_max [mrad]  (+1e-6 offset, log)")
        ax[1].set_ylabel("Korelasyon")
        ax[1].set_title("Geri catim korelasyonu")
        ax[1].set_ylim(-0.1, 1.05)
        ax[1].grid(True, which='both', alpha=0.3)
        ax[1].legend()

        fig.tight_layout()
        fig.savefig("scan_qtilt_contamination.png", dpi=120)
        print("\nGrafik: scan_qtilt_contamination.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
