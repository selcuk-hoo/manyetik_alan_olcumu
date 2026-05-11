#!/usr/bin/env python3
"""
scan_quad_tilt.py

quad_random_tilt_max degerini tarayip x-y kuplaj metriklerinin
beklenen olceklerle uyumunu dogrular.

Beklenen yasalar (zayif kuplaj rejimi, rezonanstan uzak):
  Jx_osc       ∝ theta_max      (egim ≈ 1.0, log-log)
  cross_peak   ∝ theta_max^2    (egim ≈ 2.0, log-log)

Cikti:
  scan_quad_tilt.csv  — theta_max, Jx_osc, Jy_osc, cross_y, cross_x
  scan_quad_tilt.png  — log-log grafik + fit egimleri
"""
import json
import os
import time
import numpy as np
from numpy.fft import rfft, rfftfreq

from integrator import integrate_particle, FieldParams

BASE = os.path.dirname(os.path.abspath(__file__))


def setup_fields(config):
    M2  = 0.938272046
    AMU = 1.792847356
    C   = 299792458.0
    M1  = 1.672621777e-27
    p_magic = M2 / np.sqrt(AMU)
    beta0   = p_magic / np.sqrt(p_magic**2 + M2**2)
    gamma0  = 1.0 / np.sqrt(1.0 - beta0**2)
    R0      = config["R0"]
    E0_V_m  = -(p_magic * beta0 / R0) * 1e9
    direction = float(config.get("direction", -1))
    p_mag = gamma0 * M1 * C * beta0

    a = FieldParams()
    a.R0       = R0
    a.E0       = E0_V_m
    a.E0_power = config.get("E0_power", 1.0)
    a.B0ver    = config.get("B0ver", 0.0)
    a.B0rad    = config.get("B0rad", 0.0)
    a.B0long   = config.get("B0long", 0.0)
    a.quadG1   = config.get("g1", 0.21)
    a.quadG0   = config.get("g0", a.quadG1)
    a.sextK1   = config.get("sextK1", 0.0)
    a.quadSwitch = float(config.get("quadSwitch", 1))
    a.sextSwitch = float(config.get("sextSwitch", 0))
    a.EDMSwitch  = 0.0
    a.direction  = direction
    a.nFODO    = float(config.get("nFODO", 24))
    a.quadLen  = float(config.get("quadLen", 0.4))
    a.driftLen = float(config.get("driftLen", 2.0))
    a.poincare_quad_index = -1.0
    a.rfSwitch = 0.0
    a.rfVoltage = 0.0
    a.h = float(config.get("h", 1.0))

    x0 = config.get("dev0", 1e-5)
    y0v = config.get("y0", 1e-5)
    state0 = [x0, y0v, 0.0,
              0.0, 0.0, p_mag * direction,
              0.0, 0.0, direction]
    return a, state0


def compute_metrics(poin_local):
    """Poincare verisinden Jx/Jy salinim ve FFT capraz pik metrikleri."""
    x  = poin_local[:, 0]
    y  = poin_local[:, 1]
    px = poin_local[:, 3]
    py = poin_local[:, 4]
    pz = poin_local[:, 5]
    xp = px / pz
    yp = py / pz

    # Normalize: Jx ~ x^2 + xp^2 (beta=1 approx, mutlak deger onemli degil)
    # Olcekleme oranlari iceriyor, normalizasyon log-log fit'i etkilemez.
    Jx = x**2 + xp**2
    Jy = y**2 + yp**2
    Jx_osc = float(np.std(Jx) / np.mean(Jx)) if np.mean(Jx) > 0 else 0.0
    Jy_osc = float(np.std(Jy) / np.mean(Jy)) if np.mean(Jy) > 0 else 0.0

    xc = x - x.mean()
    yc = y - y.mean()
    Fx = np.abs(rfft(xc))
    Fy = np.abs(rfft(yc))
    ix = int(np.argmax(Fx[1:]) + 1)
    iy = int(np.argmax(Fy[1:]) + 1)
    cross_y = float(Fy[ix] / Fy[iy]) if Fy[iy] > 0 else 0.0
    cross_x = float(Fx[iy] / Fx[ix]) if Fx[ix] > 0 else 0.0
    return Jx_osc, Jy_osc, cross_y, cross_x


def run_one(config, theta_max, seed=44):
    alanlar, state0 = setup_fields(config)
    n_q = 2 * int(alanlar.nFODO)
    n_d = 2 * int(alanlar.nFODO)
    quad_dy_arr = np.zeros(n_q)
    quad_dx_arr = np.zeros(n_q)
    dipole_tilt = np.zeros(n_d)
    if theta_max > 0:
        rng = np.random.default_rng(seed)
        quad_tilt = rng.uniform(-theta_max, theta_max, n_q)
    else:
        quad_tilt = np.zeros(n_q)

    _, poin_local, _ = integrate_particle(
        state0, t0=0.0,
        t_end=config.get("t2", 1e-3),
        h=config.get("dt", 1e-11),
        fields=alanlar,
        return_steps=100,
        quad_dy=quad_dy_arr, quad_dx=quad_dx_arr,
        dipole_tilt=dipole_tilt, quad_tilt=quad_tilt,
    )
    if poin_local.shape[0] < 16:
        return None
    return compute_metrics(poin_local)


def main():
    os.chdir(BASE)
    with open("params.json") as f:
        config = json.load(f)

    # Tarama degerleri (rad). 0 noktasi ayri ele aliniyor (log icin).
    thetas = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3])

    print("=" * 60)
    print("quad_random_tilt_max taramasi")
    print("=" * 60)
    print(f"{'theta_max[mrad]':>16s}  {'Jx_osc':>10s}  {'Jy_osc':>10s}  "
          f"{'cross_y':>10s}  {'cross_x':>10s}")
    print("-" * 60)

    results = []
    t0 = time.time()
    for th in thetas:
        out = run_one(config, th)
        if out is None:
            print(f"  theta={th:.1e}: yetersiz Poincare verisi")
            continue
        Jx_o, Jy_o, cy, cx = out
        results.append((th, Jx_o, Jy_o, cy, cx))
        print(f"{th*1e3:16.4f}  {Jx_o:10.4e}  {Jy_o:10.4e}  "
              f"{cy:10.4e}  {cx:10.4e}")
    print(f"\nToplam sure: {time.time()-t0:.1f}s")

    if len(results) < 3:
        print("Fit icin yeterli nokta yok.")
        return

    arr = np.array(results)
    th = arr[:, 0]
    Jx_o = arr[:, 1]
    cy   = arr[:, 3]

    # log-log fit
    mask_J = (Jx_o > 0)
    mask_F = (cy > 0)
    slope_J, _ = np.polyfit(np.log(th[mask_J]), np.log(Jx_o[mask_J]), 1)
    slope_F, _ = np.polyfit(np.log(th[mask_F]), np.log(cy[mask_F]), 1)

    print("\n" + "=" * 60)
    print("log-log fit egimleri (beklenen: Jx_osc=1.0, cross=2.0)")
    print("=" * 60)
    print(f"  d log(Jx_osc)     / d log(theta)  = {slope_J:.3f}  (bekl. ~1.0)")
    print(f"  d log(cross_y@x)  / d log(theta)  = {slope_F:.3f}  (bekl. ~2.0)")
    print()
    if abs(slope_J - 1.0) < 0.2 and abs(slope_F - 2.0) < 0.4:
        print("  -> quad_tilt fizigi DOGRU olceklenyor (skew-quadrupol).")
    else:
        print("  -> Sapma var: rezonansa yakin olabilir veya")
        print("     dogrusal-olmayan rejimde takiliyorsun.")

    # CSV ve grafik
    np.savetxt("scan_quad_tilt.csv", arr,
               header="theta_max_rad Jx_osc Jy_osc cross_y_at_x cross_x_at_y",
               fmt="%.6e")
    print("\nVeri: scan_quad_tilt.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

        ax[0].loglog(th*1e3, Jx_o, 'o-', label=f"Jx_osc (egim={slope_J:.2f})")
        ax[0].loglog(th*1e3, arr[:, 2], 's--', label="Jy_osc", alpha=0.6)
        ref = Jx_o[0] * (th / th[0])
        ax[0].loglog(th*1e3, ref, 'k:', label="ref: egim=1", alpha=0.5)
        ax[0].set_xlabel("theta_max [mrad]")
        ax[0].set_ylabel("J salinim orani")
        ax[0].set_title("Courant-Snyder invariant salinimi")
        ax[0].grid(True, which='both', alpha=0.3)
        ax[0].legend()

        ax[1].loglog(th*1e3, cy, 'o-', label=f"cross_y@nu_x (egim={slope_F:.2f})")
        ax[1].loglog(th*1e3, arr[:, 4], 's--', label="cross_x@nu_y", alpha=0.6)
        ref2 = cy[0] * (th / th[0])**2
        ax[1].loglog(th*1e3, ref2, 'k:', label="ref: egim=2", alpha=0.5)
        ax[1].set_xlabel("theta_max [mrad]")
        ax[1].set_ylabel("FFT capraz pik orani")
        ax[1].set_title("Spektral kuplaj kaniti")
        ax[1].grid(True, which='both', alpha=0.3)
        ax[1].legend()

        fig.tight_layout()
        fig.savefig("scan_quad_tilt.png", dpi=120)
        print("Grafik: scan_quad_tilt.png")
    except ImportError:
        print("matplotlib yok, grafik atlandi.")


if __name__ == "__main__":
    main()
