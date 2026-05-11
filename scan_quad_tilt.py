#!/usr/bin/env python3
"""
scan_quad_tilt.py

quad_random_tilt_max degerini tarayip x-y kuplaj metriklerinin
beklenen olceklerle uyumunu dogrular.

Beklenen yasalar (zayif kuplaj rejimi, rezonanstan uzak):
  Hem Jx_osc hem cross_peak AMPLITUD orani oldugundan, kuplaj
  katsayisi |C-| ∝ theta ile dogrudan dogru orantili olmali:
    Delta_Jx_osc     ∝ theta_max   (egim ≈ 1.0, log-log)
    Delta_cross_peak ∝ theta_max   (egim ≈ 1.0, log-log)

  ONEMLI: theta=0 baseline'i cikarilmali — yoksa sextupol/lineer
  olmayan kaynaklardan gelen artalan kuplaji egimi sifirlar.

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

    # Once theta=0 baseline (sextupol vb. kaynakli artalan kuplaji)
    thetas = np.array([0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3])

    print("=" * 70)
    print("quad_random_tilt_max taramasi (baseline cikarmali)")
    print("=" * 70)
    print(f"{'theta_max[mrad]':>16s}  {'Jx_osc':>10s}  {'Jy_osc':>10s}  "
          f"{'cross_y':>10s}  {'cross_x':>10s}")
    print("-" * 70)

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

    if len(results) < 4:
        print("Fit icin yeterli nokta yok.")
        return

    arr = np.array(results)
    # Baseline (theta=0) ayir
    base_mask = arr[:, 0] == 0.0
    pert_mask = arr[:, 0] > 0.0
    base = arr[base_mask][0] if base_mask.any() else np.zeros(5)
    pert = arr[pert_mask]

    th  = pert[:, 0]
    dJx = np.abs(pert[:, 1] - base[1])
    dJy = np.abs(pert[:, 2] - base[2])
    dcy = np.abs(pert[:, 3] - base[3])
    dcx = np.abs(pert[:, 4] - base[4])

    print("\nBaseline (theta=0):")
    print(f"  Jx_osc={base[1]:.4e}  Jy_osc={base[2]:.4e}  "
          f"cross_y={base[3]:.4e}  cross_x={base[4]:.4e}")
    print("\nBaseline cikarilmis sinyaller (|metric(theta) - metric(0)|):")
    print(f"{'theta_max[mrad]':>16s}  {'dJx':>10s}  {'dJy':>10s}  "
          f"{'dcross_y':>10s}  {'dcross_x':>10s}")
    for i in range(len(th)):
        print(f"{th[i]*1e3:16.4f}  {dJx[i]:10.4e}  {dJy[i]:10.4e}  "
              f"{dcy[i]:10.4e}  {dcx[i]:10.4e}")

    def _slope(y):
        m = y > 0
        if m.sum() < 3:
            return float('nan')
        return np.polyfit(np.log(th[m]), np.log(y[m]), 1)[0]

    sJx = _slope(dJx)
    sJy = _slope(dJy)
    scy = _slope(dcy)
    scx = _slope(dcx)

    print("\n" + "=" * 70)
    print("log-log fit egimleri (beklenen tum metrikler icin ~1.0)")
    print("=" * 70)
    print(f"  d log(|dJx_osc|)     / d log(theta) = {sJx:.3f}")
    print(f"  d log(|dJy_osc|)     / d log(theta) = {sJy:.3f}")
    print(f"  d log(|dcross_y|)    / d log(theta) = {scy:.3f}")
    print(f"  d log(|dcross_x|)    / d log(theta) = {scx:.3f}")
    print()

    good = [s for s in [sJx, sJy, scy, scx] if not np.isnan(s)]
    n_good = sum(1 for s in good if abs(s - 1.0) < 0.25)
    if n_good >= 3:
        print("  -> quad_tilt fizigi DOGRU olceklenyor (|C-| ∝ theta).")
    else:
        print("  -> Bazi metrikler beklenenden sapiyor. Olasi sebepler:")
        print("     * Rezonansa cok yakin (nu_x ~ nu_y)")
        print("     * Artalan kuplaji (sextupol) baseline > delta")
        print("     * Tarama araligi cok genis (lineer rejim disinda)")

    # Sonraki adimlar icin kullanilabilir bir kalibrasyon sabiti
    if not np.isnan(scx) and abs(scx - 1.0) < 0.3:
        # k = dcross_x / theta
        k = dcx[-1] / th[-1]
        print(f"\n  Kalibrasyon sabiti  k = d(cross_x) / theta ≈ {k:.3e}")
        print(f"  Tersine kullanim: theta_max ≈ d(cross_x) / k")

    # CSV
    np.savetxt("scan_quad_tilt.csv", arr,
               header="theta_max_rad Jx_osc Jy_osc cross_y_at_x cross_x_at_y",
               fmt="%.6e")
    print("\nVeri: scan_quad_tilt.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

        ax[0].loglog(th*1e3, dJx, 'o-', label=f"|dJx_osc| (egim={sJx:.2f})")
        ax[0].loglog(th*1e3, dJy, 's--', label=f"|dJy_osc| (egim={sJy:.2f})", alpha=0.6)
        if dJx[-1] > 0:
            ref = dJx[-1] * (th / th[-1])
            ax[0].loglog(th*1e3, ref, 'k:', label="ref: egim=1", alpha=0.5)
        ax[0].set_xlabel("theta_max [mrad]")
        ax[0].set_ylabel("|metric(theta) - metric(0)|")
        ax[0].set_title("J salinim deltasi (baseline cikarilmis)")
        ax[0].grid(True, which='both', alpha=0.3)
        ax[0].legend()

        ax[1].loglog(th*1e3, dcx, 'o-', label=f"|dcross_x| (egim={scx:.2f})")
        ax[1].loglog(th*1e3, dcy, 's--', label=f"|dcross_y| (egim={scy:.2f})", alpha=0.6)
        if dcx[-1] > 0:
            ref2 = dcx[-1] * (th / th[-1])
            ax[1].loglog(th*1e3, ref2, 'k:', label="ref: egim=1", alpha=0.5)
        ax[1].set_xlabel("theta_max [mrad]")
        ax[1].set_ylabel("|metric(theta) - metric(0)|")
        ax[1].set_title("FFT capraz pik deltasi (baseline cikarilmis)")
        ax[1].grid(True, which='both', alpha=0.3)
        ax[1].legend()

        fig.tight_layout()
        fig.savefig("scan_quad_tilt.png", dpi=120)
        print("Grafik: scan_quad_tilt.png")
    except ImportError:
        print("matplotlib yok, grafik atlandi.")


if __name__ == "__main__":
    main()
