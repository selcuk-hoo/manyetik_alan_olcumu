#!/usr/bin/env python3
"""
compare_field_harmonics.py  —  Omarov Fig.7 analog (B_r harmonikleri)

Omarov 2022 (PRD 105, 032001) Şekil 7 sağ paneli analogu:
  Perturbasyon : dipole_tilt[j] = alpha_0 * cos(N * theta_j)
  Fiziksel anlam: j'inci yay bölümünde ΔB_r = B_eq * sin(alpha_0) ≈ B_eq * alpha_0
  Hedef  : dS_y/dt vs N  taraması (N = 0..N_max)

C++ kodu (integrator.cpp, satır 160-167):
  p_magic = M_PROTON_GEV / sqrt(G_P)          [GeV/c]
  B_eq    = p_magic * 1e9 / (C_LIGHT * R0)    [T]   ← ~24.5 mT
  b_tilt  = B_eq * sin(dipole_tilt[j])        → B_r katkısı (lokal radyal)

Parametre seçimi:
  alpha_0 = Br_target / B_eq  →  B_r genliği = Br_target [T]

Not: dipole_tilt yalnızca 48 yay bölümüne (deflektör) uygulanır;
     driftler ve quadlar kapsanmaz (~%80 doluluk oranı).

Yöntem: false_edm_mode_scan.py ile aynı — CO fırlatması + stroboskopik S_y eğimi.

Çıktı:
  field_harmonic_scan.png   — dS_y/dt vs N (bizim ring, B_r harmoniği)
  field_vs_misalign.png     — yan yana: B_r harmoniği + misalignment (false_edm_mode_scan)
"""
import json, os, sys, time
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)

from integrator import integrate_particle, FieldParams

# ── Sabitler ────────────────────────────────────────────────────────────────
M2  = 0.938272046      # proton kütlesi [GeV/c²]
AMU = 1.792847356      # G_P = anomalous magnetic moment
C_  = 299792458.0      # ışık hızı [m/s]
M1  = 1.672621777e-27  # proton kütlesi [kg]


def setup_fields(config):
    p_magic = M2 / np.sqrt(AMU)            # magic momentum [GeV/c]
    E_tot   = np.sqrt(p_magic**2 + M2**2)
    beta0   = p_magic / E_tot
    gamma0  = 1.0 / np.sqrt(1.0 - beta0**2)
    R0      = config["R0"]
    E0_V_m  = -(p_magic * beta0 / R0) * 1e9  # kılavuz elektrik alan [V/m]

    # Eşdeğer manyetik dönüş alanı B_eq = p/(qR0) [T]
    # dipole_tilt[j] = phi_j → ΔB_r = B_eq * sin(phi_j) ≈ B_eq * phi_j
    B_eq = p_magic * 1e9 / (C_ * R0)         # ~24.5 mT

    f = FieldParams()
    f.R0        = R0
    f.E0        = E0_V_m
    f.E0_power  = config.get("E0_power", 1.0)
    f.quadG1    = config.get("g1", 0.21)
    f.quadG0    = config.get("g0", f.quadG1)
    f.sextK1    = config.get("sextK1", 0.0)
    f.quadSwitch  = float(config.get("quadSwitch", 1))
    f.sextSwitch  = float(config.get("sextSwitch", 0))
    f.EDMSwitch   = 0.0
    f.direction   = float(config.get("direction", -1))
    f.nFODO     = float(config.get("nFODO", 24))
    f.quadLen   = float(config.get("quadLen", 0.4))
    f.driftLen  = float(config.get("driftLen", 2.0833))
    f.poincare_quad_index = -1.0
    f.rfSwitch  = 0.0
    f.h         = float(config.get("h", 100))

    p_mag = gamma0 * M1 * C_ * beta0
    direction = f.direction
    y0 = [0.0, 0.0, 0.0,
          0.0, 0.0, p_mag * direction,
          0.0, 0.0, direction]
    return f, y0, beta0, gamma0, R0, p_mag, direction, B_eq


def _make_state(v, p_mag, direction, spin):
    return [v[0], v[1], 0.0,
            p_mag * direction * v[2], p_mag * direction * v[3], p_mag * direction,
            spin[0], spin[1], spin[2]]


def find_co_with_dipole_tilt(fields, p_mag, direction, dipole_tilt_arr,
                              dt, T_rev, n_turns=60, n_iter=2):
    """Kapalı yörünge (y0, y') — dipole_tilt pertürbasyonlu halka için."""
    n_q = int(2 * fields.nFODO)
    spin = [0.0, 0.0, direction]
    fields.poincare_quad_index = 0.0
    t_probe = n_turns * T_rev
    quad_dy_zero = np.zeros(n_q)

    def var2(yc, ypc):
        st = _make_state([0.0, yc, 0.0, ypc], p_mag, direction, spin)
        _, poin, _ = integrate_particle(
            st, 0.0, t_probe, dt, fields=fields,
            return_steps=10,
            quad_dy=quad_dy_zero,
            dipole_tilt=dipole_tilt_arr)
        if poin is None or len(poin) < 5:
            return 1e30
        return float(np.var(poin[:, 1]))

    yc, ypc = 0.0, 0.0
    sy, syp = 2e-4, 2e-5
    for it in range(n_iter):
        f0   = var2(yc,      ypc)
        fp_y = var2(yc + sy, ypc);  fm_y = var2(yc - sy, ypc)
        fp_p = var2(yc, ypc + syp); fm_p = var2(yc, ypc - syp)
        fpp  = var2(yc + sy, ypc + syp)
        gy = (fp_y - fm_y) / (2*sy)
        gp = (fp_p - fm_p) / (2*syp)
        Hyy = (fp_y - 2*f0 + fm_y) / (sy**2)
        Hpp = (fp_p - 2*f0 + fm_p) / (syp**2)
        Hyp = (fpp - fp_y - fp_p + f0) / (sy*syp)
        det = Hyy*Hpp - Hyp**2
        if det <= 0 or Hyy <= 0:
            break
        dy = -(Hpp*gy - Hyp*gp) / det
        dp = -(-Hyp*gy + Hyy*gp) / det
        yc += dy; ypc += dp
        sy *= 0.2; syp *= 0.2

    resid = np.sqrt(max(var2(yc, ypc), 0.0))
    fields.poincare_quad_index = -1.0
    return np.array([0.0, yc, 0.0, ypc]), float(resid)


def measure_one_N(N, alpha_0, t2, dt, fields, p_mag, direction, T_rev,
                   do_co=True, co_turns=60):
    """N'inci B_r alan harmoniği için dS_y/dt ölçümü."""
    n_q = int(2 * fields.nFODO)
    # 48 yay bölümünün azimut açıları (düzgün aralıklı yaklaşım)
    theta_j = 2 * np.pi * (np.arange(n_q) + 0.5) / n_q
    dipole_tilt = alpha_0 * np.cos(N * theta_j)

    # CO fırlatması
    if do_co:
        v_co, resid = find_co_with_dipole_tilt(
            fields, p_mag, direction, dipole_tilt, dt, T_rev,
            n_turns=co_turns)
        y_launch = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
        co_mm = float(np.hypot(v_co[0], v_co[1]) * 1e3)
    else:
        y_launch = [0.0, 0.0, 0.0, 0.0, 0.0, p_mag*direction, 0.0, 0.0, direction]
        co_mm = float("nan")

    # Stroboskopik ölçüm
    fields.poincare_quad_index = 0.0
    _, poin, poin_t = integrate_particle(
        y_launch, 0.0, t2, dt, fields=fields,
        return_steps=200,
        dipole_tilt=dipole_tilt)
    fields.poincare_quad_index = -1.0

    if poin is None or len(poin) < 10:
        return float("nan"), co_mm

    sy = np.asarray(poin[:, 7], float)
    ts = np.asarray(poin_t, float)
    slope = float(np.polyfit(ts, sy, 1)[0])
    return slope, co_mm


def run_scan(N_max=12, Br_target=1e-9, t2=5e-3,
             do_co=True, co_turns=60, dt=None):
    with open("params.json") as f:
        config = json.load(f)

    fields, _, beta0, gamma0, R0, p_mag, direction, B_eq = setup_fields(config)
    circ = 2*np.pi*R0 + 4*fields.nFODO*fields.driftLen + 2*fields.nFODO*fields.quadLen
    T_rev = circ / (beta0 * C_)
    if dt is None:
        dt = config.get("dt", 1e-11)

    # alpha_0 [rad]: dipole_tilt genliği → ΔB_r = B_eq * alpha_0
    alpha_0 = Br_target / B_eq
    N_list = list(range(N_max + 1))

    print("=" * 70)
    print("  B_r ALAN HARMONİĞİ TARAMASI  (Omarov Fig.7 sağ panel analog)")
    print(f"  B_r_target = {Br_target:.2e} T  →  alpha_0 = {alpha_0:.3e} rad")
    print(f"  B_eq = {B_eq:.4f} T,  t2 = {t2*1e3:.1f} ms")
    print(f"  N_list = {N_list},  CO fırlatma: {'AÇIK' if do_co else 'KAPALI'}")
    print("=" * 70)

    results = []
    t_wall = time.time()
    for N in N_list:
        t0 = time.time()
        slope, co_mm = measure_one_N(
            N, alpha_0, t2, dt, fields, p_mag, direction, T_rev,
            do_co=do_co, co_turns=co_turns)
        elapsed = time.time() - t0
        print(f"  N={N:2d}  CO={co_mm:7.4f}mm  dSy/dt={slope:+.3e} rad/s"
              f"   ({elapsed:.0f}s)")
        results.append({"N": N, "dSy_dt": slope, "co_mm": co_mm,
                         "Br_target": Br_target, "B_eq": B_eq, "alpha_0": alpha_0})

    print(f"\n  Toplam: {time.time()-t_wall:.0f}s")
    return results, Br_target, B_eq


def plot_results(results, Br_target, B_eq,
                 misalign_data=None, outdir="."):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ns   = np.array([r["N"]       for r in results])
    dsy  = np.array([abs(r["dSy_dt"]) for r in results])
    co   = np.array([r["co_mm"]   for r in results])

    BLUE  = "#2166ac"
    RED   = "#d6604d"

    Br_nT = Br_target * 1e9  # nT cinsinden

    # ── Fig A: B_r alan harmoniği taraması ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(Ns, dsy, "o-", color=BLUE, lw=2, ms=7,
                label=rf"$|dS_y/dt|$ — $B_r={Br_nT:.1f}$ nT harmonic")
    ax.set_xlabel("Field harmonic $N$")
    ax.set_ylabel(r"$|dS_y/dt|$ [rad/s]")
    ax.set_title(r"Vertical spin precession rate vs. $B_r$ field harmonic" + "\n"
                 r"(Omarov Fig. 7 right analog, CW beam, $Q_y \approx 2.68$)")
    ax.set_xticks(Ns)
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.4)
    for Nr, label in [(2, r"$N\approx Q_y$"), (3, r"$N\approx Q_y$")]:
        if Nr in Ns:
            ax.axvline(Nr, color=RED, lw=0.8, ls="--", alpha=0.6)
    fig.tight_layout()
    out_a = os.path.join(outdir, "field_harmonic_scan.png")
    fig.savefig(out_a, dpi=150)
    print(f"  -> {out_a}")
    plt.close(fig)

    # ── Fig B: yan yana karşılaştırma (varsa misalignment verisi) ────────────
    if misalign_data is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        ax0 = axes[0]
        ax0.semilogy(Ns, dsy, "o-", color=BLUE, lw=2, ms=7)
        ax0.set_xlabel("Field harmonic $N$")
        ax0.set_ylabel(r"$|dS_y/dt|$ [rad/s]")
        ax0.set_title(rf"$B_r = {Br_nT:.1f}$ nT radyal alan harmoniği"
                      "\n(bu halka, CW)")
        ax0.set_xticks(Ns)
        ax0.grid(True, which="both", ls=":", alpha=0.4)
        for Nr in [2, 3]:
            if Nr in Ns:
                ax0.axvline(Nr, color=RED, lw=0.8, ls="--", alpha=0.6)

        ax1 = axes[1]
        ks_m  = np.array([r["k"]          for r in misalign_data])
        dsy_m = np.array([abs(r["dSy_dt"]) for r in misalign_data])
        ax1.semilogy(ks_m, dsy_m, "s-", color=RED, lw=2, ms=7)
        ax1.set_xlabel("Quad misalignment harmonic $k$")
        ax1.set_ylabel(r"$|dS_y/dt|$ [rad/s]")
        ax1.set_title("$A=10\\ \\mu$m quad kaçıklığı harmoniği\n(bu halka, CW)")
        ax1.set_xticks(ks_m)
        ax1.grid(True, which="both", ls=":", alpha=0.4)
        for Nr in [2, 3]:
            if Nr in ks_m:
                ax1.axvline(Nr, color=BLUE, lw=0.8, ls="--", alpha=0.6)

        fig.suptitle(r"$B_r$ alan harmoniği vs. quad kaçıklık harmoniği "
                     r"— $Q_y\approx2.68$",
                     fontsize=12)
        fig.tight_layout()
        out_b = os.path.join(outdir, "field_vs_misalign.png")
        fig.savefig(out_b, dpi=150)
        print(f"  -> {out_b}")
        plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Omarov Fig.7 sağ panel analog: dS_y/dt vs B_r alan harmoniği N")
    p.add_argument("--Nmax", type=int, default=12,
                   help="Taranacak maksimum harmonic sayısı (default 12)")
    p.add_argument("--Br", type=float, default=1e-9,
                   help="B_r genliği [T] (default 1e-9 = 1 nT)")
    p.add_argument("--t2", type=float, default=5e-3,
                   help="Simülasyon süresi [s] (default 5ms)")
    p.add_argument("--no-co", action="store_true")
    p.add_argument("--co-turns", type=int, default=60)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--misalign-log", type=str, default=None,
                   help="false_edm_mode_scan.py JSON çıktısı (yan yana plot için)")
    args = p.parse_args()

    results, Br_target, B_eq = run_scan(
        N_max=args.Nmax, Br_target=args.Br, t2=args.t2,
        do_co=not args.no_co, co_turns=args.co_turns, dt=args.dt)

    misalign_data = None
    if args.misalign_log and os.path.exists(args.misalign_log):
        with open(args.misalign_log) as f:
            misalign_data = json.load(f)

    plot_results(results, Br_target, B_eq, misalign_data=misalign_data)

    print("\n  N   dSy/dt [rad/s]    CO [mm]")
    for r in results:
        print(f"  {r['N']:2d}   {r['dSy_dt']:+.3e}    {r['co_mm']:.4f}")

    out_json = "field_harmonic_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  JSON: {out_json}")
