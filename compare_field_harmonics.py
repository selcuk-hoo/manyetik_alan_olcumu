#!/usr/bin/env python3
"""
compare_field_harmonics.py  —  Omarov 2022 (PRD 105, 032001) Fig. 8 analogu
                               + N=0/N=1 ARTEFAKT TANISI + PARALELLEŞTİRME

Omarov Şekil 8: "vertical spin precession rate vs. B_x = 1 nT field N harmonic
around the ring".  Yani halka boyunca azimutal olarak değişen, N-harmonikli,
1 nT genlikli bir RADYAL manyetik alan B_x(θ) = A_r·cos(N·θ) uygulanır ve
dikey spin presesyon hızı dS_y/dt'nin N'e bağımlılığı ölçülür.

Bu alan integratörde doğrudan desteklenir (tilt YOK):
  FieldParams.B0rad_harm_amp = A_r   [T]   (genlik, ör. 1e-9 = 1 nT)
  FieldParams.B0rad_harm_N   = N           (azimut harmonik numarası)
C++ tarafı bu alanı halkadaki HER elemana YEREL RADYAL yönde uygular
(integrator.cpp:260, B[0] += A_r·cos(N·θ), θ = θ_e + atan2(Y,X)).

────────────────────────────────────────────────────────────────────────────
NEDEN BU SÜRÜM:  N=0 (ve N=1) baskınlığı fizik mi, artefakt mı?

Fiziksel beklenti (kullanıcı itirazı, Omarov Fig.8 ile uyumlu):
  Uniform radyal B (N=0) dikey Lorentz kuvveti üretir → kapalı yörünge dikeyde
  kayar → quad odaklaması dengeler. Kaymış yörüngede görülen radyal E-alanı,
  B-radyal presesyonunu BÜYÜK ORANDA TELAFİ eder (frozen-spin manyetometri).
  Geriye yalnızca geometrik-faz mertebesinde KÜÇÜK bir dS_y/dt kalır.
  → N=0'ın diğerlerinden büyük olması için sebep yoktur; Omarov Fig.8'de
    N=0 < N=1.

Eğer simülasyonda N=0 baskınsa olası sebep: kapalı-yörünge (CO) bulucusu
1 nT'lik alanın ürettiği KÜÇÜK orbit kaymasını yakalayamıyor → E-alan
telafisi spin takibine girmiyor → ham doğrudan-kuplaj (büyük) kalıyor.

Bu sürüm bunu AYIRT EDER. Her N için şunları raporlar:
  - CO offset (bulunan kapalı-yörünge dikey kayması)  [mm]
  - CO residual betatron RMS (yakınsama göstergesi)   [mm]
  - dS_y/dt  WITH CO   (telafi DAHİL)                 [rad/s]
  - dS_y/dt  NO  CO    (y=0'dan fırlat, telafi YOK)   [rad/s]
  - oran (NO/WITH): telafinin ne kadar bastırdığını gösterir
  --linearity: A_r ve A_r/2'de ölç → ölçekleme üssü (1 = lineer doğrudan kuplaj)

Yorum:
  * WITH ≈ NO (her ikisi de büyük)  → CO telafisi YAKALANMIYOR (artefakt!).
    --co-refine ile güçlendirilmiş CO bulucu N=0'ı bastırmalı.
  * WITH ≪ NO  → telafi çalışıyor; WITH küçükse fizik doğru (N=0 küçük).
  * WITH hâlâ N=0'da baskınsa ve --co-refine değiştirmiyorsa → gerçek fizik
    (uniform radyal B = en büyük sahte EDM); Omarov farkı modelleme tercihidir.

Çıktı:
  field_harmonic_scan.png   — dS_y/dt vs N (with-CO + no-CO)
  field_vs_misalign.png     — yan yana: B_x harmoniği + quad kaçıklık harmoniği
  field_harmonic_results.json
"""
import json
import os
import sys
import time
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)

# ── Fiziksel sabitler ─────────────────────────────────────────────────────────
M2  = 0.938272046      # proton kütlesi [GeV/c²]
AMU = 1.792847356      # G_P = anormal manyetik moment
C_  = 299792458.0      # ışık hızı [m/s]
M1  = 1.672621777e-27  # proton kütlesi [kg]


def setup_fields(config):
    p_magic = M2 / np.sqrt(AMU)            # magic momentum [GeV/c]
    E_tot   = np.sqrt(p_magic**2 + M2**2)
    beta0   = p_magic / E_tot
    gamma0  = 1.0 / np.sqrt(1.0 - beta0**2)
    R0      = config["R0"]
    E0_V_m  = -(p_magic * beta0 / R0) * 1e9  # kılavuz elektrik alan [V/m]

    from integrator import FieldParams
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
    return f, beta0, gamma0, R0, p_mag, direction


def _make_state(v, p_mag, direction, spin):
    return [v[0], v[1], 0.0,
            p_mag * direction * v[2], p_mag * direction * v[3], p_mag * direction,
            spin[0], spin[1], spin[2]]


def find_co(fields, p_mag, direction, dt, T_rev, n_turns=60, n_iter=3,
            refine=False, verbose=False):
    """Kapalı yörünge (y0, y') — radyal B_x harmoniği zaten fields'te ayarlı.

    Sabit azimutta (Poincaré) tur-başına dikey konum VARYANSINI minimize eder;
    lineer latiste varyans kapalı-yörüngeden sapmanın tam kuadratik formudur,
    sonlu-fark Hessian + Newton ile minimuma atlanır.

    refine=True: ADAPTİF adım — 1 nT gibi KÜÇÜK alanlar için kapalı-yörünge
    kayması nm mertebesinde olabilir. Önce kaba bir tur ile orbit ölçeğini
    kestirip sonlu-fark adımlarını o ölçeğe oturtur, böylece çok küçük orbit
    kaymaları da sayısal olarak çözülür (N=0/N=1 telafi testinin kalbi).
    """
    from integrator import integrate_particle
    spin = [0.0, 0.0, direction]
    fields.poincare_quad_index = 0.0
    t_probe = n_turns * T_rev

    def orbit_stats(yc, ypc):
        st = _make_state([0.0, yc, 0.0, ypc], p_mag, direction, spin)
        _, poin, _ = integrate_particle(
            st, 0.0, t_probe, dt, fields=fields, return_steps=10)
        if poin is None or len(poin) < 5:
            return 1e30, 0.0
        col = poin[:, 1]
        return float(np.var(col)), float(np.mean(col))

    def var2(yc, ypc):
        return orbit_stats(yc, ypc)[0]

    # ── Adaptif adım kestirimi (refine) ──────────────────────────────────────
    sy, syp = 2e-4, 2e-5            # varsayılan sonlu-fark adımları
    if refine:
        v0, mean0 = orbit_stats(0.0, 0.0)
        # y=0'dan fırlatınca betatron genliği ~ √(2·var); kapalı-yörünge DC
        # kayması ~ mean0. Adımı bu ölçeğin ~%30'una oturt (alt sınır 1e-9 m).
        scale_y = max(np.sqrt(max(v0, 0.0)), abs(mean0))
        if scale_y > 0:
            sy  = max(min(0.3 * scale_y, 2e-4), 1e-9)
            syp = sy / 10.0
        if verbose:
            print(f"    [CO refine] orbit ölçeği≈{scale_y:.2e} m → "
                  f"sy={sy:.2e}, syp={syp:.2e}")

    yc, ypc = 0.0, 0.0
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
            if verbose:
                print(f"    [CO iter {it}] Hessian pos-def değil, durdu")
            break
        dy = -(Hpp*gy - Hyp*gp) / det
        dp = -(-Hyp*gy + Hyy*gp) / det
        yc += dy; ypc += dp
        sy *= 0.3; syp *= 0.3

    resid = np.sqrt(max(var2(yc, ypc), 0.0))
    fields.poincare_quad_index = -1.0
    return np.array([0.0, yc, 0.0, ypc]), float(resid)


def _measure_slope(fields, y_launch, t2, dt):
    """Stroboskopik (sabit-azimut, tur-başına) S_y seküler eğimi [rad/s]."""
    from integrator import integrate_particle
    fields.poincare_quad_index = 0.0
    _, poin, poin_t = integrate_particle(
        y_launch, 0.0, t2, dt, fields=fields, return_steps=200)
    fields.poincare_quad_index = -1.0
    if poin is None or len(poin) < 10:
        return float("nan")
    sy = np.asarray(poin[:, 7], float)
    ts = np.asarray(poin_t, float)
    return float(np.polyfit(ts, sy, 1)[0])


def _run_one_N(task):
    """Tek N harmoniği için tam tanı (paralel worker).

    Her alt-süreç integrator'ı (ctypes lib) taze yükler → C++ çağrıları
    süreçler arası paylaşımsız, güvenle paralel çalışır (false_edm_mode_scan
    ile aynı 'spawn' kalıbı).
    """
    (N, Ar, t2, dt, do_co, co_turns, co_iter, co_refine,
     diag_noco, lin_half) = task
    import os, json, time
    import numpy as np
    os.chdir(BASE)
    sys.path.insert(0, BASE)

    with open("params.json") as f:
        config = json.load(f)
    fields, beta0, gamma0, R0, p_mag, direction = setup_fields(config)
    circ = (2*np.pi*R0 + 4*fields.nFODO*fields.driftLen
            + 2*fields.nFODO*fields.quadLen)
    T_rev = circ / (beta0 * C_)

    def measure(Ar_use, with_co):
        fields.B0rad_harm_amp = Ar_use
        fields.B0rad_harm_N   = float(N)
        if with_co:
            v_co, resid = find_co(fields, p_mag, direction, dt, T_rev,
                                  n_turns=co_turns, n_iter=co_iter,
                                  refine=co_refine)
            y_launch = _make_state(v_co, p_mag, direction,
                                   [0.0, 0.0, direction])
            co_mm = float(v_co[1] * 1e3)        # işaretli dikey CO kayması
            resid_mm = resid * 1e3
        else:
            y_launch = [0.0, 0.0, 0.0, 0.0, 0.0, p_mag*direction,
                        0.0, 0.0, direction]
            co_mm = 0.0
            resid_mm = float("nan")
        slope = _measure_slope(fields, y_launch, t2, dt)
        return slope, co_mm, resid_mm

    t0 = time.time()
    out = {"N": N, "Br_target": Ar}

    # ── Ana ölçüm: WITH-CO (varsayılan) veya NO-CO ───────────────────────────
    slope_wc, co_mm, resid_mm = measure(Ar, with_co=do_co)
    out["dSy_dt"]   = slope_wc
    out["co_mm"]    = co_mm
    out["resid_mm"] = resid_mm

    # ── Tanı: aynı N için telafisiz (NO-CO) ölçüm ────────────────────────────
    if diag_noco and do_co:
        slope_nc, _, _ = measure(Ar, with_co=False)
        out["dSy_dt_noco"] = slope_nc
        out["supp_ratio"]  = (abs(slope_nc / slope_wc)
                              if slope_wc not in (0.0,) and np.isfinite(slope_wc)
                              else float("nan"))
    else:
        out["dSy_dt_noco"] = float("nan")
        out["supp_ratio"]  = float("nan")

    # ── Lineerlik: A_r/2'de ölç → ölçekleme üssü ─────────────────────────────
    if lin_half:
        slope_half, _, _ = measure(0.5 * Ar, with_co=do_co)
        out["dSy_dt_half"] = slope_half
        if (np.isfinite(slope_half) and slope_half != 0.0
                and np.isfinite(slope_wc) and slope_wc != 0.0):
            out["lin_exp"] = float(np.log(abs(slope_wc / slope_half)) / np.log(2.0))
        else:
            out["lin_exp"] = float("nan")
    else:
        out["dSy_dt_half"] = float("nan")
        out["lin_exp"]     = float("nan")

    out["runtime"] = time.time() - t0
    return out


def run_scan(N_max=12, Br_target=1e-9, t2=5e-3, do_co=True, co_turns=60,
             co_iter=3, co_refine=False, diag_noco=True, lin_half=False,
             dt=None, nproc=None):
    with open("params.json") as f:
        config = json.load(f)
    fields, beta0, gamma0, R0, p_mag, direction = setup_fields(config)
    if dt is None:
        dt = config.get("dt", 1e-11)

    N_list = list(range(N_max + 1))
    Br_nT = Br_target * 1e9
    if nproc is None:
        nproc = min(len(N_list), max(1, os.cpu_count() or 1))

    print("=" * 78)
    print("  RADYAL B_x ALAN HARMONİĞİ TARAMASI  (Omarov Fig. 8 analog + tanı)")
    print(f"  B_x = {Br_nT:.2f} nT genlikli, halka boyunca N-harmonikli radyal alan")
    print(f"  t2 = {t2*1e3:.1f} ms,  N_list = {N_list}")
    print(f"  CO fırlatma: {'AÇIK' if do_co else 'KAPALI'}"
          f"  (iter={co_iter}, refine={'AÇIK' if co_refine else 'kapalı'})")
    print(f"  tanı NO-CO: {'AÇIK' if diag_noco else 'kapalı'},  "
          f"lineerlik: {'AÇIK' if lin_half else 'kapalı'}")
    print(f"  paralel süreç sayısı = {nproc}")
    print("=" * 78)

    tasks = [(N, Br_target, t2, dt, do_co, co_turns, co_iter, co_refine,
              diag_noco, lin_half) for N in N_list]

    t_wall = time.time()
    if nproc > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(nproc) as pool:
            results = pool.map(_run_one_N, tasks)
    else:
        results = [_run_one_N(t) for t in tasks]
    results.sort(key=lambda r: r["N"])
    wall = time.time() - t_wall

    # ── Tanı tablosu ─────────────────────────────────────────────────────────
    print(f"\n  {'N':>3} {'CO[mm]':>11} {'resid[mm]':>11} "
          f"{'dSy/dt WITH':>14} {'dSy/dt NO-CO':>14} {'NO/WITH':>9} {'linExp':>7}")
    for r in results:
        print(f"  {r['N']:>3} {r['co_mm']:>11.3e} {r['resid_mm']:>11.2e} "
              f"{r['dSy_dt']:>14.3e} {r['dSy_dt_noco']:>14.3e} "
              f"{r['supp_ratio']:>9.2f} {r['lin_exp']:>7.2f}")
    print(f"\n  toplam duvar-saati: {wall:.0f}s  "
          f"(seri tahmini ~{sum(r['runtime'] for r in results):.0f}s)")

    # ── Otomatik yorum ───────────────────────────────────────────────────────
    _interpret(results, do_co, diag_noco)
    return results, Br_target


def _interpret(results, do_co, diag_noco):
    """N=0/N=1 baskınlığı fizik mi artefakt mı — otomatik teşhis."""
    by_N = {r["N"]: r for r in results}
    print("\n  " + "-" * 60)
    print("  TEŞHİS:")
    d0 = abs(by_N.get(0, {}).get("dSy_dt", np.nan))
    d1 = abs(by_N.get(1, {}).get("dSy_dt", np.nan))
    d2 = abs(by_N.get(2, {}).get("dSy_dt", np.nan))
    if np.isfinite(d0) and np.isfinite(d2) and d2 > 0:
        print(f"    N=0 / N=2 = {d0/d2:.1f}×   (N=0 / N=1 = "
              f"{d0/d1:.1f}× )" if d1 > 0 else f"    N=0 / N=2 = {d0/d2:.1f}×")
    if diag_noco and do_co:
        r0 = by_N.get(0, {}).get("supp_ratio", np.nan)
        if np.isfinite(r0):
            if r0 < 1.3:
                print(f"    N=0: NO/WITH = {r0:.2f} → CO telafisi N=0'ı "
                      f"BASTIRAMIYOR. CO kayması yakalanmıyorsa ARTEFAKT "
                      f"şüphesi → --co-refine ile tekrar deneyin.")
            else:
                print(f"    N=0: NO/WITH = {r0:.2f} → CO telafisi N=0'ı "
                      f"{r0:.1f}× bastırıyor; WITH değeri gerçek artıktır.")
    co0 = by_N.get(0, {}).get("co_mm", np.nan)
    if np.isfinite(co0) and abs(co0) < 1e-7:
        print(f"    UYARI: N=0 CO kayması ≈ {co0:.2e} mm ≈ 0 → bulucu orbit "
              f"kaymasını yakalamamış olabilir. --co-refine deneyin.")
    print("  " + "-" * 60)


def plot_results(results, Br_target, misalign_data=None, outdir="."):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ns   = np.array([r["N"]                  for r in results])
    dsy  = np.array([abs(r["dSy_dt"])        for r in results])
    dnc  = np.array([abs(r.get("dSy_dt_noco", np.nan)) for r in results])

    BLUE  = "#2166ac"
    GRAY  = "#888888"
    RED   = "#d6604d"
    Br_nT = Br_target * 1e9

    # ── Fig A: B_x alan harmoniği taraması (with-CO + no-CO) ──────────────────
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(Ns, dsy, "o-", color=BLUE, lw=2, ms=7,
                label=rf"$|dS_y/dt|$ WITH-CO ($B_x={Br_nT:.1f}$ nT)")
    if np.any(np.isfinite(dnc)):
        ax.semilogy(Ns, dnc, "s--", color=GRAY, lw=1.5, ms=5, alpha=0.8,
                    label=r"$|dS_y/dt|$ NO-CO (telafi yok)")
    ax.set_xlabel("Radial-field harmonic $N$")
    ax.set_ylabel(r"$|dS_y/dt|$ [rad/s]")
    ax.set_title(r"Vertical spin precession rate vs. $B_x$ field harmonic" + "\n"
                 r"(Omarov 2022 Fig. 8 analog, CW beam, $Q_y \approx 2.68$)")
    ax.set_xticks(Ns)
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.4)
    for Nr in [2, 3]:
        if Nr in Ns:
            ax.axvline(Nr, color=RED, lw=0.8, ls="--", alpha=0.6)
    ax.text(0.02, 0.04, r"$N\approx Q_y$ rezonansı", transform=ax.transAxes,
            color=RED, fontsize=9)
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
        ax0.set_xlabel("Radial-field harmonic $N$")
        ax0.set_ylabel(r"$|dS_y/dt|$ [rad/s]")
        ax0.set_title(rf"$B_x = {Br_nT:.1f}$ nT radyal alan harmoniği"
                      "\n(bu halka, CW)")
        ax0.set_xticks(Ns)
        ax0.grid(True, which="both", ls=":", alpha=0.4)
        for Nr in [2, 3]:
            if Nr in Ns:
                ax0.axvline(Nr, color=RED, lw=0.8, ls="--", alpha=0.6)

        ax1 = axes[1]
        ks_m  = np.array([r["k"]           for r in misalign_data])
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

        fig.suptitle(r"$B_x$ alan harmoniği vs. quad kaçıklık harmoniği "
                     r"— $Q_y\approx2.68$", fontsize=12)
        fig.tight_layout()
        out_b = os.path.join(outdir, "field_vs_misalign.png")
        fig.savefig(out_b, dpi=150)
        print(f"  -> {out_b}")
        plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Omarov Fig. 8 analog: dS_y/dt vs radyal B_x alan harmoniği N"
                    " (+ N=0/N=1 artefakt tanısı, paralel)")
    p.add_argument("--Nmax", type=int, default=12,
                   help="Taranacak maksimum harmonic sayısı (default 12)")
    p.add_argument("--Br", type=float, default=1e-9,
                   help="B_x genliği [T] (default 1e-9 = 1 nT, Omarov parametresi)")
    p.add_argument("--t2", type=float, default=5e-3,
                   help="Simülasyon süresi [s] (default 5ms)")
    p.add_argument("--no-co", action="store_true",
                   help="kapalı yörünge fırlatmasını KAPAT")
    p.add_argument("--co-turns", type=int, default=60)
    p.add_argument("--co-iter", type=int, default=3,
                   help="kapalı yörünge Newton yineleme sayısı (default 3)")
    p.add_argument("--co-refine", action="store_true",
                   help="ADAPTİF sonlu-fark adımı: küçük (nT) alanların ürettiği "
                        "nm-mertebesi orbit kaymasını çözmek için (N=0 tanısı)")
    p.add_argument("--no-diag", action="store_true",
                   help="NO-CO tanı ölçümünü kapat (yalnız with-CO, daha hızlı)")
    p.add_argument("--linearity", action="store_true",
                   help="her N için A_r/2'de de ölç → ölçekleme üssü raporla")
    p.add_argument("--nproc", type=int, default=None,
                   help="paralel süreç sayısı (default: min(N sayısı, CPU))")
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--misalign-log", type=str, default=None,
                   help="false_edm_mode_scan.py JSON çıktısı (yan yana plot için)")
    args = p.parse_args()

    results, Br_target = run_scan(
        N_max=args.Nmax, Br_target=args.Br, t2=args.t2,
        do_co=not args.no_co, co_turns=args.co_turns, co_iter=args.co_iter,
        co_refine=args.co_refine, diag_noco=not args.no_diag,
        lin_half=args.linearity, dt=args.dt, nproc=args.nproc)

    misalign_data = None
    if args.misalign_log and os.path.exists(args.misalign_log):
        with open(args.misalign_log) as f:
            misalign_data = json.load(f)

    plot_results(results, Br_target, misalign_data=misalign_data)

    out_json = "field_harmonic_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  JSON: {out_json}")
