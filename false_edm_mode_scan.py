#!/usr/bin/env python3
"""false_edm_mode_scan.py — False-EDM vs Fourier mode of quad misalignment.

Amaç (makalenin kalbi):
  "k=2 modu false EDM'yi domine ediyor" iddiasını spin takibiyle KANITLAMAK.

Yöntem:
  Her k = 0,1,2,3,4,5 modu için quad dikey misalignment'ı
      Δy_j = A · F_k[j]          (FODO-antisimetrik Fourier modu)
  olarak ver, gerçek EDM kapalı (EDMSwitch=0), spin takibi yap ve
  dikey spin presesyon hızını ölç:
      false EDM sinyali  ≡  dS_y/dt   [rad/s]

  Tüm modlar için A aynı (10 μm cos katsayısı, makaledeki kurulum).
  Beklenti: dS_y/dt, orbit kazancı ‖RF_k‖ ile birlikte k=2'de zirve yapar.

Çıktı:
  - false_edm_mode_scan.png  (|dS_y/dt| vs k, ‖RF_k‖ ile karşılaştırma)
  - terminal tablosu
"""
import json
import time
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from integrator import integrate_particle, FieldParams
from fourier_reconstruct import fodo_basis

# ── Fiziksel sabitler / magic momentum ───────────────────────────────────────
M2  = 0.938272046     # proton kütlesi [GeV/c^2]
AMU = 1.792847356     # anormal manyetik moment
C   = 299792458.0     # ışık hızı [m/s]
M1  = 1.672621777e-27 # proton kütlesi [kg]


def setup_fields(config):
    p_magic_base = M2 / np.sqrt(AMU)
    E_tot  = np.sqrt(p_magic_base**2 + M2**2)
    beta0  = p_magic_base / E_tot
    gamma0 = 1.0 / np.sqrt(1.0 - beta0**2)
    R0     = config["R0"]
    E0_V_m = -(p_magic_base * (p_magic_base / np.sqrt(p_magic_base**2 + M2**2)) / R0) * 1e9

    f = FieldParams()
    f.R0 = R0
    f.E0 = E0_V_m
    f.E0_power = config.get("E0_power", 1.0)
    f.quadG1 = config.get("g1", 0.21)
    f.quadG0 = config.get("g0", f.quadG1)
    f.sextK1 = config.get("sextK1", 0.0)
    f.quadSwitch = float(config.get("quadSwitch", 1))
    f.sextSwitch = float(config.get("sextSwitch", 0))
    f.EDMSwitch  = 0.0    # GERÇEK EDM KAPALI — sadece false EDM ölçüyoruz
    f.direction  = float(config.get("direction", -1))
    f.nFODO    = float(config.get("nFODO", 24))
    f.quadLen  = float(config.get("quadLen", 0.4))
    f.driftLen = float(config.get("driftLen", 2.0833))
    f.poincare_quad_index = -1.0
    f.rfSwitch = 0.0
    f.h = float(config.get("h", 100))

    p_mag = gamma0 * M1 * C * beta0
    direction = f.direction
    # Başlangıç koşulu: spin boylamsal frozen (Sy=0). Yörünge başlangıcı
    # AŞAĞIDA find_closed_orbit ile kaymış kapalı yörüngeye oturtulur
    # (betatron salınımını yok etmek için — bkz. measure_dSy_dt notu).
    y0 = [0.0, 0.0, 0.0,
          0.0, 0.0, p_mag * direction,
          0.0, 0.0, direction]   # spin başlangıç: boylamsal frozen (Sy=0)
    return f, y0, beta0, R0, p_mag, direction


def _make_state(v, p_mag, direction, spin):
    """(x0, y0, x', y') faz-uzayı noktasından 9-bileşenli yerel başlangıç durumu.
    Açılar boyutsuz (x'=px/pz); momentuma p_mag·direction ile çevrilir."""
    return [v[0], v[1], 0.0,
            p_mag * direction * v[2], p_mag * direction * v[3], p_mag * direction,
            spin[0], spin[1], spin[2]]


def find_closed_orbit(fields, p_mag, direction, quad_dy, dt, T_rev,
                      n_turns=48, n_iter=2, verbose=False):
    """Kaymış kapalı yörüngenin dikey fırlatma noktasını (y0, y') bulur.

    NEDEN: quad misalignment kapalı yörüngeyi kaydırır. İdeal y=0'dan
    fırlatırsak parçacık yeni kapalı yörünge etrafında betatron salınımı yapar
    (genlik ~ kapalı-yörünge ofseti ~0.2mm). Bu salınımın S_y'ye katkısı (~1e-5)
    aradığımız seküler false-EDM driftinden (~1e-8) ~1000× büyük ve hiçbir
    estimator onu ayıramaz. Çözüm: parçacığı kapalı yörünge ÜZERİNDE fırlat →
    betatron yok → S_y saf seküler drift.

    YÖNTEM: sabit azimutta (Poincaré, tur-başına) betatron VARYANSI,
    fırlatma noktasının kapalı-yörüngeden sapmasının tam KUADRATİK formudur
    (sextSwitch=0 → lineer lattis). 2B (y, y') faz-uzayında varyans bir eğik
    elips (çapraz terim Hyp≠0) → koordinat-bazlı iniş bu vadide YAVAŞ ilerler.
    Bunun yerine sonlu-fark Hessian'ı (çapraz terim dahil) kurup tek Newton
    adımıyla minimuma (= kapalı yörünge) atlanır; lineer lattis için 1 adım
    yeterli, varyans dalgalanmasına karşı 1-2 yineleme ile sağlamlaştırılır.
    Quad_dy düz halkada SADECE dikey yörüngeyi kaydırır (x ile kuplaj yok,
    quad_tilt=0) → yatay düzlem y=0'da bırakılır.
    Azimut-bağımsız: betatron aksiyonu J→0 olunca parçacık HER azimutta
    kapalı yörünge üzerindedir.
    """
    from integrator import integrate_particle
    spin = [0.0, 0.0, direction]
    fields.poincare_quad_index = 0.0   # cell-0 ARC1 girişi: tur-başına 1 örnek
    t_probe = n_turns * T_rev

    def var2(yc, ypc):
        """Sabit azimutta tur-başına dikey konumun VARYANSI [m²].
        Kapalı yörüngede sabit → 0; sapmada ∝ sapma² (kuadratik)."""
        st = _make_state([0.0, yc, 0.0, ypc], p_mag, direction, spin)
        _, poin, _ = integrate_particle(st, 0.0, t_probe, dt, fields=fields,
                                        return_steps=10, quad_dy=quad_dy)
        if poin is None or len(poin) < 5:
            return 1e30
        return float(np.var(poin[:, 1]))

    yc, ypc = 0.0, 0.0
    sy, syp = 2e-4, 2e-5         # finite-fark adımları (y [m], y' [rad])
    for it in range(n_iter):
        f0   = var2(yc,      ypc)
        fp_y = var2(yc + sy, ypc);  fm_y = var2(yc - sy, ypc)
        fp_p = var2(yc, ypc + syp); fm_p = var2(yc, ypc - syp)
        fpp  = var2(yc + sy, ypc + syp)
        gy = (fp_y - fm_y) / (2*sy)
        gp = (fp_p - fm_p) / (2*syp)
        Hyy = (fp_y - 2*f0 + fm_y) / (sy*sy)
        Hpp = (fp_p - 2*f0 + fm_p) / (syp*syp)
        Hyp = (fpp - fp_y - fp_p + f0) / (sy*syp)
        det = Hyy*Hpp - Hyp*Hyp
        if det <= 0 or Hyy <= 0:     # pozitif-tanımlı değil → güvenli çık
            if verbose:
                print(f"    [CO iter {it}] Hessian pos-def değil, durdu")
            break
        dy = -( Hpp*gy - Hyp*gp) / det
        dp = -(-Hyp*gy + Hyy*gp) / det
        yc += dy; ypc += dp
        if verbose:
            rms = np.sqrt(max(var2(yc, ypc), 0.0))
            print(f"    [CO iter {it}] resid betatron rms = {rms:.3e} m, "
                  f"y={yc:.6e} y'={ypc:.6e}")
        sy *= 0.2; syp *= 0.2        # min'e yaklaşınca daha küçük fark adımı
    resid_rms = np.sqrt(max(var2(yc, ypc), 0.0))   # sabit-azimut betatron RMS
    fields.poincare_quad_index = -1.0   # ana koşum için Poincaré'yi kapat
    return np.array([0.0, yc, 0.0, ypc]), float(resid_rms)


def _savgol_or_movavg(sig, win):
    """Savitzky-Golay (scipy varsa) ya da numpy hareketli-ortalama fallback."""
    if win < 5:
        return sig.copy()
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(sig, window_length=win, polyorder=1)
    except Exception:
        # numpy-only fallback: kenar yansıtmalı kayan ortalama
        k = win
        pad = k // 2
        ext = np.concatenate([sig[pad:0:-1], sig, sig[-2:-pad-2:-1]])
        kern = np.ones(k) / k
        return np.convolve(ext, kern, mode="same")[pad:pad+len(sig)]


def measure_dSy_dt(hist, t_array):
    """S_y'nin sekuler (false-EDM) eğimi [rad/s].

    KRİTİK ÖN KOŞUL — parçacık KAPALI YÖRÜNGE üzerinde fırlatılmalı
    (find_closed_orbit). Aksi halde betatron salınımı (genlik ~1e-5) seküler
    drift'i (~1e-8) ~1000× domine eder; salınımın lineer fite sızması işaret
    ve büyüklük olarak ÇIKTI örneklemesine göre değişir → ölçüm anlamsız.
    Bu, betatron amplitüdünün C, tur sayısının N olduğu durumda sızma tabanı
    ~C/N ≈ 5e-9 rad/tur ≫ sinyal 5e-12 rad/tur olmasından kaynaklanır:
    hiçbir estimator betatron varken sinyali çıkaramaz. Kapalı yörünge
    üzerinde fırlatılırsa betatron yok → S_y düz seküler çizgi.

    Proje standardı (plot_results.py / run_simulation.py): büyük pencereli
    Savitzky-Golay (pencere = N/4) ile kalan hızlı g-2 salınımı bastırılır,
    kenarların %10'u atılıp düz çizgi fit edilir.
    """
    sy = np.asarray(hist[:, 7], float)
    t  = np.asarray(t_array, float)
    n  = len(sy)
    win = (n // 4) * 2 + 1
    sy_f = _savgol_or_movavg(sy, win)
    trim = int(n * 0.1)
    if trim > 0 and n - 2 * trim > 10:
        tt, yy = t[trim:-trim], sy_f[trim:-trim]
    else:
        tt, yy = t, sy_f
    slope, _ = np.polyfit(tt, yy, 1)
    return slope


def _run_one_k(task):
    """Tek bir k modu için simülasyon + eğim ölçümü (paralel worker).

    Her alt-süreç integrator'ı (ctypes lib) yeniden yükler → C++ çağrıları
    süreçler arası paylaşımsız, güvenle paralel çalışır.
    """
    k, amp_coef, t2, return_steps, dt, do_co, co_turns = task
    # worker içinde taze import (multiprocessing 'spawn' güvenliği)
    import os, json, time
    import numpy as np
    from integrator import integrate_particle
    from fourier_reconstruct import fodo_basis
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open("params.json") as f:
        config = json.load(f)
    fields, y0, beta0, R0, p_mag, direction = setup_fields(config)
    n_q = 2 * int(fields.nFODO)
    antisym = config.get("smooth_antisym_fodo", True)

    F_k, _ = fodo_basis(n_q, [k], antisym)
    mode = F_k[:, 0]
    quad_dy = amp_coef * mode

    circ = (2*np.pi*R0 + 4*fields.nFODO*fields.driftLen
            + 2*fields.nFODO*fields.quadLen)
    T_rev = circ / (beta0 * C)

    t0 = time.time()
    # ── Kapalı yörünge fırlatması (betatron salınımını yok eder) ──
    if do_co:
        v_co, resid_rms = find_closed_orbit(fields, p_mag, direction, quad_dy,
                                            dt, T_rev, n_turns=co_turns)
        y_launch = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
        co_off_mm = float(np.hypot(v_co[0], v_co[1]) * 1e3)
        resid_beta_mm = resid_rms * 1e3   # sabit-azimut kalan betatron [mm]
    else:
        y_launch = y0
        co_off_mm = 0.0
        resid_beta_mm = float("nan")

    # Sabit azimutta tur-başına örnekleme (STROBOSKOPİK). CO fırlatmasıyla
    # tur-başına spin haritası sabit bir dönme → bu azimutta her tur AYNI
    # faz → tur-içi (~1e-5) salınım TAMAMEN çıkar; geriye seküler false-EDM
    # driftı (+ yavaş spin-tune dalgası) kalır. SG/örnekleme-oranı yok.
    fields.poincare_quad_index = 0.0
    hist, poin, poin_t = integrate_particle(
        y_launch, 0.0, t2, dt, fields=fields, return_steps=return_steps,
        quad_dy=quad_dy)
    t_array = np.arange(hist.shape[0]) * (t2 / hist.shape[0])

    # Birincil ölçüm: stroboskopik S_y(tur) doğrusal eğimi [rad/s].
    sy_strobe = np.asarray(poin[:, 7], float) if poin is not None and len(poin) > 5 else None
    if sy_strobe is not None:
        ts = np.asarray(poin_t, float)
        slope = float(np.polyfit(ts, sy_strobe, 1)[0])
        # Richardson integration-error estimate: re-run at 2·dt (same y_launch,
        # no CO redo — half the steps, ~50% extra wall time per k).
        # |slope(dt) - slope(2dt)| ≈ leading-order truncation error of slope(dt).
        _, poin_c, poin_t_c = integrate_particle(
            y_launch, 0.0, t2, 2*dt, fields=fields, return_steps=return_steps,
            quad_dy=quad_dy)
        sy_c = (np.asarray(poin_c[:, 7], float)
                if poin_c is not None and len(poin_c) > 5 else None)
        if sy_c is not None:
            slope_c = float(np.polyfit(np.asarray(poin_t_c, float), sy_c, 1)[0])
            slope_err = abs(slope - slope_c)
        else:
            slope_err = float("nan")
    else:
        ts = None
        slope = float("nan")
        slope_err = float("nan")
    # Karşılaştırma: eski sürekli-SG yöntemi (örnekleme-bağımlı, güvenilmez)
    slope_sg = float(measure_dSy_dt(hist, t_array))
    sy = hist[:, 7].copy()   # sürekli S_y (hızlı salınım bağlamı için)
    dt_run = time.time() - t0
    # resid_beta_mm: sabit azimutta kalan betatron RMS (CO aramasından).
    # Kapalı yörünge üzerinde fırlatıldıysa ~0 (≪ CO ofseti) → temiz fırlatma.
    return {"k": k, "co_off_mm": co_off_mm, "resid_beta_mm": resid_beta_mm,
            "dSy_dt": slope, "dSy_dt_err": slope_err,
            "dSy_dt_sg": slope_sg, "runtime": dt_run,
            "sy": sy, "t_array": t_array,
            "sy_strobe": (sy_strobe.copy() if sy_strobe is not None else None),
            "t_strobe": (ts.copy() if ts is not None else None)}


def run_scan(k_list, amp_coef=1e-5, t2=5e-4, return_steps=5000, nproc=None,
             do_co=True, co_turns=60, dt=None):
    with open("params.json") as f:
        config = json.load(f)
    fields, y0, beta0, R0, p_mag, direction = setup_fields(config)

    circ = 2*np.pi*R0 + 4*fields.nFODO*fields.driftLen + 2*fields.nFODO*fields.quadLen
    T_rev = circ / (beta0 * C)
    if dt is None:                       # --dt verilmezse config'ten al
        dt = config.get("dt", 1e-11)

    if nproc is None:
        # mod başına bir süreç: tüm k'lar tek dalgada paralel koşsun
        # (kmax=5 → 6 mod → 6 süreç). Bağımsız C++ işleri, az sayıda mod
        # olduğundan hafif aşırı-abonelik sorun değil. CPU sayısıyla
        # sınırlamak istersen --nproc ile elle ver.
        nproc = len(k_list)

    print("=" * 72)
    print("  FALSE EDM — FOURIER MODU TARAMASI")
    print(f"  amplitude (cos katsayısı) = {amp_coef*1e6:.1f} μm  (tüm modlar eşit)")
    print(f"  t2 = {t2:.1e} s  (~{t2/T_rev:.0f} tur),  dt = {dt:.1e} s,  "
          f"EDMSwitch = 0 (gerçek EDM yok)")
    print(f"  kapalı yörünge fırlatması = {'AÇIK' if do_co else 'KAPALI'}"
          f"{f' ({co_turns} tur prob)' if do_co else ''}")
    print(f"  paralel süreç sayısı = {nproc}")
    print("=" * 72)

    tasks = [(k, amp_coef, t2, return_steps, dt, do_co, co_turns) for k in k_list]
    t_wall = time.time()
    if nproc > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(nproc) as pool:
            results = pool.map(_run_one_k, tasks)
    else:
        results = [_run_one_k(t) for t in tasks]
    wall = time.time() - t_wall

    results.sort(key=lambda r: r["k"])
    print(f"  {'k':>3}  {'CO ofset':>10}  {'kalan betatron':>16}  "
          f"{'dS_y/dt strobe':>16}  {'(SG, güvenilmez)':>16}")
    for r in results:
        print(f"  {r['k']:>3}  {r['co_off_mm']:>8.3f}mm  "
              f"{r['resid_beta_mm']:>14.2e}mm  "
              f"{r['dSy_dt']:>16.3e}  {r.get('dSy_dt_sg', float('nan')):>16.2e}"
              f"   ({r['runtime']:.0f}s)")
    if do_co:
        print("  (dS_y/dt strobe = sabit-azimut tur-başına örneklemeden seküler "
              "eğim [rad/s]; SG sütunu eski sürekli yöntem, örnekleme-bağımlı)")
    print(f"  toplam duvar-saati: {wall:.0f}s  "
          f"(seri tahmini ~{sum(r['runtime'] for r in results):.0f}s)")

    return results, config


def plot_results(results, amp_coef):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks      = np.array([r["k"] for r in results])
    dsy     = np.array([abs(r["dSy_dt"]) for r in results])
    dsy_err = np.array([r.get("dSy_dt_err", 0.0) for r in results])
    # Orbit rezonans göstergesi: bulunan kapalı-yörünge fırlatma ofseti
    # (parçacık bunun üzerine oturtulduğu için kalan betatron ~0'dır).
    orbit = np.array([r["co_off_mm"] for r in results])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "tab:red"
    ax1.errorbar(ks, dsy, yerr=dsy_err, fmt="o-", color=color1, lw=1.8,
                 ms=8, mfc="white", mew=2, capsize=4, capthick=1.5,
                 elinewidth=1.3,
                 label=r"$|dS_y/dt|$  [err: Richardson $\Delta t$ convergence]")
    ax1.set_yscale("log")
    ax1.set_xlabel("Fourier mode $k$ of quad misalignment", fontsize=12)
    ax1.set_ylabel(r"$|dS_y/dt|$  [rad/s]  (false EDM signal)", color=color1,
                   fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(ks)

    ax2 = ax1.twinx()
    color2 = "tab:blue"
    ax2.plot(ks, orbit, "o-", color=color2, lw=2, ms=7,
             label=r"closed-orbit amplitude")
    ax2.set_ylabel(r"closed-orbit amplitude [mm]", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(f"False EDM resonance at $k=2$ "
                  f"(misalignment amp = {amp_coef*1e6:.0f} $\\mu$m, all modes)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               fontsize=9)
    fig.tight_layout()
    fig.savefig("false_edm_mode_scan.png", dpi=140)
    print("\n  -> false_edm_mode_scan.png kaydedildi")


def plot_sy_timeseries(results, amp_coef):
    """Her k modu için S_y zaman serisini 2×3 grid halinde çizer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = sorted(results, key=lambda x: x["k"])
    ncols = 3
    nrows = (len(res) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows))
    axes = np.array(axes).flatten()

    for i, r in enumerate(res):
        ax = axes[i]
        k  = r["k"]
        sy = np.asarray(r["sy"])
        t_s = np.asarray(r["t_array"])
        t_ms = t_s * 1e3

        # Sol eksen (gri): sürekli S_y — tur-içi ~1e-5 salınım (ham bağlam)
        ax.plot(t_ms, sy, lw=0.4, alpha=0.30, color="gray")
        ax.set_ylabel(r"$S_y$ (sürekli)", fontsize=8, color="gray")
        ax.tick_params(axis="y", labelcolor="gray", labelsize=7)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

        # Sağ eksen (mavi/kırmızı): stroboskopik S_y, DC ofset (fit kesişimi)
        # çıkarılmış → SEKÜLER DRIFT artık kendi ölçeğinde GÖRÜNÜR.
        if r.get("sy_strobe") is not None:
            ts = np.asarray(r["t_strobe"]); ss = np.asarray(r["sy_strobe"])
            fit = np.polyfit(ts, ss, 1)               # [slope, intercept]
            axr = ax.twinx()
            axr.plot((ts*1e3), (ss - fit[1]) * 1e9, ".", ms=2.6,
                     color="tab:blue", label="stroboskopik − DC")
            axr.plot((ts*1e3), (np.polyval(fit, ts) - fit[1]) * 1e9, "-",
                     lw=1.8, color="tab:red",
                     label=f"eğim {r['dSy_dt']:.2e} rad/s")
            axr.set_ylabel(r"$\Delta S_y$ stroboskopik [$\times10^{-9}$]",
                           fontsize=8, color="tab:red")
            axr.tick_params(axis="y", labelcolor="tab:red", labelsize=7)
            axr.legend(fontsize=6.5, loc="lower right")

        ax.set_title(f"k = {k}  |  CO {r['co_off_mm']:.3f} mm  "
                     f"|  resid β {r['resid_beta_mm']:.1e} mm", fontsize=9)
        ax.set_xlabel("t [ms]", fontsize=9)

    for j in range(len(res), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        rf"$S_y$ zaman serisi — Fourier modu taraması  "
        f"($A$ = {amp_coef*1e6:.0f} μm, EDMSwitch=0)",
        fontsize=12)
    fig.tight_layout()
    fig.savefig("false_edm_sy_traces.png", dpi=140)
    print("  -> false_edm_sy_traces.png kaydedildi")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--kmax", type=int, default=5)
    p.add_argument("--amp", type=float, default=1e-5, help="cos katsayısı [m]")
    p.add_argument("--t2", type=float, default=5e-4, help="simülasyon süresi [s]")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--nproc", type=int, default=None,
                   help="paralel süreç sayısı (varsayılan: mod sayısı = kmax+1)")
    p.add_argument("--no-co", action="store_true",
                   help="kapalı yörünge fırlatmasını KAPAT (eski davranış; "
                        "betatron salınımı sinyali domine eder)")
    p.add_argument("--co-turns", type=int, default=60,
                   help="kapalı yörünge arama probu için tur sayısı")
    p.add_argument("--dt", type=float, default=None,
                   help="entegrasyon adımı [s] (varsayılan: params.json'dan). "
                        "dt-yakınsama testi için override")
    args = p.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    k_list = list(range(0, args.kmax + 1))
    results, config = run_scan(k_list, amp_coef=args.amp, t2=args.t2,
                               return_steps=args.steps, nproc=args.nproc,
                               dt=args.dt,
                               do_co=not args.no_co, co_turns=args.co_turns)
    plot_results(results, args.amp)
    plot_sy_timeseries(results, args.amp)

    # özet
    dsy = {r["k"]: abs(r["dSy_dt"]) for r in results}
    kmax_signal = max(dsy, key=dsy.get)
    print(f"\n  ÖZET: en büyük false EDM sinyali  k={kmax_signal}")
    if 2 in dsy and dsy[2] > 0:
        for k in sorted(dsy):
            if k != 2:
                print(f"    k=2 / k={k}:  {dsy[2]/dsy[k]:6.1f}×" if dsy[k] > 0
                      else f"    k=2 / k={k}:  inf")
