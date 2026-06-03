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
    # Başlangıç koşulu: IDEAL kapalı yörünge (x=y=0, spin boylamsal frozen).
    # params.json'daki dev0/y0 değerleri KULLANILMIYOR — onlar genel simülasyon
    # için; burada amacımız misalignment'tan gelen saf false-EDM sinyalini ölçmek.
    # Herhangi bir x/y sapması, misalignmentsiz lattisin COD'ından bağımsız bir
    # betatron titreşimi yaratır ve dSy/dt'ye arka plan katkı ekler (tüm k için
    # aynı → modlar ayırt edilemez). x=y=0 ile bu arka plan sıfırlanır.
    y0 = [0.0, 0.0, 0.0,
          0.0, 0.0, p_mag * direction,
          0.0, 0.0, direction]   # spin başlangıç: boylamsal frozen (Sy=0)
    return f, y0, beta0, R0


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

    Proje standardı (plot_results.py / run_simulation.py): büyük pencereli
    Savitzky-Golay (pencere = N/4) ile hızlı g-2 ve betatron salınımları
    alçak-geçirenle bastırılır, kenarların %10'u atılıp doğru fit edilir.
    Salınımın çok sayıda periyodu olduğunda (UZUN simülasyon) sekuler eğim
    baskın hale gelir ve kestirim kararlılaşır.

    DİKKAT — turda-bir (Poincaré) örnekleme ALIASING yaratır (Q_y≈0.68 ile
    sahte yavaş salınım); bu yüzden SÜREKLİ veri kullanılır.
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
    k, amp_coef, t2, return_steps, dt = task
    # worker içinde taze import (multiprocessing 'spawn' güvenliği)
    import os, json, time
    import numpy as np
    from integrator import integrate_particle
    from fourier_reconstruct import fodo_basis
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open("params.json") as f:
        config = json.load(f)
    fields, y0, beta0, R0 = setup_fields(config)
    n_q = 2 * int(fields.nFODO)
    antisym = config.get("smooth_antisym_fodo", True)

    F_k, _ = fodo_basis(n_q, [k], antisym)
    mode = F_k[:, 0]
    quad_dy = amp_coef * mode

    # R-tabanlı ‖RF_k‖ SADECE referans amaçlı (fizik için gerekmez; bu
    # simülasyon saf demet/spin dinamiğidir). Dosya yoksa ya da bayat/sıfır
    # ise atlanır — gerçek rezonans aşağıda entegre yörüngeden ölçülür.
    R = np.load("R_dy_1.npy") if os.path.exists("R_dy_1.npy") else None
    if R is not None and np.linalg.norm(R) > 1e-12:
        RFk = float(np.linalg.norm(R @ (mode / np.linalg.norm(mode))))
    else:
        RFk = None

    t0 = time.time()
    hist, _, _ = integrate_particle(
        y0, 0.0, t2, dt, fields=fields, return_steps=return_steps,
        quad_dy=quad_dy)
    t_array = np.arange(hist.shape[0]) * (t2 / hist.shape[0])
    slope = float(measure_dSy_dt(hist, t_array))
    # R-bağımsız orbit teşhisi: entegre edilen dikey yörünge genliği [mm]
    y_orbit_mm = float(np.std(hist[:, 1]) * 1e3)
    sy = hist[:, 7].copy()   # S_y zaman serisi (grafik için)
    dt_run = time.time() - t0
    return {"k": k, "RFk": RFk, "orbit_mm": y_orbit_mm,
            "dSy_dt": slope, "runtime": dt_run,
            "sy": sy, "t_array": t_array}


def run_scan(k_list, amp_coef=1e-5, t2=5e-4, return_steps=5000, nproc=None):
    with open("params.json") as f:
        config = json.load(f)
    fields, y0, beta0, R0 = setup_fields(config)

    circ = 2*np.pi*R0 + 4*fields.nFODO*fields.driftLen + 2*fields.nFODO*fields.quadLen
    T_rev = circ / (beta0 * C)
    dt = config.get("dt", 1e-11)

    if nproc is None:
        # mod başına bir süreç: tüm k'lar tek dalgada paralel koşsun
        # (kmax=5 → 6 mod → 6 süreç). Bağımsız C++ işleri, az sayıda mod
        # olduğundan hafif aşırı-abonelik sorun değil. CPU sayısıyla
        # sınırlamak istersen --nproc ile elle ver.
        nproc = len(k_list)

    print("=" * 68)
    print("  FALSE EDM — FOURIER MODU TARAMASI")
    print(f"  amplitude (cos katsayısı) = {amp_coef*1e6:.1f} μm  (tüm modlar eşit)")
    print(f"  t2 = {t2:.1e} s  (~{t2/T_rev:.0f} tur),  EDMSwitch = 0 (gerçek EDM yok)")
    print(f"  paralel süreç sayısı = {nproc}")
    print("=" * 68)

    tasks = [(k, amp_coef, t2, return_steps, dt) for k in k_list]
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
    has_R = any(r["RFk"] is not None for r in results)
    print(f"  {'k':>3}  {'‖RF_k‖':>9}  {'y-orbit (sim)':>14}  {'dS_y/dt [rad/s]':>18}")
    for r in results:
        rfk_s = f"{r['RFk']:>9.3f}" if r["RFk"] is not None else f"{'—':>9}"
        print(f"  {r['k']:>3}  {rfk_s}  {r['orbit_mm']:>12.3f}mm  "
              f"{r['dSy_dt']:>18.3e}   ({r['runtime']:.0f}s)")
    if not has_R:
        print("  (‖RF_k‖ referansı atlandı: R_dy_1.npy yok/sıfır — fizik için "
              "gerekmez; y-orbit sütunu R-bağımsız rezonans göstergesidir)")
    print(f"  toplam duvar-saati: {wall:.0f}s  "
          f"(seri tahmini ~{sum(r['runtime'] for r in results):.0f}s)")

    return results, config


def plot_results(results, amp_coef):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks   = np.array([r["k"] for r in results])
    dsy  = np.array([abs(r["dSy_dt"]) for r in results])
    # R-bağımsız karşılaştırma: simülasyondan entegre edilen y-orbit genliği.
    # (‖RF_k‖ varsa onu da gösterebiliriz ama gerekmez.)
    orbit = np.array([r["orbit_mm"] for r in results])

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    color1 = "tab:red"
    ax1.bar(ks, dsy, width=0.55, color=color1, alpha=0.75,
            label=r"$|dS_y/dt|$ (false EDM)")
    ax1.set_xlabel("Fourier mode $k$ of quad misalignment")
    ax1.set_ylabel(r"$|dS_y/dt|$  [rad/s]  (false EDM signal)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(ks)

    ax2 = ax1.twinx()
    color2 = "tab:blue"
    ax2.plot(ks, orbit, "o-", color=color2, lw=2, ms=7,
             label=r"integrated $y$-orbit amplitude")
    ax2.set_ylabel(r"$y$-orbit amplitude [mm] (from tracking)", color=color2)
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

        ax.plot(t_ms, sy, lw=0.4, alpha=0.45, color="gray")

        win = (len(sy) // 4) * 2 + 1
        sy_f = _savgol_or_movavg(sy, win)
        ax.plot(t_ms, sy_f, lw=1.4, color="tab:blue", label="smoothed")

        n_pts = len(sy)
        trim = int(n_pts * 0.1)
        tt = t_s[trim:-trim] if trim > 0 and n_pts - 2*trim > 10 else t_s
        yy = sy_f[trim:-trim] if trim > 0 and n_pts - 2*trim > 10 else sy_f
        coef = np.polyfit(tt, yy, 1)
        ax.plot(t_ms, np.polyval(coef, t_s), "--", lw=1.4, color="tab:red",
                label=f"{r['dSy_dt']:.2e} rad/s")

        ax.set_title(f"k = {k}  |  orbit {r['orbit_mm']:.3f} mm", fontsize=10)
        ax.set_xlabel("t [ms]", fontsize=9)
        ax.set_ylabel(r"$S_y$", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

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
    args = p.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    k_list = list(range(0, args.kmax + 1))
    results, config = run_scan(k_list, amp_coef=args.amp, t2=args.t2,
                               return_steps=args.steps, nproc=args.nproc)
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
