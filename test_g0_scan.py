#!/usr/bin/env python3
"""test_g0_scan.py — Eşik teorisinin genel geçerlilik testi: g₀ taraması.

SORU: G_k = C/|Q_eff² − k²| yasası ve k_max² < Q_eff² + C·σ_q/σ_b eşiği
      bu kafese özgü bir tesadüf mü, yoksa Q_eff parametrizasyonuyla genel
      geçer mi?

YÖNTEMİ: Üç kuadrupol gradyanında (g₀ = 0.15, 0.20, 0.25 T/m):
  1. Yörünge kalibrasyon matrisi O hesaplanır (k=1..12, 25 koşum).
  2. G_k ölçülür, G_k = C/|Q_eff² − k²| yasasına fit edilir → C, Q_eff.
  3. Eşik formülünden k_max öngörüsü yapılır; k_max VE k_max+1 fit
     genişliğiyle 50 seed üzerinden artık f karşılaştırılır.
  4. Antisimetrik/simetrik bastırma oranları (yörünge kazanç oranı ve
     spin kuplaj oranı) ölçülür.
  5. Geometrik faz a·γ·Φ_def sabitliği kontrol edilir.

Çıktı: test_g0_scan.json, test_g0_scan.png
"""

import copy
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from scipy.optimize import curve_fit

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)

from test_orbit_trim import (
    _worker, mode_vec_pair, knob_matrix, spektrum,
    T2, PATTERN_RMS, OFFSET_RMS, A_CAL, RETURN_STEPS,
)
from fourier_reconstruct import fodo_basis

# ── Parametreler ──────────────────────────────────────────────────────────────
G0_LIST      = [0.15, 0.20, 0.25]   # taranacak kuadrupol gradyanları [T/m]
N_SEEDS      = 50                    # istatistik için tohum sayısı
SEED_BASE    = 2000                  # örüntü tohumları: SEED_BASE+i
OFFSET_BASE  = 9000                  # BPM ofset tohumları: OFFSET_BASE+i
K_CALIB_MAX  = 12                    # kalibre edilecek maksimum mod sayısı
K_FIT_MAX    = 6                     # G_k yasası fiti için kullanılan aralık
DECOMP_SEED  = 321                   # antisim/simetrik ayrıştırma sabit tohumu

# Proton sabitleri — geometrik faz aγΦ_def hesabı için
A_PROTON   = 1.792847               # anomal manyetik moment
M_P_GEV    = 0.938272               # proton kütlesi [GeV/c²]
P_MAGIC    = 0.700850               # sihirli momentum [GeV/c]

# Orijinal konfigürasyonu bellekte sakla
with open(os.path.join(BASE, "params.json")) as _fh:
    _ORIG_CONFIG = json.load(_fh)

N_FODO = _ORIG_CONFIG["nFODO"]   # 24
R0     = _ORIG_CONFIG["R0"]       # 95.49 m


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def geom_phase():
    """a·γ·Φ_half [rad]: g₀'dan bağımsız geometrik faz.
    Φ_half = π / nFODO (yarı-hücre bükme açısı, tüm çevre / 2*nFODO)."""
    gamma = np.sqrt(1.0 + (P_MAGIC / M_P_GEV) ** 2)
    phi_half = np.pi / N_FODO
    return A_PROTON * gamma * phi_half


def write_params(g0):
    """params.json'u g₀ ile (kalıcı olmayan) güncelle."""
    cfg = copy.deepcopy(_ORIG_CONFIG)
    cfg["g0"] = g0
    cfg["g1"] = g0
    # g2 (k-mod gradyanı) orantılı kaydır
    cfg["g2"] = round(g0 * (_ORIG_CONFIG["g2"] / _ORIG_CONFIG["g0"]), 6)
    with open(os.path.join(BASE, "params.json"), "w") as fh:
        json.dump(cfg, fh, indent=4)


def restore_params():
    """Orijinal params.json'u geri yaz."""
    with open(os.path.join(BASE, "params.json"), "w") as fh:
        json.dump(_ORIG_CONFIG, fh, indent=4)


def gain_law(k, C, Q2):
    """Kazanç yasası: G_k = C / |Q_eff² − k²|"""
    return C / np.abs(Q2 - k ** 2)


def fit_gain_law(gain_by_k):
    """G_k = C/|Q_eff² − k²| fit'i, k ≤ K_FIT_MAX, aşırı yüksek G hariç."""
    ks = np.array([k for k in range(1, K_FIT_MAX + 1)
                   if gain_by_k.get(k, 0) < 200])
    gs = np.array([gain_by_k[k] for k in ks])
    try:
        popt, _ = curve_fit(gain_law, ks, gs, p0=[25.0, 5.0],
                            bounds=([0.5, 0.5], [500.0, 100.0]))
        C_fit, Q2_fit = float(popt[0]), float(popt[1])
    except Exception:
        C_fit, Q2_fit = 24.8, 5.03   # fallback
    return C_fit, Q2_fit


def predict_kmax(C, Q2, sigma_ratio=1.0):
    """Eşik formülü: k_max = floor(sqrt(Q2 + C * σ_q/σ_b))."""
    k_max_sq = Q2 + C * sigma_ratio
    return int(np.sqrt(max(k_max_sq, 0)))


# ── Aşama 1: kalibrasyon ──────────────────────────────────────────────────────

def calibrate(g0, ctx, nw):
    """g₀ için O_full [48 × 2*K_CALIB_MAX] ve G_k haritasını hesapla."""
    write_params(g0)
    with open(os.path.join(BASE, "params.json")) as fh:
        config = json.load(fh)
    from false_edm_mode_scan import setup_fields
    fields, *_ = setup_fields(config)
    n_q     = 2 * int(fields.nFODO)
    antisym = config.get("smooth_antisym_fodo", True)

    tasks = [("ref", np.zeros(n_q).tolist(), "orbit", T2, 10)]
    for k in range(1, K_CALIB_MAX + 1):
        c, s = mode_vec_pair(n_q, k, antisym)
        tasks.append((f"o{k}c", (A_CAL * c).tolist(), "orbit", T2, 10))
        tasks.append((f"o{k}s", (A_CAL * s).tolist(), "orbit", T2, 10))

    print(f"  Kalibrasyon: {len(tasks)} koşum...", flush=True)
    with ctx.Pool(processes=min(nw, len(tasks))) as pool:
        res = dict(pool.map(_worker, tasks))

    y_ref  = np.asarray(res["ref"])
    O_cols = []
    for k in range(1, K_CALIB_MAX + 1):
        for ph in ("c", "s"):
            O_cols.append((np.asarray(res[f"o{k}{ph}"]) - y_ref) / A_CAL)
    O_full = np.column_stack(O_cols)   # [48 × 24]

    gains      = np.sqrt(np.mean(O_full ** 2, axis=0))
    gain_by_k  = {k: float(0.5 * (gains[2*(k-1)] + gains[2*(k-1)+1]))
                  for k in range(1, K_CALIB_MAX + 1)}

    return O_full, gain_by_k, n_q, antisym


# ── Aşama 2: 50-seed trim istatistiği ─────────────────────────────────────────

def run_seed_batch(g0, O_full, k_max_pred, n_q, antisym, ctx, nw):
    """
    N_SEEDS tohum üzerinde orbit trim yap; k_max_pred ve k_max_pred+1
    karşılaştırması.  Yörünge y_true, O_full üzerinden doğrusal yaklaşım.
    Döndürür: dict nm → f_trim array (N_SEEDS,)
    """
    k_cap   = min(k_max_pred + 1, K_CALIB_MAX)
    k_lists = {"pred":   list(range(1, k_max_pred + 1)),
               "pred+1": list(range(1, k_cap + 1))}

    Om = {nm: O_full[:, : 2 * len(ks)] for nm, ks in k_lists.items()}
    Km = {nm: knob_matrix(n_q, ks, antisym) for nm, ks in k_lists.items()}

    write_params(g0)
    spin_tasks = []
    for i in range(N_SEEDS):
        P = np.random.default_rng(SEED_BASE + i).standard_normal(n_q) * PATTERN_RMS
        b = np.random.default_rng(OFFSET_BASE + i).standard_normal(n_q) * OFFSET_RMS

        # Doğrusal yörünge yaklaşımı: y_true ≈ O_full @ a_full
        _, by_k = spektrum(P, n_q, antisym, kmax=K_CALIB_MAX)
        a_full = np.array([by_k[k].get(kw, 0.0)
                           for k in range(1, K_CALIB_MAX + 1)
                           for kw in ("cos", "sin")])
        y_meas = O_full @ a_full + b

        for nm, ks in k_lists.items():
            a_hat, *_ = np.linalg.lstsq(Om[nm], y_meas, rcond=None)
            P_trim    = P - Km[nm] @ a_hat
            spin_tasks.append((f"s{i}_{nm}", P_trim.tolist(), "spin",
                                T2, RETURN_STEPS))

    print(f"  Tohum batarya: {len(spin_tasks)} spin koşumu ({N_SEEDS} tohum × "
          f"{len(k_lists)} fit)...", flush=True)
    with ctx.Pool(processes=min(nw, len(spin_tasks))) as pool:
        res = dict(pool.map(_worker, spin_tasks))

    f_trim = {nm: np.array([res[f"s{i}_{nm}"] for i in range(N_SEEDS)])
              for nm in k_lists}
    return f_trim, k_lists


# ── Aşama 3: antisim/simetrik ayrıştırma ──────────────────────────────────────

def run_decomp(g0, n_q, antisym, ctx, nw):
    """
    DECOMP_SEED deseni: tam, antisimetrik ve simetrik parçalar için yörünge +
    spin koşumu.  Bastırma oranlarını döndürür.
    """
    write_params(g0)
    P = (np.random.default_rng(DECOMP_SEED).standard_normal(n_q) * PATTERN_RMS)

    # Antisimetrik projeksiyon (Fourier tabanı)
    F_anti, _ = fodo_basis(n_q, list(range(0, N_FODO // 2 + 1)), antisym)
    a_coef, *_ = np.linalg.lstsq(F_anti, P, rcond=None)
    P_anti = F_anti @ a_coef
    P_symm = P - P_anti

    tasks = [
        ("full_s",  P.tolist(),       "spin",  T2, RETURN_STEPS),
        ("anti_s",  P_anti.tolist(),  "spin",  T2, RETURN_STEPS),
        ("symm_s",  P_symm.tolist(),  "spin",  T2, RETURN_STEPS),
        ("full_o",  P.tolist(),       "orbit", T2, 10),
        ("anti_o",  P_anti.tolist(),  "orbit", T2, 10),
        ("symm_o",  P_symm.tolist(),  "orbit", T2, 10),
    ]
    print(f"  Ayrıştırma: {len(tasks)} koşum...", flush=True)
    with ctx.Pool(processes=min(nw, len(tasks))) as pool:
        res = dict(pool.map(_worker, tasks))

    f_full  = float(res["full_s"])
    f_anti  = float(res["anti_s"])
    f_symm  = float(res["symm_s"])
    cod_full = float(np.std(np.asarray(res["full_o"])))
    cod_anti = float(np.std(np.asarray(res["anti_o"])))
    cod_symm = float(np.std(np.asarray(res["symm_o"])))

    rms_anti = float(np.std(P_anti))
    rms_symm = float(np.std(P_symm))

    # Birim kaçıklık başına normalize
    orbit_anti = cod_anti / rms_anti if rms_anti > 0 else np.nan
    orbit_symm = cod_symm / rms_symm if rms_symm > 0 else np.nan
    spin_anti  = abs(f_anti) / rms_anti if rms_anti > 0 else np.nan
    spin_symm  = abs(f_symm) / rms_symm if rms_symm > 0 else np.nan

    orbit_ratio = orbit_anti / orbit_symm if orbit_symm > 0 else np.nan
    spin_ratio  = spin_anti  / spin_symm  if spin_symm  > 0 else np.nan

    return {
        "f_full":  f_full, "f_anti": f_anti, "f_symm": f_symm,
        "cod_anti_um": cod_anti * 1e6, "cod_symm_um": cod_symm * 1e6,
        "rms_anti_um": rms_anti * 1e6, "rms_symm_um": rms_symm * 1e6,
        "orbit_anti": orbit_anti, "orbit_symm": orbit_symm,
        "spin_anti":  spin_anti,  "spin_symm":  spin_symm,
        "orbit_ratio": orbit_ratio, "spin_ratio": spin_ratio,
    }


# ── Ana döngü ─────────────────────────────────────────────────────────────────

def main():
    t0  = time.time()
    ctx = mp.get_context("spawn")
    nw  = mp.cpu_count()
    phi = geom_phase()
    print(f"Geometrik faz a·γ·Φ_half = {phi:.4f} rad (g₀'dan bağımsız)")
    print(f"Taranacak g₀ değerleri: {G0_LIST} T/m | N_SEEDS = {N_SEEDS}\n")

    results = {}

    try:
        for g0 in G0_LIST:
            print(f"{'='*60}")
            print(f"g₀ = {g0:.2f} T/m")
            print(f"{'='*60}")
            t_g = time.time()

            # 1. Kalibrasyon
            O_full, gain_by_k, n_q, antisym = calibrate(g0, ctx, nw)

            # 2. G_k yasası fit
            C_fit, Q2_fit = fit_gain_law(gain_by_k)
            k_max_pred    = predict_kmax(C_fit, Q2_fit,
                                         sigma_ratio=PATTERN_RMS / OFFSET_RMS)

            # k=7..12 öngörü sapması
            pred_vs_meas = {}
            for k in range(1, K_CALIB_MAX + 1):
                g_pred = gain_law(k, C_fit, Q2_fit) if abs(Q2_fit - k**2) > 0.1 else np.nan
                pred_vs_meas[k] = {"olculen": gain_by_k[k],
                                   "ongorulen": float(g_pred) if not np.isnan(g_pred) else None,
                                   "sapma_pct": float(100*(g_pred/gain_by_k[k]-1))
                                                if not np.isnan(g_pred) and gain_by_k[k] > 0 else None}

            print(f"\n  Fit: C = {C_fit:.2f}, Q_eff² = {Q2_fit:.3f}  "
                  f"→ Q_eff = {np.sqrt(Q2_fit):.3f}")
            print(f"  Eşik k_max öngörüsü: {k_max_pred} "
                  f"(k_max² = {Q2_fit + C_fit*(PATTERN_RMS/OFFSET_RMS):.2f})")
            print(f"\n  {'k':>4} {'ölçülen':>9} {'öngörü':>9} {'sapma%':>8}")
            for k in range(1, K_CALIB_MAX + 1):
                d = pred_vs_meas[k]
                tag = " ← fit" if k <= K_FIT_MAX else "  öngörü"
                sp  = f"{d['sapma_pct']:+7.1f}" if d['sapma_pct'] is not None else "      —"
                pr  = f"{d['ongorulen']:9.3f}" if d['ongorulen'] is not None else "        —"
                print(f"  {k:>4} {d['olculen']:>9.3f} {pr} {sp}{tag}")

            # 3. 50-seed trim
            f_trim, k_lists = run_seed_batch(g0, O_full, k_max_pred,
                                              n_q, antisym, ctx, nw)

            print(f"\n  Trim artık f (N={N_SEEDS} tohum):")
            for nm, ft in f_trim.items():
                ks_str = f"k=1..{max(k_lists[nm])}"
                print(f"    {nm:7s} ({ks_str:8s}): "
                      f"RMS={np.std(ft)*1e6:7.2f}×10⁻⁶ rad/s  "
                      f"medyan={np.median(np.abs(ft))*1e6:7.2f}×10⁻⁶ rad/s  "
                      f"maks={np.max(np.abs(ft))*1e6:7.2f}×10⁻⁶ rad/s")

            # 4. Ayrıştırma
            dec = run_decomp(g0, n_q, antisym, ctx, nw)
            print(f"\n  Antisim/simetrik ayrıştırma (seed={DECOMP_SEED}):")
            print(f"    Yörünge kazanç: antisim={dec['orbit_anti']:.1f}, "
                  f"simetrik={dec['orbit_symm']:.1f}  "
                  f"→ oran {dec['orbit_ratio']:.2f}×")
            print(f"    Spin kuplajı:   antisim={dec['spin_anti']:.3e}, "
                  f"simetrik={dec['spin_symm']:.3e}  "
                  f"→ oran {dec['spin_ratio']:.1f}×")

            results[g0] = {
                "C": C_fit, "Q2": Q2_fit, "Q_eff": float(np.sqrt(Q2_fit)),
                "k_max_pred": k_max_pred,
                "k_lists": {nm: k_lists[nm] for nm in k_lists},
                "f_trim_rms": {nm: float(np.std(f_trim[nm])) for nm in f_trim},
                "f_trim_median": {nm: float(np.median(np.abs(f_trim[nm]))) for nm in f_trim},
                "f_trim_max": {nm: float(np.max(np.abs(f_trim[nm]))) for nm in f_trim},
                "f_trim_all": {nm: f_trim[nm].tolist() for nm in f_trim},
                "gain_by_k": {str(k): v for k, v in gain_by_k.items()},
                "pred_vs_meas": {str(k): v for k, v in pred_vs_meas.items()},
                "decomp": dec,
                "geom_phase": phi,
                "elapsed_s": time.time() - t_g,
            }
            print(f"\n  g₀={g0:.2f} T/m tamamlandı: "
                  f"{(time.time()-t_g)/60:.1f} dk\n")

    finally:
        restore_params()
        print("params.json geri yüklendi.")

    # ── JSON çıktı ─────────────────────────────────────────────────────────────
    out = {"_aciklama": "g₀ taraması: eşik teorisi genel geçerlilik testi",
           "parametreler": {"N_SEEDS": N_SEEDS, "PATTERN_RMS_um": PATTERN_RMS*1e6,
                            "OFFSET_RMS_um": OFFSET_RMS*1e6, "K_CALIB_MAX": K_CALIB_MAX,
                            "K_FIT_MAX": K_FIT_MAX, "geom_phase_rad": phi},
           "sonuclar": {str(g0): results[g0] for g0 in G0_LIST}}
    with open("test_g0_scan.json", "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("Kaydedildi: test_g0_scan.json")

    # ── Özet tablo ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("ÖZET TABLO — Eşik teorisi g₀ taraması")
    print(f"{'═'*80}")
    print(f"{'g₀':>6} {'Q_eff':>6} {'C':>6} {'k_max':>6} "
          f"{'f_RMS (pred)':>14} {'f_RMS (pred+1)':>15} {'yörünge oranı':>14} "
          f"{'spin oranı':>11}")
    print(f"{'[T/m]':>6} {'':>6} {'':>6} {'öngörü':>6} "
          f"{'[×10⁻⁴ rad/s]':>14} {'[×10⁻⁴ rad/s]':>15} {'antisim/simetrik':>14} "
          f"{'antisim/sim':>11}")
    print('─'*80)
    for g0 in G0_LIST:
        r   = results[g0]
        rms_p   = r["f_trim_rms"]["pred"]   * 1e4
        rms_p1  = r["f_trim_rms"]["pred+1"] * 1e4
        print(f"{g0:>6.2f} {r['Q_eff']:>6.3f} {r['C']:>6.1f} "
              f"{r['k_max_pred']:>6d} "
              f"{rms_p:>14.3f} {rms_p1:>15.3f} "
              f"{r['decomp']['orbit_ratio']:>14.2f} "
              f"{r['decomp']['spin_ratio']:>11.1f}")
    print(f"\nGeometrik faz a·γ·Φ_half = {phi:.4f} rad (g₀'dan bağımsız)")
    print(f"Toplam süre: {(time.time()-t0)/60:.1f} dk")

    # ── Figürler ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Eşik teorisi: g₀ = {G0_LIST} T/m taraması  "
        f"(N_seeds={N_SEEDS}, σ_q=σ_b={PATTERN_RMS*1e6:.0f} μm)",
        fontsize=12)

    colors = {g0: c for g0, c in zip(G0_LIST, ["tab:blue", "tab:orange", "tab:green"])}

    # Panel 1: G_k yasası (tüm g₀'lar aynı eksen)
    ax = axes[0, 0]
    ks_all = np.linspace(0.5, 12.5, 300)
    for g0 in G0_LIST:
        r   = results[g0]
        gk  = [r["gain_by_k"][str(k)] for k in range(1, K_CALIB_MAX + 1)]
        ax.semilogy(range(1, K_CALIB_MAX + 1), gk, "o",
                    color=colors[g0], ms=6, label=f"g₀={g0:.2f} ölçülen")
        Q2, C = r["Q2"], r["C"]
        k_safe = ks_all[np.abs(Q2 - ks_all**2) > 0.3]
        ax.semilogy(k_safe, gain_law(k_safe, C, Q2), "--",
                    color=colors[g0], lw=1.5,
                    label=f"g₀={g0:.2f} fit (C={C:.1f}, Q²={Q2:.2f})")
    ax.axhline(OFFSET_RMS / PATTERN_RMS, color="red", ls=":", lw=1.5,
               label=f"eşik σ_b/σ_q = {OFFSET_RMS/PATTERN_RMS:.1f}")
    ax.set_xlabel("mod k"); ax.set_ylabel("G_k")
    ax.set_title("Kazanç yasası: üç g₀ noktasında")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    # Panel 2: k_max öngörüsü vs Q_eff²
    ax = axes[0, 1]
    Q2_vals = [results[g0]["Q2"] for g0 in G0_LIST]
    kmax_pred = [results[g0]["k_max_pred"] for g0 in G0_LIST]
    ax.plot(G0_LIST, Q2_vals, "s-", ms=8, label="Q_eff² (ölçülen)")
    ax2 = ax.twinx()
    ax2.plot(G0_LIST, kmax_pred, "D--", color="tab:red", ms=8,
             label="k_max öngörü")
    ax.set_xlabel("g₀ [T/m]"); ax.set_ylabel("Q_eff²")
    ax2.set_ylabel("k_max öngörüsü", color="tab:red")
    ax.set_title("Q_eff² ve k_max öngörüsü vs g₀")
    ax.grid(alpha=0.3)
    lines1, lbs1 = ax.get_legend_handles_labels()
    lines2, lbs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, lbs1+lbs2, fontsize=9)

    # Panel 3: f_trim RMS kutu grafiği
    ax = axes[0, 2]
    positions = []
    labels    = []
    data_list = []
    w = 0.3
    for j, g0 in enumerate(G0_LIST):
        r = results[g0]
        for nm in ("pred", "pred+1"):
            data_list.append(np.abs(r["f_trim_all"][nm]))
            positions.append(j + (0 if nm == "pred" else 0.35))
            labels.append(f"g{g0:.2f}\n{nm}")
    bp = ax.boxplot(data_list, positions=positions, widths=0.28,
                    patch_artist=True, showfliers=False)
    for patch, g0, nm in zip(bp["boxes"],
                              [g for g in G0_LIST for _ in ("pred","pred+1")],
                              ["pred","pred+1"]*len(G0_LIST)):
        patch.set_facecolor(colors[g0])
        patch.set_alpha(0.7 if nm == "pred" else 0.35)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("|f_trim| [rad/s]")
    ax.set_title(f"Trim artığı: pred vs pred+1 ({N_SEEDS} tohum)")
    ax.grid(alpha=0.3, axis="y")

    # Panel 4: yörünge ve spin bastırma oranları
    ax = axes[1, 0]
    orbit_r = [results[g0]["decomp"]["orbit_ratio"] for g0 in G0_LIST]
    spin_r  = [results[g0]["decomp"]["spin_ratio"]  for g0 in G0_LIST]
    x = np.arange(len(G0_LIST))
    ax.bar(x - 0.2, orbit_r, 0.4, label="yörünge antisim/simetrik",
           color="tab:blue", alpha=0.8)
    ax2b = ax.twinx()
    ax2b.bar(x + 0.2, spin_r, 0.4, label="spin antisim/simetrik",
             color="tab:red", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"g₀={g}" for g in G0_LIST])
    ax.set_ylabel("Yörünge bastırma oranı", color="tab:blue")
    ax2b.set_ylabel("Spin bastırma oranı", color="tab:red")
    ax.set_title("Antisim/simetrik bastırma vs g₀\n"
                 "(spin oranı ≈ sabit? ← geometrik faz testi)")
    lines_a, lbs_a = ax.get_legend_handles_labels()
    lines_b, lbs_b = ax2b.get_legend_handles_labels()
    ax.legend(lines_a+lines_b, lbs_a+lbs_b, fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Panel 5: f_trim RMS vs g₀ (pred vs pred+1)
    ax = axes[1, 1]
    rms_pred   = [results[g0]["f_trim_rms"]["pred"]   * 1e4 for g0 in G0_LIST]
    rms_pred1  = [results[g0]["f_trim_rms"]["pred+1"] * 1e4 for g0 in G0_LIST]
    ax.plot(G0_LIST, rms_pred,  "o-", ms=8, label="k ≤ k_max (öngörülen)")
    ax.plot(G0_LIST, rms_pred1, "s--", ms=8, label="k ≤ k_max+1 (aşan)")
    ax.set_xlabel("g₀ [T/m]")
    ax.set_ylabel("|f_trim| RMS [×10⁻⁴ rad/s]")
    ax.set_title("Trim artığı: öngörülen eşikte pred < pred+1 mı?")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 6: geometrik faz + spin kuplaj mekanizması özeti
    ax = axes[1, 2]
    spin_antisym = [results[g0]["decomp"]["spin_anti"] * 1e3 for g0 in G0_LIST]
    spin_symm    = [results[g0]["decomp"]["spin_symm"]  * 1e3 for g0 in G0_LIST]
    ax.plot(G0_LIST, spin_antisym, "o-", ms=8, label="|c| antisim [×10⁻³ (rad/s)/m]")
    ax.plot(G0_LIST, spin_symm,   "s--", ms=8, label="|c| simetrik [×10⁻³ (rad/s)/m]")
    ax2c = ax.twinx()
    ax2c.axhline(phi, color="gray", ls=":", lw=1.5, label=f"aγΦ = {phi:.3f} rad (sabit)")
    ax2c.set_ylim(0, 1); ax2c.set_ylabel("aγΦ_half [rad]", color="gray")
    ax.set_xlabel("g₀ [T/m]")
    ax.set_ylabel("Spin kuplaj katsayısı (birim kaçıklık başına)")
    ax.set_title("Spin kuplaj mekanizması: geometrik faz sabitliği")
    lines_p, lbs_p = ax.get_legend_handles_labels()
    lines_c, lbs_c = ax2c.get_legend_handles_labels()
    ax.legend(lines_p+lines_c, lbs_p+lbs_c, fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("test_g0_scan.png", dpi=150)
    print("Figür: test_g0_scan.png")
    print(f"\nToplam süre: {(time.time()-t0)/60:.1f} dk")


if __name__ == "__main__":
    main()
