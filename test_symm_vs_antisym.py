#!/usr/bin/env python3
"""test_symm_vs_antisym.py — Antisimetrik vs simetrik hizalama hata bileşenlerinin
yörünge (COD) ve sahte EDM sinyaline etkisi.

TEORİK ARKA PLAN:
  48 kuadrupol dizisi iki alt uzaya ayrılabilir:
    1. Antisimetrik alt uzay (25 boyut): QF ve QD'nin ZIT yönlerde hareket ettiği
       Fourier modları (fodo_basis, antisym=True, k=0..12). Bu modlar kapalı
       yörünge bozulması (COD) üretir ve sahte EDM sinyaline yol açar.
    2. Simetrik tümleme (23 boyut): QF ve QD'nin AYNI yönde hareket ettiği
       bileşenler. COD üretmemeli ama "taban" sahte EDM üretebilir.

TEST: 4 koşum (CO=False, t2=1ms, seed=321, RMS=100μm):
  1. full    : orijinal rastgele desen
  2. antisym : antisimetrik izdüşümü
  3. symm    : simetrik tümleme (full - antisym)
  4. zero    : sıfır referans

Çıktı: test_symm_vs_antisym.json, test_symm_vs_antisym.png
"""

import json
import os
import shutil
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)

from fourier_reconstruct import fodo_basis

# ── Parametreler ───────────────────────────────────────────────────────────────
PATTERN_RMS  = 1e-4      # rastgele hizalama hatası RMS [m] = 100 μm
PATTERN_SEED = 321
T2           = 1e-3      # simülasyon süresi [s]
RETURN_STEPS = 6000      # spin Poincaré noktası sayısı


# ── Gizli stdout bastırıcılar ─────────────────────────────────────────────────
def _suppress_stdout():
    fd = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.close(null)
    return fd


def _restore_stdout(fd):
    os.dup2(fd, 1)
    os.close(fd)


# ── İşçi fonksiyonlar ─────────────────────────────────────────────────────────

def _orbit_worker(task):
    """Yörünge koşumu: 48 BPM tur-ortalamalı y değerlerini döndürür.

    task = (label, dy_list)
    """
    label, dy_list = task

    import os, sys, json, tempfile, shutil
    import numpy as np
    sys.path.insert(0, BASE)
    tmp = tempfile.mkdtemp(prefix=f"symm_{os.getpid()}_")
    os.chdir(tmp)

    from false_edm_mode_scan import setup_fields
    from integrator import integrate_particle

    with open(os.path.join(BASE, "params.json")) as fh:
        config = json.load(fh)
    fields, y0, beta0, R0, p_mag, direction = setup_fields(config)
    dt = float(config.get("dt", 1e-11))
    dy = np.asarray(dy_list, dtype=float)

    saved = _suppress_stdout()
    try:
        fields.poincare_quad_index = 999.0
        integrate_particle(y0, 0.0, T2, dt, fields=fields,
                           return_steps=10, quad_dy=dy)
        cd = np.loadtxt("cod_data.txt", skiprows=1)
        cd[:, 1:3] *= 1e-3            # mm → m
        n = int(fields.nFODO)
        y_bpm = np.empty(2 * n)
        for k in range(n):
            y_bpm[2*k]     = cd[k*8 + 2, 2]   # QF girişi
            y_bpm[2*k + 1] = cd[k*8 + 6, 2]   # QD girişi
        result = y_bpm.tolist()
    finally:
        _restore_stdout(saved)
        os.chdir(BASE)
        shutil.rmtree(tmp, ignore_errors=True)

    return label, result


def _spin_worker(task):
    """Spin koşumu: dSy/dt eğimini döndürür (boylamsal başlangıç, EDMSwitch=0).

    task = (label, dy_list)
    """
    label, dy_list = task

    import os, sys, json, tempfile, shutil
    import numpy as np
    sys.path.insert(0, BASE)
    tmp = tempfile.mkdtemp(prefix=f"sspin_{os.getpid()}_")
    os.chdir(tmp)

    from false_edm_mode_scan import setup_fields
    from integrator import integrate_particle

    with open(os.path.join(BASE, "params.json")) as fh:
        config = json.load(fh)
    fields, y0, beta0, R0, p_mag, direction = setup_fields(config)
    fields.poincare_quad_index = 0.0
    fields.EDMSwitch = 0.0
    dt = float(config.get("dt", 1e-11))
    dy = np.asarray(dy_list, dtype=float)

    fd = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1); os.close(null)
    try:
        _, poin, poin_t = integrate_particle(
            y0, 0.0, T2, dt, fields=fields,
            return_steps=RETURN_STEPS, quad_dy=dy)
        t_arr = np.asarray(poin_t, float)
        sy = np.asarray(poin[:, 7], float)
        slope = float(np.polyfit(t_arr, sy, 1)[0])
    finally:
        os.dup2(fd, 1); os.close(fd)
        os.chdir(BASE)
        shutil.rmtree(tmp, ignore_errors=True)

    return label, slope


# ── Antisimetrik iz düşüm matrisi ─────────────────────────────────────────────

def build_antisym_projector(n_q, kmax=12):
    """Antisimetrik alt uzay için iz düşüm matrisi P (n_q × n_q).

    Antisimetrik baz: fodo_basis(n_q, k=0..kmax, antisym=True)
    B sütunları bu baza karşılık gelir.
    P = B @ pinv(B) = B @ (B^T B)^{-1} B^T
    """
    k_list = list(range(0, kmax + 1))   # k=0..12 → 1 + 2×12 = 25 sütun
    B, meta = fodo_basis(n_q, k_list, antisym=True)
    # Ortanormal bir baz oluştur (sayısal kararlılık için QR)
    Q, R_mat = np.linalg.qr(B, mode='reduced')
    P = Q @ Q.T
    return P, B, meta


def main():
    t0 = time.time()
    with open("params.json") as fh:
        config = json.load(fh)

    from false_edm_mode_scan import setup_fields
    fields, *_ = setup_fields(config)
    n_q     = 2 * int(fields.nFODO)   # = 48

    # ── Antisimetrik iz düşüm matrisi ─────────────────────────────────────────
    P_antisym, B, meta = build_antisym_projector(n_q, kmax=12)
    print(f"Baz boyutu: n_q={n_q}, antisym sütun sayısı={B.shape[1]}")
    print(f"P_antisym rank ≈ {int(np.round(np.trace(P_antisym)))}")

    # ── Rastgele hizalama deseni ───────────────────────────────────────────────
    rng = np.random.default_rng(PATTERN_SEED)
    dy_full   = rng.standard_normal(n_q) * PATTERN_RMS
    dy_antisym = P_antisym @ dy_full
    dy_symm    = dy_full - dy_antisym
    dy_zero    = np.zeros(n_q)

    # Kontrol
    overlap = float(np.dot(dy_antisym, dy_symm))
    print(f"\nDesen özeti:")
    print(f"  full   : RMS = {np.std(dy_full)*1e6:7.2f} μm")
    print(f"  antisym: RMS = {np.std(dy_antisym)*1e6:7.2f} μm")
    print(f"  symm   : RMS = {np.std(dy_symm)*1e6:7.2f} μm")
    print(f"  İç çarpım (antisym·symm) = {overlap:.2e}  (≈0 olmalı)")
    print(f"  Enerji koruma: {np.dot(dy_full,dy_full):.4e} ≈ "
          f"{np.dot(dy_antisym,dy_antisym)+np.dot(dy_symm,dy_symm):.4e}")

    patterns = {
        "full":    dy_full,
        "antisym": dy_antisym,
        "symm":    dy_symm,
        "zero":    dy_zero,
    }

    # ── Paralel simülasyon ─────────────────────────────────────────────────────
    ctx = mp.get_context("spawn")
    nw  = mp.cpu_count()

    orbit_tasks = [(lbl, v.tolist()) for lbl, v in patterns.items()]
    spin_tasks  = [(lbl, v.tolist()) for lbl, v in patterns.items()]

    print(f"\n{len(orbit_tasks)} yörünge + {len(spin_tasks)} spin simülasyonu "
          f"başlatılıyor ({nw} işçi)...")

    with ctx.Pool(processes=min(nw, len(orbit_tasks))) as pool:
        orbit_res = dict(pool.map(_orbit_worker, orbit_tasks))

    with ctx.Pool(processes=min(nw, len(spin_tasks))) as pool:
        spin_res = dict(pool.map(_spin_worker, spin_tasks))

    # ── Metrikleri hesapla ─────────────────────────────────────────────────────
    results = {}
    for lbl in patterns:
        bpm = np.asarray(orbit_res[lbl])
        cod_rms = float(np.sqrt(np.mean(bpm**2)) * 1e6)   # μm cinsinden RMS
        dSy_dt  = float(spin_res[lbl])
        results[lbl] = {"COD_rms_um": cod_rms, "dSy_dt": dSy_dt}

    # ── Tablo ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*56}")
    print(f"{'desen':>10} | {'COD_rms [μm]':>14} | {'dSy/dt [rad/s]':>16}")
    print(f"{'─'*56}")
    for lbl in ["full", "antisym", "symm", "zero"]:
        r = results[lbl]
        print(f"{lbl:>10} | {r['COD_rms_um']:>14.2f} | {r['dSy_dt']:>+16.4e}")
    print(f"{'─'*56}")

    # Yorumlar
    f_full    = abs(results["full"]["dSy_dt"])
    f_antisym = abs(results["antisym"]["dSy_dt"])
    f_symm    = abs(results["symm"]["dSy_dt"])
    if f_full > 0:
        print(f"\nAntisim/full oranı (sahte EDM): {f_antisym/f_full:.3f}")
        if f_symm > 0:
            print(f"Simetrik taban (full'a kıyasla): {f_symm/f_full:.3e}")

    # ── JSON kaydet ────────────────────────────────────────────────────────────
    out = {
        "_aciklama": ("Antisimetrik vs simetrik hizalama hata bileşenlerinin "
                      "COD ve sahte EDM sinyaline etkisi (CO=False, t2=1ms, "
                      f"seed={PATTERN_SEED}, RMS={PATTERN_RMS*1e6:.0f}μm)"),
        "parametreler": {
            "PATTERN_RMS_um": PATTERN_RMS * 1e6,
            "PATTERN_SEED": PATTERN_SEED,
            "T2_s": T2,
            "n_q": n_q,
            "antisym_basis_cols": int(B.shape[1]),
        },
        "desen_rms_um": {
            lbl: float(np.std(v) * 1e6) for lbl, v in patterns.items()
        },
        "ic_carpim_antisym_symm": overlap,
        "sonuclar": results,
    }
    with open("test_symm_vs_antisym.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nKaydedildi: test_symm_vs_antisym.json")

    # ── Figür ─────────────────────────────────────────────────────────────────
    labels_tr = {"full": "full\n(orijinal)",
                 "antisym": "antisym\n(izdüşüm)",
                 "symm": "symm\n(tümleme)",
                 "zero": "zero\n(referans)"}
    lbls = ["full", "antisym", "symm", "zero"]
    colors = ["tab:gray", "tab:blue", "tab:orange", "tab:green"]
    x = np.arange(len(lbls))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Antisimetrik vs Simetrik Hizalama Bileşenleri\n"
        f"(seed={PATTERN_SEED}, RMS={PATTERN_RMS*1e6:.0f}μm, CO=False, t2=1ms)",
        fontsize=12)

    # Sol: COD_rms
    ax = axes[0]
    vals_cod = [results[l]["COD_rms_um"] for l in lbls]
    bars = ax.bar(x, vals_cod, color=colors, alpha=0.85, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([labels_tr[l] for l in lbls])
    ax.set_ylabel("COD RMS [μm]")
    ax.set_title("Kapalı Yörünge Bozulması (BPM RMS)")
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, vals_cod):
        ax.annotate(f"{val:.1f}", (bar.get_x() + bar.get_width()/2, val),
                    ha='center', va='bottom', fontsize=9)

    # Sağ: |dSy/dt|
    ax = axes[1]
    vals_edm = [abs(results[l]["dSy_dt"]) for l in lbls]
    bars = ax.bar(x, vals_edm, color=colors, alpha=0.85, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([labels_tr[l] for l in lbls])
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("Sahte EDM Sinyali")
    ax.set_yscale('log')
    ax.grid(True, which='both', axis='y', alpha=0.3)
    for bar, val in zip(bars, vals_edm):
        if val > 0:
            ax.annotate(f"{val:.2e}",
                        (bar.get_x() + bar.get_width()/2,
                         val * 1.15),
                        ha='center', va='bottom', fontsize=8, rotation=15)

    plt.tight_layout()
    plt.savefig("test_symm_vs_antisym.png", dpi=150)
    print("Figür kaydedildi: test_symm_vs_antisym.png")
    print(f"Toplam süre: {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
