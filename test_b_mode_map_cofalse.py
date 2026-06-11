#!/usr/bin/env python3
"""test_b_mode_map_cofalse.py — Mod haritası evrenselliği, GERÇEKÇİ koşulda.

test_b_mode_map.py'nin CO=False versiyonu: parçacık EKSENDEN fırlatılır,
kapalı yörünge hiç aranmaz/oturtulmaz (o yöntemin pratik olmadığına karar
verildi). Soru aynı: c_k vektörü (k=1..12) hizalama konfigürasyonundan
bağımsız mı?

Not — k aralığı: aliasing özdeşliği (cos(2πkn/24)=cos(2π(24−k)n/24)) baz
vektörlerinin kendisinin eşitliğidir; k ve 24−k modları kuadrupol
konumlarında AYNI desendir. Bu fırlatma koşulundan bağımsız kesin bir
kimliktir → k=13..24'ü yeniden ölçmek bilgi katmaz, tarama k=1..12.

Yöntem (üniform 0.2 lattice, t2=1ms, eksenden fırlatma):
  c_k^eff(P) = [f(P + A·F_k) − f(P)] / A,  A=10μm cos prob
  Üç arka plan: boş (test_b_ck_cofalse.json'dan), seed-7, seed-21 (RMS 10μm).

Çıktı:
  test_b_mode_map_cofalse.png
  test_b_mode_map_cofalse.json
  Terminal tablo
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fourier_reconstruct import fodo_basis

# ── Parametreler ─────────────────────────────────────────────────────────────
PROBE_A      = 1e-5
BG_RMS       = 1e-5
SEED_A       = 7
SEED_B       = 21
K_LIST       = list(range(1, 13))   # bağımsız modlar (aliasing: k=1..12)
T2           = 1e-3
RETURN_STEPS = 6000
CK_FILE      = "test_b_ck_cofalse.json"   # boş arka plan (CO=False) kalibrasyonu


def _suppress_stdout():
    fd = os.dup(1); null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1); os.close(null); return fd


def _restore_stdout(fd):
    os.dup2(fd, 1); os.close(fd)


def _worker(task):
    """EKSENDEN fırlatma (CO aranmaz) → dSy/dt."""
    label, dy_list, t2, return_steps = task

    import os, sys, json, numpy as np
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from false_edm_mode_scan import setup_fields
    from integrator import integrate_particle

    with open("params.json") as fh:
        config = json.load(fh)
    fields, y0, beta0, R0, p_mag, direction = setup_fields(config)
    dt = float(config.get("dt", 1e-11))
    dy = np.asarray(dy_list, dtype=float)

    saved = _suppress_stdout()
    try:
        fields.poincare_quad_index = 0.0
        _, poin, poin_t = integrate_particle(
            y0, 0.0, t2, dt,
            fields=fields, return_steps=return_steps, quad_dy=dy)
    finally:
        _restore_stdout(saved)

    slope = float(np.polyfit(np.asarray(poin_t, float),
                             np.asarray(poin[:, 7], float), 1)[0])
    return label, slope


def mode_vec(n_q, k, amp, antisym):
    Fk, _ = fodo_basis(n_q, [k], antisym)
    return Fk[:, 0] * amp


def main():
    t0 = time.time()
    with open("params.json") as fh:
        config = json.load(fh)
    with open(CK_FILE) as fh:
        ck_data = json.load(fh)
    c_bare = {int(k): float(v) for k, v in ck_data["c_k"].items()}

    from false_edm_mode_scan import setup_fields
    fields, *_ = setup_fields(config)
    n_q     = 2 * int(fields.nFODO)
    antisym = config.get("smooth_antisym_fodo", True)
    ctx     = mp.get_context("spawn")
    nw      = mp.cpu_count()

    rngA = np.random.default_rng(SEED_A)
    rngB = np.random.default_rng(SEED_B)
    P_A  = rngA.standard_normal(n_q) * BG_RMS
    P_B  = rngB.standard_normal(n_q) * BG_RMS

    print(f"Lattice üniform {fields.quadG1} T/m — EKSENDEN fırlatma, "
          f"COD oturtma YOK (t2={T2*1e3:.0f}ms)")
    print(f"Boş arka plan c_k: {CK_FILE} (k=1..{max(c_bare)})")

    tasks = [("bgA_base", P_A.tolist(), T2, RETURN_STEPS),
             ("bgB_base", P_B.tolist(), T2, RETURN_STEPS)]
    for k in K_LIST:
        tasks.append((f"bgA_{k}", (P_A + mode_vec(n_q, k, PROBE_A, antisym)).tolist(),
                      T2, RETURN_STEPS))
        tasks.append((f"bgB_{k}", (P_B + mode_vec(n_q, k, PROBE_A, antisym)).tolist(),
                      T2, RETURN_STEPS))

    print(f"\n{len(tasks)} simülasyon ({nw} işçi) — ilerleme:")
    res = {}
    with ctx.Pool(processes=nw) as pool:
        for i, (label, slope) in enumerate(
                pool.imap_unordered(_worker, tasks), 1):
            res[label] = slope
            print(f"  [{i:>3}/{len(tasks)}] {label:>10} = {slope:+.3e}", flush=True)

    fA = res["bgA_base"]; fB = res["bgB_base"]
    c_effA = {k: (res[f"bgA_{k}"] - fA) / PROBE_A for k in K_LIST}
    c_effB = {k: (res[f"bgB_{k}"] - fB) / PROBE_A for k in K_LIST}

    print(f"\n  Arka plan tabanları: f(P_A)={fA:+.3e}, f(P_B)={fB:+.3e} rad/s")
    print(f"\n{'─'*76}")
    print("c_k vektörleri [rad/s/m] (CO=False): boş / seed-A / seed-B")
    print(f"{'─'*76}")
    print(f"{'k':>3}  {'c_bare':>12}  {'c_eff(A)':>12}  {'c_eff(B)':>12}  "
          f"{'A/bare':>8}  {'B/bare':>8}")
    print('─'*76)
    for k in K_LIST:
        rA = c_effA[k]/c_bare[k] if c_bare[k] != 0 else float('nan')
        rB = c_effB[k]/c_bare[k] if c_bare[k] != 0 else float('nan')
        print(f"{k:>3}  {c_bare[k]:>12.3e}  {c_effA[k]:>12.3e}  "
              f"{c_effB[k]:>12.3e}  {rA:>8.3f}  {rB:>8.3f}")

    vb = np.array([c_bare[k] for k in K_LIST])
    vA = np.array([c_effA[k] for k in K_LIST])
    vB = np.array([c_effB[k] for k in K_LIST])
    corr_AB    = float(np.corrcoef(vA, vB)[0, 1])
    corr_bareA = float(np.corrcoef(vb, vA)[0, 1])
    corr_bareB = float(np.corrcoef(vb, vB)[0, 1])
    rms_dev_AB = float(np.sqrt(np.mean((vA - vB)**2)) / np.sqrt(np.mean(vb**2)))

    print(f"\n  Korelasyon  c_bare  ↔ c_eff(A) : {corr_bareA:.4f}")
    print(f"  Korelasyon  c_bare  ↔ c_eff(B) : {corr_bareB:.4f}")
    print(f"  Korelasyon  c_eff(A) ↔ c_eff(B) : {corr_AB:.4f}")
    print(f"  Bağıl RMS sapma |A−B|/|bare|   : {rms_dev_AB*100:.2f}%")

    elapsed = time.time() - t0
    print(f"\nToplam süre: {elapsed:.1f} s")
    print("\nYORUM:")
    print("  Vektörler çakışıyorsa → harita CO=False'ta da evrensel; trim")
    print("  reçetesi oturtma olmadan, her konfigürasyonda geçerli.")

    with open("test_b_mode_map_cofalse.json", "w") as fh:
        json.dump({
            "_aciklama": "CO=False mod haritası: c_k üç arka planda "
                         "(eksenden fırlatma, cos prob 10μm, t2=1ms)",
            "probe_A_m": PROBE_A, "bg_rms_m": BG_RMS,
            "seed_A": SEED_A, "seed_B": SEED_B,
            "f_base_A": fA, "f_base_B": fB,
            "c_bare":  {str(k): c_bare[k]  for k in K_LIST},
            "c_eff_A": {str(k): c_effA[k] for k in K_LIST},
            "c_eff_B": {str(k): c_effB[k] for k in K_LIST},
        }, fh, indent=2)
    print("Kaydedildi: test_b_mode_map_cofalse.json")

    # ── Figür ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5))
    fig.suptitle("Mod haritası evrenselliği — CO=False (eksenden fırlatma, "
                 "oturtma yok)", fontsize=13)
    kk = np.array(K_LIST)

    ax = axes[0]
    ax.plot(kk, vb, 'o-',  color='k',        ms=6, label='boş arka plan')
    ax.plot(kk, vA, 's--', color='tab:red',  ms=6, label=f'seed-{SEED_A}')
    ax.plot(kk, vB, 'd:',  color='tab:blue', ms=6, label=f'seed-{SEED_B}')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xlabel("Fourier modu k"); ax.set_ylabel("c_k [rad/s/m]")
    ax.set_title("c_k üç arka planda")
    ax.set_xticks(kk); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(kk, np.abs(vb)+1e-12, 'o-',  color='k',        ms=6, label='|c_bare|')
    ax.semilogy(kk, np.abs(vA)+1e-12, 's--', color='tab:red',  ms=6, label='|c_eff(A)|')
    ax.semilogy(kk, np.abs(vB)+1e-12, 'd:',  color='tab:blue', ms=6, label='|c_eff(B)|')
    ax.axvline(2.68, color='purple', ls='--', alpha=0.6, label='Q_y≈2.68')
    ax.set_xlabel("Fourier modu k"); ax.set_ylabel("|c_k| [rad/s/m]")
    ax.set_title("Büyüklük (log)")
    ax.set_xticks(kk); ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)

    ax = axes[2]
    lim = [min(vb.min(), vA.min(), vB.min()), max(vb.max(), vA.max(), vB.max())]
    ax.plot(lim, lim, 'k--', alpha=0.5, label='y=x')
    ax.scatter(vA, vB, c=kk, cmap='viridis', s=70, zorder=3)
    for k, x, y in zip(K_LIST, vA, vB):
        if k <= 4:
            ax.annotate(f"k={k}", (x, y), fontsize=8,
                        textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel(f"c_eff (seed-{SEED_A}) [rad/s/m]")
    ax.set_ylabel(f"c_eff (seed-{SEED_B}) [rad/s/m]")
    ax.set_title(f"Evrensellik: korelasyon = {corr_AB:.4f}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "test_b_mode_map_cofalse.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()
