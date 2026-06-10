#!/usr/bin/env python3
"""test_b_mode_map.py — Test B (harita): k=1..24 mod katsayılarının evrenselliği.

Soru (kullanıcı):
  k=1..24'ten k=1..24'e bir "telafi haritası" kurabilir miyiz — hangi mod
  hangi modla nasıl korelasyonlu? Ve bu harita hizalama hatası
  konfigürasyonundan (seed) bağımsız mı (evrensel mi)?

Fizik / tasarım notu:
  Sistem lineerse dSy/dt = Σ c_k·a_k → 24×24 telafi haritası
  M[k,k'] = −c_k/c_k' RANK-1'dir: tüm harita tek bir c_k vektöründen türetilir.
  Dolayısıyla haritanın bağımsız bilgi içeriği c_k vektörüdür ve evrensellik
  testi şudur: c_k'yi FARKLI ARKA PLANLARDA ölç, karşılaştır.

  Efektif katsayı (arka plan P üzerinde):
      c_k^eff(P) = [f(P + A·F_k) − f(P)] / A
  P=0 (boş), P=seed-A, P=seed-B için ölçülür.
   - Üç vektör çakışırsa → çapraz terim yok, harita evrensel,
     M[k,k'] = −c_k/c_k' her konfigürasyonda geçerli.
   - Çakışmazsa → mod etkileşimi var; sapma deseni etkileşimin yapısını verir.

  Bu tasarım 24×24=576 çift simülasyonu yerine ~64 simülasyonla aynı bilgiyi
  toplar.

Yöntem (üniform 0.2 lattice, CO=True):
  1. Boş arka plan: c_k, k=11..24 ölçülür (k=1..10 test_b_ck_table.json'dan).
  2. Seed-A (rastgele, RMS 10μm): f(P_A) + 24 prob → c_k^eff(A).
  3. Seed-B (farklı seed): aynısı → c_k^eff(B).
  4. Karşılaştırma: korelasyon, bağıl sapma, harita M[k,k'] farkı.

Çıktı:
  test_b_mode_map.png — 4 panel
  test_b_mode_map.json — üç c_k vektörü
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
PROBE_A      = 1e-5     # prob mod genliği [m] = 10 μm
BG_RMS       = 1e-5     # arka plan hizalama RMS [m] = 10 μm
SEED_A       = 7        # arka plan A (önceki testlerin referans seed'i)
SEED_B       = 21       # arka plan B (bağımsız konfigürasyon)
K_LIST       = list(range(1, 25))   # k=1..24 (k=24: antisim DC-eşleniği, sin'i yok)
T2           = 5e-4
CO_TURNS     = 24
CO_ITER      = 1
RETURN_STEPS = 3000
CK_FILE      = "test_b_ck_table.json"   # k=1..10 boş-arka-plan katsayıları


def _suppress_stdout():
    fd = os.dup(1); null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1); os.close(null); return fd


def _restore_stdout(fd):
    os.dup2(fd, 1); os.close(fd)


def _worker(task):
    """CO=True spin takibi → dSy/dt. Görev: (label, dy_list, t2, ct, ci, rs)."""
    label, dy_list, t2, co_turns, co_iter, return_steps = task

    import os, sys, json, numpy as np
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from false_edm_mode_scan import setup_fields, find_closed_orbit, _make_state, C
    from integrator import integrate_particle

    with open("params.json") as fh:
        config = json.load(fh)
    fields, _, beta0, R0, p_mag, direction = setup_fields(config)
    dt    = float(config.get("dt", 1e-11))
    circ  = (2*np.pi*R0 + 4*fields.nFODO*fields.driftLen
             + 2*fields.nFODO*fields.quadLen)
    T_rev = circ / (beta0 * C)
    dy    = np.asarray(dy_list, dtype=float)

    saved = _suppress_stdout()
    try:
        v_co, _ = find_closed_orbit(fields, p_mag, direction, dy, dt, T_rev,
                                     n_turns=co_turns, n_iter=co_iter)
        y_launch = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
        fields.poincare_quad_index = 0.0
        _, poin, poin_t = integrate_particle(
            y_launch, 0.0, t2, dt,
            fields=fields, return_steps=return_steps, quad_dy=dy)
    finally:
        _restore_stdout(saved)

    slope = float(np.polyfit(np.asarray(poin_t, float),
                             np.asarray(poin[:, 7], float), 1)[0])
    return label, slope


def mode_vec(n_q, k, amp, antisym):
    """Tek FODO Fourier modu (cos fazı) → quad dy vektörü [m]."""
    Fk, _ = fodo_basis(n_q, [k], antisym)
    return Fk[:, 0] * amp


def main():
    t0 = time.time()
    with open("params.json") as fh:
        config = json.load(fh)
    with open(CK_FILE) as fh:
        ck_data = json.load(fh)
    c_bare_known = {int(k): float(v) for k, v in ck_data["c_k"].items()}

    from false_edm_mode_scan import setup_fields
    fields, *_ = setup_fields(config)
    n_q     = 2 * int(fields.nFODO)
    antisym = config.get("smooth_antisym_fodo", True)
    ctx     = mp.get_context("spawn")
    nw      = mp.cpu_count()

    # Arka plan desenleri
    rngA = np.random.default_rng(SEED_A)
    rngB = np.random.default_rng(SEED_B)
    P_A  = rngA.standard_normal(n_q) * BG_RMS
    P_B  = rngB.standard_normal(n_q) * BG_RMS

    print(f"Lattice üniform {fields.quadG1} T/m. Prob A={PROBE_A*1e6:.0f}μm, "
          f"arka plan RMS={BG_RMS*1e6:.0f}μm (seed {SEED_A} ve {SEED_B})")

    # ── Görevler ──────────────────────────────────────────────────────────
    tasks = []
    # 1) Boş arka plan: eksik k'lar (JSON'da olmayanlar)
    k_missing = [k for k in K_LIST if k not in c_bare_known]
    for k in k_missing:
        tasks.append((f"bare_{k}", mode_vec(n_q, k, PROBE_A, antisym).tolist(),
                      T2, CO_TURNS, CO_ITER, RETURN_STEPS))
    # 2) Seed-A: taban + 24 prob
    tasks.append(("bgA_base", P_A.tolist(), T2, CO_TURNS, CO_ITER, RETURN_STEPS))
    for k in K_LIST:
        dy = P_A + mode_vec(n_q, k, PROBE_A, antisym)
        tasks.append((f"bgA_{k}", dy.tolist(), T2, CO_TURNS, CO_ITER, RETURN_STEPS))
    # 3) Seed-B: taban + 24 prob
    tasks.append(("bgB_base", P_B.tolist(), T2, CO_TURNS, CO_ITER, RETURN_STEPS))
    for k in K_LIST:
        dy = P_B + mode_vec(n_q, k, PROBE_A, antisym)
        tasks.append((f"bgB_{k}", dy.tolist(), T2, CO_TURNS, CO_ITER, RETURN_STEPS))

    print(f"\n{len(tasks)} simülasyon ({nw} işçi) — ilerleme:")
    res = {}
    with ctx.Pool(processes=nw) as pool:
        for i, (label, slope) in enumerate(
                pool.imap_unordered(_worker, tasks), 1):
            res[label] = slope
            print(f"  [{i:>3}/{len(tasks)}] {label:>10} = {slope:+.3e}", flush=True)

    # ── Katsayılar ────────────────────────────────────────────────────────
    c_bare = {}
    for k in K_LIST:
        if k in c_bare_known:
            c_bare[k] = c_bare_known[k]
        else:
            c_bare[k] = res[f"bare_{k}"] / PROBE_A

    fA = res["bgA_base"]; fB = res["bgB_base"]
    c_effA = {k: (res[f"bgA_{k}"] - fA) / PROBE_A for k in K_LIST}
    c_effB = {k: (res[f"bgB_{k}"] - fB) / PROBE_A for k in K_LIST}

    print(f"\n  Arka plan tabanları: f(P_A)={fA:+.3e}, f(P_B)={fB:+.3e} rad/s")
    print(f"\n{'─'*76}")
    print("c_k vektörleri [rad/s/m]: boş / seed-A üzerinde / seed-B üzerinde")
    print(f"{'─'*76}")
    print(f"{'k':>3}  {'c_bare':>12}  {'c_eff(A)':>12}  {'c_eff(B)':>12}  "
          f"{'A/bare':>8}  {'B/bare':>8}")
    print('─'*76)
    for k in K_LIST:
        rA = c_effA[k]/c_bare[k] if c_bare[k] != 0 else float('nan')
        rB = c_effB[k]/c_bare[k] if c_bare[k] != 0 else float('nan')
        print(f"{k:>3}  {c_bare[k]:>12.3e}  {c_effA[k]:>12.3e}  "
              f"{c_effB[k]:>12.3e}  {rA:>8.3f}  {rB:>8.3f}")

    # Evrensellik metrikleri
    vb = np.array([c_bare[k] for k in K_LIST])
    vA = np.array([c_effA[k] for k in K_LIST])
    vB = np.array([c_effB[k] for k in K_LIST])
    corr_AB    = float(np.corrcoef(vA, vB)[0, 1])
    corr_bareA = float(np.corrcoef(vb, vA)[0, 1])
    rms_dev_AB = float(np.sqrt(np.mean((vA - vB)**2)) / np.sqrt(np.mean(vb**2)))

    print(f"\n  Korelasyon  c_eff(A) ↔ c_eff(B) : {corr_AB:.4f}")
    print(f"  Korelasyon  c_bare  ↔ c_eff(A) : {corr_bareA:.4f}")
    print(f"  Bağıl RMS sapma |A−B|/|bare|   : {rms_dev_AB*100:.2f}%")

    elapsed = time.time() - t0
    print(f"\nToplam süre: {elapsed:.1f} s")
    print("\nYORUM:")
    print("  Üç vektör çakışıyorsa → lineerlik + evrensellik: 24×24 telafi haritası")
    print("  M[k,k'] = −c_k/c_k' tek vektörden kurulur ve seed'den bağımsızdır.")
    print("  Sapma varsa → mod etkileşimi (çapraz terim); sapan k'lar haritayı bozar.")

    # JSON kaydet
    out_json = {
        "_aciklama": "c_k vektörleri: boş arka plan, seed-A, seed-B üzerinde "
                     "(cos fazı, prob 10μm, CO=True, üniform 0.2 T/m)",
        "probe_A_m": PROBE_A, "bg_rms_m": BG_RMS,
        "seed_A": SEED_A, "seed_B": SEED_B,
        "f_base_A": fA, "f_base_B": fB,
        "c_bare":  {str(k): c_bare[k]  for k in K_LIST},
        "c_eff_A": {str(k): c_effA[k] for k in K_LIST},
        "c_eff_B": {str(k): c_effB[k] for k in K_LIST},
    }
    with open("test_b_mode_map.json", "w") as fh:
        json.dump(out_json, fh, indent=2)
    print("Kaydedildi: test_b_mode_map.json")

    # ── Figür ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    fig.suptitle("Test B (harita): c_k vektörünün arka plan bağımsızlığı "
                 "(k=1..24)", fontsize=13)
    kk = np.array(K_LIST)

    # Panel 1: üç c_k vektörü
    ax = axes[0, 0]
    ax.plot(kk, vb, 'o-',  color='k',          ms=5, label='boş arka plan')
    ax.plot(kk, vA, 's--', color='tab:red',    ms=5, label=f'seed-{SEED_A} üzerinde')
    ax.plot(kk, vB, 'd:',  color='tab:blue',   ms=5, label=f'seed-{SEED_B} üzerinde')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xlabel("Fourier modu k"); ax.set_ylabel("c_k [rad/s/m]")
    ax.set_title("c_k üç arka planda")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 2: |c_k| log ölçek
    ax = axes[0, 1]
    ax.semilogy(kk, np.abs(vb)+1e-12, 'o-',  color='k',        ms=5, label='|c_bare|')
    ax.semilogy(kk, np.abs(vA)+1e-12, 's--', color='tab:red',  ms=5, label='|c_eff(A)|')
    ax.semilogy(kk, np.abs(vB)+1e-12, 'd:',  color='tab:blue', ms=5, label='|c_eff(B)|')
    ax.axvline(2.68, color='purple', ls='--', alpha=0.6, label='Q_y≈2.68')
    ax.set_xlabel("Fourier modu k"); ax.set_ylabel("|c_k| [rad/s/m]")
    ax.set_title("Büyüklük (log) — rezonans yapısı")
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)

    # Panel 3: seed-A vs seed-B saçılımı
    ax = axes[1, 0]
    ax.plot([vb.min(), vb.max()], [vb.min(), vb.max()], 'k--', alpha=0.5, label='y=x')
    ax.scatter(vA, vB, c=kk, cmap='viridis', s=60, zorder=3)
    for k, x, y in zip(K_LIST, vA, vB):
        if k <= 4:
            ax.annotate(f"k={k}", (x, y), fontsize=8,
                        textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel(f"c_eff (seed-{SEED_A}) [rad/s/m]")
    ax.set_ylabel(f"c_eff (seed-{SEED_B}) [rad/s/m]")
    ax.set_title(f"Evrensellik: korelasyon = {corr_AB:.4f}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 4: 24×24 telafi haritası M[k,k'] = −c_k/c_k' (boş arka plan, log10|·|)
    ax = axes[1, 1]
    M = -np.outer(vb, 1.0/vb)
    im = ax.imshow(np.log10(np.abs(M)), origin='lower', cmap='RdBu_r',
                   extent=[0.5, 24.5, 0.5, 24.5], aspect='auto')
    ax.set_xlabel("telafi modu k'"); ax.set_ylabel("hedef mod k")
    ax.set_title("Telafi haritası log₁₀|A*_k'/A_k| = log₁₀|c_k/c_k'|\n"
                 "(rank-1: tek c_k vektöründen)")
    plt.colorbar(im, ax=ax, label="log₁₀|oran|")

    plt.tight_layout()
    out = "test_b_mode_map.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()
