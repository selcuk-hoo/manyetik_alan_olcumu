#!/usr/bin/env python3
"""test_harmonic_cancellation.py — Test 2: Modlar arası telafi ve evrensellik.

Soru:
  k=2'yi tek başına yörüngeden silmek false EDM'yi sıfırlamıyor çünkü bazı
  modlar zıt işaretli false EDM üretiyor. Acaba k=1,2,3'ün her birini telafi
  eden modları bulabilir miyiz? Bulursak bu HER rastgele dağılımda geçerli mi?

Fizik:
  dSy/dt, mod genliklerinde yaklaşık LİNEER (test_cross_correlation.py'de
  doğrulandı: dSy/dt ∝ A¹). Dolayısıyla her mod k bir c_k katsayısıyla
  katkı verir:
      dSy/dt ≈ Σ_k c_k · A_k
  Rezonans işaret kuralı (Q_y ≈ 2.68): k < Q_y modları (k=1,2) pozitif,
  k > Q_y modları (k=3,4,…) negatif. Bir pozitif modu, zıt işaretli bir
  modla telafi etmek mümkün:
      c_k·A_k + c_{k'}·A_{k'} = 0  →  A_{k'}* = -(c_k / c_{k'})·A_k

Test akışı:
  Bölüm 1: c_k katsayı tablosu (k=1..4, cos & sin fazları, A=10μm, CO=True).
           İşaret yapısı (+ / −) doğrulanır.
  Bölüm 2: Telafi doğrulaması — k=2(+)/k=3(−) için A_3 taraması ile dSy/dt=0
           geçişi; ayrıca k=1(+)/k=4(−) için tek-nokta doğrulaması.
  Bölüm 3: Evrensellik — M rastgele desen için GERÇEK dSy/dt vs lineer-model
           TAHMİNİ (Σ c_k·a_k). Yüksek korelasyon → c_k evrensel → telafi
           her dağılımda hesaplanabilir.

Çıktı:
  test_harmonic_cancellation.png — 3 panel
  Terminal tablolar
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

# ── Test parametreleri ───────────────────────────────────────────────────────
A_MODE       = 1e-5     # tek-mod genliği [m] = 10 μm
T2           = 5e-4     # simülasyon süresi [s]
CO_TURNS     = 24       # kapalı yörünge bulma tur sayısı
CO_ITER      = 1        # Newton yinelemesi
RETURN_STEPS = 3000
K_TABLE      = [1, 2, 3, 4]            # katsayı tablosu modları
A3_SCAN_UM   = [0, 10, 22, 30, 45]    # k=3 telafi taraması [μm]
M_RANDOM     = 6        # evrensellik için rastgele desen sayısı
RANDOM_RMS   = 1e-5     # rastgele hizalama RMS [m]
RANDOM_SEED  = 321


# ── Paralel worker ───────────────────────────────────────────────────────────

def _suppress_stdout():
    fd = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.close(null)
    return fd


def _restore_stdout(fd):
    os.dup2(fd, 1)
    os.close(fd)


def _worker(task):
    """CO=True spin takibi → dSy/dt ölçümü.

    Görev demeti: (label, dy_list, t2, co_turns, co_iter, return_steps)
    Dönüş: (label, slope [rad/s])
    """
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
            fields=fields, return_steps=return_steps, quad_dy=dy,
        )
    finally:
        _restore_stdout(saved)

    slope = float(np.polyfit(np.asarray(poin_t, float),
                             np.asarray(poin[:, 7], float), 1)[0])
    return label, slope


# ── Yardımcılar ─────────────────────────────────────────────────────────────

def make_mode(n_q, k, a_cos, a_sin, antisym):
    """Tek FODO Fourier modu → quad dy vektörü [m]."""
    Fk, _ = fodo_basis(n_q, [k], antisym)
    return Fk[:, 0] * a_cos + Fk[:, 1] * a_sin


def project_fodo(dy, n_q, k_list, antisym):
    """dy vektörünü FODO modlarına ayrıştır → {k: (a_cos, a_sin)} [m]."""
    cols = [fodo_basis(n_q, [k], antisym)[0] for k in k_list]
    B = np.column_stack(cols)
    coeffs, _, _, _ = np.linalg.lstsq(B, dy, rcond=None)
    out = {}
    for i, k in enumerate(k_list):
        out[k] = (float(coeffs[2*i]), float(coeffs[2*i + 1]))
    return out


# ── Ana rutin ────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    with open("params.json") as fh:
        config = json.load(fh)
    from false_edm_mode_scan import setup_fields
    fields, *_ = setup_fields(config)
    n_q     = 2 * int(fields.nFODO)
    antisym = config.get("smooth_antisym_fodo", True)
    ctx     = mp.get_context("spawn")
    nw      = mp.cpu_count()

    # ══ BÖLÜM 1: c_k katsayı tablosu (cos & sin) ═════════════════════════
    tasks1 = []
    for k in K_TABLE:
        tasks1.append((f"c{k}_cos",
                       make_mode(n_q, k, A_MODE, 0.0, antisym).tolist(),
                       T2, CO_TURNS, CO_ITER, RETURN_STEPS))
        tasks1.append((f"c{k}_sin",
                       make_mode(n_q, k, 0.0, A_MODE, antisym).tolist(),
                       T2, CO_TURNS, CO_ITER, RETURN_STEPS))

    print(f"BÖLÜM 1: c_k tablosu — {len(tasks1)} simülasyon ({nw} işçi)...")
    with ctx.Pool(processes=min(nw, len(tasks1))) as pool:
        res1 = dict(pool.map(_worker, tasks1))

    # c_k katsayıları: slope / A  [rad/s per metre]
    c_cos = {k: res1[f"c{k}_cos"] / A_MODE for k in K_TABLE}
    c_sin = {k: res1[f"c{k}_sin"] / A_MODE for k in K_TABLE}

    print(f"\n{'─'*60}")
    print("Mod başına false-EDM katsayısı (CO=True, A=10μm)")
    print(f"{'─'*60}")
    print(f"{'k':>3}  {'dSy/dt cos':>14}  {'işaret':>7}  {'dSy/dt sin':>14}")
    print('─'*60)
    for k in K_TABLE:
        sgn = '+' if res1[f"c{k}_cos"] > 0 else '−'
        print(f"{k:>3}  {res1[f'c{k}_cos']:>14.3e}  {sgn:>7}  "
              f"{res1[f'c{k}_sin']:>14.3e}")
    print(f"\nRezonans kuralı: Q_y≈2.68 → k<Q_y (k=1,2) pozitif, "
          f"k>Q_y (k=3,4) negatif")

    # ══ BÖLÜM 2: Telafi doğrulaması ══════════════════════════════════════
    # k=2(+) ↔ k=3(−): teorik telafi genliği
    A3_star = -(c_cos[2] / c_cos[3]) * A_MODE       # [m]
    # k=1(+) ↔ k=4(−)
    A4_star = -(c_cos[1] / c_cos[4]) * A_MODE       # [m]

    tasks2 = []
    # k=2 + k=3 taraması
    for a3_um in A3_SCAN_UM:
        dy = (make_mode(n_q, 2, A_MODE, 0.0, antisym)
              + make_mode(n_q, 3, a3_um*1e-6, 0.0, antisym))
        tasks2.append((f"scan23_{a3_um}", dy.tolist(),
                       T2, CO_TURNS, CO_ITER, RETURN_STEPS))
    # k=2 + k=3(A3*) tam telafi noktası
    dy = (make_mode(n_q, 2, A_MODE, 0.0, antisym)
          + make_mode(n_q, 3, A3_star, 0.0, antisym))
    tasks2.append(("comp23", dy.tolist(), T2, CO_TURNS, CO_ITER, RETURN_STEPS))
    # k=1 + k=4(A4*) tam telafi noktası
    dy = (make_mode(n_q, 1, A_MODE, 0.0, antisym)
          + make_mode(n_q, 4, A4_star, 0.0, antisym))
    tasks2.append(("comp14", dy.tolist(), T2, CO_TURNS, CO_ITER, RETURN_STEPS))

    print(f"\nBÖLÜM 2: Telafi doğrulaması — {len(tasks2)} simülasyon...")
    print(f"  Teorik telafi: A_3* = {A3_star*1e6:+.1f} μm (k=2 için),  "
          f"A_4* = {A4_star*1e6:+.1f} μm (k=1 için)")
    with ctx.Pool(processes=min(nw, len(tasks2))) as pool:
        res2 = dict(pool.map(_worker, tasks2))

    print(f"\n{'─'*60}")
    print("k=2 (A=10μm) + k=3 telafi taraması")
    print(f"{'─'*60}")
    print(f"{'A_3 [μm]':>10}  {'dSy/dt [rad/s]':>16}")
    print('─'*30)
    scan23 = [(a, res2[f"scan23_{a}"]) for a in A3_SCAN_UM]
    for a3_um, sl in scan23:
        print(f"{a3_um:>10d}  {sl:>16.3e}")
    base23 = res2["scan23_0"]      # k=2 tek başına
    print(f"\n  k=2 tek başına           : {base23:.3e} rad/s")
    print(f"  k=2 + k=3(A_3*={A3_star*1e6:.1f}μm) : {res2['comp23']:.3e} rad/s  "
          f"→ {abs(base23/res2['comp23']):.0f}× bastırma")
    base14 = res1["c1_cos"]
    print(f"  k=1 tek başına           : {base14:.3e} rad/s")
    print(f"  k=1 + k=4(A_4*={A4_star*1e6:.1f}μm) : {res2['comp14']:.3e} rad/s  "
          f"→ {abs(base14/res2['comp14']):.0f}× bastırma")

    # ══ BÖLÜM 3: Evrensellik — rastgele desen tahmini ════════════════════
    rng = np.random.default_rng(RANDOM_SEED)
    rand_patterns = [rng.standard_normal(n_q) * RANDOM_RMS
                     for _ in range(M_RANDOM)]
    tasks3 = [(f"rand_{m}", p.tolist(), T2, CO_TURNS, CO_ITER, RETURN_STEPS)
              for m, p in enumerate(rand_patterns)]

    print(f"\nBÖLÜM 3: Evrensellik — {len(tasks3)} rastgele desen...")
    with ctx.Pool(processes=min(nw, len(tasks3))) as pool:
        res3 = dict(pool.map(_worker, tasks3))

    # Lineer model tahmini: Σ_k c_k^cos·a_k + c_k^sin·b_k
    truth, pred = [], []
    for m, p in enumerate(rand_patterns):
        proj = project_fodo(p, n_q, K_TABLE, antisym)
        pr = sum(c_cos[k]*proj[k][0] + c_sin[k]*proj[k][1] for k in K_TABLE)
        truth.append(res3[f"rand_{m}"])
        pred.append(pr)
    truth = np.array(truth); pred = np.array(pred)
    corr = float(np.corrcoef(truth, pred)[0, 1])

    print(f"\n{'─'*60}")
    print("Rastgele desen: GERÇEK vs lineer-model TAHMİN (k=1..4)")
    print(f"{'─'*60}")
    print(f"{'desen':>6}  {'gerçek [rad/s]':>16}  {'tahmin [rad/s]':>16}")
    print('─'*44)
    for m in range(M_RANDOM):
        print(f"{m:>6d}  {truth[m]:>16.3e}  {pred[m]:>16.3e}")
    print(f"\n  Korelasyon (gerçek, tahmin) = {corr:.3f}")
    print(f"  → c_k evrensel ise telafi her dağılımda hesaplanabilir")

    elapsed = time.time() - t0
    print(f"\nToplam süre: {elapsed:.1f} s")

    # ══ Figür ═════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Test 2: Modlar Arası Telafi ve Evrensellik", fontsize=13)

    # Panel 1: c_k işaret tablosu
    ax = axes[0]
    cvals = [res1[f"c{k}_cos"] for k in K_TABLE]
    colors = ['tab:red' if v > 0 else 'tab:blue' for v in cvals]
    ax.bar(K_TABLE, cvals, color=colors, alpha=0.8)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel("Fourier modu k")
    ax.set_ylabel("dSy/dt [rad/s]  (A=10μm)")
    ax.set_title("Mod başına işaret yapısı\n(kırmızı +, mavi −)")
    ax.set_xticks(K_TABLE)
    ax.grid(True, axis='y', alpha=0.3)

    # Panel 2: k=2+k=3 telafi taraması
    ax = axes[1]
    a3s = np.array(A3_SCAN_UM, float)
    sls = np.array([res2[f"scan23_{a}"] for a in A3_SCAN_UM])
    ax.plot(a3s, sls, 'bo-', markersize=8)
    ax.axhline(0, color='gray', ls='--', alpha=0.7)
    ax.axvline(A3_star*1e6, color='green', ls=':', alpha=0.8,
               label=f"teorik A_3* = {A3_star*1e6:.1f} μm")
    ax.scatter([A3_star*1e6], [res2["comp23"]], color='red', s=120,
               zorder=5, marker='*', label=f"telafi: {res2['comp23']:.1e}")
    ax.set_xlabel("k=3 genliği A_3 [μm]")
    ax.set_ylabel("dSy/dt [rad/s]")
    ax.set_title("k=2(+) → k=3(−) ile telafi\n(A_2=10μm sabit)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: evrensellik — gerçek vs tahmin
    ax = axes[2]
    ax.scatter(pred, truth, color='purple', s=60, zorder=3)
    lim = max(np.max(np.abs(truth)), np.max(np.abs(pred))) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.6, label='y = x')
    ax.set_xlabel("Lineer model tahmini [rad/s]")
    ax.set_ylabel("Gerçek simülasyon [rad/s]")
    ax.set_title(f"Evrensellik (M={M_RANDOM} rastgele)\nkorelasyon = {corr:.3f}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')

    plt.tight_layout()
    out = "test_harmonic_cancellation.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()
