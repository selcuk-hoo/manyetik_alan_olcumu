#!/usr/bin/env python3
"""test_b_partner_search.py — Test B: k=2'yi bastıran eş-modu sistematik arama.

Soru:
  k=1,2,3 modlarını yörüngeden tek tek silmek yerine, onları TELAFI eden zıt
  işaretli modları bulup hepsini birlikte bastırmak istiyoruz. İlk adım: k=2'yi
  en iyi bastıran ortağı bulmak.

Fizik:
  dSy/dt mod genliklerinde lineer: dSy/dt ≈ Σ_k c_k·A_k. Rezonans işaret
  kuralı (Q_y ≈ 2.68): k < Q_y → c_k > 0; k > Q_y → c_k < 0. Pozitif bir mod
  (k=2) zıt işaretli bir ortakla telafi edilebilir:
      c_2·A_2 + c_k'·A_k'* = 0  →  A_k'* = −(c_2/c_k')·A_2
  En iyi ortak: |c_k'| en büyük olan negatif mod → en KÜÇÜK telafi genliği
  gerektirir (büyük telafi genliği yeni hizalama bütçesi yer).

Test akışı:
  Bölüm 1: c_k tablosu k=1..10 (cos fazı, A=10μm, CO=True) — geniş tarama.
  Bölüm 2: tüm negatif-c adayları için A_k'* hesapla, k=2+aday birleşik
           simülasyonla doğrula → artığa göre sırala.
  Bölüm 3: en iyi ortak için A* etrafında ince tarama → gerçek sıfır geçişi
           (lineer modelin sapması) ve ulaşılabilir en derin bastırma.

Çıktı:
  test_b_partner_search.png — 3 panel
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

# ── Parametreler ─────────────────────────────────────────────────────────────
A_MODE       = 1e-5     # referans mod genliği [m] = 10 μm
T2           = 5e-4
CO_TURNS     = 24
CO_ITER      = 1
RETURN_STEPS = 3000
K_SCAN       = list(range(1, 11))   # c_k tablosu: k=1..10
K_TARGET     = 2                    # bastırılacak mod
FINE_FRACS   = [0.85, 1.0, 1.15]    # ince tarama: A* etrafında ±%15


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


def make_mode(n_q, k, a_cos, a_sin, antisym):
    """Tek FODO Fourier modu → quad dy vektörü [m]."""
    Fk, _ = fodo_basis(n_q, [k], antisym)
    return Fk[:, 0] * a_cos + Fk[:, 1] * a_sin


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

    print(f"Lattice üniform {fields.quadG1} T/m. Hedef mod: k={K_TARGET}")

    # ══ BÖLÜM 1: c_k tablosu (k=1..10, cos fazı) ═════════════════════════
    tasks1 = [(f"k{k}", make_mode(n_q, k, A_MODE, 0.0, antisym).tolist(),
               T2, CO_TURNS, CO_ITER, RETURN_STEPS) for k in K_SCAN]
    print(f"\nBÖLÜM 1: c_k tablosu — {len(tasks1)} simülasyon ({nw} işçi)...")
    with ctx.Pool(processes=min(nw, len(tasks1))) as pool:
        res1 = dict(pool.map(_worker, tasks1))

    c = {k: res1[f"k{k}"] / A_MODE for k in K_SCAN}   # [rad/s per m]

    print(f"\n{'─'*56}")
    print(f"c_k katsayıları (A={A_MODE*1e6:.0f}μm cos fazı, CO=True)")
    print(f"{'─'*56}")
    print(f"{'k':>3}  {'dSy/dt [rad/s]':>16}  {'c_k [rad/s/m]':>15}  {'işaret':>7}")
    print('─'*56)
    for k in K_SCAN:
        print(f"{k:>3}  {res1[f'k{k}']:>16.3e}  {c[k]:>15.3e}  "
              f"{'+' if c[k] > 0 else '−':>7}")

    # ══ BÖLÜM 2: aday ortaklar — A* hesapla, doğrula, sırala ═════════════
    s_target = res1[f"k{K_TARGET}"]
    candidates = [k for k in K_SCAN
                  if k != K_TARGET and np.sign(c[k]) == -np.sign(c[K_TARGET])
                  and abs(c[k]) > 1e-12]
    cand_info = []
    for k in candidates:
        A_star = -(c[K_TARGET] / c[k]) * A_MODE     # [m]
        cand_info.append((k, A_star))

    print(f"\nBÖLÜM 2: k={K_TARGET} için {len(candidates)} aday ortak "
          f"(zıt işaretli modlar) doğrulanıyor...")
    tasks2 = []
    for k, A_star in cand_info:
        dy = (make_mode(n_q, K_TARGET, A_MODE, 0.0, antisym)
              + make_mode(n_q, k, A_star, 0.0, antisym))
        tasks2.append((f"pair{k}", dy.tolist(), T2, CO_TURNS, CO_ITER, RETURN_STEPS))
    with ctx.Pool(processes=min(nw, len(tasks2))) as pool:
        res2 = dict(pool.map(_worker, tasks2))

    print(f"\n{'─'*74}")
    print(f"k={K_TARGET} (A={A_MODE*1e6:.0f}μm) + ortak k' (A_k'*) → birleşik artık")
    print(f"{'─'*74}")
    print(f"{'k_ortak':>8}  {'A* [μm]':>9}  {'artık [rad/s]':>15}  "
          f"{'bastırma':>9}  {'not':>20}")
    print('─'*74)
    rank = []
    for k, A_star in cand_info:
        resid = res2[f"pair{k}"]
        supp  = abs(s_target / resid) if resid != 0 else float('inf')
        note  = "küçük genlik" if abs(A_star) < 2*A_MODE else "BÜYÜK genlik"
        rank.append((k, A_star, resid, supp))
        print(f"{k:>8}  {A_star*1e6:>9.1f}  {resid:>15.3e}  {supp:>8.0f}×  {note:>20}")

    # En iyi ortak: en derin bastırma
    rank.sort(key=lambda r: -r[3])
    k_best, A_best, resid_best, supp_best = rank[0]
    print(f"\n  → EN İYİ ORTAK: k'={k_best}  (A*={A_best*1e6:+.1f}μm, "
          f"{supp_best:.0f}× bastırma)")

    # ══ BÖLÜM 3: en iyi ortak için ince tarama (gerçek sıfır) ════════════
    print(f"\nBÖLÜM 3: k'={k_best} için A* etrafında ince tarama "
          f"({len(FINE_FRACS)} nokta)...")
    tasks3 = []
    for fr in FINE_FRACS:
        dy = (make_mode(n_q, K_TARGET, A_MODE, 0.0, antisym)
              + make_mode(n_q, k_best, A_best*fr, 0.0, antisym))
        tasks3.append((f"fine{fr}", dy.tolist(), T2, CO_TURNS, CO_ITER, RETURN_STEPS))
    with ctx.Pool(processes=min(nw, len(tasks3))) as pool:
        res3 = dict(pool.map(_worker, tasks3))

    fine_A = np.array([A_best*fr for fr in FINE_FRACS])
    fine_s = np.array([res3[f"fine{fr}"] for fr in FINE_FRACS])
    # lineer fit ile gerçek sıfır geçişi
    p = np.polyfit(fine_A, fine_s, 1)
    A_zero = -p[1] / p[0]
    print(f"\n{'─'*58}")
    print(f"İnce tarama: k={K_TARGET}(10μm) + k'={k_best}(A)")
    print(f"{'─'*58}")
    for fr, A, s in zip(FINE_FRACS, fine_A, fine_s):
        print(f"  A = {A*1e6:>7.2f} μm ({fr:.2f}·A*)  →  {s:+.3e} rad/s")
    print(f"\n  Lineer model A*      = {A_best*1e6:+.2f} μm")
    print(f"  Gerçek sıfır geçişi  = {A_zero*1e6:+.2f} μm "
          f"(sapma {abs(A_zero-A_best)/abs(A_best)*100:.1f}%)")
    s_at_zero = np.polyval(p, A_zero)
    print(f"  Sıfırda kalan artık (fit) ≈ {s_at_zero:+.1e} rad/s")

    elapsed = time.time() - t0
    print(f"\nToplam süre: {elapsed:.1f} s")
    print("\nYORUM:")
    print(f"  k={K_TARGET} tek başına: {s_target:+.3e} rad/s")
    print(f"  En iyi ortak k'={k_best} ile birleşik artık: {resid_best:+.3e} rad/s")
    print("  Sonraki adım: k=1 ve k=3 için aynı arama; sonra üçü birden +")
    print("  ortakları tek desende birleştirip toplam bastırmayı ölçmek.")

    # ══ Figür ═════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5))
    fig.suptitle(f"Test B: k={K_TARGET} modunu bastıran ortak arama "
                 f"(üniform 0.2 lattice)", fontsize=13)

    # Panel 1: c_k tablosu
    ax = axes[0]
    cv = [c[k]*A_MODE for k in K_SCAN]
    colors = ['tab:red' if v > 0 else 'tab:blue' for v in cv]
    ax.bar(K_SCAN, cv, color=colors, alpha=0.85)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel("Fourier modu k")
    ax.set_ylabel(f"dSy/dt [rad/s]  (A={A_MODE*1e6:.0f}μm)")
    ax.set_title("c_k işaret tablosu k=1..10\n(kırmızı +, mavi −)")
    ax.set_xticks(K_SCAN)
    ax.grid(True, axis='y', alpha=0.3)

    # Panel 2: aday sıralaması
    ax = axes[1]
    ks    = [r[0] for r in rank]
    supps = [r[3] for r in rank]
    As    = [abs(r[1])*1e6 for r in rank]
    x = np.arange(len(ks))
    ax.bar(x, supps, color='tab:green', alpha=0.85)
    for xi, (s_v, a_v) in enumerate(zip(supps, As)):
        ax.text(xi, s_v*1.05, f"A*={a_v:.0f}μm", ha='center', fontsize=8)
    ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels([f"k'={k}" for k in ks])
    ax.set_ylabel("bastırma çarpanı [×]")
    ax.set_title(f"Aday ortaklar (k={K_TARGET} + k')\nlineer A* ile ulaşılan bastırma")
    ax.grid(True, axis='y', which='both', alpha=0.3)

    # Panel 3: ince tarama
    ax = axes[2]
    ax.plot(fine_A*1e6, fine_s, 'o-', color='tab:blue', ms=8)
    ax.axhline(0, color='gray', ls='--', alpha=0.7)
    ax.axvline(A_best*1e6, color='green', ls=':', label=f'lineer A*={A_best*1e6:.1f}μm')
    ax.axvline(A_zero*1e6, color='red',   ls=':', label=f'gerçek sıfır={A_zero*1e6:.1f}μm')
    ax.set_xlabel(f"k'={k_best} genliği A [μm]")
    ax.set_ylabel("dSy/dt [rad/s]")
    ax.set_title(f"İnce tarama: k={K_TARGET}(10μm)+k'={k_best}(A)\nsıfır geçişi")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "test_b_partner_search.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()
