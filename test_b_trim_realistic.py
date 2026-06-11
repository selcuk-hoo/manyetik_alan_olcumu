#!/usr/bin/env python3
"""test_b_trim_realistic.py — Trim döngüsü GERÇEKÇİ koşulda: COD'ye oturtma YOK.

Gerekçe:
  Kick ile kapalı yörüngeye oturtmanın pratik olmadığına karar verildi
  (injection_kick_raporu.md): tek kick yalnız sentroidi oturtur, mikron-altı
  tolerans ister. Bu testte parçacık EKSENDEN fırlatılır (CO=False,
  use_closed_orbit anahtarındaki gibi), kapalı yörünge hiç aranmaz.
  Betatron salınımı ölçümün doğal parçasıdır.

Sorular:
  1. CO=False'ta mod katsayıları c_k hâlâ iyi tanımlı ve işaret-yapılı mı?
     (Betatron kirlenmesi de COD bozulmasınca sürülür → a_k ile lineer
     büyümesi beklenir; o zaman c_k tanımı geçerli kalır.)
  2. Doğrusallık: f(A) ∝ A sağlanıyor mu? (k=2'de A=5,10,20 μm)
  3. İteratif ölç-trimle döngüsü CO=False'ta da çalışır mı? Kaç tur,
     hangi tabana iner?

Yöntem (üniform 0.2 lattice, t2=1ms, eksenden fırlatma):
  Bölüm 1: c_k tablosu k=1..12 (cos, A=10μm) — CO ARAMADAN.
  Bölüm 2: k=2 doğrusallık (A=5, 20 μm ek noktalar).
  Bölüm 3: taze desen (seed 99) üzerinde 3 adımlı trim döngüsü,
           kol A: k=3 tek mod; kol B: k=3+k=4 bölüşmüş.

Çıktı:
  test_b_trim_realistic.png
  test_b_ck_cofalse.json (CO=False kalibrasyonu)
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
LIN_AMPS     = [5e-6, 2e-5]          # doğrusallık kontrol genlikleri (k=2)
PATTERN_SEED = 99
BG_RMS       = 1e-5
N_ITER       = 3
T2           = 1e-3     # CO=False için 1 ms (betatron üzerinden eğim fiti)
RETURN_STEPS = 6000
K_LIST       = list(range(1, 13))    # bağımsız modlar k=1..12 (aliasing)


def _suppress_stdout():
    fd = os.dup(1); null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1); os.close(null); return fd


def _restore_stdout(fd):
    os.dup2(fd, 1); os.close(fd)


def _worker(task):
    """EKSENDEN fırlatma (CO aranmaz) → dSy/dt.

    Görev: (label, dy_list, t2, return_steps)
    """
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
    """Tek FODO Fourier modu (cos fazı) → quad dy vektörü [m]."""
    Fk, _ = fodo_basis(n_q, [k], antisym)
    return Fk[:, 0] * amp


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

    rng = np.random.default_rng(PATTERN_SEED)
    P = rng.standard_normal(n_q) * BG_RMS

    def run(tasks):
        with ctx.Pool(processes=min(nw, len(tasks))) as pool:
            return dict(pool.map(_worker, tasks))

    print(f"Lattice üniform {fields.quadG1} T/m — EKSENDEN fırlatma, "
          f"COD oturtma YOK (t2={T2*1e3:.0f}ms)")

    # ══ BÖLÜM 1+2: c_k tablosu + doğrusallık + taban (tek havuz) ═════════
    tasks = [(f"k{k}", mode_vec(n_q, k, A_MODE, antisym).tolist(),
              T2, RETURN_STEPS) for k in K_LIST]
    for A in LIN_AMPS:
        tasks.append((f"lin_{A}", mode_vec(n_q, 2, A, antisym).tolist(),
                      T2, RETURN_STEPS))
    tasks.append(("f0", P.tolist(), T2, RETURN_STEPS))
    tasks.append(("bos", np.zeros(n_q).tolist(), T2, RETURN_STEPS))

    print(f"\nBölüm 1+2: {len(tasks)} simülasyon ({nw} işçi)...")
    res = run(tasks)

    f_bos = res["bos"]
    c = {k: res[f"k{k}"] / A_MODE for k in K_LIST}

    print(f"\n{'─'*60}")
    print("c_k tablosu — CO=False (eksenden fırlatma)")
    print(f"{'─'*60}")
    print(f"{'k':>3}  {'dSy/dt [rad/s]':>16}  {'c_k [rad/s/m]':>15}  {'işaret':>7}")
    print('─'*60)
    for k in K_LIST:
        print(f"{k:>3}  {res[f'k{k}']:>16.3e}  {c[k]:>15.3e}  "
              f"{'+' if c[k] > 0 else '−':>7}")
    print(f"\n  Boş kafes tabanı (misalignment yok): {f_bos:+.3e} rad/s")

    # Doğrusallık (k=2)
    print(f"\n{'─'*60}")
    print("Doğrusallık kontrolü (k=2): f(A)/A sabit mi?")
    print(f"{'─'*60}")
    amps_all = LIN_AMPS[:1] + [A_MODE] + LIN_AMPS[1:]
    for A in amps_all:
        fv = res[f"lin_{A}"] if A != A_MODE else res["k2"]
        print(f"  A = {A*1e6:>5.1f} μm  →  f = {fv:+.3e}  →  f/A = {fv/A:+.3e}")
    lin_ratios = [(res[f"lin_{A}"] if A != A_MODE else res["k2"]) / A
                  for A in amps_all]
    lin_spread = (max(lin_ratios) - min(lin_ratios)) / abs(np.mean(lin_ratios))
    print(f"  → f/A yayılımı: %{lin_spread*100:.1f}")

    # ══ BÖLÜM 3: iteratif trim döngüsü (CO=False) ════════════════════════
    f0 = res["f0"]
    print(f"\n{'─'*60}")
    print(f"Bölüm 3: trim döngüsü — taze desen seed={PATTERN_SEED}, "
          f"f0 = {f0:+.3e} rad/s")
    print(f"{'─'*60}")

    trimA = np.zeros(n_q); trimB = np.zeros(n_q)
    fA_hist = [f0]; fB_hist = [f0]
    ampA_hist, ampB_hist = [], []

    for it in range(1, N_ITER + 1):
        aA = -fA_hist[-1] / c[3]
        xB = -fB_hist[-1] / (c[3] + c[4])
        trimA_new = trimA + mode_vec(n_q, 3, aA, antisym)
        trimB_new = (trimB + mode_vec(n_q, 3, xB, antisym)
                            + mode_vec(n_q, 4, xB, antisym))
        print(f"\nAdım {it}: kol A trim k=3 ({aA*1e6:+.2f}μm), "
              f"kol B k=3+4 ({xB*1e6:+.2f}μm × 2)...")
        r = run([(f"A{it}", (P + trimA_new).tolist(), T2, RETURN_STEPS),
                 (f"B{it}", (P + trimB_new).tolist(), T2, RETURN_STEPS)])
        trimA, trimB = trimA_new, trimB_new
        fA_hist.append(r[f"A{it}"]); fB_hist.append(r[f"B{it}"])
        ampA_hist.append(aA);         ampB_hist.append(xB)
        print(f"  kol A: f{it} = {fA_hist[-1]:+.3e}  "
              f"(bastırma {abs(f0/fA_hist[-1]):.0f}×)")
        print(f"  kol B: f{it} = {fB_hist[-1]:+.3e}  "
              f"(bastırma {abs(f0/fB_hist[-1]):.0f}×)")

    print(f"\n{'─'*70}")
    print("Döngü özeti (CO=False — kapalı yörünge hiç aranmadı)")
    print(f"{'─'*70}")
    print(f"{'adım':>5}  {'kol A (k=3)':>14}  {'bastırma':>9}  "
          f"{'kol B (k=3+4)':>14}  {'bastırma':>9}")
    print('─'*70)
    for i in range(N_ITER + 1):
        sA = abs(f0/fA_hist[i]) if fA_hist[i] != 0 else float('inf')
        sB = abs(f0/fB_hist[i]) if fB_hist[i] != 0 else float('inf')
        print(f"{i:>5}  {fA_hist[i]:>14.3e}  {sA:>8.0f}×  "
              f"{fB_hist[i]:>14.3e}  {sB:>8.0f}×")
    print(f"\n  Boş kafes tabanı (alt sınır): {f_bos:+.3e} rad/s")
    totA = sum(abs(a) for a in ampA_hist)
    totB = sum(abs(x) for x in ampB_hist) * 2
    print(f"  Trim bütçesi: kol A {totA*1e6:.1f}μm, kol B {totB*1e6:.1f}μm")

    # CO=False kalibrasyonunu kaydet
    with open("test_b_ck_cofalse.json", "w") as fh:
        json.dump({
            "_aciklama": "CO=False (eksenden fırlatma) mod katsayıları; "
                         "cos fazı, A=10μm, t2=1ms, üniform 0.2 T/m",
            "A_mode_m": A_MODE, "t2_s": T2, "f_bos": f_bos,
            "c_k": {str(k): c[k] for k in K_LIST},
        }, fh, indent=2)
    print("\nKaydedildi: test_b_ck_cofalse.json")

    elapsed = time.time() - t0
    print(f"Toplam süre: {elapsed:.1f} s")

    # ══ Figür ═════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5))
    fig.suptitle("Trim döngüsü gerçekçi koşulda: COD'ye oturtma YOK "
                 "(eksenden fırlatma)", fontsize=13)

    ax = axes[0]
    cv = [c[k]*A_MODE for k in K_LIST]
    colors = ['tab:red' if v > 0 else 'tab:blue' for v in cv]
    ax.bar(K_LIST, cv, color=colors, alpha=0.85)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel("Fourier modu k")
    ax.set_ylabel(f"dSy/dt [rad/s]  (A={A_MODE*1e6:.0f}μm)")
    ax.set_title("c_k işaret tablosu (CO=False)")
    ax.set_xticks(K_LIST); ax.grid(True, axis='y', alpha=0.3)

    ax = axes[1]
    AA = np.array(amps_all)
    ff = np.array([res[f"lin_{A}"] if A != A_MODE else res["k2"]
                   for A in amps_all])
    ax.plot(AA*1e6, ff, 'o-', color='tab:purple', ms=9)
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xlabel("k=2 genliği A [μm]")
    ax.set_ylabel("dSy/dt [rad/s]")
    ax.set_title(f"Doğrusallık (k=2): f/A yayılımı %{lin_spread*100:.1f}")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    steps = np.arange(N_ITER + 1)
    ax.semilogy(steps, np.abs(fA_hist), 'o-', color='tab:green', ms=9, lw=2,
                label='kol A: trim k=3')
    ax.semilogy(steps, np.abs(fB_hist), 's--', color='tab:blue', ms=9, lw=2,
                label='kol B: k=3+k=4')
    ax.axhline(abs(f_bos), color='gray', ls='--', alpha=0.8,
               label=f'boş kafes tabanı ({abs(f_bos):.1e})')
    ax.set_xticks(steps)
    ax.set_xlabel("iterasyon adımı")
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("Ölç-trimle döngüsü (CO=False)")
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    out = "test_b_trim_realistic.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()
