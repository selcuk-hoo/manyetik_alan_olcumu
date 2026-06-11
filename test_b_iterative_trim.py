#!/usr/bin/env python3
"""test_b_iterative_trim.py — Test B (iteratif): ölç-trimle-tekrarla stratejisi.

Soru:
  Mod haritası evrensel ama %3-10 belirsizlikle (test_b_mode_map). Tek atışlık
  trim bastırmayı ~10-30× ile sınırlar. Ölçüm-trim döngüsü iterasyonla 468×
  sınıfına (hatta ötesine) ulaşır mı?

Deneysel gerçekçilik:
  Bu şemada hizalama desenini BİLMEK GEREKMEZ. Ölçülen tek şey dSy/dt
  (spin tabanlı, deneyde erişilebilir). Trim reçetesi:
      A_trim = −f_ölçülen / c_trim
  c_trim bir kez kalibre edilmiş katsayı (test_b_ck_table.json). Trim
  uygulanır, kalan f tekrar ölçülür, küçük ikinci trim eklenir.

Yöntem (üniform 0.2 lattice, CO=True):
  Yeni rastgele desen P (seed 99, RMS 10μm) — haritanın hiç görmediği
  konfigürasyon.
  İki kol:
    A) trim tek modda: k=3
    B) trim bölüşmüş: k=3 + k=4 (eşit genlik x, c3+c4 ile)
  Adımlar: f0 = f(P) → trim1 → f1 → trim2 → f2  (gerekirse trim3 → f3)

Çıktı:
  test_b_iterative_trim.png
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
PATTERN_SEED = 99       # haritanın hiç görmediği taze konfigürasyon
BG_RMS       = 1e-5     # desen RMS [m] = 10 μm
N_ITER       = 3        # ölçüm-trim döngüsü sayısı
T2           = 5e-4
CO_TURNS     = 24
CO_ITER      = 1
RETURN_STEPS = 3000
CK_FILE      = "test_b_ck_table.json"


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
    c = {int(k): float(v) for k, v in ck_data["c_k"].items()}

    from false_edm_mode_scan import setup_fields
    fields, *_ = setup_fields(config)
    n_q     = 2 * int(fields.nFODO)
    antisym = config.get("smooth_antisym_fodo", True)
    ctx     = mp.get_context("spawn")

    rng = np.random.default_rng(PATTERN_SEED)
    P = rng.standard_normal(n_q) * BG_RMS

    print(f"Lattice üniform {fields.quadG1} T/m. "
          f"Taze desen seed={PATTERN_SEED} (RMS {BG_RMS*1e6:.0f}μm)")
    print(f"Trim katsayıları (kalibrasyon): c_3={c[3]:.3e}, c_4={c[4]:.3e} rad/s/m")

    # ── Adım 0: taban ölçümü ─────────────────────────────────────────────
    print("\nAdım 0: f0 = f(P) ölçülüyor...")
    with ctx.Pool(processes=1) as pool:
        (_, f0), = pool.map(_worker, [("f0", P.tolist(), T2, CO_TURNS,
                                        CO_ITER, RETURN_STEPS)])
    print(f"  f0 = {f0:+.3e} rad/s")

    # ── İteratif döngü: iki kol paralel ──────────────────────────────────
    # Kol A: tek mod k=3.  Kol B: bölüşmüş k=3 + k=4.
    trimA = np.zeros(n_q)        # birikmiş trim deseni (kol A)
    trimB = np.zeros(n_q)
    fA_hist = [f0]
    fB_hist = [f0]
    ampA_hist, ampB_hist = [], []   # adım başına trim genlikleri [m]

    for it in range(1, N_ITER + 1):
        # Yeni trim genlikleri: ölçülen son f'den
        aA = -fA_hist[-1] / c[3]                  # kol A: k=3
        xB = -fB_hist[-1] / (c[3] + c[4])         # kol B: k=3+k=4 eşit pay
        trimA_new = trimA + mode_vec(n_q, 3, aA, antisym)
        trimB_new = (trimB + mode_vec(n_q, 3, xB, antisym)
                            + mode_vec(n_q, 4, xB, antisym))

        tasks = [
            (f"A{it}", (P + trimA_new).tolist(), T2, CO_TURNS, CO_ITER, RETURN_STEPS),
            (f"B{it}", (P + trimB_new).tolist(), T2, CO_TURNS, CO_ITER, RETURN_STEPS),
        ]
        print(f"\nAdım {it}: kol A trim k=3 ({aA*1e6:+.2f}μm), "
              f"kol B trim k=3+4 ({xB*1e6:+.2f}μm × 2) — ölçülüyor...")
        with ctx.Pool(processes=2) as pool:
            r = dict(pool.map(_worker, tasks))

        trimA, trimB = trimA_new, trimB_new
        fA_hist.append(r[f"A{it}"]); fB_hist.append(r[f"B{it}"])
        ampA_hist.append(aA);         ampB_hist.append(xB)
        print(f"  kol A: f{it} = {fA_hist[-1]:+.3e}  "
              f"(bastırma {abs(f0/fA_hist[-1]):.0f}×)")
        print(f"  kol B: f{it} = {fB_hist[-1]:+.3e}  "
              f"(bastırma {abs(f0/fB_hist[-1]):.0f}×)")

    # ── Özet ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("İteratif trim özeti (ölç → trimle → tekrar ölç)")
    print(f"{'─'*70}")
    print(f"{'adım':>5}  {'kol A (k=3)':>14}  {'bastırma':>9}  "
          f"{'kol B (k=3+4)':>14}  {'bastırma':>9}")
    print('─'*70)
    for i in range(N_ITER + 1):
        sA = abs(f0/fA_hist[i]) if fA_hist[i] != 0 else float('inf')
        sB = abs(f0/fB_hist[i]) if fB_hist[i] != 0 else float('inf')
        print(f"{i:>5}  {fA_hist[i]:>14.3e}  {sA:>8.0f}×  "
              f"{fB_hist[i]:>14.3e}  {sB:>8.0f}×")

    totA = sum(abs(a) for a in ampA_hist)
    totB = sum(abs(x) for x in ampB_hist) * 2
    print(f"\n  Toplam trim genlik bütçesi: kol A {totA*1e6:.1f}μm (k=3), "
          f"kol B {totB*1e6:.1f}μm (k=3,4 toplam)")

    elapsed = time.time() - t0
    print(f"\nToplam süre: {elapsed:.1f} s")
    print("\nYORUM:")
    print("  Desen bilgisi KULLANILMADI — yalnızca ölçülen dSy/dt ve kalibre c_k.")
    print("  1. atış katsayı belirsizliğiyle sınırlı (~10-30× beklenir);")
    print("  2.-3. atış belirsizliğin karesi/küpü ile derinleşmeli.")

    # ── Figür ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 6))
    steps = np.arange(N_ITER + 1)
    ax.semilogy(steps, np.abs(fA_hist), 'o-', color='tab:green', ms=9, lw=2,
                label='kol A: trim k=3')
    ax.semilogy(steps, np.abs(fB_hist), 's--', color='tab:blue', ms=9, lw=2,
                label='kol B: trim k=3+k=4 bölüşmüş')
    ax.axhline(1e-9, color='gray', ls='--', alpha=0.7, label='1e-9 (Omarov hedef)')
    for i in range(N_ITER + 1):
        sA = abs(f0/fA_hist[i]) if fA_hist[i] != 0 else float('inf')
        ax.annotate(f"{sA:.0f}×", (i, abs(fA_hist[i])), fontsize=8,
                    textcoords="offset points", xytext=(8, 6), color='tab:green')
    ax.set_xticks(steps)
    ax.set_xlabel("iterasyon adımı (0 = trimsiz)")
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title(f"İteratif ölç-trimle döngüsü (taze desen seed={PATTERN_SEED})\n"
                 "desen bilgisi yok — yalnız ölçülen dSy/dt + kalibre c_k")
    ax.legend(fontsize=10); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    out = "test_b_iterative_trim.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()
