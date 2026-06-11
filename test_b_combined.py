#!/usr/bin/env python3
"""test_b_combined.py — Test B (devam): k=1 ve k=3 ortakları + birleşik bastırma.

Önceki adım (test_b_partner_search.py):
  c_k tablosu k=1..10 ölçüldü (test_b_ck_table.json). k=2 için en iyi tekil
  ortaklar: k'=3 (A*=41μm, 12×) ve k'=4 (A*=121μm, 24×) — ikisi de hedef
  genlikten (10μm) çok büyük telafi genliği istiyor çünkü |c_2| baskın.

Bu test:
  Bölüm 1: k=1 hedefi → zıt işaretli ortaklar (k'=3,4) doğrulaması.
           Beklenti: |c_1|≈|c_3| → A_3* ≈ 9.5μm (KÜÇÜK genlik, ideal çift).
  Bölüm 2: k=3 hedefi → pozitif ortaklar (k'=1,2) doğrulaması.
           Beklenti: A_2* ≈ 2.4μm (c_2 baskın → çok küçük genlik yeter).
  Bölüm 3: BİRLEŞİK — a_1=a_2=a_3=10μm karışımı:
           (a) telafisiz artık,
           (b) tek serbest modla telafi: k=3'e trim ekle
               A_3^trim = −(c_1·a_1 + c_2·a_2 + c_3·a_3)/c_3,
           (c) iki modla bölüştürülmüş telafi (k=3 + k=4, genlik dengeli).
           Her durumda gerçek simülasyon artığı ölçülür.

Çıktı:
  test_b_combined.png
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


def make_mode(n_q, k, a_cos, a_sin, antisym):
    """Tek FODO Fourier modu → quad dy vektörü [m]."""
    Fk, _ = fodo_basis(n_q, [k], antisym)
    return Fk[:, 0] * a_cos + Fk[:, 1] * a_sin


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
    nw      = mp.cpu_count()

    print(f"Lattice üniform {fields.quadG1} T/m. "
          f"c_k tablosu {CK_FILE}'dan yüklendi (k=1..{max(c)}).")

    # ── Görev listesi (tek havuz, hepsi paralel) ─────────────────────────
    tasks = []

    def pair_dy(k_t, A_t, k_p, A_p):
        return (make_mode(n_q, k_t, A_t, 0.0, antisym)
                + make_mode(n_q, k_p, A_p, 0.0, antisym))

    # Bölüm 1: k=1 hedefi, ortaklar k'=3,4
    A13 = -(c[1]/c[3]) * A_MODE
    A14 = -(c[1]/c[4]) * A_MODE
    tasks.append(("p1_k3", pair_dy(1, A_MODE, 3, A13).tolist(),
                  T2, CO_TURNS, CO_ITER, RETURN_STEPS))
    tasks.append(("p1_k4", pair_dy(1, A_MODE, 4, A14).tolist(),
                  T2, CO_TURNS, CO_ITER, RETURN_STEPS))

    # Bölüm 2: k=3 hedefi, ortaklar k'=1,2
    A31 = -(c[3]/c[1]) * A_MODE
    A32 = -(c[3]/c[2]) * A_MODE
    tasks.append(("p3_k1", pair_dy(3, A_MODE, 1, A31).tolist(),
                  T2, CO_TURNS, CO_ITER, RETURN_STEPS))
    tasks.append(("p3_k2", pair_dy(3, A_MODE, 2, A32).tolist(),
                  T2, CO_TURNS, CO_ITER, RETURN_STEPS))

    # Bölüm 3: birleşik a1=a2=a3=10μm
    base = (make_mode(n_q, 1, A_MODE, 0.0, antisym)
            + make_mode(n_q, 2, A_MODE, 0.0, antisym)
            + make_mode(n_q, 3, A_MODE, 0.0, antisym))
    S_pred = (c[1] + c[2] + c[3]) * A_MODE       # lineer tahmin [rad/s]

    # (a) telafisiz
    tasks.append(("comb_raw", base.tolist(), T2, CO_TURNS, CO_ITER, RETURN_STEPS))

    # (b) tek modla telafi: k=3 trim
    A3_trim = -S_pred / c[3]
    dy_b = base + make_mode(n_q, 3, A3_trim, 0.0, antisym)
    tasks.append(("comb_k3", dy_b.tolist(), T2, CO_TURNS, CO_ITER, RETURN_STEPS))

    # (c) iki modla bölüştürme: k=3 ve k=4'e eşit |genlik| payı
    #     c_3·x + c_4·x = −S_pred → x = −S_pred/(c_3+c_4)
    x = -S_pred / (c[3] + c[4])
    dy_c = (base + make_mode(n_q, 3, x, 0.0, antisym)
                 + make_mode(n_q, 4, x, 0.0, antisym))
    tasks.append(("comb_k34", dy_c.tolist(), T2, CO_TURNS, CO_ITER, RETURN_STEPS))

    # Tekil mod referansları (artık oranı için)
    tasks.append(("solo_k1", make_mode(n_q, 1, A_MODE, 0.0, antisym).tolist(),
                  T2, CO_TURNS, CO_ITER, RETURN_STEPS))
    tasks.append(("solo_k3", make_mode(n_q, 3, A_MODE, 0.0, antisym).tolist(),
                  T2, CO_TURNS, CO_ITER, RETURN_STEPS))

    print(f"\n{len(tasks)} simülasyon ({nw} işçi)...")
    with ctx.Pool(processes=min(nw, len(tasks))) as pool:
        res = dict(pool.map(_worker, tasks))

    # ── Sonuçlar ─────────────────────────────────────────────────────────
    s1 = res["solo_k1"]; s3 = res["solo_k3"]

    print(f"\n{'─'*72}")
    print("BÖLÜM 1: k=1 hedefi (tek başına {:.3e} rad/s)".format(s1))
    print(f"{'─'*72}")
    for kp, A, lab in [(3, A13, "p1_k3"), (4, A14, "p1_k4")]:
        r = res[lab]; sup = abs(s1/r) if r != 0 else float('inf')
        print(f"  k=1(10μm) + k'={kp}(A*={A*1e6:+.1f}μm) → artık {r:+.3e}  "
              f"({sup:.0f}× bastırma)")

    print(f"\n{'─'*72}")
    print("BÖLÜM 2: k=3 hedefi (tek başına {:.3e} rad/s)".format(s3))
    print(f"{'─'*72}")
    for kp, A, lab in [(1, A31, "p3_k1"), (2, A32, "p3_k2")]:
        r = res[lab]; sup = abs(s3/r) if r != 0 else float('inf')
        print(f"  k=3(10μm) + k'={kp}(A*={A*1e6:+.1f}μm) → artık {r:+.3e}  "
              f"({sup:.0f}× bastırma)")

    raw  = res["comb_raw"]
    cb   = res["comb_k3"]
    cc   = res["comb_k34"]
    sup_b = abs(raw/cb) if cb != 0 else float('inf')
    sup_c = abs(raw/cc) if cc != 0 else float('inf')

    print(f"\n{'─'*72}")
    print("BÖLÜM 3: birleşik desen a_1=a_2=a_3=10μm")
    print(f"{'─'*72}")
    print(f"  Lineer tahmin (Σc_k·a_k)      : {S_pred:+.3e} rad/s")
    print(f"  (a) telafisiz gerçek artık    : {raw:+.3e} rad/s")
    print(f"  (b) k=3 trim ({A3_trim*1e6:+.1f}μm)     : {cb:+.3e} rad/s  "
          f"({sup_b:.0f}× bastırma)")
    print(f"  (c) k=3+k=4 bölüşmüş ({x*1e6:+.1f}μm her biri): {cc:+.3e} rad/s  "
          f"({sup_c:.0f}× bastırma)")

    elapsed = time.time() - t0
    print(f"\nToplam süre: {elapsed:.1f} s")
    print("\nYORUM:")
    print("  Bölüm 1-2: |c| dengeli çiftlerde (k=1↔k=3) telafi genliği küçük → pratik.")
    print("  Bölüm 3: tek denklem Σc_k·a_k=0 olduğundan TEK serbest mod yeterli;")
    print("  telafi genliği büyükse iki moda bölüştürmek hizalama bütçesini korur.")

    # ── Figür ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Test B (devam): k=1, k=3 ortakları ve birleşik bastırma", fontsize=13)

    ax = axes[0]
    labels = [f"k=1+k'3\n(A*={A13*1e6:.0f}μm)", f"k=1+k'4\n(A*={A14*1e6:.0f}μm)",
              f"k=3+k'1\n(A*={A31*1e6:.0f}μm)", f"k=3+k'2\n(A*={A32*1e6:.0f}μm)"]
    solos  = [abs(s1), abs(s1), abs(s3), abs(s3)]
    resids = [abs(res["p1_k3"]), abs(res["p1_k4"]),
              abs(res["p3_k1"]), abs(res["p3_k2"])]
    xx = np.arange(4)
    ax.bar(xx-0.18, solos,  width=0.36, color='tab:gray',  alpha=0.8, label='hedef tek başına')
    ax.bar(xx+0.18, resids, width=0.36, color='tab:green', alpha=0.85, label='çift artığı')
    ax.set_yscale('log')
    ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("Tekil çift telafileri")
    ax.legend(fontsize=9); ax.grid(True, axis='y', which='both', alpha=0.3)

    ax = axes[1]
    labels = ['telafisiz', f'k=3 trim\n({A3_trim*1e6:+.0f}μm)',
              f'k=3+k=4\n({x*1e6:+.0f}μm × 2)']
    vals   = [abs(raw), abs(cb), abs(cc)]
    ax.bar(range(3), vals, color=['tab:red', 'tab:green', 'tab:blue'], alpha=0.85)
    ax.set_yscale('log')
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("Birleşik desen a₁=a₂=a₃=10μm\ntek modla / iki modla bastırma")
    ax.grid(True, axis='y', which='both', alpha=0.3)

    plt.tight_layout()
    out = "test_b_combined.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()
