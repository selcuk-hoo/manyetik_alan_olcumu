#!/usr/bin/env python3
"""test_false_edm_modes.py — Fourier mod başına false EDM taraması ve kombine testler.

Dört bölüm:
  1. k=1..10 tek-mod taraması (her mod 100μm cos)
     → CO genliği [mm] + |dSy/dt| [rad/s], aralarındaki korelasyon
  2. k=1..10 aynı anda, 100μm rastgele fazlar → tek false EDM değeri
  3. k=1,2,3=0μm, k=4..10=100μm → false EDM (sadece yüksek-k katkısı)
  4. [Dünkü test tekrarı]
     k=1,2,3=10μm + k=4..10=100μm → k=1,2,3=1μm: neden %20-30 değişim?

Kapalı yörünge araması params.json["use_closed_orbit"] ile kontrol edilir.
"""

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from false_edm_mode_scan import (
    setup_fields, find_closed_orbit, _make_state, _run_one_k, run_scan, C
)
from fourier_reconstruct import fodo_basis

A100 = 100e-6   # 100 μm test genliği
A10  = 10e-6    # 10 μm
A1   = 1e-6     # 1 μm


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı: çoklu-mod quad_dy oluşturma
# ─────────────────────────────────────────────────────────────────────────────

def build_dy(mode_specs, n_q, antisym=True):
    """mode_specs: [(k, amp_cos, amp_sin), ...] → quad_dy [m]"""
    dy = np.zeros(n_q)
    for k, ac, asn in mode_specs:
        Fk, _ = fodo_basis(n_q, [k], antisym)
        dy += Fk[:, 0] * ac + Fk[:, 1] * asn
    return dy


# ─────────────────────────────────────────────────────────────────────────────
# Paralel worker: tam quad_dy vektörü → false EDM
# ─────────────────────────────────────────────────────────────────────────────

def _combo_worker(task):
    """Worker for multi-mode scenarios. Her alt-süreç bağımsız."""
    label, dy_list, use_co, t2, steps = task

    import os, sys, json, time
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from integrator import integrate_particle
    from false_edm_mode_scan import setup_fields, find_closed_orbit, _make_state, C

    with open("params.json") as fh:
        config = json.load(fh)

    dy_vec = np.asarray(dy_list)
    fields, _, beta0, R0, p_mag, direction = setup_fields(config)
    dt = float(config.get("dt", 1e-11))
    circ = (2*np.pi*R0 + 4*fields.nFODO*fields.driftLen
            + 2*fields.nFODO*fields.quadLen)
    T_rev = circ / (beta0 * C)

    t0 = time.time()
    if use_co:
        v_co, _ = find_closed_orbit(fields, p_mag, direction, dy_vec,
                                    dt, T_rev, n_turns=60)
        co_mm = float(np.hypot(v_co[0], v_co[1]) * 1e3)
        y_launch = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
    else:
        co_mm = float("nan")
        y_launch = [0.0, 0.0, 0.0, 0.0, 0.0,
                    p_mag * direction, 0.0, 0.0, direction]

    fields.poincare_quad_index = 0.0
    _, poin, poin_t = integrate_particle(
        y_launch, 0.0, t2, dt, fields=fields,
        return_steps=steps, quad_dy=dy_vec)

    sy = np.asarray(poin[:, 7], float)
    ts = np.asarray(poin_t, float)
    slope = float(np.polyfit(ts, sy, 1)[0])
    return (label, slope, co_mm, time.time() - t0)


# ─────────────────────────────────────────────────────────────────────────────
# Bölüm 1 — Tek-mod taraması
# ─────────────────────────────────────────────────────────────────────────────

def run_single_mode_scan(config, use_co, t2, steps):
    """k=1..24, her biri 100μm cos → CO genliği ve |dSy/dt|.

    k=13..24 = k=11..1'in kopyası (antisimetrik FODO bazında N=24 aliasing).
    k=22 = k=2, k=26 = k=2 gibi; 'yeni' mod k=11, k=12 (Nyquist).
    """
    print("\n" + "="*68)
    print("  BÖLÜM 1 — TEK-MOD TARAMASI  k=1..24,  A=100 μm cos")
    print(f"  use_closed_orbit = {use_co}")
    print("  (k=13..24 aliasing kontrolü: beklenen k=11..1 tekrarı)")
    print("="*68)

    results, _ = run_scan(
        k_list=list(range(1, 25)),
        amp_coef=A100,
        t2=t2,
        return_steps=steps,
        do_co=use_co,
        co_turns=60,
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Bölüm 2-4 — Kombine senaryolar
# ─────────────────────────────────────────────────────────────────────────────

def run_combo_tests(config, n_q, antisym, use_co, t2, steps):
    """Çoklu-mod test senaryoları paralel çalışır."""
    rng = np.random.default_rng(99)
    phases = rng.uniform(0, 2 * np.pi, 10)  # rastgele faz, seed=99

    # ── Senaryo tanımları ────────────────────────────────────────────────────
    combos = {}

    # B2: tüm k=1..10, 100μm, rastgele faz (seed=99)
    combos["B2: k=1..10 rastgele faz (100μm)"] = build_dy(
        [(k, A100 * np.cos(phases[k-1]), A100 * np.sin(phases[k-1]))
         for k in range(1, 11)], n_q, antisym)

    # B3: k=1,2,3=0, k=4..10=100μm cos
    combos["B3: k=4..10=100μm (k=1,2,3=0)"] = build_dy(
        [(k, A100, 0.0) for k in range(4, 11)], n_q, antisym)

    # B4a: k=1,2,3=10μm, k=4..10=100μm  [dünkü başlangıç]
    combos["B4a: k=1..3=10μm, k=4..10=100μm"] = build_dy(
        [(k, A10, 0.0) for k in [1, 2, 3]] +
        [(k, A100, 0.0) for k in range(4, 11)], n_q, antisym)

    # B4b: k=1,2,3=1μm, k=4..10=100μm   [dünkü "azaltılmış"]
    combos["B4b: k=1..3=1μm,  k=4..10=100μm"] = build_dy(
        [(k, A1, 0.0) for k in [1, 2, 3]] +
        [(k, A100, 0.0) for k in range(4, 11)], n_q, antisym)

    # B5: k=1,2,3=0 (mükemmel düzeltme), k=4..10=100μm — üst sınır kontrolü
    combos["B5: k=1..3=0,    k=4..10=100μm"] = build_dy(
        [(k, A100, 0.0) for k in range(4, 11)], n_q, antisym)
    # (B5 == B3: aynı dy vektörü — kasıtlı olarak tutuldu, kontrol amaçlı)

    print("\n" + "="*68)
    print("  BÖLÜM 2-4 — KOMBİNE SENARYOLAR")
    print(f"  use_closed_orbit = {use_co}")
    for lbl, dy in combos.items():
        rms = np.sqrt(np.mean(dy**2)) * 1e6
        print(f"  {lbl:45s}  rms={rms:.1f}μm")
    print("="*68)

    import multiprocessing as mp
    tasks = [(lbl, dy.tolist(), use_co, t2, steps)
             for lbl, dy in combos.items()]
    nproc = min(len(tasks), max(1, mp.cpu_count()))
    print(f"\n  {nproc} paralel süreç...")
    t_wall = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(nproc) as pool:
        raw = pool.map(_combo_worker, tasks)
    wall = time.time() - t_wall

    results = {lbl: (slope, co, rt) for lbl, slope, co, rt in raw}
    return combos, results, wall


# ─────────────────────────────────────────────────────────────────────────────
# Çıktı ve grafik
# ─────────────────────────────────────────────────────────────────────────────

def print_single_mode_table(scan_results):
    print(f"\n  {'k':>3}  {'CO genl [mm]':>14}  {'|dSy/dt| [r/s]':>16}  {'CO∝EDM?':>10}")
    print(f"  {'-'*3}  {'-'*14}  {'-'*16}  {'-'*10}")
    for r in scan_results:
        k = r["k"]
        co = r["co_off_mm"]
        edm = abs(r["dSy_dt"])
        print(f"  {k:>3}  {co:>14.4f}  {edm:>16.3e}  {r['runtime']:.0f}s")


def print_combo_table(combos, results, ref_label=None):
    print(f"\n  {'Senaryo':46s}  {'dSy/dt [r/s]':>14}  {'CO [mm]':>8}  {'oran':>8}")
    print(f"  {'-'*46}  {'-'*14}  {'-'*8}  {'-'*8}")
    ref_edm = None
    if ref_label and ref_label in results:
        ref_edm = abs(results[ref_label][0])
    for lbl in combos:
        slope, co_mm, rt = results[lbl]
        ratio_str = ""
        if ref_edm and ref_edm > 0:
            ratio = abs(slope) / ref_edm
            ratio_str = f"{ratio:>7.3f}"
        print(f"  {lbl:46s}  {slope:>+14.3e}  {co_mm:>8.4f}  {ratio_str}")


def make_plots(scan_results, combos, combo_results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # ── Panel 1: k vs CO genliği ve false EDM ────────────────────────────
        ax = axes[0]
        ks  = [r["k"]          for r in scan_results]
        cos = [r["co_off_mm"]  for r in scan_results]
        edm = [abs(r["dSy_dt"]) for r in scan_results]

        color1, color2 = "tab:blue", "tab:red"
        ax.bar(ks, cos, color=color1, alpha=0.6, label="CO genliği [mm]")
        ax.set_xlabel("Fourier modu k", fontsize=11)
        ax.set_ylabel("Kapalı yörünge genliği [mm]", color=color1, fontsize=10)
        ax.tick_params(axis="y", labelcolor=color1)
        ax.set_xticks(ks)
        # k=13..24 aliasing bölgesi gri arka plan
        if max(ks) >= 13:
            ax.axvspan(12.5, max(ks) + 0.5, color="lightgray", alpha=0.4,
                       label="aliasing (k=N-k)")

        ax2 = ax.twinx()
        ax2.plot(ks, edm, "o-", color=color2, lw=2, ms=7, label="|dSy/dt|")
        ax2.set_ylabel(r"$|dS_y/dt|$ [rad/s]", color=color2, fontsize=10)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_yscale("log")
        ax.set_title("Tek-mod taraması: CO genliği vs false EDM", fontsize=10)

        lines1 = [plt.Line2D([0], [0], color=color1, lw=6, alpha=0.6)]
        lines2 = [plt.Line2D([0], [0], color=color2, marker="o", lw=2)]
        ax.legend(lines1 + lines2,
                  ["CO genliği [mm]", r"$|dS_y/dt|$ [r/s]"],
                  fontsize=8, loc="upper right")

        # ── Panel 2: CO genliği vs |dSy/dt| korelasyon scatter ──────────────
        ax = axes[1]
        ax.scatter(cos, edm, c=ks, cmap="tab10", s=80, zorder=3)
        for k, c, e in zip(ks, cos, edm):
            ax.annotate(f"k={k}", (c, e), textcoords="offset points",
                        xytext=(4, 4), fontsize=8)
        # log-log korelasyon
        cos_arr, edm_arr = np.array(cos), np.array(edm)
        mask = (cos_arr > 0) & (edm_arr > 0)
        if mask.sum() > 2:
            p = np.polyfit(np.log10(cos_arr[mask]), np.log10(edm_arr[mask]), 1)
            x_line = np.linspace(min(cos_arr[mask]), max(cos_arr[mask]), 50)
            y_line = 10 ** np.polyval(p, np.log10(x_line))
            ax.plot(x_line, y_line, "k--", lw=1.2,
                    label=f"log-log eğim = {p[0]:.2f}")
            ax.legend(fontsize=9)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Kapalı yörünge genliği [mm]", fontsize=10)
        ax.set_ylabel(r"$|dS_y/dt|$ [rad/s]", fontsize=10)
        ax.set_title("CO genliği — false EDM korelasyonu (log-log)", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)

        # ── Panel 3: Kombine senaryo karşılaştırması ─────────────────────────
        ax = axes[2]
        labels = list(combos.keys())
        edm_c  = [abs(combo_results[lbl][0]) for lbl in labels]
        short  = [lbl.split(":")[0] for lbl in labels]
        colors_c = ["tab:blue", "tab:orange", "tab:red",
                    "tab:green", "tab:purple"][:len(labels)]
        bars = ax.bar(range(len(labels)), np.array(edm_c) * 1e9,
                      color=colors_c, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(short, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(r"$|dS_y/dt|$ [rad/s × 10⁻⁹]", fontsize=10)
        ax.set_yscale("log")
        ax.set_title("Kombine senaryolar: false EDM karşılaştırması", fontsize=10)
        # B4a → B4b ok çiz (dünkü test)
        lbl_a = "B4a: k=1..3=10μm, k=4..10=100μm"
        lbl_b = "B4b: k=1..3=1μm,  k=4..10=100μm"
        if lbl_a in labels and lbl_b in labels:
            ia, ib = labels.index(lbl_a), labels.index(lbl_b)
            ea = abs(combo_results[lbl_a][0]) * 1e9
            eb = abs(combo_results[lbl_b][0]) * 1e9
            ax.annotate(
                f"10μm→1μm: ×{ea/eb:.2f}",
                xy=(ib, eb), xytext=(ib + 0.5, (ea + eb) / 2),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=8, color="black")

        fig.suptitle("False EDM Mod Analizi — 100 μm test genliği", fontsize=12)
        fig.tight_layout()
        out = "test_false_edm_modes.png"
        fig.savefig(out, dpi=140)
        print(f"\n  → {out} kaydedildi")
    except Exception as e:
        print(f"  [grafik hatası: {e}]")


# ─────────────────────────────────────────────────────────────────────────────
# Ana program
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    par = argparse.ArgumentParser()
    par.add_argument("--t2",    type=float, default=1e-3)
    par.add_argument("--steps", type=int,   default=5000)
    args = par.parse_args()

    with open("params.json") as fh:
        config = json.load(fh)

    n_q     = 2 * int(config.get("nFODO", 24))
    antisym = bool(config.get("smooth_antisym_fodo", True))
    use_co  = bool(config.get("use_closed_orbit", True))

    print(f"\nFalse EDM mod analizi")
    print(f"  use_closed_orbit = {use_co}  (params.json)")
    print(f"  t2 = {args.t2*1e3:.1f} ms,  steps = {args.steps}")

    t_total = time.time()

    # ── Bölüm 1: Tek-mod taraması ─────────────────────────────────────────
    scan_results = run_single_mode_scan(config, use_co, args.t2, args.steps)
    print_single_mode_table(scan_results)

    # Hızlı korelasyon özeti
    cos = np.array([r["co_off_mm"]   for r in scan_results])
    edm = np.array([abs(r["dSy_dt"]) for r in scan_results])
    mask = (cos > 0) & (edm > 0)
    if mask.sum() > 2:
        corr = np.corrcoef(np.log10(cos[mask]), np.log10(edm[mask]))[0, 1]
        print(f"\n  log-log Pearson korelasyonu (CO ↔ |dSy/dt|): r = {corr:.3f}")

    # ── Bölüm 2-4: Kombine senaryolar ─────────────────────────────────────
    combos, combo_results, wall2 = run_combo_tests(
        config, n_q, antisym, use_co, args.t2, args.steps)

    print(f"\n  Kombine test sonuçları (B4a referans alınarak):")
    print_combo_table(combos, combo_results,
                      ref_label="B4a: k=1..3=10μm, k=4..10=100μm")
    print(f"  Duvar-saati (kombin): {wall2:.0f}s")

    # ── Dünkü test özeti ──────────────────────────────────────────────────
    lbl_a = "B4a: k=1..3=10μm, k=4..10=100μm"
    lbl_b = "B4b: k=1..3=1μm,  k=4..10=100μm"
    if lbl_a in combo_results and lbl_b in combo_results:
        ea = abs(combo_results[lbl_a][0])
        eb = abs(combo_results[lbl_b][0])
        pct = (ea - eb) / ea * 100
        print(f"\n  ── DÜNKÜ TEST TEKRARI ──────────────────────────────────────")
        print(f"  k=1,2,3: 10μm → 1μm  (k=4..10 = 100μm sabit)")
        print(f"  |dSy/dt| önce: {ea:.3e}  sonra: {eb:.3e}")
        print(f"  Düşüş: %{pct:.1f}   (beklenen: %90,  gözlemlenen: %{pct:.0f})")
        b3_lbl = "B3: k=4..10=100μm (k=1,2,3=0)"
        if b3_lbl in combo_results:
            e3 = abs(combo_results[b3_lbl][0])
            frac_highk = e3 / ea * 100
            print(f"  k=4..10 katkısı (B3/B4a): %{frac_highk:.1f} — ")
            if frac_highk > 70:
                print(f"  → Yüksek-k modlar toplam false EDM'in %{frac_highk:.0f}'ini oluşturuyor.")
                print(f"     k=1,2,3'ü düzeltsek bile %{frac_highk:.0f} hala mevcut.")

    make_plots(scan_results, combos, combo_results)
    print(f"\n  Toplam süre: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
