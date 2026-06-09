#!/usr/bin/env python3
"""harmonic_orbit_correction.py — Harmonik yörünge düzeltmesi ve false EDM bastırımı.

Temel soru:
  "k=1,2,3 düzeltmesi neden yalnızca %20-30 iyileştirme veriyor?"

Yanıt:
  false_edm_mode_scan.py F(k)'yı eşit genlikle ölçer → k=2 baskın görünür.
  Ancak absolute katkı F(k) × A(k)'dır:
    k=2, A=10μm  → F(2) × 10μm
    k=4, A=100μm → F(4) × 100μm ≈ F(2)/10 × 100μm = 10 × (F(2) × 1μm)
  Yüksek genlikli k≥4 modları toplam false EDM'i domine edebilir.

Akış:
  1. params.json dy_harmonics veya rastgele misalignment → quad_dy
  2. Eşleşmiş-filtre proxy: her modun orbit katkısı → false EDM tahmini
  3. Kademeli R-LS düzeltme (k=1,2,3 → baskın modlar → tam düzeltme)
  4. Her aşamada spin simülasyonu → gerçek false EDM bastırım oranı

Kullanım:
  python3 harmonic_orbit_correction.py
  python3 harmonic_orbit_correction.py --sigma 1e-4      # rastgele 100μm RMS
  python3 harmonic_orbit_correction.py --bpm-noise 1e-5  # BPM gürültüsüyle
  python3 harmonic_orbit_correction.py --kmax-corr 8     # k=1..8 tam düzeltme
"""

import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from false_edm_mode_scan import setup_fields, find_closed_orbit, _make_state, C
from fourier_reconstruct import fodo_basis


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı: misalignment vektörü oluşturma
# ─────────────────────────────────────────────────────────────────────────────

def build_dy_from_harmonics(harmonics, n_q, antisym=True):
    """params.json dy_harmonics listesinden quad_dy [m] vektörü oluşturur."""
    dy = np.zeros(n_q)
    for h in harmonics:
        k = int(h["k"])
        if k == 0:
            continue
        Fk, _ = fodo_basis(n_q, [k], antisym)
        dy += Fk[:, 0] * float(h.get("amp_cos", 0.0))
        dy += Fk[:, 1] * float(h.get("amp_sin", 0.0))
    return dy


# ─────────────────────────────────────────────────────────────────────────────
# Proxy analiz: orbit eşleşmiş-filtre katkıları
# ─────────────────────────────────────────────────────────────────────────────

def mode_contributions_proxy(dy, R, n_q, k_max=12, antisym=True):
    """Her modun orbit normuna katkısı — false EDM için hızlı proxy.

    Gerçek false EDM ∝ F(k) × A(k), ancak F(k) ∝ ‖M_k‖ = ‖R F_k‖ olduğundan
    matched-filter projeksiyon normu makul bir göstergedir.
    Kesin değer için spin simülasyonu gerekir; bu proxy hangi modların
    düzeltileceğine karar vermek içindir.
    """
    y_orbit = R @ dy
    out = {}
    for k in range(1, k_max + 1):
        Fk, _ = fodo_basis(n_q, [k], antisym)
        Mk = R @ Fk
        # Matched-filter: modu M_k altuzayına yansıt
        a_k, *_ = np.linalg.lstsq(Mk, y_orbit, rcond=None)
        y_k = Mk @ a_k
        amp_um = float(np.linalg.norm(a_k)) * 1e6
        orbit_um = float(np.linalg.norm(y_k)) * 1e6
        out[k] = {"amp_um": amp_um, "orbit_norm_um": orbit_um}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# R-LS harmonik düzeltmesi
# ─────────────────────────────────────────────────────────────────────────────

def apply_rls_correction(dy, ks_correct, R, n_q, antisym=True,
                         bpm_noise_sigma=0.0, rng_seed=42):
    """Orbit ölçümünden R-LS ile belirtilen modları düzelt.

    Fiziksel model: orbiti ölç, baskın harmonikleri kestir,
    dipol düzeltici eşdeğeri olarak dy'den çıkar.

    Returns
    -------
    dy_corr : ndarray — düzeltilmiş misalignment vektörü
    est_amps : dict   — k → kestirilen genlik [μm]
    """
    if len(ks_correct) == 0:
        return dy.copy(), {}

    rng = np.random.default_rng(rng_seed)
    y_meas = R @ dy + rng.normal(0, bpm_noise_sigma, n_q)

    # Tüm düzeltilecek modlar için ortak F ve M = R F matrisi
    F_blocks = []
    for k in ks_correct:
        Fk, _ = fodo_basis(n_q, [k], antisym)
        F_blocks.append(Fk)
    F = np.column_stack(F_blocks)   # [n_q, 2*len(ks_correct)]
    M = R @ F                        # [n_q, 2*len(ks_correct)]

    a_hat, *_ = np.linalg.lstsq(M, y_meas, rcond=None)
    dy_corr = dy - F @ a_hat

    est_amps = {}
    idx = 0
    for k in ks_correct:
        est_amps[k] = float(np.hypot(a_hat[idx], a_hat[idx + 1])) * 1e6
        idx += 2

    return dy_corr, est_amps


# ─────────────────────────────────────────────────────────────────────────────
# Spin simülasyonu (false EDM ölçümü)
# ─────────────────────────────────────────────────────────────────────────────

def measure_false_edm(dy_vec, config, t2=5e-4, return_steps=5000,
                      dt=None, co_turns=60):
    """Verilen quad_dy için stroboskopik dS_y/dt [rad/s]."""
    from integrator import integrate_particle
    fields, _, beta0, R0, p_mag, direction = setup_fields(config)
    if dt is None:
        dt = float(config.get("dt", 1e-11))

    circ = (2*np.pi*R0 + 4*fields.nFODO*fields.driftLen
            + 2*fields.nFODO*fields.quadLen)
    T_rev = circ / (beta0 * C)

    v_co, _ = find_closed_orbit(fields, p_mag, direction, dy_vec,
                                dt, T_rev, n_turns=co_turns)
    y_launch = _make_state(v_co, p_mag, direction, [0.0, 0.0, direction])
    co_off_mm = float(np.hypot(v_co[0], v_co[1]) * 1e3)

    fields.poincare_quad_index = 0.0
    _, poin, poin_t = integrate_particle(
        y_launch, 0.0, t2, dt, fields=fields,
        return_steps=return_steps, quad_dy=dy_vec)

    sy = np.asarray(poin[:, 7], float)
    ts = np.asarray(poin_t, float)
    slope = float(np.polyfit(ts, sy, 1)[0])
    return slope, co_off_mm


def _spin_worker(task):
    """Paralel spin-takip işçisi."""
    import os, sys, json, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    label, dy_list, t2, return_steps = task
    with open("params.json") as fh:
        config = json.load(fh)
    t0 = time.time()
    slope, co_off = measure_false_edm(np.asarray(dy_list), config,
                                      t2=t2, return_steps=return_steps)
    return label, slope, co_off, time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Ana program
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse, multiprocessing as mp
    par = argparse.ArgumentParser(
        description="Harmonik yörünge düzeltmesi — false EDM bastırımı")
    par.add_argument("--sigma", type=float, default=None,
                     help="Rastgele misalignment RMS [m]. Verilmezse params.json kullanılır.")
    par.add_argument("--seed", type=int, default=42)
    par.add_argument("--t2", type=float, default=5e-4, help="Simülasyon süresi [s]")
    par.add_argument("--steps", type=int, default=5000)
    par.add_argument("--bpm-noise", type=float, default=0.0,
                     help="BPM elektronik gürültüsü [m] (düzeltme adımında)")
    par.add_argument("--kmax-corr", type=int, default=8,
                     help="Tam düzeltme aşamasında maksimum k (varsayılan: 8)")
    par.add_argument("--proxy-only", action="store_true",
                     help="Yalnızca proxy analizi yap, spin simülasyonu çalıştırma")
    args = par.parse_args()

    with open("params.json") as fh:
        config = json.load(fh)

    if not os.path.exists("R_dy_1.npy"):
        print("HATA: R_dy_1.npy bulunamadı. Önce çalıştırın:")
        print("  python3 build_response_matrix.py")
        return
    R = np.load("R_dy_1.npy")
    n_q = R.shape[0]
    antisym = config.get("smooth_antisym_fodo", True)

    # ── Misalignment ──────────────────────────────────────────────────────────
    if args.sigma is not None:
        rng = np.random.default_rng(args.seed)
        dy = rng.normal(0, args.sigma, n_q)
        src = f"rastgele σ={args.sigma*1e6:.0f} μm (seed={args.seed})"
    else:
        dy = build_dy_from_harmonics(config.get("dy_harmonics", []), n_q, antisym)
        src = "params.json dy_harmonics"
        if np.max(np.abs(dy)) < 1e-9:
            print("UYARI: dy_harmonics boş. --sigma parametresi kullanın.")
            return

    print(f"\n{'='*70}")
    print(f"  HARMONİK YÖRÜNGE DÜZELTMESİ")
    print(f"  Kaynak: {src}")
    print(f"  |dy| RMS = {np.sqrt(np.mean(dy**2))*1e6:.1f} μm  "
          f"  maks = {np.max(np.abs(dy))*1e6:.1f} μm")
    print(f"{'='*70}")

    # ── Proxy katkı analizi ───────────────────────────────────────────────────
    contribs = mode_contributions_proxy(dy, R, n_q, k_max=12, antisym=antisym)
    total_orbit = sum(c["orbit_norm_um"] for c in contribs.values())

    print(f"\n  {'k':>3}  {'A_k [μm]':>10}  {'orbit [μm]':>12}  {'katkı %':>9}  {'':>4}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*12}  {'-'*9}  {'-'*4}")
    dominant = []
    for k in range(1, 13):
        c = contribs[k]
        frac = c["orbit_norm_um"] / total_orbit * 100 if total_orbit > 0 else 0.0
        mark = " ◄" if frac >= 5.0 else ""
        if frac >= 5.0:
            dominant.append(k)
        print(f"  {k:>3}  {c['amp_um']:>10.2f}  "
              f"{c['orbit_norm_um']:>12.2f}  {frac:>8.1f}%{mark}")
    print(f"  {'':>3}  {'':>10}  {total_orbit:>12.2f}  {'100.0':>9}%  (toplam)")
    print(f"\n  Baskın modlar (>5% orbit katkısı): k = {dominant}")
    print(f"\n  [NOT] Bu proxy orbit normunu kullanır. Gerçek false EDM için")
    print(f"  spin simülasyonu aşağıda çalışacak.")

    if args.proxy_only:
        print("\n  (--proxy-only: spin simülasyonu atlandı)")
        return

    # ── Düzeltme aşamaları ────────────────────────────────────────────────────
    # Mevcut params.json'daki harmonik modları bul
    harm_ks = sorted({int(h["k"]) for h in config.get("dy_harmonics", []) if h["k"] > 0})

    stages_def = [
        ("Baseline (düzeltme yok)",     []),
        ("k=1,2,3",                     [1, 2, 3]),
    ]
    # Baskın modlar aşaması (k=1,2,3'ten farklıysa)
    if sorted(dominant) != sorted([1, 2, 3]) and dominant:
        stages_def.append((f"Baskın modlar {dominant}", sorted(dominant)))
    # params.json modları (yukardakilerden farklıysa)
    if harm_ks and sorted(harm_ks) not in [sorted([1,2,3]), sorted(dominant)]:
        stages_def.append((f"params.json modları {harm_ks}", harm_ks))
    # Tam düzeltme
    full_ks = list(range(1, args.kmax_corr + 1))
    if sorted(full_ks) not in [sorted(s[1]) for s in stages_def]:
        stages_def.append((f"Tam k=1..{args.kmax_corr}", full_ks))

    # Tekrar edenleri çıkar
    seen_keys = set()
    stages = []
    for label, ks in stages_def:
        key = frozenset(ks)
        if key not in seen_keys:
            seen_keys.add(key)
            stages.append((label, ks))

    # Her aşama için dy_corrected hesapla
    print(f"\n{'='*70}")
    print(f"  SPİN SİMÜLASYONU  ({len(stages)} aşama, paralel)")
    print(f"  t2={args.t2*1e3:.1f} ms  steps={args.steps}  "
          f"BPM gürültüsü={args.bpm_noise*1e6:.1f} μm")
    print(f"{'='*70}")

    tasks = []
    for label, ks in stages:
        if len(ks) == 0:
            dy_stage = dy.copy()
        else:
            dy_stage, est = apply_rls_correction(
                dy, ks, R, n_q, antisym=antisym,
                bpm_noise_sigma=args.bpm_noise, rng_seed=args.seed)
            amps_str = ", ".join(f"k{k}={est.get(k,0):.1f}μm" for k in ks[:4])
            if len(ks) > 4:
                amps_str += "…"
            print(f"  [{label}] kestirilen genlikler: {amps_str}")
        tasks.append((label, dy_stage.tolist(), args.t2, args.steps))

    nproc = min(len(tasks), max(1, mp.cpu_count()))
    print(f"\n  {nproc} paralel süreç başlatılıyor...")
    t_wall = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(nproc) as pool:
        raw = pool.map(_spin_worker, tasks)
    wall = time.time() - t_wall

    res = {lbl: (slope, co, rt) for lbl, slope, co, rt in raw}
    baseline_edm = abs(res["Baseline (düzeltme yok)"][0])

    # ── Sonuç tablosu ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {'Aşama':40}  {'dSy/dt [r/s]':>14}  {'bastırım':>10}  {'CO [mm]':>8}")
    print(f"  {'-'*40}  {'-'*14}  {'-'*10}  {'-'*8}")
    suppression_vals = []
    for label, ks in stages:
        edm, co, rt = res[label]
        supp = baseline_edm / abs(edm) if abs(edm) > 1e-30 else float("inf")
        suppression_vals.append(supp)
        flag = " ◄◄" if supp >= 10 else (" ◄" if supp >= 3 else "")
        print(f"  {label:40}  {edm:>+14.3e}  {supp:>9.1f}×{flag}  {co:>7.3f}")
    print(f"{'='*70}")
    print(f"  Toplam duvar-saati: {wall:.0f} s")

    # ── Grafik ────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Panel 1: Mod katkıları (proxy)
        ax = axes[0]
        ks_p = list(range(1, 13))
        orb = [contribs[k]["orbit_norm_um"] for k in ks_p]
        amps = [contribs[k]["amp_um"] for k in ks_p]
        colors_p = ["tab:red" if k in dominant else "steelblue" for k in ks_p]
        ax.bar(ks_p, orb, color=colors_p, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Fourier modu k", fontsize=11)
        ax.set_ylabel("Eşleşmiş-filtre orbit katkısı [μm]", fontsize=10)
        ax.set_title("False EDM Proxy: Her Modun Katkısı", fontsize=11)
        ax.set_xticks(ks_p)
        ax.legend(handles=[Patch(color="tab:red", label="Baskın (>5%)"),
                            Patch(color="steelblue", label="Küçük (<5%)")],
                  fontsize=9)
        # Misalignment genliği ikincil eksen
        ax2 = ax.twinx()
        ax2.plot(ks_p, amps, "o--", color="gray", ms=5, lw=1.2, label="A_k [μm]")
        ax2.set_ylabel("Misalignment A_k [μm]", fontsize=9, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        ax2.legend(fontsize=8, loc="upper right")

        # Panel 2: Kademeli düzeltme bastırım oranları
        ax = axes[1]
        stage_names = [lbl.replace("Baseline (düzeltme yok)", "Baseline")
                       for lbl, _ in stages]
        xpos = np.arange(len(stages))
        bar_colors = ["#d62728"] + ["#2ca02c"] * (len(stages) - 1)
        bars = ax.bar(xpos, suppression_vals, color=bar_colors,
                      edgecolor="black", linewidth=0.5)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels(stage_names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("False EDM bastırım oranı (×)", fontsize=11)
        ax.set_title("Kademeli Harmonik Düzeltme Etkinliği", fontsize=11)
        for bar, val in zip(bars, suppression_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(suppression_vals) * 0.02,
                    f"{val:.1f}×", ha="center", va="bottom", fontsize=9)

        # Panel 3: |dSy/dt| karşılaştırması
        ax = axes[2]
        edm_vals = [abs(res[lbl][0]) for lbl, _ in stages]
        bars2 = ax.bar(xpos, np.array(edm_vals) * 1e9, color=bar_colors,
                       edgecolor="black", linewidth=0.5)
        ax.set_xticks(xpos)
        ax.set_xticklabels(stage_names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(r"$|dS_y/dt|$ [rad/s × 10⁻⁹]", fontsize=11)
        ax.set_title("Mutlak False EDM Sinyali", fontsize=11)
        ax.set_yscale("log")

        fig.suptitle(f"Harmonik Yörünge Düzeltmesi  |  {src}", fontsize=12)
        fig.tight_layout()
        out = "harmonic_correction_result.png"
        fig.savefig(out, dpi=140)
        print(f"\n  → {out} kaydedildi")

    except Exception as e:
        print(f"  [grafik hatası: {e}]")


if __name__ == "__main__":
    main()
