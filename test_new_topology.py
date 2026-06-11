#!/usr/bin/env python3
"""test_new_topology.py — Yeni topoloji (6-elemanlı FODO) testleri.

Hipotez: Tüm bükmesi tek deflektörde toplanan yeni FODO topolojisinde
simetrik kuadrupol hizalama modları daha yüksek yörünge kazancına
ve daha düşük spin kuplajına sahip olacak → sahte EDM tabanı
2.5×10⁻⁴ rad/s'den 10⁻⁵ rad/s düzeyine inebilir.

Bölüm A: k=1..4 antisimetrik ve simetrik modların yörünge kazanç karşılaştırması
Bölüm B: Rastgele desen (seed=321) ile tam trim testi
Bölüm C: Simetrik vs antisimetrik ayrışımı
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

# ── Parametreler ─────────────────────────────────────────────────────────────
PATTERN_RMS  = 1e-4      # rastgele hizalama hatası RMS [m] = 100 μm
PATTERN_SEED = 321
A_CAL        = 1e-4      # kalibrasyon mod genliği [m] = 100 μm
T2           = 1e-3      # simülasyon süresi [s]
RETURN_STEPS = 6000      # spin koşumları için Poincaré adımı
K_CAL        = [1, 2, 3, 4]


def _suppress_stdout():
    fd = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.close(null)
    return fd


def _restore_stdout(fd):
    os.dup2(fd, 1)
    os.close(fd)


def mode_vec_antisym(n_q, k):
    """k modunun antisimetrik (cos, sin) birim desen çifti.
    QF ve QD zıt yönlerde: c[2*i] = cos, c[2*i+1] = -cos."""
    n_fodo = n_q // 2
    c = np.zeros(n_q)
    s = np.zeros(n_q)
    for i in range(n_fodo):
        phi = 2 * np.pi * k * i / n_fodo
        c[2*i]   =  np.cos(phi)
        c[2*i+1] = -np.cos(phi)
        s[2*i]   =  np.sin(phi)
        s[2*i+1] = -np.sin(phi)
    nc = np.linalg.norm(c)
    ns = np.linalg.norm(s)
    return c / nc if nc > 0 else c, s / ns if ns > 0 else s


def mode_vec_symm(n_q, k):
    """k modunun simetrik (cos, sin) birim desen çifti.
    QF ve QD aynı yönde: c[2*i] = cos, c[2*i+1] = cos."""
    n_fodo = n_q // 2
    c = np.zeros(n_q)
    s = np.zeros(n_q)
    for i in range(n_fodo):
        phi = 2 * np.pi * k * i / n_fodo
        c[2*i]   = np.cos(phi)
        c[2*i+1] = np.cos(phi)
        s[2*i]   = np.sin(phi)
        s[2*i+1] = np.sin(phi)
    nc = np.linalg.norm(c)
    ns = np.linalg.norm(s)
    return c / nc if nc > 0 else c, s / ns if ns > 0 else s


def _worker(task):
    """Tek koşum — integrator2 (yeni topoloji).
    mode='orbit': 48 BPM tur-ortalamalı y [m]
    mode='spin': dSy/dt [rad/s]
    """
    label, dy_list, mode, t2, return_steps = task

    import os, sys, json, tempfile, shutil
    import numpy as np
    sys.path.insert(0, BASE)
    tmp = tempfile.mkdtemp(prefix=f"ntopo_{os.getpid()}_")
    os.chdir(tmp)

    from false_edm_mode_scan import setup_fields
    from integrator2 import integrate_particle

    with open(os.path.join(BASE, "params.json")) as fh:
        config = json.load(fh)
    fields, y0, beta0, R0, p_mag, direction = setup_fields(config)
    # setup_fields, integrator'dan FieldParams alır; integrator2 için aynı
    # FieldParams yapısını kullanıyoruz (aynı arayüz)
    from integrator2 import FieldParams as FieldParams2
    fields2 = FieldParams2()
    fields2.R0 = fields.R0
    fields2.E0 = fields.E0
    fields2.E0_power = fields.E0_power
    fields2.B0ver = fields.B0ver
    fields2.B0rad = fields.B0rad
    fields2.B0long = fields.B0long
    fields2.quadG1 = fields.quadG1
    fields2.quadG0 = fields.quadG0
    fields2.sextK1 = fields.sextK1
    fields2.quadSwitch = fields.quadSwitch
    fields2.sextSwitch = fields.sextSwitch
    fields2.EDMSwitch = fields.EDMSwitch
    fields2.direction = fields.direction
    fields2.nFODO = fields.nFODO
    fields2.quadLen = fields.quadLen
    fields2.driftLen = fields.driftLen
    fields2.rfSwitch = fields.rfSwitch
    fields2.rfVoltage = fields.rfVoltage
    fields2.h = fields.h
    fields2.quadModA = fields.quadModA
    fields2.quadModF = fields.quadModF
    fields2.B0rad_harm_amp = fields.B0rad_harm_amp
    fields2.B0rad_harm_N = fields.B0rad_harm_N

    dt = float(config.get("dt", 1e-11))
    dy = np.asarray(dy_list, dtype=float)

    n = int(fields2.nFODO)

    saved = _suppress_stdout()
    try:
        if mode == "orbit":
            fields2.poincare_quad_index = 999.0
            integrate_particle(y0, 0.0, t2, dt, fields=fields2,
                               return_steps=10, quad_dy=dy)
            cd = np.loadtxt("cod_data.txt", skiprows=1)
            cd[:, 1:3] *= 1e-3            # mm → m
            # Yeni topoloji: 6 element/cell: QF(0) DRIFT(1) QD(2) DRIFT(3) DEFL(4) DRIFT(5)
            # QF girişi: elem 0, QD girişi: elem 2
            y_bpm = np.empty(2 * n)
            for k in range(n):
                y_bpm[2*k]     = cd[k*6 + 0, 2]   # QF girişi (elem 0)
                y_bpm[2*k + 1] = cd[k*6 + 2, 2]   # QD girişi (elem 2)
            result = y_bpm.tolist()
        else:
            fields2.poincare_quad_index = 0.0
            _, poin, poin_t = integrate_particle(
                y0, 0.0, t2, dt, fields=fields2,
                return_steps=return_steps, quad_dy=dy)
            if poin is None or len(poin) < 10:
                result = 0.0
            else:
                result = float(np.polyfit(np.asarray(poin_t, float),
                                          np.asarray(poin[:, 7], float), 1)[0])
    finally:
        _restore_stdout(saved)
        os.chdir(BASE)
        shutil.rmtree(tmp, ignore_errors=True)

    return label, result


def main():
    t_start = time.time()

    with open("params.json") as fh:
        config = json.load(fh)

    from false_edm_mode_scan import setup_fields
    fields, *_ = setup_fields(config)
    n_q = 2 * int(fields.nFODO)

    ctx = mp.get_context("spawn")
    nw  = mp.cpu_count()

    def run(tasks):
        with ctx.Pool(processes=min(nw, len(tasks))) as pool:
            return dict(pool.map(_worker, tasks))

    print("=" * 70)
    print("YENİ TOPOLOJİ TESTİ — 6-elemanlı FODO (tek deflektör)")
    print("=" * 70)

    # ══ BÖLÜM A: Kalibrasyon — kazanç karşılaştırması ════════════════════════
    print(f"\nBölüm A: Yörünge kazancı kalibrasyonu (k=1..4, antisim+sim)...")

    # 1 referans + 4×(cos+sin)×2 = 17 koşum
    tasks = [("ref", np.zeros(n_q).tolist(), "orbit", T2, 10)]
    for k in K_CAL:
        c_a, s_a = mode_vec_antisym(n_q, k)
        c_s, s_s = mode_vec_symm(n_q, k)
        tasks.append((f"a{k}c", (A_CAL * c_a).tolist(), "orbit", T2, 10))
        tasks.append((f"a{k}s", (A_CAL * s_a).tolist(), "orbit", T2, 10))
        tasks.append((f"s{k}c", (A_CAL * c_s).tolist(), "orbit", T2, 10))
        tasks.append((f"s{k}s", (A_CAL * s_s).tolist(), "orbit", T2, 10))

    print(f"  {len(tasks)} simülasyon ({nw} işçi)...")
    res_cal = run(tasks)

    y_ref = np.asarray(res_cal["ref"])

    # Kazanç hesabı: sütun normu / A_CAL
    G_antisym_new = {}
    G_symm_new = {}
    O_antisym_cols = {}
    O_symm_cols = {}

    for k in K_CAL:
        col_ac = (np.asarray(res_cal[f"a{k}c"]) - y_ref) / A_CAL
        col_as = (np.asarray(res_cal[f"a{k}s"]) - y_ref) / A_CAL
        col_sc = (np.asarray(res_cal[f"s{k}c"]) - y_ref) / A_CAL
        col_ss = (np.asarray(res_cal[f"s{k}s"]) - y_ref) / A_CAL
        G_antisym_new[k] = 0.5 * (np.linalg.norm(col_ac) + np.linalg.norm(col_as))
        G_symm_new[k]    = 0.5 * (np.linalg.norm(col_sc) + np.linalg.norm(col_ss))
        O_antisym_cols[k] = (col_ac, col_as)
        O_symm_cols[k]    = (col_sc, col_ss)

    # Orijinal değerler (test_orbit_trim.json ve test_symm_basis_fit.json'dan)
    G_antisym_orig = {1: 6.20, 2: 24.08, 3: 6.29, 4: 2.26}
    G_symm_orig    = {1: 0.975, 2: 4.73, 3: 1.577, 4: 0.715}

    print(f"\n{'─'*80}")
    print("Yörünge kazanç karşılaştırması (A_CAL sütun normu)")
    print(f"{'─'*80}")
    print(f"{'k':>3} | {'G_antisim (orig)':>17} | {'G_antisim (yeni)':>17} | "
          f"{'G_sim (orig)':>14} | {'G_sim (yeni)':>14} | {'antisim oran':>12} | {'sim oran':>10}")
    print('─' * 80)
    for k in K_CAL:
        r_a = G_antisym_new[k] / G_antisym_orig[k] if G_antisym_orig[k] > 0 else float('nan')
        r_s = G_symm_new[k]    / G_symm_orig[k]    if G_symm_orig[k]    > 0 else float('nan')
        print(f"{k:>3} | {G_antisym_orig[k]:>17.3f} | {G_antisym_new[k]:>17.3f} | "
              f"{G_symm_orig[k]:>14.3f} | {G_symm_new[k]:>14.3f} | "
              f"{r_a:>12.3f} | {r_s:>10.3f}")

    # ══ BÖLÜM B: Tam trim testi (seed=321) ═══════════════════════════════════
    print(f"\n{'─'*70}")
    print("Bölüm B: Tam trim testi (seed=321, PATTERN_RMS=100μm)")
    print(f"{'─'*70}")

    rng_p = np.random.default_rng(PATTERN_SEED)
    P = rng_p.standard_normal(n_q) * PATTERN_RMS
    print(f"  Desen RMS: {np.std(P)*1e6:.1f} μm")

    # f0: trims öncesi sahte EDM
    print("  f0 (trim öncesi spin) koşuluyor...")
    res_f0 = run([("f0", P.tolist(), "spin", T2, RETURN_STEPS)])
    f0 = res_f0["f0"]
    print(f"  f0 = {f0:+.4e} rad/s")

    # Fit 1: antisimetrik k=1..4 ile trim
    print("  Antisimetrik k=1..4 fit+trim...")
    O_a = np.column_stack([O_antisym_cols[k][0] for k in K_CAL] +
                          [O_antisym_cols[k][1] for k in K_CAL])
    # Sıra: k1_cos, k2_cos, k3_cos, k4_cos, k1_sin, k2_sin, k3_sin, k4_sin
    # Düzenle: (k1c, k1s, k2c, k2s, ...)
    O_a_ordered = np.column_stack([O_antisym_cols[k][j] for k in K_CAL for j in range(2)])

    res_P = run([("oP_a", P.tolist(), "orbit", T2, 10)])
    y_P_a = np.asarray(res_P["oP_a"])
    a_hat_a, *_ = np.linalg.lstsq(O_a_ordered, y_P_a, rcond=None)

    # Trim vektörü oluştur
    P_trim_a = P.copy()
    for ki, k in enumerate(K_CAL):
        c_a, s_a = mode_vec_antisym(n_q, k)
        P_trim_a -= a_hat_a[2*ki]   * c_a * A_CAL * np.linalg.norm(c_a)**(-1) * A_CAL
        P_trim_a -= a_hat_a[2*ki+1] * s_a * A_CAL * np.linalg.norm(s_a)**(-1) * A_CAL
    # Düzeltme: a_hat zaten [m] biriminde (O normalize edilmemiş), doğrudan çıkar
    P_trim_a = P.copy()
    for ki, k in enumerate(K_CAL):
        c_a, s_a = mode_vec_antisym(n_q, k)
        P_trim_a -= a_hat_a[2*ki]   * c_a
        P_trim_a -= a_hat_a[2*ki+1] * s_a

    # Fit 2: antisimetrik + simetrik k=1..4 ile trim
    O_as_ordered = np.column_stack(
        [O_antisym_cols[k][j] for k in K_CAL for j in range(2)] +
        [O_symm_cols[k][j]    for k in K_CAL for j in range(2)]
    )
    res_P2 = run([("oP_as", P.tolist(), "orbit", T2, 10)])
    y_P_as = np.asarray(res_P2["oP_as"])
    a_hat_as, *_ = np.linalg.lstsq(O_as_ordered, y_P_as, rcond=None)

    P_trim_as = P.copy()
    for ki, k in enumerate(K_CAL):
        c_a, s_a = mode_vec_antisym(n_q, k)
        P_trim_as -= a_hat_as[2*ki]   * c_a
        P_trim_as -= a_hat_as[2*ki+1] * s_a
    for ki, k in enumerate(K_CAL):
        c_s, s_s = mode_vec_symm(n_q, k)
        offset = 2 * len(K_CAL)
        P_trim_as -= a_hat_as[offset + 2*ki]   * c_s
        P_trim_as -= a_hat_as[offset + 2*ki+1] * s_s

    # Spin ölçümleri
    print("  Trim sonrası spin koşumları (antisim+antisim+sim)...")
    spin_tasks = [
        ("f_antisym", P_trim_a.tolist(),  "spin", T2, RETURN_STEPS),
        ("f_antisym_symm", P_trim_as.tolist(), "spin", T2, RETURN_STEPS),
    ]
    res_spin = run(spin_tasks)
    f_antisym      = res_spin["f_antisym"]
    f_antisym_symm = res_spin["f_antisym_symm"]

    print(f"\n{'─'*60}")
    print("Bölüm B: Spin tabanı karşılaştırması")
    print(f"{'─'*60}")
    print(f"  {'durum':>30}  {'dSy/dt [rad/s]':>15}  {'bastırma':>10}")
    print('─' * 60)
    print(f"  {'trim öncesi':>30}  {f0:>15.4e}  {'—':>10}")
    sup_a  = abs(f0/f_antisym)  if abs(f_antisym)      > 1e-30 else float('inf')
    sup_as = abs(f0/f_antisym_symm) if abs(f_antisym_symm) > 1e-30 else float('inf')
    print(f"  {'antisim k=1..4 trim':>30}  {f_antisym:>15.4e}  {sup_a:>9.1f}×")
    print(f"  {'antisim+sim k=1..4 trim':>30}  {f_antisym_symm:>15.4e}  {sup_as:>9.1f}×")

    # ══ BÖLÜM C: Simetrik mod spin kuplajı ═══════════════════════════════════
    print(f"\n{'─'*70}")
    print("Bölüm C: Simetrik mod spin kuplajı (yeni topoloji)")
    print(f"{'─'*70}")

    A_SPIN = 1e-4   # spin kuplaj ölçümü için mod genliği [m]
    spin_coup_tasks = []
    for k in K_CAL:
        c_a, s_a = mode_vec_antisym(n_q, k)
        c_s, s_s = mode_vec_symm(n_q, k)
        spin_coup_tasks.append((f"sa{k}c", (A_SPIN*c_a).tolist(), "spin", T2, RETURN_STEPS))
        spin_coup_tasks.append((f"ss{k}c", (A_SPIN*c_s).tolist(), "spin", T2, RETURN_STEPS))

    print(f"  {len(spin_coup_tasks)} spin koşumu ({nw} işçi)...")
    res_coup = run(spin_coup_tasks)

    print(f"\n{'─'*60}")
    print("Bölüm C: Spin kuplajı karşılaştırması (tek mod, A=100μm)")
    print(f"{'─'*60}")
    print(f"{'k':>3} | {'antisim dSy/dt':>16} | {'sim dSy/dt':>16} | "
          f"{'sim/antisim oran':>16}")
    print('─' * 60)

    spin_coupling = {}
    for k in K_CAL:
        f_a = res_coup[f"sa{k}c"]
        f_s = res_coup[f"ss{k}c"]
        ratio = abs(f_s / f_a) if abs(f_a) > 1e-30 else float('nan')
        spin_coupling[k] = {"antisim": f_a, "sim": f_s, "ratio": ratio}
        print(f"{k:>3} | {f_a:>16.4e} | {f_s:>16.4e} | {ratio:>16.4f}")

    # ══ JSON çıktı ════════════════════════════════════════════════════════════
    out = {
        "_aciklama": "Yeni topoloji testi: 6-elemanlı FODO, tek deflektör",
        "topoloji": "QF|DRIFT|QD|DRIFT|DEFLECTOR(2*Phi_def)|DRIFT(2*driftLen)",
        "parametreler": {
            "PATTERN_RMS_um": PATTERN_RMS * 1e6,
            "PATTERN_SEED": PATTERN_SEED,
            "A_CAL_um": A_CAL * 1e6,
            "T2_ms": T2 * 1e3,
            "RETURN_STEPS": RETURN_STEPS,
            "K_CAL": K_CAL,
        },
        "bolum_A_kazanclar": {
            "G_antisim_orig": G_antisym_orig,
            "G_antisim_yeni": G_antisym_new,
            "G_sim_orig": G_symm_orig,
            "G_sim_yeni": G_symm_new,
            "antisim_oran_yeni_div_orig": {k: float(G_antisym_new[k] / G_antisym_orig[k]) for k in K_CAL},
            "sim_oran_yeni_div_orig":     {k: float(G_symm_new[k]    / G_symm_orig[k])    for k in K_CAL},
        },
        "bolum_B_spin_tabani": {
            "f0": float(f0),
            "f_antisim_trim": float(f_antisym),
            "f_antisim_sim_trim": float(f_antisym_symm),
            "bastirma_antisim": float(sup_a),
            "bastirma_antisim_sim": float(sup_as),
        },
        "bolum_C_spin_kuplaji": {
            str(k): {
                "antisim_dSydt": float(spin_coupling[k]["antisim"]),
                "sim_dSydt":     float(spin_coupling[k]["sim"]),
                "sim_antisim_oran": float(spin_coupling[k]["ratio"]),
            }
            for k in K_CAL
        },
        "toplam_sure_s": float(time.time() - t_start),
    }

    with open("test_new_topology.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nKaydedildi: test_new_topology.json")

    # ══ Figür ════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Yeni Topoloji Testi: 6-Elemanlı FODO (Tek Deflektör)", fontsize=13)

    # Panel 1: Antisimetrik kazanç karşılaştırması
    ax = axes[0, 0]
    kk = list(K_CAL)
    x = np.arange(len(kk))
    ax.bar(x - 0.2, [G_antisym_orig[k] for k in kk], 0.4,
           label='Orijinal (8-elem)', color='tab:blue', alpha=0.8)
    ax.bar(x + 0.2, [G_antisym_new[k]  for k in kk], 0.4,
           label='Yeni (6-elem)',     color='tab:orange', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in kk])
    ax.set_ylabel("Yörünge kazancı (RMS, boyutsuz)")
    ax.set_title("Antisimetrik mod yörünge kazancı")
    ax.set_yscale('log'); ax.legend(); ax.grid(True, which='both', axis='y', alpha=0.3)

    # Panel 2: Simetrik kazanç karşılaştırması
    ax = axes[0, 1]
    ax.bar(x - 0.2, [G_symm_orig[k] for k in kk], 0.4,
           label='Orijinal (8-elem)', color='tab:green', alpha=0.8)
    ax.bar(x + 0.2, [G_symm_new[k]  for k in kk], 0.4,
           label='Yeni (6-elem)',     color='tab:red', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in kk])
    ax.set_ylabel("Yörünge kazancı (RMS, boyutsuz)")
    ax.set_title("Simetrik mod yörünge kazancı")
    ax.set_yscale('log'); ax.legend(); ax.grid(True, which='both', axis='y', alpha=0.3)

    # Panel 3: Spin tabanı karşılaştırması
    ax = axes[1, 0]
    labels3 = ["Trim öncesi", "Antisim\nk=1..4", "Antisim+Sim\nk=1..4"]
    vals3   = [abs(f0), abs(f_antisym), abs(f_antisym_symm)]
    cols3   = ['tab:gray', 'tab:blue', 'tab:purple']
    ax.bar(labels3, vals3, color=cols3, alpha=0.85)
    ax.set_yscale('log')
    for i, v in enumerate(vals3):
        txt = "—" if i == 0 else f"{abs(f0)/v:.1f}×"
        ax.annotate(txt, (i, v), ha='center', va='bottom', fontsize=10)
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("Sahte EDM tabanı (yeni topoloji)")
    ax.grid(True, which='both', axis='y', alpha=0.3)

    # Panel 4: Simetrik mod spin kuplajı
    ax = axes[1, 1]
    f_antisym_coup = [abs(spin_coupling[k]["antisim"]) for k in kk]
    f_symm_coup    = [abs(spin_coupling[k]["sim"])     for k in kk]
    ax.bar(x - 0.2, f_antisym_coup, 0.4, label='Antisimetrik', color='tab:blue',   alpha=0.8)
    ax.bar(x + 0.2, f_symm_coup,    0.4, label='Simetrik',     color='tab:orange', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in kk])
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("Spin kuplajı: simetrik vs antisimetrik modlar")
    ax.set_yscale('log'); ax.legend(); ax.grid(True, which='both', axis='y', alpha=0.3)

    plt.tight_layout()
    fout = "test_new_topology.png"
    plt.savefig(fout, dpi=150)
    print(f"Figür kaydedildi: {fout}")
    print(f"Toplam süre: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
