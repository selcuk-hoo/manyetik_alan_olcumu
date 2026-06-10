#!/usr/bin/env python3
"""test_b_trim_bpm.py — BPM ofseti/gürültüsünün ölç-trimle döngüsüne etkisi.

Önemli gözlem:
  Trim döngüsünün girdisi spin tabanlı dSy/dt ölçümü; BPM verisi döngüde
  DOĞRUDAN kullanılmaz. BPM hataları üç dolaylı kanaldan girebilir:

  Kanal A — dSy/dt ölçüm gürültüsü (polarimetre istatistiği):
    f_ölçülen = f_gerçek + ε. Beklenti: döngü tabanı = σ_ε (her adım son
    gürültü gerçekleşmesini trimler).

  Kanal B — trim aktüasyon hatası (trimler quad mover yerine BPM-referanslı
    yörünge tümseğiyle uygulanırsa): statik BPM ofseti σ_b, tümseğin k'inci
    harmonik genliğine b_k ≈ σ_b/√24 hata ekler (σ_b=100μm → b_3≈20μm →
    c_3·b_3 ≈ 1e-9 rad/s = sinyalin kendisi!). AMA: ofset statik olduğundan
    ARTIMSAL (diferansiyel) aktüasyonda iptal olur → döngü kendi hatasını
    ölçüp bir sonraki turda temizler. Test: 1. trime kasıtlı +20μm statik
    hata ekle, sonraki turların diferansiyel toparlamasını ölç.

  Kanal C — COD üzerine oturma hatası (yörünge düzeltmesi BPM'lerle yapılır;
    demet sentroidi COD'den δ kadar sapıksa koherent betatron dSy/dt
    ölçümüne bias ekler): bias(δ) eğrisi ölçülür, tolerans çıkarılır.
    Not: faz karışmış (decohere olmuş) demette sentroid COD'ye kendiliğinden
    oturur; bu kanal koherent salınım sönmeden ölçüm yapılırsa devreye girer.

Yöntem (üniform 0.2 lattice, CO=True, taze desen seed=99):
  Batch 1: f0 taban.
  Batch 2: A-adım1 (gürültülü trim), B-adım1 (aktüasyon hatalı trim),
           C: 1-atış trimli konfigde fırlatma sapması δ taraması.
  Batch 3: A-adım2, B-adım2 (diferansiyel).
  Batch 4: B-adım3.

Çıktı:
  test_b_trim_bpm.png
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
PATTERN_SEED = 99       # test_b_iterative_trim ile aynı taze desen
BG_RMS       = 1e-5
T2           = 5e-4
CO_TURNS     = 24
CO_ITER      = 1
RETURN_STEPS = 3000
CK_FILE      = "test_b_ck_table.json"

SIGMA_F_FRAC = 0.10     # Kanal A: ölçüm gürültüsü, |f0|'ın oranı
B3_ERR       = 20e-6    # Kanal B: statik aktüasyon hatası [m]
                        # (σ_b=100μm BPM ofsetinin k=3 bileşeni ≈ σ_b/√24)
DELTA_UM     = [0.0, 0.01, 0.1, 1.0, 10.0]   # Kanal C: COD'den sapma [μm]


def _suppress_stdout():
    fd = os.dup(1); null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1); os.close(null); return fd


def _restore_stdout(fd):
    os.dup2(fd, 1); os.close(fd)


def _worker(task):
    """CO bul (+opsiyonel fırlatma sapması) → dSy/dt.

    Görev: (label, dy_list, dlaunch, t2, ct, ci, rs)
    dlaunch: kapalı yörünge bulunduktan sonra fırlatma konumuna eklenen
             dikey sapma [m] (Kanal C; 0 → tam COD üzerinde).
    """
    label, dy_list, dlaunch, t2, co_turns, co_iter, return_steps = task

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
        v = list(v_co)
        v[1] = float(v[1]) + float(dlaunch)
        y_launch = _make_state(v, p_mag, direction, [0.0, 0.0, direction])
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
    Fk, _ = fodo_basis(n_q, [k], antisym)
    return Fk[:, 0] * amp


def main():
    t0 = time.time()
    with open("params.json") as fh:
        config = json.load(fh)
    with open(CK_FILE) as fh:
        ck_data = json.load(fh)
    c = {int(k): float(v) for k, v in ck_data["c_k"].items()}
    c3 = c[3]

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

    print(f"Lattice üniform {fields.quadG1} T/m, taze desen seed={PATTERN_SEED}")
    print(f"Trim modu k=3 (c_3={c3:.3e} rad/s/m)")

    # ── Batch 1: taban ───────────────────────────────────────────────────
    print("\nBatch 1: f0 ölçülüyor...")
    r = run([("f0", P.tolist(), 0.0, T2, CO_TURNS, CO_ITER, RETURN_STEPS)])
    f0 = r["f0"]
    sigma_f = SIGMA_F_FRAC * abs(f0)
    print(f"  f0 = {f0:+.3e} rad/s;  Kanal A gürültüsü σ_f = {sigma_f:.1e} (%10)")

    # ── Batch 2 ──────────────────────────────────────────────────────────
    # Kanal A adım 1: ölçüm gürültülü trim (deterministik gerçekleşme +σ_f)
    epsA0 = +sigma_f
    trimA1 = -(f0 + epsA0) / c3
    dyA1 = P + mode_vec(n_q, 3, trimA1, antisym)
    # Kanal B adım 1: doğru reçete + statik aktüasyon hatası
    trimB1_req  = -f0 / c3
    trimB1_real = trimB1_req + B3_ERR
    dyB1 = P + mode_vec(n_q, 3, trimB1_real, antisym)
    # Kanal C: 1-atış trimli konfig, fırlatma sapması taraması
    dyC = P + mode_vec(n_q, 3, trimB1_req, antisym)

    tasks2 = [("A1", dyA1.tolist(), 0.0, T2, CO_TURNS, CO_ITER, RETURN_STEPS),
              ("B1", dyB1.tolist(), 0.0, T2, CO_TURNS, CO_ITER, RETURN_STEPS)]
    for d_um in DELTA_UM:
        tasks2.append((f"C_{d_um}", dyC.tolist(), d_um*1e-6,
                       T2, CO_TURNS, CO_ITER, RETURN_STEPS))
    print(f"\nBatch 2: {len(tasks2)} simülasyon "
          f"(A1: trim {trimA1*1e6:+.2f}μm; B1: trim {trimB1_req*1e6:+.2f}"
          f"{B3_ERR*1e6:+.0f}μm hata; C: δ taraması)...")
    r2 = run(tasks2)
    fA1 = r2["A1"]; fB1 = r2["B1"]
    print(f"  A1 = {fA1:+.3e}  (beklenti ≈ −ε0 = {-epsA0:+.1e})")
    print(f"  B1 = {fB1:+.3e}  (beklenti ≈ c_3·b_3 = {c3*B3_ERR:+.1e})")

    # ── Batch 3: adım 2 ──────────────────────────────────────────────────
    epsA1 = -sigma_f
    trimA2 = -(fA1 + epsA1) / c3
    dyA2 = dyA1 + mode_vec(n_q, 3, trimA2, antisym)
    # B: diferansiyel aktüasyon — statik ofset artımda iptal olur (hatasız uygulanır)
    trimB2 = -fB1 / c3
    dyB2 = dyB1 + mode_vec(n_q, 3, trimB2, antisym)
    print(f"\nBatch 3: A2 (trim {trimA2*1e6:+.3f}μm), "
          f"B2 (diferansiyel trim {trimB2*1e6:+.2f}μm)...")
    r3 = run([("A2", dyA2.tolist(), 0.0, T2, CO_TURNS, CO_ITER, RETURN_STEPS),
              ("B2", dyB2.tolist(), 0.0, T2, CO_TURNS, CO_ITER, RETURN_STEPS)])
    fA2 = r3["A2"]; fB2 = r3["B2"]
    print(f"  A2 = {fA2:+.3e}  (beklenti ≈ −ε1 = {-epsA1:+.1e})")
    print(f"  B2 = {fB2:+.3e}")

    # ── Batch 4: B adım 3 ────────────────────────────────────────────────
    trimB3 = -fB2 / c3
    dyB3 = dyB2 + mode_vec(n_q, 3, trimB3, antisym)
    print(f"\nBatch 4: B3 (diferansiyel trim {trimB3*1e6:+.3f}μm)...")
    r4 = run([("B3", dyB3.tolist(), 0.0, T2, CO_TURNS, CO_ITER, RETURN_STEPS)])
    fB3 = r4["B3"]
    print(f"  B3 = {fB3:+.3e}")

    # ── Sonuç tabloları ──────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("KANAL A: dSy/dt ölçüm gürültüsü (σ_f = %10·|f0|)")
    print(f"{'═'*72}")
    print(f"  adım 0: {f0:+.3e}")
    print(f"  adım 1: {fA1:+.3e}  (gürültü gerçekleşmesi ε0={epsA0:+.1e})")
    print(f"  adım 2: {fA2:+.3e}  (ε1={epsA1:+.1e})")
    print(f"  → döngü tabanı ≈ σ_f: her adım SON gürültüyü trimler.")
    print(f"    Artık/σ_f oranları: adım1 {abs(fA1)/sigma_f:.2f}, "
          f"adım2 {abs(fA2)/sigma_f:.2f}")

    print(f"\n{'═'*72}")
    print(f"KANAL B: statik aktüasyon hatası (b_3={B3_ERR*1e6:.0f}μm ≈ "
          f"σ_b=100μm BPM ofsetinin k=3 bileşeni)")
    print(f"{'═'*72}")
    print(f"  adım 0: {f0:+.3e}   (trimsiz)")
    print(f"  adım 1: {fB1:+.3e}   (hatalı trim → {abs(fB1/f0):.1f}× KÖTÜLEŞME)")
    print(f"  adım 2: {fB2:+.3e}   (diferansiyel → {abs(f0/fB2):.0f}× bastırma)")
    print(f"  adım 3: {fB3:+.3e}   (diferansiyel → {abs(f0/fB3):.0f}× bastırma)")
    print(f"  → statik ofset 1. trimi bozar ama döngü ölçüp temizler;")
    print(f"    kalıcı hasar yok, 1 ek tur maliyeti var.")

    print(f"\n{'═'*72}")
    print("KANAL C: COD üzerine oturma hatası (koherent sentroid sapması δ)")
    print(f"{'═'*72}")
    fC0 = r2["C_0.0"]
    print(f"{'δ [μm]':>9}  {'f(δ) [rad/s]':>14}  {'bias=f(δ)−f(0)':>16}")
    print('─'*45)
    bias = {}
    for d_um in DELTA_UM:
        fC = r2[f"C_{d_um}"]
        bias[d_um] = fC - fC0
        print(f"{d_um:>9.2f}  {fC:>14.3e}  {bias[d_um]:>16.3e}")
    # bias eğimi (lineer kısmından)
    dd = [d for d in DELTA_UM if d > 0]
    slope_bias = np.polyfit([d*1e-6 for d in dd],
                            [bias[d] for d in dd], 1)[0]
    # tolerans: bias < |f0|/100 (1% bozulma)
    tol = abs(f0) / 100 / abs(slope_bias) if slope_bias != 0 else float('inf')
    print(f"\n  bias eğimi ≈ {slope_bias:.2e} rad/s/m "
          f"({slope_bias*1e-6:.1e} rad/s per μm)")
    print(f"  %1 ölçüm bozulması için tolerans: δ < {tol*1e9:.2f} nm (!)")
    print(f"  → koherent sentroid sapması ölçümü hızla domine eder;")
    print(f"    dSy/dt ölçümü faz-karışmış (decohere) demetle yapılmalı —")
    print(f"    o zaman sentroid COD'ye kendiliğinden oturur ve birinci")
    print(f"    mertebe bias faz ortalamasında silinir.")

    elapsed = time.time() - t0
    print(f"\nToplam süre: {elapsed:.1f} s")
    print("\nÖZET:")
    print("  BPM ofseti/gürültüsü döngüye DOĞRUDAN girmez (girdi spin ölçümü).")
    print("  A) ölçüm gürültüsü → taban = σ_f (uzun ölçümle düşer)")
    print("  B) statik aktüasyon hatası → diferansiyel döngü kendini onarır")
    print("  C) koherent oturma hatası → en sert kanal; decoherence şart")

    # ── Figür ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5))
    fig.suptitle("BPM ofseti/gürültüsünün ölç-trimle döngüsüne etkisi (3 kanal)",
                 fontsize=13)

    ax = axes[0]
    ax.semilogy([0, 1, 2], [abs(f0), abs(fA1), abs(fA2)], 'o-',
                color='tab:blue', ms=9, lw=2)
    ax.axhline(sigma_f, color='red', ls='--', alpha=0.8,
               label=f'σ_f = {sigma_f:.1e}')
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("iterasyon adımı")
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("Kanal A: ölçüm gürültüsü\ndöngü tabanı = σ_f")
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    ax.semilogy([0, 1, 2, 3], [abs(f0), abs(fB1), abs(fB2), abs(fB3)], 's-',
                color='tab:orange', ms=9, lw=2)
    ax.axhline(abs(f0), color='gray', ls=':', alpha=0.8, label='|f0| (trimsiz)')
    ax.annotate("statik hata:\n4× kötüleşme", (1, abs(fB1)), fontsize=8,
                textcoords="offset points", xytext=(10, 0))
    ax.annotate("diferansiyel\ntoparlama", (2, abs(fB2)), fontsize=8,
                textcoords="offset points", xytext=(10, 0))
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("iterasyon adımı")
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title(f"Kanal B: statik aktüasyon hatası (b₃={B3_ERR*1e6:.0f}μm)\n"
                 "döngü kendini onarır")
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)

    ax = axes[2]
    dd_pos = [d for d in DELTA_UM if d > 0]
    ax.loglog(dd_pos, [abs(bias[d]) for d in dd_pos], 'D-',
              color='tab:red', ms=8, lw=2, label='|bias(δ)|')
    ax.axhline(abs(f0), color='gray', ls=':', label='|f0| (sinyal)')
    ax.axhline(abs(f0)/100, color='green', ls='--', label='%1 tolerans')
    ax.set_xlabel("koherent sentroid sapması δ [μm]")
    ax.set_ylabel("|bias| [rad/s]")
    ax.set_title("Kanal C: COD oturma hatası\nkoherent betatron bias'ı")
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    out = "test_b_trim_bpm.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()
