#!/usr/bin/env python3
"""test_b_trim_bpm.py — BPM ofseti/gürültüsünün trim döngüsüne etkisi (CO=False).

Önemli gözlem:
  Trim döngüsünün girdisi spin tabanlı dSy/dt ölçümü; BPM verisi döngüde
  DOĞRUDAN kullanılmaz. Üstelik gerçekçi koşulda (eksenden fırlatma, COD'ye
  oturtma YOK) yörünge düzeltmesi diye bir adım da yoktur → BPM'lerin
  "oturtma hatası" kanalı tanım gereği ortadan kalkar. Kalan iki dolaylı
  kanal:

  Kanal A — dSy/dt ölçüm gürültüsü (polarimetre istatistiği):
    f_ölçülen = f_gerçek + ε. Sistem tam lineer olduğundan (CO=False'ta
    f/A yayılımı %0) beklenti kesindir: adım n sonrası artık = −ε_{n-1},
    yani döngü tabanı = σ_ε. Gürültü birikmez; her adım son gerçekleşmeyi
    trimler.

  Kanal B — trim aktüasyon hatası:
    Trimler quad mover yerine BPM-referanslı yörünge tümseğiyle uygulanırsa
    statik BPM ofseti σ_b tümseğin k harmonik genliğine b_k ≈ σ_b/√24 hata
    ekler (σ_b=100μm → b_3 ≈ 20μm → ilk trim büyük hata alır). AMA ofset
    statik olduğundan ARTIMSAL aktüasyonda iptal olur → döngü hatayı ölçüp
    sonraki turda temizler. Test: 1. trime kasıtlı +20μm statik hata,
    sonraki turların diferansiyel toparlaması.

Yöntem (üniform 0.2 lattice, eksenden fırlatma, t2=1ms, taze desen seed=99):
  Batch 1: f0.
  Batch 2: A-adım1 (gürültülü trim, ε0=+0.1|f0|), B-adım1 (+20μm hata).
  Batch 3: A-adım2 (ε1=−0.1|f0|), B-adım2 (diferansiyel).
  Batch 4: B-adım3 (diferansiyel).

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
PATTERN_SEED = 99
BG_RMS       = 1e-5
T2           = 1e-3
RETURN_STEPS = 6000
CK_FILE      = "test_b_ck_cofalse.json"

SIGMA_F_FRAC = 0.10     # Kanal A: ölçüm gürültüsü, |f0| oranı
B3_ERR       = 20e-6    # Kanal B: statik aktüasyon hatası [m]
                        # (σ_b=100μm BPM ofsetinin k=3 bileşeni ≈ σ_b/√24)


def _suppress_stdout():
    fd = os.dup(1); null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1); os.close(null); return fd


def _restore_stdout(fd):
    os.dup2(fd, 1); os.close(fd)


def _worker(task):
    """EKSENDEN fırlatma (CO aranmaz) → dSy/dt."""
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

    print(f"Lattice üniform {fields.quadG1} T/m — EKSENDEN fırlatma "
          f"(oturtma yok), taze desen seed={PATTERN_SEED}")
    print(f"Trim modu k=3 (c_3={c3:.3e} rad/s/m, CO=False kalibrasyonu)")

    # ── Batch 1: taban ───────────────────────────────────────────────────
    print("\nBatch 1: f0 ölçülüyor...")
    r = run([("f0", P.tolist(), T2, RETURN_STEPS)])
    f0 = r["f0"]
    sigma_f = SIGMA_F_FRAC * abs(f0)
    print(f"  f0 = {f0:+.3e} rad/s;  Kanal A gürültüsü σ_f = {sigma_f:.1e} (%10)")

    # ── Batch 2: adım 1 ──────────────────────────────────────────────────
    epsA0 = +sigma_f
    trimA1 = -(f0 + epsA0) / c3
    dyA1 = P + mode_vec(n_q, 3, trimA1, antisym)
    trimB1_req  = -f0 / c3
    trimB1_real = trimB1_req + B3_ERR
    dyB1 = P + mode_vec(n_q, 3, trimB1_real, antisym)
    print(f"\nBatch 2: A1 (trim {trimA1*1e6:+.2f}μm, gürültülü ölçümden), "
          f"B1 (trim {trimB1_req*1e6:+.2f}μm istek + {B3_ERR*1e6:.0f}μm hata)...")
    r2 = run([("A1", dyA1.tolist(), T2, RETURN_STEPS),
              ("B1", dyB1.tolist(), T2, RETURN_STEPS)])
    fA1 = r2["A1"]; fB1 = r2["B1"]
    print(f"  A1 = {fA1:+.3e}  (beklenti −ε0 = {-epsA0:+.1e})")
    print(f"  B1 = {fB1:+.3e}  (beklenti c_3·b_3 = {c3*B3_ERR:+.1e})")

    # ── Batch 3: adım 2 ──────────────────────────────────────────────────
    epsA1 = -sigma_f
    trimA2 = -(fA1 + epsA1) / c3
    dyA2 = dyA1 + mode_vec(n_q, 3, trimA2, antisym)
    trimB2 = -fB1 / c3        # diferansiyel: statik ofset artımda iptal
    dyB2 = dyB1 + mode_vec(n_q, 3, trimB2, antisym)
    print(f"\nBatch 3: A2 (trim {trimA2*1e6:+.3f}μm), "
          f"B2 (diferansiyel {trimB2*1e6:+.2f}μm)...")
    r3 = run([("A2", dyA2.tolist(), T2, RETURN_STEPS),
              ("B2", dyB2.tolist(), T2, RETURN_STEPS)])
    fA2 = r3["A2"]; fB2 = r3["B2"]
    print(f"  A2 = {fA2:+.3e}  (beklenti −ε1 = {-epsA1:+.1e})")
    print(f"  B2 = {fB2:+.3e}")

    # ── Batch 4: B adım 3 ────────────────────────────────────────────────
    trimB3 = -fB2 / c3
    dyB3 = dyB2 + mode_vec(n_q, 3, trimB3, antisym)
    print(f"\nBatch 4: B3 (diferansiyel {trimB3*1e6:+.4f}μm)...")
    r4 = run([("B3", dyB3.tolist(), T2, RETURN_STEPS)])
    fB3 = r4["B3"]
    print(f"  B3 = {fB3:+.3e}")

    # ── Sonuçlar ─────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"KANAL A: dSy/dt ölçüm gürültüsü (σ_f = %10·|f0| = {sigma_f:.1e})")
    print(f"{'═'*72}")
    print(f"  adım 0: {f0:+.3e}")
    print(f"  adım 1: {fA1:+.3e}  (ε0={epsA0:+.1e};  artık/σ_f = "
          f"{abs(fA1)/sigma_f:.3f})")
    print(f"  adım 2: {fA2:+.3e}  (ε1={epsA1:+.1e};  artık/σ_f = "
          f"{abs(fA2)/sigma_f:.3f})")
    print(f"  → döngü tabanı = σ_f: her adım SON gürültü gerçekleşmesini")
    print(f"    trimler, gürültü birikmez. Daha uzun/temiz spin ölçümü →")
    print(f"    taban doğrudan düşer.")

    print(f"\n{'═'*72}")
    print(f"KANAL B: statik aktüasyon hatası (b_3={B3_ERR*1e6:.0f}μm ≈ "
          f"σ_b=100μm'lik BPM ofsetinin k=3 bileşeni)")
    print(f"{'═'*72}")
    print(f"  adım 0: {f0:+.3e}   (trimsiz)")
    print(f"  adım 1: {fB1:+.3e}   ({abs(fB1/f0):.1f}× KÖTÜLEŞME)")
    print(f"  adım 2: {fB2:+.3e}   ({abs(f0/fB2):.0f}× bastırma)")
    print(f"  adım 3: {fB3:+.3e}   ({abs(f0/fB3):.0f}× bastırma)")
    print(f"  → statik ofset 1. trimi bozar ama döngü kendini onarır:")
    print(f"    artımsal aktüasyonda statik hata iptal olur. Maliyet: 1 ek tur.")

    elapsed = time.time() - t0
    print(f"\nToplam süre: {elapsed:.1f} s")
    print("\nÖZET:")
    print("  Gerçekçi koşulda (oturtma yok) BPM'lerin oturtma kanalı zaten yok.")
    print("  A) ölçüm gürültüsü → taban = σ_f (birikmez, uzun ölçümle düşer)")
    print("  B) statik aktüasyon hatası → diferansiyel döngü kendini onarır")
    print("  → BPM ofseti/gürültüsü trim yönteminin etkinliğini SINIRLAMAZ;")
    print("    nihai taban polarimetre (spin ölçümü) istatistiğidir.")

    # ── Figür ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("BPM ofseti/gürültüsünün ölç-trimle döngüsüne etkisi "
                 "(CO=False — oturtma yok)", fontsize=13)

    ax = axes[0]
    ax.semilogy([0, 1, 2], [abs(f0), abs(fA1), abs(fA2)], 'o-',
                color='tab:blue', ms=9, lw=2)
    ax.axhline(sigma_f, color='red', ls='--', alpha=0.8,
               label=f'σ_f = {sigma_f:.1e}')
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("iterasyon adımı")
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title("Kanal A: ölçüm gürültüsü\ndöngü tabanı = σ_f, birikme yok")
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    ax.semilogy([0, 1, 2, 3], [abs(f0), abs(fB1), abs(fB2), abs(fB3)], 's-',
                color='tab:orange', ms=9, lw=2)
    ax.axhline(abs(f0), color='gray', ls=':', alpha=0.8, label='|f0| (trimsiz)')
    ax.annotate("statik hata:\nkötüleşme", (1, abs(fB1)), fontsize=8,
                textcoords="offset points", xytext=(10, 0))
    ax.annotate("diferansiyel\ntoparlama", (2, abs(fB2)), fontsize=8,
                textcoords="offset points", xytext=(10, 0))
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("iterasyon adımı")
    ax.set_ylabel("|dSy/dt| [rad/s]")
    ax.set_title(f"Kanal B: statik aktüasyon hatası (b₃={B3_ERR*1e6:.0f}μm)\n"
                 "döngü kendini onarır")
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    out = "test_b_trim_bpm.png"
    plt.savefig(out, dpi=150)
    print(f"Figür kaydedildi: {out}")


if __name__ == "__main__":
    main()
