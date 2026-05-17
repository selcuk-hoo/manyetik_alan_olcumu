"""
mode_transfer.py — Test 2 (yapilacaklar-2.md §Test 2)

Her estimator'ın uzaysal transfer fonksiyonu: sinüsoidal misalignment
patternlerini enjekte et, geri alınan amplitüd / gerçek amplitüd oranını
Fourier mod indeksi k'nın fonksiyonu olarak ölç.

  Δq_j^(k) = A·cos(2π·k·j/N + φ),  k = 0, 1, ..., N/2

Forward: y = R·Δq  (analitik, simülasyon koşmadan — Test 1 sonucuyla
analitik R'nin yeterince doğru olduğu doğrulandı).

Estimator'lar Test 1 ile aynı: direct, ham ΔR, Tikhonov, TSVD.

Makalenin signature figure'ını üretir: estimator transfer fonksiyonları
tek bir grafikte karşılaştırmalı.

Kullanım:
    python mode_transfer.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fodo_lattice import (
    compute_twiss_at_quads, signed_KL, build_response_matrix,
    calibrate_K_x_arc, direct_invert,
)
from compare_regularization import tikhonov, tsvd

EPS = 0.02
AMPLITUDE = 100e-6   # 100 μm
TIKHONOV_LAMBDA = {'y': 3.57e-2, 'x': 4.94e-2}   # Test 1 L-curve optimumları
TSVD_K = {'y': 3, 'x': 5}                          # Test 1 oracle optimumları
BPM_NOISE = 1e-6   # 1 μm RMS BPM gürültüsü (gürültülü senaryo için)
N_REALIZATIONS = 40   # gürültü ortalamasını yumuşatmak için tekrarlar


def build_R(config, g, plane, K_x_arc=None):
    cfg = dict(config); cfg['g1'] = g
    beta, phi, Q = compute_twiss_at_quads(cfg, plane, K_x_arc=K_x_arc)
    KL = signed_KL(cfg, plane)
    return build_response_matrix(beta, phi, Q, KL)


def transfer_function(plane, config, k_range, noise_rms=0.0, n_real=1):
    """Her k için her estimator'ın transfer ratio'sunu hesapla.
    noise_rms > 0 ise BPM okumalarına Gaussian gürültü eklenir; n_real
    realization ortalaması alınır.
    """
    K_x_arc = calibrate_K_x_arc(config) if plane == 'x' else None
    g_nom = config['g1']; g_pert = g_nom * (1 + EPS)
    R1 = build_R(config, g_nom,  plane, K_x_arc)
    R2 = build_R(config, g_pert, plane, K_x_arc)
    dR = R1 - R2

    N = R1.shape[0]
    j = np.arange(N)

    lam = TIKHONOV_LAMBDA[plane]
    k_tsvd = TSVD_K[plane]

    names = ["Direct (v₁+v₂)/2", "Ham ΔR⁻¹",
             f"Tikhonov ΔR (λ={lam:.1e})", f"TSVD ΔR (k={k_tsvd})"]
    ratios = {n: [] for n in names}
    rng = np.random.default_rng(123)

    for k in k_range:
        r_dir = []; r_raw = []; r_tik = []; r_tsv = []
        # k=0 ve k=N/2 (Nyquist) Nyquist için sadece tek faz: phase=0
        # (diğer fazda mod kimliksel olarak sıfır olabilir)
        phases = [0.0] if (k == 0 or 2 * k == N) else [0.0, np.pi / 2]
        for phase in phases:
            dq_true = AMPLITUDE * np.cos(2 * np.pi * k * j / N + phase)
            if np.dot(dq_true, dq_true) < (AMPLITUDE * 1e-3) ** 2:
                continue  # güvenlik
            y1_clean = R1 @ dq_true
            y2_clean = R2 @ dq_true

            for _ in range(n_real):
                if noise_rms > 0:
                    y1 = y1_clean + rng.normal(0, noise_rms, N)
                    y2 = y2_clean + rng.normal(0, noise_rms, N)
                else:
                    y1, y2 = y1_clean, y2_clean
                dy = y1 - y2

                v_dir = 0.5 * (direct_invert(R1, y1) + direct_invert(R2, y2))
                v_raw = direct_invert(dR, dy)
                v_tik = tikhonov(dR, dy, lam)
                v_tsv = tsvd(dR, dy, k_tsvd)

                def proj(dq_hat):
                    return float(np.dot(dq_hat, dq_true)
                                 / np.dot(dq_true, dq_true))

                r_dir.append(proj(v_dir))
                r_raw.append(proj(v_raw))
                r_tik.append(proj(v_tik))
                r_tsv.append(proj(v_tsv))

        ratios[names[0]].append(np.mean(r_dir))
        ratios[names[1]].append(np.mean(r_raw))
        ratios[names[2]].append(np.mean(r_tik))
        ratios[names[3]].append(np.mean(r_tsv))

    return {n: np.array(v) for n, v in ratios.items()}


def main():
    with open("params.json", "r") as f:
        config = json.load(f)

    print("=" * 64)
    print("mode_transfer.py — Test 2 (signature figure)")
    print("=" * 64)
    print(f"Mod amplitüdü: {AMPLITUDE*1e6:.0f} μm")
    print(f"Tikhonov λ (Test 1 L-curve): {TIKHONOV_LAMBDA}")
    print(f"TSVD k (Test 1 oracle):      {TSVD_K}")

    N = 48
    k_range = np.arange(0, N // 2 + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey='row')

    scenarios = [
        ("Noiseless (analytic)",            0.0,       1),
        (f"BPM noise σ={BPM_NOISE*1e6:.0f}μm, {N_REALIZATIONS} realizations",
         BPM_NOISE, N_REALIZATIONS),
    ]

    for row, (label, noise, nr) in enumerate(scenarios):
        for col, plane in enumerate(['y', 'x']):
            ax = axes[row, col]
            print(f"\n[{plane}, {label}] hesaplanıyor...")
            ratios = transfer_function(plane, config, k_range,
                                       noise_rms=noise, n_real=nr)

            colors = {"Direct (v₁+v₂)/2": "C0", "Ham ΔR⁻¹": "C3"}
            for name, r in ratios.items():
                c = colors.get(name, None)
                ax.plot(k_range, r, '-o', markersize=4, label=name, color=c)

            ax.axhline(1.0, color='gray', lw=0.5, ls=':')
            ax.axhline(0.0, color='gray', lw=0.5, ls=':')
            ax.set_xlabel("Fourier mod indeksi k")
            ax.set_title(f"{label}, plane {plane}")
            ax.grid(True, alpha=0.3)
            if row == 1:
                ax.set_ylim(-1.5, 2.0)
            else:
                ax.set_ylim(-0.5, 2.0)
            ax.legend(fontsize=7, loc='upper right')

            print(f"  k        Direct   ΔR ham   Tikhonov  TSVD")
            for k in [0, 1, 4, 8, 12, 16, 20, 24]:
                i = list(k_range).index(k)
                vals = [ratios["Direct (v₁+v₂)/2"][i],
                        ratios["Ham ΔR⁻¹"][i],
                        ratios[f"Tikhonov ΔR (λ={TIKHONOV_LAMBDA[plane]:.1e})"][i],
                        ratios[f"TSVD ΔR (k={TSVD_K[plane]})"][i]]
                print(f"  {k:3d}    " + "  ".join(f"{v:+7.3f}" for v in vals))

    for row in range(2):
        axes[row, 0].set_ylabel("Transfer ratio (recovered / true)")
    fig.suptitle("Spatial transfer function of two-gradient estimators",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("test2_mode_transfer.png", dpi=140)
    print("\nKaydedildi: test2_mode_transfer.png")


if __name__ == "__main__":
    main()
