"""
bpm_offset_drift_sim.py — Test 5 (yapilacaklar-2.md §Test 5, genişletilmiş)

İki tamamlayıcı drift-tracking estimator'ının BPM ofset kaymasına karşı
robustluk karşılaştırması.

  Estimator A: Calibrated-reference direct
        δq_hat(t) = R⁻¹·(y(t) - y_0)
      • Tek-epoch'ta hızlı ve hassas
      • BPM ofseti sabit kaldığı sürece bağışık
      • BPM ofseti kayarsa bozulur

  Estimator B: Per-epoch ΔR with time averaging
        δq_hat_T(t) = (1/T) · Σ_{τ=t-T+1}^{t} ΔR⁻¹·(y₁(τ) - y₂(τ))
      • Her epoch'ta ofseti iptal eder (k-modülasyon)
      • Tek-epoch'ta çok gürültülü (κ(ΔR) ≈ 27000)
      • T epoch ortalaması ile gürültü √T kat azalır
      • BPM ofset kaymasından BAĞIMSIZ

Senaryo:
  • Misalignment drift: küçük ramp (yön belirli)
  • BPM ofset drift hızı σ_b_dot ∈ [0, 10] μm/epoch — taranır
  • Her σ_b_dot için iki estimator'ın RMS hatası, N_epoch ortalama
    sonrası karşılaştırılır

Beklenti: yavaş BPM kaymasında A kazanır, hızlı kaymada B kazanır.
Geçiş noktası operasyonel karar noktasıdır.

Kullanım:
    python bpm_offset_drift_sim.py
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

EPS         = 0.02
DQ0_RMS     = 100e-6   # başlangıç misalignment
DRIFT_RAMP  = 10e-6    # toplam misalignment drift
BPM_OFFSET  = 50e-6    # başlangıç BPM ofset RMS
BPM_NOISE   = 1e-6     # her okumadaki BPM gürültüsü
N_EPOCHS    = 60       # gözlem süresi
AVG_WINDOW  = 30       # B estimator'ında ortalama penceresi

# BPM ofset drift hızı taraması [μm/epoch RMS]
DRIFT_RATES = np.array([0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]) * 1e-6


def build_R(config, g, plane, K_x_arc=None):
    cfg = dict(config); cfg['g1'] = g
    beta, phi, Q = compute_twiss_at_quads(cfg, plane, K_x_arc=K_x_arc)
    KL = signed_KL(cfg, plane)
    return build_response_matrix(beta, phi, Q, KL)


def run_scenario(plane, config, rng, sigma_b_dot):
    """Bir bpm-drift hızı için A ve B estimator'larının ortalama RMS hatası."""
    K_x_arc = calibrate_K_x_arc(config) if plane == 'x' else None
    R1 = build_R(config, config['g1'],            plane, K_x_arc)
    R2 = build_R(config, config['g1'] * (1+EPS),  plane, K_x_arc)
    dR = R1 - R2
    N = R1.shape[0]

    R1_inv = np.linalg.inv(R1)
    dR_inv = np.linalg.inv(dR)

    # Başlangıç durumu
    dq_0    = rng.normal(0, DQ0_RMS,    N)
    b_0     = rng.normal(0, BPM_OFFSET, N)
    dq_ramp = rng.normal(0, DRIFT_RAMP, N)
    b_drift_dir = rng.normal(0, 1.0, N); b_drift_dir /= np.linalg.norm(b_drift_dir) / np.sqrt(N)
    # ↑ birim-RMS yön vektörü; her epoch'ta sigma_b_dot · b_drift_dir eklenir

    # Kalibrasyon (sadece A için)
    y1_0 = R1 @ dq_0 + b_0 + rng.normal(0, BPM_NOISE, N)
    # B için kalibrasyon gerekmiyor (her epoch'ta kendiliğinden iptal)

    # Drift takibi
    err_A = []
    delta_B_history = []   # her epoch'taki tek-epoch ΔR tahmini

    for t in range(1, N_EPOCHS + 1):
        dq_t = dq_0 + dq_ramp * (t / N_EPOCHS)
        b_t  = b_0  + b_drift_dir * sigma_b_dot * t
        y1_t = R1 @ dq_t + b_t + rng.normal(0, BPM_NOISE, N)
        y2_t = R2 @ dq_t + b_t + rng.normal(0, BPM_NOISE, N)

        true_delta = dq_t - dq_0

        # A: calibrated direct
        delta_A = R1_inv @ (y1_t - y1_0)
        err_A.append(np.sqrt(np.mean((delta_A - true_delta) ** 2)))

        # B: per-epoch ΔR (her epoch'ta TAM dq_t tahmini, ofset-bağımsız)
        full_B = dR_inv @ (y1_t - y2_t)  # dq_t için ofset-iptalli tahmin
        delta_B_history.append((full_B, dq_t))

    # B estimator'ı için zaman-ortalama
    # δq_B_T(t) = full_B(t) - <full_B over baseline window>
    # Burada baseline: ilk AVG_WINDOW epoch'un ortalaması, drift düşük olduğunda
    # Tracking için: full_B(t)'nin running average'ı ile son AVG_WINDOW epoch
    base_avg = np.mean([h[0] for h in delta_B_history[:AVG_WINDOW]], axis=0)
    dq_baseline_t = np.mean([h[1] for h in delta_B_history[:AVG_WINDOW]], axis=0)

    err_B = []
    for t in range(1, N_EPOCHS + 1):
        # Son AVG_WINDOW epoch'un ortalaması (running window)
        lo = max(0, t - AVG_WINDOW)
        window = delta_B_history[lo:t]
        avg_estimate = np.mean([h[0] for h in window], axis=0)
        avg_true     = np.mean([h[1] for h in window], axis=0)

        # B'nin drift tahmini: window ortalama - baseline
        delta_B = avg_estimate - base_avg
        true_drift_avg = avg_true - dq_baseline_t
        err_B.append(np.sqrt(np.mean((delta_B - true_drift_avg) ** 2)))

    err_A = np.array(err_A); err_B = np.array(err_B)
    # Geçici (transient) ilk AVG_WINDOW epoch'u atla, kararlı bölgenin
    # ortalama hatasını al
    return np.mean(err_A[AVG_WINDOW:]), np.mean(err_B[AVG_WINDOW:])


def main():
    with open("params.json", "r") as f:
        config = json.load(f)

    print("=" * 72)
    print("bpm_offset_drift_sim.py — Test 5")
    print("=" * 72)
    print(f"Pencere: {N_EPOCHS} epoch, ortalama penceresi: {AVG_WINDOW} epoch")
    print(f"BPM gürültüsü: {BPM_NOISE*1e6:.1f} μm/epoch")
    print(f"Misalignment drift toplam: {DRIFT_RAMP*1e6:.0f} μm")

    rng = np.random.default_rng(2027)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, plane in zip(axes, ['y', 'x']):
        print(f"\n========== Düzlem: {plane} ==========")
        print(f"  {'σ_b_drift [μm/ep]':>20s}  "
              f"{'A: calibrated direct':>22s}  "
              f"{'B: ΔR avg':>15s}")
        print("  " + "-" * 60)

        A_errs = []; B_errs = []
        for rate in DRIFT_RATES:
            # Her rate için yeni rng (deterministik karşılaştırma)
            sub_rng = np.random.default_rng(int(2027 + rate * 1e9))
            errA, errB = run_scenario(plane, config, sub_rng, rate)
            A_errs.append(errA); B_errs.append(errB)
            print(f"  {rate*1e6:>20.3f}  {errA*1e6:>20.3f}μm  "
                  f"{errB*1e6:>13.3f}μm")
        A_errs = np.array(A_errs); B_errs = np.array(B_errs)

        ax.semilogy(DRIFT_RATES * 1e6, A_errs * 1e6, '-o',
                    label="A: Calibrated direct  R⁻¹(y(t)-y₀)",
                    color='C0')
        ax.semilogy(DRIFT_RATES * 1e6, B_errs * 1e6, '-s',
                    label=f"B: ΔR avg over {AVG_WINDOW} epochs",
                    color='C2')
        # Geçiş noktası
        if np.any(A_errs > B_errs):
            i_cross = int(np.argmax(A_errs > B_errs))
            x_cross = DRIFT_RATES[i_cross] * 1e6
            ax.axvline(x_cross, color='gray', ls='--', alpha=0.6,
                       label=f"Geçiş ≈ {x_cross:.2f} μm/epoch")
        ax.axhline(5.0, color='red', ls=':', alpha=0.5,
                   label="5 μm hedef eşiği")
        ax.set_xlabel("BPM ofset drift hızı σ_ḃ [μm/epoch]")
        ax.set_ylabel("Drift tahmin RMS hatası [μm]")
        ax.set_title(f"Düzlem: {plane}")
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=8)

    fig.suptitle("Estimator robustness vs BPM offset drift", fontsize=12)
    fig.tight_layout()
    fig.savefig("test5_bpm_offset_drift.png", dpi=140)
    print("\nKaydedildi: test5_bpm_offset_drift.png")


if __name__ == "__main__":
    main()
