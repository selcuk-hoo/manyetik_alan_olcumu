"""
drift_monitor_sim.py — Test 4 (yapilacaklar-2.md §Test 4)

Kalibre-referans drift takibinin somut gösterimi.

Senaryo:
  t = 0   : kalibrasyon epoch'u
              dq_0  ~ 100 μm RMS rastgele misalignment
              b_0   ~  50 μm RMS rastgele BPM ofseti  ← BÜYÜK!
              y_0   = R·dq_0 + b_0 + η_0           (kaydedilir)
  t = 1...N : misalignment yavaş kayar, b sabit
              dq(t) = dq_0 + δq_ramp · t/N     (toplam 10 μm RMS ramp)
              y(t)  = R·dq(t) + b_0 + η(t)
              tahmin: δq_hat(t) = R⁻¹·(y(t) - y_0)

Beklenti: tahmin, gerçek δq(t)'yi BPM ofsetinden bağımsız olarak izlemeli.
b_0'ın 50 μm büyüklüğü, mutlak rekonstrüksiyonu mahveder ama drift
rekonstrüksiyonunu etkilemez.

Kullanım:
    python drift_monitor_sim.py
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

with open("params.json", "r") as _f:
    _cfg = json.load(_f)
with open("test_params.json", "r") as _f:
    _tp = json.load(_f)
_t4 = _tp["test4"]

# Başlangıç misalignment büyüklüğü — params.json'dan (y ve x ortalaması)
DQ0_RMS    = 0.5 * (float(_cfg["quad_random_dy_max"]) + float(_cfg["quad_random_dx_max"]))
BPM_OFFSET = float(_t4["BPM_OFFSET"])
BPM_NOISE  = float(_t4["BPM_NOISE"])
DRIFT_RAMP = float(_t4["DRIFT_RAMP"])
N_EPOCHS   = int(_t4["N_EPOCHS"])


def build_R(config, g, plane, K_x_arc=None):
    cfg = dict(config); cfg['g1'] = g
    beta, phi, Q = compute_twiss_at_quads(cfg, plane, K_x_arc=K_x_arc)
    KL = signed_KL(cfg, plane)
    return build_response_matrix(beta, phi, Q, KL)


def run_plane(plane, config, rng, ax_drift, ax_err):
    K_x_arc = calibrate_K_x_arc(config) if plane == 'x' else None
    R = build_R(config, config['g1'], plane, K_x_arc)
    N = R.shape[0]

    dq_0    = rng.normal(0, DQ0_RMS,    N)
    b_0     = rng.normal(0, BPM_OFFSET, N)
    dq_ramp = rng.normal(0, DRIFT_RAMP, N)   # bir yöne çekilen rastgele vektör

    y_0 = R @ dq_0 + b_0 + rng.normal(0, BPM_NOISE, N)

    times = np.arange(0, N_EPOCHS + 1)
    drift_true_rms = np.zeros_like(times, dtype=float)
    drift_hat_rms  = np.zeros_like(times, dtype=float)
    track_err_rms  = np.zeros_like(times, dtype=float)
    corr           = np.zeros_like(times, dtype=float)
    abs_err_rms    = np.zeros_like(times, dtype=float)  # eğer mutlak yapsaydık

    R_inv = np.linalg.inv(R)

    print(f"\n========== Düzlem: {plane} ==========")
    print(f"  ‖dq_0‖_RMS      = {np.std(dq_0)*1e6:7.2f} μm")
    print(f"  ‖b_0‖_RMS       = {np.std(b_0)*1e6:7.2f} μm  (BÜYÜK)")
    print(f"  ‖drift_ramp‖    = {np.std(dq_ramp)*1e6:7.2f} μm  (toplam)")
    print()
    print(f"  {'epoch':>5s}  {'δq_true':>9s}  {'δq_hat':>9s}  "
          f"{'tracking err':>13s}  {'corr':>6s}  {'abs naive':>10s}")
    print("  " + "-" * 65)

    for t in times:
        dq_t = dq_0 + dq_ramp * (t / N_EPOCHS)
        delta_q_true = dq_t - dq_0
        y_t = R @ dq_t + b_0 + rng.normal(0, BPM_NOISE, N)

        # Drift tahmini (calibrated reference)
        delta_q_hat = R_inv @ (y_t - y_0)

        # Naive absolute reconstruction (karşılaştırma)
        dq_hat_abs  = R_inv @ y_t
        abs_err_rms[t] = np.sqrt(np.mean((dq_hat_abs - dq_t) ** 2))

        drift_true_rms[t] = np.sqrt(np.mean(delta_q_true ** 2))
        drift_hat_rms[t]  = np.sqrt(np.mean(delta_q_hat ** 2))
        track_err_rms[t]  = np.sqrt(np.mean((delta_q_hat - delta_q_true) ** 2))
        if drift_true_rms[t] > 1e-12 and np.std(delta_q_hat) > 1e-12:
            corr[t] = np.corrcoef(delta_q_hat, delta_q_true)[0, 1]
        else:
            corr[t] = np.nan

        print(f"  {t:5d}  {drift_true_rms[t]*1e6:8.2f}μm  "
              f"{drift_hat_rms[t]*1e6:8.2f}μm  "
              f"{track_err_rms[t]*1e6:12.3f}μm  {corr[t]:6.3f}  "
              f"{abs_err_rms[t]*1e6:9.2f}μm")

    # Grafik 1: gerçek vs tahmin edilen drift RMS, zamanın fonksiyonu
    ax_drift.plot(times, drift_true_rms * 1e6, '-o', label="True drift", color='C0')
    ax_drift.plot(times, drift_hat_rms  * 1e6, '-s', label="Estimated drift",
                  color='C1', markersize=5)
    ax_drift.fill_between(times,
                          (drift_true_rms - BPM_NOISE) * 1e6,
                          (drift_true_rms + BPM_NOISE) * 1e6,
                          alpha=0.15, color='C0', label="True ± 1μm BPM band")
    ax_drift.set_xlabel("Epoch")
    ax_drift.set_ylabel("RMS [μm]")
    ax_drift.set_title(f"Drift tracking, plane {plane}")
    ax_drift.legend(fontsize=8); ax_drift.grid(True, alpha=0.3)

    # Grafik 2: drift tahmin hatası ve naive absolute hatası karşılaştırması
    ax_err.semilogy(times, track_err_rms * 1e6, '-o',
                    label="Drift tracking error", color='C2')
    ax_err.semilogy(times, abs_err_rms * 1e6, '-s',
                    label="Absolute reconstruction error (naive)", color='C3')
    ax_err.axhline(np.std(b_0) * 1e6, color='gray', ls='--',
                   label=f"‖b_0‖ = {np.std(b_0)*1e6:.0f} μm")
    ax_err.set_xlabel("Epoch")
    ax_err.set_ylabel("Reconstruction error RMS [μm]")
    ax_err.set_title(f"Drift vs absolute, plane {plane}")
    ax_err.legend(fontsize=7); ax_err.grid(True, alpha=0.3, which='both')

    # Özet
    print()
    print(f"  Ortalama drift tracking hatası : {np.mean(track_err_rms)*1e6:.2f} μm")
    print(f"  Ortalama mutlak rek. hatası    : {np.mean(abs_err_rms)*1e6:.2f} μm")
    print(f"  Mutlak/drift hata oranı        : {np.mean(abs_err_rms)/np.mean(track_err_rms[1:]):.0f}×")


def main():
    with open("params.json", "r") as f:
        config = json.load(f)

    print("=" * 72)
    print("drift_monitor_sim.py — Test 4")
    print("=" * 72)
    print(f"DQ0_RMS    = {DQ0_RMS*1e6:.0f} μm  (params.json quad_random_dy/dx_max ortalaması)")
    print(f"BPM_OFFSET = {BPM_OFFSET*1e6:.0f} μm  BPM_NOISE = {BPM_NOISE*1e6:.1f} μm  "
          f"DRIFT_RAMP = {DRIFT_RAMP*1e6:.0f} μm  N_EPOCHS = {N_EPOCHS}  (test_params.json)")

    rng = np.random.default_rng(2026)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    run_plane('y', config, rng, axes[0, 0], axes[0, 1])
    run_plane('x', config, rng, axes[1, 0], axes[1, 1])
    fig.tight_layout()
    fig.savefig("test4_drift_monitor.png", dpi=140)
    print("\nKaydedildi: test4_drift_monitor.png")


if __name__ == "__main__":
    main()
