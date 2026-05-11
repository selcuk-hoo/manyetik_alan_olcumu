#!/usr/bin/env python3
"""
test_kmod_reconstruction.py

K-modülasyon ile quad hizalama hatası (dy, dx) geri çatımı.

Yöntem:
  İki farklı gradyan konfigürasyonunda (g_nom, g_pert = g_nom × 1.02)
  kapalı yörünge sapması (COD) ölçülür ve fark alınır:

    Δy = y(g_pert) - y(g_nom) = ΔR_dy @ dy + [dipol tilt katkısı]

  Model yalnızca ΔR_dy kullanır. Dipol tilt modelde yoktur — ölçümde
  gürültü olarak görünür ve geri çatım hatasını büyütür.

BPM hataları:
  Ofset: her iki ölçümde aynı → Δy farkında otomatik iptal (common-mode).
  Gürültü: bağımsız iki realizasyon → Δy'de √2·σ olarak kalır.

Ön koşul: build_response_matrix.py çalıştırılmış olmalı
          (R_dy_1.npy, R_dy_2.npy, R_dx_1.npy, R_dx_2.npy üretir).
"""
import json
import numpy as np
import os
from build_response_matrix import setup_fields, run_sim

BASE = os.path.dirname(os.path.abspath(__file__))


def apply_bpm_errors(arr, sigma_noise, sigma_offset, offset_vec, rng):
    noisy = arr.copy()
    if sigma_noise > 0:
        noisy += rng.normal(0, sigma_noise, len(arr))
    if sigma_offset > 0:
        noisy += offset_vec
    return noisy


def print_results(label, gercek, geri):
    hata = geri - gercek
    corr = np.corrcoef(gercek, geri)[0, 1]
    print(f"  {label:30s}  hata RMS={np.std(hata)*1e6:7.2f} um   korelasyon={corr:.6f}")


def main():
    os.chdir(BASE)

    with open("params.json") as f:
        config = json.load(f)

    R_dy_1 = np.load("R_dy_1.npy")
    R_dy_2 = np.load("R_dy_2.npy")
    R_dx_1 = np.load("R_dx_1.npy")
    R_dx_2 = np.load("R_dx_2.npy")

    dR_dy = R_dy_2 - R_dy_1
    dR_dx = R_dx_2 - R_dx_1

    n_q     = R_dy_1.shape[0]
    g1_nom  = config.get("g1", 0.21)
    eps     = 0.02
    g1_pert = g1_nom * (1.0 + eps)

    print(f"dR_dy kosul sayisi : {np.linalg.cond(dR_dy):.3e}")
    print(f"dR_dx kosul sayisi : {np.linalg.cond(dR_dx):.3e}")
    print(f"dg/g               : {eps*100:.1f}%  (g_nom={g1_nom}, g_pert={g1_pert:.4f})")

    # BPM hata parametreleri
    sigma_noise  = config.get("bpm_noise_sigma",  0.0)
    sigma_offset = config.get("bpm_offset_sigma", 0.0)
    offset_seed  = config.get("bpm_offset_seed",  55)

    rng_offset = np.random.default_rng(seed=offset_seed)
    bpm_offset_dy = rng_offset.normal(0, sigma_offset, n_q) if sigma_offset > 0 else np.zeros(n_q)
    bpm_offset_dx = rng_offset.normal(0, sigma_offset, n_q) if sigma_offset > 0 else np.zeros(n_q)
    rng_noise = np.random.default_rng(seed=99)

    # Quad hizalama hataları (geri çatılacak)
    dy_max    = config.get("quad_random_dy_max", 0.3e-3)
    dx_max    = config.get("quad_random_dx_max", 0.3e-3)
    quad_seed = config.get("quad_random_seed", 13)
    rng = np.random.default_rng(seed=quad_seed)
    dy_gercek = rng.uniform(-dy_max, dy_max, n_q)
    dx_gercek = rng.uniform(-dx_max, dx_max, n_q)

    # Dipol tiltler — modelde yok, her iki ölçümde aynı (common-mode)
    tilt_max  = config.get("dipole_random_tilt_max", 0.2e-3)
    tilt_seed = config.get("dipole_random_seed", 43)
    rng_tilt  = np.random.default_rng(seed=tilt_seed)
    tilt_sabit = rng_tilt.uniform(-tilt_max, tilt_max, n_q)

    print(f"\nGercek hatalar:")
    print(f"  dy   RMS = {np.std(dy_gercek)*1e3:.3f} mm")
    print(f"  dx   RMS = {np.std(dx_gercek)*1e3:.3f} mm")
    print(f"  tilt RMS = {np.std(tilt_sabit)*1e3:.3f} mrad  (gorulmez — modelde yok)")
    if sigma_noise > 0 or sigma_offset > 0:
        print(f"  BPM gurultu s={sigma_noise*1e6:.1f} um, ofset s={sigma_offset*1e6:.1f} um")
        print(f"  -> Ofset, delta_y farki ile otomatik iptal olur")

    # İki konfigürasyonda simülasyon — tilt fiziksel olarak var ama modelde yok
    alanlar1, state01 = setup_fields(config)
    alanlar2, state02 = setup_fields(config, g1_override=g1_pert)

    print("\nKonfigurasyon 1 (g_nom) kosumu...")
    x1, y1 = run_sim(alanlar1, state01, config, dy_gercek, dx_gercek,
                     dipole_tilt=tilt_sabit)
    print("Konfigurasyon 2 (g_pert) kosumu...")
    x2, y2 = run_sim(alanlar2, state02, config, dy_gercek, dx_gercek,
                     dipole_tilt=tilt_sabit)

    # BPM hatalarını uygula — farkta ofset iptal olur
    y1_meas = apply_bpm_errors(y1, sigma_noise, sigma_offset, bpm_offset_dy, rng_noise)
    y2_meas = apply_bpm_errors(y2, sigma_noise, sigma_offset, bpm_offset_dy, rng_noise)
    x1_meas = apply_bpm_errors(x1, sigma_noise, sigma_offset, bpm_offset_dx, rng_noise)
    x2_meas = apply_bpm_errors(x2, sigma_noise, sigma_offset, bpm_offset_dx, rng_noise)

    delta_y = y2_meas - y1_meas
    delta_x = x2_meas - x1_meas

    # Sinyal büyüklükleri — ΔR_dy·dy ideal sinyali, gerçek Δy buna tilt+gürültü eklenmiş
    quad_signal_y = dR_dy @ dy_gercek
    quad_signal_x = dR_dx @ dx_gercek
    print(f"\nSinyal ve kirlilik:")
    print(f"  quad_signal_y (tiltsiz beklenti) RMS = {np.std(quad_signal_y)*1e6:.1f} um")
    print(f"  gercek delta_y                   RMS = {np.std(delta_y)*1e6:.1f} um")
    print(f"  fark (tilt + gurultu)            RMS = {np.std(delta_y - quad_signal_y)*1e6:.1f} um")

    # Geri çatım
    print("\n" + "=" * 60)
    print("Geri catim sonuclari")
    print("=" * 60)

    dy_geri = np.linalg.solve(dR_dy, delta_y)
    dx_geri = np.linalg.solve(dR_dx, delta_x)
    print_results("\n  dy", dy_gercek, dy_geri)
    print_results(  "  dx", dx_gercek, dx_geri)

    np.savez("kmod_reconstruction_test.npz",
             dy_gercek=dy_gercek, dy_geri=dy_geri,
             dx_gercek=dx_gercek, dx_geri=dx_geri,
             tilt_sabit=tilt_sabit,
             quad_signal_y=quad_signal_y, quad_signal_x=quad_signal_x,
             delta_y=delta_y, delta_x=delta_x)
    print("\nSonuclar 'kmod_reconstruction_test.npz' dosyasina kaydedildi.")


if __name__ == "__main__":
    main()
