#!/usr/bin/env python3
"""
test_loco_reconstruction.py

İki optik konfigürasyondaki COD ölçümlerinden quad dy, dx ve dipol tilt
hatalarını eş zamanlı olarak geri çatar.

Sistem:
  M @ [dy; tilt] = [y_COD_1; y_COD_2]   (dikey, 96×96 LOCO)
  R_dx @ dx      = x_COD_1               (radyal, 48×48)

BPM hataları (params.json'dan):
  bpm_noise_sigma  : her ölçümde bağımsız Gaussian gürültü [m]
                     (elektronik gürültü, tur-tur değişen)
  bpm_offset_sigma : her BPM için sabit sistematik ofset [m]
                     (kalibrasyon hatası, ortalama almakla yok olmaz)

Çözüm yöntemleri:
  linalg.solve : doğrudan LU — yedek, iyi koşullu sistemler için
  SVD          : kırpmalı tekil değer ayrışımı

Ön koşul: build_response_matrix.py çalıştırılmış ve .npy dosyaları mevcut.
"""
import json
import numpy as np
import os
from build_response_matrix import setup_fields, run_sim

BASE = os.path.dirname(os.path.abspath(__file__))


def apply_bpm_errors(cod, sigma_noise, sigma_offset, offset_vec, rng):
    """COD vektörüne BPM gürültüsü ve ofseti ekler.

    sigma_noise  : her ölçümde yeniden çekilen Gaussian gürültü std [m]
    sigma_offset : sabit BPM ofseti (offset_vec ile verilir)
    offset_vec   : önceden üretilmiş sabit ofset dizisi [m]
    """
    noisy = cod.copy()
    if sigma_noise > 0:
        noisy += rng.normal(0, sigma_noise, len(cod))
    if sigma_offset > 0:
        noisy += offset_vec
    return noisy


def solve_svd(M, b, rcond=1e-6):
    """Kırpmalı SVD ile Mx = b çözer.

    rcond: en büyük tekil değere oranla eşik.
    """
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    threshold = rcond * s[0]
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)
    kept    = int(np.sum(s > threshold))
    dropped = len(s) - kept
    x = Vt.T @ (s_inv * (U.T @ b))
    return x, kept, dropped, s


def print_results(label, dy_g, dy_r, dx_g, dx_r, tilt_g, tilt_r):
    hata_dy   = dy_r   - dy_g
    hata_dx   = dx_r   - dx_g
    hata_tilt = tilt_r - tilt_g
    corr_dy   = np.corrcoef(dy_g,   dy_r)[0, 1]
    corr_dx   = np.corrcoef(dx_g,   dx_r)[0, 1]
    corr_tilt = np.corrcoef(tilt_g, tilt_r)[0, 1]
    print(f"\n--- {label} ---")
    print(f"dy   hata RMS : {np.std(hata_dy)*1e6:8.2f} μm    korelasyon: {corr_dy:.6f}")
    print(f"dx   hata RMS : {np.std(hata_dx)*1e6:8.2f} μm    korelasyon: {corr_dx:.6f}")
    print(f"tilt hata RMS : {np.std(hata_tilt)*1e6:8.2f} μrad  korelasyon: {corr_tilt:.6f}")


def main():
    os.chdir(BASE)

    with open("params.json") as f:
        config = json.load(f)

    R_dy_1   = np.load("R_dy_1.npy")
    R_dx_1   = np.load("R_dx_1.npy")
    R_tilt_1 = np.load("R_tilt_1.npy")
    R_dy_2   = np.load("R_dy_2.npy")
    R_tilt_2 = np.load("R_tilt_2.npy")
    M_loco   = np.load("M_loco.npy")

    n_q    = R_dy_1.shape[0]   # 48
    g1_nom = config.get("g1", 0.21)
    eps    = 0.02
    g1_pert = g1_nom * (1.0 + eps)

    # BPM hata parametreleri
    sigma_noise  = config.get("bpm_noise_sigma",  0.0)   # [m]
    sigma_offset = config.get("bpm_offset_sigma", 0.0)   # [m]
    offset_seed  = config.get("bpm_offset_seed",  55)

    alanlar1, state01 = setup_fields(config)
    alanlar2, state02 = setup_fields(config, g1_override=g1_pert)

    # Sabit BPM ofseti: kalibrasyon hatası, her ölçümde aynı
    rng_offset = np.random.default_rng(seed=offset_seed)
    bpm_offset_dy = rng_offset.normal(0, sigma_offset, n_q) if sigma_offset > 0 else np.zeros(n_q)
    bpm_offset_dx = rng_offset.normal(0, sigma_offset, n_q) if sigma_offset > 0 else np.zeros(n_q)

    # Simülasyon gürültüsü için ayrı RNG (tur-tur değişen)
    rng_noise = np.random.default_rng(seed=99)

    # Bilinen rastgele hatalar
    rng = np.random.default_rng(seed=13)
    dy_max   = 0.3e-3
    dx_max   = 0.3e-3
    tilt_max = 0.2e-3
    dy_gercek   = rng.uniform(-dy_max,   dy_max,   n_q)
    dx_gercek   = rng.uniform(-dx_max,   dx_max,   n_q)
    tilt_gercek = rng.uniform(-tilt_max, tilt_max, n_q)

    print("Gerçek hatalar:")
    print(f"  dy   RMS = {np.std(dy_gercek)*1e3:.3f} mm")
    print(f"  dx   RMS = {np.std(dx_gercek)*1e3:.3f} mm")
    print(f"  tilt RMS = {np.std(tilt_gercek)*1e3:.3f} mrad")

    if sigma_noise > 0 or sigma_offset > 0:
        print(f"\nBPM hataları:")
        if sigma_noise > 0:
            print(f"  Gürültü σ  = {sigma_noise*1e6:.1f} μm  (her ölçümde bağımsız)")
        if sigma_offset > 0:
            print(f"  Ofset  σ   = {sigma_offset*1e6:.1f} μm  (sabit, bilinmiyor)")

    print("\nReferans koşumları...")
    x0_1, y0_1 = run_sim(alanlar1, state01, config, np.zeros(n_q), np.zeros(n_q))
    x0_2, y0_2 = run_sim(alanlar2, state02, config, np.zeros(n_q), np.zeros(n_q))

    print("Hatalı koşumlar...")
    x_cod_1, y_cod_1 = run_sim(alanlar1, state01, config,
                                dy_gercek, dx_gercek, dipole_tilt=tilt_gercek)
    x_cod_2, y_cod_2 = run_sim(alanlar2, state02, config,
                                dy_gercek, dx_gercek, dipole_tilt=tilt_gercek)

    # Gerçek COD farkı (gürültüsüz)
    dy_cod_1_ideal = y_cod_1 - y0_1
    dy_cod_2_ideal = y_cod_2 - y0_2
    dx_cod_1_ideal = x_cod_1 - x0_1

    # BPM hataları uygulanmış ölçümler
    dy_cod_1 = apply_bpm_errors(dy_cod_1_ideal, sigma_noise, sigma_offset, bpm_offset_dy, rng_noise)
    dy_cod_2 = apply_bpm_errors(dy_cod_2_ideal, sigma_noise, sigma_offset, bpm_offset_dy, rng_noise)
    dx_cod_1 = apply_bpm_errors(dx_cod_1_ideal, sigma_noise, sigma_offset, bpm_offset_dx, rng_noise)

    print(f"\nÖlçülen COD (BPM hataları dahil):")
    print(f"  dikey [nom]  RMS = {np.std(dy_cod_1)*1e3:.3f} μm")
    print(f"  dikey [pert] RMS = {np.std(dy_cod_2)*1e3:.3f} μm")
    print(f"  radyal[nom]  RMS = {np.std(dx_cod_1)*1e3:.3f} μm")

    rhs = np.concatenate([dy_cod_1, dy_cod_2])

    # --- Yöntem 1: linalg.solve (yedek) ---
    sol_direct = np.linalg.solve(M_loco, rhs)
    dy_geri_direct   = sol_direct[:n_q]
    tilt_geri_direct = sol_direct[n_q:]
    dx_geri_direct   = np.linalg.solve(R_dx_1, dx_cod_1)

    # --- Yöntem 2: Kırpmalı SVD ---
    sol_svd, kept, dropped, sv = solve_svd(M_loco, rhs, rcond=1e-6)
    dy_geri_svd   = sol_svd[:n_q]
    tilt_geri_svd = sol_svd[n_q:]
    dx_geri_svd   = np.linalg.solve(R_dx_1, dx_cod_1)  # R_dx iyi koşullu

    print(f"\nSVD analizi: tutulan {kept}/{len(sv)}, atılan {dropped}/{len(sv)}")

    print_results("linalg.solve",
                  dy_gercek, dy_geri_direct,
                  dx_gercek, dx_geri_direct,
                  tilt_gercek, tilt_geri_direct)

    print_results("SVD (rcond=1e-6)",
                  dy_gercek, dy_geri_svd,
                  dx_gercek, dx_geri_svd,
                  tilt_gercek, tilt_geri_svd)

    np.savez("loco_reconstruction_test.npz",
             dy_gercek=dy_gercek, dy_geri=dy_geri_direct,
             dx_gercek=dx_gercek, dx_geri=dx_geri_direct,
             tilt_gercek=tilt_gercek, tilt_geri=tilt_geri_direct,
             singular_values=sv)
    print("\nSonuçlar 'loco_reconstruction_test.npz' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()
