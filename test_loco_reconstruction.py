#!/usr/bin/env python3
"""
test_loco_reconstruction.py

İki optik konfigürasyondaki COD ölçümlerinden quad dy ve dipol tilt hatalarını
eş zamanlı olarak geri çatar.

Kullanılan sistem:
  M @ [dy; tilt] = [y_COD_1; y_COD_2]

  M = [[R_dy_1, R_tilt_1],   (nominal g1)
       [R_dy_2, R_tilt_2]]   (pertürbe g1)

Boyutlar: M [96×96], [dy;tilt] [96], [y_COD_1;y_COD_2] [96]

Ön koşul: build_response_matrix.py çalıştırılmış ve tüm .npy dosyaları mevcut.
"""
import json
import numpy as np
import os
from build_response_matrix import setup_fields, run_sim

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    os.chdir(BASE)

    with open("params.json") as f:
        config = json.load(f)

    # Matrisleri yükle
    R_dy_1   = np.load("R_dy_1.npy")
    R_dx_1   = np.load("R_dx_1.npy")
    R_tilt_1 = np.load("R_tilt_1.npy")
    R_dy_2   = np.load("R_dy_2.npy")
    R_tilt_2 = np.load("R_tilt_2.npy")
    M_loco   = np.load("M_loco.npy")

    n_q = R_dy_1.shape[0]   # 48
    eps = 0.02
    g1_nom  = config.get("g1", 0.21)
    g1_pert = g1_nom * (1.0 + eps)

    # Optik konfigürasyonları
    alanlar1, state01 = setup_fields(config, g1_override=g1_nom)
    alanlar2, state02 = setup_fields(config, g1_override=g1_pert)

    # Bilinen rastgele hatalar (tekrarlanabilir)
    rng = np.random.default_rng(seed=13)
    dy_max   = 0.3e-3   # ±0.3 mm quad dikey kaçıklık
    tilt_max = 0.2e-3   # ±0.2 mrad dipol tilt
    dy_gercek   = rng.uniform(-dy_max,   dy_max,   n_q)
    tilt_gercek = rng.uniform(-tilt_max, tilt_max, n_q)

    print(f"Gerçek hatalar:")
    print(f"  dy   RMS = {np.std(dy_gercek)*1e3:.3f} mm,  max = {np.max(np.abs(dy_gercek))*1e3:.3f} mm")
    print(f"  tilt RMS = {np.std(tilt_gercek)*1e3:.3f} mrad, max = {np.max(np.abs(tilt_gercek))*1e3:.3f} mrad")

    # Referans COD (sıfır hata, her iki konfigürasyon)
    print("\nReferans koşumları...")
    x0_1, y0_1 = run_sim(alanlar1, state01, config, np.zeros(n_q), np.zeros(n_q))
    x0_2, y0_2 = run_sim(alanlar2, state02, config, np.zeros(n_q), np.zeros(n_q))

    # Hatalı simülasyon (quad dy + dipol tilt)
    print("Hatalı koşumlar...")
    _, y_cod_1 = run_sim(alanlar1, state01, config,
                         dy_gercek, np.zeros(n_q), dipole_tilt=tilt_gercek)
    _, y_cod_2 = run_sim(alanlar2, state02, config,
                         dy_gercek, np.zeros(n_q), dipole_tilt=tilt_gercek)

    dy_cod_1 = y_cod_1 - y0_1
    dy_cod_2 = y_cod_2 - y0_2

    print(f"\nÖlçülen COD (dikey):")
    print(f"  [nom]  RMS = {np.std(dy_cod_1)*1e3:.3f} μm, max = {np.max(np.abs(dy_cod_1))*1e3:.3f} μm")
    print(f"  [pert] RMS = {np.std(dy_cod_2)*1e3:.3f} μm, max = {np.max(np.abs(dy_cod_2))*1e3:.3f} μm")

    # LOCO geri çatım: M @ [dy; tilt] = [y_COD_1; y_COD_2]
    rhs = np.concatenate([dy_cod_1, dy_cod_2])
    sol = np.linalg.solve(M_loco, rhs)

    dy_geri   = sol[:n_q]
    tilt_geri = sol[n_q:]

    hata_dy   = dy_geri   - dy_gercek
    hata_tilt = tilt_geri - tilt_gercek

    print("\n--- LOCO Geri Çatım Sonuçları ---")
    print(f"dy   gerçek  RMS : {np.std(dy_gercek)*1e3:.4f} mm")
    print(f"dy   geri ç. RMS : {np.std(dy_geri)*1e3:.4f} mm")
    print(f"dy   hata    RMS : {np.std(hata_dy)*1e6:.2f} μm")
    print(f"dy   hata    max : {np.max(np.abs(hata_dy))*1e6:.2f} μm")
    print()
    print(f"tilt gerçek  RMS : {np.std(tilt_gercek)*1e6:.4f} μrad")
    print(f"tilt geri ç. RMS : {np.std(tilt_geri)*1e6:.4f} μrad")
    print(f"tilt hata    RMS : {np.std(hata_tilt)*1e6:.2f} μrad")
    print(f"tilt hata    max : {np.max(np.abs(hata_tilt))*1e6:.2f} μrad")

    corr_dy   = np.corrcoef(dy_gercek, dy_geri)[0, 1]
    corr_tilt = np.corrcoef(tilt_gercek, tilt_geri)[0, 1]
    print(f"\nKorelasyon — dy  : {corr_dy:.6f}")
    print(f"             tilt: {corr_tilt:.6f}")

    # Ayrıca quad-only dx geri çatımı (radyal, tilt etkilemez)
    print("\n--- Radyal Quad (dx) Geri Çatımı ---")
    dx_gercek = rng.uniform(-dy_max, dy_max, n_q)
    x_cod_1, _ = run_sim(alanlar1, state01, config,
                          np.zeros(n_q), dx_gercek)
    dx_cod_1 = x_cod_1 - x0_1
    dx_geri = np.linalg.solve(R_dx_1, dx_cod_1)
    hata_dx = dx_geri - dx_gercek
    corr_dx = np.corrcoef(dx_gercek, dx_geri)[0, 1]
    print(f"dx   hata    RMS : {np.std(hata_dx)*1e6:.2f} μm")
    print(f"Korelasyon — dx  : {corr_dx:.6f}")

    np.savez("loco_reconstruction_test.npz",
             dy_gercek=dy_gercek, dy_geri=dy_geri,
             tilt_gercek=tilt_gercek, tilt_geri=tilt_geri,
             dx_gercek=dx_gercek, dx_geri=dx_geri)
    print("\nSonuçlar 'loco_reconstruction_test.npz' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()
