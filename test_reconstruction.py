#!/usr/bin/env python3
"""
test_reconstruction.py

Bilinen rastgele quad kaçıklıkları uygular, simülasyonla COD ölçer,
R_dy / R_dx tepki matrisleriyle hataları geri çatar ve karşılaştırır.

params.json değiştirilmez.
"""
import json
import numpy as np
import os
from integrator import integrate_particle, FieldParams
from build_response_matrix import setup_fields, run_sim

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    os.chdir(BASE)

    with open("params.json") as f:
        config = json.load(f)

    alanlar, state0 = setup_fields(config)
    n_q = 2 * int(alanlar.nFODO)  # 48

    # Tepki matrislerini yükle
    R_dy = np.load("R_dy.npy")
    R_dx = np.load("R_dx.npy")

    # Bilinen rastgele hatalar (seed sabit → tekrarlanabilir)
    rng = np.random.default_rng(seed=7)
    dy_max = 0.05e-3   # 50 μm RMS civarı
    dx_max = 0.05e-3
    dy_gercek = rng.uniform(-dy_max, dy_max, n_q)
    dx_gercek = rng.uniform(-dx_max, dx_max, n_q)

    print(f"Gerçek hatalar — dy: RMS={np.std(dy_gercek)*1e3:.3f} mm, "
          f"max={np.max(np.abs(dy_gercek))*1e3:.3f} mm")
    print(f"                 dx: RMS={np.std(dx_gercek)*1e3:.3f} mm, "
          f"max={np.max(np.abs(dx_gercek))*1e3:.3f} mm")

    # Referans COD (hatasız)
    print("\nReferans koşumu...")
    x0, y0 = run_sim(alanlar, state0, config, np.zeros(n_q), np.zeros(n_q))

    # Hatalı simülasyon
    print("Hatalı koşum...")
    x_cod, y_cod = run_sim(alanlar, state0, config, dy_gercek, dx_gercek)

    # Net COD değişimi
    dx_cod = x_cod - x0
    dy_cod = y_cod - y0

    print(f"\nÖlçülen COD — x: RMS={np.std(dx_cod)*1e3:.3f} mm, "
          f"max={np.max(np.abs(dx_cod))*1e3:.3f} mm")
    print(f"              y: RMS={np.std(dy_cod)*1e3:.3f} mm, "
          f"max={np.max(np.abs(dy_cod))*1e3:.3f} mm")

    # Geri çatım: R @ hata = COD  →  hata = R^{-1} @ COD
    dy_geri = np.linalg.solve(R_dy, dx_cod)
    dx_geri = np.linalg.solve(R_dx, dy_cod)

    # Hata analizi
    hata_dy = dy_geri - dy_gercek
    hata_dx = dx_geri - dx_gercek

    print("\n--- Geri Çatım Sonuçları ---")
    print(f"dy  gerçek  RMS : {np.std(dy_gercek)*1e3:.4f} mm")
    print(f"dy  geri ç. RMS : {np.std(dy_geri)*1e3:.4f} mm")
    print(f"dy  hata    RMS : {np.std(hata_dy)*1e6:.2f} μm")
    print(f"dy  hata    max : {np.max(np.abs(hata_dy))*1e6:.2f} μm")
    print()
    print(f"dx  gerçek  RMS : {np.std(dx_gercek)*1e3:.4f} mm")
    print(f"dx  geri ç. RMS : {np.std(dx_geri)*1e3:.4f} mm")
    print(f"dx  hata    RMS : {np.std(hata_dx)*1e6:.2f} μm")
    print(f"dx  hata    max : {np.max(np.abs(hata_dx))*1e6:.2f} μm")

    # Korelasyon katsayısı
    corr_dy = np.corrcoef(dy_gercek, dy_geri)[0, 1]
    corr_dx = np.corrcoef(dx_gercek, dx_geri)[0, 1]
    print(f"\nKorelasyon — dy: {corr_dy:.6f}")
    print(f"             dx: {corr_dx:.6f}")

    np.savez("reconstruction_test.npz",
             dy_gercek=dy_gercek, dy_geri=dy_geri,
             dx_gercek=dx_gercek, dx_geri=dx_geri)
    print("\nSonuçlar 'reconstruction_test.npz' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()
