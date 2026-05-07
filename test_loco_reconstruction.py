#!/usr/bin/env python3
"""
test_loco_reconstruction.py

İki optik konfigürasyondaki COD ölçümlerinden quad dy ve dipol tilt hatalarını
eş zamanlı olarak geri çatar.

Kullanılan sistem:
  M @ [dy; tilt] = [y_COD_1; y_COD_2]

  M = [[R_dy_1, R_tilt_1],   (nominal: g1=0.21, g0=0.20)
       [R_dy_2, R_tilt_2]]   (pertürbe: g1=0.21, g0×1.10)

Boyutlar: M [96×96], [dy;tilt] [96], [y_COD_1;y_COD_2] [96]

Çözüm yöntemleri:
  linalg.solve : doğrudan LU çözümü (iyi koşullu sistemler için)
  SVD          : kırpmalı tekil değer ayrışımı (kötü koşullu sistemler için)

Ön koşul: build_response_matrix.py çalıştırılmış ve tüm .npy dosyaları mevcut.
"""
import json
import numpy as np
import os
from build_response_matrix import setup_fields, run_sim

BASE = os.path.dirname(os.path.abspath(__file__))


def solve_svd(M, b, rcond=1e-6):
    """Kırpmalı SVD ile Mx = b çözer.

    rcond: en büyük tekil değere oranla eşik (bu oranın altındakiler sıfırlanır)
    Döndürür: (çözüm, tutulan_sv_sayısı, atılan_sv_sayısı)
    """
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    threshold = rcond * s[0]
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)
    kept   = int(np.sum(s > threshold))
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
    print(f"dy   hata RMS : {np.std(hata_dy)*1e6:.2f} μm    korelasyon: {corr_dy:.6f}")
    print(f"dx   hata RMS : {np.std(hata_dx)*1e6:.2f} μm    korelasyon: {corr_dx:.6f}")
    print(f"tilt hata RMS : {np.std(hata_tilt)*1e6:.2f} μrad  korelasyon: {corr_tilt:.6f}")


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
    g0_nom = config.get("g0", g1_nom)
    eps    = 0.10
    g0_pert = g0_nom * (1.0 + eps)

    # Optik konfigürasyonları — tek quad pertürbasyonu
    alanlar1, state01 = setup_fields(config)
    alanlar2, state02 = setup_fields(config, g0_override=g0_pert)

    # Bilinen rastgele hatalar (tekrarlanabilir) — üç hata türü aynı anda
    rng = np.random.default_rng(seed=13)
    dy_max   = 0.3e-3
    dx_max   = 0.3e-3
    tilt_max = 0.2e-3
    dy_gercek   = rng.uniform(-dy_max,   dy_max,   n_q)
    dx_gercek   = rng.uniform(-dx_max,   dx_max,   n_q)
    tilt_gercek = rng.uniform(-tilt_max, tilt_max, n_q)

    print("Gerçek hatalar (üç tür aynı anda):")
    print(f"  dy   RMS = {np.std(dy_gercek)*1e3:.3f} mm,  max = {np.max(np.abs(dy_gercek))*1e3:.3f} mm")
    print(f"  dx   RMS = {np.std(dx_gercek)*1e3:.3f} mm,  max = {np.max(np.abs(dx_gercek))*1e3:.3f} mm")
    print(f"  tilt RMS = {np.std(tilt_gercek)*1e3:.3f} mrad, max = {np.max(np.abs(tilt_gercek))*1e3:.3f} mrad")

    print("\nReferans koşumları...")
    x0_1, y0_1 = run_sim(alanlar1, state01, config, np.zeros(n_q), np.zeros(n_q))
    x0_2, y0_2 = run_sim(alanlar2, state02, config, np.zeros(n_q), np.zeros(n_q))

    print("Hatalı koşumlar (dy + dx + tilt birlikte)...")
    x_cod_1, y_cod_1 = run_sim(alanlar1, state01, config,
                                dy_gercek, dx_gercek, dipole_tilt=tilt_gercek)
    x_cod_2, y_cod_2 = run_sim(alanlar2, state02, config,
                                dy_gercek, dx_gercek, dipole_tilt=tilt_gercek)

    dy_cod_1 = y_cod_1 - y0_1
    dy_cod_2 = y_cod_2 - y0_2
    dx_cod_1 = x_cod_1 - x0_1

    print(f"\nÖlçülen COD:")
    print(f"  dikey [nom]  RMS = {np.std(dy_cod_1)*1e3:.3f} μm")
    print(f"  dikey [pert] RMS = {np.std(dy_cod_2)*1e3:.3f} μm")
    print(f"  radyal[nom]  RMS = {np.std(dx_cod_1)*1e3:.3f} μm")

    rhs = np.concatenate([dy_cod_1, dy_cod_2])

    # --- Yöntem 1: linalg.solve (yedek — iyi koşullu sistemler için) ---
    # sol_direct = np.linalg.solve(M_loco, rhs)
    # dy_geri_direct   = sol_direct[:n_q]
    # tilt_geri_direct = sol_direct[n_q:]

    # --- Yöntem 2: Kırpmalı SVD ---
    print("\nTekil değer analizi...")
    sol_svd, kept, dropped, sv = solve_svd(M_loco, rhs, rcond=1e-6)
    dy_geri_svd   = sol_svd[:n_q]
    tilt_geri_svd = sol_svd[n_q:]

    print(f"  En büyük σ : {sv[0]:.3e}")
    print(f"  En küçük σ : {sv[-1]:.3e}")
    print(f"  κ = σ_max/σ_min : {sv[0]/sv[-1]:.3e}")
    print(f"  Tutulan  : {kept}/{len(sv)}")
    print(f"  Atılan   : {dropped}/{len(sv)}")
    if dropped > 0:
        print(f"  Eşik σ   : {1e-6*sv[0]:.3e}  (rcond=1e-6)")
        print(f"  Atılan modlar: σ ∈ [{sv[kept]:.3e}, {sv[-1]:.3e}]")

    # Radyal geri çatım (SVD gerekmiyor, R_dx iyi koşullu)
    dx_geri = np.linalg.solve(R_dx_1, dx_cod_1)

    print_results("SVD Geri Çatım Sonuçları",
                  dy_gercek, dy_geri_svd,
                  dx_gercek, dx_geri,
                  tilt_gercek, tilt_geri_svd)

    np.savez("loco_reconstruction_test.npz",
             dy_gercek=dy_gercek, dy_geri=dy_geri_svd,
             dx_gercek=dx_gercek, dx_geri=dx_geri,
             tilt_gercek=tilt_gercek, tilt_geri=tilt_geri_svd,
             singular_values=sv)
    print("\nSonuçlar 'loco_reconstruction_test.npz' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()
