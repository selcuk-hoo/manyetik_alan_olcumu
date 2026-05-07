#!/usr/bin/env python3
"""
test_kmod_reconstruction.py

K-modülasyon benzeri geri çatım: referans ölçüm olmadan quad
misalignment'larını geri çat, dipol tilt'in ne kadar bozduğunu ölç.

Fikir:
  Tüm hatalar (dy, dx, tilt) aynıyken iki farklı optik konfigürasyonda
  COD ölçülür. Fark:

    Δy = y(g_pert) - y(g_nom)
       = [R_dy(g_pert) - R_dy(g_nom)] @ dy
       + [R_tilt(g_pert) - R_tilt(g_nom)] @ tilt
       = ΔR_dy @ dy  +  ΔR_tilt @ tilt
                ^              ^
            büyük          bastırılmış

  Quad katkısı g ile orantılı (ΔR_dy/R_dy ≈ δg/g ≈ %2).
  Tilt katkısı yalnızca beta fonksiyonu değişiminden gelir (ΔR_tilt << ΔR_dy).

BPM hataları:
  Ofset: her iki ölçümde de aynı BPM → Δy'de common-mode rejection ile
         tamamen iptal olur. Referans ölçümüne gerek yok.
  Gürültü: bağımsız iki realization, Δy'de √2·σ_noise olarak kalır.

Matris kaynağı (öncelik sırasıyla):
  dR_dy_kmod.npy / dR_dx_kmod.npy  — sabit dipol tilt arka planı ile inşa
  edilmişse kullanılır (params.json'da dipole_random_tilt_max > 0 olmalı).
  Yoksa R_dy_1/2.npy farkına döner (ideal — tilt arka planı yok).

Ön koşul: build_response_matrix.py çalıştırılmış olmalı.
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
    print(f"  {label:30s}  hata RMS={np.std(hata)*1e6:7.2f} μm   korelasyon={corr:.6f}")


def main():
    os.chdir(BASE)

    with open("params.json") as f:
        config = json.load(f)

    # Matris kaynağını seç: sabit tilt arka planı ile inşa edilmiş k-mod
    # matrisleri varsa onları kullan (daha gerçekçi), yoksa ideal ΔR'ye dön.
    kmod_files = ("dR_dy_kmod.npy", "dR_dx_kmod.npy")
    use_kmod_matrices = all(os.path.exists(f) for f in kmod_files)

    if use_kmod_matrices:
        dR_dy = np.load("dR_dy_kmod.npy")
        dR_dx = np.load("dR_dx_kmod.npy")
        print("Matris kaynağı: dR_dy_kmod.npy / dR_dx_kmod.npy  (sabit tilt arka planı ile)")
    else:
        R_dy_1 = np.load("R_dy_1.npy")
        R_dy_2 = np.load("R_dy_2.npy")
        R_dx_1 = np.load("R_dx_1.npy")
        R_dx_2 = np.load("R_dx_2.npy")
        dR_dy  = R_dy_2 - R_dy_1
        dR_dx  = R_dx_2 - R_dx_1
        print("Matris kaynağı: R_dy_1/2.npy farkı  (ideal — tilt arka planı yok)")

    # Tilt kirliliği hesabı için R_tilt farkı (her iki durumda mevcut matrisler)
    R_tilt_1 = np.load("R_tilt_1.npy")
    R_tilt_2 = np.load("R_tilt_2.npy")
    dR_tilt = R_tilt_2 - R_tilt_1  # küçük — yalnızca β değişiminden

    n_q    = dR_dy.shape[0]   # 48
    g1_nom = config.get("g1", 0.21)
    eps    = 0.02
    g1_pert = g1_nom * (1.0 + eps)

    cond_dR_dy = np.linalg.cond(dR_dy)
    cond_dR_dx = np.linalg.cond(dR_dx)
    print(f"ΔR_dy koşul sayısı : {cond_dR_dy:.3e}")
    print(f"ΔR_dx koşul sayısı : {cond_dR_dx:.3e}")
    print(f"δg/g               : {eps*100:.1f}%  (g_nom={g1_nom}, g_pert={g1_pert:.4f})")

    # BPM hata parametreleri
    sigma_noise  = config.get("bpm_noise_sigma",  0.0)
    sigma_offset = config.get("bpm_offset_sigma", 0.0)
    offset_seed  = config.get("bpm_offset_seed",  55)

    rng_offset = np.random.default_rng(seed=offset_seed)
    bpm_offset_dy = rng_offset.normal(0, sigma_offset, n_q) if sigma_offset > 0 else np.zeros(n_q)
    bpm_offset_dx = rng_offset.normal(0, sigma_offset, n_q) if sigma_offset > 0 else np.zeros(n_q)
    rng_noise = np.random.default_rng(seed=99)

    # Quad hataları (geri çatılacak)
    dy_max = config.get("quad_random_dy_max", 0.3e-3)
    dx_max = config.get("quad_random_dx_max", 0.3e-3)
    quad_seed = config.get("quad_random_seed", 13)
    rng = np.random.default_rng(seed=quad_seed)
    dy_gercek = rng.uniform(-dy_max, dy_max, n_q)
    dx_gercek = rng.uniform(-dx_max, dx_max, n_q)

    # Dipol tiltler — sabit, tüm ölçümlerde aynı (BPM ofseti gibi davranır)
    tilt_max  = config.get("dipole_random_tilt_max", 0.2e-3)
    tilt_seed = config.get("dipole_random_seed", 43)
    rng_tilt  = np.random.default_rng(seed=tilt_seed)
    tilt_sabit = rng_tilt.uniform(-tilt_max, tilt_max, n_q)

    print(f"\nGerçek hatalar:")
    print(f"  dy   RMS = {np.std(dy_gercek)*1e3:.3f} mm")
    print(f"  dx   RMS = {np.std(dx_gercek)*1e3:.3f} mm")
    print(f"  tilt RMS = {np.std(tilt_sabit)*1e3:.3f} mrad  (sabit, her ölçümde aynı)")
    if sigma_noise > 0:
        print(f"\nBPM gürültü σ={sigma_noise*1e6:.1f} μm, ofset σ={sigma_offset*1e6:.1f} μm")
        print(f"  Δy'de ofset: tamamen iptal (common-mode rejection)")
        print(f"  Δy'de gürültü: √2·σ = {np.sqrt(2)*sigma_noise*1e6:.1f} μm kalır")

    # Optik konfigürasyonlar
    alanlar1, state01 = setup_fields(config)
    alanlar2, state02 = setup_fields(config, g1_override=g1_pert)

    # --- İki konfigürasyonda COD ölçümü ---
    print("\nKonfigürasyon 1 (g_nom) koşumu...")
    x1, y1 = run_sim(alanlar1, state01, config, dy_gercek, dx_gercek, dipole_tilt=tilt_sabit)

    print("Konfigürasyon 2 (g_pert) koşumu...")
    x2, y2 = run_sim(alanlar2, state02, config, dy_gercek, dx_gercek, dipole_tilt=tilt_sabit)

    # BPM hatalarını ham ölçümlere uygula → farkta ofset iptal
    y1_meas = apply_bpm_errors(y1, sigma_noise, sigma_offset, bpm_offset_dy, rng_noise)
    y2_meas = apply_bpm_errors(y2, sigma_noise, sigma_offset, bpm_offset_dy, rng_noise)
    x1_meas = apply_bpm_errors(x1, sigma_noise, sigma_offset, bpm_offset_dx, rng_noise)
    x2_meas = apply_bpm_errors(x2, sigma_noise, sigma_offset, bpm_offset_dx, rng_noise)

    delta_y = y2_meas - y1_meas   # ofset iptal, quad + tilt sinyali kalır
    delta_x = x2_meas - x1_meas

    # Sinyallerin büyüklüğü
    delta_y_ideal = (y2 - y1)   # gürültüsüz
    quad_signal   = dR_dy @ dy_gercek
    tilt_contam   = dR_tilt @ tilt_sabit
    print(f"\nΔy bileşenleri (gürültüsüz):")
    print(f"  Quad sinyali  ΔR_dy·dy  RMS = {np.std(quad_signal)*1e6:.1f} μm")
    print(f"  Tilt kirliliği ΔR_tilt·θ RMS = {np.std(tilt_contam)*1e6:.1f} μm")
    print(f"  Oran (kirlilik/sinyal)       = {np.std(tilt_contam)/np.std(quad_signal)*100:.1f}%")

    # --- Geri çatım ---
    print("\nGeri çatım sonuçları:")

    # 1) Sadece quad sinyali ile (ideal — tilt sıfır olsaydı)
    dy_ideal = np.linalg.solve(dR_dy, quad_signal)
    dx_ideal = np.linalg.solve(dR_dx, dR_dx @ dx_gercek)
    print("\n[İdeal: tilt=0, gürültü=0]")
    print_results("dy", dy_gercek, dy_ideal)

    # 2) Gerçekçi: tilt var, gürültü var
    dy_geri = np.linalg.solve(dR_dy, delta_y)
    dx_geri = np.linalg.solve(dR_dx, delta_x)
    print("\n[Gerçekçi: tilt sabit, BPM hataları var]")
    print_results("dy", dy_gercek, dy_geri)
    print_results("dx", dx_gercek, dx_geri)

    # Hata kaynakları
    hata_tilt  = np.linalg.solve(dR_dy, tilt_contam)          # tilt katkısı
    hata_noise = dy_geri - np.linalg.solve(dR_dy, delta_y_ideal)  # gürültü katkısı
    print(f"\nHata kaynakları (dy):")
    print(f"  Tilt'ten   : {np.std(hata_tilt)*1e6:.2f} μm")
    print(f"  Gürültüden : {np.std(hata_noise)*1e6:.2f} μm")
    print(f"  Toplam     : {np.std(dy_geri - dy_gercek)*1e6:.2f} μm")

    np.savez("kmod_reconstruction_test.npz",
             dy_gercek=dy_gercek, dy_geri=dy_geri,
             dx_gercek=dx_gercek, dx_geri=dx_geri,
             tilt_sabit=tilt_sabit,
             quad_signal=quad_signal, tilt_contam=tilt_contam)
    print("\nSonuçlar 'kmod_reconstruction_test.npz' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()
