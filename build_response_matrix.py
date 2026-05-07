#!/usr/bin/env python3
"""
build_response_matrix.py

İki aşamalı tepki matrisi hesabı:

Aşama 1 — Quad-only (tek optik konfigürasyon):
  R_dy [48×48] : quad_dy (dikey kaçıklık)  → y_COD (quad girişlerinde) [mm/m]
  R_dx [48×48] : quad_dx (radyal kaçıklık) → x_COD (quad girişlerinde) [mm/m]

Aşama 2 — LOCO benzeri (iki optik konfigürasyon, dipol tilt de dahil):
  R_tilt [48×48] : dipol_tilt (s-ekseni etrafında açı) → y_COD [mm/rad]

  Dikey düzlem karışıklığı:
    y_COD = R_dy @ dy + R_tilt @ tilt  (tek ölçüm → 96 bilinmeyene 48 denklem: belirsiz)

  Çözüm — iki konfigürasyon:
    nominal (g1):       y_COD_1 = R_dy_1 @ dy + R_tilt_1 @ tilt
    pertürbe (g1×ε):    y_COD_2 = R_dy_2 @ dy + R_tilt_2 @ tilt

  Birleşik 96×96 sistem:
    M @ [dy; tilt] = [y_COD_1; y_COD_2]
    M = [[R_dy_1, R_tilt_1],
         [R_dy_2, R_tilt_2]]

  κ(M) küçük olduğunda dy ve tilt eş zamanlı geri çatılabilir.

Kaydedilen dosyalar:
  R_dy.npy, R_dx.npy                         — nominal, quad-only (mevcut format)
  R_dy_1.npy, R_dx_1.npy, R_tilt_1.npy      — nominal konfigürasyon
  R_dy_2.npy, R_dx_2.npy, R_tilt_2.npy      — pertürbe konfigürasyon
  M_loco.npy                                  — 96×96 birleşik LOCO matrisi
"""
import json
import numpy as np
import os
import time
from integrator import integrate_particle, FieldParams

BASE = os.path.dirname(os.path.abspath(__file__))


def setup_fields(config, g1_override=None, g0_override=None):
    """FieldParams ve başlangıç koşullarını oluşturur.

    g1_override: tüm quadların gradyanı (None → config["g1"])
    g0_override: yalnızca 1. FODO'nun 1. QF gradyanı (None → config["g0"])
    """
    M2  = 0.938272046      # proton kütlesi [GeV/c²]
    AMU = 1.792847356      # G = (g-2)/2
    C   = 299792458.0
    M1  = 1.672621777e-27  # proton kütlesi [kg]

    p_magic = M2 / np.sqrt(AMU)
    beta0   = p_magic / np.sqrt(p_magic**2 + M2**2)
    gamma0  = 1.0 / np.sqrt(1.0 - beta0**2)
    R0      = config["R0"]
    E0_V_m  = -(p_magic * beta0 / R0) * 1e9

    direction = float(config.get("direction", -1))
    p_mag = gamma0 * M1 * C * beta0

    g1 = g1_override if g1_override is not None else config.get("g1", 0.0)

    alanlar = FieldParams()
    alanlar.R0          = R0
    alanlar.E0          = E0_V_m
    alanlar.E0_power    = config.get("E0_power", 1.0)
    alanlar.B0ver       = config.get("B0ver", 0.0)
    alanlar.B0rad       = config.get("B0rad", 0.0)
    alanlar.B0long      = config.get("B0long", 0.0)
    g0 = g0_override if g0_override is not None else config.get("g0", g1)
    alanlar.quadG1      = g1
    alanlar.quadG0      = g0
    alanlar.sextK1      = config.get("sextK1", 0.0)
    alanlar.quadSwitch  = float(config.get("quadSwitch", 1))
    alanlar.sextSwitch  = float(config.get("sextSwitch", 0))
    alanlar.EDMSwitch   = 0.0
    alanlar.direction   = direction
    alanlar.nFODO       = float(config.get("nFODO", 24))
    alanlar.quadLen     = float(config.get("quadLen", 0.4))
    alanlar.driftLen    = float(config.get("driftLen", 2.0))
    alanlar.poincare_quad_index = 999.0  # Poincaré kapalı
    alanlar.rfSwitch    = 0.0
    alanlar.rfVoltage   = 0.0
    alanlar.h           = float(config.get("h", 1.0))
    alanlar.quadModA    = 0.0
    alanlar.quadModF    = 0.0

    state0 = [
        0.0, 0.0, 0.0,
        0.0, 0.0, p_mag * direction,
        0.0, 0.0, direction,
    ]
    return alanlar, state0


def read_cod_quads(nFODO):
    """QF ve QD giriş noktalarında x ve y COD değerlerini döndürür.
    Sıralama: [QF_0, QD_0, QF_1, QD_1, ..., QF_23, QD_23]
    """
    cd = np.loadtxt(os.path.join(BASE, "cod_data.txt"), skiprows=1)
    n = int(nFODO)
    x_bpm = np.empty(2 * n)
    y_bpm = np.empty(2 * n)
    for k in range(n):
        qf = k * 8 + 2
        qd = k * 8 + 6
        x_bpm[2*k]     = cd[qf, 1]
        y_bpm[2*k]     = cd[qf, 2]
        x_bpm[2*k + 1] = cd[qd, 1]
        y_bpm[2*k + 1] = cd[qd, 2]
    return x_bpm, y_bpm


def run_sim(alanlar, state0, config, quad_dy, quad_dx, dipole_tilt=None):
    """Tek bir parçacık koşumu yapar ve BPM konumlarında COD değerlerini döndürür.

    dipole_tilt: None → sıfır tilt (mevcut davranış)
    """
    for fname in ("cod_data.txt", "rf.txt"):
        p = os.path.join(BASE, fname)
        if os.path.exists(p):
            os.remove(p)
    n_q = 2 * int(alanlar.nFODO)
    if dipole_tilt is None:
        dipole_tilt = np.zeros(n_q)
    integrate_particle(
        state0,
        t0=0.0,
        t_end=config.get("t2", 1e-3),
        h=config.get("dt", 1e-11),
        fields=alanlar,
        return_steps=10,
        quad_dy=quad_dy,
        quad_dx=quad_dx,
        dipole_tilt=dipole_tilt,
    )
    return read_cod_quads(int(alanlar.nFODO))


def build_matrices(alanlar, state0, config, delta_q=1e-4, delta_tilt=1e-4, label=""):
    """Verilen optik konfigürasyon için R_dy, R_dx ve R_tilt matrislerini hesaplar.

    delta_q    : quad kaçıklık pertürbasyonu [m]  (varsayılan 0.1 mm)
    delta_tilt : dipol tilt pertürbasyonu    [rad] (varsayılan 0.1 mrad)
    label      : ilerleme çıktısı için etiket

    Matrisler gürültüsüz hesaplanır — gerçek deneyde tepki matrisi ölçümleri
    defalarca tekrarlanıp ortalandığından gürültü bastırılır.
    """
    n_q = 2 * int(alanlar.nFODO)

    # Referans COD
    t0 = time.time()
    if label:
        print(f"  [{label}] Referans koşumu...")
    x0, y0 = run_sim(alanlar, state0, config, np.zeros(n_q), np.zeros(n_q))
    if label:
        print(f"  [{label}] Referans tamamlandı ({time.time()-t0:.1f}s). "
              f"x_max={np.max(np.abs(x0))*1e3:.2f} μm, "
              f"y_max={np.max(np.abs(y0))*1e3:.2f} μm")

    # Quad matrisleri: dy ve dx aynı anda pertürbe edilir (düzlemler ayrışık)
    R_dy = np.zeros((n_q, n_q))
    R_dx = np.zeros((n_q, n_q))
    if label:
        print(f"  [{label}] R_dy ve R_dx ({n_q}×{n_q})...")
    t_start = time.time()
    for i in range(n_q):
        dy = np.zeros(n_q); dy[i] = delta_q
        dx = np.zeros(n_q); dx[i] = delta_q
        x_cod, y_cod = run_sim(alanlar, state0, config, dy, dx)
        R_dy[:, i] = (y_cod - y0) / delta_q
        R_dx[:, i] = (x_cod - x0) / delta_q
        if label and (i + 1) % 8 == 0:
            el = time.time() - t_start
            rem = el / (i + 1) * (n_q - i - 1)
            print(f"  [{label}] quad {i+1}/{n_q}  ({el:.0f}s geçti, ~{rem:.0f}s kaldı)")

    # Dipol tilt matrisi
    R_tilt = np.zeros((n_q, n_q))
    if label:
        print(f"  [{label}] R_tilt ({n_q}×{n_q})...")
    t_start = time.time()
    for i in range(n_q):
        tilt = np.zeros(n_q); tilt[i] = delta_tilt
        _, y_cod = run_sim(alanlar, state0, config,
                           np.zeros(n_q), np.zeros(n_q), dipole_tilt=tilt)
        R_tilt[:, i] = (y_cod - y0) / delta_tilt
        if label and (i + 1) % 8 == 0:
            el = time.time() - t_start
            rem = el / (i + 1) * (n_q - i - 1)
            print(f"  [{label}] dipol {i+1}/{n_q}  ({el:.0f}s geçti, ~{rem:.0f}s kaldı)")

    return R_dy, R_dx, R_tilt


def main():
    os.chdir(BASE)
    with open("params.json") as f:
        config = json.load(f)

    n_q      = 2 * int(config.get("nFODO", 24))   # 48
    delta_q  = 1e-4   # 0.1 mm quad kaçıklık pertürbasyonu
    delta_t  = 1e-4   # 0.1 mrad dipol tilt pertürbasyonu
    g1_nom   = config.get("g1", 0.21)
    eps      = 0.02   # %2 global optik pertürbasyon
    g1_pert  = g1_nom * (1.0 + eps)

    print("=" * 60)
    print("Konfigürasyon 1: nominal optik")
    print("=" * 60)
    print(f"  n_quad = {n_q},  δ_q = {delta_q*1e3:.2f} mm,  δ_tilt = {delta_t*1e3:.2f} mrad")
    print(f"  g1 = {g1_nom} T/m")
    print()

    alanlar1, state01 = setup_fields(config)
    t_total = time.time()

    R_dy_1, R_dx_1, R_tilt_1 = build_matrices(
        alanlar1, state01, config,
        delta_q=delta_q, delta_tilt=delta_t, label="nom"
    )

    np.save("R_dy.npy",    R_dy_1)   # geriye dönük uyumluluk
    np.save("R_dx.npy",    R_dx_1)
    np.save("R_dy_1.npy",  R_dy_1)
    np.save("R_dx_1.npy",  R_dx_1)
    np.save("R_tilt_1.npy", R_tilt_1)

    cond_dy1   = np.linalg.cond(R_dy_1)
    cond_dx1   = np.linalg.cond(R_dx_1)
    cond_tilt1 = np.linalg.cond(R_tilt_1)
    print(f"\n  [nom] R_dy  koşul sayısı: {cond_dy1:.3e}")
    print(f"  [nom] R_dx  koşul sayısı: {cond_dx1:.3e}")
    print(f"  [nom] R_tilt koşul sayısı: {cond_tilt1:.3e}")

    print()
    print("=" * 60)
    print("Konfigürasyon 2: global optik pertürbasyon (LOCO için)")
    print("=" * 60)
    print(f"  g1 = {g1_pert:.4f} T/m  (tüm quadlar  →  %{eps*100:.0f} değişim)")
    print()

    alanlar2, state02 = setup_fields(config, g1_override=g1_pert)

    R_dy_2, R_dx_2, R_tilt_2 = build_matrices(
        alanlar2, state02, config,
        delta_q=delta_q, delta_tilt=delta_t, label="pert"
    )

    np.save("R_dy_2.npy",   R_dy_2)
    np.save("R_dx_2.npy",   R_dx_2)
    np.save("R_tilt_2.npy", R_tilt_2)

    cond_dy2   = np.linalg.cond(R_dy_2)
    cond_dx2   = np.linalg.cond(R_dx_2)
    cond_tilt2 = np.linalg.cond(R_tilt_2)
    print(f"\n  [pert] R_dy  koşul sayısı: {cond_dy2:.3e}")
    print(f"  [pert] R_dx  koşul sayısı: {cond_dx2:.3e}")
    print(f"  [pert] R_tilt koşul sayısı: {cond_tilt2:.3e}")

    # LOCO birleşik matrisi: dikey düzlem
    #   M @ [dy; tilt] = [y_COD_1; y_COD_2]
    M_loco = np.block([[R_dy_1, R_tilt_1],
                        [R_dy_2, R_tilt_2]])
    np.save("M_loco.npy", M_loco)

    cond_loco = np.linalg.cond(M_loco)

    print()
    print("=" * 60)
    print("LOCO birleşik matrisi özeti")
    print("=" * 60)
    print(f"  M_loco boyutu : {M_loco.shape[0]}×{M_loco.shape[1]}")
    print(f"  κ(M_loco)     : {cond_loco:.3e}")

    if cond_loco < 1e6:
        print("  → Koşul sayısı makul: dy ve dipol tilt eş zamanlı geri çatılabilir.")
    elif cond_loco < 1e10:
        print("  → Koşul sayısı yüksek: SVD/Tikhonov regularizasyonu önerilir.")
    else:
        print("  UYARI: Çok yüksek koşul sayısı — tek quad pertürbasyonu yetersiz.")
        print(f"  İpucu: eps={eps} yerine daha büyük bir değer veya tüm quadları değiştirmeyi deneyin.")

    total_elapsed = time.time() - t_total
    print(f"\nToplam süre: {total_elapsed:.0f}s")
    print("Kaydedilen dosyalar:")
    print("  R_dy.npy, R_dx.npy            (nominal, geriye dönük uyumluluk)")
    print("  R_dy_1.npy, R_dx_1.npy, R_tilt_1.npy  (nominal konfigürasyon)")
    print("  R_dy_2.npy, R_dx_2.npy, R_tilt_2.npy  (pertürbe konfigürasyon)")
    print("  M_loco.npy                     (96×96 birleşik LOCO matrisi)")


if __name__ == "__main__":
    main()
