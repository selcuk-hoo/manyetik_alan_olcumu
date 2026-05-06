#!/usr/bin/env python3
"""
build_response_matrix.py

Quad kaçıklıklarının tepki matrisini hesaplar:
  R_dy [48×48] : quad_dy (dikey kaçıklık) → y_COD (quad girişlerinde)
  R_dx [48×48] : quad_dx (radyal kaçıklık) → x_COD (quad girişlerinde)

Her sütun: tek bir quada δ=0.1 mm kaçıklık → tüm BPM'lerde ölçülen COD / δ
Toplam koşum sayısı: 1 referans + 48 dy + 48 dx = 97
"""
import json
import numpy as np
import os
import time
from integrator import integrate_particle, FieldParams

BASE = os.path.dirname(os.path.abspath(__file__))


def setup_fields(config):
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

    alanlar = FieldParams()
    alanlar.R0          = R0
    alanlar.E0          = E0_V_m
    alanlar.E0_power    = config.get("E0_power", 1.0)
    alanlar.B0ver       = config.get("B0ver", 0.0)
    alanlar.B0rad       = config.get("B0rad", 0.0)
    alanlar.B0long      = config.get("B0long", 0.0)
    alanlar.quadG1      = config.get("g1", 0.0)
    alanlar.quadG0      = config.get("g0", alanlar.quadG1)
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

    # Nominal kapalı yörünge başlangıç koşulları (sıfır sapma)
    state0 = [
        0.0, 0.0, 0.0,                   # x, y_dikey, z_boylamsal [m]
        0.0, 0.0, p_mag * direction,      # px, py, pz [kg·m/s]
        0.0, 0.0, direction,              # sx, sy, sz
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
        qf = k * 8 + 2  # QF giriş satırı (elem=2)
        qd = k * 8 + 6  # QD giriş satırı (elem=6)
        x_bpm[2*k]     = cd[qf, 1]  # x [mm]
        y_bpm[2*k]     = cd[qf, 2]  # y [mm]
        x_bpm[2*k + 1] = cd[qd, 1]
        y_bpm[2*k + 1] = cd[qd, 2]
    return x_bpm, y_bpm


def run_sim(alanlar, state0, config, quad_dy, quad_dx):
    for fname in ("cod_data.txt", "rf.txt"):
        p = os.path.join(BASE, fname)
        if os.path.exists(p):
            os.remove(p)
    n_q = 2 * int(alanlar.nFODO)
    integrate_particle(
        state0,
        t0=0.0,
        t_end=config.get("t2", 1e-3),
        h=config.get("dt", 1e-11),
        fields=alanlar,
        return_steps=10,
        quad_dy=quad_dy,
        quad_dx=quad_dx,
        dipole_tilt=np.zeros(n_q),
    )
    return read_cod_quads(int(alanlar.nFODO))


def main():
    os.chdir(BASE)
    with open("params.json") as f:
        config = json.load(f)

    alanlar, state0 = setup_fields(config)
    n_q   = 2 * int(alanlar.nFODO)  # 48
    delta = 1e-4                      # 0.1 mm pertürbasyon

    print(f"Tepki matrisi: {n_q} quad, δ = {delta*1e3:.2f} mm")
    print(f"Toplam koşum: {1 + n_q} (dy ve dx aynı anda uygulanıyor)")
    print()

    # Referans COD (hatasız)
    print("Referans koşumu...")
    t0 = time.time()
    x0, y0 = run_sim(alanlar, state0, config, np.zeros(n_q), np.zeros(n_q))
    print(f"  Tamamlandı ({time.time()-t0:.1f}s). "
          f"x_max={np.max(np.abs(x0))*1e3:.2f} μm, "
          f"y_max={np.max(np.abs(y0))*1e3:.2f} μm")

    # dy ve dx aynı anda uygulanır: düzlemler ayrıştığından
    #   x_COD değişimi → yalnızca dy'den gelir → R_dy sütunu
    #   y_COD değişimi → yalnızca dx'den gelir → R_dx sütunu
    R_dy = np.zeros((n_q, n_q))
    R_dx = np.zeros((n_q, n_q))
    print(f"\nR_dy ve R_dx ({n_q}×{n_q}) — birleşik koşumlar:")
    t_start = time.time()
    for i in range(n_q):
        dy = np.zeros(n_q); dy[i] = delta
        dx = np.zeros(n_q); dx[i] = delta
        x_cod, y_cod = run_sim(alanlar, state0, config, dy, dx)
        R_dy[:, i] = (y_cod - y0) / delta  # dy → y_COD [mm/m]
        R_dx[:, i] = (x_cod - x0) / delta  # dx → x_COD [mm/m]
        if (i + 1) % 8 == 0:
            elapsed = time.time() - t_start
            remaining = elapsed / (i + 1) * (n_q - i - 1)
            print(f"  {i+1}/{n_q}  ({elapsed:.0f}s geçti, ~{remaining:.0f}s kaldı)")

    np.save("R_dy.npy", R_dy)
    np.save("R_dx.npy", R_dx)

    cond_dy = np.linalg.cond(R_dy)
    cond_dx = np.linalg.cond(R_dx)
    print(f"\nKaydedildi: R_dy.npy, R_dx.npy")
    print(f"R_dy koşul sayısı: {cond_dy:.3e}")
    print(f"R_dx koşul sayısı: {cond_dx:.3e}")

    if cond_dy > 1e8 or cond_dx > 1e8:
        print("UYARI: Yüksek koşul sayısı — inversiyon gürültüye duyarlı olabilir.")
    else:
        print("Koşul sayıları makul — doğrudan inversiyon veya SVD uygulanabilir.")


if __name__ == "__main__":
    main()
