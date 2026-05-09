#!/usr/bin/env python3
"""
build_response_matrix.py

İki aşamalı tepki matrisi hesabı:

Aşama 1 — Quad-only (tek optik konfigürasyon):
  R_dy [48×48] : quad_dy (dikey kaçıklık)  → y_COD (quad girişlerinde) [m/m]
  R_dx [48×48] : quad_dx (radyal kaçıklık) → x_COD (quad girişlerinde) [m/m]

Aşama 2 — LOCO benzeri (iki optik konfigürasyon, dipol tilt de dahil):
  R_tilt [48×48] : dipol_tilt (s-ekseni etrafında açı) → y_COD [m/rad]

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

Düzeltmeler (önceki sürüme göre):
  • dx ve dy AYRI koşumlarla pertürbe ediliyor (önceki sürüm aynı koşumda
    ikisini birden yapıyor ve düzlemler arası kuplaj sızıntısına açıktı).
  • Tüm pertürbasyon koşumları artık ProcessPoolExecutor ile paralel çalışıyor.
    Her worker süreci kendi geçici çalışma dizinine chdir eder; böylece
    C++ entegratörünün yazdığı `cod_data.txt` dosyaları çakışmaz.

Komut satırı:
  python build_response_matrix.py --workers 7

Kaydedilen dosyalar:
  R_dy.npy, R_dx.npy                         — nominal, quad-only (mevcut format)
  R_dy_1.npy, R_dx_1.npy, R_tilt_1.npy      — nominal konfigürasyon
  R_dy_2.npy, R_dx_2.npy, R_tilt_2.npy      — pertürbe konfigürasyon
  M_loco.npy                                  — 96×96 birleşik LOCO matrisi
"""
import argparse
import atexit
import json
import numpy as np
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from integrator import integrate_particle, FieldParams

BASE = os.path.dirname(os.path.abspath(__file__))


# ── Fiziksel kurulum ──────────────────────────────────────────────────────
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

    NOT: Dosya yolu **CWD'ye** görelidir (BASE değil). Bu, paralelleştirmede
    her worker'ın kendi geçici dizininde çalışmasını mümkün kılar.

    Sıralama: [QF_0, QD_0, QF_1, QD_1, ..., QF_23, QD_23]
    """
    cd = np.loadtxt("cod_data.txt", skiprows=1)
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

    Çıktı dosyaları (`cod_data.txt`, `rf.txt`) her zaman **CWD**'ye yazılır;
    paralel modda her worker süreci kendi geçici dizininde olduğundan dosyalar
    çakışmaz.
    """
    for fname in ("cod_data.txt", "rf.txt"):
        if os.path.exists(fname):
            os.remove(fname)
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


# ── Paralelleştirme yardımcıları ──────────────────────────────────────────
def _worker_init():
    """Her worker süreci kendi geçici dizinine chdir eder.

    delta_q    : quad kaçıklık pertürbasyonu [m]  (varsayılan 0.1 mm)
    delta_tilt : dipol tilt pertürbasyonu    [rad] (varsayılan 0.1 mrad)
    label      : ilerleme çıktısı için etiket
    sigma_noise: her ölçüme eklenen Gaussian gürültü std [m]
                 (gerçek deneyde ortalama alınsa da tamamen bastırılamaz)
                 BPM ofseti farkta iptal olduğundan burada uygulanmaz.
    rng        : numpy RNG nesnesi (sigma_noise > 0 ise gerekli)
    """
    tmp = tempfile.mkdtemp(prefix=f"kmod_w{os.getpid()}_")
    os.chdir(tmp)
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)

    def add_noise(arr):
        if sigma_noise > 0 and rng is not None:
            return arr + rng.normal(0, sigma_noise, n_q)
        return arr

    # Referans COD
    t0 = time.time()
    if label:
        print(f"  [{label}] Referans koşumu...")
    x0, y0 = run_sim(alanlar, state0, config, np.zeros(n_q), np.zeros(n_q))
    x0, y0 = add_noise(x0), add_noise(y0)
    if label:
        print(f"  [{label}] Referans tamamlandı ({time.time()-t0:.1f}s). "
              f"x_max={np.max(np.abs(x0))*1e3:.2f} μm, "
              f"y_max={np.max(np.abs(y0))*1e3:.2f} μm")


def build_matrices(config, g1_override=None, g0_override=None,
                   delta_q=1e-4, delta_tilt=1e-4, label="",
                   sigma_noise=0.0, noise_seed=77,
                   background_tilt=None, n_workers=1):
    """Verilen optik konfigürasyon için R_dy, R_dx ve R_tilt matrislerini hesaplar.

    delta_q        : quad kaçıklık pertürbasyonu [m]  (varsayılan 0.1 mm)
    delta_tilt     : dipol tilt pertürbasyonu    [rad] (varsayılan 0.1 mrad)
    label          : ilerleme çıktısı için etiket
    sigma_noise    : R inşası sırasında BPM'lere eklenen Gaussian gürültü std [m]
                     (gerçek deneyde ortalama alınsa da tamamen bastırılamaz)
    noise_seed     : gürültü RNG tohumu
    background_tilt: sabit dipol tilt arka planı [rad dizisi].
                     None → sıfır tilt (mevcut davranış).
                     Verildiğinde referans ve tüm pertürbasyon koşumları
                     bu tilt ile yapılır. Sütun farkında iptal olan yalnızca
                     tilt'in sıfırıncı mertebe katkısı (R_tilt @ bg_tilt)'tır;
                     birinci mertebe etki (∂R_dy/∂tilt * bg_tilt) hesaplanan
                     R_dy sütunlarına yansır.
    n_workers      : paralel worker süreci sayısı (1 → seri).
    """
    n_q = 2 * int(config.get("nFODO", 24))
    bg_tilt = (background_tilt if background_tilt is not None
               else np.zeros(n_q))
    zeros = np.zeros(n_q)

    # Görev listesi: 1 referans + n_q dy + n_q dx + n_q tilt = 3*n_q + 1
    # ÖNEMLİ: dy ve dx artık AYRI koşumlarda pertürbe ediliyor (önceki sürüm
    # aynı koşumda ikisini birden yapıyor → düzlemler arası kuplaj sızıntısı).
    tasks = []
    tasks.append((config, g1_override, g0_override,
                  'ref', 0, zeros, zeros, bg_tilt))
    for i in range(n_q):
        dy = np.zeros(n_q); dy[i] = delta_q
        dx = np.zeros(n_q); dx[i] = delta_q
        x_cod, y_cod = run_sim(alanlar, state0, config, dy, dx)
        R_dy[:, i] = (add_noise(y_cod) - y0) / delta_q
        R_dx[:, i] = (add_noise(x_cod) - x0) / delta_q
        if label and (i + 1) % 8 == 0:
            el = time.time() - t_start
            rem = el / (i + 1) * (n_q - i - 1)
            print(f"  [{label}] quad {i+1}/{n_q}  ({el:.0f}s geçti, ~{rem:.0f}s kaldı)")

    # Dipol tilt matrisi
    R_tilt = np.zeros((n_q, n_q))
    if label:
        print(f"  [{label}] {n_total} koşum, n_workers={n_workers}")
    t0 = time.time()

    results = {}
    progress_step = max(1, n_total // 10)

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_worker_init) as pool:
            futures = [pool.submit(_run_one, t) for t in tasks]
            for j, fut in enumerate(as_completed(futures), 1):
                kind, idx, x_cod, y_cod = fut.result()
                results[(kind, idx)] = (x_cod, y_cod)
                if label and (j % progress_step == 0 or j == n_total):
                    el = time.time() - t0
                    rem = el / j * (n_total - j)
                    print(f"  [{label}] {j}/{n_total}  "
                          f"({el:.0f}s geçti, ~{rem:.0f}s kaldı)")
    else:
        for j, t in enumerate(tasks, 1):
            kind, idx, x_cod, y_cod = _run_one(t)
            results[(kind, idx)] = (x_cod, y_cod)
            if label and (j % progress_step == 0 or j == n_total):
                el = time.time() - t0
                rem = el / j * (n_total - j)
                print(f"  [{label}] {j}/{n_total}  "
                      f"({el:.0f}s geçti, ~{rem:.0f}s kaldı)")

    # Referans ve gürültü uygulaması (deterministik, tek thread'de)
    x0_clean, y0_clean = results[('ref', 0)]
    rng = np.random.default_rng(noise_seed) if sigma_noise > 0 else None

    def _noisy(arr):
        return arr + rng.normal(0, sigma_noise, len(arr)) if rng is not None else arr

    x0 = _noisy(x0_clean)
    y0 = _noisy(y0_clean)

    R_dy   = np.zeros((n_q, n_q))
    R_dx   = np.zeros((n_q, n_q))
    R_tilt = np.zeros((n_q, n_q))

    for i in range(n_q):
        tilt = np.zeros(n_q); tilt[i] = delta_tilt
        _, y_cod = run_sim(alanlar, state0, config,
                           np.zeros(n_q), np.zeros(n_q), dipole_tilt=tilt)
        R_tilt[:, i] = (add_noise(y_cod) - y0) / delta_tilt
        if label and (i + 1) % 8 == 0:
            el = time.time() - t_start
            rem = el / (i + 1) * (n_q - i - 1)
            print(f"  [{label}] dipol {i+1}/{n_q}  ({el:.0f}s geçti, ~{rem:.0f}s kaldı)")

    if label:
        print(f"  [{label}] tamamlandı ({time.time()-t0:.1f}s)")
    return R_dy, R_dx, R_tilt


def main():
    parser = argparse.ArgumentParser(
        description="Tepki matrisi inşası — paralel."
    )
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Paralel worker süreç sayısı (öneri: çekirdek-1)")
    args = parser.parse_args()

    os.chdir(BASE)
    with open("params.json") as f:
        config = json.load(f)

    n_q      = 2 * int(config.get("nFODO", 24))   # 48
    delta_q  = 1e-4   # 0.1 mm quad kaçıklık pertürbasyonu
    delta_t  = 1e-4   # 0.1 mrad dipol tilt pertürbasyonu
    g1_nom   = config.get("g1", 0.21)
    eps      = 0.02   # %2 global optik pertürbasyon
    g1_pert  = g1_nom * (1.0 + eps)

    sigma_noise = config.get("bpm_noise_sigma", 0.0)

    print("=" * 60)
    print(f"Konfigürasyon 1: nominal optik  (n_workers={args.workers})")
    print("=" * 60)
    print(f"  n_quad = {n_q},  δ_q = {delta_q*1e3:.2f} mm,  "
          f"δ_tilt = {delta_t*1e3:.2f} mrad")
    print(f"  g1 = {g1_nom} T/m")
    if sigma_noise > 0:
        print(f"  BPM gürültüsü (R inşasında): σ = {sigma_noise*1e6:.1f} μm")
    print()

    t_total = time.time()

    R_dy_1, R_dx_1, R_tilt_1 = build_matrices(
        config, g1_override=g1_nom,
        delta_q=delta_q, delta_tilt=delta_t, label="nom",
        sigma_noise=sigma_noise, n_workers=args.workers,
    )

    np.save("R_dy.npy",     R_dy_1)   # geriye dönük uyumluluk
    np.save("R_dx.npy",     R_dx_1)
    np.save("R_dy_1.npy",   R_dy_1)
    np.save("R_dx_1.npy",   R_dx_1)
    np.save("R_tilt_1.npy", R_tilt_1)

    print(f"\n  [nom] κ(R_dy)   = {np.linalg.cond(R_dy_1):.3e}")
    print(f"  [nom] κ(R_dx)   = {np.linalg.cond(R_dx_1):.3e}")
    print(f"  [nom] κ(R_tilt) = {np.linalg.cond(R_tilt_1):.3e}")

    print()
    print("=" * 60)
    print("Konfigürasyon 2: global optik pertürbasyon (LOCO için)")
    print("=" * 60)
    print(f"  g1 = {g1_pert:.4f} T/m  (tüm quadlar  →  %{eps*100:.0f} değişim)")
    print()

    R_dy_2, R_dx_2, R_tilt_2 = build_matrices(
        config, g1_override=g1_pert,
        delta_q=delta_q, delta_tilt=delta_t, label="pert",
        sigma_noise=sigma_noise, n_workers=args.workers,
    )

    np.save("R_dy_2.npy",   R_dy_2)
    np.save("R_dx_2.npy",   R_dx_2)
    np.save("R_tilt_2.npy", R_tilt_2)

    print(f"\n  [pert] κ(R_dy)   = {np.linalg.cond(R_dy_2):.3e}")
    print(f"  [pert] κ(R_dx)   = {np.linalg.cond(R_dx_2):.3e}")
    print(f"  [pert] κ(R_tilt) = {np.linalg.cond(R_tilt_2):.3e}")

    # LOCO birleşik matrisi: dikey düzlem
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
        print("  → Koşul sayısı makul: dy ve tilt eş zamanlı geri çatılabilir.")
    elif cond_loco < 1e10:
        print("  → Koşul sayısı yüksek: SVD/Tikhonov regularizasyonu önerilir.")
    else:
        print("  UYARI: Çok yüksek koşul sayısı — global pertürbasyon yetersiz.")

    # ── K-modülasyon matrisleri: sabit dipol tilt arka planı ile ──────────────
    tilt_max  = config.get("dipole_random_tilt_max", 0.0)
    tilt_seed = config.get("dipole_random_seed", 43)

    if tilt_max > 0:
        print()
        print("=" * 60)
        print("K-modülasyon matrisleri: sabit dipol tilt arka planı ile")
        print("=" * 60)
        rng_tilt = np.random.default_rng(seed=tilt_seed)
        bg_tilt  = rng_tilt.uniform(-tilt_max, tilt_max, n_q)
        print(f"  tilt_max = {tilt_max*1e3:.3f} mrad,  "
              f"RMS = {np.std(bg_tilt)*1e3:.3f} mrad,  seed={tilt_seed}")
        print()

        print("Konfigürasyon 1 (kmod, g_nom + sabit tilt)...")
        R_dy_1t, R_dx_1t, _ = build_matrices(
            config, g1_override=g1_nom,
            delta_q=delta_q, delta_tilt=delta_t, label="kmod-nom",
            sigma_noise=sigma_noise, background_tilt=bg_tilt,
            n_workers=args.workers,
        )

        print("\nKonfigürasyon 2 (kmod, g_pert + sabit tilt)...")
        R_dy_2t, R_dx_2t, _ = build_matrices(
            config, g1_override=g1_pert,
            delta_q=delta_q, delta_tilt=delta_t, label="kmod-pert",
            sigma_noise=sigma_noise, background_tilt=bg_tilt,
            n_workers=args.workers,
        )

        dR_dy_kmod = R_dy_2t - R_dy_1t
        dR_dx_kmod = R_dx_2t - R_dx_1t
        np.save("dR_dy_kmod.npy", dR_dy_kmod)
        np.save("dR_dx_kmod.npy", dR_dx_kmod)
        np.save("bg_tilt_kmod.npy", bg_tilt)

        print(f"\n  κ(ΔR_dy_kmod) = {np.linalg.cond(dR_dy_kmod):.3e}")
        print(f"  κ(ΔR_dx_kmod) = {np.linalg.cond(dR_dx_kmod):.3e}")
        print("  Kaydedildi: dR_dy_kmod.npy, dR_dx_kmod.npy, bg_tilt_kmod.npy")
    else:
        print("\n  dipole_random_tilt_max = 0 → K-modülasyon matrisleri atlandı.")

    total_elapsed = time.time() - t_total
    print(f"\nToplam süre: {total_elapsed:.0f}s  (n_workers={args.workers})")


if __name__ == "__main__":
    main()
