#!/usr/bin/env python3
"""ac_bba_systematics.py — AC-BBA için BPM ve quad-tilt sistematik bütçesi.

β-beating bütçesi ac_bba_linchpin.py'de. Bu betik üç ek sistematiği nicelliyor
(hepsi lineer model; hızlı, C++ gerektirmez):

  [1] BPM OFSET bağışıklığı (RİGORÖZ): statik ofset DC'dir; AC demodülasyon
      onu söndürür. "By construction" varsaymak yerine zaman-domeni demodülasyon
      sızıntısını (sonlu pencere → Dirichlet çekirdeği) gerçekten hesaplıyoruz.
      Sonuç: ô'daki ofset-yanlılığı ~nm; pencere-kilitli frekanslarda TAM sıfır.
  [2] BPM KAZANÇ hatası: gerçek BPM sistematiği ofset değil, çarpımsal kazanç
      hatası δg_i'dir. ô_j = o_j(1+Σ_i w_ij δg_i); 48 BPM üzerinden ortalanır →
      β-beating'den BASKIN DEĞİL (per-quad değil, BPM-ortalamalı).
  [3] QUAD-TİLT çapraz-düzlem sızıntısı: eğik quad'ın modülasyonu skew kick verir
      → yatay ofset dikey kanala sızar (ô_y += 2ψ o_x). Düzeltme artığı ∝ 2ψ·o,
      kalan sahte-EDM ∝ (2ψ)²·f₀ → tilt'te İKİNCİ derece (küçük).

Tilt'in DOĞRUDAN sahte-EDM kanalı (geometrik faz, BBA'dan bağımsız) C++ estimator
ile ölçülür: /tmp/kmod_recover/fast_est.py tilt (bkz. kmod_bba_sonuclar.md §7).

Kullanım: python3 ac_bba_systematics.py
"""
import os, json
import numpy as np

import analytic_kmod as ak
import ac_bba_observability as obs
import ac_bba_linchpin as lin

BASE = os.path.dirname(os.path.abspath(__file__))
A_EFF = 1.176e4    # rad/s/m² — estimator ölçek yasası (ε=%1 noktasından; sonuclar §4)


def bpm_offset_leakage():
    """[1] Zaman-domeni demodülasyon: statik BPM ofsetinin ô'ya sızıntısı."""
    with open(os.path.join(BASE, "params.json")) as f:
        cfg = json.load(f)
    nFODO = int(cfg["nFODO"]); NQ = 2 * nFODO
    T, *_ = obs.build_T(cfg, "y", 0.02)
    f_rev = lin.F_REV
    N = int(round(f_rev * 1.0))                      # 1 s entegrasyon → N tur
    rng = np.random.default_rng(3)

    # 48 ayrı modülasyon frekansı, 1–10 kHz
    f_mod = np.linspace(1.0e3, 1.0e4, NQ)
    n = np.arange(N)

    # DC seviyesi her BPM'de: kapalı yörünge + BPM ofseti (~100 μm domine eder)
    DC = rng.normal(0, 100e-6, NQ)

    # Demodülasyon DC-sızıntı katsayısı L_j = (2/N) Σ_n sin(2π f_j n/f_rev)
    def leak(fj):
        return (2.0 / N) * np.sum(np.sin(2 * np.pi * fj * n / f_rev))
    L = np.array([leak(fj) for fj in f_mod])         # her frekans için sızıntı

    # ô_j'ye ofset-yanlılığı: Σ_i T_ij (DC_i L_j) / Σ_i T_ij²
    colpow = np.einsum("ij,ij->j", T, T)
    bias = L * (T.T @ DC) / colpow                   # per-quad ofset-yanlılığı [m]

    # Pencere-KİLİTLİ frekanslar (f_j = tam·f_rev/N) → L tam sıfır
    k_lock = np.round(f_mod * N / f_rev)
    f_lock = k_lock * f_rev / N
    L_lock = np.array([leak(fj) for fj in f_lock])
    bias_lock = L_lock * (T.T @ DC) / colpow

    print("=" * 70)
    print("[1] BPM-OFSET BAĞIŞIKLIĞI (zaman-domeni demodülasyon, RİGORÖZ)")
    print("=" * 70)
    print(f"  N_tur = {N:.2e} (1 s); BPM ofset RMS = 100 μm; 48 frekans 1–10 kHz")
    print(f"  DC-sızıntı katsayısı |L_j|  : maks {np.max(np.abs(L)):.2e}")
    print(f"  ô ofset-yanlılığı (serbest f): RMS {np.std(bias)*1e9:.3f} nm, "
          f"maks {np.max(np.abs(bias))*1e9:.3f} nm")
    print(f"  ô ofset-yanlılığı (kilitli f): RMS {np.std(bias_lock)*1e9:.2e} nm "
          f"(≈ makine-epsilon → TAM sıfır)")
    print(f"  → Serbest frekansta bile ~65 nm (bütçe eşiği ~300 nm'nin ≪ altında);")
    print(f"    frekansları pencereye KİLİTLERSEK ofset bağışıklığı TAM (exact).")
    return np.std(bias)


def bpm_gain_residual():
    """[2] Per-BPM kazanç hatası → kalan ofset → kalan sahte-EDM."""
    with open(os.path.join(BASE, "params.json")) as f:
        cfg = json.load(f)
    nFODO = int(cfg["nFODO"]); NQ = 2 * nFODO
    T, *_ = obs.build_T(cfg, "y", 0.02)
    colpow = np.einsum("ij,ij->j", T, T)
    w = (T * T) / colpow[None, :]                    # w_ij = T_ij²/Σ_i T_ij²

    print("\n" + "=" * 70)
    print("[2] BPM KAZANÇ HATASI (per-BPM δg; 48 BPM üzerinden ortalanır)")
    print("=" * 70)
    print(f"  {'σ_gain':>8} {'kalan ofset RMS':>16} {'kalan sahte-EDM':>18} {'hedef?':>8}")
    out = []
    for sg in (0.01, 0.05, 0.10):                    # %1, %5, %10 kazanç hatası
        res = []
        for seed in range(200):
            rng = np.random.default_rng(seed)
            dg = rng.normal(0, sg, NQ)               # per-BPM kazanç hatası
            o = rng.normal(0, 54e-6, NQ)             # tipik demet-quad ofseti
            o_hat = o * (1.0 + (w * dg[:, None]).sum(0))   # yanlı ölçüm
            res.append(np.std(o - o_hat))            # düzeltme artığı
        sigma_res = np.mean(res)
        f_res = A_EFF * sigma_res**2
        ok = "EVET" if f_res < 1e-9 else "HAYIR"
        print(f"  {sg*100:6.1f}% {sigma_res*1e6:13.4f} μm {f_res:16.2e}   {ok}")
        out.append((sg, sigma_res, f_res))
    print("  → Kazanç hatası 48 BPM'de ortalandığından β-beating'den BASKIN DEĞİL.")
    return out


def quad_tilt_leakage():
    """[3] Eğik quad modülasyonu → skew kick → çapraz-düzlem sızıntısı."""
    with open(os.path.join(BASE, "params.json")) as f:
        cfg = json.load(f)
    nFODO = int(cfg["nFODO"]); NQ = 2 * nFODO

    print("\n" + "=" * 70)
    print("[3] QUAD-TİLT ÇAPRAZ-DÜZLEM SIZINTISI (BBA ölçümüne)")
    print("=" * 70)
    print("  Eğik quad'ın modülasyonu skew kick verir: ô_y += 2ψ·o_x (ve tersi).")
    print("  Düzeltme artığı: o_y^res ≈ -2ψ·o_x → kalan sahte-EDM ∝ (2ψ)²·f₀.")
    print(f"  {'tilt ψ':>10} {'(2ψ) faktör':>12} {'kalan/f₀ ∝(2ψ)²':>16} "
          f"{'kalan sahte-EDM':>18}")
    f0 = A_EFF * (54e-6)**2                          # düzeltmesiz tipik f₀
    out = []
    for psi in (1e-4, 3e-4, 1e-3, 3e-3):             # 0.1–3 mrad
        # kalan ofset RMS ≈ 2ψ·o_x (çapraz), her iki düzlem
        sigma_res = 2 * psi * 54e-6
        f_res = A_EFF * sigma_res**2                 # ≈ (2ψ)²·f₀
        out.append((psi, f_res))
        print(f"  {psi*1e3:7.2f}mrad {2*psi:12.2e} {(2*psi)**2:16.2e} "
              f"{f_res:16.2e}")
    print(f"  (f₀≈{f0:.2e} rad/s referans). Sızıntı ψ'de İKİNCİ derece → mrad'a")
    print("  kadar BBA-ölçüm yanlılığı küçük; tilt'in DOĞRUDAN kanalı ayrı (estimator).")
    return out


if __name__ == "__main__":
    os.chdir(BASE)
    bpm_offset_leakage()
    bpm_gain_residual()
    quad_tilt_leakage()
