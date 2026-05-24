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
import argparse
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


def tsvd_solve(A, b, tau_rel):
    """Truncated SVD: σ_i < tau_rel * σ_max olan modlar kesilir.
    Geriye çözüm + tutulan mod sayısı dönülür."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > tau_rel * s[0]
    s_inv = np.where(keep, 1.0 / s, 0.0)
    x = Vt.T @ (s_inv * (U.T @ b))
    return x, int(keep.sum()), s


def fourier_basis(n_q, N=None, k_list=None):
    """Fourier baz matrisi. İki mod:
      N       : k=1..N harmonikleri + DC  → boyut (n_q, 2N+1)
      k_list  : yalnızca belirtilen harmonikler, DC yok → boyut (n_q, 2·len(k_list))
    """
    j = np.arange(n_q)
    if k_list is not None:
        cols = []
        for k in k_list:
            cols.append(np.cos(2 * np.pi * k * j / n_q))
            cols.append(np.sin(2 * np.pi * k * j / n_q))
        return np.column_stack(cols)
    cols = [np.ones(n_q)]
    for k in range(1, N + 1):
        cols.append(np.cos(2 * np.pi * k * j / n_q))
        cols.append(np.sin(2 * np.pi * k * j / n_q))
    return np.column_stack(cols)


def _fit_fourier(dR, delta, dy_gercek, F):
    """ΔR·F üzerinden LSQ çöz, kappa/model/ölçüm hatası ve korelasyon döndür."""
    M = dR @ F
    a, _, _, sv = np.linalg.lstsq(M, delta, rcond=None)
    dy_geri = F @ a
    kappa = sv[0] / sv[-1] if len(sv) > 1 and sv[-1] > 0 else (sv[0] if len(sv) == 1 else np.inf)
    a_ref, _, _, _ = np.linalg.lstsq(F, dy_gercek, rcond=None)
    model_err = np.std(dy_gercek - F @ a_ref) * 1e6
    olcum_err = np.std(dy_geri - dy_gercek) * 1e6
    corr = np.corrcoef(dy_gercek, dy_geri)[0, 1]
    return dy_geri, kappa, model_err, olcum_err, corr


def fourier_reconstruct(dR, delta, dy_gercek, n_q, N_list):
    """k=1..N harmonikleri + DC taraması."""
    results = {}
    print(f"\n{'N':>3}  {'baz':>6}  {'κ(ΔR·F)':>10}  {'model RMS':>11}  {'ölçüm RMS':>11}  {'korelasyon':>11}")
    print("-" * 60)
    for N in N_list:
        F = fourier_basis(n_q, N=N)
        dy_geri, kappa, model_err, olcum_err, corr = _fit_fourier(dR, delta, dy_gercek, F)
        print(f"  {N:1d}  {F.shape[1]:>6d}  {kappa:10.3e}  {model_err:8.2f} um  {olcum_err:8.2f} um  {corr:11.6f}")
        results[N] = dy_geri
    return results


def fourier_reconstruct_targeted(dR, delta, dy_gercek, n_q, k_lists, label="hedefli"):
    """Belirli harmonik listelerine odaklı rekonstrüksiyon.
    k_lists: [[2], [2,4], [1,2,3,4], ...]  — her biri bir ölçüm stratejisi."""
    results = {}
    print(f"\n{'k listesi':>12}  {'baz':>6}  {'κ(ΔR·F)':>10}  {'model RMS':>11}  {'ölçüm RMS':>11}  {'korelasyon':>11}")
    print("-" * 68)
    for k_list in k_lists:
        F = fourier_basis(n_q, k_list=k_list)
        dy_geri, kappa, model_err, olcum_err, corr = _fit_fourier(dR, delta, dy_gercek, F)
        tag = "{" + ",".join(str(k) for k in k_list) + "}"
        print(f"  {tag:>10}  {F.shape[1]:>6d}  {kappa:10.3e}  {model_err:8.2f} um  {olcum_err:8.2f} um  {corr:11.6f}")
        results[tuple(k_list)] = dy_geri
    return results


def _resolve_kmod_test(config, n_q, cfg_idx=None):
    """Kmod parametrelerini çöz (test için)."""
    from build_response_matrix import _resolve_kmod
    return _resolve_kmod(config, n_q, cfg_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smooth", action="store_true",
                        help="Rastgele yerine sinüzoidal quad hizalama hatası üret (algoritma testi)")
    parser.add_argument("--config", "-c", type=int, default=None,
                        help="kmod_configs[N] konfig indeksi. Verilirse kmod_test_cN.npz kaydeder.")
    args = parser.parse_args()

    os.chdir(BASE)

    with open("params.json") as f:
        config = json.load(f)

    cfg_idx = args.config
    suffix  = f"_c{cfg_idx}" if cfg_idx is not None else ""

    R_dy_1 = np.load(f"R_dy_1{suffix}.npy")
    R_dy_2 = np.load(f"R_dy_2{suffix}.npy")
    R_dx_1 = np.load(f"R_dx_1{suffix}.npy")
    R_dx_2 = np.load(f"R_dx_2{suffix}.npy")

    dR_dy = R_dy_2 - R_dy_1
    dR_dx = R_dx_2 - R_dx_1

    n_q = R_dy_1.shape[0]

    g1_nom, g1_pert, quad_dG_pert, mode_label = _resolve_kmod_test(config, n_q, cfg_idx)
    if cfg_idx is not None:
        print(f"K-mod modu: kmod_configs[{cfg_idx}] — {mode_label}")
    else:
        print(f"K-mod modu: {mode_label}")

    print(f"dR_dy kosul sayisi : {np.linalg.cond(dR_dy):.3e}")
    print(f"dR_dx kosul sayisi : {np.linalg.cond(dR_dx):.3e}")

    # BPM hata parametreleri
    sigma_noise  = config.get("bpm_noise_sigma",  0.0)
    sigma_offset = config.get("bpm_offset_sigma", 0.0)
    offset_seed  = config.get("bpm_offset_seed",  55)

    rng_offset = np.random.default_rng(seed=offset_seed)
    bpm_offset_dy = rng_offset.normal(0, sigma_offset, n_q) if sigma_offset > 0 else np.zeros(n_q)
    bpm_offset_dx = rng_offset.normal(0, sigma_offset, n_q) if sigma_offset > 0 else np.zeros(n_q)
    rng_noise = np.random.default_rng(seed=99)

    # Quad hizalama hataları (geri çatılacak)
    # Üretim modları, öncelik sırasıyla:
    #   1) config'de dy_harmonics varsa → FODO antisim. smooth + RMS gürültü
    #   2) --smooth bayrağı → hardcoded sinüzoidal (geriye uyumluluk)
    #   3) varsayılan: rastgele uniform
    j = np.arange(n_q)
    if "dy_harmonics" in config or "dx_harmonics" in config:
        n_fodo = n_q // 2
        antisym = config.get("smooth_antisym_fodo", True)
        sign = np.where(j % 2 == 0, 1.0, -1.0 if antisym else 1.0)
        fodo_idx = j // 2

        def build_smooth(harmonics):
            f = np.zeros(n_fodo)
            for h in harmonics:
                k  = h["k"]
                ac = h.get("amp_cos", 0.0)
                as_= h.get("amp_sin", 0.0)
                if k == 0:
                    f += ac
                else:
                    f += ac * np.cos(2*np.pi*k*np.arange(n_fodo)/n_fodo)
                    f += as_* np.sin(2*np.pi*k*np.arange(n_fodo)/n_fodo)
            return sign * f[fodo_idx]

        dy_smooth = build_smooth(config.get("dy_harmonics", []))
        dx_smooth = build_smooth(config.get("dx_harmonics", []))

        dy_rms  = config.get("dy_random_RMS", 0.0)
        dx_rms  = config.get("dx_random_RMS", 0.0)
        dy_seed = config.get("dy_random_seed", 42)
        dx_seed = config.get("dx_random_seed", 43)
        rng_dy  = np.random.default_rng(seed=dy_seed)
        rng_dx  = np.random.default_rng(seed=dx_seed)
        dy_noise = rng_dy.normal(0, dy_rms, n_q) if dy_rms > 0 else np.zeros(n_q)
        dx_noise = rng_dx.normal(0, dx_rms, n_q) if dx_rms > 0 else np.zeros(n_q)

        dy_gercek = dy_smooth + dy_noise
        dx_gercek = dx_smooth + dx_noise

        ks_dy = [h["k"] for h in config.get("dy_harmonics", [])]
        ks_dx = [h["k"] for h in config.get("dx_harmonics", [])]
        antisym_str = "ANTİSİM" if antisym else "SİMETRİK"
        print(f"Mod: HARMONIC ({antisym_str} FODO)")
        print(f"  dy harmonikleri k={ks_dy}, smooth RMS={np.std(dy_smooth)*1e6:.1f} um, gurultu RMS={dy_rms*1e6:.1f} um")
        print(f"  dx harmonikleri k={ks_dx}, smooth RMS={np.std(dx_smooth)*1e6:.1f} um, gurultu RMS={dx_rms*1e6:.1f} um")
    elif args.smooth:
        A = 1.0e-4
        dy_gercek = A * (np.sin(2*np.pi*2*j/n_q) + 0.5*np.cos(2*np.pi*4*j/n_q))
        dx_gercek = A * (np.cos(2*np.pi*1*j/n_q) + 0.5*np.sin(2*np.pi*3*j/n_q))
        print("Mod: SMOOTH (sinüzoidal, hardcoded)  dy→k={2,4}  dx→k={1,3}")
    else:
        dy_max    = config.get("quad_random_dy_max", 1e-4)
        dx_max    = config.get("quad_random_dx_max", 1e-4)
        quad_seed = config.get("quad_random_seed", 42)
        rng = np.random.default_rng(seed=quad_seed)
        dy_gercek = rng.uniform(-dy_max, dy_max, n_q)
        dx_gercek = rng.uniform(-dx_max, dx_max, n_q)
        print("Mod: RASTGELE (uniform)")

    # Dipol tiltler — modelde yok, her iki ölçümde aynı (common-mode)
    d_tilt_max  = config.get("dipole_random_tilt_max", 0.0)
    d_tilt_seed = config.get("dipole_random_seed", 43)
    rng_dtilt   = np.random.default_rng(seed=d_tilt_seed)
    dipole_tilt_sabit = rng_dtilt.uniform(-d_tilt_max, d_tilt_max, n_q) if d_tilt_max > 0 else np.zeros(n_q)

    # Quad tilt — modelde yok, skew-quadrupol bileşeni → x-y kuplaji
    q_tilt_max  = config.get("quad_random_tilt_max", 0.0)
    q_tilt_seed = config.get("quad_random_tilt_seed", 44)
    rng_qtilt   = np.random.default_rng(seed=q_tilt_seed)
    quad_tilt_sabit = rng_qtilt.uniform(-q_tilt_max, q_tilt_max, n_q) if q_tilt_max > 0 else np.zeros(n_q)

    print(f"\nGercek hatalar:")
    print(f"  dy        RMS = {np.std(dy_gercek)*1e3:.3f} mm")
    print(f"  dx        RMS = {np.std(dx_gercek)*1e3:.3f} mm")
    print(f"  dipol tilt RMS = {np.std(dipole_tilt_sabit)*1e3:.3f} mrad  (gorulmez — modelde yok)")
    print(f"  quad tilt RMS = {np.std(quad_tilt_sabit)*1e3:.3f} mrad  (gorulmez — x-y kuplaji yaratir)")
    if sigma_noise > 0 or sigma_offset > 0:
        print(f"  BPM gurultu s={sigma_noise*1e6:.1f} um, ofset s={sigma_offset*1e6:.1f} um")
        print(f"  -> Ofset, delta_y farki ile otomatik iptal olur")

    # İki konfigürasyonda simülasyon — tilt'ler fiziksel olarak var ama modelde yok
    alanlar1, state01 = setup_fields(config, g1_override=g1_nom)
    alanlar2, state02 = setup_fields(config, g1_override=g1_pert)

    print("\nKonfigurasyon 1 (g_nom) kosumu...")
    x1, y1 = run_sim(alanlar1, state01, config, dy_gercek, dx_gercek,
                     dipole_tilt=dipole_tilt_sabit, quad_tilt=quad_tilt_sabit)
    print("Konfigurasyon 2 (g_pert) kosumu...")
    x2, y2 = run_sim(alanlar2, state02, config, dy_gercek, dx_gercek,
                     dipole_tilt=dipole_tilt_sabit, quad_tilt=quad_tilt_sabit,
                     quad_dG=quad_dG_pert)

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

    # Direkt çözüm (referans, kötü koşullanmışta patlar)
    dy_direct = np.linalg.solve(dR_dy, delta_y)
    dx_direct = np.linalg.solve(dR_dx, delta_x)
    print("\nDirekt çözüm (np.linalg.solve):")
    print_results("  dy", dy_gercek, dy_direct)
    print_results("  dx", dx_gercek, dx_direct)

    # TSVD: eşik = residual/sinyal oranı (otomatik)
    tau_y = np.std(delta_y - quad_signal_y) / np.std(quad_signal_y)
    tau_x = np.std(delta_x - quad_signal_x) / np.std(quad_signal_x)
    dy_geri, ny, sy = tsvd_solve(dR_dy, delta_y, tau_y)
    dx_geri, nx, sx = tsvd_solve(dR_dx, delta_x, tau_x)
    print(f"\nTSVD çözüm (otomatik eşik = residual/sinyal):")
    print(f"  dy: tau={tau_y:.2e}, tutulan mod={ny}/{n_q}, σ_max/σ_min(kalan)={sy[0]/sy[ny-1]:.2e}")
    print(f"  dx: tau={tau_x:.2e}, tutulan mod={nx}/{n_q}, σ_max/σ_min(kalan)={sx[0]/sx[nx-1]:.2e}")
    print_results("  dy", dy_gercek, dy_geri)
    print_results("  dx", dx_gercek, dx_geri)

    # Fourier tabanlı rekonstrüksiyon — k=1..N taraması
    N_list = [1, 2, 3, 4, 5]
    print("\n" + "=" * 68)
    print("Fourier rekonstrüksiyonu (k=1..N + DC) — dikey (dy)")
    print("  'model RMS': gerçek dy'nin N harmonikle temsil hatası")
    print("  'ölçüm RMS': ΔR·F üzerinden geri çatım hatası")
    print("=" * 68)
    dy_fourier = fourier_reconstruct(dR_dy, delta_y, dy_gercek, n_q, N_list)

    print("\n" + "=" * 68)
    print("Fourier rekonstrüksiyonu (k=1..N + DC) — yatay (dx)")
    print("=" * 68)
    dx_fourier = fourier_reconstruct(dR_dx, delta_x, dx_gercek, n_q, N_list)

    # Hedefli rekonstrüksiyon — yalnızca ilgilenilen harmonikler
    # k_listesi: her satır bir strateji (kaç harmonik, hangisi)
    k_lists = [[2], [4], [2, 4], [1, 2, 3, 4]]
    print("\n" + "=" * 68)
    print("Hedefli Fourier rekonstrüksiyonu — dikey (dy)")
    print("  (DC olmadan, yalnızca seçili harmonikler)")
    print("=" * 68)
    dy_targeted = fourier_reconstruct_targeted(dR_dy, delta_y, dy_gercek, n_q, k_lists)

    k_lists_dx = [[1], [3], [1, 3], [1, 2, 3, 4]]
    print("\n" + "=" * 68)
    print("Hedefli Fourier rekonstrüksiyonu — yatay (dx)")
    print("=" * 68)
    dx_targeted = fourier_reconstruct_targeted(dR_dx, delta_x, dx_gercek, n_q, k_lists_dx)

    out_name = f"kmod_test{suffix}.npz" if cfg_idx is not None else "kmod_reconstruction_test.npz"
    np.savez(out_name,
             dy_gercek=dy_gercek, dy_geri=dy_geri,
             dx_gercek=dx_gercek, dx_geri=dx_geri,
             dipole_tilt_sabit=dipole_tilt_sabit,
             quad_tilt_sabit=quad_tilt_sabit,
             quad_signal_y=quad_signal_y, quad_signal_x=quad_signal_x,
             delta_y=delta_y, delta_x=delta_x,
             **{f"dy_fourier_N{N}": dy_fourier[N] for N in N_list},
             **{f"dx_fourier_N{N}": dx_fourier[N] for N in N_list})
    print(f"\nSonuclar '{out_name}' dosyasina kaydedildi.")


if __name__ == "__main__":
    main()
