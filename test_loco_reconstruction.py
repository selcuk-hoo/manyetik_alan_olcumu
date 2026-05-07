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
  linalg.solve : doğrudan LU — iyi koşullu sistemler için yedek
  SVD          : kırpmalı tekil değer ayrışımı (rcond eşiği)
  Tikhonov     : min ||Mx-b||² + λ||x||²  —  gürültülü sistemler için

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


def compute_svd(M):
    """M matrisinin SVD'sini hesaplar."""
    return np.linalg.svd(M, full_matrices=False)


def solve_svd(b, U, s, Vt, rcond=1e-6):
    """Kırpmalı SVD ile Mx = b çözer.

    rcond: en büyük tekil değere oranla eşik — altındakiler sıfırlanır.
    """
    threshold = rcond * s[0]
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)
    kept    = int(np.sum(s > threshold))
    dropped = len(s) - kept
    x = Vt.T @ (s_inv * (U.T @ b))
    return x, kept, dropped


def solve_tikhonov(b, U, s, Vt, lambda_):
    """Tikhonov regularizasyonu: min ||Mx-b||² + λ||x||²

    SVD üzerinden filtre faktörleriyle çözüm:
      x = V @ diag(σᵢ/(σᵢ²+λ)) @ Uᵀ @ b

    Büyük σᵢ → faktör ≈ 1/σᵢ  (korunur)
    Küçük σᵢ → faktör ≈ σᵢ/λ  (bastırılır, kırpmalı SVD'deki sıfır yerine)
    """
    filter_factors = s / (s**2 + lambda_)
    x = Vt.T @ (filter_factors * (U.T @ b))
    residual = np.linalg.norm(U @ (s * (Vt @ x)) - b)  # ||Mx - b||
    return x, residual


def find_lambda_discrepancy(b, U, s, Vt, sigma_noise, n_obs, eta=1.0):
    """Uyumsuzluk ilkesiyle λ seçer: ||Mx-b|| ≈ η × σ_noise × √n_obs

    eta: güvenlik faktörü (varsayılan 1.0)
    Gerçek deneyde uygulanabilir — yalnızca σ_noise bilinmesi yeterli.
    """
    target = eta * sigma_noise * np.sqrt(n_obs)
    lam_lo, lam_hi = 1e-10 * s[0]**2, 1e10 * s[0]**2
    for _ in range(60):
        lam_mid = np.sqrt(lam_lo * lam_hi)
        _, res = solve_tikhonov(b, U, s, Vt, lam_mid)
        if res < target:
            lam_lo = lam_mid   # λ çok küçük → artır
        else:
            lam_hi = lam_mid   # λ çok büyük → azalt
    return np.sqrt(lam_lo * lam_hi)


def find_lambda_optimal(b, U, s, Vt, x_true):
    """Simülasyona özgü: gerçek hatayı minimize eden λ'yı bulur.

    Gerçek deneyde KULLANILAMAZ — yalnızca doğrulama amacıyla.
    """
    lambdas = np.logspace(-4, 8, 500) * s[-1]**2
    best_lam, best_err = lambdas[0], np.inf
    for lam in lambdas:
        x, _ = solve_tikhonov(b, U, s, Vt, lam)
        err = np.linalg.norm(x - x_true)
        if err < best_err:
            best_err, best_lam = err, lam
    return best_lam


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

    R_dx_1 = np.load("R_dx_1.npy")
    M_loco = np.load("M_loco.npy")

    n_q     = R_dx_1.shape[0]   # 48
    g1_nom  = config.get("g1", 0.21)
    g1_pert = g1_nom * 1.02

    sigma_noise  = config.get("bpm_noise_sigma",  0.0)
    sigma_offset = config.get("bpm_offset_sigma", 0.0)
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
    dy_gercek   = rng.uniform(-0.3e-3, 0.3e-3, n_q)
    dx_gercek   = rng.uniform(-0.3e-3, 0.3e-3, n_q)
    tilt_gercek = rng.uniform(-0.2e-3, 0.2e-3, n_q)

    print("Gerçek hatalar:")
    print(f"  dy   RMS = {np.std(dy_gercek)*1e3:.3f} mm")
    print(f"  dx   RMS = {np.std(dx_gercek)*1e3:.3f} mm")
    print(f"  tilt RMS = {np.std(tilt_gercek)*1e3:.3f} mrad")
    if sigma_noise > 0:
        print(f"\nBPM gürültü σ = {sigma_noise*1e6:.1f} μm,  ofset σ = {sigma_offset*1e6:.1f} μm")

    print("\nReferans koşumları...")
    x0_1, y0_1 = run_sim(alanlar1, state01, config, np.zeros(n_q), np.zeros(n_q))
    x0_2, y0_2 = run_sim(alanlar2, state02, config, np.zeros(n_q), np.zeros(n_q))

    print("Hatalı koşumlar...")
    x_cod_1, y_cod_1 = run_sim(alanlar1, state01, config,
                                dy_gercek, dx_gercek, dipole_tilt=tilt_gercek)
    x_cod_2, y_cod_2 = run_sim(alanlar2, state02, config,
                                dy_gercek, dx_gercek, dipole_tilt=tilt_gercek)

    # Önce ham ölçümlere gürültü + ofset ekle, sonra fark al
    # → ofset her iki ölçümde aynı olduğundan farkta iptal olur
    # → sadece iki bağımsız gürültü kalır (etkin std = √2 × σ_noise)
    y0_1_meas  = apply_bpm_errors(y0_1,    sigma_noise, sigma_offset, bpm_offset_dy, rng_noise)
    y_cod_1_meas = apply_bpm_errors(y_cod_1, sigma_noise, sigma_offset, bpm_offset_dy, rng_noise)
    y0_2_meas  = apply_bpm_errors(y0_2,    sigma_noise, sigma_offset, bpm_offset_dy, rng_noise)
    y_cod_2_meas = apply_bpm_errors(y_cod_2, sigma_noise, sigma_offset, bpm_offset_dy, rng_noise)
    x0_1_meas  = apply_bpm_errors(x0_1,    sigma_noise, sigma_offset, bpm_offset_dx, rng_noise)
    x_cod_1_meas = apply_bpm_errors(x_cod_1, sigma_noise, sigma_offset, bpm_offset_dx, rng_noise)

    dy_cod_1 = y_cod_1_meas - y0_1_meas
    dy_cod_2 = y_cod_2_meas - y0_2_meas
    dx_cod_1 = x_cod_1_meas - x0_1_meas

    print(f"\nÖlçülen COD:")
    print(f"  dikey [nom]  RMS = {np.std(dy_cod_1)*1e3:.3f} mm")
    print(f"  dikey [pert] RMS = {np.std(dy_cod_2)*1e3:.3f} mm")
    print(f"  radyal[nom]  RMS = {np.std(dx_cod_1)*1e3:.3f} mm")

    rhs    = np.concatenate([dy_cod_1, dy_cod_2])
    x_true = np.concatenate([dy_gercek, tilt_gercek])

    # SVD — bir kez hesapla, tüm yöntemler paylaşır
    U, s, Vt = compute_svd(M_loco)
    print(f"\nSVD: σ_max={s[0]:.3e}, σ_min={s[-1]:.3e}, κ={s[0]/s[-1]:.3e}")

    # Yöntem 1: linalg.solve (yedek)
    sol_dir = np.linalg.solve(M_loco, rhs)
    dx_dir  = np.linalg.solve(R_dx_1, dx_cod_1)
    print_results("linalg.solve",
                  dy_gercek, sol_dir[:n_q], dx_gercek, dx_dir,
                  tilt_gercek, sol_dir[n_q:])

    # Yöntem 2: Kırpmalı SVD
    sol_svd, kept, dropped = solve_svd(rhs, U, s, Vt, rcond=1e-6)
    print(f"\nSVD: tutulan {kept}/{len(s)}, atılan {dropped}/{len(s)}")
    print_results("SVD (rcond=1e-6)",
                  dy_gercek, sol_svd[:n_q], dx_gercek, dx_dir,
                  tilt_gercek, sol_svd[n_q:])

    # Yöntem 3: Tikhonov — optimal λ (simülasyon doğrulaması)
    lam_opt = find_lambda_optimal(rhs, U, s, Vt, x_true)
    sol_tik_opt, _ = solve_tikhonov(rhs, U, s, Vt, lam_opt)
    print(f"\nTikhonov optimal λ = {lam_opt:.3e}  (gerçek hata minimize edildi — sim. only)")
    print_results("Tikhonov (optimal λ)",
                  dy_gercek, sol_tik_opt[:n_q], dx_gercek, dx_dir,
                  tilt_gercek, sol_tik_opt[n_q:])

    # Yöntem 4: Tikhonov — uyumsuzluk ilkesi
    # Efektif σ: hem gürültü hem ofset residual'a katkıda bulunur
    if sigma_noise > 0:
        sigma_eff = np.sqrt(2) * sigma_noise  # her farkta iki bağımsız gürültü
        lam_disc = find_lambda_discrepancy(rhs, U, s, Vt, sigma_eff, len(rhs))
        sol_tik_disc, res_disc = solve_tikhonov(rhs, U, s, Vt, lam_disc)
        print(f"\nTikhonov uyumsuzluk ilkesi: λ = {lam_disc:.3e},  ||Mx-b|| = {res_disc:.3e}")
        print_results("Tikhonov (uyumsuzluk ilkesi)",
                      dy_gercek, sol_tik_disc[:n_q], dx_gercek, dx_dir,
                      tilt_gercek, sol_tik_disc[n_q:])

    np.savez("loco_reconstruction_test.npz",
             dy_gercek=dy_gercek,   dy_geri=sol_tik_opt[:n_q],
             dx_gercek=dx_gercek,   dx_geri=dx_dir,
             tilt_gercek=tilt_gercek, tilt_geri=sol_tik_opt[n_q:],
             singular_values=s)
    print("\nSonuçlar 'loco_reconstruction_test.npz' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()
