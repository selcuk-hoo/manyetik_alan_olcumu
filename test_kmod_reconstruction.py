#!/usr/bin/env python3
"""
test_kmod_reconstruction.py

K-modulasyon benzeri geri catim: referans olcum olmadan quad
misalignment'larini geri cat, dipol tilt'in ne kadar bozdugunu olc.

Sonunda M_loco varsa LOCO cozumunu de dener ve karsilastirir.

On kosul: build_response_matrix.py calistirilmis olmali.
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
    print(f"  {label:30s}  hata RMS={np.std(hata)*1e6:7.2f} um   korelasyon={corr:.6f}")


def main():
    os.chdir(BASE)

    with open("params.json") as f:
        config = json.load(f)

    R_dy_1   = np.load("R_dy_1.npy")
    R_dy_2   = np.load("R_dy_2.npy")
    R_dx_1   = np.load("R_dx_1.npy")
    R_dx_2   = np.load("R_dx_2.npy")
    R_tilt_1 = np.load("R_tilt_1.npy")
    R_tilt_2 = np.load("R_tilt_2.npy")

    dR_dy   = R_dy_2   - R_dy_1
    dR_dx   = R_dx_2   - R_dx_1
    dR_tilt = R_tilt_2 - R_tilt_1

    n_q    = R_dy_1.shape[0]
    g1_nom = config.get("g1", 0.21)
    eps    = 0.02
    g1_pert = g1_nom * (1.0 + eps)

    print(f"dR_dy kosul sayisi : {np.linalg.cond(dR_dy):.3e}")
    print(f"dR_dx kosul sayisi : {np.linalg.cond(dR_dx):.3e}")
    print(f"dg/g               : {eps*100:.1f}%  (g_nom={g1_nom}, g_pert={g1_pert:.4f})")

    sigma_noise  = config.get("bpm_noise_sigma",  0.0)
    sigma_offset = config.get("bpm_offset_sigma", 0.0)
    offset_seed  = config.get("bpm_offset_seed",  55)

    rng_offset = np.random.default_rng(seed=offset_seed)
    bpm_offset_dy = rng_offset.normal(0, sigma_offset, n_q) if sigma_offset > 0 else np.zeros(n_q)
    bpm_offset_dx = rng_offset.normal(0, sigma_offset, n_q) if sigma_offset > 0 else np.zeros(n_q)
    rng_noise = np.random.default_rng(seed=99)

    dy_max = config.get("quad_random_dy_max", 0.3e-3)
    dx_max = config.get("quad_random_dx_max", 0.3e-3)
    quad_seed = config.get("quad_random_seed", 13)
    rng = np.random.default_rng(seed=quad_seed)
    dy_gercek = rng.uniform(-dy_max, dy_max, n_q)
    dx_gercek = rng.uniform(-dx_max, dx_max, n_q)

    tilt_max  = config.get("dipole_random_tilt_max", 0.2e-3)
    tilt_seed = config.get("dipole_random_seed", 43)
    rng_tilt  = np.random.default_rng(seed=tilt_seed)
    tilt_sabit = rng_tilt.uniform(-tilt_max, tilt_max, n_q)

    print(f"\nGercek hatalar:")
    print(f"  dy   RMS = {np.std(dy_gercek)*1e3:.3f} mm")
    print(f"  dx   RMS = {np.std(dx_gercek)*1e3:.3f} mm")
    print(f"  tilt RMS = {np.std(tilt_sabit)*1e3:.3f} mrad")

    alanlar1, state01 = setup_fields(config)
    alanlar2, state02 = setup_fields(config, g1_override=g1_pert)

    print("\nKonfigurasyon 1 (g_nom) kosumu...")
    x1, y1 = run_sim(alanlar1, state01, config, dy_gercek, dx_gercek, dipole_tilt=tilt_sabit)
    print("Konfigurasyon 2 (g_pert) kosumu...")
    x2, y2 = run_sim(alanlar2, state02, config, dy_gercek, dx_gercek, dipole_tilt=tilt_sabit)

    y1_meas = apply_bpm_errors(y1, sigma_noise, sigma_offset, bpm_offset_dy, rng_noise)
    y2_meas = apply_bpm_errors(y2, sigma_noise, sigma_offset, bpm_offset_dy, rng_noise)
    x1_meas = apply_bpm_errors(x1, sigma_noise, sigma_offset, bpm_offset_dx, rng_noise)
    x2_meas = apply_bpm_errors(x2, sigma_noise, sigma_offset, bpm_offset_dx, rng_noise)

    delta_y = y2_meas - y1_meas
    delta_x = x2_meas - x1_meas

    delta_y_ideal = (y2 - y1)
    quad_signal   = dR_dy @ dy_gercek
    tilt_contam   = dR_tilt @ tilt_sabit
    print(f"\nDelta_y bilesenleri (gurultusuz):")
    print(f"  Quad sinyali  dR_dy*dy   RMS = {np.std(quad_signal)*1e6:.1f} um")
    print(f"  Tilt kirliligi dR_tilt*t RMS = {np.std(tilt_contam)*1e6:.1f} um")
    print(f"  Oran (kirlilik/sinyal)       = {np.std(tilt_contam)/np.std(quad_signal)*100:.1f}%")

    print("\nGeri catim sonuclari:")

    dy_ideal = np.linalg.solve(dR_dy, quad_signal)
    print("\n[Ideal: tilt=0, gurultu=0]")
    print_results("dy", dy_gercek, dy_ideal)

    dy_geri = np.linalg.solve(dR_dy, delta_y)
    dx_geri = np.linalg.solve(dR_dx, delta_x)
    print("\n[Gercekci: tilt sabit, BPM hatalari var]")
    print_results("dy", dy_gercek, dy_geri)
    print_results("dx", dx_gercek, dx_geri)

    hata_tilt  = np.linalg.solve(dR_dy, tilt_contam)
    hata_noise = dy_geri - np.linalg.solve(dR_dy, delta_y_ideal)
    print(f"\nHata kaynaklari (dy):")
    print(f"  Tilt'ten   : {np.std(hata_tilt)*1e6:.2f} um")
    print(f"  Gurultuden : {np.std(hata_noise)*1e6:.2f} um")
    print(f"  Toplam     : {np.std(dy_geri - dy_gercek)*1e6:.2f} um")

    # ── LOCO cozumu: dy ve tilt birlikte (96x96) ─────────────────────────
    print("\n" + "=" * 60)
    print("LOCO cozumu: dy ve tilt birlikte (96x96)")
    print("=" * 60)

    dy_loco = tilt_loco = None
    dy_loco_reg = tilt_loco_reg = None

    if os.path.exists("M_loco.npy"):
        M_loco = np.load("M_loco.npy")
        cond_M = np.linalg.cond(M_loco)
        print(f"  M_loco sekli      : {M_loco.shape}")
        print(f"  kappa(M_loco)     : {cond_M:.3e}")

        rhs = np.concatenate([y1_meas, y2_meas])

        try:
            sol = np.linalg.solve(M_loco, rhs)
            dy_loco   = sol[:n_q]
            tilt_loco = sol[n_q:]
            print("\n[LOCO dogrudan cozum]")
            print_results("dy   (LOCO)",   dy_gercek,  dy_loco)
            print_results("tilt (LOCO)",   tilt_sabit, tilt_loco)
        except np.linalg.LinAlgError as e:
            print(f"  Dogrudan cozum basarisiz: {e}")

        U, S, Vt = np.linalg.svd(M_loco, full_matrices=False)
        cutoff = S[0] * 1e-6
        S_inv = np.where(S > cutoff, 1.0 / S, 0.0)
        n_kept = int(np.sum(S > cutoff))
        sol_reg = Vt.T @ (S_inv * (U.T @ rhs))
        dy_loco_reg   = sol_reg[:n_q]
        tilt_loco_reg = sol_reg[n_q:]
        print(f"\n[LOCO + SVD truncate, cutoff = sigma_max * 1e-6, "
              f"{n_kept}/{len(S)} mod tutuldu]")
        print_results("dy   (LOCO+SVD)", dy_gercek,  dy_loco_reg)
        print_results("tilt (LOCO+SVD)", tilt_sabit, tilt_loco_reg)
    else:
        print("  M_loco.npy bulunamadi - once build_response_matrix.py'i calistirin.")

    print("\n" + "=" * 60)
    print("Ozet: dy geri catim hatasi RMS [um]")
    print("=" * 60)
    print(f"  K-mod (yalniz dR_dy ile)           : "
          f"{np.std(dy_geri - dy_gercek)*1e6:9.2f}")
    if dy_loco is not None:
        print(f"  LOCO (M_loco dogrudan)             : "
              f"{np.std(dy_loco - dy_gercek)*1e6:9.2f}")
    if dy_loco_reg is not None:
        print(f"  LOCO (M_loco + SVD truncate)       : "
              f"{np.std(dy_loco_reg - dy_gercek)*1e6:9.2f}")

    save_dict = dict(
        dy_gercek=dy_gercek, dy_geri=dy_geri,
        dx_gercek=dx_gercek, dx_geri=dx_geri,
        tilt_sabit=tilt_sabit,
        quad_signal=quad_signal, tilt_contam=tilt_contam,
    )
    if dy_loco is not None:
        save_dict["dy_loco"]   = dy_loco
        save_dict["tilt_loco"] = tilt_loco
    if dy_loco_reg is not None:
        save_dict["dy_loco_reg"]   = dy_loco_reg
        save_dict["tilt_loco_reg"] = tilt_loco_reg

    np.savez("kmod_reconstruction_test.npz", **save_dict)
    print("\nSonuclar 'kmod_reconstruction_test.npz' dosyasina kaydedildi.")


if __name__ == "__main__":
    main()
