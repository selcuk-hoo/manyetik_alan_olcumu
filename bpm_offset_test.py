#!/usr/bin/env python3
"""bpm_offset_test.py — R-tabanlı Fourier fit vs kmod (ΔR-tabanlı), BPM ofseti altında

Test sorusu:
  y = R·Δq + b  ölçümünü Fourier bazıyla direkt fit edersek,
  kmod (Δy = ΔR·Δq, b-serbest) ile karşılaştırıldığında ne kadar iyi?

Beklenti:
  - kmod: b tamamen sıfırlanır; bilgi kaybı tek-quad → rank-1
  - R-tabanlı: full rank, ama b'nin Fourier bazına sızıntısı var
    beklenen taban ≈ σ_b · ||(R·F)⁺||  (κ(R)≈160 ile bastırılmış)
"""
import json
import numpy as np
import sys
import os

# fourier_reconstruct.py'deki fodo_basis'i kullan
sys.path.insert(0, os.path.dirname(__file__))
from fourier_reconstruct import fodo_basis

# ── Yardımcılar ──────────────────────────────────────────────────────────────

def load_matrices():
    needed = ["R_dy_1.npy", "dR_dy.npy"]
    for f in needed:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"{f} bulunamadı — önce build_response_matrix.py çalıştır")
    R  = np.load("R_dy_1.npy")   # nominal tepki matrisi (48×48)
    dR = np.load("dR_dy.npy")    # ΔR = R2-R1 kmod matrisi (48×48)
    return R, dR


def make_truth_dy(config, n_q=48, add_random=False):
    """params.json'dan gerçek quad misalignment vektörü üret.

    add_random=True: dy_random_RMS değerinde rastgele bileşen ekler.
    Bu bileşen Fourier bazının dışında → R-fit için SNR sorunu (Sorun 3).
    """
    harmonics = config.get("dy_harmonics", [])
    antisym   = config.get("smooth_antisym_fodo", True)
    N = n_q // 2
    j = np.arange(n_q)
    s = (-1.0)**j if antisym else np.ones(n_q)
    n = j // 2
    dy = np.zeros(n_q)
    for h in harmonics:
        k = h["k"]
        ac = h.get("amp_cos", 0.0)
        as_ = h.get("amp_sin", 0.0)
        if k == 0:
            dy += s * ac
        else:
            dy += s * (ac * np.cos(2*np.pi*k*n/N) + as_ * np.sin(2*np.pi*k*n/N))
    if add_random:
        sigma_r = config.get("dy_random_RMS", 0.0)
        if sigma_r > 0:
            rng_r = np.random.default_rng(config.get("dy_random_seed", 42))
            dy += rng_r.normal(0, sigma_r, n_q)
    return dy


def fourier_fit(M, y, rcond=1e-3):
    """â = lstsq(M, y); döndürür â, κ(M), rank(M)."""
    sv = np.linalg.svd(M, compute_uv=False)
    rank = int(np.sum(sv > sv[0] * rcond))
    kappa = sv[0] / sv[rank-1] if rank > 0 else np.inf
    a, *_ = np.linalg.lstsq(M, y, rcond=rcond)
    return a, kappa, rank


def report_k2(a, meta, dy_true, label, n_q=48, antisym=True):
    """k=2 harmonik için genlik/faz/hata raporla."""
    idx = [i for i, (k, _) in enumerate(meta) if k == 2]
    if not idx:
        print(f"  {label}: k=2 bazda yok")
        return
    a2c, a2s = a[idx[0]], a[idx[1]] if len(idx) > 1 else (a[idx[0]], 0.0)
    amp_fit  = np.sqrt(a2c**2 + a2s**2)
    phi_fit  = np.arctan2(a2s, a2c)

    # gerçek k=2
    F2, m2 = fodo_basis(n_q, [2], antisym)
    a2_true, *_ = np.linalg.lstsq(F2, dy_true, rcond=None)
    amp_true = np.sqrt(a2_true[0]**2 + a2_true[1]**2)
    phi_true = np.arctan2(a2_true[1], a2_true[0])

    err_pct  = abs(amp_fit - amp_true) / amp_true * 100
    dphi     = abs(phi_fit - phi_true)
    dphi     = min(dphi, 2*np.pi - dphi)

    print(f"  {label}:")
    print(f"    k=2 fit  : {amp_fit*1e6:7.2f} μm  ∠{phi_fit:.2f} rad")
    print(f"    k=2 gerçek: {amp_true*1e6:7.2f} μm  ∠{phi_true:.2f} rad")
    print(f"    genlik hatası {err_pct:.0f}%   faz Δ={dphi:.2f} rad")


def profile_error(dy_fit, dy_true):
    err  = np.linalg.norm(dy_fit - dy_true) * 1e6
    corr = np.corrcoef(dy_fit, dy_true)[0, 1]
    return err, corr


# ── Ana test ─────────────────────────────────────────────────────────────────

def run(bpm_offset_sigma=100e-6, n_trials=50, seed=0, add_random=False):
    print("=" * 64)
    print(f"  BPM ofset testi   σ_b = {bpm_offset_sigma*1e6:.0f} μm   ({n_trials} deneme)")
    print("=" * 64)

    with open("params.json") as f:
        config = json.load(f)

    R, dR = load_matrices()
    n_q = R.shape[0]   # 48
    antisym = config.get("smooth_antisym_fodo", True)

    # gerçek misalignment (random bileşen opsiyonel)
    dy_true = make_truth_dy(config, n_q, add_random=add_random)
    if add_random and config.get("dy_random_RMS", 0) > 0:
        print(f"  [UYARI] dy_random_RMS = {config['dy_random_RMS']*1e6:.0f} μm eklendi "
              f"— Fourier bazı dışı bileşen var")
    truth_k2_amp = np.sqrt(
        sum(config["dy_harmonics"][0]["amp_cos"]**2 +
            config["dy_harmonics"][0].get("amp_sin", 0)**2
            for h in config["dy_harmonics"] if h["k"] == 2))
    # daha temiz:
    F_all, m_all = fodo_basis(n_q, [2,4,6,8], antisym)

    # gerçek harmonik bilgisi için k listesi
    truth_ks = sorted(set(h["k"] for h in config["dy_harmonics"]))

    # Fourier bazları
    F_full, meta_full = fodo_basis(n_q, truth_ks, antisym)  # baz = truth
    F_k2,  meta_k2   = fodo_basis(n_q, [2],       antisym)  # yalnız k=2

    # ΔR + Fourier: kmod (b-serbest) —————————————————————————————
    dRF_full = dR @ F_full
    dRF_k2   = dR @ F_k2

    sv_dRF = np.linalg.svd(dRF_full, compute_uv=False)
    rank_dRF = int(np.sum(sv_dRF > sv_dRF[0] * 1e-3))
    kappa_dRF = sv_dRF[0] / sv_dRF[rank_dRF - 1]

    # R + Fourier: direkt (b var) —————————————————————————————————
    RF_full = R @ F_full
    RF_k2   = R @ F_k2

    sv_RF = np.linalg.svd(RF_full, compute_uv=False)
    rank_RF = int(np.sum(sv_RF > sv_RF[0] * 1e-3))
    kappa_RF = sv_RF[0] / sv_RF[rank_RF - 1]

    print(f"\n  Matris özellikleri:")
    print(f"    κ(R)          = {np.linalg.cond(R):.0f}")
    print(f"    κ(ΔR)         = {np.linalg.cond(dR):.2e}")
    print(f"    κ(ΔR·F_full)  = {kappa_dRF:.1f}   rank = {rank_dRF}/{F_full.shape[1]}")
    print(f"    κ(R·F_full)   = {kappa_RF:.1f}   rank = {rank_RF}/{F_full.shape[1]}")

    # teorik b-sızıntısı tahmini
    RF_pinv = np.linalg.pinv(RF_full, rcond=1e-3)
    b_leak_theory = bpm_offset_sigma * np.linalg.norm(RF_pinv, 'fro') / np.sqrt(n_q)
    print(f"\n  Teorik b-sızıntısı (R·F bazı): "
          f"≈ {b_leak_theory*1e6:.1f} μm RMS katsayı hatası")

    # Monte Carlo: çok deneme üzerinden ortalama
    rng = np.random.default_rng(seed)

    errs = {
        "kmod_full"   : [],   # ΔR, baz=truth
        "kmod_k2"     : [],   # ΔR, baz={k=2}
        "R_full"      : [],   # R,  baz=truth
        "R_k2"        : [],   # R,  baz={k=2}
    }
    k2_amps = {k: [] for k in errs}
    k2_phis = {k: [] for k in errs}

    dy_true_orb = R @ dy_true   # gerçek orbit (b olmadan)

    for _ in range(n_trials):
        b = rng.normal(0, bpm_offset_sigma, n_q)

        # ölçümler
        y_with_b = dy_true_orb + b        # y = R·Δq + b
        dy_kmod  = dR @ dy_true           # Δy = ΔR·Δq  (b-serbest)

        # kmod (b-serbest) — full baz
        a, *_ = np.linalg.lstsq(dRF_full, dy_kmod, rcond=1e-3)
        dy_fit = F_full @ a
        errs["kmod_full"].append(profile_error(dy_fit, dy_true))
        idx2 = [i for i,(k,_) in enumerate(meta_full) if k==2]
        k2_amps["kmod_full"].append(np.sqrt(a[idx2[0]]**2 + a[idx2[1]]**2))
        k2_phis["kmod_full"].append(np.arctan2(a[idx2[1]], a[idx2[0]]))

        # kmod — yalnız k=2
        a2, *_ = np.linalg.lstsq(dRF_k2, dy_kmod, rcond=1e-3)
        dy_fit2 = F_k2 @ a2
        errs["kmod_k2"].append(profile_error(dy_fit2, dy_true))
        k2_amps["kmod_k2"].append(np.sqrt(a2[0]**2 + a2[1]**2))
        k2_phis["kmod_k2"].append(np.arctan2(a2[1], a2[0]))

        # R-tabanlı — full baz
        a, *_ = np.linalg.lstsq(RF_full, y_with_b, rcond=1e-3)
        dy_fit = F_full @ a
        errs["R_full"].append(profile_error(dy_fit, dy_true))
        idx2 = [i for i,(k,_) in enumerate(meta_full) if k==2]
        k2_amps["R_full"].append(np.sqrt(a[idx2[0]]**2 + a[idx2[1]]**2))
        k2_phis["R_full"].append(np.arctan2(a[idx2[1]], a[idx2[0]]))

        # R-tabanlı — yalnız k=2
        a2, *_ = np.linalg.lstsq(RF_k2, y_with_b, rcond=1e-3)
        dy_fit2 = F_k2 @ a2
        errs["R_k2"].append(profile_error(dy_fit2, dy_true))
        k2_amps["R_k2"].append(np.sqrt(a2[0]**2 + a2[1]**2))
        k2_phis["R_k2"].append(np.arctan2(a2[1], a2[0]))

    # gerçek k=2
    a2t, *_ = np.linalg.lstsq(F_k2, dy_true, rcond=None)
    amp_true = np.sqrt(a2t[0]**2 + a2t[1]**2)
    phi_true = np.arctan2(a2t[1], a2t[0])

    print(f"\n  Gerçek k=2: {amp_true*1e6:.2f} μm @ ∠{phi_true:.3f} rad")

    print(f"\n  {'Yöntem':<22} {'RMS hata (μm)':>14} {'Korelasyon':>12} "
          f"{'k=2 genlik (μm)':>17} {'k=2 faz hatası (rad)':>21}")
    print("  " + "-" * 90)

    labels = {
        "kmod_full" : "kmod  baz=truth",
        "kmod_k2"   : "kmod  baz={k=2}",
        "R_full"    : "R-fit baz=truth",
        "R_k2"      : "R-fit baz={k=2}",
    }

    for key, lbl in labels.items():
        rms_errs  = [e[0] for e in errs[key]]
        corrs     = [e[1] for e in errs[key]]
        amps      = k2_amps[key]
        phis      = k2_phis[key]

        mean_rms  = np.mean(rms_errs)
        mean_corr = np.mean(corrs)
        mean_amp  = np.mean(amps) * 1e6
        dphi      = np.abs(np.array(phis) - phi_true)
        dphi      = np.minimum(dphi, 2*np.pi - dphi)
        mean_dphi = np.mean(dphi)
        amp_err   = abs(mean_amp - amp_true*1e6) / (amp_true*1e6) * 100

        print(f"  {lbl:<22} {mean_rms:>13.1f}  {mean_corr:>11.3f}  "
              f"{mean_amp:>13.2f} ({amp_err:+.0f}%)  {mean_dphi:>15.3f}")

    print(f"\n  NOT: kmod = ΔR kullanır (b iptal, ama rank sınırlı)")
    print(f"       R-fit = R kullanır (full rank, ama b sızar)")


def model_error_sweep(bpm_offset_sigma=100e-6,
                      rel_errors=(0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05),
                      n_trials=50, seed=0, add_random=False):
    """R modelindeki göreli gradyen hatası arttıkça R-fit kalitesi nasıl bozuluyor?

    Fiziksel model: her quad'ın gradyeni K_j yerine K_j*(1+ε_j) olarak bilinir.
    Bu R'nin j. sütununu (1+ε_j) ile ölçekler → R_model[:,j] = R_true[:,j]*(1+ε_j).
    Gerçek orbit y = R_true·Δq + b; fit ise R_model·F'yi kullanır.
    """
    with open("params.json") as f:
        config = json.load(f)

    R, _ = load_matrices()
    n_q = R.shape[0]
    antisym = config.get("smooth_antisym_fodo", True)
    dy_true  = make_truth_dy(config, n_q, add_random=add_random)
    if add_random and config.get("dy_random_RMS", 0) > 0:
        print(f"  [UYARI] dy_random_RMS = {config['dy_random_RMS']*1e6:.0f} μm eklendi")
    truth_ks = sorted(set(h["k"] for h in config["dy_harmonics"]))
    F_full, meta_full = fodo_basis(n_q, truth_ks, antisym)
    F_k2,  meta_k2   = fodo_basis(n_q, [2],       antisym)

    # gerçek k=2
    a2t, *_ = np.linalg.lstsq(F_k2, dy_true, rcond=None)
    amp_true = np.sqrt(a2t[0]**2 + a2t[1]**2)
    phi_true = np.arctan2(a2t[1], a2t[0])

    print("=" * 70)
    print(f"  Model hatası süpürmesi   σ_b = {bpm_offset_sigma*1e6:.0f} μm   ({n_trials} deneme)")
    print(f"  Gerçek k=2: {amp_true*1e6:.2f} μm @ ∠{phi_true:.3f} rad")
    print("=" * 70)
    print(f"  {'δK/K (%)':>10} {'RMS hata (μm)':>15} {'Korelasyon':>12} "
          f"{'k=2 (μm)':>10} {'k=2 hata%':>10} {'faz Δ (rad)':>12}")
    print("  " + "-" * 75)

    for sigma_model in rel_errors:
        rng = np.random.default_rng(seed)
        rms_list, corr_list, amp_list, phi_list = [], [], [], []

        for _ in range(n_trials):
            b = rng.normal(0, bpm_offset_sigma, n_q)

            # Gerçek orbit: R_true kullanılır
            y = R @ dy_true + b

            # Model hatası: her quad'ın gradyeni σ_model göreli hatayla bilinir
            if sigma_model > 0:
                eps = rng.normal(0, sigma_model, n_q)
                R_model = R @ np.diag(1.0 + eps)
            else:
                R_model = R

            # Fourier fit: baz = truth harmonikleri
            M = R_model @ F_full
            a, *_ = np.linalg.lstsq(M, y, rcond=1e-3)
            dy_fit = F_full @ a

            rms, corr = profile_error(dy_fit, dy_true)
            rms_list.append(rms)
            corr_list.append(corr)

            idx2 = [i for i,(k,_) in enumerate(meta_full) if k==2]
            amp_list.append(np.sqrt(a[idx2[0]]**2 + a[idx2[1]]**2))
            phi_list.append(np.arctan2(a[idx2[1]], a[idx2[0]]))

        mean_rms  = np.mean(rms_list)
        mean_corr = np.mean(corr_list)
        mean_amp  = np.mean(amp_list) * 1e6
        dphi = np.abs(np.array(phi_list) - phi_true)
        dphi = np.minimum(dphi, 2*np.pi - dphi)
        mean_dphi = np.mean(dphi)
        amp_err   = abs(mean_amp - amp_true*1e6) / (amp_true*1e6) * 100

        label = f"{sigma_model*100:.2f}"
        print(f"  {label:>10} {mean_rms:>14.1f}  {mean_corr:>11.3f}  "
              f"{mean_amp:>9.2f}  {amp_err:>9.1f}  {mean_dphi:>11.3f}")

    print()
    print("  NOT: δK/K=0 → saf BPM ofset etkisi (model mükemmel)")
    print("       δK/K>0 → model hatası + BPM ofset birlikte")


def clean_orbit(R, y, candidate_ks, antisym=True,
                gain=0.2, max_iter=300, tol=1e-4):
    """CLEAN: tek orbit y = R·Δq + b üzerinde harmonik keşfi.

    Aday k kümesi verilir (hangi k'ların var olduğu bilinmez).
    Her turda artığı en çok düşüren k seçilip kesirli çıkarılır.
    b beyaz gürültü olduğu için R·F_k'ya projeksiyon yaparken doğal
    olarak bastırılır — kmod'a gerek yok.

    Döndürür: {k: a_katsayı_vektörü} birikmiş katsayılar.
    """
    n_q = R.shape[0]
    # Tüm aday bazları önceden hesapla
    bases = {}
    for k in candidate_ks:
        F_k, _ = fodo_basis(n_q, [k], antisym)
        bases[k] = (F_k, R @ F_k)        # (F_k, M_k = R·F_k)

    r = y.astype(float).copy()
    r0_norm = np.linalg.norm(r)
    accum = {k: np.zeros(2) for k in candidate_ks}

    for _ in range(max_iter):
        best_k, best_red, best_a, best_M = None, 0.0, None, None
        r_norm_sq = float(np.dot(r, r))
        for k, (F_k, M_k) in bases.items():
            a_k, *_ = np.linalg.lstsq(M_k, r, rcond=1e-3)
            red = r_norm_sq - float(np.dot(r - M_k @ a_k, r - M_k @ a_k))
            if red > best_red:
                best_red, best_k, best_a, best_M = red, k, a_k, M_k
        if best_k is None or best_red <= 0:
            break
        r -= gain * (best_M @ best_a)
        accum[best_k] += gain * best_a
        if np.linalg.norm(r) / r0_norm < tol:
            break

    return accum


def clean_test(bpm_offset_sigma=100e-6, n_trials=50, seed=0,
               candidate_ks=None, gain=0.2, max_iter=300):
    """CLEAN ile tek orbit ölçümünden k=2 çekme testi.

    Aday k kümesi oracle değil: hangi harmoniklerin var olduğu bilinmez.
    CLEAN kendi keşfeder, biz sadece k=2 sonucuna bakırız.
    """
    with open("params.json") as f:
        config = json.load(f)

    R, _ = load_matrices()
    n_q     = R.shape[0]
    antisym = config.get("smooth_antisym_fodo", True)
    dy_true = make_truth_dy(config, n_q, add_random=False)

    # k=2 gerçeği
    F_k2, _ = fodo_basis(n_q, [2], antisym)
    a2t, *_ = np.linalg.lstsq(F_k2, dy_true, rcond=None)
    amp_true = np.sqrt(a2t[0]**2 + a2t[1]**2)
    phi_true = np.arctan2(a2t[1], a2t[0])

    if candidate_ks is None:
        # Geniş aday kümesi — oracle değil
        candidate_ks = list(range(1, 13))   # k=1..12

    present_ks = sorted(set(h["k"] for h in config["dy_harmonics"]))

    print("=" * 64)
    print(f"  CLEAN tek-orbit testi   σ_b = {bpm_offset_sigma*1e6:.0f} μm   ({n_trials} deneme)")
    print(f"  Gerçek k=2: {amp_true*1e6:.2f} μm @ ∠{phi_true:.3f} rad")
    print(f"  Gerçek harmonikler: {present_ks}   (algoritmaya söylenmedi)")
    print(f"  Aday k kümesi: {candidate_ks}")
    print(f"  gain={gain}  max_iter={max_iter}")
    print("=" * 64)

    rng = np.random.default_rng(seed)
    k2_amps, k2_phis, found_ks_all = [], [], []

    for _ in range(n_trials):
        b = rng.normal(0, bpm_offset_sigma, n_q)
        y = R @ dy_true + b

        accum = clean_orbit(R, y, candidate_ks, antisym, gain, max_iter)

        # k=2 katsayısı
        a2 = accum.get(2, np.zeros(2))
        k2_amps.append(np.sqrt(a2[0]**2 + a2[1]**2))
        k2_phis.append(np.arctan2(a2[1], a2[0]))

        # Hangi k'lar baskın çıktı?
        found = sorted(k for k, a in accum.items()
                       if np.linalg.norm(a)*1e6 > 1.0)   # >1 μm eşiği
        found_ks_all.append(found)

    mean_amp  = np.mean(k2_amps) * 1e6
    std_amp   = np.std(k2_amps)  * 1e6
    dphi      = np.abs(np.array(k2_phis) - phi_true)
    dphi      = np.minimum(dphi, 2*np.pi - dphi)
    mean_dphi = np.mean(dphi)
    amp_err   = abs(mean_amp - amp_true*1e6) / (amp_true*1e6) * 100

    print(f"\n  k=2 sonucu ({n_trials} deneme ortalaması):")
    print(f"    Bulunan genlik : {mean_amp:.2f} ± {std_amp:.2f} μm")
    print(f"    Gerçek genlik  : {amp_true*1e6:.2f} μm")
    print(f"    Genlik hatası  : {amp_err:.1f}%")
    print(f"    Faz hatası     : {mean_dphi:.3f} rad")

    # Tipik bir denemede hangi k'lar bulundu?
    from collections import Counter
    all_found_flat = [k for trial in found_ks_all for k in trial]
    freq = Counter(all_found_flat)
    print(f"\n  Hangi k'lar >1 μm bulundu (kaç denemede / {n_trials}):")
    for k in sorted(freq):
        bar = '█' * int(freq[k] / n_trials * 20)
        print(f"    k={k:2d}: {freq[k]:3d}/{n_trials}  {bar}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, default=100e-6,
                   help="BPM ofset sigma (m), varsayılan=100e-6")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep", action="store_true",
                   help="Model hatası süpürmesi çalıştır")
    p.add_argument("--random", action="store_true",
                   help="dy_random_RMS ekle (Fourier bazı dışı bileşen)")
    p.add_argument("--clean", action="store_true",
                   help="CLEAN tek-orbit testi: aday k=1..12, oracle yok")
    p.add_argument("--gain", type=float, default=0.2,
                   help="CLEAN loop gain (varsayılan 0.2)")
    p.add_argument("--max-k", type=int, default=12,
                   help="CLEAN aday k üst sınırı (varsayılan 12)")
    args = p.parse_args()
    if args.clean:
        clean_test(bpm_offset_sigma=args.sigma, n_trials=args.trials,
                   seed=args.seed, gain=args.gain,
                   candidate_ks=list(range(1, args.max_k + 1)))
    elif args.sweep:
        model_error_sweep(bpm_offset_sigma=args.sigma,
                          n_trials=args.trials, seed=args.seed,
                          add_random=args.random)
    else:
        run(bpm_offset_sigma=args.sigma, n_trials=args.trials, seed=args.seed,
            add_random=args.random)
