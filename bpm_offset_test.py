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

    # ── TANI 1: sinyal vs ofset büyüklüğü ───────────────────────────────
    orb_signal = R @ dy_true
    rng_diag = np.random.default_rng(seed)
    b_sample = rng_diag.normal(0, bpm_offset_sigma, n_q)
    print(f"\n  [TANI 1] Sinyal vs ofset büyüklüğü:")
    print(f"    ‖R·dy_true‖ (orbit sinyali) = {np.linalg.norm(orb_signal)*1e6:8.1f} μm")
    print(f"    ‖b‖ (BPM ofseti, örnek)     = {np.linalg.norm(b_sample)*1e6:8.1f} μm")
    # k=2'nin tek başına orbit katkısı
    F_k2_d, _ = fodo_basis(n_q, [2], antisym)
    orb_k2 = R @ (F_k2_d @ a2t)
    print(f"    ‖R·dy(k=2)‖ (yalnız k=2)    = {np.linalg.norm(orb_k2)*1e6:8.1f} μm")

    # ── TANI 2: SADECE ofset (sinyal yok) → CLEAN ne uyduruyor? ──────────
    print(f"\n  [TANI 2] Sinyal YOK (dy_true=0), yalnız b={bpm_offset_sigma*1e6:.0f}μm:")
    rng_d2 = np.random.default_rng(seed + 999)
    b_only_k2 = []
    for _ in range(n_trials):
        b = rng_d2.normal(0, bpm_offset_sigma, n_q)
        acc = clean_orbit(R, b, candidate_ks, antisym, gain, max_iter)  # y = b sadece
        a2_b = acc.get(2, np.zeros(2))
        b_only_k2.append(np.sqrt(a2_b[0]**2 + a2_b[1]**2))
    print(f"    CLEAN'in b'den uydurduğu sahte k=2: "
          f"{np.mean(b_only_k2)*1e6:.3f} ± {np.std(b_only_k2)*1e6:.3f} μm")
    print(f"    (Eğer ~0 ise b gerçekten reddediliyor; büyükse sızıntı var)")

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
    print(f"    Bulunan genlik : {mean_amp:.3f} ± {std_amp:.3f} μm")
    print(f"    Gerçek genlik  : {amp_true*1e6:.3f} μm")
    print(f"    Genlik hatası  : {amp_err:.2f}%")
    print(f"    Faz hatası     : {mean_dphi:.3f} rad")

    # ── TANI 3: σ_b süpürmesi → hata b ile ölçekleniyor mu? ─────────────
    print(f"\n  [TANI 3] σ_b süpürmesi (k=2 std, b ile ölçekleniyor mu?):")
    print(f"    {'σ_b (μm)':>10} {'k=2 ort (μm)':>14} {'k=2 std (μm)':>14}")
    for sb in [0.0, 100e-6, 1000e-6, 10000e-6]:
        rng_s = np.random.default_rng(seed + 7)
        amps_s = []
        for _ in range(n_trials):
            b = rng_s.normal(0, sb, n_q) if sb > 0 else np.zeros(n_q)
            acc = clean_orbit(R, R @ dy_true + b, candidate_ks, antisym, gain, max_iter)
            a2s = acc.get(2, np.zeros(2))
            amps_s.append(np.sqrt(a2s[0]**2 + a2s[1]**2))
        print(f"    {sb*1e6:>10.0f} {np.mean(amps_s)*1e6:>14.3f} {np.std(amps_s)*1e6:>14.4f}")
    print(f"    (std σ_b ile orantılı artmalı; sabit ~0 kalırsa b girmiyor demektir)")

    # Tipik bir denemede hangi k'lar bulundu?
    from collections import Counter
    all_found_flat = [k for trial in found_ks_all for k in trial]
    freq = Counter(all_found_flat)
    print(f"\n  Hangi k'lar >1 μm bulundu (kaç denemede / {n_trials}):")
    for k in sorted(freq):
        bar = '█' * int(freq[k] / n_trials * 20)
        print(f"    k={k:2d}: {freq[k]:3d}/{n_trials}  {bar}")


def bpm_whiteness_test(n_trials=200, seed=0, gain=0.2, max_iter=300):
    """BPM ofseti 'beyazlık' testi.

    k=2 tahminini bozan şey b'nin toplam normu değil, yalnızca b'nin k=2
    orbit yönü v_k2 = RF_2/‖RF_2‖ üzerindeki bileşeni b_∥ = b·v_k2'dir:

        δa_k2 = b_∥ / ‖M_{k2}‖   (direct lstsq; CLEAN ≈ lstsq, γ≈1)

    burada ‖M_{k2}‖ = ‖R @ F_k2[:,0]‖ = √24 · ‖RF_2‖ ≈ 167
    (normalize edilmemiş Fourier sütunu normu — sadece ‖RF_2‖=34.1 değil!)

    Üç senaryo:
    1. b = beyaz(σ=100μm)         — referans
    2. b = A · v_k2              — saf yapısal (worst case, deterministic)
    3. b = beyaz(σ=100μm) + A·v_k2 — karma
    """
    with open("params.json") as f:
        config = json.load(f)

    R, _ = load_matrices()
    n_q     = R.shape[0]
    antisym = config.get("smooth_antisym_fodo", True)
    dy_true = make_truth_dy(config, n_q, add_random=False)

    # k=2 gerçek genlik
    F_k2, _ = fodo_basis(n_q, [2], antisym)
    a2t, *_ = np.linalg.lstsq(F_k2, dy_true, rcond=None)
    amp_true = np.sqrt(a2t[0]**2 + a2t[1]**2)
    phi_true = np.arctan2(a2t[1], a2t[0])

    # k=2 orbit yönü (BPM uzayında birim vektör)
    Fc2 = F_k2[:, 0] / np.linalg.norm(F_k2[:, 0])
    orbit_k2  = R @ Fc2
    RF2_norm  = np.linalg.norm(orbit_k2)         # ‖RF_2‖ = 34.1  (normaliz. girdi)
    M_k2_norm = np.linalg.norm(R @ F_k2[:, 0])  # ‖M_{k2}‖ = √24·‖RF_2‖ ≈ 167
    v_k2      = orbit_k2 / RF2_norm              # birim k=2 orbit vektörü

    # Direct lstsq için tam M matrisi (k=1..12)
    candidate_ks = list(range(1, 13))
    F_all, meta_all = fodo_basis(n_q, candidate_ks, antisym)
    M = R @ F_all
    idx2 = [i for i,(k,_) in enumerate(meta_all) if k == 2]

    y_signal = R @ dy_true
    rng = np.random.default_rng(seed)
    sigma_white = 100e-6

    print("=" * 70)
    print("  BPM OFSETİ BEYAZLIK TESTİ")
    print(f"  ‖RF_2‖  = {RF2_norm:.3f}  (k=2 orbit kazancı, normaliz. girdi)")
    print(f"  ‖M_k2‖  = {M_k2_norm:.3f}  (M sütun normu = √24·‖RF_2‖)")
    print(f"  k=2 gerçek genlik: {amp_true*1e6:.2f} μm")
    print("=" * 70)

    # ── Teorik limit ──────────────────────────────────────────────────────
    print("\n  [TEORİ] Saf yapısal b = A·v_k2 için δa_k2 = A/‖M_k2‖")
    print(f"  {'A (μm)':>8}  {'Teorik bias (μm)':>18}  {'Limit (10μm) için A_max':>24}")
    A_max = 10e-6 * M_k2_norm
    for A_um in [50, 100, 200, 300, int(A_max*1e6)]:
        print(f"  {A_um:>8}  {A_um/M_k2_norm:>18.2f}  {'← limit' if A_um >= int(A_max*1e6) else ''}")
    print(f"  → 10μm hedefi için b_∥ < {A_max*1e6:.0f} μm")

    # ── Senaryo 1: Beyaz b ───────────────────────────────────────────────
    print(f"\n  [SENARYO 1] b = beyaz(σ={sigma_white*1e6:.0f}μm)  —  referans")
    print(f"  Teorik lstsq: std(δa_k2) = σ/‖M_k2‖ = {sigma_white/M_k2_norm*1e6:.2f} μm")

    errs_c, errs_l = [], []
    b_par_vals = []
    for _ in range(n_trials):
        b = rng.normal(0, sigma_white, n_q)
        b_par_vals.append(abs(float(b @ v_k2)) / M_k2_norm * 1e6)  # δa_k2 = b_∥/‖M_k2‖
        y = y_signal + b
        # CLEAN
        acc = clean_orbit(R, y, candidate_ks, antisym, gain, max_iter)
        a2 = acc.get(2, np.zeros(2))
        errs_c.append(np.sqrt(a2[0]**2+a2[1]**2)*1e6 - amp_true*1e6)
        # Direct lstsq
        a_l, *_ = np.linalg.lstsq(M, y, rcond=1e-3)
        a2l = np.array([a_l[idx2[0]], a_l[idx2[1]]])
        errs_l.append(np.sqrt(a2l[0]**2+a2l[1]**2)*1e6 - amp_true*1e6)

    print(f"  b_∥/‖M_k2‖ (tahmini k=2 hata): ort {np.mean(b_par_vals):.2f} ± "
          f"{np.std(b_par_vals):.2f} μm")
    print(f"  CLEAN k=2 hatası   : {np.mean(errs_c):+.3f} ± {np.std(errs_c):.3f} μm")
    print(f"  Direct lstsq hatası: {np.mean(errs_l):+.3f} ± {np.std(errs_l):.3f} μm")
    ratio = np.std(errs_l) / np.std(errs_c) if np.std(errs_c) > 0 else np.nan
    print(f"  CLEAN bastırma faktörü vs lstsq: {ratio:.1f}×")

    # ── Senaryo 2: Saf yapısal b = A·v_k2 ────────────────────────────────
    print(f"\n  [SENARYO 2] b = A·v_k2  —  saf yapısal, deterministik")
    print(f"  {'A (μm)':>8}  {'Teorik bias':>13}  {'CLEAN hata':>12}  {'lstsq hata':>12}")
    for A_um in [0, 50, 100, 200, 300, 341]:
        A = A_um * 1e-6
        b_s = A * v_k2
        theor = A / M_k2_norm * 1e6
        y = y_signal + b_s
        # CLEAN (deterministik: tek deneme yeterli)
        acc = clean_orbit(R, y, candidate_ks, antisym, gain, max_iter)
        a2 = acc.get(2, np.zeros(2))
        c_err = np.sqrt(a2[0]**2+a2[1]**2)*1e6 - amp_true*1e6
        # Direct lstsq
        a_l, *_ = np.linalg.lstsq(M, y, rcond=1e-3)
        a2l = np.array([a_l[idx2[0]], a_l[idx2[1]]])
        l_err = np.sqrt(a2l[0]**2+a2l[1]**2)*1e6 - amp_true*1e6
        limit_str = "← 10μm limit" if abs(theor - 10.0) < 0.5 else ""
        print(f"  {A_um:>8}  {theor:>11.2f}μm  {c_err:>+10.2f}μm  {l_err:>+10.2f}μm  {limit_str}")

    # ── Senaryo 3: Karma ─────────────────────────────────────────────────
    print(f"\n  [SENARYO 3] b = beyaz(σ={sigma_white*1e6:.0f}μm) + A·v_k2  —  karma")
    print(f"  {'A (μm)':>8}  {'CLEAN ort':>12}  {'CLEAN std':>10}  {'lstsq ort':>12}  {'lstsq std':>10}")
    rng2 = np.random.default_rng(seed + 42)
    for A_um in [0, 50, 100, 200, 300]:
        A = A_um * 1e-6
        ec, el = [], []
        for _ in range(n_trials):
            b = rng2.normal(0, sigma_white, n_q) + A * v_k2
            y = y_signal + b
            acc = clean_orbit(R, y, candidate_ks, antisym, gain, max_iter)
            a2 = acc.get(2, np.zeros(2))
            ec.append(np.sqrt(a2[0]**2+a2[1]**2)*1e6 - amp_true*1e6)
            a_l, *_ = np.linalg.lstsq(M, y, rcond=1e-3)
            a2l = np.array([a_l[idx2[0]], a_l[idx2[1]]])
            el.append(np.sqrt(a2l[0]**2+a2l[1]**2)*1e6 - amp_true*1e6)
        print(f"  {A_um:>8}  {np.mean(ec):>+10.3f}μm  {np.std(ec):>8.3f}μm  "
              f"{np.mean(el):>+10.3f}μm  {np.std(el):>8.3f}μm")

    # ── Spektrum: b'nin Fourier spektrumu ────────────────────────────────
    print(f"\n  [SPEKTRUM] Beyaz b'nin k=2..8 Fourier projeksiyon gücü")
    print(f"  (‖RF_k‖ büyükse k=2 etkisi büyür)")
    print(f"  {'k':>4}  {'‖RF_k‖(norm)':>14}  {'‖M_k‖=√24·‖RF_k‖':>18}  {'σ/‖M_k‖ (μm)':>14}")
    for k_test in range(1, 9):
        Fk, _ = fodo_basis(n_q, [k_test], antisym)
        Fkc = Fk[:, 0] / np.linalg.norm(Fk[:, 0])
        RF_k_norm = np.linalg.norm(R @ Fkc)          # normalized gain
        M_k_norm  = np.linalg.norm(R @ Fk[:, 0])    # actual M column norm
        theory_std = sigma_white / M_k_norm * 1e6
        print(f"  {k_test:>4}  {RF_k_norm:>14.3f}  {M_k_norm:>18.3f}  {theory_std:>12.2f}")

    print()
    print("  ÖZET:")
    print(f"  • k=2 tahminini etkileyen yalnızca b'nin v_k2 yönündeki bileşeni")
    print(f"  • Saf yapısal b=A·v_k2: δa_k2 = A/‖M_k2‖ = A/{M_k2_norm:.1f}")
    print(f"    → 10μm hata için A < {M_k2_norm*10:.0f} μm gerekir (tipik 300μm << bu sınır)")
    print(f"  • Beyaz b(σ): beklenen lstsq hata std ≈ σ/‖M_k2‖ = σ/{M_k2_norm:.1f}")
    print(f"    CLEAN ≈ lstsq (γ≈{ratio:.1f}×); CLEAN avantajı: oracle gerektirmez")
    print(f"  • 300μm tamamen k=2-orbit-hizalı ofset: bias = 300/{M_k2_norm:.1f} "
          f"= {300/M_k2_norm:.2f}μm (10μm hedefin altında)")


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
    p.add_argument("--whiteness", action="store_true",
                   help="BPM ofseti beyazlık testi: yapısal vs beyaz b")
    p.add_argument("--gain", type=float, default=0.2,
                   help="CLEAN loop gain (varsayılan 0.2)")
    p.add_argument("--max-k", type=int, default=12,
                   help="CLEAN aday k üst sınırı (varsayılan 12)")
    args = p.parse_args()
    if args.clean:
        clean_test(bpm_offset_sigma=args.sigma, n_trials=args.trials,
                   seed=args.seed, gain=args.gain,
                   candidate_ks=list(range(1, args.max_k + 1)))
    elif args.whiteness:
        bpm_whiteness_test(n_trials=args.trials, seed=args.seed,
                           gain=args.gain, max_iter=300)
    elif args.sweep:
        model_error_sweep(bpm_offset_sigma=args.sigma,
                          n_trials=args.trials, seed=args.seed,
                          add_random=args.random)
    else:
        run(bpm_offset_sigma=args.sigma, n_trials=args.trials, seed=args.seed,
            add_random=args.random)
