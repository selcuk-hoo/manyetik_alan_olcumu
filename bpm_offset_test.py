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


def make_truth_dy(config, n_q=48):
    """params.json'dan gerçek quad misalignment vektörü üret."""
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

def run(bpm_offset_sigma=100e-6, n_trials=50, seed=0):
    print("=" * 64)
    print(f"  BPM ofset testi   σ_b = {bpm_offset_sigma*1e6:.0f} μm   ({n_trials} deneme)")
    print("=" * 64)

    with open("params.json") as f:
        config = json.load(f)

    R, dR = load_matrices()
    n_q = R.shape[0]   # 48
    antisym = config.get("smooth_antisym_fodo", True)

    # gerçek misalignment
    dy_true = make_truth_dy(config, n_q)
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


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, default=100e-6,
                   help="BPM ofset sigma (m), varsayılan=100e-6")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(bpm_offset_sigma=args.sigma, n_trials=args.trials, seed=args.seed)
