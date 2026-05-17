"""
spectral_inversion.py — DFT/FFT tabanlı quad hizalama geri çatımı

Dört aşamalı analiz akışı:
    Aşama A : İdeal geri çatım üst-sınır testi (sim yok)
    Aşama B : Kondisyon sayısı haritası — R₁, R₂, ΔR mod bazlı (sim yok)
    Aşama C : İki-kmod rekonstrüksiyonu (simülasyon = "gerçek makine")
    Aşama D : Gürbüzlük taraması (tilt, BPM gürültüsü, β hatası, ofset)

Her aşama bağımsız çalıştırılabilir. fodo_lattice.py'nin doğru Twiss verdiği
varsayılır (Qx = 2.6824 kalibre, Qy = 2.3617 fizikten).
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # ekran gerekmez; PNG çıktısı
import matplotlib.pyplot as plt

from fodo_lattice import (
    compute_twiss_at_quads,
    signed_KL,
    build_response_matrix,
    fft_invert,
    direct_invert,
    magic_momentum_proton,
    compute_Brho,
)


# =============================================================================
# Yardımcı: belirli bir gradient g için R inşa et
# =============================================================================
def build_R_for_gradient(config, g, plane, K_x_arc_x=None):
    """
    Verilen kuadrupol gradyanı g [T/m] için yanıt matrisini inşa eder.

    Twiss parametreleri (β, φ, Q) örgüden gelir — K_x_arc (yatay arc odaklaması)
    bir kez kalibre edilir ve farklı g değerleri için sabit tutulur (arc
    geometrisi g'den bağımsız). KL ise g ile orantılı.
    """
    # config'in g1 alanını geçici olarak güncelle (Twiss için K_abs hesabı)
    cfg = dict(config)
    cfg['g1'] = g

    beta, phi, Q = compute_twiss_at_quads(cfg, plane, K_x_arc=K_x_arc_x)
    KL = signed_KL(cfg, plane)
    R = build_response_matrix(beta, phi, Q, KL)
    return R, beta, phi, Q, KL


# =============================================================================
# Aşama A — İdeal geri çatım üst-sınır testi
# =============================================================================
def stage_A_ideal(config, plane='y', N_real=20, sigma_q=100e-6, seed=0,
                  verbose=True):
    """
    Aşama A: Eğer model = gerçek makine olsaydı geri çatım ne kadar iyi olurdu?

    Adımlar:
      1. Analitik R hesapla
      2. Rastgele Δq üret (~100 μm RMS, Gauss)
      3. y = R·Δq (sim yok, doğrudan model)
      4. FFT_invert ile Δq̂ kurtar; np.linalg.solve ile karşılaştır
      5. RMS hata ve korelasyon raporla

    Beklenti: Direct çözüm makine hassasiyetinde (< 1e-12 m). FFT, ideal FODO
    için block-circulant yaklaşımı nedeniyle az ama ölçülebilir hata bırakır
    (QF ve QD farklı β'da olduğu için R tam sirkülant değil).
    """
    if plane == 'x':
        # x için K_x_arc'ı bir kez kalibre et
        from fodo_lattice import calibrate_K_x_arc
        K_x_arc = calibrate_K_x_arc(config)
    else:
        K_x_arc = None

    beta, phi, Q = compute_twiss_at_quads(config, plane, K_x_arc=K_x_arc)
    KL = signed_KL(config, plane)
    R = build_response_matrix(beta, phi, Q, KL)
    N = len(beta)

    rng = np.random.default_rng(seed)
    err_fft     = np.empty(N_real)
    err_direct  = np.empty(N_real)
    corr_fft    = np.empty(N_real)
    corr_direct = np.empty(N_real)

    for i in range(N_real):
        dq_true = rng.normal(0.0, sigma_q, N)
        y       = R @ dq_true
        dq_fft    = fft_invert(y, beta, phi, Q, KL)
        dq_direct = direct_invert(R, y)
        err_fft[i]     = np.sqrt(np.mean((dq_fft    - dq_true) ** 2))
        err_direct[i]  = np.sqrt(np.mean((dq_direct - dq_true) ** 2))
        corr_fft[i]    = np.corrcoef(dq_fft,    dq_true)[0, 1]
        corr_direct[i] = np.corrcoef(dq_direct, dq_true)[0, 1]

    if verbose:
        print(f"\n[Aşama A] Düzlem={plane}, N_quad={N}, "
              f"σ_Δq={sigma_q*1e6:.0f} μm, {N_real} realizasyon")
        print(f"  Q = {Q:.6f},  κ(R) = {np.linalg.cond(R):.3e}")
        print(f"  Direct  geri dönüşüm RMS : {err_direct.mean()*1e6:.3e} μm   "
              f"(corr = {corr_direct.mean():.6f})")
        print(f"  FFT     geri dönüşüm RMS : {err_fft.mean()*1e6:.3e} μm   "
              f"(corr = {corr_fft.mean():.6f})")

    return {
        'plane': plane, 'Q': Q, 'beta': beta, 'phi': phi, 'KL': KL, 'R': R,
        'err_fft': err_fft, 'err_direct': err_direct,
        'corr_fft': corr_fft, 'corr_direct': corr_direct,
    }


# =============================================================================
# Aşama B — Kondisyon sayısı haritası
# =============================================================================
def stage_B_condition_map(config, plane='y', g_pert_frac=0.02,
                          out_dir='.', verbose=True):
    """
    Aşama B: R₁, R₂ ve ΔR matrislerinin mod bazlı kondisyonunu karşılaştırır.

    Beta-normalize edilmiş yanıt operatörünün özdeğerleri = ilk satırının
    DFT'si. |λ_k| → mod k'nın "kazancı"; |λ_k|⁻¹ → o mod için gürültü
    yükseltme faktörü.

    Adımlar:
      1. g_nom = config['g1'],  g_pert = (1+g_pert_frac)·g_nom
      2. R₁ = R(g_nom),  R₂ = R(g_pert),  ΔR = R₁ − R₂
      3. Her matris için β-normalize ilk satır → λ_k = FFT(ilk_satır)
      4. log10|λ_k|⁻¹ grafiği (k modu vs. zorluk)
      5. Global κ değerleri (SVD oranı) raporla

    Beklenti: ΔR mod gücü, R₁/R₂'nin "farkı" kadar küçük → ΔR çok daha
    zayıf koşullu modlara sahip.
    """
    if plane == 'x':
        from fodo_lattice import calibrate_K_x_arc
        K_x_arc = calibrate_K_x_arc(config)
    else:
        K_x_arc = None

    g_nom  = config['g1']
    g_pert = g_nom * (1.0 + g_pert_frac)

    R1, beta1, phi1, Q1, KL1 = build_R_for_gradient(config, g_nom,  plane, K_x_arc)
    R2, beta2, phi2, Q2, KL2 = build_R_for_gradient(config, g_pert, plane, K_x_arc)
    dR = R1 - R2
    N  = len(beta1)

    # β-normalize ilk satır → modal özdeğerler
    def modal_eigs(M, beta_i, beta_j):
        sqi = np.sqrt(beta_i); sqj = np.sqrt(beta_j)
        # M'in β-normalize hali: M / (sqrt(β_i)·sqrt(β_j))
        # Bu hal, sirkülant yaklaşımına en yakındır
        M_norm = M / np.outer(sqi, sqj)
        return np.fft.fft(M_norm[0, :])

    lam_R1 = modal_eigs(R1, beta1, beta1)
    lam_R2 = modal_eigs(R2, beta2, beta2)
    lam_dR = modal_eigs(dR, beta1, beta1)

    # Genel kondisyon sayıları (SVD)
    kappa = {
        'R1': np.linalg.cond(R1),
        'R2': np.linalg.cond(R2),
        'dR': np.linalg.cond(dR),
    }

    if verbose:
        print(f"\n[Aşama B] Düzlem={plane},  g_pert/g_nom-1 = {g_pert_frac:+.1%}")
        print(f"  Q1 = {Q1:.6f},  Q2 = {Q2:.6f}")
        print(f"  κ(R1) = {kappa['R1']:.3e}")
        print(f"  κ(R2) = {kappa['R2']:.3e}")
        print(f"  κ(ΔR) = {kappa['dR']:.3e}    ← ΔR yöntemi bunu kullanır")
        print(f"  Oran  κ(ΔR)/κ(R1) ≈ {kappa['dR']/kappa['R1']:.1f}")

    # Mod bazlı plot
    fig, ax = plt.subplots(figsize=(9, 5))
    k = np.arange(N)
    ax.semilogy(k, 1.0/np.abs(lam_R1), 'o-', label='|λ_k|⁻¹  R₁',  ms=4)
    ax.semilogy(k, 1.0/np.abs(lam_R2), 's-', label='|λ_k|⁻¹  R₂',  ms=4)
    ax.semilogy(k, 1.0/np.abs(lam_dR), '^-', label='|λ_k|⁻¹  ΔR',  ms=4, color='C3')
    ax.set_xlabel('Fourier mod indeksi k')
    ax.set_ylabel('Modal kondisyon  |λ_k|⁻¹  [m / m]')
    ax.set_title(f"Aşama B — Modal kondisyon haritası (düzlem={plane}, "
                 f"Δg/g={g_pert_frac:+.1%})")
    ax.grid(True, which='both', alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, f'stage_B_condition_{plane}.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    if verbose:
        print(f"  Grafik: {out_path}")

    return {
        'plane': plane, 'Q1': Q1, 'Q2': Q2,
        'lam_R1': lam_R1, 'lam_R2': lam_R2, 'lam_dR': lam_dR,
        'kappa': kappa,
        'R1': R1, 'R2': R2, 'dR': dR,
        'beta': beta1, 'phi': phi1, 'KL1': KL1, 'KL2': KL2,
    }


# =============================================================================
# Çalıştırma
# =============================================================================
if __name__ == "__main__":
    with open("params.json", "r") as f:
        config = json.load(f)

    print("=" * 64)
    print("spectral_inversion.py — DFT/FFT tabanlı quad geri çatım")
    print("=" * 64)

    # Aşama A: her iki düzlem için ideal test
    for plane in ['x', 'y']:
        stage_A_ideal(config, plane=plane, N_real=20)

    # Aşama B: kondisyon haritası
    for plane in ['x', 'y']:
        stage_B_condition_map(config, plane=plane, g_pert_frac=0.02)
