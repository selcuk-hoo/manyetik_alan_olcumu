#!/usr/bin/env python3
"""
analytic_kmod.py

K-modülasyon tepki matrisini FODO Twiss parametrelerinden analitik olarak
hesaplar ve quad hizalama geri çatımı yapar.

Simülasyon tabanlı build_response_matrix.py'den farkı:
  Her quad için birim misalignment koşumu YAPILMAZ.
  Bunun yerine β(s) ve φ(s) FODO transfer matrislerinden hesaplanır:

    R[i,j] = √(βᵢβⱼ) / (2sin(πQ)) · KL_j · cos(|φᵢ-φⱼ| - πQ)

İki gradyan konfigürasyonu (g_nom, g_pert) için ΔR = R₂ - R₁ hesaplanır;
ölçülen Δy farkından dy geri çatılır.

Hata senaryoları (modele dahil edilmeyen, "gerçek makine"de var):
  - quad_tilt  : skew bileşeni → x-y kuplajı
  - dipole_tilt: deflektör tilti → COD kirliliği
  - beta_error : gradyan hatası → gerçek β tasarım β'sından sapıyor

Kullanım:
  python analytic_kmod.py
"""

import json
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────
# Fiziksel sabitler
# ─────────────────────────────────────────────────────

def compute_Brho(config):
    """Protonun magic momentumda Bρ [T·m]"""
    M2  = 0.938272046   # proton kütlesi [GeV/c²]
    AMU = 1.792847356   # (g-2)/2 anomalous moment
    p_magic = M2 / np.sqrt(AMU)          # [GeV/c]
    return p_magic / 0.299792458          # Bρ [T·m]


# ─────────────────────────────────────────────────────
# Transfer matrisleri
# ─────────────────────────────────────────────────────

def quad_matrix(K, L):
    """2×2 transfer matrisi, kalın quadrupol. K>0 foklayıcı."""
    if abs(K) < 1e-14:
        return np.array([[1.0, L], [0.0, 1.0]])
    if K > 0:
        sq = np.sqrt(K)
        return np.array([[np.cos(sq*L),      np.sin(sq*L)/sq],
                         [-sq*np.sin(sq*L),   np.cos(sq*L)   ]])
    else:
        sq = np.sqrt(-K)
        return np.array([[np.cosh(sq*L),      np.sinh(sq*L)/sq],
                         [ sq*np.sinh(sq*L),  np.cosh(sq*L)   ]])


def drift_matrix(L):
    return np.array([[1.0, L], [0.0, 1.0]])


def propagate_twiss(M, beta, alpha):
    """Twiss parametrelerini M matrisiyle ilerletir."""
    gamma = (1.0 + alpha**2) / beta
    b = M[0,0]**2*beta - 2*M[0,0]*M[0,1]*alpha + M[0,1]**2*gamma
    a = (-M[1,0]*M[0,0]*beta
         + (M[0,0]*M[1,1] + M[0,1]*M[1,0])*alpha
         - M[0,1]*M[1,1]*gamma)
    return b, a


def phase_step(M, beta, alpha):
    """
    Mevcut Twiss (β,α) verilen bir elemandan geçerken oluşan faz artışı.
    Arctan2 formülü; dallanma kesimi için düzeltme uygulanır.
    """
    denom = M[0,0]*beta - M[0,1]*alpha
    dphi = np.arctan2(M[0,1], denom)
    if dphi <= 0.0:
        dphi += np.pi
    return dphi


# ─────────────────────────────────────────────────────
# FODO Twiss hesabı
# ─────────────────────────────────────────────────────

def compute_twiss_at_quads(config, g_override=None, plane='y'):
    """
    FODO halkasında her quad girişinde Twiss parametreleri (β, φ) hesaplar.

    Dizi sırası: QF0, QD0, QF1, QD1, ..., QF(n-1), QD(n-1)

    Hücre yapısı (C++'daki integrator.cpp ile uyumlu):
      elem2=QF  elem6=QD konumlarında BPM.
      ARC deflektörleri (uzunluk L_def) dikey düzlemde drift olarak
      modellenir (elektrik alan yalnızca radyal, dikey Twiss'e katkısı ihmal).

    plane : 'y' (dikey) veya 'x' (radyal/yatay)
    """
    nFODO = int(config['nFODO'])
    L_q   = float(config['quadLen'])
    L_d   = float(config['driftLen'])
    R0    = float(config['R0'])
    g     = g_override if g_override is not None else float(config.get('g1', 0.21))

    Brho  = compute_Brho(config)
    K_abs = abs(g) / Brho          # [1/m²]

    # ARC (deflektör) uzunluğu: C++ aynı formülü kullanıyor
    L_def = np.pi * R0 / nFODO     # arc length per deflector [m]

    # Efektif drift: ARC1 (L_def) + drift (L_d) = sonraki quad'a kadar
    # Hücre (QF girişinden QF girişine):
    #   QF(L_q) → drift(L_d) → ARC2(L_def) → drift(L_d) → QD(L_q)
    #           → drift(L_d) → ARC1(L_def) → drift(L_d) → QF(...)
    # Aralar kombine edilerek:
    L_mid  = 2.0*L_d + L_def     # QF çıkışı → QD girişi arası efektif drift
    L_wrap = 2.0*L_d + L_def     # QD çıkışı → sonraki QF girişi arası

    # Düzleme göre odaklama işareti
    if plane == 'y':
        K_QF, K_QD = -K_abs, +K_abs   # QF defokalize, QD fokalize (dikey)
    else:
        K_QF, K_QD = +K_abs, -K_abs   # QF fokalize, QD defokalize (radyal)

    M_QF  = quad_matrix(K_QF, L_q)
    M_QD  = quad_matrix(K_QD, L_q)
    M_mid = drift_matrix(L_mid)
    M_end = drift_matrix(L_wrap)

    # Tek hücre matrisi: QF girişinden QF girişine
    # M_cell = M_end · M_QD · M_mid · M_QF
    M_cell = M_end @ M_QD @ M_mid @ M_QF

    # Periyodik Twiss çözümü
    cos_mu = (M_cell[0,0] + M_cell[1,1]) / 2.0
    if abs(cos_mu) > 1.0 - 1e-9:
        cos_mu = np.clip(cos_mu, -1.0, 1.0)
    mu = np.arccos(cos_mu)         # hücre faz artışı [rad]
    Q  = nFODO * mu / (2.0*np.pi)  # betatron tune

    sin_mu = np.sin(mu)
    if abs(sin_mu) < 1e-12:
        raise ValueError(f"Dejenere FODO ({plane}): sin(μ)≈0, μ={np.degrees(mu):.2f}°")

    beta0  = M_cell[0,1] / sin_mu
    alpha0 = (M_cell[0,0] - M_cell[1,1]) / (2.0*sin_mu)

    # Her quad girişinde β ve φ kaydet
    n_q      = 2 * nFODO
    beta_arr = np.zeros(n_q)
    phi_arr  = np.zeros(n_q)

    beta  = beta0
    alpha = alpha0
    phi   = 0.0

    for k in range(nFODO):
        # ── QF girişi (BPM) ──────────────────────────────────
        beta_arr[2*k]   = beta
        phi_arr[2*k]    = phi

        # QF içinden geç
        dphi  = phase_step(M_QF, beta, alpha)
        beta, alpha = propagate_twiss(M_QF, beta, alpha)
        phi  += dphi

        # QF çıkışı → QD girişi (efektif drift)
        dphi  = phase_step(M_mid, beta, alpha)
        beta, alpha = propagate_twiss(M_mid, beta, alpha)
        phi  += dphi

        # ── QD girişi (BPM) ──────────────────────────────────
        beta_arr[2*k+1] = beta
        phi_arr[2*k+1]  = phi

        # QD içinden geç
        dphi  = phase_step(M_QD, beta, alpha)
        beta, alpha = propagate_twiss(M_QD, beta, alpha)
        phi  += dphi

        # QD çıkışı → sonraki QF girişi
        dphi  = phase_step(M_end, beta, alpha)
        beta, alpha = propagate_twiss(M_end, beta, alpha)
        phi  += dphi

    return beta_arr, phi_arr, Q


# ─────────────────────────────────────────────────────
# Analitik tepki matrisi
# ─────────────────────────────────────────────────────

def signed_KL(nFODO, K_abs, L_q, plane):
    """
    Her quad için işaretli K·L kick büyüklüğü.

    Dikey (dy): QF quadrupol y'de defokalize → +Δy → aşağı kick → KL = -K·L
                QD quadrupol y'de fokalize   → +Δy → yukarı kick → KL = +K·L
    Radyal (dx): QF x'de fokalize  → +Δx → dışarı kick → KL = +K·L
                 QD x'de defokalize → +Δx → içeri kick  → KL = -K·L
    """
    KL = np.zeros(2 * nFODO)
    for k in range(nFODO):
        if plane == 'y':
            KL[2*k]   = -K_abs * L_q   # QF
            KL[2*k+1] = +K_abs * L_q   # QD
        else:
            KL[2*k]   = +K_abs * L_q   # QF
            KL[2*k+1] = -K_abs * L_q   # QD
    return KL


def build_R_analytic(beta, phi, Q, KL):
    """
    R[i,j] = √(βᵢ·βⱼ) / (2sin(πQ)) · KL_j · cos(|φᵢ-φⱼ| - πQ)

    βᵢ, φᵢ : BPM i konumunda Twiss parametreleri
    KL[j]   : quad j'nin işaretli K·L değeri
    """
    n = len(beta)
    denom = 2.0 * np.sin(np.pi * Q)
    sqrt_b = np.sqrt(beta)
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dphi = abs(phi[i] - phi[j])
            R[i, j] = (sqrt_b[i] * sqrt_b[j] / denom
                       * KL[j]
                       * np.cos(dphi - np.pi * Q))
    return R


def build_analytic_dR(config, g_nom, g_pert, plane='y'):
    """İki optik konfigürasyondan analitik ΔR matrisi."""
    Brho  = compute_Brho(config)
    nFODO = int(config['nFODO'])
    L_q   = float(config['quadLen'])

    beta1, phi1, Q1 = compute_twiss_at_quads(config, g_nom,  plane)
    beta2, phi2, Q2 = compute_twiss_at_quads(config, g_pert, plane)

    KL1 = signed_KL(nFODO, abs(g_nom)  / Brho, L_q, plane)
    KL2 = signed_KL(nFODO, abs(g_pert) / Brho, L_q, plane)

    R1 = build_R_analytic(beta1, phi1, Q1, KL1)
    R2 = build_R_analytic(beta2, phi2, Q2, KL2)
    return R2 - R1, R1, R2


# ─────────────────────────────────────────────────────
# Geri çatım yardımcıları
# ─────────────────────────────────────────────────────

def print_results(label, true_vals, reconstructed):
    err  = reconstructed - true_vals
    corr = np.corrcoef(true_vals, reconstructed)[0, 1] if np.std(true_vals) > 0 else float('nan')
    print(f"  {label:30s}  hata RMS={np.std(err)*1e6:8.2f} um   korelasyon={corr:.6f}")


def reconstruct(dR, delta):
    """SVD ile geri çatım (truncated SVD ile regularize edilmiş)."""
    U, s, Vt = np.linalg.svd(dR)
    # Koşul sayısı çok büyükse küçük singular değerleri sıfırla
    threshold = s[0] * 1e-10
    s_inv = np.where(s > threshold, 1.0/s, 0.0)
    return Vt.T @ (s_inv * (U.T @ delta))


# ─────────────────────────────────────────────────────
# Ana program
# ─────────────────────────────────────────────────────

def main():
    os.chdir(BASE)
    with open("params.json") as f:
        config = json.load(f)

    nFODO = int(config['nFODO'])
    n_q   = 2 * nFODO
    L_q   = float(config['quadLen'])
    Brho  = compute_Brho(config)

    # Gradyan konfigürasyonları (uniform k-mod modu)
    g_nom  = float(config.get('g1', 0.21))
    g_pert = g_nom * 1.02

    print("=" * 65)
    print("Analitik FODO Twiss")
    print("=" * 65)

    beta_y, phi_y, Q_y = compute_twiss_at_quads(config, g_nom, 'y')
    beta_x, phi_x, Q_x = compute_twiss_at_quads(config, g_nom, 'x')

    print(f"  g_nom = {g_nom:.4f} T/m   Bρ = {Brho:.4f} T·m   K = {g_nom/Brho:.5f} m⁻²")
    print(f"  Tune  Q_y = {Q_y:.4f}   Q_x = {Q_x:.4f}")
    print(f"  β_y : min={beta_y.min():.2f} m   max={beta_y.max():.2f} m")
    print(f"  β_x : min={beta_x.min():.2f} m   max={beta_x.max():.2f} m")

    # Analitik ΔR matrisleri
    dR_dy, R1_dy, R2_dy = build_analytic_dR(config, g_nom, g_pert, 'y')
    dR_dx, R1_dx, R2_dx = build_analytic_dR(config, g_nom, g_pert, 'x')

    print(f"\n  kappa(dR_dy) = {np.linalg.cond(dR_dy):.3e}")
    print(f"  kappa(dR_dx) = {np.linalg.cond(dR_dx):.3e}")

    # ─── Simülasyon R matrisleriyle karşılaştırma ───────────────────
    sim_files = ["R_dy_1.npy", "R_dx_1.npy"]
    if all(os.path.exists(f) for f in sim_files):
        print("\n" + "=" * 65)
        print("Simülasyon R matrisiyle karşılaştırma (R_dy_1, R_dx_1)")
        print("=" * 65)
        R_sim_dy = np.load("R_dy_1.npy") * 1e-3   # mm/m → m/m
        R_sim_dx = np.load("R_dx_1.npy") * 1e-3   # mm/m → m/m

        corr_dy = np.corrcoef(R1_dy.ravel(), R_sim_dy.ravel())[0, 1]
        corr_dx = np.corrcoef(R1_dx.ravel(), R_sim_dx.ravel())[0, 1]
        rel_dy  = np.std(R1_dy - R_sim_dy) / (np.std(R_sim_dy) + 1e-30)
        rel_dx  = np.std(R1_dx - R_sim_dx) / (np.std(R_sim_dx) + 1e-30)
        print(f"  R_dy: korelasyon={corr_dy:.6f}   bağıl fark RMS={rel_dy:.4f}")
        print(f"  R_dx: korelasyon={corr_dx:.6f}   bağıl fark RMS={rel_dx:.4f}")
    else:
        print("\n  (R_dy_1.npy bulunamadı — simülasyon karşılaştırması atlanıyor)")

    # ─── Rastgele hatalar ────────────────────────────────────────────
    dy_max    = float(config.get('quad_random_dy_max', 1e-4))
    dx_max    = float(config.get('quad_random_dx_max', 1e-4))
    quad_seed = int(config.get('quad_random_seed', 42))
    rng = np.random.default_rng(seed=quad_seed)
    dy_true = rng.uniform(-dy_max, dy_max, n_q)
    dx_true = rng.uniform(-dx_max, dx_max, n_q)

    # Tilt ve BPM hata parametreleri
    sigma_noise  = float(config.get('bpm_noise_sigma',  0.0))
    sigma_offset = float(config.get('bpm_offset_sigma', 0.0))
    offset_seed  = int(config.get('bpm_offset_seed', 55))
    d_tilt_max   = float(config.get('dipole_random_tilt_max', 0.0))
    q_tilt_max   = float(config.get('quad_random_tilt_max',   0.0))
    d_tilt_seed  = int(config.get('dipole_random_seed', 43))
    q_tilt_seed  = int(config.get('quad_random_tilt_seed', 44))

    rng_off = np.random.default_rng(seed=offset_seed)
    bpm_off_y = rng_off.normal(0, sigma_offset, n_q) if sigma_offset > 0 else np.zeros(n_q)
    bpm_off_x = rng_off.normal(0, sigma_offset, n_q) if sigma_offset > 0 else np.zeros(n_q)

    rng_dt = np.random.default_rng(seed=d_tilt_seed)
    dipole_tilt = (rng_dt.uniform(-d_tilt_max, d_tilt_max, n_q)
                   if d_tilt_max > 0 else np.zeros(n_q))

    rng_qt = np.random.default_rng(seed=q_tilt_seed)
    quad_tilt = (rng_qt.uniform(-q_tilt_max, q_tilt_max, n_q)
                 if q_tilt_max > 0 else np.zeros(n_q))

    print(f"\n  dy_max={dy_max*1e3:.3f} mm  dx_max={dx_max*1e3:.3f} mm")
    if d_tilt_max > 0: print(f"  dipole_tilt_max={d_tilt_max*1e3:.3f} mrad")
    if q_tilt_max  > 0: print(f"  quad_tilt_max={q_tilt_max*1e3:.3f} mrad")
    if sigma_noise  > 0: print(f"  BPM noise σ={sigma_noise*1e6:.1f} μm")
    if sigma_offset > 0: print(f"  BPM offset σ={sigma_offset*1e6:.1f} μm (farkta iptal)")

    # ─── "Gerçek makine" simülasyonu ────────────────────────────────
    # Simülasyon kütüphanesi mevcutsa kullan, yoksa analitik R ile taklit et
    try:
        from build_response_matrix import setup_fields, run_sim

        rng_noise = np.random.default_rng(seed=99)

        def meas(g, dG=None):
            alanlar, state0 = setup_fields(config, g1_override=g)
            x, y = run_sim(alanlar, state0, config, dy_true, dx_true,
                           dipole_tilt=dipole_tilt, quad_tilt=quad_tilt,
                           quad_dG=dG)
            # run_sim → read_cod_quads → cod_data.txt → mm; analitik R m/m biriminde
            y_m = y * 1e-3   # mm → m
            x_m = x * 1e-3   # mm → m
            if sigma_noise > 0:
                y_m += rng_noise.normal(0, sigma_noise, n_q)
                x_m += rng_noise.normal(0, sigma_noise, n_q)
            if sigma_offset > 0:
                y_m += bpm_off_y
                x_m += bpm_off_x
            return x_m, y_m

        print("\nSimülasyon koşumları yapılıyor...")
        x1, y1 = meas(g_nom)
        x2, y2 = meas(g_pert)
        sim_available = True
        print("Koşumlar tamamlandı.")

    except Exception as e:
        print(f"\n  Simülasyon yüklenemedi ({e})")
        print("  → Analitik 'forward model' kullanılıyor (kontrol testi)")
        # Analitik R ile sentetik ölçüm oluştur
        KL_y = signed_KL(nFODO, g_nom/Brho, L_q, 'y')
        KL_x = signed_KL(nFODO, g_nom/Brho, L_q, 'x')
        y1 = R1_dy @ dy_true + bpm_off_y
        y2 = R2_dy @ dy_true + bpm_off_y
        x1 = R1_dx @ dx_true + bpm_off_x
        x2 = R2_dx @ dx_true + bpm_off_x
        sim_available = False

    delta_y = y2 - y1
    delta_x = x2 - x1

    # Sinyal kalitesi
    ideal_y = dR_dy @ dy_true
    ideal_x = dR_dx @ dx_true
    print(f"\n  Sinyal (ideal ΔR·dy)  RMS = {np.std(ideal_y)*1e6:.1f} μm")
    print(f"  Gerçek Δy             RMS = {np.std(delta_y)*1e6:.1f} μm")
    kirli_y = np.std(delta_y - ideal_y)*1e6
    print(f"  Kirlilik (tilt+gürültü) RMS = {kirli_y:.1f} μm")

    # ─── Geri çatım ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Analitik ΔR ile geri çatım sonuçları")
    print("=" * 65)

    dy_an = reconstruct(dR_dy, delta_y)
    dx_an = reconstruct(dR_dx, delta_x)
    print_results("dy (analitik ΔR)", dy_true, dy_an)
    print_results("dx (analitik ΔR)", dx_true, dx_an)

    # Simülasyon ΔR varsa karşılaştır
    if all(os.path.exists(f) for f in ["dR_dy.npy", "dR_dx.npy"]):
        # build_response_matrix.py cod_data.txt'i mm cinsinden okur, delta_q m cinsinden →
        # dR_dy.npy birimi mm/m; analitik sinyal m cinsinde → 1e-3 ile m/m'e çevir.
        dR_sim_y = np.load("dR_dy.npy") * 1e-3
        dR_sim_x = np.load("dR_dx.npy") * 1e-3
        dy_sim = reconstruct(dR_sim_y, delta_y)
        dx_sim = reconstruct(dR_sim_x, delta_x)
        print()
        print_results("dy (simülasyon ΔR)", dy_true, dy_sim)
        print_results("dx (simülasyon ΔR)", dx_true, dx_sim)

    # ─── Self-consistency testi ─────────────────────────────────────
    # Analitik forward model ile: delta_y = dR @ dy_true → mükemmel geri çatım
    print("\n" + "=" * 65)
    print("Self-consistency testi (analitik forward model)")
    print("=" * 65)
    dy_self = reconstruct(dR_dy, dR_dy @ dy_true)
    dx_self = reconstruct(dR_dx, dR_dx @ dx_true)
    print_results("dy (öz-tutarlılık)", dy_true, dy_self)
    print_results("dx (öz-tutarlılık)", dx_true, dx_self)

    # ─── Beta hatası senaryosu ───────────────────────────────────────
    # Gerçek halka g_err gradyanıyla çalışıyor (tasarım g_nom'dan sapıyor).
    # Analitik ΔR tasarım β'sını (g_nom) kullanıyor → model hatası → bozuk geri çatım.
    # Forward model: analitik R(g_err) ile — hızlı, simülasyon gerektirmez.
    print("\n" + "=" * 65)
    print("Beta-fonksiyonu hatası etkisi (analitik forward, tasarım ΔR sabit)")
    print("=" * 65)
    print("  (Gerçek halka g_err'de, model hâlâ g_nom β'sını kullanıyor)")
    print(f"  {'g_hata':>8}  {'dy hata RMS':>12}  {'corr':>8}  {'dx hata RMS':>12}  {'corr':>8}")

    for beta_err_pct in [0, 1, 2, 5, 10, 20]:
        g_err = g_nom * (1.0 + beta_err_pct / 100.0)
        _, R1e_y, R2e_y = build_analytic_dR(config, g_err, g_err*1.02, 'y')
        _, R1e_x, R2e_x = build_analytic_dR(config, g_err, g_err*1.02, 'x')
        dy_e = (R2e_y - R1e_y) @ dy_true
        dx_e = (R2e_x - R1e_x) @ dx_true
        dy_rec = reconstruct(dR_dy, dy_e)
        dx_rec = reconstruct(dR_dx, dx_e)
        ey = dy_rec - dy_true
        ex = dx_rec - dx_true
        cy = np.corrcoef(dy_true, dy_rec)[0,1] if np.std(dy_true) > 0 else float('nan')
        cx = np.corrcoef(dx_true, dx_rec)[0,1] if np.std(dx_true) > 0 else float('nan')
        print(f"  {beta_err_pct:+7d}%  {np.std(ey)*1e6:10.2f} μm  {cy:8.4f}  "
              f"{np.std(ex)*1e6:10.2f} μm  {cx:8.4f}")

    # ─── Kayıt ──────────────────────────────────────────────────────
    np.savez("analytic_kmod_result.npz",
             beta_y=beta_y, phi_y=phi_y, Q_y=np.array([Q_y]),
             beta_x=beta_x, phi_x=phi_x, Q_x=np.array([Q_x]),
             dR_dy=dR_dy, dR_dx=dR_dx,
             dy_true=dy_true, dy_analytic=dy_an,
             dx_true=dx_true, dx_analytic=dx_an)
    print("\nSonuçlar 'analytic_kmod_result.npz' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()
