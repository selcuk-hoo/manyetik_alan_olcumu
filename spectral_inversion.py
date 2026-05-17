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
# Simülasyon arayüzü: belirli g ve Δq ile koş, BPM verilerini al
# =============================================================================
def _run_simulation_with(config, g_value, dy_arr, dx_arr,
                         t_end=1e-4, dt=None, return_steps=1000,
                         quad_tilt_arr=None):
    """
    Simülasyonu belirli kuadrupol gradyanı g_value ve verilen Δq vektörleriyle
    çalıştırır. cod_data.txt'den her quad GİRİŞİNDEKİ (elem=2 ve elem=6) ortalama
    yatay/dikey kapalı yörünge bozulmasını çıkarır.

    Parametreler:
      g_value : kuadrupol gradyanı [T/m] (kmod akımına karşılık gelir)
      dy_arr, dx_arr : shape (2*nFODO,) — quad hizalama hataları [m]
      t_end   : simülasyon süresi [s]  (50 tur ≈ 2.3e-4 s)

    Geri dönüş: dict(x_bpm, y_bpm) — her biri shape (2*nFODO,)
        Sıralama: [QF_0, QD_0, QF_1, QD_1, ..., QF_{nF-1}, QD_{nF-1}]
    """
    from integrator import integrate_particle, FieldParams

    M2 = 0.938272046; AMU = 1.792847356; C = 299792458.0; M1 = 1.672621777e-27
    p_magic = M2 / np.sqrt(AMU)
    E_tot = np.sqrt(p_magic**2 + M2**2)
    beta0 = p_magic / E_tot
    gamma0 = 1.0 / np.sqrt(1.0 - beta0**2)
    R0 = config['R0']
    E0_V_m = -(p_magic * beta0 / R0) * 1e9
    direction = config.get('direction', -1)

    # Başlangıç: tam merkezde, küçük açısal kick yok (saf COD ölçümü için)
    p_mag = gamma0 * M1 * C * beta0
    y0 = [config.get('dev0', 0.0), config.get('y0', 0.0), 0.0,
          0.0, 0.0, p_mag * direction,
          0.0, 0.0, 1.0]

    fields = FieldParams()
    fields.R0 = R0
    fields.E0 = E0_V_m
    fields.E0_power = config.get('E0_power', 1.0)
    fields.quadG1 = float(g_value)
    fields.quadG0 = float(g_value)
    fields.quadSwitch = 1.0
    fields.sextSwitch = 0.0
    fields.EDMSwitch = 0.0
    fields.direction = float(direction)
    fields.nFODO = float(config['nFODO'])
    fields.quadLen = float(config['quadLen'])
    fields.driftLen = float(config['driftLen'])
    fields.poincare_quad_index = -1.0
    fields.rfSwitch = 0.0

    nF = int(config['nFODO'])
    n_q = 2 * nF
    if dt is None:
        dt = float(config.get('dt', 1e-11))

    q_tilt = np.zeros(n_q) if quad_tilt_arr is None else np.asarray(quad_tilt_arr)
    integrate_particle(
        y0, 0.0, t_end, dt, fields=fields, return_steps=return_steps,
        quad_dy=np.asarray(dy_arr), quad_dx=np.asarray(dx_arr),
        dipole_tilt=np.zeros(n_q), quad_tilt=q_tilt,
        quad_dG=np.zeros(n_q),
    )

    # cod_data.txt'yi oku — 8*nFODO+1 satır (boundary kapanışı dahil)
    cod = np.loadtxt("cod_data.txt", skiprows=1)  # [s_m, x_mm, y_mm]
    x_bpm = np.empty(n_q); y_bpm = np.empty(n_q)
    for k in range(nF):
        x_bpm[2*k    ] = cod[k*8 + 2, 1] * 1e-3   # QF entry: elem=2
        y_bpm[2*k    ] = cod[k*8 + 2, 2] * 1e-3
        x_bpm[2*k + 1] = cod[k*8 + 6, 1] * 1e-3   # QD entry: elem=6
        y_bpm[2*k + 1] = cod[k*8 + 6, 2] * 1e-3
    return {'x_bpm': x_bpm, 'y_bpm': y_bpm}


# =============================================================================
# Aşama C — İki-kmod rekonstrüksiyonu (simülasyon = "gerçek makine")
# =============================================================================
def stage_C_two_kmod(config, plane='y', g_pert_frac=0.02,
                     sigma_dq=100e-6, seed=42,
                     t_end=1e-4, verbose=True):
    """
    Aşama C: Simülasyon = gerçek makine, analitik model = R kaynağı.

    Adımlar:
      1. Rastgele Δq vektörü üret (verilen seed ile, σ=100 μm)
      2. Simülasyon (g_nom, Δq) → y₁_sim
      3. Simülasyon (g_pert, Δq) → y₂_sim
      4. Analitik R₁, R₂ inşa et
      5. v₁ = R₁⁻¹·y₁,  v₂ = R₂⁻¹·y₂
      6. Δq̂ = (v₁ + v₂)/2  (ana rekonstrüksiyon)
      7. Δq_true ile karşılaştır

    Başarı kriteri (YAPILACAKLAR.md): RMS < 10 μm, corr > 0.95
    """
    g_nom  = config['g1']
    g_pert = g_nom * (1.0 + g_pert_frac)
    nF     = int(config['nFODO'])
    N      = 2 * nF

    # Hangi düzlem hangi misalignment?
    rng = np.random.default_rng(seed)
    dq_y = rng.normal(0.0, sigma_dq, N)
    dq_x = rng.normal(0.0, sigma_dq, N)
    dq_true = dq_y if plane == 'y' else dq_x

    if verbose:
        print(f"\n[Aşama C] Düzlem={plane},  Δg/g={g_pert_frac:+.1%},  "
              f"σ_Δq={sigma_dq*1e6:.0f} μm,  seed={seed}")
        print(f"  Simülasyon 1/2 (g = {g_nom:.4f} T/m) ...")

    run1 = _run_simulation_with(config, g_nom,  dq_y, dq_x, t_end=t_end)

    if verbose:
        print(f"  Simülasyon 2/2 (g = {g_pert:.4f} T/m) ...")

    run2 = _run_simulation_with(config, g_pert, dq_y, dq_x, t_end=t_end)

    y1 = run1['y_bpm'] if plane == 'y' else run1['x_bpm']
    y2 = run2['y_bpm'] if plane == 'y' else run2['x_bpm']

    # Analitik R₁, R₂
    K_x_arc = None
    if plane == 'x':
        from fodo_lattice import calibrate_K_x_arc
        K_x_arc = calibrate_K_x_arc(config)
    R1, beta, phi, Q1, KL1 = build_R_for_gradient(config, g_nom,  plane, K_x_arc)
    R2, _,    _,   Q2, KL2 = build_R_for_gradient(config, g_pert, plane, K_x_arc)

    # İki-kmod rekonstrüksiyonu
    v1 = direct_invert(R1, y1)
    v2 = direct_invert(R2, y2)
    dq_hat = 0.5 * (v1 + v2)

    # Karşılaştırma için tek-kmod sonuçları da
    err_v1   = np.sqrt(np.mean((v1     - dq_true)**2))
    err_v2   = np.sqrt(np.mean((v2     - dq_true)**2))
    err_avg  = np.sqrt(np.mean((dq_hat - dq_true)**2))
    corr_v1  = np.corrcoef(v1,     dq_true)[0, 1]
    corr_v2  = np.corrcoef(v2,     dq_true)[0, 1]
    corr_avg = np.corrcoef(dq_hat, dq_true)[0, 1]

    # ΔR yöntemiyle karşılaştırma (ne kadar kötü olduğunu görmek için)
    dR  = R1 - R2
    dy  = y1 - y2
    dq_dR = direct_invert(dR, dy)
    err_dR  = np.sqrt(np.mean((dq_dR - dq_true)**2))
    corr_dR = np.corrcoef(dq_dR, dq_true)[0, 1]

    if verbose:
        print(f"\n  Sonuçlar (gerçek RMS Δq = {np.std(dq_true)*1e6:.1f} μm):")
        print(f"    v₁ tek-kmod-1   : RMS = {err_v1 *1e6:8.2f} μm   corr = {corr_v1 :.4f}")
        print(f"    v₂ tek-kmod-2   : RMS = {err_v2 *1e6:8.2f} μm   corr = {corr_v2 :.4f}")
        print(f"    (v₁+v₂)/2 (ana) : RMS = {err_avg*1e6:8.2f} μm   corr = {corr_avg:.4f}")
        print(f"    ΔR doğrudan     : RMS = {err_dR *1e6:8.2f} μm   corr = {corr_dR :.4f}")
        print(f"    İyileştirme oranı (ΔR / iki-kmod) ≈ {err_dR/err_avg:.1f}×")

    return {
        'plane': plane, 'dq_true': dq_true, 'dq_hat': dq_hat,
        'v1': v1, 'v2': v2, 'y1': y1, 'y2': y2,
        'err_avg': err_avg, 'corr_avg': corr_avg,
        'err_dR': err_dR, 'corr_dR': corr_dR,
        'R1': R1, 'R2': R2, 'beta': beta, 'phi': phi,
    }


# =============================================================================
# Aşama D — Gürbüzlük testi
# =============================================================================
def stage_D_robustness(config, plane='y', g_pert_frac=0.02, sigma_dq=100e-6,
                       seed=42, t_end=1.5e-4, N_trials=8, out_dir='.',
                       verbose=True):
    """
    Aşama D: Dört hata kaynağında iki-kmod rekonstrüksiyonu gürbüzlük taraması.

      1. BPM gürültüsü  : σ_noise ∈ [0, 1, 2, 5, 10, 20] μm
      2. BPM sabit ofseti: σ_b    ∈ [0, 10, 50, 100, 200] μm
      3. Model beta hatası: δβ/β  ∈ [0, 0.5, 1, 2, 3, 5] %
      4. Kuadrupol eğimi : σ_tilt ∈ [0, 0.5, 1, 2] mrad

    1-3 için baz simülasyon çifti yeterlidir (hızlı).
    4 için her seviyede ayrı simülasyon çifti gerekir.
    """
    g_nom  = config['g1']
    g_pert = g_nom * (1.0 + g_pert_frac)
    nF     = int(config['nFODO'])
    N      = 2 * nF

    rng = np.random.default_rng(seed)
    dq_y   = rng.normal(0.0, sigma_dq, N)
    dq_x   = rng.normal(0.0, sigma_dq, N)
    dq_true = dq_y if plane == 'y' else dq_x

    if plane == 'x':
        from fodo_lattice import calibrate_K_x_arc
        K_x_arc = calibrate_K_x_arc(config)
    else:
        K_x_arc = None

    R1, beta, phi, Q1, KL1 = build_R_for_gradient(config, g_nom,  plane, K_x_arc)
    R2, _,    _,   Q2, KL2 = build_R_for_gradient(config, g_pert, plane, K_x_arc)

    if verbose:
        print(f"\n[Aşama D] Düzlem={plane}  —  baz simülasyon çifti koşuluyor ...")
    run1 = _run_simulation_with(config, g_nom,  dq_y, dq_x, t_end=t_end)
    run2 = _run_simulation_with(config, g_pert, dq_y, dq_x, t_end=t_end)
    y1_base = run1['y_bpm'] if plane == 'y' else run1['x_bpm']
    y2_base = run2['y_bpm'] if plane == 'y' else run2['x_bpm']

    def _rms_corr(y1, y2, R1_, R2_):
        v1 = direct_invert(R1_, y1)
        v2 = direct_invert(R2_, y2)
        dq_hat = 0.5 * (v1 + v2)
        err  = np.sqrt(np.mean((dq_hat - dq_true) ** 2))
        corr = np.corrcoef(dq_hat, dq_true)[0, 1]
        return err, corr

    # ---- 1. BPM gürültüsü ----
    noise_levels = np.array([0, 1, 2, 5, 10, 20]) * 1e-6
    rng_n = np.random.default_rng(seed + 100)
    noise_rms  = np.zeros(len(noise_levels))
    noise_corr = np.zeros(len(noise_levels))
    for i, sig in enumerate(noise_levels):
        errs = []
        for _ in range(N_trials):
            y1n = y1_base + rng_n.normal(0, sig, N)
            y2n = y2_base + rng_n.normal(0, sig, N)
            e, _ = _rms_corr(y1n, y2n, R1, R2)
            errs.append(e)
        noise_rms[i]  = np.mean(errs)
        noise_corr[i] = _rms_corr(y1_base, y2_base, R1, R2)[1]  # baz corr

    # ---- 2. BPM sabit ofseti ----
    offset_levels = np.array([0, 10, 50, 100, 200]) * 1e-6
    rng_b = np.random.default_rng(seed + 200)
    offset_rms  = np.zeros(len(offset_levels))
    offset_corr = np.zeros(len(offset_levels))
    for i, sig in enumerate(offset_levels):
        errs = []
        for _ in range(N_trials):
            b   = rng_b.normal(0, sig, N)
            y1b = y1_base + b
            y2b = y2_base + b
            e, _ = _rms_corr(y1b, y2b, R1, R2)
            errs.append(e)
        offset_rms[i]  = np.mean(errs)
        offset_corr[i] = _rms_corr(y1_base, y2_base, R1, R2)[1]

    # ---- 3. Model beta hatası ----
    beta_err_levels = np.array([0, 0.5, 1, 2, 3, 5]) / 100.0
    rng_bb = np.random.default_rng(seed + 300)
    beta_rms  = np.zeros(len(beta_err_levels))
    beta_corr = np.zeros(len(beta_err_levels))
    for i, dbb in enumerate(beta_err_levels):
        errs = []
        for _ in range(N_trials):
            bp   = beta * (1.0 + rng_bb.normal(0, dbb, N))
            R1p  = build_response_matrix(bp, phi, Q1, KL1)
            R2p  = build_response_matrix(bp, phi, Q2, KL2)
            e, c = _rms_corr(y1_base, y2_base, R1p, R2p)
            errs.append(e)
        beta_rms[i]  = np.mean(errs)
        beta_corr[i] = _rms_corr(y1_base, y2_base, R1, R2)[1]

    # ---- 4. Kuadrupol eğimi (yeni sim çifti per seviye) ----
    tilt_levels = np.array([0, 0.5, 1, 2]) * 1e-3
    rng_t = np.random.default_rng(seed + 400)
    tilt_rms  = np.zeros(len(tilt_levels))
    tilt_corr = np.zeros(len(tilt_levels))
    for i, sig_t in enumerate(tilt_levels):
        if verbose:
            print(f"  Eğim taraması {i+1}/{len(tilt_levels)}: "
                  f"σ_tilt = {sig_t*1e3:.1f} mrad ...")
        if sig_t == 0.0:
            # Baz simülasyonu yeniden kullan
            tilt_rms[i], tilt_corr[i] = _rms_corr(y1_base, y2_base, R1, R2)
            continue
        errs = []
        for _ in range(max(1, N_trials // 4)):  # eğim sim pahalı → az deneme
            ta   = rng_t.normal(0, sig_t, N)
            r1t  = _run_simulation_with(config, g_nom,  dq_y, dq_x, t_end=t_end,
                                        quad_tilt_arr=ta)
            r2t  = _run_simulation_with(config, g_pert, dq_y, dq_x, t_end=t_end,
                                        quad_tilt_arr=ta)
            y1t  = r1t['y_bpm'] if plane == 'y' else r1t['x_bpm']
            y2t  = r2t['y_bpm'] if plane == 'y' else r2t['x_bpm']
            e, c = _rms_corr(y1t, y2t, R1, R2)
            errs.append(e)
        tilt_rms[i]  = np.mean(errs)
        tilt_corr[i] = np.mean([_rms_corr(
            r1t['y_bpm'] if plane == 'y' else r1t['x_bpm'],
            r2t['y_bpm'] if plane == 'y' else r2t['x_bpm'],
            R1, R2)[1] for _ in [0]])  # tek deneme korelasyonu

    if verbose:
        print(f"\n  Gürbüzlük özeti — baz RMS = "
              f"{_rms_corr(y1_base, y2_base, R1, R2)[0]*1e6:.2f} μm")
        for lev, rms_ in zip(noise_levels * 1e6, noise_rms * 1e6):
            print(f"    BPM gürültüsü {lev:5.1f} μm → RMS = {rms_:.2f} μm")
        for lev, rms_ in zip(offset_levels * 1e6, offset_rms * 1e6):
            print(f"    BPM ofseti    {lev:5.0f} μm → RMS = {rms_:.2f} μm")
        for lev, rms_ in zip(beta_err_levels * 100, beta_rms * 1e6):
            print(f"    β hatası      {lev:5.1f}  % → RMS = {rms_:.2f} μm")
        for lev, rms_ in zip(tilt_levels * 1e3, tilt_rms * 1e6):
            print(f"    Eğim σ        {lev:5.1f} mrad→ RMS = {rms_:.2f} μm")

    # ---- Grafik ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Aşama D — Gürbüzlük taraması (düzlem={plane})", fontsize=13)

    ax = axes[0, 0]
    ax.plot(noise_levels * 1e6, noise_rms * 1e6, 'o-')
    ax.axhline(10, ls='--', color='r', lw=1, label='10 μm eşiği')
    ax.set_xlabel('BPM gürültüsü σ [μm]')
    ax.set_ylabel('Rekonstrüksiyon RMS [μm]')
    ax.set_title('1. BPM ölçüm gürültüsü')
    ax.legend(); ax.grid(True, alpha=0.4)

    ax = axes[0, 1]
    ax.plot(offset_levels * 1e6, offset_rms * 1e6, 's-', color='C1')
    ax.axhline(10, ls='--', color='r', lw=1, label='10 μm eşiği')
    ax.set_xlabel('BPM sabit ofseti σ [μm]')
    ax.set_ylabel('Rekonstrüksiyon RMS [μm]')
    ax.set_title('2. BPM sabit ofseti')
    ax.legend(); ax.grid(True, alpha=0.4)

    ax = axes[1, 0]
    ax.plot(beta_err_levels * 100, beta_rms * 1e6, '^-', color='C2')
    ax.axhline(10, ls='--', color='r', lw=1, label='10 μm eşiği')
    ax.set_xlabel('Model β hatası δβ/β [%]')
    ax.set_ylabel('Rekonstrüksiyon RMS [μm]')
    ax.set_title('3. Model beta hatası')
    ax.legend(); ax.grid(True, alpha=0.4)

    ax = axes[1, 1]
    ax.plot(tilt_levels * 1e3, tilt_rms * 1e6, 'D-', color='C3')
    ax.axhline(10, ls='--', color='r', lw=1, label='10 μm eşiği')
    ax.set_xlabel('Kuadrupol eğimi σ [mrad]')
    ax.set_ylabel('Rekonstrüksiyon RMS [μm]')
    ax.set_title('4. Kuadrupol eğimi')
    ax.legend(); ax.grid(True, alpha=0.4)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f'stage_D_robustness_{plane}.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    if verbose:
        print(f"  Grafik: {out_path}")

    return {
        'plane': plane,
        'noise_levels': noise_levels, 'noise_rms': noise_rms,
        'offset_levels': offset_levels, 'offset_rms': offset_rms,
        'beta_err_levels': beta_err_levels, 'beta_rms': beta_rms,
        'tilt_levels': tilt_levels, 'tilt_rms': tilt_rms,
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

    # Aşama C: iki-kmod rekonstrüksiyonu (simülasyon = gerçek makine)
    # ~30 tur, 2 simülasyon ≈ 2-3 dk toplam
    for plane in ['y', 'x']:
        stage_C_two_kmod(config, plane=plane, g_pert_frac=0.02,
                         sigma_dq=100e-6, seed=42, t_end=1.5e-4)

    # Aşama D: gürbüzlük taraması
    # Tilt taraması 3 ek sim çifti ≈ +6 dk
    for plane in ['y', 'x']:
        stage_D_robustness(config, plane=plane, g_pert_frac=0.02,
                           sigma_dq=100e-6, seed=42, t_end=1.5e-4,
                           N_trials=8)
