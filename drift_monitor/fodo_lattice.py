"""
fodo_lattice.py — Twiss parametreleri ve DFT tabanlı yanıt matrisi geri dönüşümü

Bu modül periyodik FODO örgüsünün simetrisinden faydalanır:
  - Hücre transfer matrisinden Twiss parametreleri (β, φ, Q) çıkarılır
  - Yanıt matrisi R, Courant-Snyder formülünden inşa edilir
  - Sirkülant yapı DFT (FFT algoritmasıyla) ile diyagonalize edilerek
    O(N log N) maliyetinde tersi alınır

Simülasyon kütüphanesine (integrator) bağımlılığı yoktur; tamamen analitiktir.

Hücre yapısı (integrator.cpp ile uyumlu):
    QF → drift → arc → drift → QD → drift → arc → drift → (sonraki QF)

Arc deflektör fiziksel davranışı (n=1 silindirik elektrik kapasitör):
  * DİKEY düzlem: E_z = 0 tam olarak (Maxwell, n=1) → arc = saf drift, K_y_arc = 0
  * YATAY düzlem: merkezkaç + relativistik dispersiyon kuplajı sıfır olmayan
    odaklama yaratır. Basit analitik formül (1-n±β²)/ρ² sonucu tam vermez
    (dönen referans çerçevesindeki Coriolis kuplajı sebebiyle).
    K_x_arc, "ideal örgü" simülasyonundan (hizalama hatası sıfır, başlangıç
    yalnız açısal kick) elde edilen Qx_ref ile bisection kalibre edilir.

Doğrulama: ideal sim'de Qx ≈ 2.68, Qy ≈ 2.36 (params.json varsayılan
örgüsünde, R0=95.49 m, g1=0.21, nFODO=24).

Örnekleme konvansiyonu: her quad GİRİŞİNDE (quad merkezinde değil).
"""

import json
import numpy as np

# =============================================================================
# Fiziksel sabitler
# =============================================================================
C_LIGHT = 299792458.0
M_PROTON_GEV = 0.938272046
AMU_PROTON = 1.792847356
# 1 GeV/c manyetik rijitlik = 1/0.299792458 T·m
GEVC_TO_TM = 1.0 / 0.299792458


# =============================================================================
# 1. Manyetik rijitlik ve sihirli momentum
# =============================================================================
def magic_momentum_proton(mom_error=0.0):
    """Proton EDM sihirli momentumu [GeV/c]. p_magic = M / sqrt(AMU)."""
    return (M_PROTON_GEV / np.sqrt(AMU_PROTON)) * (1.0 + mom_error)


def compute_Brho(p_GeV_c):
    """Manyetik rijitlik [T·m]. Brho = p/q, q=e için Brho = p[GeV/c] / 0.2998."""
    return p_GeV_c * GEVC_TO_TM


# =============================================================================
# 2. Temel transfer matrisleri (2×2, tek düzlem)
# =============================================================================
def drift_matrix(L):
    """Serbest drift."""
    return np.array([[1.0, L], [0.0, 1.0]])


def thick_quad_matrix(K, L, focusing):
    """
    Kalın-lens kuadrupol matrisi.
    K        : pozitif gradient büyüklüğü [1/m²]
    L        : kuadrupol uzunluğu [m]
    focusing : True ise bu düzlemde odaklayan, False ise dağıtan
    """
    if K <= 0.0:
        return drift_matrix(L)
    sqk = np.sqrt(K)
    phi = sqk * L
    if focusing:
        return np.array([[np.cos(phi),       np.sin(phi) / sqk],
                         [-sqk * np.sin(phi), np.cos(phi)]])
    else:
        return np.array([[np.cosh(phi),       np.sinh(phi) / sqk],
                         [sqk * np.sinh(phi), np.cosh(phi)]])


def arc_matrix(L, K):
    """
    Arc transfer matrisi, verilen odaklama gücü K [1/m²] için.

    K > 0  → odaklayan (cos/sin)
    K < 0  → dağıtan (cosh/sinh)
    K = 0  → saf drift

    Elektrik halka için K_y_arc = 0 (Maxwell, n=1 → E_z=0).
    K_x_arc, ideal simülasyon Qx referansından kalibrasyonla bulunur
    (bkz. calibrate_K_x_arc).
    """
    if K > 0:
        sqk = np.sqrt(K); phi = sqk * L
        return np.array([[np.cos(phi),       np.sin(phi) / sqk],
                         [-sqk * np.sin(phi), np.cos(phi)]])
    elif K < 0:
        sqk = np.sqrt(-K); phi = sqk * L
        return np.array([[np.cosh(phi),       np.sinh(phi) / sqk],
                         [sqk * np.sinh(phi), np.cosh(phi)]])
    else:
        return drift_matrix(L)


# Temiz simülasyondan kalibre edilmiş referans yatay tune
# (params.json varsayılanı: R0=95.49, g1=0.21, nFODO=24,
#  hizalama hatası = 0, başlangıç yalnız açısal kick → Qx ≈ 2.6824 arctan2)
QX_REF_CLEAN_SIM = 2.6824


def calibrate_K_x_arc(config, Q_x_target=QX_REF_CLEAN_SIM,
                     K_bounds=(0.0, 1.0e-3), tol=1e-10):
    """
    Yatay arc odaklamasını, verilen hedef tune Q_x_target'a uydurmak için
    bisection ile bulur.

    Hedef varsayılanı (QX_REF_CLEAN_SIM = 2.6824) ideal koşullarda
    simülasyonun verdiği temiz Qx değeridir — hizalama hatası kapalı,
    başlangıç yalnız 1 mrad açısal kick.

    Dikey için ayrı bir fonksiyon yok: K_y_arc = 0 (Maxwell n=1 → E_z=0).

    Geri dönüş: K_x_arc [m⁻²], pozitif (odaklayıcı)
    """
    nF      = int(config['nFODO'])
    L_d     = config['driftLen']
    L_q     = config['quadLen']
    R0      = config['R0']
    L_arc   = np.pi * R0 / nF

    p_GeV = magic_momentum_proton(config.get('momError', 0.0))
    Brho  = compute_Brho(p_GeV)
    K_abs = config['g1'] / Brho

    M_QF = thick_quad_matrix(K_abs, L_q, focusing=True)
    M_QD = thick_quad_matrix(K_abs, L_q, focusing=False)
    M_D  = drift_matrix(L_d)

    def Q_for_K(K_arc):
        M_A   = arc_matrix(L_arc, K_arc)
        M_mid = M_D @ M_A @ M_D
        M_end = M_D @ M_A @ M_D
        M_cell = M_end @ M_QD @ M_mid @ M_QF
        tr2 = (M_cell[0, 0] + M_cell[1, 1]) / 2.0
        if abs(tr2) >= 1.0:
            return None
        mu = np.arccos(tr2)
        if M_cell[0, 1] < 0:
            mu = 2.0 * np.pi - mu
        return nF * mu / (2.0 * np.pi)

    lo, hi = K_bounds
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        Q_mid = Q_for_K(mid)
        if Q_mid is None or Q_mid > Q_x_target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# =============================================================================
# 3. Twiss parametrelerinin çıkarımı
# =============================================================================
def twiss_from_periodic_matrix(M):
    """
    Periyodik hücre matrisi M'den (β, α, μ) çıkarır.
    sin(μ) işareti M[0,1] işaretiyle alınır (β > 0 konvansiyonu).
    """
    trace = M[0, 0] + M[1, 1]
    cos_mu = trace / 2.0
    if abs(cos_mu) >= 1.0:
        raise ValueError(f"Karasız hücre: |tr/2|={abs(cos_mu):.4f} >= 1")
    sin_mu = np.sign(M[0, 1]) * np.sqrt(1.0 - cos_mu * cos_mu)
    mu = np.arctan2(sin_mu, cos_mu)
    if mu < 0:
        mu += 2.0 * np.pi
    beta  = M[0, 1] / sin_mu
    alpha = (M[0, 0] - M[1, 1]) / (2.0 * sin_mu)
    return beta, alpha, mu


def propagate_twiss(beta0, alpha0, M):
    """
    Twiss parametrelerini matris M ile bir noktadan diğerine taşır.
    Standart Courant-Snyder taşıma.
    """
    gamma0 = (1.0 + alpha0 * alpha0) / beta0
    m11, m12 = M[0, 0], M[0, 1]
    m21, m22 = M[1, 0], M[1, 1]
    beta  =  m11 * m11 * beta0  - 2.0 * m11 * m12 * alpha0 + m12 * m12 * gamma0
    alpha = -m11 * m21 * beta0  + (m11 * m22 + m12 * m21) * alpha0 - m12 * m22 * gamma0
    return beta, alpha


# =============================================================================
# 4. Twiss parametrelerinin her quad girişinde hesaplanması
# =============================================================================
def compute_twiss_at_quads(config, plane, K_x_arc=None, Q_x_target=None):
    """
    Her kuadrupol GİRİŞİNDE Twiss parametrelerini hesaplar.

    Hücre yapısı (QF girişinden başlayarak):
        QF → drift(L_d) → arc(L_def) → drift(L_d) → QD
           → drift(L_d) → arc(L_def) → drift(L_d) → (sonraki QF)

    Arc modeli:
      plane='y':  arc = saf drift  (K_y_arc = 0, fizikten)
      plane='x':  arc = ince odaklama, K_x_arc kalibre edilir
                  - K_x_arc parametresi verilirse o kullanılır
                  - Verilmezse Q_x_target (varsayılan: temiz sim Qx=2.6824)
                    ile bisection kalibrasyonu yapılır

    Geri dönüş: (beta_arr [N], phi_arr [N], Q)
        N = 2 * nFODO  (toplam quad sayısı; çift indeksler QF, tek indeksler QD)
    """
    nF = int(config['nFODO'])
    N  = 2 * nF

    p_GeV = magic_momentum_proton(config.get('momError', 0.0))
    Brho  = compute_Brho(p_GeV)
    K_abs = config['g1'] / Brho

    L_d     = config['driftLen']
    L_q     = config['quadLen']
    R0      = config['R0']
    arc_len = np.pi * R0 / nF   # L_def = 2πR0 / (2*nFODO) = πR0/nFODO

    if plane == 'x':
        QF_foc, QD_foc = True, False
        if K_x_arc is None:
            target = QX_REF_CLEAN_SIM if Q_x_target is None else float(Q_x_target)
            K_x_arc = calibrate_K_x_arc(config, Q_x_target=target)
        K_arc_eff = K_x_arc
    else:
        QF_foc, QD_foc = False, True
        K_arc_eff = 0.0   # fizik: n=1 → E_z=0 → arc dikeyde saf drift

    M_QF = thick_quad_matrix(K_abs, L_q, focusing=QF_foc)
    M_QD = thick_quad_matrix(K_abs, L_q, focusing=QD_foc)
    M_D   = drift_matrix(L_d)
    M_A   = arc_matrix(arc_len, K_arc_eff)

    # QF çıkışından QD girişine (drift + arc + drift)
    M_mid = M_D @ M_A @ M_D
    # QD çıkışından sonraki QF girişine (drift + arc + drift)
    M_end = M_D @ M_A @ M_D

    # Tek hücre transfer matrisi (QF girişinden QF girişine)
    M_cell = M_end @ M_QD @ M_mid @ M_QF
    beta0, alpha0, mu_cell = twiss_from_periodic_matrix(M_cell)
    Q = nF * mu_cell / (2.0 * np.pi)

    # QF girişi → QD girişi transfer matrisi
    M_to_QD = M_mid @ M_QF

    beta_QD, _ = propagate_twiss(beta0, alpha0, M_to_QD)

    # Faz ilerleme: QF girişinde φ=0 (referans), QD girişinde φ_QD
    phi_QD = np.arctan2(M_to_QD[0, 1],
                        beta0 * M_to_QD[0, 0] - alpha0 * M_to_QD[0, 1])
    if phi_QD < 0:
        phi_QD += 2.0 * np.pi

    # Tüm hücreler için dizi oluştur (her hücre mu_cell kadar faz artışı)
    beta_arr = np.zeros(N)
    phi_arr  = np.zeros(N)
    for i_cell in range(nF):
        beta_arr[2 * i_cell    ] = beta0
        beta_arr[2 * i_cell + 1] = beta_QD
        phi_arr [2 * i_cell    ] = i_cell * mu_cell          # QF girişi
        phi_arr [2 * i_cell + 1] = i_cell * mu_cell + phi_QD # QD girişi

    return beta_arr, phi_arr, Q


# =============================================================================
# 5. İşaretli integre kuadrupol gücü
# =============================================================================
def signed_KL(config, plane):
    """
    Her quad için işaretli KL [1/m].
    Yatay düzlemde QF (+) odaklar, QD (-) dağıtır; dikey düzlemde tersi.

    İndeks düzeni: [QF_0, QD_0, QF_1, QD_1, ..., QF_{nF-1}, QD_{nF-1}]
    """
    nF   = int(config['nFODO'])
    L_q  = config['quadLen']
    p    = magic_momentum_proton(config.get('momError', 0.0))
    Brho = compute_Brho(p)
    K_abs = config['g1'] / Brho
    sign = +1.0 if plane == 'x' else -1.0
    KL = np.empty(2 * nF)
    KL[0::2] =  sign * K_abs * L_q   # QF: x'de odaklayan
    KL[1::2] = -sign * K_abs * L_q   # QD: x'de dağıtan
    return KL


# =============================================================================
# 6. Yanıt matrisi inşası
# =============================================================================
def build_response_matrix(beta, phi, Q, KL):
    """
    Courant-Snyder kapalı-yörünge yanıt matrisi:

        R_ij = sqrt(β_i β_j) · cos(|φ_i - φ_j| - πQ) · KL_j / (2 sin(πQ))

    İşaret kuralı: KL_j içinde quad tipi (QF/QD) ve düzlem (x/y) işareti
    saklanır (bkz. signed_KL). Bu formülün önünde ek bir "-" yoktur —
    simülasyon ile karşılaştırmada doğrulanmıştır.

    Boyut: N×N (N = len(beta))
    """
    denom = 2.0 * np.sin(np.pi * Q)
    sqb = np.sqrt(beta)
    dphi = np.abs(phi[:, None] - phi[None, :])
    R = np.outer(sqb, sqb) * np.cos(dphi - np.pi * Q) * KL[None, :] / denom
    return R


# =============================================================================
# 7. Sirkülant yaklaşım: özdeğerler ve DFT terslemesi
# =============================================================================
def circulant_eigenvalues_from_first_row(first_row):
    """
    Sirkülant matrisin özdeğerleri = ilk satırın DFT'si.
    Pratikte FFT algoritmasıyla O(N log N) hesaplanır.
    """
    return np.fft.fft(first_row)


def fft_invert(y, beta, phi, Q, KL):
    """
    Beş adımlı DFT (FFT ile) tabanlı geri dönüşüm.

    Adım 1: ỹ_i = y_i / sqrt(β_i)             (β normalizasyonu, soldan)
    Adım 2: Normalize sirkülant operatörün ilk satırı:
                m_k = cos(|φ_k - φ_0| - πQ) / (2 sin(πQ))
            Özdeğerler λ_k = DFT(m)
    Adım 3: Bilinmeyen dönüşümü u_j = sqrt(β_j) · KL_j · Δq_j
            ỹ_i = Σ_j M_ij u_j olduğundan DFT uzayında:
                Ũ_k = Ỹ_k / λ_k
    Adım 4: u = IDFT(Ũ)
    Adım 5: Δq_j = u_j / (sqrt(β_j) · KL_j)

    Bu algoritma, β ve KL non-uniform olsa bile, faz aralıkları |φ_i - φ_j|
    yalnızca (i-j)'ye bağlı olduğu sürece geçerlidir (gerçekte block-circulant
    yapıda küçük sapma olur — bkz. spectral_inversion.py Aşama B).
    """
    sqb = np.sqrt(beta)

    # Adım 1
    y_tilde = y / sqb

    # Adım 2 — normalize sirkülant matrisin ilk satırı
    dphi0 = np.abs(phi - phi[0])
    first_row = np.cos(dphi0 - np.pi * Q) / (2.0 * np.sin(np.pi * Q))
    lam = circulant_eigenvalues_from_first_row(first_row)

    # Adım 3-4
    Y = np.fft.fft(y_tilde)
    U = np.fft.ifft(Y / lam).real

    # Adım 5
    return U / (sqb * KL)


def direct_invert(R, y):
    """Referans geri dönüşüm: doğrudan np.linalg.solve(R, y)."""
    return np.linalg.solve(R, y)


# =============================================================================
# 8. Dahili tutarlılık testi
# =============================================================================
def _self_test(config_path="params.json"):
    """
    Bu modül doğrudan çalıştırıldığında bütün adımları sıfırdan kurar:
      1. Analitik Twiss hesapla (arc = saf drift, v2.8 referansı)
      2. R inşa et
      3. Rastgele Δq üret, y = R·Δq hesapla
      4. FFT geri dönüşümüyle Δq'yu kurtar
      5. Doğrudan np.linalg.solve ile karşılaştır
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    print("=" * 60)
    print("fodo_lattice.py — dahili tutarlılık testi")
    print("Arc deflektörler: SAF DRIFT (v2.8 analitik referansı)")
    print("=" * 60)

    for plane in ['x', 'y']:
        beta, phi, Q = compute_twiss_at_quads(config, plane)
        KL = signed_KL(config, plane)
        R  = build_response_matrix(beta, phi, Q, KL)
        N  = len(beta)

        print(f"\n--- Düzlem: {plane} ---")
        print(f"  N (quad sayısı) : {N}")
        print(f"  Q (tune)        : {Q:.6f}")
        print(f"  β aralığı       : [{beta.min():.3f}, {beta.max():.3f}] m")
        print(f"  φ_son           : {phi[-1]:.4f} rad   (2π·Q = {2*np.pi*Q:.4f})")
        print(f"  |KL| aralığı    : [{abs(KL).min():.4e}, {abs(KL).max():.4e}] 1/m")
        print(f"  κ(R)            : {np.linalg.cond(R):.3e}")

        rng = np.random.default_rng(0)
        dq_true = rng.normal(0.0, 100e-6, N)    # 100 μm RMS
        y       = R @ dq_true

        dq_fft    = fft_invert(y, beta, phi, Q, KL)
        dq_direct = direct_invert(R, y)

        err_fft    = np.sqrt(np.mean((dq_fft    - dq_true) ** 2))
        err_direct = np.sqrt(np.mean((dq_direct - dq_true) ** 2))
        corr_fft   = np.corrcoef(dq_fft,    dq_true)[0, 1]
        corr_direct= np.corrcoef(dq_direct, dq_true)[0, 1]

        print(f"  FFT     geri dönüşüm RMS hatası : {err_fft*1e6:.3e} μm   "
              f"(corr={corr_fft:.6f})")
        print(f"  Direct  geri dönüşüm RMS hatası : {err_direct*1e6:.3e} μm   "
              f"(corr={corr_direct:.6f})")


if __name__ == "__main__":
    _self_test()
