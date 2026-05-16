"""
fodo_lattice.py — Twiss parametreleri ve DFT tabanlı yanıt matrisi geri dönüşümü

Bu modül periyodik FODO örgüsünün simetrisinden faydalanır:
  - Hücre transfer matrisinden Twiss parametreleri (β, φ, Q) çıkarılır
  - Yanıt matrisi R, Courant-Snyder formülünden inşa edilir
  - Sirkülant yapı DFT (FFT algoritmasıyla) ile diyagonalize edilerek
    O(N log N) maliyetinde tersi alınır

Simülasyon kütüphanesine (integrator) bağımlılığı yoktur; tamamen analitiktir.

Hibrit Twiss kaynağı: compute_twiss_at_quads çağrısında Q_measured veya
beta_measured verilirse analitik değer yerine kullanılır.
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

    K > 0: odaklayan (cos/sin), K < 0: dağıtan (cosh/sinh), K = 0: drift.

    Elektrik halka için K_x_arc ve K_y_arc analitik formüllerden tam türetmek
    karmaşıktır; standart (1-n+1/γ²)/ρ² yaklaşımı simülasyonu tam tutmaz.
    Bu nedenle K değerleri calibrate_arc_K() ile sayısal olarak elde edilir.
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


def calibrate_arc_K(config, Q_target, plane, K_bounds=(-3e-4, 3e-4), tol=1e-7):
    """
    Hedef tune Q_target'a karşılık gelen arc odaklama gücünü sayısal arama ile
    bulur (bisection). Halkanın gerçek (ölçülen) Q'sundan başlanır; analitik
    formüllerin yetersiz kaldığı durumlarda (elektrik halka, çoklu fiziksel
    etki) kalibre edilmiş tek-parametre modeli sağlar.

    Q_target  : ölçülen veya simülasyondan alınan tune (skaler)
    plane     : 'x' veya 'y'
    K_bounds  : arama aralığı [m⁻²]
    """
    p_GeV = magic_momentum_proton(config.get('momError', 0.0))
    Brho  = compute_Brho(p_GeV)
    K_abs = config['g1'] / Brho

    def tune_for_K(K_arc):
        elements, _ = cell_element_sequence(config, K_abs, plane, K_arc)
        M = propagate_through(elements)
        trace = M[0, 0] + M[1, 1]
        c = trace / 2.0
        if abs(c) >= 1.0:
            return None
        mu = np.arccos(c)
        return int(config['nFODO']) * mu / (2.0 * np.pi)

    lo, hi = K_bounds
    # nFODO/2π·μ olduğundan tune K ile monoton artar (büyük K → büyük μ)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        Q_mid = tune_for_K(mid)
        if Q_mid is None or Q_mid > Q_target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# =============================================================================
# 3. FODO hücresi: parçacığın gördüğü eleman sırası
# =============================================================================
def cell_element_sequence(config, K_abs, plane, K_arc):
    """
    Bir FODO hücresinin transfer matrislerini, parçacığın geçeceği sırayla
    döndürür. Quad merkezlerinde "marker" yerleştirir → Twiss buralarda örneklenir.

    Hücre yapısı (COD verisinden çıkarılan gerçek sıra):
        arc + drift(L_d) + QF + drift(L_d) + arc + drift(L_d) + QD + drift(L_d)

    QF ve QD aynı mutlak gradient büyüklüğünü paylaşır (|K| = g1/Brho).
    C++ kodunda: current_G1 = (elem==QF) ? +quadG1 : -quadG1  → aynı |K|.

    Marker konvansiyonu: her quad'ın MERKEZİnde örnekleme yapılır
    (yani yarım-quad öncesi + yarım-quad sonrası şeklinde bölünür).

    Geri dönüş: (matrices, marker_indices)
        matrices         : sırayla 2×2 matris listesi
        marker_indices   : matrices listesinde quad-merkezi pozisyonlarına
                           karşılık gelen indeksler (her hücre için 2 marker)
    """
    L_d  = config['driftLen']
    L_q  = config['quadLen']
    R0   = config['R0']
    nF   = int(config['nFODO'])
    arc_len = 2.0 * np.pi * R0 / (2 * nF)

    # QF ve QD'nin bu düzlemde odaklayıp odaklamadığı
    if plane == 'x':
        QF_foc, QD_foc = True, False
    else:
        QF_foc, QD_foc = False, True

    half_QF = thick_quad_matrix(K_abs, L_q / 2.0, focusing=QF_foc)
    half_QD = thick_quad_matrix(K_abs, L_q / 2.0, focusing=QD_foc)
    D = drift_matrix(L_d)
    A = arc_matrix(arc_len, K_arc)   # K_arc düzleme göre ayrı geçilir

    # Gerçek hücre sırası (arc önce, quad ortada):
    # A → D → [QF_half → MARKER → QF_half] → D → A → D → [QD_half → MARKER → QD_half] → D
    elements = [A, D, half_QF]        # arc + drift + QF ilk yarısı → MARKER
    marker_idx = [len(elements) - 1]
    elements += [half_QF, D, A, D, half_QD]   # QF ikinci yarısı + drift + arc + drift + QD ilk yarısı → MARKER
    marker_idx.append(len(elements) - 1)
    elements += [half_QD, D]          # QD ikinci yarısı + drift → hücre sonu
    return elements, marker_idx


def propagate_through(elements):
    """Element listesini soldan sağa çarparak toplam transfer matrisini döndürür."""
    M = np.eye(2)
    for E in elements:
        M = E @ M
    return M


# =============================================================================
# 4. Twiss parametrelerinin çıkarımı
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


def compute_twiss_at_quads(config, plane, Q_measured=None, beta_measured=None,
                           K_arc=None):
    """
    Her kuadrupol merkezinde Twiss parametrelerini hesaplar.

    Hibrit Twiss kaynağı:
      Q_measured      : verilirse arc odaklama K_arc bunu sağlayacak şekilde
                        kalibre edilir (sayısal bisection). Önerilen mod —
                        elektrik halkalarda analitik formüller yetersizdir.
      beta_measured   : skaler veya N-vektör; verilirse analitik β yerine kullanılır
      K_arc           : doğrudan arc odaklama gücü [1/m²]. Q_measured varsa
                        bu parametre yok sayılır.

    Geri dönüş: (beta_arr [N], phi_arr [N], Q)
        N = 2 * nFODO     (toplam quad sayısı)
    """
    nF = int(config['nFODO'])
    N  = 2 * nF

    p_GeV = magic_momentum_proton(config.get('momError', 0.0))
    Brho  = compute_Brho(p_GeV)
    # C++ kodu: QF ve QD aynı mutlak gradient kullanır → quadG1 = g1
    # (g0 yalnızca kmod modülasyonu için cell-0 QF'de kullanılır)
    K_abs = config['g1'] / Brho

    # K_arc: deneysel/ölçülen Q'dan kalibrasyon ya da kullanıcı verir
    if Q_measured is not None:
        K_arc_eff = calibrate_arc_K(config, float(Q_measured), plane)
    elif K_arc is not None:
        K_arc_eff = float(K_arc)
    else:
        K_arc_eff = 0.0   # saf drift; analitik en alt sınır

    elements, marker_idx = cell_element_sequence(config, K_abs, plane, K_arc_eff)
    M_cell = propagate_through(elements)
    beta0, alpha0, mu_cell = twiss_from_periodic_matrix(M_cell)

    # Tek hücre içinde marker konumlarında β ve φ örnekle
    cell_beta = []
    cell_phi  = []
    M_accum = np.eye(2)
    for k, E in enumerate(elements):
        M_accum = E @ M_accum
        if k in marker_idx:
            beta_k, _ = propagate_twiss(beta0, alpha0, M_accum)
            cell_beta.append(beta_k)
            # Faz: tek-noktalı transfer matrisinden φ = atan2(M[0,1], β0·M[0,0] - α0·M[0,1])
            phi_k = np.arctan2(M_accum[0, 1],
                               beta0 * M_accum[0, 0] - alpha0 * M_accum[0, 1])
            if phi_k < 0:
                phi_k += 2.0 * np.pi
            cell_phi.append(phi_k)

    cell_beta = np.array(cell_beta)
    cell_phi  = np.array(cell_phi)

    # Hücre boyunca tekrarlat
    beta_arr = np.tile(cell_beta, nF)
    phi_arr  = np.zeros(N)
    for i_cell in range(nF):
        phi_arr[2 * i_cell    ] = i_cell * mu_cell + cell_phi[0]
        phi_arr[2 * i_cell + 1] = i_cell * mu_cell + cell_phi[1]

    # K_arc kalibre edildiyse mu_cell zaten doğru Q'yu verir
    Q = nF * mu_cell / (2.0 * np.pi)

    if beta_measured is not None:
        if np.isscalar(beta_measured):
            beta_arr = np.full(N, float(beta_measured))
        else:
            beta_arr = np.asarray(beta_measured, dtype=float)
            if beta_arr.shape != (N,):
                raise ValueError(f"beta_measured boyutu {N} olmalı, {beta_arr.shape} verildi")

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
    # C++ kodu: QF ve QD aynı |K| kullanır (g1); g0 sadece kmod cell-0 içindir.
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
    Courant-Snyder yanıt matrisi:

        R_ij = -sqrt(β_i β_j) · cos(|φ_i - φ_j| - πQ) · KL_j / (2 sin(πQ))

    Boyut: N×N (N = len(beta))
    """
    N = len(beta)
    denom = 2.0 * np.sin(np.pi * Q)
    sqb = np.sqrt(beta)
    # |φ_i - φ_j| matrisi
    dphi = np.abs(phi[:, None] - phi[None, :])
    R = -np.outer(sqb, sqb) * np.cos(dphi - np.pi * Q) * KL[None, :] / denom
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
                m_k = -cos(|φ_k - φ_0| - πQ) / (2 sin(πQ))
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
    N = len(y)
    sqb = np.sqrt(beta)

    # Adım 1
    y_tilde = y / sqb

    # Adım 2 — normalize sirkülant matrisin ilk satırı
    dphi0 = np.abs(phi - phi[0])
    first_row = -np.cos(dphi0 - np.pi * Q) / (2.0 * np.sin(np.pi * Q))
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
      1. Twiss hesapla
      2. R inşa et
      3. Rastgele Δq üret, y = R·Δq hesapla
      4. FFT geri dönüşümüyle Δq'yu kurtar
      5. Doğrudan np.linalg.solve ile karşılaştır
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    print("=" * 60)
    print("fodo_lattice.py — dahili tutarlılık testi")
    print("=" * 60)

    # Simülasyonun gerçek tune'ları (run_simulation.py çıktısından)
    Q_sim = {'x': 2.6877, 'y': 2.0918}

    for plane in ['x', 'y']:
        print(f"\n--- Düzlem: {plane}  (hedef Q_sim={Q_sim[plane]}) ---")
        beta, phi, Q = compute_twiss_at_quads(config, plane,
                                              Q_measured=Q_sim[plane])
        KL = signed_KL(config, plane)
        R  = build_response_matrix(beta, phi, Q, KL)
        N  = len(beta)

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
