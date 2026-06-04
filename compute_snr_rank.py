#!/usr/bin/env python3
"""
compute_snr_rank.py  —  Taslaktaki sayısal iddiaların doğrulanması

Doğrular:
  1. k=2 yörünge genliği (BPM uzayı): 10 μm kaçıklık → ? mm
  2. Beyaz BPM ofseti (100 μm) → k=2 Fourier genliği → SNR
  3. R'nin tekil değer spektrumu: rank, koşul sayısı, hangi k en iyi koşullu
  4. k-modülasyon için ΔR rank argümanı (≈ quad sayısı)
"""
import numpy as np, os, sys, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

spec = importlib.util.spec_from_file_location("mpf", "make_paper_figures.py")
mpf = importlib.util.module_from_spec(spec); spec.loader.exec_module(mpf)
R = mpf._load_R()
n_q = R.shape[0]
rng = np.random.default_rng(7)

def Fcos(k, n=n_q):
    j = np.arange(n)
    if k == 0: return (-1.0)**j
    return (-1.0)**j * np.cos(2*np.pi*k*(j//2)/(n//2))
def Fsin(k, n=n_q):
    j = np.arange(n)
    return (-1.0)**j * np.sin(2*np.pi*k*(j//2)/(n//2))

def fourier_amp(vec, k):
    """vec'in k modundaki FİZİKSEL genliği (katsayı konvansiyonu):
       vec ≈ Σ a_k cos_k + b_k sin_k,  a_k = <vec,cos_k>/||cos_k||²
       (make_paper_figures fig5 ile aynı: beyaz 100μm → ~26μm)"""
    fc = Fcos(k); fs = Fsin(k)
    ac = (vec @ fc) / (fc @ fc)
    if np.allclose(fs, 0):   # k=0 ve k=12: sin yok
        return abs(float(ac))
    as_ = (vec @ fs) / (fs @ fs)
    return float(np.hypot(ac, as_))

def matched_filter_sigma(sigma_b, k):
    """Eşleşmiş-filtre kestiriminin σ(Â_k): M_k=R·F_k pattern'ine projeksiyon.
       σ(Â_k) = σ_b / ||M_k||  (M_k = R·F_k, F_k normalize edilmemiş)"""
    Mc = R @ Fcos(k)
    Mk_norm = np.linalg.norm(Mc) if np.allclose(Fsin(k), 0) else \
              np.sqrt(np.linalg.norm(Mc)**2 + np.linalg.norm(R @ Fsin(k))**2)
    return sigma_b / Mk_norm * np.sqrt(2)  # cos+sin iki parametre

# ── 1. k=2 yörünge genliği (10 μm kaçıklık) ──────────────────────────────────
print("=" * 70)
print("1. k=2 YÖRÜNGE GENLİĞİ (BPM uzayında), 10 μm kaçıklık")
A = 10e-6
dq = A * Fcos(2)              # saf k=2 cos modu kaçıklık
y  = R @ dq                  # BPM yörüngesi
print(f"  Kaçıklık:    A=10 μm,  ||dq||={np.linalg.norm(dq)*1e6:.1f} μm")
print(f"  Yörünge:     ||y||={np.linalg.norm(y)*1e6:.0f} μm  (=||M_k=2||×A)")
print(f"  Pik BPM:     {np.max(np.abs(y))*1e6:.0f} μm")
print(f"  RMS BPM:     {np.std(y)*1e6:.0f} μm")
print(f"  k=2 Fourier genliği (BPM uzayı): {fourier_amp(y,2)*1e6:.0f} μm")
print(f"  → Taslaktaki '2.5 mm' yerine doğru değer yukarıda.")

# ── 2. Beyaz BPM ofseti → k=2 genliği → SNR ──────────────────────────────────
print()
print("=" * 70)
print("2. BEYAZ BPM OFSETİ → k=2 genliği → SNR (iki yöntem)")
sig_amp_k2 = fourier_amp(y, 2)    # sinyalin k=2 genliği (katsayı konv.)
A_sig = 10e-6                      # gerçek kaçıklık genliği
print(f"  Sinyal k=2 yörünge genliği (katsayı): {sig_amp_k2*1e6:.1f} μm")
print(f"  (gerçek kaçıklık 10 μm; eşleşmiş-filtre bunu A=10μm olarak geri verir)")
print()
for off_um in [20, 100]:
    off = off_um*1e-6
    amps = np.array([fourier_amp(rng.normal(0, off, n_q), 2) for _ in range(5000)])
    off_k2 = amps.mean()
    sig_mf = matched_filter_sigma(off, 2)   # σ(Â_2) eşleşmiş filtre
    print(f"  σ_off={off_um} μm (beyaz):")
    print(f"     ham ofsetin k=2 genliği     = {off_k2*1e6:.1f} μm")
    print(f"     [A] NAİF Fourier SNR        = {sig_amp_k2/off_k2:.1f}  "
          f"(F_k modu projeksiyonu)")
    print(f"     [B] EŞLEŞMİŞ-FİLTRE σ(Â_2)   = {sig_mf*1e6:.2f} μm  →  "
          f"SNR = {A_sig/sig_mf:.1f}  (M_k=R·F_k projeksiyonu)")

# ── 3. R tekil değer spektrumu / rank ────────────────────────────────────────
print()
print("=" * 70)
print("3. R TEKİL DEĞER SPEKTRUMU (rank, koşul sayısı)")
U, S, Vt = np.linalg.svd(R)
print(f"  R boyut: {R.shape},  rank = {np.linalg.matrix_rank(R)}")
print(f"  σ_max = {S[0]:.3e},  σ_min = {S[-1]:.3e},  κ = {S[0]/S[-1]:.1f}")
print()
# Her sağ tekil vektörün baskın k modu
print("  En büyük 6 tekil değerin baskın Fourier modu:")
for i in range(6):
    v = Vt[i]
    k_amps = [(k, fourier_amp(v, k)) for k in range(13)]
    kbest = max(k_amps, key=lambda t: t[1])
    print(f"     σ_{i}={S[i]:.3e}  →  baskın mod k={kbest[0]}  "
          f"(genlik {kbest[1]:.3f})")
print()
# Her k modunun yörünge kazancı = ||R Fk_unit||
print("  k modu → yörünge kazancı ||R·F_k(birim)||:")
for k in range(9):
    fc = Fcos(k); fc=fc/np.linalg.norm(fc)
    print(f"     k={k}:  kazanç = {np.linalg.norm(R@fc):.1f}")

# ── 4. ΔR rank argümanı (k-modülasyon) ───────────────────────────────────────
print()
print("=" * 70)
print("4. ΔR RANK (k-modülasyon: tüm quadlar birlikte modüle)")
print("  Misalignment kestirim Jacobian'ı J[i,j] = ∂y_i/∂(δy_j) = R[i,j]")
print("  (gradyan modülasyonu birinci derecede R ile orantılı sapma üretir)")
print(f"  → rank(J) = rank(R) = {np.linalg.matrix_rank(R)} = quad sayısı ({n_q})")
print()
print("  Tek-tek modülasyon vs birlikte modülasyon:")
print("  Eğer modülasyon deseni M (n_mod × n_q) ise gözlemlenebilir")
print("  parametre sayısı = rank(R @ M^T). Tüm quadlar bağımsız → tam rank.")
# Örnek: tek harmonik modülasyon (kötü) vs rastgele ortogonal (iyi)
M_single = np.zeros((n_q, n_q)); np.fill_diagonal(M_single, 1.0)  # birim = her quad ayrı
M_onemode = np.outer(np.ones(n_q), Fcos(2))  # hepsi aynı k=2 deseni (kötü)
print(f"     birim (her quad bağımsız):     rank(R·Mᵀ) = "
      f"{np.linalg.matrix_rank(R @ M_single.T)}")
print(f"     tek-mod (hepsi aynı desen):    rank(R·Mᵀ) = "
      f"{np.linalg.matrix_rank(R @ M_onemode.T)}  (rank çöküyor!)")
