#!/usr/bin/env python3
"""random_harmonic_analysis.py — Rastgele hizalama dağılımının harmonik içeriği

Soru: Gerçekçi bir deneyde quad kaymaları ~100 μm RMS rastgele dağılır.
Bu dağılımın Fourier (harmonik) bileşenleri ne kadar büyük, ve hangileri
kmod ile ölçülebilir?

Bu script:
  1) Verilen spektrumla (beyaz veya kırmızı/power-law) rastgele dy üretir
  2) Harmonik genliklerini (tam Fourier ayrışımı) hesaplar
  3) İstatistik: her harmoniğin beklenen genliği, dağılımı
  4) Rank-sınırlı ölçümde hangi harmoniklerin geri çatılabileceğini gösterir

Fizik: Beyaz gürültü tüm harmoniklere eşit güç dağıtır → hiçbiri ayrıcalıklı
değil, tam karakterizasyon rank 48 (48 quad) ister. Kırmızı spektrum
(uzun-dalga baskın, fiziksel) → düşük harmonikler hâkim, az quad yeter.
"""
import numpy as np


def generate_random_dy(N=48, sigma=100e-6, alpha=0.0, seed=0):
    """Power-law spektrumlu rastgele dy üret.

    alpha = 0 → beyaz gürültü (düz spektrum)
    alpha > 0 → kırmızı (P(k) ∝ k^-alpha, düşük-k baskın)

    Toplam RMS = sigma olacak şekilde normalize edilir.
    """
    rng = np.random.default_rng(seed)
    # Frekans uzayında üret: her mod için karmaşık katsayı
    freqs = np.fft.rfftfreq(N, d=1.0) * N        # 0, 1, ..., N/2
    # Güç profili: P(k) ∝ k^-alpha (k=0 için 1)
    power = np.ones_like(freqs)
    nz = freqs > 0
    power[nz] = freqs[nz] ** (-alpha)
    amp = np.sqrt(power)
    # Rastgele faz + genlik
    re = rng.standard_normal(len(freqs)) * amp
    im = rng.standard_normal(len(freqs)) * amp
    spec = re + 1j * im
    spec[0] = spec[0].real          # DC reel
    if N % 2 == 0:
        spec[-1] = spec[-1].real    # Nyquist reel
    dy = np.fft.irfft(spec, n=N)
    # RMS normalize
    dy = dy / np.std(dy) * sigma
    return dy


def harmonic_amplitudes(dy):
    """dy'nin tam Fourier harmonik genliklerini hesapla.
    Döndürür: k dizisi ve A_k = √(a_kc² + a_ks²) genlikleri."""
    N = len(dy)
    D = np.fft.rfft(dy)
    k = np.arange(len(D))
    # rfft → cos/sin genlik dönüşümü
    A = np.abs(D) * 2.0 / N
    A[0] = np.abs(D[0]) / N                 # DC
    if N % 2 == 0:
        A[-1] = np.abs(D[-1]) / N           # Nyquist
    return k, A


def theoretical_white_amplitude(sigma, N):
    """Beyaz gürültüde her harmoniğin RMS genliği: 2σ/√N."""
    return 2 * sigma / np.sqrt(N)


def spectrum_report(sigma=100e-6, N=48, alphas=(0.0, 2.0), n_real=2000):
    """Farklı spektrumlar için harmonik istatistiği."""
    print(f"{'='*64}")
    print(f"  Rastgele dağılımın harmonik içeriği")
    print(f"  N = {N} quad   toplam RMS = {sigma*1e6:.0f} μm")
    print(f"{'='*64}")
    print(f"  Beyaz gürültü teorisi: her harmonik ≈ 2σ/√N = "
          f"{theoretical_white_amplitude(sigma, N)*1e6:.1f} μm (RMS)\n")

    for alpha in alphas:
        # Çok sayıda gerçekleşme üzerinden ortalama harmonik genliği
        acc = None
        for s in range(n_real):
            dy = generate_random_dy(N, sigma, alpha, seed=s)
            k, A = harmonic_amplitudes(dy)
            acc = A.copy() if acc is None else acc + A
        Amean = acc / n_real

        tag = "BEYAZ (düz)" if alpha == 0 else f"KIRMIZI (k^-{alpha:.0f})"
        print(f"  ── Spektrum: {tag} ──")
        print(f"  {'k':>3} {'<A_k> (μm)':>12} {'% toplam güç':>13}")
        tot = np.sum(Amean[1:]**2)
        for kk in range(1, min(10, len(Amean))):
            pct = Amean[kk]**2 / tot * 100
            bar = '█' * int(pct / 2)
            print(f"  {kk:>3} {Amean[kk]*1e6:>11.1f} {pct:>11.1f}%  {bar}")
        print()


def recovery_demo(N=48, sigma=100e-6, alpha=2.0, K_model=4, rank=8,
                  seed=1, n_real=500):
    """Rank-sınırlı ölçümde düşük harmonikleri geri çatma kalitesi.

    K_model: bazda kaç harmonik modelleniyor (k=1..K_model)
    rank: yığılmış ΔR rank'ı (≈ modüle edilen quad sayısı)

    Yüksek harmonikler (k>K_model) modellenmediği için sızıntı yaratır.
    Spektrum kırmızıysa (alpha büyük) bu sızıntı küçük; beyazsa büyük.
    """
    from fourier_reconstruct import fodo_basis  # antisym FODO bazı

    print(f"{'='*64}")
    print(f"  Rank-sınırlı geri çatım: k=1..{K_model} modelleniyor, rank={rank}")
    print(f"  Spektrum: {'beyaz' if alpha==0 else f'kırmızı k^-{alpha:.0f}'}")
    print(f"{'='*64}")

    rng = np.random.default_rng(seed)

    def rank1_dR():
        u = rng.standard_normal(N); u /= np.linalg.norm(u)
        v = rng.standard_normal(N); v /= np.linalg.norm(v)
        return np.outer(u, v) * 1000.0

    ks = list(range(1, K_model + 1))
    F, meta = fodo_basis(N, ks, antisym=True)

    err_per_k = {k: [] for k in ks}
    for s in range(n_real):
        dy = generate_random_dy(N, sigma, alpha, seed=1000 + s)
        # gerçek düşük-k harmonik genlikleri (referans)
        _, A_true = harmonic_amplitudes(dy)

        dRl = [rank1_dR() for _ in range(rank)]
        M = np.vstack([dR @ F for dR in dRl])
        b = np.concatenate([dR @ dy for dR in dRl])
        a = np.linalg.pinv(M, rcond=1e-3) @ b

        # geri çatılan harmonik genlikleri
        for k in ks:
            idx = [i for i, (kk, _) in enumerate(meta) if kk == k]
            A_fit = np.sqrt(sum(a[i]**2 for i in idx))
            # antisym baz ile tam Fourier farklı; oransal hatayı bazda ölç
            err_per_k[k].append(A_fit)

    print(f"  (Not: antisym FODO bazı tam Fourier'den farklı; bu demo")
    print(f"   rank yeterliliğinin spektrumla ilişkisini gösterir.)\n")
    print(f"  Ortalama geri çatılan |a_k| (μm), {n_real} gerçekleşme:")
    for k in ks:
        print(f"    k={k}: {np.mean(err_per_k[k])*1e6:7.1f} μm")


if __name__ == "__main__":
    spectrum_report(sigma=100e-6, N=48, alphas=(0.0, 2.0))
    print()
    print("YORUM:")
    print("  Beyaz gürültüde her harmonik ~29 μm — k=2 dahil hiçbiri")
    print("  ayrıcalıklı değil. Tam karakterizasyon 48 quad (rank 48) ister.")
    print("  Kırmızı spektrumda (fiziksel: zemin oturması, termal) güç")
    print("  düşük-k'da toplanır; birkaç harmonik baskın, az quad yeter.")
    print()
    print("  KRİTİK SORU: gerçek makinedeki misalignment spektrumu nedir?")
    print("  Bu, ölçüm fizibilitesini belirleyen tek en önemli parametre.")
