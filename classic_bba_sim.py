#!/usr/bin/env python3
"""
classic_bba_sim.py — T5: null-arayan KLASİK BBA'nın simetrik-mod performansı
(separation_bba_testleri.md §3; analitik per-quad Twiss, nefes DAHİL)

Soru: her quad'ı tek tek modüle edip demeti YEREL bump'la tarayarak tepkinin
null'unu bulmak (klasik BBA), simetrik (orbit-kör) kaçıklık bileşenini de
merkez-merkez ölçebilir mi? Projede öldürülen yöntemler genlik-okuma (nefes)
ve global inversiyonlardı; null-arayan yerel ölçüm hiç test edilmemişti.

Fizik notu (birinci mertebe, tam): quad i'nin alanı modülasyonla
g(1+ε)(y−m_i) olur; ekstra alan ε·g·(y−m_i) → TEK, YEREL kick
θ = ε·K·L·(y(i)−m_i). Yani tepki, demetin O quad'ın merkezine uzaklığıyla
orantılı; null tam merkezde. "Nefes", sabit yörüngede genlik okuyanın
y(i)≫m_i yüzünden gördüğü şeydi; tarama+null bunu tanım gereği aşar.
Bu betik bunu pertürbatif varsaymaz: tepki tam [R(K_i)−R(K0)]@m ile
(optik yeniden hesaplanarak) üretilir.

Ölçüm modeli:
  - BPM ofsetleri (100 μm): modülasyon FARKINDA ve tarama-ekseni FARKINDA
    otomatik düşer (statik) → sonuçlar ofset-bağışık.
  - BPM gürültüsü: nokta başına σ_pt = √2·σ_n/√N_avg (mod aç/kapa farkı,
    N_avg atış ortalaması); hem tepkiye hem yerel-konum okumasına eklenir.
  - Bump: u_i = R0_model⁻¹ e_i (modelden). Null bir sıfır-geçişi olduğundan
    bump'ın kusuru (β-beat'li modelle kurulması) bias yaratmamalı — bu da
    test ediliyor (--bbeat).

Çıktı: per-quad merkez-hata RMS + SİMETRİK/antisimetrik bileşen RMS
(mean-çıkarmadan; metrik tuzağına dikkat) + kestirilen kalan sahte-EDM
(f ≈ A_eff·σ_res², A_eff=1.18e4 rad/s/m², iki düzlem aynı seviyede varsayımı).

Kullanım: python3 classic_bba_sim.py [--noise 1e-6] [--navg 100] [--bbeat 0.01]
"""
import argparse
import numpy as np

from make_orbit_figures import (R_perquad, sym_anti_projectors, NQ, G_NOM)

A_EFF = 1.18e4          # rad/s/m² (kmod_bba_sonuclar §4; f = A_eff·σ²)
TARGET = 1e-9           # rad/s (1 nrad/s ↔ 1e-29 e·cm)


def run_bba(sigma_mis=10e-6, sigma_bpm_noise=1e-6, n_avg=100, n_scan=9,
            scan_half=150e-6, eps=0.02, bbeat=0.0, seed=0, verbose=True):
    rng = np.random.default_rng(seed)

    # Gerçek makine: kaçıklıklar + (ops.) per-quad gradyan hatası (β-beat)
    m_true = rng.normal(0.0, sigma_mis, NQ)                # merkezler [m]
    g_true = np.full(NQ, G_NOM)
    if bbeat > 0:
        g_true *= (1.0 + rng.normal(0.0, bbeat, NQ))

    # Gerçek makinenin tepki matrisleri (nefes dahil): nominal + her quad modülasyonlu
    R0_true, _ = R_perquad(g_true)
    R_mod = []
    for i in range(NQ):
        gi = g_true.copy(); gi[i] *= (1.0 + eps)
        R_mod.append(R_perquad(gi)[0])

    # Bump: TEK komşu düzeltici (gerçekçi, mütevazı güç). R⁻¹e_i tipi "tek
    # noktada oynat" desenleri simetrik yönlerde ~mm'lik dev kaynaklar üretir
    # ve modelin küçük iç tutarsızlıklarını amplifiye eder — gerçek BBA'da da
    # kullanılmaz. Null ilkesi bump şeklinden bağımsızdır (tarama ekseni BPM
    # i'den ölçülür); tek knob yeter.
    R0_model, _ = R_perquad(np.full(NQ, G_NOM))

    sigma_pt = np.sqrt(2.0) * sigma_bpm_noise / np.sqrt(n_avg)
    deltas = np.linspace(-scan_half, scan_half, n_scan)

    est = np.zeros(NQ)          # kestirilen merkezler (mutlak)
    bump_norm = np.zeros(NQ)    # düzeltici gücü göstergesi
    for i in range(NQ):
        # komşular arasından demeti quad i'de en verimli oynatan tek düzeltici
        cands = [(i - 1) % NQ, (i + 1) % NQ, (i - 2) % NQ, (i + 2) % NQ]
        j = max(cands, key=lambda jj: abs(R0_model[i, jj]))
        u = np.zeros(NQ)
        u[j] = 1.0 / R0_model[i, j]      # modele göre y(i)'de birim hareket
        bump_norm[i] = np.linalg.norm(u)
        # tarama: her δ için (a) tepki vektörü, (b) BPM i'de yerel konum farkı
        A_pts = np.zeros((n_scan, NQ))
        x_pts = np.zeros(n_scan)
        y_base = R0_true @ m_true                          # bump'sız yörünge
        for kd, d in enumerate(deltas):
            mm = m_true + d * u                            # bump = merkez kaydırma
            A = (R_mod[i] - R0_true) @ mm                  # mod aç−kapa farkı
            A_pts[kd] = A + rng.normal(0.0, sigma_pt, NQ)
            # tarama ekseni: BPM i okuma farkı (ofset düşer, gürültü kalır)
            x_pts[kd] = ((R0_true @ mm)[i] - y_base[i]
                         + rng.normal(0.0, sigma_pt))
        # null: tüm BPM'lerde A ≈ s_b·(x − x*) modeline LSQ
        # önce her BPM için eğim, sonra ağırlıklı sıfır-geçişi
        X = np.vstack([x_pts, np.ones(n_scan)]).T
        coef, _, _, _ = np.linalg.lstsq(X, A_pts, rcond=None)   # (2, NQ)
        s_b, a_b = coef[0], coef[1]                        # A_b = s_b·x + a_b
        w = s_b**2
        x_star = -np.sum(w * (a_b / np.where(s_b == 0, 1, s_b))) / np.sum(w)
        est[i] = y_base[i] + x_star                        # mutlak merkez kestirimi

    err = est - m_true
    P_sym, P_anti = sym_anti_projectors()
    rms = lambda a: float(np.sqrt(np.mean(a**2)))
    res = {"rms": rms(err), "sym": rms(P_sym @ err), "anti": rms(P_anti @ err),
           "bump_norm_max": float(bump_norm.max()),
           "sym_signal": rms(P_sym @ m_true)}
    # kalan sahte-EDM kestirimi: iki düzlem de bu seviyeye hizalanırsa
    res["f_res"] = A_EFF * res["rms"]**2
    if verbose:
        print(f"  seed {seed}: merkez-hata RMS = {res['rms']*1e6:.3f} μm "
              f"(sym {res['sym']*1e6:.3f}, anti {res['anti']*1e6:.3f}) "
              f"[sym sinyal {res['sym_signal']*1e6:.1f} μm]  "
              f"f_res ≈ {res['f_res']:.2e} rad/s ({res['f_res']/TARGET:.2f}× hedef)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=1e-6, help="BPM tek-atış gürültü [m]")
    ap.add_argument("--navg", type=int, default=100, help="nokta başına atış ortalaması")
    ap.add_argument("--bbeat", type=float, default=0.0, help="per-quad gradyan hatası (β-beat)")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    print("=== T5: NULL-ARAYAN KLASİK BBA — simetrik mod merkez ölçümü ===")
    print(f"  σ_mis=10 μm, BPM ofset 100 μm (farkta düşer), gürültü "
          f"{args.noise*1e6:.1f} μm/atış, N_avg={args.navg}, β-beat={args.bbeat*100:.1f}%")
    print(f"  nokta gürültüsü σ_pt = {np.sqrt(2)*args.noise/np.sqrt(args.navg)*1e9:.0f} nm")

    R = [run_bba(sigma_bpm_noise=args.noise, n_avg=args.navg,
                 bbeat=args.bbeat, seed=s) for s in range(args.seeds)]
    sym = np.array([r["sym"] for r in R]); anti = np.array([r["anti"] for r in R])
    tot = np.array([r["rms"] for r in R]); f = np.array([r["f_res"] for r in R])
    print(f"\n  ENSEMBLE ({args.seeds} seed):")
    print(f"  merkez-hata RMS      = {tot.mean()*1e6:.3f} ± {tot.std()*1e6:.3f} μm")
    print(f"  SİMETRİK bileşen     = {sym.mean()*1e6:.3f} ± {sym.std()*1e6:.3f} μm"
          f"   (sinyal ~7 μm)")
    print(f"  antisimetrik bileşen = {anti.mean()*1e6:.3f} ± {anti.std()*1e6:.3f} μm")
    print(f"  kalan sahte-EDM      ≈ {f.mean():.2e} rad/s = {f.mean()/TARGET:.2f}× hedef")
    print(f"  bump gücü (maks ‖u‖) = {max(r['bump_norm_max'] for r in R)*1e6:.0f} μm-eşdeğer")
    print("\n  Yorum: simetrik bileşen hatası gürültü-sınırlıysa (β-beat'le "
          "büyümüyorsa), klasik BBA no-go'nun DIŞINDADIR (yerel null ≠ inversiyon).")


if __name__ == "__main__":
    main()
