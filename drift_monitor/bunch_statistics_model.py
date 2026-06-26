#!/usr/bin/env python3
"""
bunch_statistics_model.py — Drift monitörün EDM istatistiğine değer modeli.

SORU: Orbit-tabanlı drift monitör, spin-tabanlı hizalamadan (radyal-polarize
bunch) iş üstlenirse, o bunch'lar EDM'ye döner. Deney ne kadar hızlanır?

──────────────────────────────────────────────────────────────────────────────
MODEL (basit, varsayım-tabanlı — sayılar SENİN girdin):
  d0   = monitörsüz, demet zamanının spin-hizalamaya giden kesri (EDM'ye sayılmaz)
  beta = driftin antisimetrik (orbit-GÖRÜNÜR) kesri = monitörün üstlenebileceği pay
  Monitör antisimetriği bedava üstlenince spin-hizalama duty'si: d1 = d0*(1-beta)
  EDM hassasiyeti ~ 1/sqrt(N_edm * T); sabit hedef için süre  T ~ 1/(1-d)
  => Hızlanma (zaman oranı) = T0/T1 = (1-d1)/(1-d0)
     Tasarruf edilen süre %  = (1 - T1/T0)*100

──────────────────────────────────────────────────────────────────────────────
*** KRİTİK VARSAYIM (make-or-break — betiği koşmadan önce oku) ***
  beta>0 OLMASI, monitörün rutin orbit-düzeltmesinin YAPAMADIĞI bir antisimetrik
  hizalama işini spin-hizalamadan alabilmesine bağlıdır. İki uç:
   (a) Rutin orbit-düzeltmesi antisimetriği zaten hallediyor + spin-hizalama
       SADECE simetriği yapıyorsa  -> beta ~ 0 -> KAZANÇ YOK. Argüman çöker.
   (b) Spin-hizalama (radyal bunch) antisimetrik orbit-corrugation geri-beslemesi
       de yapıyorsa (Omarov §5.2 "dikey hız / orbit corrugation feedback") ve
       monitör onu offset-bağışık biçimde üstlenebiliyorsa -> beta > 0 -> kazanç.
  Hangisi doğru, deneyin gerçek iş bölümüne bağlı; bunu SEN değerlendir. Bu model
  yalnız "EĞER beta şu ise kazanç bu" haritasını verir.
──────────────────────────────────────────────────────────────────────────────
Kullanım:  python3 bunch_statistics_model.py
Bağımlılık: numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def speedup(d0, beta):
    """Zaman oranı T0/T1 (>1 = hızlanma) ve tasarruf%."""
    d1 = d0 * (1.0 - beta)
    ratio = (1.0 - d1) / (1.0 - d0)        # T0/T1
    saved = (1.0 - 1.0 / ratio) * 100.0    # % daha kısa süre
    return ratio, saved


def main():
    # ---- SENİN GİRDİLERİN (değiştir) -------------------------------------
    d0_list = [0.1, 0.2, 0.3, 0.5]   # monitörsüz spin-hizalama duty kesri (varsayım)
    beta_grid = np.linspace(0.0, 0.8, 17)  # antisimetrik (offloadable) kesir
    # ---------------------------------------------------------------------

    print("="*70)
    print("Drift monitör -> EDM istatistik kazancı (varsayım-tabanlı)")
    print("="*70)
    print(f"{'d0 (duty)':>10} {'beta':>6} {'hızlanma T0/T1':>15} {'tasarruf %':>12}")
    for d0 in d0_list:
        for beta in (0.2, 0.5, 0.8):
            r, s = speedup(d0, beta)
            print(f"{d0:>10.2f} {beta:>6.2f} {r:>15.2f} {s:>12.1f}")
        print("-"*45)

    # grafik: hızlanma vs beta, her d0 için
    plt.figure(figsize=(7, 4.5))
    for d0 in d0_list:
        rr = [speedup(d0, b)[0] for b in beta_grid]
        plt.plot(beta_grid, rr, "o-", ms=3, label=f"d0={d0:.1f} (hizalamaya ayrılan pay)")
    plt.axhline(1.0, color="gray", lw=0.8, ls=":")
    plt.xlabel(r"$\beta$ = driftin antisimetrik (monitörün üstlendiği) kesri")
    plt.ylabel(r"EDM hızlanması  $T_0/T_1$  (>1 daha hızlı)")
    plt.title("Drift monitörün EDM deneyine istatistik/süre kazancı\n(varsayım-tabanlı; KRİTİK: betadaki varsayımı dosya başından oku)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("bunch_statistics_gain.png", dpi=140)
    print("\nKaydedildi: bunch_statistics_gain.png")
    print("\nYORUM: kazanç d0 (hizalamaya ne kadar zaman gidiyor) ve beta (ne kadarı")
    print("       offloadable antisimetrik) ile büyür. beta~0 ise (dosya başı uyarısı)")
    print("       kazanç yok; bu yüzden asıl iş 'beta gerçekten >0 mı' sorusunda.")


if __name__ == "__main__":
    main()
