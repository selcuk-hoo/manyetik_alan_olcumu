# Literatür taraması: simetrik/antisimetrik kaçıklığın yörünge kazanç oranı

Tarih: 2026-11-20. İddia (bizim, `revizyon/tests/teshis_tan_mu4.py`):
G_sym/G_anti ≈ √[ ((√βF−√βD)/(√βF+√βD))² + (πm/2N)² ], ince mercek FODO'da ilk terim = tan(μ/4);
μ ≲ 60°'de ≤%8 doğru. Sonuç: hiçbir kararlı FODO'da "100×" gizlenme yok.

## Aranan ifadeler
1. closed orbit amplification factor girder motion focusing defocusing cancellation sqrt beta
2. ground motion long wavelength closed orbit FODO correlated misalignment sqrt beta_F beta_D
3. "amplification factor" correlated quadrupole displacement same girder focusing defocusing
4. Tiefenback 1985 FODO symmetric antisymmetric quadrupole misalignment
5. Fischer lattice response function ground wave C/λ quadrupoles in phase
Ayrıca yerel `literatur/ref_rossbach_groundwaves.md` (tam metin) okundu.

## Bulgular
- **BELGELENMİŞ — fizik biliniyor.** Rossbach 1989 (Part. Accel. 23, 121), Denk. (4): ince mercek FODO'da
  düzlem yer dalgasına kapalı yörünge tepkisi, odaklayan kuadrupollerde +√βF/f, dağıtanlarda −√βD/f ağırlıklı
  iki toplam. Bizim harmonik-m simetrik desenimiz, C/λ=m'lik bir yer dalgasıyla aynı şeydir; yani G_sym
  Rossbach'ın tepki fonksiyonunda (Denk. 8) örtük olarak vardır.
- **BELGELENMİŞ — girder korelasyonu.** NSLS-II hizalama makalesi (PAC 2009, MO6PFP003): aynı girder'daki
  odaklayan/dağıtan kuadrupoller zıt tekme verdiği için korelasyonlu hareket kısmen söner; "girder closed
  orbit amplification factor" tanımlanır. Fizik aynı, çerçeve (girder) farklı.
- **BELGELENMİŞ — uzun dalga bastırması.** Halka analizleri: λ ≫ C/ν dalgalar ~(C/νλ)² ile bastırılır.
- **BELGELENMİŞ (ders kitabı).** İnce mercek FODO: βmax/βmin = (1+sin(μ/2))/(1−sin(μ/2)).
  tan(μ/4) bundan iki satırda çıkar; ayrıca aranmadı.
- **Doğrudan eşleşme bulunamadı:** simetrik/antisimetrik ayrışımının kazanç ORANI için kapalı formül
  ((√βF−√βD)/(√βF+√βD) ⊕ πm/2N), ve bunun kararlı FODO'da alt sınır koyması (oran ≥ πm/2N).
  "Yenidir" DEĞİL: tarama kapsamlı değil.

## En yakın önceki çalışma ve fark
Rossbach 1989: aynı tepki toplamı, ama amaç rastgele/dalga hareketinin rms yörünge büyüklüğü; antisimetrik
referansa oran, gözlenebilirlik veya EDM çerçevesi yok. Bizim katkı (YORUM): bilinen toplamın,
simetrik/antisimetrik ayrışımda basit bir orana indirgenmesi ve "100×" gibi düzgün-halka tahminlerinin neden
yanlış olduğunun açıklanması. Makalede yeni bir sonuç gibi değil, Rossbach'a atıfla "bilinen tepkiden
türetilen basit oran" olarak sunulmalı.

## Kapsam dışı / elle bakılabilecekler
- Tiefenback 1985 (FODO sim/antisim mod) bulunamadı; INSPIRE'da "Tiefenback" + "closed orbit" aranmalı.
- Fischer (SLAC, yer hareketi tepki fonksiyonu) özgün raporu okunmadı.
- Montag & Rossbach yer titreşimi notu, Parkhomchuk–Shiltsev–Stupakov 1994 tam metinleri okunmadı.
- JACoW tam metin indekslenmiyor; elle arama: "girder amplification factor", "coherent quadrupole motion
  FODO", "correlated misalignment closed orbit focusing defocusing".

Kaynaklar:
- Rossbach, Part. Accel. 23, 121 (1989) — yerel tam metin
- https://accelconf.web.cern.ch/PAC2009/papers/mo6pfp003.pdf (NSLS-II)
- https://accelconf.web.cern.ch/medsi2018/papers/tuph28.pdf (APS-U girder amplification)
- https://arxiv.org/pdf/1202.5490 (Tevatron, √β ağırlıklı COD toplamı)
- https://core.ac.uk/download/44197474.pdf (SSC dönemi FODO düzlem dalga analizi)
