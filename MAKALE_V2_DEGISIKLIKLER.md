# Makale v2 — değişiklik listesi

Kaynak: `makale_orbit_bastirma_en_v2.tex` (tek kaynak). PDF'ler:
`makale_orbit_bastirma_en_v2_review.pdf` (TODO'suz, review için), `makale_orbit_bastirma_en_v2_todo.pdf`
(kırmızı TODO notlu). Cevap mektubu: `cevap_mektubu_v2.pdf`. Orijinal `makale_orbit_bastirma_en.tex` değişmedi.

## Yapı
- I.C (yeni): simetrik/antisimetrik tanımları başa alındı, indisler tanımlı (Hakem A4, B2).
- I.B: örgü/optik şekli eklendi (Fig. 1), Tablo I'e QF/QD β değerleri (Hakem B1).
- II.C (yeni içerik): sahte EDM donmuş spin enerjisinde; fırlatma noktası testi; Ek A.
- II.D: benchmark'lar güncellendi (p = 2.0002, roll analitik formülü, σ² testinin sınırı).
- III.A: "100 kat bastırma" düzeltildi → 4–7 kat; √β açıklaması; SVD ↔ simetrik kanıtı (Fig. 2) (Hakem B3).
- III.B: kanallar şekli yeniden ölçülene kadar review sürümünden çıkarıldı (TODO).
- IV.A: BBA bölümü kısaltıldı; BBA şekli kaldırıldı (eski prosedür, desenler kayıtlı değil) — kalan simetrik
  kaçıklık sayıları verildi; sahte EDM ifadesi TODO ile işaretli.
- IV.B: tek demet; rcond bağımlılığı ve uyuşmazlık ilkesi.
- IV.C: iki demet yöntemi 5 adımlı algoritma olarak (Hakem B4); ofset dejenerasyonu; Fig. 4 donmuş spinle.
- V (yeni): geçerlilik alanı — A BPM modeli (ρ, E1–E4, rijit BPM, σ_δ); B saptırıcı hataları (Hakem A1),
  trim, genlik taraması; C düzeltici türü (Hakem A3); D korelasyonlu kaçıklık (Hakem A2).
- VI (yeni): bir haftalık işletim senaryosu (Hakem B4).
- VII: sonuçlar geçerlilik koşullarıyla yeniden yazıldı.
- Ekler: A (yeni) donmuş spin ve fırlatma; B nefes; C fark matrisi (boyut düzeltmesi, Hakem A6–7); D 2B model.
- Teşekkür: yapay zekâ kullanım cümlesi genişletildi (TODO: yazarlar onaylamalı).

## TODO'lar (yalnız todo sürümünde görünür)
1. Önceki arXiv makalesine atıf (donmuş spin enerjisinin kayması).
2. Kanallar şekli (Fig. 2 eski) yeniden ölçüm — `makale_yeniden_olcum.py ch`.
3. BBA sahte EDM değerleri — `classic_bba_iter_donmus.py` (~19 saat).
4. Fig. 4'e %0.2 gradyan hatalı 15 seed iki demet noktası ve σ=5, 20 μm tek demet noktaları.
5. Korelasyonlu modellerin sahte EDM'si (eski prosedür; makalede sayı verilmedi).
6. Kaynakça: TTF2, PACMAN, SLS 2.0 yazar/başlık; FuG kaynağı yerine atıf yapılabilir bir kaynak.
7. Yapay zekâ beyanı.

## Dikkat
- Saptırıcı genlik taraması ve T1/T2 4–8 seed; makaledeki "patterns" sayıları buna göre.
- Kısaltılmış ayar (14/2) kullanıldı; sayısal taban ≈ 0.008 × hedef.
