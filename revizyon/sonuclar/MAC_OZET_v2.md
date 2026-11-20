# İkinci tur ofis koşuları — özet (MAC_PROMPT_v2.md)

Tarih: 20.11.2026. İş B (polar) Mac'te (ofis) koşuldu ve push edildi (6ee79e9). İş A (hafta) Mac'te
başlamamıştı; laptopta (Linux, 12 çekirdek, `-w 10`, motor `lib_integrator_rev.so` yeniden derlendi) koşuldu.
C = ½|f_CW − f_CCW|, hedef (1e-9) biriminde.

## İş A — haftalık senaryo, tek demet (W1-E2) / iki demet (W1-E4)

- Seed 5–24 (yeni, 240 görev) + seed 0–4 (ilk tur). Saat 24/96/168, yön ±1, kısa ayar (14 tur / 2 iter).
- Süre: 17:43 → 20:20 (2 sa 37 dk, laptop). Hata, NaN ya da uyarı yok.

`ozet` çıktısı (25 desen × 3 saat = 75 değer):

| Senaryo | n | medyan | %84 | maks | hedef-üstü |
|---|---|---|---|---|---|
| W1-E2 (tek demet) | 75 | 0.432 | 1.36 | 3.28 | 16/75 |
| W1-E4 (iki demet) | 75 | 0.0716 | 0.130 | 0.259 | 0/75 |

İlk 15 değerle (seed 0–4) karşılaştırma:

| | E2 medyan / %84 / maks / üst | E4 medyan / %84 / maks / üst |
|---|---|---|
| eski (seed 0–4, n=15) | 0.655 / 1.14 / 2.05 / 3 | 0.0531 / 0.122 / 0.175 / 0 |
| yeni (seed 5–24, n=60) | 0.428 / 1.37 / 3.28 / 13 | 0.075 / 0.135 / 0.259 / 0 |

Saate göre (25 desen):

| saat | E2 medyan / maks / üst | E4 medyan / maks / üst |
|---|---|---|
| 24 | 0.574 / 1.58 / 5 | 0.0714 / 0.259 / 0 |
| 96 | 0.611 / 3.28 / 11 | 0.0638 / 0.153 / 0 |
| 168 | 0.285 / 0.969 / 0 | 0.0756 / 0.175 / 0 |

Sonuç: iki demetin üstünlüğü 25 desende de tutuyor, ama 15 değerdeki kadar büyük değil. Medyan oranı 0.65/0.053
≈ 12× iken şimdi 0.43/0.072 ≈ 6×. Eşli (aynı seed, aynı saat) E2/E4 oranının medyanı 6.2; 75 çiftin 8'inde E4 > E2.
İki demette hedefi aşan değer yok (en büyüğü 0.26); tek demette 16/75 hedefi aşıyor (en büyüğü 3.3).
"Bir mertebe" ifadesi medyan için abartılı, "~6 kat; kuyrukta (maks) ~13 kat" daha doğru.
Anormallik: E2'nin 168 saatteki değerleri 24 ve 96 saattekinden küçük (hiçbiri hedefi aşmıyor). Bu, sapmanın
zamanla tekdüze büyümediğini gösteriyor; nedeni incelenmedi.

## İş B — kuadrupol polaritesi ve saptırıcı ε×φ (Mac'te koşuldu)

- Seed 4–11 (yeni, 64 görev) + seed 0–3 (ilk tur); n=12. Ayrıntı: `saptirici_testleri.jsonl`.

| test | \|odd+\| medyan | \|odd−\| medyan | Omarov ½\|odd₊+odd₋\| medyan / maks | korelasyon |
|---|---|---|---|---|
| polar_defl (yalnız saptırıcı) | 3.07 | 3.07 | 0.0182 / 0.0431 | −1.00 |
| polar_tam (r = 1e-6/√h, 168 sa) | 6.17 | 6.11 | 0.0132 / 0.0374 | −1.00 |

Sonuç: polariteyi çevirmek saptırıcının sahte EDM'sinin işaretini tam çeviriyor. Omarov birleşiminde
170× (polar_defl) ile 470× (polar_tam) arası söndürülüyor; kalan değer hedefin %4'ünden küçük. Yani Omarov'un
şeması bu sistematikten muaf, bizim şemamız (polarite çevirmesiz) değil. Makalede açıkça yazılmalı.

n=12 kontrolü: `polar_tam` normal polarite satırları `saptirici_testleri.jsonl`'da yalnız seed 4–11 için var.
Seed 0–3'ün normal polarite değeri `ozet()` tarafından bilerek `dinamik_saptirici_edm.jsonl`'dan
(rate 1e-6, 168 sa) alınıyor. Bu, aynı anlık görüntüyle (`dinamik_saptirici_lin.json`, `1e-06_1_s`, 168 sa) yapılmış ölçüm.
Seed 0–3 ile 4–11'in Omarov kalıntıları aynı büyüklükte (0.006–0.037), yani birleştirme tutarlı ve n=12 doğru.
