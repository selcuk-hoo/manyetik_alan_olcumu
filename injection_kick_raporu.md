# Injection Kick ile Kapalı Yörüngeye Oturtma — Yöntem, Testler ve Pratik Sınırlar

**Tarih:** 2026-06-10
**Kafes:** üniform 0.2 T/m FODO (48 kuadrupol), 10 μm RMS dikey hizalama hatası
**İlgili testler:** `test_kick_correction.py` (Test 1), `test_quad_flip_symmetry.py` (Test 4),
`test_nonideal_cancellation.py` (Test 5), `test_kicker_precision.py` (Test 6),
`test_realistic_amplitude.py` (Test 7), `test_realistic_beam.py` (Test 8),
`test_quad_dx_effect.py` (Test 9)

> **⚠️ Kapsam ve güncel not (2026-06).** Bu rapor **injection (fırlatma)
> toleransı** konusudur: parçacığı kapalı yörüngeye oturtmak. Buradaki "sahte EDM
> **lineer** büyür" ifadeleri **tek-parçacık betatron** kontaminasyonuna aittir
> (fırlatma ofseti δy ile lineer) — bu, kuadrupol kaçıklığının ürettiği **baskın**
> sahte EDM kanalıyla **karıştırılmamalıdır**: o kanal **kuadratiktir (σ²)** ve
> **dx·dy** geometrik fazdan gelir (`false_edm_harmonic_sinir.md §13`, `README §19.1`).
> Ayrıca buradaki tek-parçacık "ideal kick / CO=True" çerçevesi, ideal-olmayan
> parçacıkların sahte-EDM üretimini temsil etmediği için sonradan **4-katlı
> simetrik örnekleme + model-fit** reçetesiyle aşıldı. Adı geçen `test_*.py`
> betikleri 2026-06 temizliğinde kaldırıldı (mantık git geçmişinde).

---

## 1. Problem: Betatron kirlenmesi ve sahte EDM

Kuadrupol hizalama hataları (dy ~ 10 μm RMS) kapalı yörüngeyi (COD) kaydırır.
Parçacık ideal eksenden (y=0) fırlatılırsa, kaymış kapalı yörünge etrafında
**betatron salınımı** yapar. Bu salınımın dikey spin bileşenine (S_y) bulaşması,
gerçek EDM sinyalini taklit eden devasa bir **sahte EDM** üretir:

| Fırlatma | dS_y/dt [rad/s] |
|----------|------------------|
| Kapalı yörünge üzerinde (ideal) | ~6×10⁻¹¹ |
| Eksenden (y=0, betatron'lu) | ~8.6×10⁻⁵ |

Aradaki oran ~**1.4 milyon kat**. Bu yüzden ilk akla gelen strateji: parçacığı
injection sırasında tek bir kick ile tam kapalı yörüngeye "oturtmak".

## 2. Yöntemin mantığı

Kapalı yörünge periyodik bir eğridir: halkanın belirli bir azimutunda tek bir
(y_co, y'_co) noktasından geçer. Dolayısıyla injection azimutunda parçacığa
doğru konum ve doğru açıyı veren **tek bir kick** (2 serbestlik derecesi)
parçacığı sonsuza dek kapalı yörünge üzerinde tutar — dağıtık düzeltici
mıknatıslara gerek yoktur.

Simülasyonda bu "ideal kick"i Newton-Hessian yöntemiyle buluyoruz
(`find_closed_orbit`): sabit azimutta tur-başına konum varyansı, fırlatma
noktasının COD'den sapmasının kuadratik formudur; tek Newton adımı minimuma
(= kapalı yörüngeye) atlar.

## 3. Ne yaptık: Test zinciri ve sonuçları

### Test 1 — Kick toleransı (tek parçacık)

İdeal kick'ten δy kadar sapan fırlatmalarda sahte EDM **lineer** büyür:

- δy = 1 μm → dS_y/dt ≈ 2×10⁻⁶ (ideal tabanın 30 000 katı)
- δy = 100 μm → ~10⁻⁴ mertebesi

Yani oturtmanın işe yaraması için injection hassasiyeti **mikron-altı** olmalı.
Bu, tek parçacık için bile zorlayıcı bir tolerans.

### Test 4 — CW/CCW + quad-flip simetrisi (Omarov şeması)

Dört konfigürasyon (CW/CCW × normal/flip) karşılaştırıldı. Sonuç:

- (CW, normal) ve (CCW, flip) **aynı kapalı yörünge konumunu** paylaşır
  (y_co = −0.254 μm her ikisinde; fark < 0.1 nm). Açı işareti terstir
  (y' = ±0.639 μrad) çünkü y' = p_y/p_z ve ışın tersinince p_z işaret değiştirir.
- Sahte EDM'leri birebir aynıdır: +6.724×10⁻¹¹ vs +6.725×10⁻¹¹.
- **Fark (CW,n) − (CCW,f) = −5.6×10⁻¹⁶ → 120 000× iptal.**
- Gerçek EDM ışın yönünde tek, quad-flip'te çift olduğundan farkta 2D olarak
  **korunur**. Sahte EDM her ikisinde de tek olduğundan farkta **silinir**.

### Test 5 — Oturtma olmadan iptal

Her iki demet kendi COD'sinden δy kadar saptırılarak (betatron'lu) fırlatıldı:

| δy [μm] | Tek tek sahte EDM | Fark |
|---------|-------------------|------|
| 0 | 6.7×10⁻¹¹ | −5.6×10⁻¹⁶ |
| 10 | 1.8×10⁻⁴ | −2.0×10⁻¹⁶ |

Tek tek sahte EDM 2.6 milyon kat büyüdü; **fark değişmedi.** İptal, oturtma
kalitesinden bağımsız.

### Test 6 — Kicker hassasiyeti

Kicker hatası δ iki senaryoda tarandı:

- **A) Kicker quad'larla birlikte ters çevrilirse** (hata ayna-simetrik):
  fark ~10⁻¹⁵ sabit kalır — hata **ortak mod**, iptal olur.
- **B) Kicker çevrilmezse** (hata her iki demette aynı yönde): fark
  ~7.7×10⁻⁴/μrad hızla **sızar**.

Sonuç: önemli olan kicker'ın mutlak hassasiyeti değil, iki demet arasındaki
**korelasyonu**. Kicker quad'larla birlikte çevrilirse hata zararsızdır.

### Test 7 — Büyük genlik (gerçekçi betatron)

Mirror açı taraması Δ = 0…300 μrad (β·Δ ≈ 9 mm ≈ 1 cm betatron):

- Δ = 300 μrad: tek tek sahte EDM **−0.115 rad/s** (devasa). Fark: **0.000** (tam).
- Simetri pertürbatif değil; Hamiltonyen'in kesin simetrisi → genlikten bağımsız.
- Pratik sınır: iki demet arasındaki **diferansiyel** uyumsuzluk.
  3 mm genlikte ε = 1 μrad uyumsuzluk → fark 3.8×10⁻⁴.

### Test 8 — Gerçek demet (N parçacık, büyük rastgele açılar)

N = 20 parçacık/demet, σ_y = 2 mm, σ_y' = 66 μrad (betatron ~mm):

- Tek parçacık sahte EDM: **±0.01…0.1 rad/s** — işaretli, açıyla ~lineer.
- **(ii) Eşlenmiş demet** (CCW flip = CW'nin (y, −y') aynası):
  demet-ortalaması farkı ~10⁻¹⁸ — parçacık-parçacık tam iptal.
- **(i) Bağımsız dolum** (gerçekçi): fark ~2×10⁻³ — yalnızca istatistik,
  ~σ_f·√(2/N) ile uyumlu.
- 10⁻⁹ hedefi için bağımsız dolumda **~10¹⁵ parçacık-ölçüm** gerekir.

### Test 9 — Yatay hizalama hatası (dx)

- Saf dx (dy=0): dikey sahte EDM **tam 0** — lineer kafeste düzlemler bağımsız.
- dx + dy birlikte: çapraz terim doğar (dx=10 μm → sahte EDM 3200× büyür;
  kaynak: yatay COD düzeltilmediği için doğan yatay betatron × dy kuplajı).
- (CW,n)−(CCW,f) farkı: **1.4×10⁻¹⁵** — iptal çapraz terimi de siler (1.5×10⁸× bastırma).

## 4. Neden pratik değil?

Kick ile oturtma **tek-parçacık idealizasyonudur** ve gerçek deneyde üç ayrı
duvara çarpar:

1. **Sonlu emittans:** Tek kick yalnızca demetin ağırlık merkezini kapalı
   yörüngeye koyabilir. Demetin σ_y ~ mm'lik yayılımı vardır; her parçacık
   kendi betatron genliğiyle salınır. Parçacık başına sahte EDM ~0.01–0.1
   rad/s mertebesindedir ve hiçbir kick bunu topluca sıfırlayamaz.

2. **Tolerans:** Tek parçacık için bile 10⁻⁹ hedefi mikron-altı injection
   hassasiyeti ister (Test 1: 1 μm → 30 000× taban aşımı).

3. **İstatistik:** Demet ortalaması alındığında bile, iki bağımsız dolum
   birbirinin tam aynası olmadığından fark istatistik-sınırlı kalır
   (Test 8: ~σ·√(2/N) → 10⁻⁹ için ~10¹⁵ parçacık-ölçüm).

## 5. Asıl çalışan mekanizma: ayna simetrisi

Testlerin ortak dersi: sahte EDM iptalini sağlayan şey oturtma değil,
**(CW, normal) ↔ (CCW, flip) ayna simetrisidir**:

- Quad-flip manyetik kick işaretini çevirir; ışın tersinmesi de çevirir.
  İkisi birlikte → her fiziksel noktada **aynı kick** → iki konfigürasyon
  birbirinin tam aynasıdır.
- Bu simetri **kesindir** (pertürbatif değil): her genlikte (9 mm betatron'da
  bile), oturtma olmadan, dx+dy karışık hatalarda dahi parçacık-parçacık iptal
  sağlar.
- Gerçek EDM farkta 2D olarak korunur; sahte EDM silinir.

**Pratik sınır** bu nedenle injection hassasiyeti değil, iki karşı-dönen
demetin **birbirinin ne kadar iyi aynası olduğudur**:

- Kicker quad'larla birlikte çevrilirse kicker hatası zararsız (ortak mod).
- Demet dağılımları eşleşmezse (ayrı dolumlar) artık istatistikseldir ve
  ancak yüksek N ile bastırılır. Omarov'un ~10⁻⁹ tabanının olası kaynağı budur.

## 6. Sonuç ve yön

- Kick ile COD'ye oturtma: tek parçacıkta çok etkili (10⁵–10⁶× kazanç) ama
  gerçek demette **uygulanabilir değil**.
- CW/CCW+flip farkı: oturtma gerektirmeden sahte EDM'yi simetri ile siler;
  deney tasarımında çaba **demet eşleştirmesine** (aynı optik, çevrilen kicker,
  ortak injection hattı) ve **istatistiğe** harcanmalı.
- Kalan artık için tamamlayıcı yol: zıt işaretli Fourier modlarıyla
  **harmonik telafi** (Test B — `test_harmonic_cancellation.py` ve devamı).
