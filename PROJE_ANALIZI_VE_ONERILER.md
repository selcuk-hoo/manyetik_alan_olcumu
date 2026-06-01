# pEDM Quad Hizalama Projesi: Fizik Analizi ve Öneriler

---

## 1. Projenin Fizik Özeti

Bu proje, **proton Elektrik Dipol Momenti (pEDM)** deneyinin
alternating-gradient (FODO) halka tasarımında quad mıknatıslarının
hizalama hatalarını BPM ölçümlerinden geri çatma problemini inceler.
Halka 24 FODO hücresinden oluşur; 48 QF/QD quad ve 48 BPM içerir.

**Temel denklem:**

$$
\mathbf{y} = R\,\Delta q + \mathbf{b} + \boldsymbol{\eta}
$$

- $R$ (48×48): Courant-Snyder tepki matrisi, $\kappa(R) \approx 160$
- $\mathbf{b}$: BPM elektronik ofsetleri, statik ama bilinmez (~300 μm)
- $\boldsymbol{\eta}$: BPM gürültüsü (~1 μm)
- Hedef hassasiyet: **10 μm** (Omarov vd., PRD 2021)

Offset sinyali boğduğundan doğrudan $R^{-1}\mathbf{y}$ çalışmaz.
Çözüm: gradient modülasyonu ile $\mathbf{b}$'yi fark alarak iptal etmek.

---

## 2. Başarılan Sonuçlar

### 2.1 Simülasyon altyapısı ✓

- **GL4 simplektik entegratör** (`integrator.cpp`): Dördüncü mertebe
  Gauss-Legendre yöntemi; enerji korunum hatası RK4'e göre belirgin
  biçimde düşük.
- Courant-Snyder **analitik tepki matrisi** ile finite-difference
  **sayısal tepki matrisi** arasında %2-7 uyum doğrulandı.
- `params.json` tabanlı esnek parametre yönetimi: BPM gürültüsü,
  ofset, quad/dipol tilt, harmonik hata üretimi hepsi parametrik.

### 2.2 Uniform k-modülasyon ✓

Tüm 48 quad aynı anda %2 modüle edildiğinde:

| Metrik | Değer |
|--------|-------|
| $\kappa(\Delta R)$ | ~160 |
| Rekonstrüksiyon RMS | **6.6 μm** |
| Korelasyon | 0.993 |

10 μm hedefinin altında, mükemmel. Sınır: gerçek bir hızlandırıcıda
tüm güç kaynaklarını eşzamanlı modüle etmek operasyonel açıdan zordur.

### 2.3 Ofset-gürültü düalitesi teoremi ✓

**Kritik teorik katkı.** BPM ofsetini iptal eden, ön-yargısız, lineer
her estimator sınıfında tek çözümün $\Delta R^{-1}$ olduğu kanıtlandı.
Bu estimatorün gürültü büyütme faktörü:

$$
\|\Delta R^{-1}\| \sim \frac{\|R^{-1}\|}{\varepsilon}
$$

$\varepsilon = 0.02$ için ~50 kat büyütme. Bu yapısal bir alt sınırdır;
herhangi bir k-mod tabanlı anlık estimator bu sınırın altına inemez.

### 2.4 Drift modu ✓

**En güçlü pratik sonuç.** Problemi yeniden tanımlayarak bu sınırın
dışına çıkıldı:

$$
\widehat{\delta q}(t) = R^{-1}\bigl(\mathbf{y}(t) - \mathbf{y}_0\bigr)
$$

Ofset iki ölçüm arasında sabit kaldığından gradient farkı yerine
**zaman farkı** ile iptal olur. $\kappa(R) \approx 160$ küçük olduğundan
gürültü büyütmesi sınırlı kalır.

| Senaryo | RMS hata |
|---------|----------|
| Mutlak $R^{-1}\mathbf{y}$ | 170-200 μm (ofset baskın) |
| Drift modu | **6.5 μm** |

6 sistematik testte (düzenlileştirme, mod transfer, ters-suç,
BPM ofseti, uzun vadeli kararlılık, adil karşılaştırma) doğrulandı.

### 2.5 FODO antisimetri parametrelendirmesi ✓

Aynı FODO hücresindeki QF ve QD'nin **zıt işaretli** kaydırılması
(antisimetrik desen) COD'da güçlü sinüs desen üretir. Simetrik kayma
ise kick'leri iptal eder. Fourier baz matrisi bu fiziksel yapıyla
tutarlı tanımlandı:

$$
F_k[j] = (-1)^j \cdot \left\{1,\; \cos\!\left(\frac{2\pi k \lfloor j/2\rfloor}{24}\right),\;
\sin(\ldots)\right\}
$$

### 2.6 Hedefli Fourier rekonstrüksiyonu ✓

İki-quad k-mod ($j_1 = 3$, $j_2 = 9$) ile doğru harmonik baz
seçildiğinde:

| Baz | $\kappa(\Delta R \cdot F)$ | RMS hata | Korelasyon |
|-----|---------------------------|----------|------------|
| Direkt $\Delta R^{-1}$ | — | 107 μm | 0.03 |
| TSVD | — | 78 μm | 0.16 |
| Geniş Fourier $k = 1..4$ | 1.3×10⁴ | 35 μm | 0.88 |
| **Sıkı Fourier $\{2, 4\}$** | **186** | **0.02 μm** | **1.000** |

**Ana bulgu:** Baz tam doğru harmoniklere hizalanırsa koşul sayısı
1.3×10⁴'ten 186'ya düşer; hata 35 μm'den 0.02 μm'e iner.
Bu, "bilgi vardı ama yanlış parametrelendirme onu gizliyordu"
gerçeğini somutlaştırıyor.

### 2.7 Çok-konfigürasyon yığma çerçevesi ✓

Tek-quad kmod başına rank ~1 katkı sağlar. 3 bilinmeyen için
(k=0, k=2 → DC + cos + sin) 3 bağımsız konfig gerekir.
$j = 1, 3, 9$ seçimi (k=2 moduna göre projeksiyon değerleri:
1.00, 0.87, −0.50) lineer bağımsızlığı garanti eder. Yığılmış
sistem (144×3) rank ~3, belirlenmiş ve aşırı-belirlenmiş.

### 2.8 Greedy ve LASSO başarısızlığının analizi ✓

**Greedy** ile rank-2 ΔR'de yanlış harmonik seçimi (tekil vektör
ötüşmesi) matematiksel olarak türetildi ve sayısal olarak doğrulandı.

**LASSO** ile rank-2 sistemde RIP koşulunun ihlali nedeniyle tüm
katsayıların sıfıra itildiği gösterildi.

---

## 3. Başarılamayan / Açık Kalan Sonuçlar

### 3.1 Örgü modeli hatasının drift moduna etkisi — TAMAMLANMADI

**En kritik açık nokta.** Drift modunun 6-7 μm performansı
$R^{-1}$'in doğruluğuna bağlıdır. Gerçek halkada:
- β-beat (beta fonksiyonu sapmalar)
- Tune kayması
- Faz ilerleme hatası

gibi örgü bozulmaları $R$ matrisini yanlış kılar ve drift tahminini
bozar. Kaç μm model hatası kaç μm rekonstrüksiyon hatası yaratır?
Bu sorunun cevabı yöntemin pratikte kullanılabilirliğini belirler.
*(README'de Test 8 olarak listelenmiş, tamamlanmamış.)*

### 3.2 Tilt taraması — TAMAMLANMADI

Quad ve dipol tilt'lerinin (skew bileşeni → x-y kuplajı) drift modu
performansına etkisi Test 6'da yalnızca 0.2 mrad sabit seviyede
incelendi. Kırılma noktası ve tilt-hassasiyet eğrisi belirlenmedi.
*(README'de Test 7 olarak listelenmiş.)*

### 3.3 Hata kaynağı ayrıştırması — TAMAMLANMADI

Test 6'daki 6-7 μm hatanın BPM gürültüsünden mi, model
uyumsuzluğundan mı, tilt'ten mi kaynaklandığı nicelenmiş değil.
Her hata kaynağı ayrı ayrı kapatılarak katkıların ayrıştırılması
yapılmadı.

### 3.4 Çok-konfigürasyon yığma sayısal doğrulaması — KISMİ

Çerçeve ve fikir detaylı belgelendi (README, YÖNTEM.md), kod altyapısı
hazır, ama gerçek bir çalışmada rank ~3 elde edildiğinin ve 3
harmonik katsayısının (DC, cos₂, sin₂) tutarlı biçimde geri
çatıldığının sayısal doğrulaması eksik.

### 3.5 Adaptif harmonik tespiti — KISMİ

Greedy rank ≥ 3'te daha güvenilir çalışacak ama bu durum için
kapsamlı test yapılmadı. Harmonik tespiti için otomatik model
seçim kriteri (BIC, F-test, cross-validation) uygulanmadı.

### 3.6 Lineer model dışı etkiler — TARTIŞILMADI

- BPM gain hataları (~1%)
- BPM roll (dönme)
- Sekstupol feed-down
- Fringe alanlar
- Manyetik histerezis
- RF akım dalgalanması

Bu etkiler şu an modelde yok. Gerçek halkada ne kadar hata yaratır,
bilinmiyor.

---

## 4. Gelecek Çalışmalar İçin Öneriler

### Öneri 1 — Test 8: β-beat hassasiyeti (öncelik: YÜKSEK)

Drift modunun pratikte kullanılabilirliği bu teste bağlı.

**Yöntem:** Nominal lattice ile hesaplanan $R$ matrisini sabit tut.
Simülasyonda beta fonksiyonunu yapay olarak boz:

$$
\beta(s) \to \beta(s)\bigl(1 + \delta_\beta \cos(2\pi s / C)\bigr)
$$

$\delta_\beta \in [0.5\%, 5\%]$ taraması yap. Her noktada drift mode
hatası ölç. Beklenen çıktı: "X% beta-beat → Y μm ek hata" eğrisi.

**Pratik önemi:** LOCO kalibrasyonu gerçek halkalarda %1'in altında
β-beat sağlar. Bu seviyede drift modunun hedef hassasiyeti koruyup
korumadığı makale için kritik bir sonuçtur.

---

### Öneri 2 — Çok-konfigürasyon yığma doğrulaması (öncelik: ORTA)

Halihazırda yazılı olan kod altyapısını (`reconstruction.py`,
`build_response_matrix.py --config N`) uçtan uca çalıştırıp sonuçları
belgele.

**Kontrol edilecekler:**
- 3 konfig yığıldığında rank gerçekten ~3 oluyor mu? (SVD ile ölç)
- k=0 + k=2 katsayıları ($a_0$, $a_{2c}$, $a_{2s}$) hatasız geri
  çatılıyor mu? Her katsayı için RMS ve korelasyon raporla.
- Gürültü 1 μm ile yığılmış sistemin tahmin hatası teorik
  $\sigma_n / \sqrt{N_c}$ ölçeklemesiyle uyuşuyor mu?

---

### Öneri 3 — BPM ofset uzun-vadeli kararlılığı (öncelik: ORTA)

Drift modu "ofset sabit kalır" varsayımına dayanır. Test 5'te
geçiş noktası ~2 μm/epoch olarak bulundu.

**Yapılabilecekler:**
- pEDM BPM donanımı için beklenen ofset kayma hızını literatürden
  derle (termal katsayı, mevcut deneyimler).
- Kayma hızı ~ 0.1 μm/saat ise drift modunun hangi zaman diliminde
  yeniden kalibrasyona ihtiyaç duyduğunu hesapla.
- Alternatif: her N epoch'ta iki-gradient sürpriz ölçüm ile drift modu
  tutarlılığını doğrulayan hibrit mod.

---

### Öneri 4 — Yapay sinir ağı karşılaştırması (öncelik: ORTA)

Paralel çalışmada aynı problem YSA ile ele alınıyor. Bu projenin
ürettiği referans çerçeve (6-7 μm drift modu performansı, 6 sistematik
test, ofset-gürültü düalitesi teoremi) YSA ile adil karşılaştırma
için hazır.

**Önerilen karşılaştırma matrisi:**

| Senaryo | Drift modu | YSA |
|---------|-----------|-----|
| BPM gürültü artışı | ? | ? |
| β-beat %2 | ? | ? |
| Bilinmeyen harmonik desen | ? | ? |
| Anlık hizalama (mutlak) | — | ? |

YSA'nın avantajı mutlak hizalamaya erişebilmesi (ofset-gürültü
dualitesi dışında). Dezavantajı: eğitim setinin dışına çıkıldığında
genelleme garantisi yok.

---

### Öneri 5 — Hata barları ve bootstrap analizi (öncelik: DÜŞÜK)

Şu an rekonstrüksiyon hataları deterministik senaryolarda tek koşumla
ölçülüyor. Gerçek deneysel kullanım için:

- 48 BPM ölçümünde bootstrap (N=500 tekrar, gürültü yeniden örnekle)
- Her Fourier katsayısı için %95 güven aralığı
- "Hangi harmonik tespiti istatistiksel olarak anlamlı?" kriteri

Bu analiz olmadan tek bir rekonstrüksiyon koşumunun ne kadar güvenilir
olduğu bilinmiyor.

---

### Öneri 6 — Sekstupol feed-down etkisi (öncelik: DÜŞÜK)

Mevcut modelde `sextSwitch = 0`. Sekstupol mıknatıslar hizalanmamış
quad yakınına konumlandığında, sekstupol gradyeni ile quad hatası
arasında çapraz terim ürer (feed-down). Bu etki:

$$
\Delta y_{\text{eff},j} \approx dy_j + \frac{K_{2,j}}{K_{1,j}}\,dx_j\,dy_j
$$

Etkinin büyüklüğü pEDM örgüsünde ne kadardır? `sextSwitch = 1` ile
sekstupol açık haldeki drift modu performansı ölçülebilir.

---

### Öneri 7 — Gerçek yer hareketi ile test (öncelik: DÜŞÜK / UZUN VADELI)

Şu an quad hataları rastgele Gaussian veya düşük-harmonikli Fourier
deseniyle üretiliyor. Gerçek tünel yerleşim hareketleri:

- Düşük frekanslı (günlük-mevsimlik termal)
- Jeolojik kayma
- Sismik titreşim

CERN, KEK gibi kurumların yayımladığı tünel yer hareketi veri setleri
kullanılarak gerçekçi hizalama hatası desenleri üretilebilir ve
drift modunun bu senaryolardaki başarımı test edilebilir.

---

## 5. Kısa Özet

| Alan | Durum |
|------|-------|
| Simülasyon altyapısı (GL4, BPM modeli) | **Tamamlandı** |
| Ofset-gürültü düalitesi teoremi | **Tamamlandı** |
| Drift modu 6 testle doğrulandı | **Tamamlandı** |
| Hedefli Fourier (0.02 μm, ideal) | **Tamamlandı** |
| FODO antisimetri parametrelendirmesi | **Tamamlandı** |
| Çok-konfigürasyon yığma çerçevesi | **Kısmi** |
| β-beat / model hatası etkisi (Test 8) | **Açık** |
| Tilt taraması (Test 7) | **Açık** |
| YSA ile karşılaştırma | **Açık** |
| Lineer model dışı etkiler | **Açık** |

En yüksek öncelikli açık konu **Test 8 (β-beat hassasiyeti)**dir.
Bu testin sonucu olmadan "drift modu 10 μm hedefine ulaşır mı?" sorusu
gerçek operasyonel koşullar için yanıtsız kalmaya devam eder.
