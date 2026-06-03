# Makalenin Fizik ve Matematik Altyapısı — Eğitici Rehber

> **Bu dosya** `paper_draft.tex` içeriğini teknik olmayan bir okuyucunun da
> takip edebileceği biçimde anlatır. Her bölüm önce "ne yapıyoruz ve neden",
> sonra matematiksel detay verir.

---

## İçindekiler

1. [Problem: Yanlış EDM nedir?](#1-problem)
2. [Neden BPM ofsetleri her şeyi bozuyor?](#2-bpm-ofseti)
3. [Klasik çözüm: K-modülasyonu (kmod) ve sınırları](#3-kmod)
4. [Bizim yaklaşımımız: Fourier-tepki projeksiyonu](#4-yontem)
5. [Neden k=2? — Spin takibiyle doğrudan kanıt](#5-k2-kanit)
6. [Yöntemin nasıl çalıştığı — üç sayısal mülk](#6-uc-mulk)
7. [Geri çatım performansı](#7-performans)
8. [BPM ofseti sağlamlığı — beyazlık varsayımı gerekmez](#8-ofset-saglamligi)
9. [Model hatası toleransı](#9-model-hatasi)
10. [Kapsam: hedefli kestirici, harmonik dedektörü değil](#10-kapsam)
11. [Açık problemler](#11-acik)

---

## 1. Problem: Yanlış EDM nedir? <a name="1-problem"></a>

### Fiziksel arka plan

Proton EDM deneyi, protonun spin vektörünü halka boyunca dondurur
(frozen-spin rejimi). Gerçek bir EDM varsa spin, yatay düzlemden dikey
eksene doğru yavaşça döner — bu dikey spin bileşeni $S_y$'nin zamana göre
doğrusal artışı olarak ölçülür:

$$\text{EDM sinyali} \equiv \frac{dS_y}{dt} \neq 0.$$

### Sorun: yanlış EDM

Dikey yönde hizalama hatası olan bir quadrupol, ışını saptırır. Bu saptırma
**dikey kapalı-yörünge bozulmasına (COD)** yol açar. COD boyunca parçacık
radyal elektrik alan bileşenleri görür — bunlar spini gerçek EDM gibi
döndürür, yani $dS_y/dt \neq 0$ üretir. Buna **yanlış EDM (false EDM)**
denir.

> **Anahtar nokta:** Quad hizalama hatası ölçülüp düzeltilmezse EDM ölçümü
> sistematik bir hata taşır. 48 quadupolun her birinin dikey konumunun
> ~10 μm hassasiyetle bilinmesi gerekir.

### Lineer model

$\mathbf{y} \in \mathbb{R}^{48}$ BPM okuması, $R$ 48×48 yörünge tepki
matrisi, $\Delta q$ quadrupol hizalama hataları, $\mathbf{b}$ BPM
elektronik ofseti (~300 μm, sabit ama bilinmez), $\boldsymbol{\eta}$
okuma gürültüsü (~1 μm) olmak üzere:

$$\mathbf{y} = R\,\Delta q + \mathbf{b} + \boldsymbol{\eta}.$$

Eğer $\mathbf{b}$ biliniyorsa $\Delta q = R^{-1}(\mathbf{y} - \mathbf{b})$
doğrudan çözülürdü. Ne yazık ki bilinmiyor.

---

## 2. Neden BPM ofsetleri her şeyi bozuyor? <a name="2-bpm-ofseti"></a>

### Koşullanma sayısı (condition number)

$\mathbf{b}$'yi bilmeden sadece $\mathbf{y}$'yi ters çevirdiğimizde
çözüm $R^{-1}\mathbf{y}$ içinde $R^{-1}\mathbf{b}$ kirliliği taşır.
Matrisin koşullanma sayısı $\kappa(R) \approx 249$ olduğundan:

$$\|\delta(\Delta q)\| \leq \kappa(R)\,\frac{\|\mathbf{b}\|}{\|R\|} \approx 249 \times 300\,\mu\text{m} \sim \text{mm düzey}$$

Bu, hedef hassasiyetin (10 μm) iki mertebe üzerinde. Yani doğrudan
ters çevirmek işe yaramaz.

---

## 3. Klasik çözüm: K-modülasyonu ve sınırları <a name="3-kmod"></a>

### Fikir

Halkayı iki farklı gradyan ayarında ölç ve farkı al:

$$\Delta\mathbf{y} = \mathbf{y}_2 - \mathbf{y}_1 = (R_2 - R_1)\,\Delta q + \text{gürültü}$$

Böylece $\mathbf{b}$ tam olarak yok olur (her iki ölçümde de aynı).

### Sorun

- **48 quadupol aynı anda modüle edilirse** $\Delta R = \varepsilon R$,
  koşullanma sayısı korunur → iyi çalışır ama operasyonel açıdan çok zor.
- **1-2 quadupol modüle edilirse** $\Delta R$ rank-yetersiz olur, koşullanma
  sayısı ~10⁶'ya fırlar → geri çatım tamamen bozulur.

### Bizim tercihimiz

$\mathbf{b}$'yi yok etmeye çalışmak yerine, soruyu daraltalım:
**48 boyutlu $\Delta q$'yu tam olarak bulmak zorunda mıyız?**

---

## 4. Bizim yaklaşımımız: Fourier-tepki projeksiyonu <a name="4-yontem"></a>

### Operasyonel hedef

Yanlış EDM'i baskın şekilde besleyen Fourier harmonikini ölç,
ardından onu uygun dipol düzeltici elemanlarla bastır.
Tüm 48 quadupolun konumunu bulmak gerekmez — yalnızca hangi
harmonikten ne kadar var sorusunu yanıtlamak yeterlidir.

### FODO-antisimetrik Fourier bazı

24 FODO hücreli halkada her hücre bir QF (odaklayan) ve bir QD
(dağıtıcı) quadupol içerir. Bunlar COD'a zıt işaretle katkıda
bulunduğundan bazımız bunu hesaba katar:

$$F_k[j] = (-1)^j \cos\!\left(\frac{2\pi k \lfloor j/2 \rfloor}{24}\right),
\quad j = 0,\ldots,47, \quad k = 0,1,\ldots,12.$$

$(-1)^j$ terimi QF/QD değişimini, $\lfloor j/2\rfloor$ hücre indeksini
yansıtır. $\Delta q = F\hat{a}$ yazarak model:

$$\mathbf{y} = \underbrace{RF}_{M}\,\hat{a} + \mathbf{b} + \boldsymbol{\eta}.$$

### Hedefli projeksiyon

$k=2$ harmonik genliğini tek adımda kestirmek için:

$$\hat{a}_{k=2} = \frac{M_{k=2}^T\,\mathbf{y}}{\|M_{k=2}\|^2}$$

yeterlidir. Burada $M_{k=2} = R\,F_{k=2}$ bir 48-boyutlu vektördür
(k=2 quad bozunumunun yörünge etkisi).

### Oracle-free çalışma: CLEAN

Hangi harmoniklerin mevcut olduğu bilinmiyorsa CLEAN döngüsü kullanılır
(radyo astronomisinden ödünç alınmış). Her adımda tüm aday modlara
projeksiyon yapılır, en büyük katkı seçilir, $g=0.2$ kazançla çıkarılır.
CLEAN ve doğrudan en küçük kareler, $k=2$ kestirimi için istatistiksel
olarak özdeştir (200 Monte Carlo'da hata oranı 1.0×). CLEAN yalnızca
oracle bilgisi gerektirmemesi açısından avantajlıdır.

---

## 5. Neden k=2? — Spin takibiyle doğrudan kanıt <a name="5-k2-kanit"></a>

### Ölçüm zorluğu

Gerçek yanlış-EDM kayması ~10⁻⁹ rad/s düzeyindedir. Ama $S_y$'nin
döngü içi salınımı ~10⁻⁵ düzeyindedir — sinyal, gürültünün 10⁴ katı
altında. Naif sürekli Savitzky-Golay eğimi: örnekleme-bağımlı ve
işaret-kararsız.

### İki adımlı çözüm

**(i) Kapalı-yörünge fırlatması:**
Newton adımı ile parçacık, bozulmuş kapalı yörüngeden tam olarak
fırlatılır. Bu, sabit bir azimutte betatron varyansını minimize eden
bir 2D optimizasyon:

$$v^* = \arg\min_{v} \text{Var}_{j}\!\left[y_{\text{poincaré}}^{(j)}\right]$$

Hessian'daki çapraz terim ($H_{yp}$) doğru ele alındığında tek Newton
adımı, rezidü betatron genliğini ~10⁻¹⁴ m'ye indirir (9 mertebe iyileşme).

**(ii) Stroboskopik örnekleme:**
$S_y$, her turda sabit bir azimutte **bir kez** örneklenir. Periyodik kısım
tam olarak iptal olur; yalnızca seküler kayma kalır. Eğim, düzleştirme
veya örnekleme hızı seçimi olmadan doğrudan doğrusal fite uyar.

### Sonuç tablosu

| k | CO genliği [mm] | dS_y/dt [rad/s]       | Oran |
|---|-----------------|----------------------|------|
| 0 | 0.058           | +1.80×10⁻¹⁰          | 0.13 |
| 1 | 0.070           | +4.93×10⁻¹⁰          | 0.34 |
| **2** | **0.198**   | **+1.44×10⁻⁹**       | **1.00** |
| 3 | 0.088           | −6.64×10⁻¹⁰          | 0.46 |
| 4 | 0.028           | −1.98×10⁻¹⁰          | 0.14 |
| 5 | 0.014           | −9.92×10⁻¹¹          | 0.07 |

k=2 baskındır: k=1'den 2.9×, k=3'ten 2.2× büyük. Bu Omarov et al.
(PRD 105, 032001, 2022) Şekil 7-8'deki N=2 tepe noktasıyla nitelik
olarak tutarlıdır (halka tasarımları farklı; doğrudan karşılaştırma yoktur).

### Sayısal sağlamlık testleri

| Test | Sonuç | Yorum |
|------|-------|-------|
| Null (A=0) | dS_y/dt = 0.0 | Sahte arka plan yok |
| dt: 1e-11 → 5e-12 | k=2'de %2.7 değişim | Entegrasyon yakınsadı |
| İşaret: A → −A | Oran = −1.000 | Tam birinci-mertebe kuplaj |
| Spin normu | max\|‖S‖−1\| = 9.9e-13 | Korunuyor |
| Genlik: A=10 → 20 μm | Oran = 2.39 (beklenen 2.0) | %20 nonlineer; fiziksel |

---

## 6. Yöntemin nasıl çalıştığı — üç sayısal mülk <a name="6-uc-mulk"></a>

### (1) Rezonans kuvvetlendirmesi

Dikey tune $Q_y \approx 2.68$ ile $k=2$ modu (hücre başına 2 tam dönme)
rezonansa en yakın moddur. Bu, tepki matrisi $R$ ile $k=2$ Fourier modunun
haritalandığı yörünge normunun 34× büyümesi anlamına gelir:

| k | \|\|RF_k\|\| | \|\|M_k\|\| = √24 × \|\|RF_k\|\| |
|---|------------|----------------------------------|
| 1 | 8.8        | 43.0                             |
| **2** | **34.1** | **167**                        |
| 3 | 8.9        | 43.6                             |
| 4 | 3.2        | 16                               |
| 6 | 1.1        | 5.5                              |
| 8 | 0.59       | 2.9                              |

$\|M_{k=2}\| = 167$ demek: BPM ofseti $\mathbf{b}$'nin $k=2$ kestirimine
kirletme etkisi $\delta a_{k=2} = (\mathbf{b} \cdot \hat{m}_{k=2}) / 167$
ile **167 ile bölünür**. Karşılaştırma: $k=4$ için aynı etki 16 ile bölünür,
yani k=2 kestirimi 10.6× daha dayanıklıdır.

### (2) Yörünge düzeyinde genlik hiyerarşisinin tersine dönmesi

Bozunumu tasarlı durum: $k=2$ sadece 10 μm; kontaminantlar $k=4,6,8$
200-300 μm. Ama rezonans kuvvetlendirmesi hiyerarşiyi yörünge uzayında
tersine çevirir:

| k | Hizalama hatası | Yörünge normu | Kazanç |
|---|-----------------|--------------|--------|
| 2 | 10 μm           | 1669 μm      | 34.1   |
| 4 | 300 μm          | 4707 μm      | 3.2    |
| 6 | 300 μm          | 1651 μm      | 1.1    |
| 8 | 200 μm          | 577 μm       | 0.59   |

k=2, yörünge uzayında artık gömülü zayıf sinyal değil — k=6 ile karşılaştırılabilir
büyüklüktedir.

### (3) Harmonik yörünge tepkilerinin yaklaşık dik açıklığı

Farklı modların normalleştirilmiş yörünge desenleri neredeyse dikeydir:

$$|\langle RF_2, RF_{4,6,8}\rangle| \approx 0.01$$

Bu, en küçük kareler yönteminin $k=2$'yi kontaminantlardan temiz şekilde
ayırması anlamına gelir. Büyük $k=4,6,8$ genlikleri $k=2$ kestirimine
**sızmaz**. Bu, $\Delta R$'nin modları karıştırdığı kmod durumunun tam tersidir.

Bu üç mülk birlikte, BPM ofseti $\mathbf{b}$'nin istatistiksel yapısından
**bağımsız olarak** $k=2$ sütununa olan projeksiyonunun küçük kalmasını
garantiler (bkz. Bölüm 8).

---

## 7. Geri çatım performansı <a name="7-performans"></a>

### Test durumu

- Hedef: $k=2$, $A = 10\,\mu$m (doğru değer)
- Kontaminantlar: $k=4,6,8$ → 300, 300, 200 μm
- BPM ofseti: $\sigma_b = 100\,\mu$m/BPM (bağımsız Gaussian)
- 50 Monte Carlo realizasyonu

### Sonuç

$$A_{k=2} = 9.94 \pm 0.66\,\mu\text{m} \quad (\text{gerçek: } 10), \qquad \Delta\phi = 0.046\,\text{rad}$$

**Genlik hatası: %0.6.** Üç tanısal test mekanizmayı doğrular:

1. **Sinyal vs. ofset:** $\|R\Delta q_{k=2}\| = 1669\,\mu$m,
   $\|\mathbf{b}\| = 611\,\mu$m → k=2 yörüngesi ofset normundan büyük.

2. **Saf-ofset sızıntısı:** $\Delta q = 0$ (yalnızca ofset) ile CLEAN
   sahte $k=2$ genliği $0.72 \pm 0.41\,\mu$m verir; gerçek sinyal sızıntının
   ~14× üzerindedir.

3. **Ofset ölçeklenmesi:** $k=2$ hatası $\sigma_b$ ile doğrusal ölçeklenir
   (küçük projeksiyon hipotezi ile tutarlı). $\sigma_b = 300\,\mu$m'e
   ekstrapolasyon → ~2 μm hata, hedefin içinde.

---

## 8. BPM ofseti sağlamlığı — beyazlık varsayımı gerekmez <a name="8-ofset-saglamligi"></a>

### Anahtar sezgi: "parmak izi" eşleştirmesi

Algoritma yörüngenin **ne kadar büyük** olduğunu ölçmüyor;
yörüngenin **tek bir özel desene** ($k=2$ parmak izine)
**ne kadar benzediğini** ölçüyor.

Gerçek $k=2$ sinyali tam o desenin kendisi → tam uyuyor.
Rastgele BPM ofseti ise o desene benzemeyen bir karmaşa → neredeyse hiç uymuyor,
o yüzden "görünmez" kalıyor.

### Grafikle anlatım (fig6)

Algoritma aslında şunu yapar — her BPM noktasında ölçümü ($y_j$) bir
**şablonla** ($\hat{m}_{k=2,j}$ — k=2 yörünge parmak izi) çarpar ve toplar.
Bu işleme "projeksiyon" veya "eşleşme skoru" denir:

$$\hat{a}_{k=2} = \frac{\sum_j y_j \cdot \hat{m}_{k=2,j}}{\|M_{k=2}\|}$$

**Gerçek $k=2$ sinyali** durumunda (fig6 üst satır):
Yörünge zaten k=2 parmak izini taşıdığından, her $y_j$ ile $\hat{m}_{k=2,j}$
**aynı işarette** → çarpımlar hep pozitif → toplam büyük → sinyal görülür.

**Rastgele BPM ofseti** durumunda (fig6 alt satır):
Ofsetin şablonla ilgisi yoktur → çarpımların yarısı pozitif, yarısı negatif
→ **birbirini götürür** → toplam ≈ 0 → ofset görünmez.

Sayısal örnek ($\sigma_b = 300\,\mu$m):
- Gerçek $k=2$ sinyali (10 μm misalignment): **kestirilen 10.0 μm** ← mükemmel
- Rastgele ofset: **kestirilen ~1.3 μm** ← 167× bastırılmış

### Temel eşitsizlik

$M = RF$ uyum matrisi için **herhangi bir** deterministik ofset $\mathbf{b}$,
$k=2$ genlik kestirimine şu kadar kirletme katar:

$$\delta a_{k=2} = \frac{\mathbf{b} \cdot \hat{m}_{k=2}}{\|M_{k=2}\|} = \frac{\mathbf{b} \cdot \hat{m}_{k=2}}{167}$$

Bu formülde $\mathbf{b}$'nin dağılımı hakkında **hiçbir varsayım** yoktur.
Güvenli limit: $\|\mathbf{b}_\parallel\| < 1669\,\mu$m (gerçekçi değerin 5.6×'ı).

### White ofset spektrumu (fig5)

fig5 ise şunu gösteriyor: $\sigma_b = 300\,\mu$m'lik rastgele bir ofset,
estimatörden geçince **tüm harmoniklerde düz ve geniş-bantlı bir taban** üretir —
hiçbir $k$'da sahte tepe yok. Ve taban **tam k=2'de en düşük** (2.2 μm),
çünkü $\|M_{k=2}\| = 167$ en büyük sütun normu. Yanlış EDM'i besleyen mod,
ofset kontaminasyonuna karşı en korunaklı olandır.

### En kötü durum testi

$\mathbf{b} = A\hat{m}_{k=2}$ (ofset tam k=2 yörünge deseniyle hizalı — en kötü durum).
Teori: $\delta a = A/167$.

| A [μm] | Teori [μm] | CLEAN [μm] | lstsq [μm] |
|--------|-----------|-----------|-----------|
| 50     | 0.30      | +0.30     | +0.30     |
| 100    | 0.60      | +0.60     | +0.60     |
| 200    | 1.20      | +1.20     | +1.20     |
| 300    | 1.80      | +1.80     | +1.80     |
| 1669   | 10.0      | +10.0     | +10.0     |

Teori ile simülasyon dört anlamlı basamakta örtüşmektedir.

---

## 9. Model hatası toleransı <a name="9-model-hatasi"></a>

Gerçek halkada $R$ yalnızca yaklaşık olarak bilinir. Gradyan hatası
$\varepsilon \sim \mathcal{N}(0, \sigma_\text{model})$ ile modele
pertürbasyon:

$$R_\text{model} = R\,\text{diag}(1 + \varepsilon)$$

$k=2$ genlik hatası $\delta A_{k=2}/A_{k=2} \sim \sigma_\text{model} \times \kappa_2$
ile ölçeklenir.

**Sayısal hata bütçesi (fig4)** — üç BPM ofseti düzeyi karşılaştırması:

| Senaryo | σ_model = 0% | σ_model = 5% | σ_model = 10% |
|---------|------------|------------|--------------|
| σ_b = 0 (model hatası yalnız) | ~0 μm | ~0 μm | ~0 μm |
| σ_b = 100 μm | ~0.6 μm | ~0.6 μm | ~0.7 μm |
| σ_b = 300 μm | ~1.8 μm | ~1.8 μm | ~2.0 μm |

Temel çıkarım: **hata tamamen BPM ofseti tarafından belirlenir, model hatası etkisizdir.**
$\sigma_b = 300\,\mu$m bile $\sigma_\text{model} \lesssim 10\%$'de 10 μm hedefinin içindedir.
Bu tolerans, beta-beat / LOCO-tipi kalibrasyon ile kolaylıkla ulaşılabilirdir.

---

## 10. Kapsam: hedefli kestirici, harmonik dedektörü değil <a name="10-kapsam"></a>

Yöntem, **önceden belirlenmiş** bir hedef harmoniğin (burada $k=2$)
genlik ve fazını ölçer. Harmonik keşif aracı değildir. Bu bilinçli bir
kapsam tercihidir:

- Hedef $k=2$, Bölüm 5'teki spin takip sonucuyla önceden belirlenmektedir.
- Oracle-free senaryoda CLEAN, ofset kaynaklı sahte modları gerçek
  harmoniklerden yalnızca genliğe bakarak ayırt edemez; ama $k=2$
  **genliğini** bozmaz (modlar $R$ altında dikeydir).

> **Söylem özeti:** "48 boyutlu hizalama problemini çözmek zorunda
> değiliz. Yanlış EDM'i besleyen baskın modu ölçmek ve onu bastırmak
> yeterlidir."

---

## 11. Açık problemler <a name="11-acik"></a>

| Konu | Durum |
|------|-------|
| Okuma gürültüsü $\boldsymbol{\eta}$ tam hata bütçesine dahil edilmeli | Yapılacak |
| Oracle-free harmonik ayırt etme (sahte vs gerçek) | Yapılacak |
| Çok yörüngeli uzantı (yavaş kayıklı **b** ortalama düşürme) | Yapılacak |
| $R$ kalibrasyonu gerçek halkada ($\delta K/K$ → $k=2$ hata bütçesi) | Yapılacak |
| $\Delta q_{k=2}$ → spin-düzeyi sistematik bağlantısı | Yapılacak |
| Yatay düzlem ve skew kuplaj (quad tilt) | Yapılacak |
| $\sigma_\text{model}$ vs $k=2$ hatası grafik onayı | **Tamamlandı** (fig4) |
| BPM ofseti dayanıklılığı — parmak izi eşleştirme sezgisi | **Tamamlandı** (fig5, fig6) |
