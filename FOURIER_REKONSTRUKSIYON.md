# Fourier Tabanlı Hizalama Hatası Ölçümü: Pedagojik Rehber

> **Okuyucu varsayımı.** Tune, beta fonksiyonu, kapalı yörünge, Courant-Snyder
> formalizmi gibi temel hızlandırıcı kavramları biliniyor. Matris tersi,
> kondisyon sayısı, SVD, Fourier ayrıştırması gibi sayısal yöntemler ise
> yabancı ya da soluk kalmış. Bu belge önce fiziği, sonra matematiği,
> sonra ikisinin nerede birleşip nerede ayrıştığını anlatır.

---

## İçindekiler

1. [Neden bu problem önemli?](#1-neden-bu-problem-önemli)
2. [Simülasyon altyapısı: halka geometrisi ve GL4 entegratör](#2-simülasyon-altyapısı)
3. [Tepki matrisi: BPM'lerin "görme" biçimi](#3-tepki-matrisi)
4. [BPM ofseti: neden doğrudan çözüm çalışmaz](#4-bpm-ofseti)
5. [K-modülasyon hilesi: ofseti fark alarak yok et](#5-k-modülasyon-hilesi)
6. [Rank: bir matrisin kaç soruya cevap verebileceği](#6-rank)
7. [İki-quad kmod neden rank-2 üretir?](#7-i̇ki-quad-kmod-neden-rank-2-üretir)
8. [Kondisyon sayısı: gürültü nasıl büyür?](#8-kondisyon-sayısı)
9. [Fourier parametreleştirme: bilinmeyen sayısını azalt](#9-fourier-parametreleştirme)
10. [Baz seçimi: doğru harmonikler neden kritik?](#10-baz-seçimi)
11. [Targeted Fourier: ne zaman çalışır, ne zaman çalışmaz?](#11-targeted-fourier)
12. [Rank kısıtı: "sayım prensibi"](#12-rank-kısıtı)
13. [Çok-konfigürasyon yığma: çözüm yolu](#13-çok-konfigürasyon-yığma)
14. [Pratik rehber ve açık sorular](#14-pratik-rehber)

---

## 1. Neden bu problem önemli?

pEDM (proton Electric Dipole Moment) deneyi, protonun elektrik dipol
momentini ölçmeyi hedefler. Deney, protonların **frozen-spin** rejiminde
bir depolama halkasında döndürüldüğü ve spin yönünün zamanla sapıp
sapmadığının izlendiği bir düzeneğe dayanır.

Bu deneyin en büyük sistematik hatası şudur: eğer halka boyunca bir
kuadrupol mıknatıs ideal konumundan $dy$ kadar kaymışsa, o quad'dan
geçen demet ekstra bir dikey kick alır. Bu kick, halka boyunca birikir
ve demetin kapalı yörüngesini bozar. Ölçülen bu yörünge sapması, gerçek
bir EDM sinyalini taklit edebilir.

Omarov ve arkadaşlarının hesaplamalarına göre bu sahteciliği önlemek için
halkadaki her quad'ın dikey konumu **~10 μm** hassasiyetle bilinmek zorunda.
Mekanik ölçüm bu hassasiyete ulaşamaz; demete dayalı ölçüm gerekir.

Problem şu: halkada **48 kuadrupol** var, her birinin konumu bağımsız
olarak bilinmiyor ve ölçüm sırasında BPM'lerin elektronik ofseti sinyali
boğuyor. Bu belge, bu sorunun nasıl çözüleceğini ve neden tam çözülmediğini
anlatıyor.

---

## 2. Simülasyon Altyapısı

### Halka geometrisi

Halka, 24 adet **FODO hücresinden** oluşur. Her FODO hücresi şu sırayla
dizilmiştir:

```
QF → drift → arc → drift → QD → drift → arc → drift → (tekrar)
```

- **QF** (Quadrupole Focusing): K > 0, dikeyde odaklar, yatayda saçar.
- **QD** (Quadrupole Defocusing): K < 0, yatayda odaklar, dikeyde saçar.
- **arc**: elektrostatik yay deflektörü, demeti eğer.

24 FODO × 2 quad = **48 kuadrupol**. Her quad'ın yanında bir **BPM**
(Beam Position Monitor) vardır: toplam 48 BPM. BPM, demeti dikey ve
yatay olarak okur; çözünürlük ~1 μm, ama elektronik ofset ~300 μm
(bkz. §4).

Tune değerleri: $Q_y \approx 2.68$ (dikey), $Q_x \approx 2.68$ (yatay).
Bu, her turda demet ekseninin ~2.68 tam salınım yaptığı anlamına gelir.

### GL4 simplektik entegratör (`integrator.cpp`)

Parçacık hareketi `integrator.cpp` içinde `C++` ile simüle edilir.
Python kodu bu kütüphaneyi `ctypes` üzerinden çağırır (`integrator.py`).

**Neden simplektik?** Newton denklemlerini sayısal olarak çözen standart
yöntemler (Euler, Runge-Kutta 4) küçük de olsa **faz uzayı hacmini**
değiştirir. Hamiltonian mekaniğinde bu hacim korunmalıdır (Liouville
teoremi). Hacim zamanla küçülürse parçacık yapay olarak "dampe"
oluyormuş gibi görünür; büyürse yapay büyüme. Uzun süreli simülasyonlarda
bu hata birikir.

**GL4** (4. mertebe Gauss-Legendre), **simplektik** bir integrasyondur:
faz uzayı hacmini makinenin yuvarlama hatasına kadar tam olarak korur.
Bu, bir depolama halkası simülasyonu için doğru seçimdir.

**Ne hesaplar?** Her turn'de parçacığın $(x, x', y, y', t, \delta)$
koordinatları güncellenir. Her element (drift, quad, arc) parçacığa
kendi transferini uygular. Quad hizalama hatası `dy_j` varsa, parçacık
quad merkezinden $dy_j$ uzakta geçer ve ek bir kick alır.

**Tepki matrisi nasıl hesaplanır?** Her quad sırayla küçük bir miktar
($\delta = 10^{-4}$ m) kaydırılır, tam turdan sonra 48 BPM'deki konum
değişikliği ölçülür ve $\delta$'ya bölünür. Bu 48 simülasyon, 48×48
boyutlu tepki matrisinin **bir sütununu** verir:

$$
R[:,j] = \frac{y_\text{CO}(dy_j = \delta) - y_\text{CO}(0)}{\delta}
$$

48 sütun için 48 simülasyon → tam $R$ matrisi.

---

## 3. Tepki Matrisi

Tepki matrisi $R$, hizalama hataları ile BPM okumalarını birbirine bağlar:

$$
\mathbf{y} = R\,\Delta q + \mathbf{b} + \boldsymbol{\eta}
$$

- $\mathbf{y}$ (48 boyutlu): BPM okumaları.
- $R$ (48×48): tepki matrisi.
- $\Delta q$ (48 boyutlu): quad hizalama hataları — **aranan**.
- $\mathbf{b}$ (48 boyutlu): BPM elektronik ofsetleri — bilinmez, ~300 μm.
- $\boldsymbol{\eta}$ (48 boyutlu): BPM ölçüm gürültüsü — ~1 μm.

$R_{ij}$ elemanı şu anlama gelir: "*j* numaralı quad 1 m kaydırılırsa
*i* numaralı BPM'de kaç metre sapma okunur?" Courant-Snyder formalizmiyle:

$$
R_{ij} = \frac{\sqrt{\beta_i \beta_j}}{2\sin(\pi Q)}\,
          \cos\!\bigl(|\phi_i - \phi_j| - \pi Q\bigr)\cdot K_j L_j
$$

Burada $\beta_{i,j}$ beta fonksiyonu, $\phi_{i,j}$ faz ilerleme,
$Q$ tune, $K_j L_j$ quad'ın integre kick gücü.

**Kondisyon sayısı $\kappa(R) \approx 160$** — bu çok iyidir.
$R$'yi "terslemek" (tüm 48 quad hatasını aynı anda bulmak) pratikte
mümkündür, **eğer** BPM ofseti olmasa.

---

## 4. BPM Ofseti

Bir BPM'in ölçüm değeri:

$$
y_i^\text{ölçülen} = y_i^\text{gerçek} + b_i + \eta_i
$$

$b_i$: elektronik ofset — BPM'in "sıfır" noktasının mekanik merkezden
sapması. Tipik değer: **~300 μm**. Zaman içinde değişmez (saatler-günler
boyunca sabit), ama bilinmez.

Hizalama hatası kaynaklı orbit sapması: **~10 μm**.

Aradığınız sinyal (10 μm), arka plan (300 μm) tarafından **30 kat**
bastırılmış durumda. Doğrudan $\Delta q = R^{-1}\mathbf{y}$ çözümü
$R^{-1}\mathbf{b}$ ofset kirliliğini taşır ve 10 μm hassasiyet hedefini
tamamen bozar.

---

## 5. K-Modülasyon Hilesi

**Fikir:** $b$ her ölçümde aynı. Gradyanları iki farklı ayarda kurup
iki kez ölç, fark al:

$$
\mathbf{y}_1 = R_1\,\Delta q + \mathbf{b} + \boldsymbol{\eta}_1
\qquad
\mathbf{y}_2 = R_2\,\Delta q + \mathbf{b} + \boldsymbol{\eta}_2
$$

$$
\Delta\mathbf{y} = \mathbf{y}_2 - \mathbf{y}_1
= \underbrace{(R_2 - R_1)}_{\Delta R}\,\Delta q
+ \underbrace{(\boldsymbol{\eta}_2 - \boldsymbol{\eta}_1)}_{\text{gürültü}}
$$

$\mathbf{b}$ iptal oldu. Şimdi $\Delta R\,\Delta q = \Delta\mathbf{y}$
denklemini çözmek gerekiyor.

Ama bir sorun var: $\Delta R$, $R$'den çok daha kötü koşullanmış.
Neden? Çünkü iki konfigürasyon arasındaki fark küçük — gradyanları %2
değiştirince $\Delta R \approx \varepsilon R$ ($\varepsilon = 0.02$),
dolayısıyla $\kappa(\Delta R) \approx \kappa(R)/\varepsilon \approx
160/0.02 = 8000$. Pratikte **~27.000** çıkıyor.

Bu sayı ne demek: $\Delta R$'yi ters çevirmek, $R$'yi ters çevirmekten
~170 kat daha çok gürültü büyütür. 1 μm gürültü → 170 μm hata.
Bu da 10 μm hedefini aşıyor.

Ama bu, **tüm 48 quad birlikte modüle edildiğinde** geçerlidir.
Ya yalnız 1 veya 2 quad modüle edilirse?

---

## 6. Rank: Bir Matrisin Kaç Soruya Cevap Verebileceği

Rank'ı anlamak bu çalışmanın tamamını anlamak demektir.

### Fizik sezgisiyle rank

Bir 48×48 matris $A$, sizi 48 boyutlu bir uzaydan alıp başka bir
48 boyutlu uzaya götürür. Ama her matris bunu gerçekten "48 bağımsız
yönde" yapmaz. Bazı matrisler, girdiyi esasen daha düşük boyutlu bir
uzaya "sıkıştırır."

Rank, matrisin kaç bağımsız bilgi boyutu taşıdığını sayar.

**Somut örnek:** Aşağıdaki 3×3 matris düşünün:

$$
A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 3 & 6 & 9 \end{pmatrix}
$$

İkinci satır birincinin 2 katı, üçüncü satır birincinin 3 katı.
Yani $A$ aslında tek bir yön taşıyor: $\text{rank}(A) = 1$.

Bu matrisle bir $\mathbf{x}$ vektörünü çarpsanız, sonuç her zaman
$\mathbf{a} = (1, 2, 3)^T$ yönünde çıkar — başka hiçbir yön gözlemleme
imkânınız yok. $A\mathbf{x} = \mathbf{b}$'yi "terslemeye" çalışsanız,
$\mathbf{b}$'nin $\mathbf{a}$'ya dik bileşenleri için hiçbir bilgi
yoktur.

### Rank ve "kaç soru sorabilirsin"

$\Delta R$ rank-$r$ ise, $\Delta\mathbf{y} = \Delta R\,\Delta q$
ölçümünden yalnızca $\Delta q$'nun **$r$ bağımsız doğrusal bileşeni**
öğrenilebilir. 48 boyutlu $\Delta q$'nun geri kalan $(48-r)$ boyutu
tamamen ölçüm dışı kalır.

- **Tek-quad kmod**: rank ~1. $\Delta q$'nun yalnız 1 kombinasyonu ölçülür.
- **İki-quad kmod**: rank ~2. $\Delta q$'nun yalnız 2 kombinasyonu ölçülür.
- **48 quad (uniform) kmod**: rank ~48. $\Delta q$'nun tamamı ölçülür.

Uniform kmod pratikte çok zordur (tüm güç kaynakları aynı anda). Peki
1-2 quad ile elde edilen 1-2 boyutluk bilgiyi nasıl işe yaratırız?

---

## 7. İki-Quad Kmod Neden Rank-2 Üretir?

### Sezgi

İki quad'ın ($j_1$, $j_2$) gradyanı değişirken 46 quad sabit kalır.
$\Delta R = R_2 - R_1$ matrisini iki terime ayıralım:

**Doğrudan terim** (~$\varepsilon$ büyüklüğünde):
Her quad'ın kick gücü $K_j$'ye orantılıdır. Yalnız $j_1$ ve $j_2$'nin
gradyanı değiştiğinden, yalnız bu iki quad'ın kick değişir. Bu değişiklik
$\Delta R$'de iki "önemli" yön yaratır.

**Dolaylı terim** (~$\varepsilon^2$ büyüklüğünde):
İki quad'ın gradyanı değişince tüm örgünün optiği değişir (beta-beat,
tune kayması). Bu dolaylı etki tüm sütunlarda iz bırakır ama $\varepsilon$
kez daha küçük.

Sonuç: $\Delta R$'nin SVD spektrumuna baktığınızda iki baskın tekil
değer ($\sigma_1 \sim \sigma_2 \gg \sigma_3, \ldots$) görürsünüz.
Etkin rank 2'dir; kondisyon sayısı $\kappa \sim 10^6$.

### Faz uzayı sezgisi

Tek bir quad'ı modüle edince, BPM'lerde oluşan orbit değişim deseni
şu biçimdedir (Courant-Snyder formülünden):

$$
\Delta y_i \propto \sqrt{\beta_i}\,\cos\!\bigl(\phi_i - \phi_{j_1} - \pi Q\bigr)
$$

Bu, **tune frekansında** ($Q \approx 2.68$) titreşen bir cosinus — halka
boyunca $j_1$ konumuna göre. İkinci bir quad'ı ($j_2$) ekleyince bu
şekle başka bir titreşim eklenir. Sonuçta $\Delta\mathbf{y}$'nin "görme
yetisi" yalnızca bu iki yönün gerildiği 2 boyutlu bir alt uzayla sınırlı.

---

## 8. Kondisyon Sayısı: Gürültü Nasıl Büyür?

### Kondisyon sayısı nedir?

Bir $A$ matrisi için kondisyon sayısı:

$$
\kappa(A) = \frac{\sigma_\text{max}}{\sigma_\text{min}}
$$

$\sigma$'lar tekil değerler — matrisi farklı yönlerde ne kadar
"esnettiğinin" ölçüsü.

**Sezgi:** $\kappa$ büyük ise bazı yönlerde girdi çok küçük çıktı
üretir. O küçük çıktıyı tersleyince — $A^{-1}$ uygulayınca — küçük
bir gürültü bile büyük bir hata yaratır.

**Somut örnek:**

$$
A = \begin{pmatrix} 1 & 0 \\ 0 & 0.001 \end{pmatrix},
\qquad \kappa(A) = 1000
$$

$A\mathbf{x} = \mathbf{b}$'yi çözerken $b_2$'de 0.001 birim gürültü
varsa, $x_2 = b_2 / 0.001$ için bu gürültü 1 birime büyür — 1000 kat
amplifikasyon.

### Pratikte ne demek?

| Matris | $\kappa$ | 1 μm BPM gürültüsü → tahmin hatası |
|--------|----------|--------------------------------------|
| $R$ | ~160 | ~14 μm |
| $\Delta R$ (uniform kmod) | ~160 | ~14 μm |
| $\Delta R$ (iki-quad kmod) | ~10⁶ | ~10⁶ μm → anlamsız |
| $\Delta R \cdot F$ (geniş Fourier) | ~13.000 | ~13 mm |
| $\Delta R \cdot F$ (sıkı Fourier) | ~186 | ~26 μm |

Hedef 10 μm olduğundan, sıkı Fourier yaklaşımı sınırda — ama diğerleri
çok uzakta. Kondisyon sayısını düşürmek bu çalışmanın temel teknik hedefi.

---

## 9. Fourier Parametreleştirme: Bilinmeyen Sayısını Azalt

### Fizik ön bilgisi

Quad hizalama hataları nereden geliyor? Termal genleşme, tünel oturması,
zemin titreşimi, mıknatıs kurulumundaki sistematik hatalar… Bu mekanizmaların
ortak özelliği: **uzun dalgalı** bozulmalar üretirler. Tünel 400 metrelik
bir çevre boyunca yavaşça eğilir; 48. quad 1. quad'dan bağımsız, rastgele
bir konumda olamaz.

Bu fizik sezgisi şunu söylüyor: $\Delta q$'nun Fourier içeriği düşük
frekanslarda yoğunlaşmalı. Yani 48 bağımsız sayı yerine birkaç Fourier
katsayısıyla işi yapabiliriz.

### Matematiksel çerçeve

Quad indeksi $j = 0, 1, \ldots, 47$ için:

$$
\Delta q_j = a_0 + a_{1c}\cos\!\left(\frac{2\pi \cdot 1 \cdot j}{48}\right)
+ a_{1s}\sin\!\left(\frac{2\pi \cdot 1 \cdot j}{48}\right)
+ a_{2c}\cos\!\left(\frac{2\pi \cdot 2 \cdot j}{48}\right)
+ \ldots
$$

Bunu matris biçiminde yaz: $\Delta q = F\hat{a}$.

Burada:
- $F$ (48 × $n_\text{baz}$): **Fourier baz matrisi** — her sütun bir
  harmonik deseni
- $\hat{a}$ ($n_\text{baz}$ boyutlu): bulunmak istenen katsayılar

K-mod denklemine yerleştir:

$$
\Delta\mathbf{y} = \Delta R\,\Delta q = \Delta R\,F\,\hat{a}
\equiv M\,\hat{a}
$$

$M = \Delta R \cdot F$ matrisi 48 × $n_\text{baz}$ boyutlu.
Eğer $n_\text{baz} = 5$ seçilirse: 48 denklem, 5 bilinmeyen.
Sistem **aşırı belirlenmiş** (overdetermined) → en küçük kareler ile
çözülür:

$$
\hat{a} = (M^T M)^{-1} M^T \Delta\mathbf{y}
$$

ve geri çatım $\widehat{\Delta q} = F\hat{a}$.

### Neden bu yardımcı oluyor?

$\Delta R$'nin kondisyon sayısı $10^6$ idi. Ama $M = \Delta R \cdot F$
için ne?

$\Delta R$'nin "güçlü yönleri" ($\sigma_1, \sigma_2$ yönleri), tune'a
yakın düşük frekanslı modlara karşılık gelir. Eğer $F$'yi bu güçlü
yönlerle hizalı seçersek — yani $\Delta R$'nin zaten "görebildiği"
frekansları modellersek — $M$ yalnızca güçlü yönleri taşır ve
$\kappa(M) \ll \kappa(\Delta R)$.

Özet: Fourier parametreleştirme, 48 bilinmeyeni birkaç bilinmeyene
indirgeyerek hem sistemi belirlenmiş hale getirir hem de kondisyon
sayısını düşürür — **eğer baz doğru seçilirse.**

---

## 10. Baz Seçimi: Doğru Harmonikler Neden Kritik?

### Üç hata türü

**Bazda eksik harmonik:** Veri $k=2$ içeriyor ama bazda $k=2$ yok.
En küçük kareler bu katkıyı en yakın baz fonksiyonuna dağıtır —
var olan harmoniklerin katsayıları da kirlenip yanlış kestiriliyor.
Bunu şöyle düşünün: portakalı elmayla açıklamaya çalışmak gibi;
hem elma katsayısı hem portakal katsayısı anlamını yitiriyor.

**Bazda fazla harmonik:** Veri yalnız $k=2$ içeriyor ama bazda
$k=1, 2, 3, 4$ var. $k=1, 3, 4$ için boş sütunlar ekleniyor.
Bu sütunlar veriyi açıklamak için rekabet ediyor — kondisyon sayısı
şişiyor. 186'dan 13.000'e çıkmak gürültü büyütmeyi 70 kata çıkarıyor.

**Baz tam doğru:** Yalnızca veride gerçekten olan harmonikler bazda.
Sistem hem tam belirlenmiş hem de iyi koşullanmış. Gürültü bastırılmış.

### Sayısal karşılaştırma

Aşağıdaki tabloda test verisi yalnızca $k=2$ ve $k=4$ harmoniklerinden
oluşturuluyor (rastgele gürültü yok, tüm sinyal bu iki modda):

| Seçilen baz | $\kappa(M)$ | Hata RMS | Korelasyon |
|-------------|-------------|----------|------------|
| Direkt $\Delta R^{-1}$ (baz yok) | ~10⁶ | 107 μm | 0.03 |
| Geniş: $k = 1, 2, 3, 4$ | 13.000 | 35 μm | 0.88 |
| Tek yanlış: $\{k=4\}$ | 14 | 1466 μm | −0.38 |
| Tek doğru: $\{k=2\}$ | 1.1 | 37 μm | 0.89 |
| **Tam doğru: $\{k=2, k=4\}$** | **186** | **0.02 μm ★** | **1.000** |

Tam doğru baz ile geniş baz arasındaki fark: **2000 kat**. Bu farkın
kaynağı kondisyon sayısındaki 70 katlık artış — gürültü büyütmesi
kondisyon sayısıyla orantılı.

> **★ İDEALİZE SENARYO:** $\Delta q$ yalnızca $k=2$ ve $k=4$
> harmoniklerinden oluşuyor, rastgele arka plan **yok**, BPM gürültüsü
> simülasyon düzeyinde küçük. 100 μm RMS rastgele arka plan eklendiğinde
> 0.02 μm sonucu tamamen bozulur. Gerçekçi sınırlar için §11 ve §13'e
> bakın.

### Neden $\{k=4\}$ bazı patladı?

Veride $k=2$ ve $k=4$ var, $k=2$ baskın. Bazda yalnız $k=4$ varsa
en küçük kareler şunu yapmak zorunda: $k=2$'nin 48 BPM'deki katkısını
yalnız $k=4$ modlarıyla açıklamaya çalış. Bu matematiksel olarak mümkün
değil (iki farklı frekans) → katsayı absürd büyüyor, sonuç patıyor.

**Ders:** Bazda bir harmonik eksikse, LSQ onun yerine en yakın bazı
zorla kullanır; bu zorlama katsayıyı patlatır.

---

## 11. Targeted Fourier: Ne Zaman Çalışır, Ne Zaman Çalışmaz?

### Temel başarı koşulu

Targeted Fourier şu koşulda güvenilir çalışır:

> **Baz boyutu ≤ etkin rank($\Delta R$)**

Baz boyutu 2 (örn. cos₂ ve cos₄) ve rank 2 → tam belirlenmiş, mükemmel.
Baz boyutu 3 (DC + cos₂ + sin₂) ve rank 2 → yetersiz belirlenmiş,
katsayılar ayrıştırılamaz.

### Ne zaman mükemmel çalıştı?

**İdealize test senaryosu:** $\Delta q$ yalnız $k=2$ ve $k=4$
harmoniklerinden oluşturuldu, sin bileşenleri sıfır, random gürültü yok.
Baz: $\{k=2, k=4\}$, 4 sütun. Ama sin bileşenleri sıfır olduğu için
efektif bilinmeyen sayısı 2 (yalnızca $a_{2c}$ ve $a_{4c}$ sıfırdan
farklı). İki-quad kmod → rank 2.

Sonuç: 2 bilinmeyen, rank-2 ölçüm → tam belirlenmiş → **0.02 μm, kor = 1.000**.

Bu, biraz talihli bir örtüşmeydi: sin bileşenlerinin sıfır olması
efektif bilinmeyen sayısını düşürdü ve rank ile tam eşleşti.

### Ne zaman bozuldu?

Gerçekçi senaryoda iki ayrı sorun devreye girer. Bunları karıştırmamak
önemli: biri rank (kaç denklem var?), diğeri SNR (sinyal ne kadar gömülü?).

#### Sorun 1: Rank yetersizliği (sayım sorunu)

**Gerçekçi senaryo:** $\Delta q$ = k=0 ve k=2 smooth bileşenler.
Baz: $\{k=0, k=2\}$ → 3 sütun (DC + cos₂ + sin₂). İki-quad kmod → rank ~2.

3 bilinmeyen, 2 bağımsız denklem → **yetersiz belirlenmiş sistem.**

`reconstruction.py` bu durumu tespit eder ve şunu söyler:

```
NOT: etkin rank < baz boyutu → çözüm minimum-norm, katsayılar
     doğrudan yorumlanamaz; rekonstrükte edilmiş profil anlamlı.
```

"Minimum-norm çözüm" ne demek? `lstsq`, sonsuz sayıda çözüm arasından
en küçük $\|\hat{a}\|$'yı döndürür. Bu çözüm rastgele bir tercih:
k=0 ile k=2 katkılarını güvenilir biçimde ayrıştıramaz. Profil
($\widehat{\Delta q} = F\hat{a}$) anlam taşıyabilir ama bireysel katsayılar
güvenilir değil.

Bu sorunu §13'teki çok-konfigürasyon yığma ile çözebilirsiniz: üçüncü
bir bağımsız kmod ekleyin, rank 3'e çıkın.

#### Sorun 2: SNR (Sinyal/Gürültü) — yığma çözmez

Daha köklü bir sorun var. Gerçek $\Delta q$ iki bileşenden oluşur:

$$
\Delta q = \underbrace{\Delta q_\text{smooth}}_{\text{10 μm, aranan}}
          + \underbrace{\Delta q_\text{random}}_{\text{100 μm RMS, bilinmez}}
$$

Bu iki bileşen $\Delta R$ üzerinden ölçüme birlikte karışır:

$$
\Delta\mathbf{y} = \Delta R\,\Delta q
= \underbrace{\Delta R\,\Delta q_\text{smooth}}_{\text{aranan sinyal}}
+ \underbrace{\Delta R\,\Delta q_\text{random}}_{\text{parazit: \,\sim\!10\times \text{ büyük}}}
$$

$|\Delta q_\text{random}|/|\Delta q_\text{smooth}| \approx 10$ olduğundan
parazit katkısı sinyalden yaklaşık 10 kat büyük. Fourier fit bu iki katkıyı
birbirinden ayırt edemez — fit "smooth bileşeni bul" komutu almaz, sadece
$\Delta\mathbf{y} = M\hat{a}$'yı minimize eder ve parazit katkısı sonuca
karışır.

**Bu BPM ölçüm gürültüsüyle karıştırılmamalı.** BPM gürültüsü
$\sigma_\eta \sim 1$ μm — elektronik titreşim. Aşırı belirlenmiş sistemde
(§13) bu gürültü bastırılabilir. Ama $\Delta R\,\Delta q_\text{random}$
terimi $\Delta\mathbf{y}$'nin içinde gerçek bir **sinyal** olarak görünür,
gürültü gibi değil. Daha fazla kmod konfigürasyonu eklemek bu parazit
katkısını azaltmaz — her yeni ölçüm de aynı $\Delta q_\text{random}$'ı
taşır.

Yapılandırılmış harmonik senaryo (k=2 = 10 μm, k=4,6,8 = 200–300 μm)
ile somut sayısal doğrulama yapıldı. `recon_k_list_dy = [2]` ile baz
yalnız {k=2}, gerçek ise {k=2,4,6,8}:

```
κ(ΔR·F) = 1.10   etkin rank(M) = 2   baz boyutu = 2
k=2:  13.50 μm @ φ = 1.12 rad   |   gerçek 10.00 μm @ φ = 0.00 rad
      %35 genlik hatası, faz tamamen yanlış
Profil: RMS hata = 76.6 μm   korelasyon = 0.030
```

**κ = 1.10 ≈ 1** (mükemmel koşullanma) ve **rank = baz boyutu = 2**
(tam belirlenmiş) koşullarında bile k=2 tahmini tamamen çöküyor.
Kondisyon sayısı ve rank doğruluğun gerekli ama yeterli koşullarıdır;
baz gerçeği kapsamıyorsa sızıntı kaçınılmaz.

### Tune-Fourier frekans uyumsuzluğu

Daha derin bir sorun var. Bir quad'ı modüle edince elde ettiğimiz
"ölçüm yönü" ($\Delta R$'nin sağ tekil vektörü $v_{j_1}$) şuna benzer:

$$
v_{j_1,j} \propto K_j\sqrt{\beta_j}\,\cos\!\bigl(|\phi_j - \phi_{j_1}| - \pi Q\bigr)
$$

Bu, **tune frekansında** ($Q \approx 2.68$) titreşen bir desen —
irrasyonel sayı. Ölçmek istediğimiz Fourier harmonikleri ise
$\cos(2\pi k j/48)$ ile **tam sayı** $k$'da titreşiyor.

İki farklı frekans dili konuşuyorlar. $M = \Delta R \cdot F$
matrisi, "tune-frekanslı ölçüm → Fourier katsayısı" dönüşümünü yapıyor.
Bu dönüşüm ne kadar kötü eşleşirlerse kondisyon sayısı o kadar büyür.

Analoji: 442 Hz'de resonans veren bir dedektörle 440 Hz'i ölçmeye
çalışmak. Sinyal var, dedektör kısmen duyuyor, ama doğal frekans tam
değil — ölçüm hassasiyeti düşük. Targeted Fourier'ın başarısı ise
dedektörün kendi frekansını tam 440 Hz'e ayarlamak gibi: eğer baz
gerçekten verideki harmoniklere hizalanırsa, dedektörün doğal frekansı
ile sinyal frekansı tesadüfen iyi örtüşüyor.

---

## 12. Rank Kısıtı: "Sayım Prensibi"

### Temel kural

$$
\boxed{\text{Belirlenebilir Fourier katsayısı sayısı} \leq
\text{etkin rank}(\Delta R)}
$$

Etkin rank ≈ bağımsız kmod ölçümü sayısı.

| Senaryo | Rank | Çözülebilir katsayı |
|---------|------|----------------------|
| Tek-quad kmod | ~1 | 1 (sadece DC veya sadece cos₂) |
| İki-quad kmod | ~2 | 2 (örn. DC + cos₂, ama sin₂ yok) |
| 3 bağımsız kmod | ~3 | 3 (DC + cos₂ + sin₂) |
| $N$ bağımsız kmod | ~N | N katsayı |

$k = 0$ ve $k = 2$ harmoniklerini tam belirlemek için:
$a_0$ (DC), $a_{2c}$ (cos₂), $a_{2s}$ (sin₂) → **3 bilinmeyen**,
dolayısıyla en az **3 bağımsız kmod ölçümü**.

### Neden iki quad yeterli değil?

Sezgide şu vardı: iki quad, faz uzayında farklı konumlarda, birbirini
tamamlıyor — halkanın tamamını kapsıyor. Bu doğru ama yetersiz.

Her kmod ölçümü tek bir bilgi yönü verir (rank ~1). İki kmod →
2 yön. Üç bilinmeyen için 2 yön yetmez; sistem yetersiz belirlenmiş
kalır.

Ayrıca, kmod ölçümünün bilgi yönü FODO harmonikleriyle değil,
tune frekansıyla hizalı (§11). Dolayısıyla iki quad ile "geniş bir
tarama" yapıyormuş gibi görünse de aslında yalnız iki dar bilgi
penceresinden bakılıyor.

---

## 13. Çok-Konfigürasyon Yığma: Rank Sorununa Çözüm (Ama Yalnız Buna)

> **Önemli uyarı:** Bu bölüm §11'deki iki sorundan yalnız **birine** —
> rank yetersizliğine — çözüm sunar. 100 μm RMS rastgele arka plan
> varlığındaki SNR sorunu farklı bir sorundur ve aşağıdaki yöntemle
> çözülmez. Her iki sorunu birlikte görmek için §11'deki "Sorun 1 ve
> Sorun 2" ayrımına bakın.

Rank sorununun çözümü kavramsal olarak basit: yeterince bağımsız kmod
ölçümü yap, denklem sayısını bilinmeyen sayısının üstüne çıkar.

### Nasıl?

Her seferinde farklı tek bir quad modüle et. Her ölçüm:

$$
\Delta\mathbf{y}_{(j)} = \Delta R_{(j)}\,\Delta q,
\qquad \Delta R_{(j)}: \text{rank} \approx 1
$$

$N$ ölçümü dikey olarak yığ:

$$
\underbrace{
\begin{pmatrix}
\Delta R_{(j_1)} \\ \Delta R_{(j_2)} \\ \vdots \\ \Delta R_{(j_N)}
\end{pmatrix}
}_{48N \times 48}
\cdot F\,\hat{a}
=
\begin{pmatrix}
\Delta\mathbf{y}_{(j_1)} \\ \Delta\mathbf{y}_{(j_2)} \\ \vdots
\end{pmatrix}
$$

$\Delta R_{(j_k)}$'lar bağımsızsa yığılmış matrisin rankı $\approx N$.
$N \geq n_\text{baz}$ olunca sistem belirlenmiş ya da aşırı belirlenmiş
→ güvenilir çözüm.

### Hangi quad'lar seçilmeli?

Bağımsızlık koşulu: her quad'ın "sondaj yönü" $v_{j_k}$, Fourier
baz uzayına farklı projeksiyonlar yapmalı.

Somut: $k=2$ harmonik, FODO hücreleri boyunca $\cos(2\pi \cdot 2 \cdot n/24)$
ile değişiyor. Üç quad'ın bulunduğu FODO hücrelerindeki bu değer:

| Quad ($j$) | FODO hücresi ($n$) | k=2 projeksiyonu |
|-----------|--------------------|-|
| $j = 1$ | 0 | $\cos(0) = +1.00$ |
| $j = 3$ | 1 | $\cos(\pi/6) = +0.87$ |
| $j = 9$ | 4 | $\cos(2\pi/3) = -0.50$ |

İkisi pozitif, biri negatif → cos₂ ve sin₂ ayrıştırılabilir →
lineer bağımsızlık sağlanmış.

### Pratik uygulamada mevcut durum

Kod altyapısı (`build_response_matrix.py --config N`,
`test_kmod_reconstruction.py --config N`, `reconstruction.py`)
bu yığma işlemini destekliyor. `params.json`'da:

```json
"kmod_configs": [
    {"j1": 3, "j2": -1},
    {"j1": 9, "j2": -1},
    {"j1": 1, "j2": -1}
]
```

Sırayla çalıştırılınca üç bağımsız kmod ölçümü elde ediliyor.
Bu 3-konfig, k=2,4,6,8 bazı (8 katsayı) için test edildi:

```
Tek konfig → rank = 2 / 8 → yetersiz belirlenmiş
  k=2:  ~2307 μm (gerçek 10 μm)

3-konfig yığma → rank = 4 / 8 → hâlâ yetersiz belirlenmiş
  k=6: %21 hata (300 μm sinyal)   k=2: %511 hata (10 μm sinyal)
  Profil: RMS = 378 μm   kor = −0.105
```

Rank 4'e çıktı, 8 katsayı için yetmiyor. K=2,4,6,8 için tam
çözüm ≥8 bağımsız konfig istiyor. Daha küçük baz (örn. k=0+k=2
= 3 katsayı) mevcut 3-konfig ile tam belirlenmiş sisteme ulaşır —
ama o zaman k=4,6,8 katkıları sızıntı yaratır (§11 Sorun 2 aynısı).

---

## 13b. En Küçük Kareler İteratif mi? Ve Gürültü Altında Zayıf Harmonik

### Yanlış bir sezgi: "lstsq iterasyonla hatayı azaltır"

Yaygın bir yanılgı: en küçük karelerin "uygula, hataya bak, sonraki
iterasyona geç" şeklinde döngülü çalıştığı. Hayır. `np.linalg.lstsq`
**tek adımda, kapalı formülle** çözer:

$$
M = \Delta R \cdot F, \qquad
\hat{a} = (M^T M)^{-1} M^T \Delta\mathbf{y}
$$

Geometrik anlamı: $\Delta\mathbf{y}$ vektörünü $M$'nin sütunlarının
gerdiği alt uzaya **dik izdüşümle** düşürür. "En küçük kare hatayı veren
nokta" o alt uzaydaki dik izdüşüm noktasıdır ve analitik formülü vardır.
İterasyon yok.

İteratif olanlar bunun üzerine kurulu sarmalayıcılardır:
- **Greedy**: her adımda yeni bir $k$ ekleyip lstsq'yi tekrar çağırır.
- **CLEAN** (aşağıda): dominant harmoniği bulup kesirli çıkarır, tekrarlar.
- **LASSO/ADMM**: gerçek anlamda iteratif konveks optimizasyon.

### Gürültü altındaki zayıf harmonik: ne işe yaramaz?

§11 Sorun 2'de gördük: büyük k=4,6,8 varken küçük k=2'yi ölçmek zor.
Birkaç bariz fikrin neden işe yaramadığını netleştirelim.

**Doğrudan FFT.** Elimizdeki ölçüm $\Delta\mathbf{y} = \Delta R\,\Delta q$,
yani $\Delta q$ değil. $\Delta R$ Fourier modlarını karıştırır; saf k=2
girişi bile tune frekansı çevresine yayılmış bir çıkış verir (§ Tune-Fourier).
$\Delta\mathbf{y}$'nin FFT'si $\Delta q$'nun FFT'si değildir.

**Periyodik katlama (folding).** Katlama farklı periyottaki rassal
gürültüyü bastırır. Ama k=4,6,8 hepsi k=2'nin tam kat harmonikleri;
k=2 periyodunda katlandığında tam sayıda dönem tamamlayıp **iptal olmaz,
güçlenir.** Kendi harmoniklerini süzemez.

**MUSIC/ESPRIT.** Bu yöntemler veri saf frekans bileşenlerinin toplamı
olduğunda güçlüdür. $\Delta\mathbf{y}$ bu form değil — $\Delta R$ matris
çarpımıdır, konvolüsyon değil. Baskın yapı $\Delta R$'nin tekil
vektörleridir, $\Delta q$'nun harmonikleri değil.

### CLEAN: dominant kaynağı soy, kalanı ölç

Radyo astronomisindeki CLEAN'in fiziği uygundur: en parlak kaynağı bul,
çıkar, tekrarla.

$$
\begin{aligned}
&\mathbf{r} \leftarrow \Delta\mathbf{y} \\
&\textbf{döngü:} \\
&\quad \text{her aday } k:\ \hat{a}_k = \text{lstsq}(\Delta R\cdot F_k,\ \mathbf{r}) \\
&\quad \text{en çok düşüreni seç } (k^\star) \\
&\quad \mathbf{r} \leftarrow \mathbf{r} - g\cdot \Delta R\,F_{k^\star}\hat{a}_{k^\star}
       \quad (g = \text{loop gain} < 1) \\
&\quad \hat{a}_\text{toplam}[k^\star] \mathrel{+}= g\cdot\hat{a}_{k^\star}
\end{aligned}
$$

**Greedy'den farkı** kesirli çıkarımdır ($g<1$). Greedy bir moda tam
taahhüt eder; CLEAN her turda yalnız bir kesrini çıkardığı için sonraki
turlarda geri dönüp düzeltebilir — mode-mixing'e daha sağlam.

**Dürüstlük notu — CLEAN rank eklemez.** Ölçümün taşımadığı bilgiyi
yaratamaz:

| Durum | CLEAN sonucu |
|-------|--------------|
| Tam rank $\Delta R$ | k=2'yi mükemmel ayıklar (9.98 / 10 μm — sentetik test) |
| Rank yetersiz (4 denklem, 8 bilinmeyen) | Joint lstsq gibi sınıra çarpar |

CLEAN'in faydası şu senaryoyla sınırlı: **rank büyük harmonikleri
ayırmaya yetiyor** ama joint fit bilgiyi minimum-norm ile saçıyorsa.
Büyükleri temiz soyup zayıfı artıkta bırakır. Rank büyükleri bile
ayıramıyorsa CLEAN da çaresizdir. Asıl darboğaz hep rank: §13'teki
çok-konfig yığma ile rank artırılmadan CLEAN tek başına yetmez.

Uygulama: `fourier_reconstruct.py` (sade kalite raporu; `clean_gain`,
`clean_candidates_dy` parametreleriyle).

---

## 14. Pratik Rehber ve Açık Sorular

### Hangi yöntem hangi durumda kullanılmalı?

| Durum | Önerilen yöntem | Neden |
|-------|----------------|-------|
| Hangi harmonikler var bilinmiyor | Çok-konfig + greedy (rank ≥ 3 ile) | Greedy rank-2'de yanlış seçim yapar |
| Harmonikler fizikten tahmin edilebilir | Targeted Fourier | Baz doğruysa en düşük κ |
| Operasyonel drift izleme | Drift modu $R^{-1}(\mathbf{y}(t)-\mathbf{y}_0)$ | Ofset iptal, κ(R)=160 yeterli |
| Mutlak hizalama | LOCO/BBA (harici) | k-mod tabanlı yöntemler ofset-gürültü dualitesiyle sınırlı |

### Targeted Fourier'ı uygulamadan önce sorulması gereken sorular

1. **Baz boyutu ≤ rank($\Delta R$) mı?** Değilse sistem yetersiz
   belirlenmiş; katsayılar ayrıştırılamaz.

2. **Verideki harmonikler gerçekten bilinyor mu?** Yalnız tahmin
   ediliyorsa: bazı biraz geniş tut (1-2 fazla harmonik tolere
   edilebilir ama κ büyür), ya da greedy ile önce tespite çalış.

3. **Random bileşen smooth sinyalden ne kadar büyük?** 100 μm
   rastgele + 10 μm smooth senaryosunda Fourier fit başarısız —
   bu yalnızca rank sorunu değil (rank 3 ile çözülür), aynı zamanda
   $\Delta R\,\Delta q_\text{random}$ paraziti ~10× büyük olduğundan
   SNR da yeterli değil. Yöntem bu senaryoda sınırına dayanmış durumda.

4. **Bağımsız kmod ölçümleri gerçekten bağımsız mı?** Farklı
   quad'lardan gelen $v_{j}$ vektörleri lineer bağımlıysa
   (örn. hepsi aynı FODO fazındaysa) yığma rankı yükselmez.

### Açık sorular

- **k=2,4,6,8 için yeterli konfig:** 3-konfig → rank=4/8, k=2 için
  %511 hata. Tam çözüm ≥8 konfig istiyor. Ancak sızıntı testi gösterdi
  ki rank=baz olsa bile SNR sorunu devam ediyor — rank yeterli değil.

- **Rastgele arka plan varlığında smooth bileşen tespiti:**
  100 μm random + 10 μm smooth senaryosunda Fourier fit başarısız —
  bu beklenen bir sonuçtur. $\Delta R\,\Delta q_\text{random}$ katkısı
  $\Delta R\,\Delta q_\text{smooth}$'tan ~10 kat büyük olduğundan fit
  paraziti sinyalden ayırt edemiyor. Bu kısıtı aşmak için ya random
  bileşen modellenmeli (istatistiksel önsel bilgi) ya da smooth sinyal
  genliği random ile karşılaştırılabilir düzeyde olmalı.

- **Adaptif baz seçimi:** Harmonikler bilinmiyorsa,
  veri-güdümlü olarak doğru baz nasıl belirlenir?
  Greedy rank ≥ 3 ile daha güvenilir ama BIC/F-test gibi
  istatistiksel model seçim kriteri henüz uygulanmadı.

- **Örgü modeli hatası:** Gerçek halkada $R$ birebir bilinmez.
  %1 beta-beat kaç μm ek hata yaratır? (Test 8, sürmekte.)

---

## Ek: Kod Akışı

```
1. integrator.cpp       → parçacık takibi (GL4 simplektik, C++)
   integrator.py        → Python sarmalayıcı (ctypes)

2. build_response_matrix.py
     R₁  ←  nominal gradyan, tüm quad'lar sırayla kaydırıldı
     R₂  ←  perturbe gradyan, aynı tarama
     ΔR = R₂ − R₁

3. test_kmod_reconstruction.py
     dy üret  ←  params.json'daki dy_harmonics + random gürültü
     Δy hesapla  ←  iki konfigürasyonda parçacık koşumu
     Direkt / TSVD / Fourier karşılaştır

4. reconstruction.py
     Hedefli fit  ←  params.json'daki harmoniklerle doğrudan
     Greedy       ←  veri-güdümlü harmonik tespiti
     LASSO        ←  L1 cezalı seyrek rekonstrüksiyon
     Çok-konfig   ←  yığılmış sistem (R_dy_1_c0.npy varsa)

5. fourier_reconstruct.py   (sade kalite raporu, LASSO/greedy yok)
     Hedefli fit  ←  genlik/faz/hata tablosu
     Sızıntı testi←  recon_k_list_dy verilirse baz ≠ truth
     CLEAN        ←  dominant harmonikleri kesirli soyma (loop gain)
```

Fourier seçimi üç satır:

```python
F, meta = fodo_fourier_basis(n_q=48, k_list=[0, 2], antisym=True)
M = dR @ F                             # 48 × 3
a, *_ = np.linalg.lstsq(M, delta_y)   # en küçük kareler
dy_geri = F @ a
```

Tüm analitik güç bu üç satırdan önce: bazın doğru seçilmesinde.
