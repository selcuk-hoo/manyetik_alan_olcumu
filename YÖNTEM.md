# Fourier Tabanlı Hizalama Hatası Geri Çatımı

> Bu bölüm hızlandırıcı fiziğinin temellerini (kapalı yörünge, beta fonksiyonu,
> faz ilerlemesi, tepki matrisi) bilen ama tepki matrisi tabanlı diagnostik
> yöntemlere yabancı bir okuyucu için ders kitabı düzeyinde yazılmıştır.
> Amaç hem "ne yapıyoruz, nasıl yapıyoruz" sorularına, hem de
> "neden işe yarıyor, neden gerekli" sorularına net cevap vermektir.

---

## İçindekiler

1. [Problemin doğuşu](#1-problemin-doğuşu)
2. [Tepki matrisi: tek-tane-değil-bütün dilini konuşmak](#2-tepki-matrisi-tek-tane-değil-bütün-dilini-konuşmak)
3. [BPM ofsetleri ve k-modülasyon hilesi](#3-bpm-ofsetleri-ve-k-modülasyon-hilesi)
4. [İki-quad k-mod neden tam çözmüyor?](#4-iki-quad-k-mod-neden-tam-çözmüyor)
5. [R₁ ve R₂'yi ayrı kullanmak yardım eder mi?](#5-r₁-ve-r₂yi-ayrı-kullanmak-yardım-eder-mi)
6. [Problemi yeniden parametrelendir: Fourier bazı](#6-problemi-yeniden-parametrelendir-fourier-bazı)
7. [Bias–variance gerilimi: doğru N kaç?](#7-biasvariance-gerilimi-doğru-n-kaç)
8. [Hedefli ölçüm: nano-metre hassasiyeti](#8-hedefli-ölçüm-nano-metre-hassasiyeti)
9. [Pratik mesajlar ve sınırlar](#9-pratik-mesajlar-ve-sınırlar)

---

## 1. Problemin doğuşu

Bir hızlandırıcı halkasında her kuadrupol mıknatıs ideal eksenine göre
küçük bir kaçıklık (`dy_j` dikeyde, `dx_j` yatayda) ile kuruludur.
Tasarımdan sapma `dy_j ≠ 0` olduğunda, mıknatıs orta ekseninden geçmesi
gereken parçacık demeti tam ortadan geçmez. Kuadrupol gradyenli alanı
parçacığı odaklayıcı/saçtırıcı olarak etkiler; demet ekseninden
sapmış parçacık ek bir dipol-benzeri kick alır:

$$
\Delta y'_j = -K_j \cdot dy_j
$$

burada $K_j = (1/B\rho) \cdot (\partial B/\partial x) \cdot L_q$ quad'ın integre kick gücü.
Her quad'daki bu küçük kick'ler birikir ve halkanın **kapalı yörüngesini**
ideal eksenden saptırır. BPM'ler (Beam Position Monitor) bu sapmayı
ölçer:

$$
y_i^{co} = \frac{\sqrt{\beta_i}}{2\sin\pi\nu}
\sum_j \sqrt{\beta_j} \cos(\pi\nu - |\phi_i - \phi_j|) \cdot \Delta y'_j
$$

Bu klasik Courant-Snyder formülüdür: BPM $i$'deki sapma, tüm $j$
quad'larının kick'lerinin beta-ağırlıklı ve faz-fark-modülasyonlu
toplamıdır.

### pEDM'de neden bu kadar önemli?

Proton EDM ölçümünde aranan sinyal son derece zayıftır. Demet
ekseninden 10 μm kalıcı bir sapma bile yanlış EDM sinyaline
benzeyebilir — sistematik hata. Bu yüzden her quad'ın hizalanması
**~10 μm** mertebesinde bilinmek istenir. Mekanik ölçüm bu hassasiyete
ulaşmaz; demete dayalı (beam-based) hizalama gerekir. Sorumuz şu:
**BPM ölçümlerinden 48 quad'ın `dy_j`'lerini tek tek geri çatabilir
miyiz?**

---

## 2. Tepki matrisi: tek-tane-değil-bütün dilini konuşmak

Yukarıdaki formül lineerdir: tüm $dy_j$'ler birlikte BPM okumalarına
katkı verir. Matris formuna alalım:

$$
y = R \cdot dy,
\qquad
R_{ij} = \frac{K_j \sqrt{\beta_i \beta_j}}{2\sin\pi\nu} \cos(\pi\nu - |\phi_i - \phi_j|)
$$

$R$, **tepki matrisi** (response matrix, ORM) olarak adlandırılır:
satır $i$ BPM'e, sütun $j$ quad'a karşılık gelir. $R_{ij}$, $j$
numaralı quad 1 m kaydırılırsa $i$ numaralı BPM'de okunacak sapmadır.

Soyut görünüyor ama operasyonel olarak basittir: simülasyonda her quad'ı
sırayla küçük bir miktar kaydırırız ($\delta = 10^{-4}$ m), kapalı
yörüngeyi hesaplayıp okuruz, fark alırız:

$$
R[:,j] = \frac{y_{co}(\,dy_j = \delta\,) - y_{co}(0)}{\delta}
$$

48 quad → 48 simülasyon → tüm $R$ matrisi (48×48).

**Eğer $R$ tersi alınabilirse:** $dy = R^{-1} y$. Problem çözüldü.
Sadece bir engel kaldı: BPM ofsetleri.

---

## 3. BPM ofsetleri ve k-modülasyon hilesi

Bir BPM'in okuduğu değer, gerçek pozisyona ek olarak **elektronik
ofset** $b_i$ içerir:

$$
y^{\text{ölçülen}} = R \cdot dy + b + n
$$

$b$ statiktir (zamanla değişmez) ama bilinmez ve büyüktür:
tipik olarak ~300 μm; oysa quad hizalama sapmasının yörüngeye katkısı
~10 μm. Ofset sinyali boğar; $R \cdot dy = y - b$ çözülemez.

### Hile: aynı $b$, farklı $R$

Quad gradyenlerini iki farklı ayarda kurup ($g_{\text{nom}}$ ve $g_{\text{pert}}$)
iki kez ölç:

$$
y_1 = R_1 \cdot dy + b + n_1
$$
$$
y_2 = R_2 \cdot dy + b + n_2
$$

İkisinin **farkı**:

$$
\Delta y = y_2 - y_1 = (R_2 - R_1) \cdot dy + (n_2 - n_1) = \Delta R \cdot dy + \Delta n
$$

$b$ otomatik iptal oldu. Şimdi denklem $\Delta R \cdot dy = \Delta y$.
Çözmek için $\Delta R$'nin iyi koşullanmış olması gerekir; bu noktada
**hangi modülasyon stratejisi** seçildiği belirleyici hale gelir.

### Üç strateji

- **Uniform**: tüm quad'lar birlikte ölçeklenir, $g \to g(1+\varepsilon)$.
  Halkanın tüm odaklaması değişir, beta-beat tüm modlarda zengindir,
  $\Delta R$ tam ranklı ve $\kappa(\Delta R) \sim 160$. Mükemmel
  çalışır. Ama gerçek bir hızlandırıcıda tüm güç kaynaklarını
  eşzamanlı kalibre etmek pratikte zordur.

- **Tek-quad**: yalnızca bir quad'ın gradyenini değiştir. $\Delta R$
  esasen rank-1, $\kappa \sim 10^8$. Tamamen başarısız.

- **İki-quad**: yalnızca iki quad'ın ($j_1$, $j_2$) gradyenini değiştir.
  $\Delta R$ rank-2 baskın bir matris, $\kappa \sim 10^6$. Hızlandırıcıda
  uygulanabilir ama matematiksel olarak zorlu. **Bu çalışmanın odak
  noktası.**

---

## 4. İki-quad k-mod neden tam çözmüyor?

İlk içgüdü şu olabilir: "iki quad'ın gradyenini değiştiriyorum; demet
tüm halkayı gezerken sadece o iki noktada farklı kick alacak, dolayısıyla
COD yalnızca o iki noktadan etkilenecek." Bu **yanlıştır**.
Gradyen değişikliği yalnızca o iki noktadaki kick'i değiştirmez;
**tüm lattice'in optiğini** (tune $\nu$, beta fonksiyonu $\beta(s)$,
faz ilerlemesi $\phi(s)$) değiştirir.

$\Delta R$'yi iki parçaya ayıralım:

$$
\Delta R[i,j] \approx
\underbrace{K_j \cdot \delta g_j \cdot C(i,j)}_{\text{DOĞRUDAN}}
+
\underbrace{K_j \cdot \Delta C(i,j)}_{\text{DOLAYLI}}
$$

burada $C(i,j) = \sqrt{\beta_i \beta_j} / (2\sin\pi\nu) \cdot \cos(\pi\nu - |\Delta\phi|)$.

- **Doğrudan terim**: yalnızca $j = j_1, j_2$ için sıfırdan farklıdır,
  çünkü sadece bu iki quad'ın $\delta g_j$'si var. Büyüklük ~ $\varepsilon$.
- **Dolaylı terim**: tüm sütunlarda sıfırdan farklıdır (beta-beat
  tüm halkaya yayılır). Büyüklük ~ $\varepsilon^2$ — beta-beat
  amplitüdü modülasyon büyüklüğü ile orantılıdır.

Bu iki terim arasındaki şiddet oranı yaklaşık **1 : ε**; tipik %5
modülasyon için 20:1. Sonuç: $\Delta R$'nin SVD spektrumunda iki
büyük tekil değer (doğrudan modlar, $j_1$ ve $j_2$ yönleri) ve 46
küçük tekil değer (dolaylı modlar) bulunur.

$$
\kappa(\Delta R) \sim \frac{1}{\varepsilon} \sim 10^6
$$

Matematiksel olarak $\Delta R$ tam ranklıdır — her quad'ın bilgisi
teorik olarak içindedir — ama **kullanılabilir** rank etkin olarak 2.
Direkt ters çevirme veya TSVD bu küçük modları gürültü tabanından
ayıramaz; rekonstrüksiyon başarısız olur.

---

## 5. R₁ ve R₂'yi ayrı kullanmak yardım eder mi?

Doğal soru şudur: $\Delta R$ yerine iki tepki matrisini sistemde ayrı
tutalım, $dy$ ve $b$'yi aynı anda çözelim.

$$
\begin{pmatrix} R_1 & I \\ R_2 & I \end{pmatrix}
\begin{pmatrix} dy \\ b \end{pmatrix}
=
\begin{pmatrix} y_1 \\ y_2 \end{pmatrix}
$$

İki bilinmeyen vektör (toplam 96 boyut), iki ölçüm vektörü (96 denklem).
Görünüşte daha fazla bilgi içeriyor.

Bloklar arasında satır indirgemesi uygulayalım — ikinci bloktan birinci
bloğu çıkaralım:

$$
\begin{pmatrix} R_1 & I \\ \Delta R & 0 \end{pmatrix}
\begin{pmatrix} dy \\ b \end{pmatrix}
=
\begin{pmatrix} y_1 \\ \Delta y \end{pmatrix}
$$

Sistem otomatik olarak iki bağımsız alt probleme ayrıştı:

1. **Alt blok:** $\Delta R \cdot dy = \Delta y$ → $dy$'yi belirler.
2. **Üst blok:** $b = y_1 - R_1 \cdot dy$ → $dy$ bulunduktan sonra $b$'yi verir.

Yani $dy$'yi bulmak için hâlâ $\Delta R$'nin iyi koşullanması gerekiyor.
**$R_1$ ve $R_2$'yi ayrı kullanmak, cebirsel olarak $\Delta R$
kullanmakla tamamen denktir.** Bilgi kazanmaz; sadece aynı bilgiyi farklı
formda yeniden yazar.

Kazanmak için **harici bilgi** eklenmesi gerekir. İki yol vardır:

- **(a) $b$ hakkında ön bilgi:** $b$'nin yapısı (örneğin düşük-frekanslı
  ofset deseni) modellenir, parametre sayısı azaltılır.
- **(b) $dy$ hakkında ön bilgi:** $dy$'nin doğru biçimde parametrelendirilmesi
  — örneğin düşük-mertebeli Fourier bileşenleri.

Bu çalışmada **(b)** yolunu izliyoruz; fizik bize $dy$'nin yapısı
hakkında doğal bir öngörü sağlıyor.

---

## 6. Problemi yeniden parametrelendir: Fourier bazı

Kapalı yörünge formülünde gizli bir filtre vardır. Kick dağılımı
$\theta(s)$'nin faz-ilerlemesi cinsinden Fourier ayrışımı:

$$
\theta(s) = \sum_n \theta_n \, e^{i n \phi(s) / \nu}
$$

Bu harmoniklerin kapalı yörüngeye katkısı:

$$
y_n^{co} \propto \frac{\theta_n}{n^2 - \nu^2}
$$

Paydanın **rezonans** yapısı belirleyicidir: $n \approx \nu$ olan harmonik
en güçlü yükseltilir; diğer harmonikler $1/(n^2 - \nu^2)$ ile
bastırılır. Dolayısıyla BPM ölçümünün taşıdığı bilgi, büyük ölçüde
**tune yakınındaki birkaç harmonik**te yoğunlaşır. Yüksek harmoniklerin
sinyali zaten küçüktür; gürültü altında kalır.

**Sonuç:** 48 boyutlu $dy$'nin her 48 bileşenini eşit hassasiyetle
geri çatmak fiziksel olarak imkânsızdır — yeterli bilgi yoktur.
Buna karşın **düşük-mertebeli Fourier bileşenleri** doğal olarak
iyi ölçülür.

### Fourier baz matrisi

Quad indeksini $j = 0, 1, \ldots, 47$ alıp her $dy$ vektörünü
düşük harmonik bileşenlerle temsil edelim:

$$
dy_j = a_0 + \sum_{k=1}^{N} \left[ a_k \cos\!\left(\frac{2\pi k j}{48}\right) + b_k \sin\!\left(\frac{2\pi k j}{48}\right) \right]
$$

Matris formunda $dy = F \cdot a$ olarak yazılır; burada $F$ boyutu
$48 \times (2N+1)$ olan **Fourier baz matrisi**, $a$ ise
$(2N+1)$-boyutlu katsayı vektörüdür.

K-mod denklemine yerleştirelim:

$$
\Delta y = \Delta R \cdot F \cdot a \equiv M \cdot a
$$

$M = \Delta R \cdot F$ matrisi $48 \times (2N+1)$ boyutludur.
$N = 4$ için 9 bilinmeyen, 48 denklem — **aşırı belirlenmiş** sistem.
En küçük kareler ile çözülür:

$$
\hat{a} = (M^T M)^{-1} M^T \Delta y
$$

ve geri çatım $\hat{dy} = F \cdot \hat{a}$ olarak elde edilir.

### Neden iyi koşullanır?

$\Delta R$'nin "güçlü" tekil vektörleri (büyük $\sigma$'lı yönler)
zaten düşük-mertebeli faz harmoniklerinin yönündedir — tune-rezonans
etkisinin doğal bir sonucu. $F$'yi bu güçlü altuzaya hizalı seçtiğimizde
$M = \Delta R \cdot F$ yalnızca "kuvvetli" modları taşır:

$$
\kappa(M) \ll \kappa(\Delta R)
$$

---

## 7. Bias–variance gerilimi: doğru N kaç?

$N$ büyüdükçe:

- **Bias (sapma) azalır:** daha çok harmonik baz eklenir; gerçek
  $dy$'yi daha iyi temsil etmek mümkün olur.
- **Variance (gürültü büyütmesi) artar:** daha çok serbestlik derecesi
  fit'in gürültüyü de modellemeye başlamasına yol açar; katsayılar şişer.

Bu klasik bir ikilemdir. Doğru $N$ veriye bağlıdır.

### Sayısal örnek

Aşağıda **sinüzoidal test verisi** kullanılmıştır: $dy$ yalnızca
$k = 2$ ve $k = 4$ harmoniklerini içerecek şekilde üretilmiştir.
$N$ taraması sonuçları:

| N | Baz boyutu | $\kappa(\Delta R \cdot F)$ | Model RMS | Ölçüm RMS | Korelasyon |
|---|-----------|---------------------------|-----------|-----------|------------|
| 1 | 3 | 8.5 | 79 μm | **1065 μm** | 0.00 |
| 2 | 5 | 7.9 × 10³ | 35 μm | 58 μm | 0.75 |
| 3 | 7 | 1.2 × 10⁴ | 35 μm | 51 μm | 0.78 |
| 4 | 9 | 1.5 × 10⁴ | 0 μm | 37 μm | 0.88 |
| 5 | 11 | 1.8 × 10⁴ | 0 μm | 38 μm | 0.88 |

Tablodaki iki ölçüt şunlardır:

- **Model RMS:** $\hat{dy}_{\text{model}} = F \cdot F^+ \cdot dy_{\text{gerçek}}$
  ile hesaplanan temsil hatası — "N harmonik, gerçek $dy$'yi ne kadar
  iyi yaklaşıklar?" sorusuna cevap verir. Ölçümden bağımsızdır.
- **Ölçüm RMS:** gerçek rekonstrüksiyon hatası; hem gürültü hem de temsil
  etkisini içerir.

Tablo okunuşu:

- **$N = 1$:** Bazda yalnız DC ve $k = 1$ var; oysa veride $k = 2$ ve $k = 4$
  bulunuyor. LSQ bu uyumsuzluğu absürd büyük katsayılarla gidermeye
  çalışır → 1065 μm hata.
- **$N = 2$:** Bazda $k = 2$ var ama $k = 4$ yok. Veriden $k = 4$'ün
  "sızıntısı" $k = 2$ katsayısını saptırıyor. 58 μm hata.
- **$N = 4$:** Tam doğru baz. Model RMS = 0; ölçüm RMS = 37 μm, yalnızca
  gürültü büyütmesinden kaynaklanıyor.
- **$N = 5$:** Gereksiz bir harmonik eklendi; zarar az ama küçük bir
  kötüleşme gözlemleniyor.

**Genel kural:** $N$, sinyalde bulunan en yüksek harmoniğe eşit ya da
hafif büyük olmalıdır. Çok daha büyük → variance baskın hale gelir.

### Neden N = 4, N = 2'den daha iyi sonuç veriyor?

Bu klasik bir **yanlış-atfetme** (misattribution) örneğidir.

$N = 2$'de bazda $k = 4$ yoktur; dolayısıyla LSQ, $k = 4$'ün $\Delta y$'ye
katkısını en yakın baz fonksiyonuna — $k = 2$'ye — dağıtmak
zorunda kalır. Sonuç: $k = 2$ katsayısı da kirlenir, yani gerçekte
temsil edebildiği bileşeni bile yanlış kestirmiş olur.

$N = 4$'te ise her harmonik kendi katsayısına gider — **temiz atfetme**.
Bunun bedeli 4 ekstra sütun eklemekten kaynaklanan biraz daha büyük
$\kappa$'dır. Net etki: doğru atfetme kazancı, kötü koşullanma
kaybından fazladır.

Daha derin mesaj: **bazınızda gerçek harmoniğin biri eksikse, var olan
harmoniklerin tahmini de yanlış olur.**

---

## 8. Hedefli ölçüm: nano-metre hassasiyeti

Önceki tablolar, k = 1..4 harmoniklerini içeren geniş baz ile çalışmanın
$\kappa \approx 10^4$ getirdiğini ve gürültüyü ~30 kat büyüttüğünü
gösterdi.

Bir adım daha gidelim: **bazı yalnızca veride gerçekten var olan
harmoniklerle sınırla.** Veride sadece $k = 2$ ve $k = 4$ olduğunu
biliyorsak (önceki ölçümlerden ya da fiziksel öngörüden), baz olarak
tam olarak bunları kullanalım:

$$
F_{\{2,4\}} = \bigl[\cos(2\pi \cdot 2 \cdot j/48),
\sin(2\pi \cdot 2 \cdot j/48),
\cos(2\pi \cdot 4 \cdot j/48),
\sin(2\pi \cdot 4 \cdot j/48)\bigr]
$$

4 sütun, 4 katsayı.

### Sonuçlar

| k listesi | Sütun | $\kappa(\Delta R \cdot F)$ | Model RMS | Ölçüm RMS | Korelasyon |
|-----------|-------|---------------------------|-----------|-----------|------------|
| {2} | 2 | 1.1 | 35 μm | 37 μm | 0.89 |
| {4} | 2 | 14 | 71 μm | **1466 μm** | −0.38 |
| **{2, 4}** | **4** | **186** | **0 μm** | **0.02 μm** | **1.000** |
| {1, 2, 3, 4} | 8 | 1.3 × 10⁴ | 0 μm | 35 μm | 0.90 |

Bulgu çarpıcıdır: **{2, 4} bazı 0.02 μm hata veriyor — gürültü tabanı.**
Geniş baz ({1, 2, 3, 4}) ile karşılaştırıldığında **2000 kat daha iyi.**

### Neden bu kadar büyük fark?

Geniş baz, $k = 1$ ve $k = 3$ için boş yere yer ayırıyor. Bu sütunlar
veride sıfır katkı taşımasına rağmen matrisin koşul sayısını bozuyor:
$\kappa$ 186'dan 13000'e çıkıyor. Gürültü büyütmesi yaklaşık $\kappa$
ile orantılıdır; bu nedenle sonuç 2000 kat daha kötüye gidiyor.

### Tek harmonik yalıtmak ne zaman güvenlidir?

- **{2}** bazı çalışıyor (37 μm, kor. 0.89), çünkü $k = 2$ harmonik
  veride $k = 4$'ten iki kat büyüktür. $k = 4$'ün sızıntısı küçük
  bir pertürbasyon olarak kalır — kabul edilebilir.
- **{4}** bazı patlar (1466 μm, kor. negatif): $k = 2$ katkısı baskın
  olduğundan LSQ bu büyük katkıyı yalnızca $k = 4$ katsayısıyla
  açıklamaya zorlanır; katsayı absürd büyür, sonuç anlamsız çıkar.

**Kural:** Tek bir harmoniği yalıtarak ölçmek istiyorsanız, ya o
harmonik veride baskın olmalıdır; ya da diğer büyük harmonikler de
baza eklenmelidir.

---

## 9. Pratik mesajlar ve sınırlar

### Yöntemin gücü nereden geliyor?

1. **Fiziksel uyum:** Düşük-mertebe Fourier, kapalı yörüngenin doğal
   "filtre yapısına" uyar. Yüksek harmonikler zaten ölçülemez; onları
   modelden çıkarmak veri kaybı değil, gürültü kazancıdır.
2. **Az parametre, çok ölçüm:** 48 BPM ile 4 parametre fit edildiğinde
   istatistiksel olarak büyük SNR avantajı elde edilir.
3. **Koşul sayısının etkisi:** Baz doğru seçilirse $\kappa$ küçük kalır
   ve gürültü büyütülmez.

### Sınırlar: gerçek senaryoya transfer

Yukarıdaki 0.02 μm sonucu **sinüzoidal test verisi** ile elde edilmiştir:
$dy$'nin yalnızca $k = 2$ ve $k = 4$ içerdiği önceden biliniyordu.
Gerçek halkada $dy$'nin Fourier içeriği tam olarak bilinmez. İki yol
vardır:

- **Fizikten tahmin:** Toprak hareketi, termal genleşme, tünel oturması
  gibi kaynaklara bağlı hizalama hataları **düşük frekanslıdır**.
  $k = 1, 2, 3$ tipik baskın modlardır. Bu senaryoda hedefli Fourier
  yaklaşımı doğal olarak uygulanabilir.
- **Veri-yönlü tahmin:** Önce geniş bir bazla ölç, rezidüel spektrumundan
  hangi harmoniklerin önemli olduğunu gör, sonra dar bir baza geç.
  Bu **adaptif baz seçimi** stratejisidir.

Eğer $dy$ tamamen rastgele ve düz-spektrumluysa — yani her quad
bağımsız ve rastgele kaydırılmışsa — düşük-N Fourier yaklaşımı
**model hatası** ile sınırlanır. Bu durumda 48 quad'ın tek tek
hizalanmasını ölçmek için **çoklu k-mod ölçümü** (her seferinde
farklı bir quad çifti) zorunludur.

### Karşılaştırma: tüm yöntemler

| Yöntem | Hata RMS | Korelasyon | Yorum |
|--------|----------|------------|-------|
| Direkt çözüm ($\Delta R^{-1}$) | 107 μm | 0.03 | $\Delta R$ tekil, başarısız |
| TSVD | 78 μm | 0.16 | Küçük modlar kesilir, biraz iyileşir |
| Fourier $N = 4$ (geniş) | 37 μm | 0.88 | Fiziksel parametrelendirme |
| **Fourier {2, 4} (sıkı)** | **0.02 μm** | **1.000** | **Doğru baz, gürültü tabanı** |

### Anahtar tasarım kararları

- **Bazda olmayan harmonik veride varsa:** sızıntı ve büyük hata.
- **Bazda olan ama veride sıfır harmonik:** zarar vermez, ama $\kappa$ büyür.
- **Optimal denge:** $F$, veride var olan harmoniklere **sıkıca** hizalanmalı;
  ne eksik ne fazla.

Bu analiz, klasik tepki-matrisi tabanlı yöntemin iki-quad k-mod
rejiminde neden başarısız göründüğünü ve neden gerçekte bilgi
taşıdığını ortaya koyuyor: bilgi vardı, ama parametrelendirme onu
çıkarmak için yanlış bir uzay üzerinde çalışıyordu. Doğru baz seçimi,
doğru sonucu getiriyor.

---

## Ek: Kod akışı

```
1. build_response_matrix.py
     R₁, R₂  ←  iki konfigürasyonda her quad'ı sırayla kaydır + simüle et
     ΔR = R₂ − R₁

2. test_kmod_reconstruction.py
     (a) Gerçek dy/dx üret (rastgele veya sinüzoidal --smooth)
     (b) İki konfigürasyonda parçacık takibi; y₁ ve y₂'yi al
     (c) Δy = y₂ − y₁
     (d) Çöz:
         Direkt:   dy = ΔR⁻¹ · Δy
         TSVD:     küçük tekil değerleri kes
         Fourier:  dy = F · (ΔR·F)⁺ · Δy
```

Fourier seçimi kod içinde üç satırdır:

```python
F = fourier_basis(n_q=48, k_list=[2, 4])   # sıkı baz
M = dR @ F
a, *_ = np.linalg.lstsq(M, delta_y, rcond=None)
dy_geri = F @ a
```

Tüm güç, bazın bu üç satırdan önce doğru seçilmesindedir.
