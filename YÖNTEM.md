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
gereken parçacık demeti tam ortadan geçmez. Quadrupol gradyenli alanı
parçacığı **odaklayıcı/saçtırıcı** olarak etkiler; demet ekseninden
sapmış parçacık ek bir dipol-benzeri kick alır:

$$
\Delta y'_j \;=\; -K_j \cdot dy_j
$$

burada $K_j = (1/B\rho)\,\partial B/\partial x \cdot L_q$ quad'ın integre kick gücü.
Her quad'da bu küçük kick'ler birikir ve halkanın **kapalı yörüngesini**
ideal eksenden saptırır. BPM'ler (Beam Position Monitor) bu sapmayı
ölçer:

$$
y_i^{co} \;=\; \frac{\sqrt{\beta_i}}{2\sin\pi\nu}\,
\sum_j \sqrt{\beta_j}\;\cos(\pi\nu - |\phi_i - \phi_j|)\;\Delta y'_j
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
**BPM ölçümlerinden 48 quad'ın `dy_j`'leri tek tek geri çatılabilir
mi?**

---

## 2. Tepki matrisi: tek-tane-değil-bütün dilini konuşmak

Yukarıdaki formül linner: tüm $dy_j$'ler birlikte BPM okumalarına
katkı verir. Matris formuna alalım:

$$
y \;=\; R \cdot dy, \qquad R_{ij} \;=\; \frac{K_j\sqrt{\beta_i\beta_j}}{2\sin\pi\nu}\,\cos(\pi\nu - |\phi_i - \phi_j|)
$$

$R$ "tepki matrisi" (response matrix, ORM): satır $i$ = BPM, sütun $j$
= quad. $R_{ij}$, $j$ quad'ı 1 m kaydırılırsa $i$ BPM'inde okunacak
sapmadır.

Soyut görünüyor ama operasyonel olarak basittir: simülasyonda her quad'ı
sırayla küçük bir miktar kaydırırız ($\delta = 10^{-4}$ m), kapalı
yörüngeyi koşturup okuruz, fark alırız:

$$
R[:,j] \;=\; \frac{y_{co}(d y_j = \delta) - y_{co}(0)}{\delta}
$$

48 quad → 48 simülasyon → tüm $R$ matrisi (48×48).

**Eğer $R$ tersi alınabilir ise:** $dy = R^{-1} y$. Probleme bittiği
gibi. Sadece bir engel kaldı: BPM ofsetleri.

---

## 3. BPM ofsetleri ve k-modülasyon hilesi

Bir BPM'in okuduğu değer, gerçek pozisyona ek olarak **elektronik
ofset** $b_i$ içerir:

$$
y^{ölçülen} \;=\; R \cdot dy \;+\; b \;+\; n
$$

$b$ statiktir (zamanla değişmez) ama bilinmez ve **büyüktür**:
tipik olarak $\sim 300\,\mu m$, oysa quad hizalama sapmasının
yörüngeye katkısı $\sim 10\,\mu m$. Ofset sinyali boğar; $R \cdot dy = y - b$
çözülemez.

### Hile: aynı $b$, farklı $R$

Quad gradyenlerini iki farklı ayarda kurup ($g_{nom}$ ve $g_{pert}$)
iki kez ölç:

$$
y_1 = R_1 \cdot dy + b + n_1, \qquad y_2 = R_2 \cdot dy + b + n_2
$$

İkisinin **farkı**:

$$
\Delta y \;=\; y_2 - y_1 \;=\; (R_2 - R_1)\cdot dy + (n_2 - n_1) \;=\; \Delta R \cdot dy + \Delta n
$$

$b$ otomatik iptal oldu! Şimdi denklem $\Delta R \cdot dy = \Delta y$.
Çözmek için $\Delta R$'nin iyi koşullanmış olması gerekir. Bu noktada
**hangi modülasyon stratejisi** seçildiği belirleyici hale gelir.

### Üç strateji

- **Uniform**: tüm quad'lar birlikte ölçeklenir, $g \to g(1+\varepsilon)$.
  Halkanın tüm odaklaması değişir, beta-beat tüm modlarda zengindir,
  $\Delta R$ tam ranklı ve $\kappa(\Delta R) \sim 160$. Mükemmel
  çalışır. **Ama gerçek bir hızlandırıcıda tüm güç kaynaklarını
  eşzamanlı kalibre etmek pratikte zor.**

- **Tek-quad**: yalnızca bir quad'ın gradyenini değiştir. $\Delta R$
  esasen 1 rank, $\kappa \sim 10^8$. Tamamen başarısız.

- **İki-quad**: yalnızca iki quad'ın ($j_1$, $j_2$) gradyenini değiştir.
  $\Delta R$ rank-2 baskın bir matris. $\kappa \sim 10^6$. Hızlandırıcıda
  uygulanabilir ama matematiksel olarak zorlu. **Bu çalışmanın odak
  noktası.**

---

## 4. İki-quad k-mod neden tam çözmüyor?

İlk içgüdü "iki quad'ın gradyenini değiştiriyorum, demet tüm halkayı
gezerken sadece o iki noktada farklı kick alacak, dolayısıyla COD
yalnızca o iki noktadan etkilenecek" şeklindedir. Bu **yanlış**.
Gradyen değişikliği yalnızca o iki noktadaki kick'i değiştirmez,
**tüm lattice'in optiğini** (tune $\nu$, beta fonksiyonu $\beta(s)$,
faz ilerlemesi $\phi(s)$) değiştirir.

$\Delta R$'yi iki parçaya ayır:

$$
\Delta R[i,j] \;\approx\; \underbrace{K_j\,\delta g_j\,C(i,j)}_{\text{DOĞRUDAN}} \;+\; \underbrace{K_j\,\Delta C(i,j)}_{\text{DOLAYLI}}
$$

burada $C(i,j) = \sqrt{\beta_i\beta_j}/(2\sin\pi\nu)\cdot\cos(\pi\nu - |\Delta\phi|)$.

- **Doğrudan terim**: yalnızca $j = j_1, j_2$ için sıfırdan farklı
  (yalnız o quad'ların $\delta g_j$'si var). Büyüklük $\sim \varepsilon$.
- **Dolaylı terim**: tüm sütunlarda sıfırdan farklı (beta-beat
  herkese yayılır). Büyüklük $\sim \varepsilon^2$ — beta-beat
  amplitüdü modülasyon büyüklüğü ile orantılı.

Bu iki terim arasındaki şiddet oranı yaklaşık **1:ε**; tipik %5
modülasyon için 20:1. Sonuç: $\Delta R$'nin SVD spektrumunda iki
büyük tekil değer (doğrudan modlar, $j_1$ ve $j_2$ yönleri) ve 46
küçük tekil değer (dolaylı modlar) vardır.

$$
\kappa(\Delta R) \;\sim\; \frac{1}{\varepsilon} \;\sim\; 20{-}10^6
$$

(tam değer beta-beat detaylarına bağlıdır). Matematiksel olarak
$\Delta R$ tam ranklı (her quad'ın bilgisi orada), ama **kullanılabilir**
rank etkin olarak 2. Direkt ters çevirme veya TSVD bu küçük modları
gürültü tabanından ayıramaz, rekonstrüksiyon başarısız olur.

---

## 5. R₁ ve R₂'yi ayrı kullanmak yardım eder mi?

Doğal soru: $\Delta R$ yerine iki tepki matrisini sistemde ayrı tut.

$$
\begin{pmatrix} R_1 & I \\ R_2 & I \end{pmatrix}
\begin{pmatrix} dy \\ b \end{pmatrix}
=
\begin{pmatrix} y_1 \\ y_2 \end{pmatrix}
$$

İki bilinmeyen vektör (toplam 96 boyut), iki ölçüm vektörü (96 denklem).
Görünüşte daha bilgi taşıyor.

Bloklar arasında satır indirgemesi uygulayalım — ikinci satırdan
birinciyi çıkar:

$$
\begin{pmatrix} R_1 & I \\ \Delta R & 0 \end{pmatrix}
\begin{pmatrix} dy \\ b \end{pmatrix}
=
\begin{pmatrix} y_1 \\ \Delta y \end{pmatrix}
$$

Sistem otomatik ayrıştı:

1. **Alt blok**: $\Delta R \cdot dy = \Delta y$ → $dy$'yi belirler.
2. **Üst blok**: $b = y_1 - R_1\cdot dy$ → $dy$ bulunduktan sonra $b$.

Yani $dy$'yi bulmak için hâlâ $\Delta R$'nin iyi koşullanması
gerekiyor. **R₁ ve R₂'yi ayrı kullanmak cebirsel olarak $\Delta R$
kullanmakla denktir.** Bilgi kazanmaz; sadece aynı bilgiyi farklı
formda yazar.

Kazanmak istiyorsak **harici bilgi** eklemek zorundayız. İki yol var:

- **(a) $b$ hakkında ön bilgi**: $b$'nin yapısı (örn. düşük-frekanslı
  ofset deseni) modellenir, parametre sayısı azaltılır.
- **(b) $dy$ hakkında ön bilgi**: $dy$'nin "akıllıca" parametrelendirilmesi —
  örneğin düşük-mertebeli Fourier bileşenleri.

Bu çalışmada (b) yolunu izliyoruz, çünkü fizik bize $dy$'nin yapısı
hakkında doğal bir öngörü sağlıyor.

---

## 6. Problemi yeniden parametrelendir: Fourier bazı

Kapalı yörünge formülünde bir gizli filtre vardır. Kick dağılımı
$\theta(s)$ ve onun Fourier ayrışımı:

$$
\theta(s) \;=\; \sum_n \theta_n e^{i n \phi(s)/\nu}
$$

COD'nin aynı harmoniği:

$$
y_n^{co} \;\propto\; \frac{\theta_n}{n^2 - \nu^2}
$$

Bu paydanın **rezonans** yapısı vardır: $n \approx \nu$ olan harmonik
en güçlü yükseltilir, diğer harmonikler $1/(n^2 - \nu^2)$ ile bastırılır.
BPM ölçümünün taşıdığı bilgi, çoğunlukla **tune yakınındaki birkaç
harmonik**te yoğunlaşmıştır. Yüksek harmoniklerin sinyali zaten doğal
olarak küçüktür; gürültü altında kalır.

**Sonuç:** 48 boyutlu $dy$'nin 48 ayrı bileşenini eşit hassasiyetle
geri çatmak fiziksel olarak imkânsız — bilgi yok. Buna karşın
**düşük-mertebeli Fourier bileşenleri** doğal olarak iyi ölçülür.

### Fourier baz matrisi

Quad indeksini $j = 0, 1, ..., 47$ alıp her $dy$ vektörünü düşük
harmonik bileşenleri ile temsil et:

$$
dy_j \;=\; a_0 + \sum_{k=1}^{N}\bigl[a_k\cos(2\pi k j/48) + b_k\sin(2\pi k j/48)\bigr]
$$

Matris formunda $dy = F\cdot a$, $F$: 48×(2N+1) baz matrisi, $a$:
(2N+1)-boyutlu katsayılar.

K-mod denkleminde yerleştir:

$$
\Delta y \;=\; \Delta R \cdot F \cdot a \;\equiv\; M \cdot a
$$

$M = \Delta R\cdot F$: 48×(2N+1). $N=4$ için 9 bilinmeyen, 48 denklem
— **aşırı belirlenmiş**. En küçük kareler ile çöz:

$$
\hat a \;=\; (M^T M)^{-1} M^T \Delta y
$$

ve geri çatım $\hat{dy} = F\cdot\hat a$.

### Neden iyi koşullanır?

$\Delta R$'nin "güçlü" tekil vektörleri (büyük $\sigma$'lı) zaten
düşük-mertebeli faz harmoniklerinin yönündedir (tune-rezonans gereği).
$F$'yi bu güçlü altuzaya hizalı seçtiğimiz için $M = \Delta R\cdot F$
yalnızca "kuvvetli" modları taşır. **$\kappa(M) \ll \kappa(\Delta R)$**.

---

## 7. Bias–variance gerilimi: doğru N kaç?

$N$ büyüdükçe:

- **Bias (sapma) azalır**: daha çok harmonik baz → gerçek $dy$'yi
  daha iyi temsil edebilir.
- **Variance (gürültü büyütmesi) artar**: daha çok serbestlik → fit
  gürültüyü de modellemeye başlar, katsayıları şişer.

Bu klasik bir trade-off'tur. Doğru $N$ veriye bağlıdır.

### Sayısal örnek

Aşağıda **sinüzoidal test verisi** (smooth dy, $k=2$ ve $k=4$
harmonikleri içeren) kullanıldı. $N$ taraması sonuçları:

| N | baz | $\kappa(\Delta R\cdot F)$ | model RMS | ölçüm RMS | kor |
|---|-----|---|-----------|-----------|-----|
| 1 | 3 | 8.5 | 79 μm | **1065 μm** | 0.00 |
| 2 | 5 | 7.9×10³ | 35 μm | 58 μm | 0.75 |
| 3 | 7 | 1.2×10⁴ | 35 μm | 51 μm | 0.78 |
| 4 | 9 | 1.5×10⁴ | 0 μm | 37 μm | 0.88 |
| 5 | 11 | 1.8×10⁴ | 0 μm | 38 μm | 0.88 |

- **"model RMS"**: $\hat{dy}_{model} = F\cdot F^+\cdot dy_{gerçek}$ kullanarak
  veriden bağımsız hesaplanmış temsil hatası — "$N$ harmonik yeterli mi?"
- **"ölçüm RMS"**: gerçek rekonstrüksiyon hatası, gürültü + temsil etkisi.

Okuma:

- $N=1$: bazda yalnız DC ve $k=1$ var, ama veride $k=2,4$. LSQ bu
  uyumsuzluğu absurd büyük katsayılarla fitlemeye çalışıyor → 1065 μm hata.
- $N=2$: bazda $k=2$ var ama $k=4$ yok. Veriden $k=4$ "sızıntı"sı
  $k=2$ katsayısını saptırıyor. 58 μm hata.
- $N=4$: tam doğru baz. Model RMS = 0, ölçüm RMS = 37 μm yalnız
  gürültü artı kötü koşullanma sonucu.
- $N=5$: ekstra bir harmonik gereksiz ama zarar da çok az — modern
  bir bias-variance kompromisinde "az fazlalık tolere edilir".

**Genel kural:** $N$, sinyalde bulunan en yüksek harmoniğe eşit ya
da hafif büyük olmalı. Çok daha fazla → variance domine.

### Neden N=4 N=2'den iyi?

Bu klasik bir yanlış-tahsis (misattribution) örneğidir. $N=2$'de
bazda $k=4$ olmadığı için LSQ $k=4$'ün $\Delta y$'ye katkısını
kendi bazına dağıtmak zorunda — en yakını $k=2$. Sonuç: $k=2$
katsayısı da kirleniyor. $N=4$'de her harmonik kendi katsayısına
gidiyor, **temiz atfetme**. Pahası: 4 ekstra sütun, biraz daha
büyük $\kappa$. Net etki: doğru atfetme kazancı > kötü koşullanma kaybı.

Daha derin mesaj: **bazınızda gerçek harmoniğin biri eksikse, var
olan harmoniklerin tahmini de yanlış olur.**

---

## 8. Hedefli ölçüm: nano-metre hassasiyeti

Yukarıdaki tablolar şunu söylüyor: 4 harmonik (DC dahil 9 katsayı)
ile çalışmak $\kappa \approx 10^4$ getiriyor, gürültü 30× büyütülüyor,
sonuç 37 μm.

Bir adım daha gidelim: **bazı yalnızca veride gerçekten var olan
harmoniklerle sınırla**. Veride sadece $k=2$ ve $k=4$ olduğunu
biliyorsak (örneğin fizikten ya da önceki ölçümlerden), baz olarak
yalnız bunları kullan:

$$
F_{\{2,4\}}: \quad [\cos(2\pi\cdot 2 j/48),\; \sin(2\pi\cdot 2 j/48),\; \cos(2\pi\cdot 4 j/48),\; \sin(2\pi\cdot 4 j/48)]
$$

4 sütun, 4 katsayı.

### Sonuçlar

| k listesi | sütun | $\kappa(\Delta R\cdot F)$ | model RMS | ölçüm RMS | kor |
|-----------|-------|---|-----------|-----------|-----|
| {2} | 2 | 1.1 | 35 μm | 37 μm | 0.89 |
| {4} | 2 | 14 | 71 μm | **1466 μm** | −0.38 |
| **{2, 4}** | **4** | **186** | **0 μm** | **0.02 μm** | **1.000** |
| {1, 2, 3, 4} | 8 | 1.3×10⁴ | 0 μm | 35 μm | 0.90 |

Bulgu çarpıcı: **{2, 4} bazı 0.02 μm hata veriyor — pratik olarak
gürültü tabanı.** Geniş baz ({1,2,3,4}) ile karşılaştırıldığında **2000×
daha iyi.**

### Niçin?

Geniş baz, $k=1$ ve $k=3$ için boş yere yer ayırıyor. Bu sütunlar
veride sıfır katkı taşıyor ama matrisin koşul sayısını berbat ediyor:
$\kappa$ 186'dan 13000'e çıkıyor. Gürültü büyütmesi yaklaşık olarak
$\kappa$ ile orantılı; sonuç 2000× daha kötü.

### Tek harmonik yalıtmak güvenli mi?

{2} ya da {4} tek başına denenince:

- **{2}** çalışıyor (37 μm, kor 0.89), çünkü $k=2$ harmoniği veride
  $k=4$'ten **iki kat büyük**. $k=4$ sızıntısı küçük bir
  pertürbasyon — kabul edilebilir.
- **{4}** patlıyor (1466 μm, kor negatif!). $k=2$ katkısı baskın,
  ve sadece $k=4$ ile fit edilmeye çalışılınca $a_4$ katsayısı
  absurd büyür. Sonuç anlamsız.

**Kural:** Tek bir harmoniği yalıtarak ölçmek istiyorsanız, **o
harmonik veride baskın olmalı** ya da diğer büyük harmonikleri
baza eklemeniz gerekir.

---

## 9. Pratik mesajlar ve sınırlar

### Yöntemin gücü nereden geliyor?

1. **Fiziksel uyum**: Düşük-mertebe Fourier, kapalı yörüngenin doğal
   "filtre yapısına" uyar. Yüksek harmonikler zaten ölçülemez; onları
   modelden çıkarmak veri kaybı değil, gürültü kazancıdır.
2. **Az parametre, çok ölçüm**: 48 BPM ile 4 parametre fit edildiğinde
   istatistiksel olarak büyük SNR avantajı vardır.
3. **Koşul sayısının zaferi**: Bazı doğru seçilirse $\kappa$ küçük
   tutulur ve gürültü büyütülmez.

### Sınırlar — gerçek senaryoya transfer

Yukarıdaki 0.02 μm sonucu **sinüzoidal test verisi** ile elde edildi:
$dy$ yalnızca $k=2$ ve $k=4$ içerdiğini biliyorduk. Gerçek halkada
$dy$'nin Fourier içeriği önceden tam bilinmez. İki sezgi yolu:

- **Fizikten tahmin**: Toprak hareketi, termal genleşme, tünelin
  oturması — bunlar **düşük frekanslı** hatalardır. $k=1, 2, 3$ tipik
  baskın modlardır. Bu senaryoda hedefli Fourier yaklaşımı doğal
  olarak uygulanabilir.
- **Veri-yönlü tahmin**: Önce geniş bir baz ile ölç, rezidüel
  spektrumdan hangi harmoniklerin önemli olduğunu gör, sonra dar bir
  baza geç. Bu **adaptif baz seçimi** stratejisidir (gelecek iş).

Eğer $dy$ tamamen rastgele ve düz-spektrumluysa (her quad bağımsız
olarak rastgele kaydırılmış), düşük-N Fourier yaklaşımı **model
hatası** ile sınırlanır — yöntem fiziksel olarak doğru cevap veremez.
Bu durumda 48 quad'ın tek tek hizalanmasını ölçmek için **çoklu
k-mod ölçümü** (her ölçümde farklı quad çifti) zorunludur.

### Karşılaştırma — tüm yöntemler

| Yöntem | Hata RMS | Korelasyon | Yorum |
|--------|----------|------------|-------|
| Direkt çözüm ($R^{-1}$) | 107 μm | 0.03 | $\Delta R$ tekil, başarısız |
| TSVD | 78 μm | 0.16 | Küçük modları keser, biraz iyileşir |
| Fourier $N=4$ (geniş) | 37 μm | 0.88 | Fiziksel parametrelendirme |
| **Fourier {2,4} (sıkı)** | **0.02 μm** | **1.000** | **Doğru baz, gürültü tabanı** |

### Anahtar tasarım kararları

- **Bazda olmayan harmonik veride var ise**: sızıntı + büyük hata.
- **Bazda olan ama veride sıfır harmonik**: zarar yok, ama $\kappa$
  büyür (gürültü büyütmesi).
- **Optimal denge**: $F$ veride var olan harmoniklere **sıkıca**
  hizalanmış olmalı, ne az ne fazla.

Bu, klasik tepki-matrisi tabanlı yöntemin neden iki-quad k-mod
rejiminde başarısız göründüğünü ve neden gerçekte bilgi taşıdığını
gösteriyor: bilgi vardı, ama parametrelendirmemiz onu çıkarmak için
yanlış üzerinde duruyordu. Doğru baz, doğru sonuç.

---

## Ek: Kod akışı

```
1. build_response_matrix.py:
     R₁, R₂  ←  iki konfigürasyonda her quad'ı sırayla kaydır + simüle
     ΔR     =  R₂ − R₁

2. test_kmod_reconstruction.py:
     (a) Gerçek dy/dx üret (rastgele veya smooth)
     (b) İki konfigürasyonda parçacık takibi koş, y₁ ve y₂'yi al
     (c) Δy = y₂ − y₁
     (d) Çöz:
         - Direkt:   dy = ΔR⁻¹ · Δy
         - TSVD:     küçük tekil değerleri kes
         - Fourier:  dy = F · (ΔR·F)⁺ · Δy
```

Fourier seçimi kod içinde:

```python
F = fourier_basis(n_q=48, k_list=[2, 4])   # sıkı baz
M = dR @ F
a, *_ = np.linalg.lstsq(M, delta_y, rcond=None)
dy_geri = F @ a
```

Üç satır. Tüm güç, bazın bu üç satırdan önce doğru seçilmesinde.
