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

> **Kritik ayrım:** Quad kaçıklığı $R$ üzerinden BPM'lere *geçer* ve k=2'de
> 167× kuvvetlenir. BPM elektronik ofseti ise $R$'yi *bypass eder* ve
> doğrudan ölçüme eklenir — hiç kuvvetlendirilmez. Bu asimetri yöntemin
> temelidir (Bölüm 8).

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

### Tepki matrisinin fiziksel yapısı: R = G · diag(K_j)

Tepki matrisi $R$'yi iki faktöre ayırmak, hem "neden BPM ofseti bypass
eder" hem de "neden k=2 özel" sorularını tek seferde yanıtlar:

$$R = G \cdot \mathrm{diag}(K_j).$$

**Birinci faktör: diag(K_j) — kaçıklık→kick kazancı**

$K_j = k_j\ell_j$, $j$'inci quadupolun normalize gradyanı ile efektif
uzunluğunun çarpımıdır; yani "1 m kaçıklık ne kadar açısal kick yaratır"
sorusunun yanıtıdır. QF quadupoller pozitif, QD quadupoller negatif kick
verir. Bu QF/QD işareti değişimi $(-1)^j$ faktörüyle Fourier bazımıza
zaten gömülü olduğundan, $\mathrm{diag}(K_j)$ sadece mekanik bir kazanç
katsayısıdır — optik içinde hiçbir yayılma yoktur.

**İkinci faktör: G — betatron Green fonksiyonu (kick→yörünge)**

$G$, $j$'inci elemanda uygulanan ince bir açısal kick'in $i$'inci BPM'de
yarattığı kapalı yörünge kaymasını verir:

$$G_{ij} = \frac{\sqrt{\beta_i\,\beta_j}}{2\sin\pi Q_y}
           \cos\!\bigl(|\mu_i-\mu_j|-\pi Q_y\bigr).$$

Burada $\beta_i$ Courant–Snyder genlik fonksiyonu, $\mu_i$ faz ilerleme
miktarı ve $Q_y \approx 2.68$ dikey tune'dur. Bu formül, tek-tur harita
denkleminin analitik çözümünden gelir. Dikkat edilecek en önemli nokta:
$G$ \emph{neredeyse simetrik}tir ($G_{ij}\approx G_{ji}$, sayısal
asimetri 0.005) — yani kick'in "hangi taraftan" uygulandığı neredeyse
fark etmez.

$R$'yi bu şekilde yazdığımızda, tepki matrisini
**"örgü boyunca manyetik alanların BPM'ler üzerindeki kazanç fonksiyonu"**
olarak okuyabiliriz: $\mathrm{diag}(K_j)$ fiziksel kaçıklığı açısal
kick'e çevirir, $G$ ise bu kick'i optik yapı aracılığıyla BPM gözlemine
yayar.

**Bu ayrışımın iki kritik sonucu**

| Soru | Cevap (R = G · diag(K_j) perspektifinden) |
|------|------------------------------------------|
| **Neden BPM ofseti R'yi bypass eder?** | $\mathbf{b}$, ölçüm uzayına eklenir — $G$'den *sonra*. Zincirleme $\Delta q \to K_j\Delta q \to G \to \text{yörünge}$ içinde hiç yer almaz; çıkışa doğrudan eklenen bir ölçüm saçılması olarak kalır ve $G$'nin yarattığı hiçbir yükseltime maruz kalmaz. |
| **Neden k=2 rezonant?** | $(-1)^j K_j$ FODO işaret değişimi, pürüzsüz $k$-harmonik kaçıklığı $G$'ye $Q_y$'ye yakın uzaysal frekansta bir kick örüntüsü olarak aktarır. $G$, rezonans frekansına ($Q_y$'ye) en yakın drive'ı en çok kuvvetlendirir. $Q_y\approx2.68$ için bu frekans $k=2$'dir — işte bu yüzden $\|M_{k=2}\|=167$, $\|M_{k=4}\|=16$ (10× fark), ve BPM ofseti sağlamlığı hiyerarşisi bu 10× üzerine kurulur. |

Kısaca: quad kaçıklığı $R$ zinciri boyunca ilerler ve $G$'nin rezonans
davranışına göre kuvvetlenir ya da zayıflar. BPM elektronik ofseti ise
bu zincirin tamamen dışındadır — sinyal işleme açısından bambaşka bir
yerde durur.

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
| 6 | 0.008           | −5.92×10⁻¹¹          | 0.041 |
| 7 | 0.005           | −4.73×10⁻¹¹          | 0.033 |

k=2 baskındır: k=1'den 2.9×, k=3'ten 2.2× büyük. k=6,7'de tepe değerin %4'üne iniyor. Bu Omarov et al.
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

| k | \|\|RF_k\|\| | \|\|M_k\|\| | dSy/dt [rad/s] |
|---|------------|---------|----------------|
| 1 | 8.8        | 43.0    | 4.9×10⁻¹¹      |
| **2** | **34.1** | **167** | **1.4×10⁻¹⁰** |
| 3 | 8.9        | 43.6    | 6.6×10⁻¹¹      |
| 4 | 3.2        | 16      | 2.0×10⁻¹¹      |
| 6 | 1.1        | 5.5     | 5.9×10⁻¹²      |
| 8 | 0.59       | 2.9     | 4.7×10⁻¹²      |

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

### Lineer cebir özü: matched filter ve koşullanma

> **Soru:** Tepki matrisi k=2'yi aşırı kuvvetlendirirken, ters-projeksiyon onu
> aşırı bastırıyor. Bu nasıl aynı anda olabilir?
>
> **Cevap:** İkisi aynı olgunun iki yüzüdür. Estimatör, tek bir sütunun
> Moore–Penrose sözde-tersidir:
>
> $$\hat{a}_{k=2} = \frac{M_{k=2}^T y}{\|M_{k=2}\|^2} = M_{k=2}^{+}\,y, \qquad \|M_{k=2}^{+}\| = \frac{1}{\|M_{k=2}\|} = \frac{1}{167}$$
>
> Bir vektörün sözde-tersinin normu, normunun tersidir → ileri kazanç ne kadar
> büyükse, geri kazanç o kadar küçük.

**Karşılıklılık (reciprocity).** Sinyal ve ofset veri yoluna farklı
noktalardan girer, dolayısıyla farklı net kazanç görür:

| | yol | net kazanç |
|---|---|---|
| **Sinyal** ($\Delta q = A F_2$) | $\xrightarrow{\times R}$ ölçüm $\xrightarrow{\times M_2^{+}}$ kestirim | $M_2^{+}(A M_2) = A$ → **1** |
| **Ofset** ($b$) | doğrudan ölçüme $\xrightarrow{\times M_2^{+}}$ kestirim | $\|M_2^{+}b\| \le \|b\|/167$ → **1/167** |

Sinyal $R$'den hem ileri (×167) hem geri (×1/167) geçer → birbirini götürür →
birim kazanç. Ofset sadece geri yoldan geçer → 167'ye bölünür.

**SVD resmi.** $R = U\Sigma V^T$ ile $\sigma_{\max} = 34.7$, $\sigma_{\min} = 0.14$,
$\kappa = 249$. k=2 misalignment yönü $f_2$'nin **%91'i en büyük tekil yöndedir**
($\|Rf_2\| = 34.1 \approx \sigma_{\max}$) — yani k=2, $Q_y \approx 2.68$ rezonansına
en yakın olduğundan $R$ spektrumunun **tepesine** oturur.

- **Naif tam ters** $R^{-1} = V\Sigma^{-1}U^T$ tehlikelidir çünkü $1/\sigma_{\min}$ ile
  küçük tekil yönleri patlatır → ofseti $\kappa = 249$ kat büyütür.
- **Tek-mod projeksiyon** ise büyük-$\sigma$ alt-uzayında çalışır; orada sözde-ters
  gürültüyü **bastırır**.

| | koşullanma sayısı |
|---|---|
| Tam 48-boyutlu ters | 249 (kötü) |
| Tek-sütun projeksiyon | **1.000** (mükemmel) |

**Matched filter kazancı.** Estimatör, şablonu k=2 yörünge parmak izi $M_{k=2}$
olan bir eşleşmiş filtredir; SNR kazancı $\|M_{k=2}\|$'dir:

$$\text{SNR}_\text{çıkış} = \frac{A\,\|M_{k=2}\|}{\sigma_b} = \frac{10 \times 167}{300} \approx 5.6$$

Yani yanlış-EDM'i en çok besleyen modun aynı zamanda en gürültü-dayanıklı mod
olması tesadüf değil — ikisi de $\|M_{k=2}\| = 167$'nin $R$ spektrumunun tepesine
oturmasının sonucudur.

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

### Üç genlik seviyesi (fig3)

BPM ofseti sağlamlığını anlamanın en doğrudan yolu, sistemin ürettiği
üç farklı genlik ölçeğini karşılaştırmaktır:

**1. Sinyal yörüngesi — 1670 μm**
10 μm'lik bir $k=2$ quad kaçıklığı, tepki matrisi $R$ üzerinden BPM'lere
yansır ve rezonant kuvvetlendirme nedeniyle $A \cdot \|M_{k=2}\| = 10 \times 167 = 1670\,\mu$m
genliğinde bir kapalı yörünge oluşturur. Bu değer pik BPM sapması (~384 μm) değil, tüm BPM'lerin RMS normudur; pik BPM sapması 384 μm'dır. κ(R)=249 koşullanma sayısı ile karıştırılmamalıdır — koşullanma sayısı yörünge kazancı değil, R matrisinin en büyük ve en küçük tekil değerlerinin oranıdır. Kaçıklık $R$'den *geçer*,
167× büyür.

**2. BPM ofseti Fourier seviyesi — ~77 μm**
BPM ofseti $\mathbf{b}$, tepki matrisinden *geçmez* — doğrudan ölçüme
eklenir. $\mathbf{b}$ beyaz gürültü olduğundan FODO-antisimetrik Fourier
bazındaki her moda eşit enerji dağılır:

$$|a_k(b)| \approx \sigma_b\sqrt{\pi/48} \approx 77\,\mu\text{m}, \quad \text{tüm } k \text{ için}$$

k=2'de özel bir yığılma yoktur. Ve bu 77 μm, sinyal yörüngesi olan
1670 μm'nin yanında zaten **22× küçüktür**.

**3. Estimatör çıkışındaki kirlenme — ~1.8 μm**
Estimatör $F_k$'ya değil, $M_k = R \cdot F_k$'ya yansıtır.
Bu, ofseti bir kez daha $\|M_{k=2}\| = 167$'ye böler:

$$\delta\hat{a}_{k=2} = \frac{\sigma_b}{\|M_{k=2}\|} = \frac{300}{167} \approx 1.8\,\mu\text{m}$$

**Hiyerarşi: $1670 \gg 77 \gg 1.8\,\mu$m**

$\|M_{k=2}\|$'nin kuvvetlendirmesi iki kez işe yarar:
gerçek sinyali BPM uzayında büyütür, ofsetin kestirime sızmasını ise bastırır.

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

---

## 12. Spin-eşdeğer radyal manyetik alan [nT] <a name="12-nT"></a>

### Dönüşüm mantığı

Quad hizalama hataları yerine doğrudan **halka boyunca yayılan radyal manyetik alan** dili kullanmak istiyorsak, iki bağımsız spin-takibi taramasının oranını kullanabiliriz:

- **(A) Harici harmonik:** B_x(θ) = A_r cos(Nθ), A_r = 1 nT → dSy/dt_ext(N)
- **(B) Quad kaçıklığı:** Δy_j = A·F_k[j], A = 10 μm → dSy/dt_quad(k)

Dönüşüm faktörü:

$$c_k = \frac{R_q(k)}{R_B(k)} = \frac{dS_y/dt_\text{quad}(k)/\mu m}{dS_y/dt_\text{ext}(k)/nT} \approx 6~\text{nT/μm}$$

Bu faktör k=1..7 için ±%17 içinde **neredeyse sabittir**. Fiziksel anlam: spin entegre alana (yörünge başına toplam radyal impulse) tepki verir; quadlar halkanın ~%3'ünü doldurduğundan lokal pik alan (~210 nT/μm) spin-eşdeğer alandan ~34× büyüktür.

### Sinyal modları

| Mod | Kaçıklık | Spin-eşdeğer B_x |
|-----|----------|-----------------|
| k=2 | 10 μm | **54 nT** harici 2-harmonik |
| k=3 | 8 μm | **46 nT** harici 3-harmonik |

(Lokal quad alanı 2100 nT ve 1680 nT olurdu — ama spin buna tepki vermez.)

### Ölçüm hassasiyeti [nT cinsinden]

BPM sistematik ofsetinden kaynaklanan k=2,3 modlarındaki belirsizlik:

$$\sigma(B_{x,\text{eq}}^{(k)}) \approx \frac{c_k \cdot \sigma_b}{\|M_k\|}$$

| σ_offset | k=2 [nT] | k=3 [nT] |
|----------|----------|----------|
| 5 μm (BBA hedefi) | 0.16 | 0.69 |
| 20 μm (BBA tipik) | 0.65 | 2.8 |
| 100 μm (BBA öncesi) | 3.3 | 14 |

**1 nT eşdeğer hassasiyet için gereken BPM sistematik ofseti:**
- k=2: σ_offset ≲ **21 μm** (BBA sonrası ulaşılabilir)
- k=3: σ_offset ≲ **5 μm** (BBA hedefi gerekir)

### Özet cümle (makale için)

> "Quad'lar ve diğer manyetik alan kaynakları ~20 μm düzeyine hizalanırsa, halka boyunca yayılan radyal manyetik alanın k=2 harmonik bileşeni ~1 nT hassasiyetle ölçülebilir; k=3 için ~4 nT hassasiyet BBA hedefiyle (σ_offset~5 μm) 1 nT'ye indirilebilir."
