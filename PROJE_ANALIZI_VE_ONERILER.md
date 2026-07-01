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
- $\mathbf{b}$: BPM elektronik ofsetleri, statik ama bilinmez (~100 μm)
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

---

### 2.6 Hedefli Fourier rekonstrüksiyonu ✓

Bu bölümde "iki quad modüle ederek halka boyunca Fourier bileşenlerini
ölçemedik" saptamasına nasıl ulaşıldığını ve doğru baz seçiminin neden
dramatik sonuç verdiğini ayrıntılı olarak açıklıyoruz.

#### Problemin kökleri: ΔR neden rank-2?

İki-quad kmod'da ($j_1, j_2$ gradyanı değişir, kalan 46 quad sabit),
$\Delta R = R(g_\text{pert}) - R(g_\text{nom})$ matrisini iki terime
ayırmak mümkündür:

$$
\Delta R_{ij} \approx
\underbrace{K_j \,\delta g_j \cdot C(i,j)}_{\text{doğrudan: yalnız } j=j_1,j_2}
\;+\;
\underbrace{K_j \,\Delta C(i,j)}_{\text{dolaylı: tüm }j\text{de, büyüklük }\sim\varepsilon^2}
$$

Birinci terim yalnız $j = j_1$ ve $j = j_2$ sütunlarında sıfırdan
farklı; büyüklüğü $\sim\varepsilon$. İkinci terim tüm sütunlarda ama
$\sim\varepsilon^2$ küçük (beta-beat etkisi). SVD spektrumunda bu
$\sigma_1, \sigma_2 \gg \sigma_3, \ldots$ olarak görünür: $\Delta R$'nin
etkin rankı 2'dir, kondisyon sayısı $\kappa \sim 1/\varepsilon \sim 10^6$.

Direkt tersini almak ($\Delta R^{-1}\Delta\mathbf{y}$) tüm 48 quad
hizalama hatasını bulmaya çalışır — ama elinde yalnızca 2 bağımsız
"boyut" vardır. Kalan 46 boyut gürültüyle dolar; hata yüzlerce μm'ye
çıkar.

#### Neden iki quad Fourier bileşenlerini ölçmeye yetmedi?

İki quad modüle ederek halka boyunca Fourier bileşenlerini belirlemek
isteniyor. Sezgi şuydu: her quad faz ilerlemesiyle sinüsoidal tepki
üretir; farklı konumlardaki iki quad birbirini tamamlayarak halkanın
tüm frekans içeriğini kapsayabilir.

Bu sezginin çöktüğü iki ayrı neden var.

**Birinci neden — sayım yetersizliği:**

Her modüle edilen quad, $\Delta q$ uzayında **tek bir doğrusal
kombinasyonu** ölçebilir (rank ~1 katkı). Modüle edilen quad $j_1$
için baskın sağ tekil vektör $v_{j_1}$, o quadin faz uzayındaki
konumuna ve örgü optiklerine göre belirlenen bir yöndür. Ölçülen
büyüklük yalnızca $v_{j_1}^T \Delta q$ — 48 boyutlu $\Delta q$'nun
bir skaler projeksiyonu.

Tek quad → 1 denklem. İki quad → 2 denklem. $k=0$ ve $k=2$
harmoniklerini çıkarmak için 3 bilinmeyen ($a_0$, $a_{2c}$, $a_{2s}$)
var. İki ölçüm üç bilinmeyeni belirleyemez; sistem yetersiz belirlenmiş.

**İkinci neden — frekans uyumsuzluğu (daha derin olan):**

Sayım yeterli olsa bile ölçüm yönleri istenen harmoniklerle hizalı
değildir. Bunu anlamak için $v_{j_1}$'in ne olduğuna bakalım.

Quad $j_1$ modüle edildiğinde, $\Delta R_{(j_1)}$'in baskın sağ tekil
vektörü yaklaşık olarak:

$$
v_{j_1,j} \;\propto\; K_j\sqrt{\beta_j}\,\cos\!\bigl(|\phi_j - \phi_{j_1}| - \pi\nu\bigr)
$$

Bu, **tune frekansında** titreşen bir fonksiyon: halka boyunca $j_1$
konumu etrafında $\cos(\nu \cdot 2\pi j / N)$ tipinde osilas yapıyor.
Burada $\nu \approx 2.68$ — **tam sayı değil, irrasyonel**.

Ölçmek istediğimiz Fourier harmonikleri ise $\cos(2\pi k j / N)$
biçiminde, **tam sayı** $k$ ile titreşiyor: $k = 0, 1, 2, \ldots$

$\nu$-frekanslı bir ölçümden $k$-frekanslı bir harmonik çıkarmak,
irrasyonel ile tam sayı frekanslar arasında dönüşüm yapmak demektir.
Bu dönüşüm $M = \Delta R \cdot F$ matrisinin yüksek koşul sayısına
($\kappa \sim 10^4$–$10^6$) yol açar.

Bir analogiyle: 442 Hz'de tepki veren bir dedektörle 440 Hz'i
ölçmeye çalışmak gibi. Sinyal var, dedektör kısmen duyuyor, ama
"doğal frekansı" tam değil; hassasiyet düşük ve gürültüye açık.
Dedektörün doğal frekansı 440 Hz'e tam eşit olsaydı ölçüm mükemmel
olurdu — targeted Fourier başarısının özü budur: baz tam doğru
harmoniklere hizalanınca $\kappa$ 13000'den 186'ya iniyor.

**Sonuç:** "İki quad faz ilerlemesiyle halkanın tamamını kapsar"
sezgisi kısmen doğru — tepki sinüsoidal olarak değişiyor. Ama bu
sinüs **tune frekansında**, ölçmek istediğimiz Fourier harmonikleri
ise **tam sayı frekanslarda**. İki dil örtüşmüyor. Doğru baz
seçildiğinde bu uyumsuzluk minimize ediliyor; bilinmeyen harmonikler
varlığında ise kaçınılmaz bir hassasiyet kaybı oluyor.

#### Yeniden parametrelendirme fikri

Peki gerçek halkada $\Delta q$ (quad hizalama hata vektörü) nasıl
bir yapıya sahiptir? Yerçekimi, termal genleşme, tünel oturması gibi
fiziksel mekanizmalar halka boyunca **uzun dalgalı** bozulmalar üretir.
Bu, $\Delta q$'nun Fourier içeriğinin düşük frekans bileşenlerinde
yoğunlaştığı anlamına gelir. 48 bağımsız sayı yerine şunu yazalım:

$$
\Delta q_j = a_0 + \sum_{k=1}^{N} \left[a_k \cos\!\left(\frac{2\pi k j}{48}\right)
+ b_k \sin\!\left(\frac{2\pi k j}{48}\right)\right]
\quad \Rightarrow \quad \Delta q = F\,\hat{a}
$$

$F$ matrisi (48 × $n_\text{baz}$) Fourier baz fonksiyonlarını içerir;
$\hat{a}$ katsayı vektörü bulunmak isteniyor. Bunu k-mod denklemine
koyalım:

$$
\Delta\mathbf{y} = \Delta R \cdot \Delta q = \Delta R \cdot F \cdot \hat{a}
\equiv M\,\hat{a}
$$

Şimdi $M = \Delta R \cdot F$ matrisi $48 \times n_\text{baz}$
boyutlu. $n_\text{baz} \ll 48$ seçilirse sistem aşırı-belirlenmiş
(overdetermined) ve en küçük kareler ile çözülebilir:

$$
\hat{a} = (M^T M)^{-1} M^T \Delta\mathbf{y},
\qquad \widehat{\Delta q} = F\,\hat{a}
$$

Sorun şu: $\kappa(M)$ ne kadardır?

#### Neden doğru baz dramatik fark yaratır?

$\Delta R$'nin "güçlü" tekil yönleri ($u_1, u_2$ sol tekil vektörleri;
$v_1, v_2$ sağ tekil vektörleri) FODO örgüsünün doğal harmoniklerine
hizalıdır — tune rezonansına yakın düşük-$k$ modlar bu yönlerde baskın.
$F$'yi bu güçlü yönlere **sıkıca hizalanmış** seçersek $M = \Delta R \cdot F$
yalnızca "kuvvetli" modları taşır ve $\kappa(M) \ll \kappa(\Delta R)$.

Sayısal olarak ne olduğu:

| Baz $F$ | $\kappa(M)$ | RMS hata | Korelasyon | Açıklama |
|---------|-------------|----------|------------|----------|
| Direkt $\Delta R^{-1}$ (48 sütun) | ~10⁶ | 107 μm | 0.03 | Tüm sütunlar, gürültü patlar |
| TSVD (4 mod tutulur) | — | 78 μm | 0.16 | Küçük modlar kesilir, biçim bozulur |
| Geniş Fourier $k = 0..4$ (9 sütun) | 1.3×10⁴ | 35 μm | 0.88 | Doğru k'lar var ama sahte k'lar da |
| **Sıkı Fourier $\{k=2, k=4\}$ (4 sütun)** | **186** | **0.02 μm** | **1.000** | Verideki tek harmonikler |

Neden $\{k=2, k=4\}$ bu kadar iyi çalışıyor? Çünkü simülasyonda dy
yalnızca bu iki harmonikten oluşacak şekilde üretildi. $M = \Delta R \cdot F_{\{2,4\}}$,
4 sütunlu bir matris. $\Delta R$'nin 2 büyük tekil değeri bu 4 sütuna
iyi yansıyor. Sistem aşırı-belirlenmiş (48 denklem, 4 bilinmeyen),
gürültü 48/4 = 12 bağımsız ölçüm ortalamasıyla baskılanıyor.
$\kappa = 186$ küçük → gürültü büyütmesi yalnızca ~14 kat.

Geniş bazda ($k = 1..4$) ise $k=1$ ve $k=3$ için boş sütunlar var.
Bu sütunlar veride sıfır katkı taşır ama $\kappa$'yı 186'dan 13000'e
çıkarır; gürültü 70 kat büyütülür → 35 μm hata.

#### Ne anlama geliyor, ne anlama gelmiyor?

**Anlama geliyor:** Hizalama hatalarının halka boyunca *Fourier
içeriği önceden biliniyorsa* (örneğin mühendislik ölçümünden veya
fiziksel öngörüden), tek-kmod ölçümü bile o katsayıları belirleyebilir.
Koşul sayısı küçük, gürültü bastırılmış.

**Anlama gelmiyor:** Bu yöntem keyfi, bilinmeyen bir hizalama
dağılımında çalışmaz. Eğer gerçek sinyal $k=3$ harmonik içeriyorsa
ama baz {2, 4} seçildiyse, LSQ bu katkıyı en yakın baz fonksiyonuna
(k=2 veya k=4) dağıtır ve katsayılar kirlenir. Baz eksik harmonik
içeriyorsa sistematik hata kaçınılmaz.

**Temel sınır:** Halka boyunca harmonik bileşen sayısı kadar
*bağımsız kmod ölçümü* gerekir. $k=0$ ve $k=2$ bileşenlerini
bulmak için 3 bilinmeyen (DC, cos₂, sin₂) ve en az 3 bağımsız ölçüm
gereklidir. İki-quad kmod bu üçü veremez; çok-konfigürasyon yığma
bunun için geliştirildi (§2.7).

---

### 2.7 Çok-konfigürasyon yığma çerçevesi ✓

#### Sorun: tek kmod ölçümü neden yetmez?

Tek bir quad ($j$) modüle edildiğinde $\Delta R_{(j)}$ matrisinin
SVD ayrışımına bakarsak:

$$
\Delta R_{(j)} \approx \sigma_j\, u_j\, v_j^T
\quad\Rightarrow\quad
\Delta R_{(j)}\,\Delta q \approx \sigma_j\,(v_j^T \Delta q)\, u_j
$$

Tüm bilgi $v_j^T \Delta q$ skalerinde — yani $\Delta q$'nun tek bir
yöne projeksiyonu. Bu bir **tek denklem** anlamına gelir.

k=0 + k=2 harmoniklerini bulmak için 3 katsayı ($a_0$, $a_{2c}$, $a_{2s}$)
belirlemek istiyoruz. Her tek-quad kmod ölçümü 1 bağımsız denklem verir.
En az 3 bağımsız kmod ölçümüne ihtiyaç var.

#### Hangi quad'lar seçilmeli?

$j_1$, $j_2$, $j_3$ quad'larından gelen ölçümlerin bağımsız olması
için, her quad'ın "sondaj yönü" $v_{j_i}^T F$ (Fourier baz uzayına
projeksiyonu) lineer bağımsız olmalıdır. Sezgisel olarak: eğer tüm üç
quad aynı FODO fazında konuşlandırılmışsa, üçünün de $k=2$ harmonik
üzerindeki projektsiyon değerleri aynı işaret ve büyüklüğe sahip olur
→ üç "farklı" ölçüm aslında aynı bilgiyi taşır.

Seçilen quad'lar için k=2 harmonik projeksiyon değerleri:

| Quad indeksi | FODO hücresi $n$ | $\cos(2\pi \cdot 2 \cdot n/24)$ |
|---|---|---|
| $j = 1$ | 0 | +1.000 |
| $j = 3$ | 1 | +0.866 |
| $j = 9$ | 4 | −0.500 |

İkisi pozitif, biri negatif: cos ve sin bileşenlerinin ayrıştırılması
için yeterli açısal çeşitlilik mevcut. $\kappa$ düşük.

#### Yığılmış sistem nasıl çözülür?

Üç ayrı kmod ölçümünden gelen diferansiyel yörünge vektörleri
($\Delta\mathbf{y}_{c0}$, $\Delta\mathbf{y}_{c1}$, $\Delta\mathbf{y}_{c2}$)
dikey olarak yığılır:

$$
\underbrace{\begin{pmatrix}\Delta R_{c0}\\\Delta R_{c1}\\\Delta R_{c2}\end{pmatrix}}_{144\times 48}
F =
\underbrace{M_\text{stack}}_{144\times 3},
\qquad
M_\text{stack}\,\hat{a} =
\begin{pmatrix}\Delta\mathbf{y}_{c0}\\\Delta\mathbf{y}_{c1}\\\Delta\mathbf{y}_{c2}\end{pmatrix}
$$

Her $\Delta R_{ci}$ rank ~1 katkı verdiğinden $M_\text{stack}$ rank ~3.
3 bilinmeyen, 144 denklem → son derece aşırı-belirlenmiş; gürültü
$\sqrt{144/3}$~7 kat bastırılır. Kod altyapısı (`build_response_matrix.py --config N`,
`reconstruction.py`) hazır; sayısal doğrulama kısmi kalmıştır.

---

### 2.8 Greedy ve LASSO başarısızlığının analizi ✓

Her iki yöntem de rank-2 $\Delta R$ ile karşılaşınca farklı ama
birbirine bağlı nedenlerle çöker.

#### Greedy: yanlış harmonik seçimi (tekil vektör ötüşmesi)

Greedy matching pursuit her adımda "rezidüeli en çok düşüren Fourier
sütununu" baza ekler. $k = 0, 1, \ldots, k_\text{max}$ aday harmonikleri
için şu miktarı minimize eder:

$$
\|r_k\|^2 = \|\Delta\mathbf{y}\|^2 -
\frac{\bigl(\Delta\mathbf{y}^T \Delta R\, f_k\bigr)^2}{\|\Delta R\,f_k\|^2}
$$

$f_k$: k'ıncı FODO-antisimetrik Fourier sütunu. Paydaki
$\Delta R\,f_k$ vektörü yalnızca $\Delta R$'nin güçlü tekil yönlerinde
bileşen taşır:

$$
\Delta R\,f_k \approx \sigma_1\,(v_1^T f_k)\,u_1 + \sigma_2\,(v_2^T f_k)\,u_2
$$

Greedy'nin seçtiği $k^*$: $\Delta R\,f_k$'yı $\Delta\mathbf{y}$'ye en
paralel yapan $k$. Ama $\Delta\mathbf{y} \approx \Delta R\,\Delta q$
olduğundan bu projeksiyon $\Delta q$'nun $v_1$ ve $v_2$ yönlerine
uyum içindeki harmonikleri — fiziksel harmonikleri değil — seçer.

$v_1$ ve $v_2$ vektörleri, iki modüle edilen quad'ın konumuna ve örgü
optiklerine bağlı keyfi vektörler. Gerçek harmonikler ($k = 0, 2, 4$)
bunlarla hizalı olmak zorunda değildir. Sayısal doğrulama:

```
Gerçek harmonikler: k = 0, 2, 4
Greedy seçimi      : k = 1, 8, 7   (korelasyon ~0.40)
```

Greedy rezidüeli düşürüyor ama fiziksel olmayan harmonikleri seçiyor.
Bu hata algoritmada değil, problem yapısındadır: rank-2 matris yalnızca
2 boyutlu bilgi taşır; greedy bu 2 boyutu en çok açıklayan iki "yönü"
seçer ve bunlar gerçek harmoniklerle örtüşmüyor.

**Rank arttığında greedy kurtarılabilir mi?** Evet. 3 veya daha fazla
bağımsız kmod ölçümü yığıldığında $M_\text{stack}$'in tekil vektörleri
artık fiziksel FODO harmoniklerine hizalanır. Greedy bu durumda doğru
harmonikleri seçer.

#### LASSO: rank-yoksulluk çöküşü

LASSO tüm harmonikleri ($k = 0, 1, \ldots, 12$, 25 sütun) adaya koyar
ve L1 ceza ile gereksizleri sıfıra iter:

$$
\hat{a} = \arg\min_a \tfrac{1}{2}\|M a - \Delta\mathbf{y}\|^2 + \lambda \|a\|_1,
\qquad M = \Delta R \cdot F_\text{full}
$$

Sorun: $M$ matrisi 48×25 boyutlu ama rank ~2. Bu demektir ki
$M$'nin 23 boyutlu bir null uzayı vardır — 25 katsayı vektörünün 23
boyutu tamamen serbesttir, hiçbir veriyi açıklamıyor. ADMM çözücüsünde
$x$ güncellemesi:

$$
x \leftarrow (M^T M + \rho I)^{-1}(M^T \Delta\mathbf{y} + \rho z)
$$

$(M^T M)$'nin 23 özdeğeri sıfır; $\rho I$ düzenlileştirmesi
tüm 25 boyutu eşit ağırlıklı hale getirir. Sinyal enerjisi 25 katsayıya
bölünür: her katsayı $|\hat{a}_i| \approx \|\Delta\mathbf{y}\| / 25
\approx 0.4\,\mu\text{m}$. LASSO eşiği $\lambda/\rho = 0.02$ ise
tüm katsayılar eşiğin altına düşer:

$$
\mathcal{S}_{0.02}(0.01) = 0 \quad \text{(her katsayı sıfırlanır)}
$$

$\lambda$ büyütülürse sinyal de silinir; küçültülürse 25 katsayının
hepsi eşit hayatta kalır — seçicilik yitirilir. Temel varsayım çökmüştür:
LASSO'nun çalışması için $M$'nin **Restricted Isometry Property (RIP)**
koşulunu sağlaması gerekir; rank-2 matris bunu sağlayamaz.

---

## 3. Başarılamayan / Açık Kalan Sonuçlar

### 3.1 Örgü modeli hatasının drift moduna etkisi — TAMAMLANMADI

**En kritik açık nokta.** Drift modunun 6-7 μm performansı
$R^{-1}$'in doğruluğuna bağlıdır. Gerçek halkada β-beat, tune kayması
ve faz ilerleme hatası gibi örgü bozulmaları $R$ matrisini yanlış kılar.
Kaç % model hatası → kaç μm rekonstrüksiyon hatası eğrisi çizilmeden
yöntemin pratikte kullanılabilirliği değerlendirilemez.

### 3.2 Tilt taraması — TAMAMLANMADI

Quad/dipol tilt'lerinin kırılma noktası ve hassasiyet eğrisi
belirlenmedi. Test 6'da yalnızca 0.2 mrad sabit seviyede incelendi.

### 3.3 Hata kaynağı ayrıştırması — TAMAMLANMADI

Test 6'daki 6-7 μm hatanın BPM gürültüsünden mi, model uyumsuzluğundan
mı, tilt'ten mi kaynaklandığı nicelenmiş değil.

### 3.4 Çok-konfigürasyon yığmanın sayısal doğrulaması — KISMİ

Kod altyapısı hazır, matematiksel çerçeve belgelendi; ama rank ~3
elde edildiğinin ve 3 katsayının ($a_0$, $a_{2c}$, $a_{2s}$) tutarlı
biçimde geri çatıldığının uçtan uca sayısal doğrulaması eksik.

### 3.5 Lineer model dışı etkiler — TARTIŞILMADI

BPM gain hataları, BPM roll, sekstupol feed-down, fringe alanlar,
manyetik histerezis, RF akım dalgalanması — bunların gerçek halkada
ne kadar hata yarattığı bilinmiyor.

---

## 4. Açık Konular İçin Öneriler

### Öneri 1 — Test 8: β-beat hassasiyeti (öncelik: YÜKSEK)

Nominal lattice ile hesaplanan $R$ matrisini sabit tut; simülasyonda
beta fonksiyonunu yapay olarak boz:

$$
\beta(s) \to \beta(s)\bigl(1 + \delta_\beta \cos(2\pi s / C)\bigr)
$$

$\delta_\beta \in [0.5\%, 5\%]$ taraması yap, her noktada drift modu
hatası ölç. LOCO kalibrasyonu gerçek halkalarda %1 altında β-beat
sağlar; bu seviyede drift modunun 10 μm hedefini koruyup korumadığı
makale için belirleyici bir sonuçtur.

---

### Öneri 2 — Çok-konfigürasyon yığmanın uçtan uca koşulması (öncelik: ORTA)

Mevcut durum: her bileşen ayrı çalışıyor ama uçtan uca sonuç
raporlanmamış.

**Somut adım:** `params.json`'da `kmod_configs` listesi üç konfig
içeriyor ($j=3$, $j=9$, $j=1$). Şu üç komut sırayla çalıştırılırsa
yığılmış sistem kurulur:

```bash
for n in 0 1 2; do python3 build_response_matrix.py --config $n; done
python3 reconstruction.py   # R_dy_1_c0.npy varlığı çok-konfig modunu tetikler
```

**Raporlanması gereken:** (a) Yığılmış $M_\text{stack}$'in SVD
spektrumu — rank gerçekten 3 oluyor mu? (b) Geri çatılan $a_0$,
$a_{2c}$, $a_{2s}$ değerleri — gerçek değerlerle korelasyon? (c)
BPM gürültüsü 1 μm iken tahmin hatası teorik $\sigma_n/\sqrt{N_c}$
ile uyuşuyor mu?

---

### Öneri 3 — BPM ofset uzun-vadeli kararlılığı (öncelik: ORTA)

Test 5'te geçiş noktası ~2 μm/epoch bulundu. pEDM BPM donanımı için
beklenen ofset kayma hızını literatürden derleyip drift modunun kaç
saatte bir yeniden kalibrasyona ihtiyaç duyduğunu hesapla.

---

## 5. Beyin Fırtınası: Bu Donanımla Başka Ne Ölçülebilir?

**Donanım:** Halka (veya linac), BPM'ler, gradyanı değiştirilebilir
quad'lar. Elde edilen büyüklük: differential orbit $\Delta\mathbf{y} =
\mathbf{y}_\text{pert} - \mathbf{y}_\text{nom}$.

Quad hizalama hatalarını ve harmonik bileşenlerini doğrudan bu
donanımla ölçemedik (en azından 1-2 quad'ı modüle ederek). Peki
aynı donanımla ölçülebilecek başka önemli fiziksel büyüklükler var mı?

---

### 5.1 Quad tilt açısı

**Fizik:** Bir quad z-ekseninde $\theta$ açısıyla dönmüşse, normal
quad bileşenine ek olarak bir **skew** bileşen ürer:

$$
K_\text{skew} = K\sin(2\theta) \approx 2K\theta \quad (\theta \ll 1)
$$

Normal quad dikey kick verir; skew quad **çapraz** kick verir —
dikey hizalama hatasına yatay orbit tepkisi, yatay hizalama hatasına
dikey orbit tepkisi.

**Nasıl ölçülür:** Tilt'li bir quad'ı kmod edince hem dikey hem yatay
orbitler değişir. Şu anda kod yalnızca $\Delta\mathbf{y}$ bakıyor; ama
$\Delta\mathbf{x}$ (yatay orbit farkı) eşzamanlı kaydedilirse, çapraz
tepki tilt miktarını verir:

$$
\frac{\Delta x_i}{\Delta y_i} \approx \frac{R_{x,ij}\cdot 2K_j\theta_j}{R_{y,ij}\cdot K_j\,dy_j}
$$

**Ne kadar zorlayıcı:** Yatay $R_x$ matrisi kodda zaten var (`R_dx_1.npy`).
Eklenmesi gereken tek şey: kmod sırasında yatay orbit de kaydedilsin
ve çapraz blok $R_{xy}$ (quad tilt varlığında sıfırdan farklı) ayrı fit
edilsin. BPM roll ile karışabilir (§5.4), ayrıştırma için farklı quad
çiftleri gerekir.

---

### 5.2 BPM gain kalibrasyonu

**Fizik:** BPM'in ölçüm kazancı nominalden sapıyorsa ($g_i = 1 + \epsilon_i$),
tüm orbit okumalarında o BPM sistematik olarak $\epsilon_i \cdot y_i$
kadar yanlış ölçer. Bu, tepki matrisinin $i$. satırını
$\tilde{R}_{ij} = (1+\epsilon_i)R_{ij}$ yapar.

**Nasıl ölçülür:** Aynı kmod ölçümünü **farklı quad'larla** tekrarla.
Kmod $j_1$ ile: ölçülen $\Delta\mathbf{y}^{(1)} = G \cdot \Delta R_{(j_1)} \cdot \Delta q$.
Kmod $j_2$ ile: ölçülen $\Delta\mathbf{y}^{(2)} = G \cdot \Delta R_{(j_2)} \cdot \Delta q$.
$G = \text{diag}(1+\epsilon_i)$ gain matrisi. Teorik $\Delta R$'ler
bilindiğinden, iki ölçümün oranından $G$'nin köşegen elemanları ($g_i$)
fit edilebilir. Bu LOCO yönteminin özüdür.

**Ne kadar zorlayıcı:** Sadece iki quad'ı kmod etmek yeterli, ama gain
değişkenleri (48 BPM) için yeterli bağımsız denklem elde etmek çok
kmod ölçümü gerektirir. Kod altyapısı büyük ölçüde hazır; `R_dy_1_cN.npy`
dosyaları bu fit için kullanılabilir.

---

### 5.3 Beta fonksiyonu kalibrasyonu (tune shift yöntemi)

**Fizik:** Tek bir quad'ın gradient'ini $\Delta K$ kadar değiştirince
tune kayması:

$$
\Delta\nu_y = \frac{\beta_{y,j}}{4\pi}\,\Delta K_j\,L_j
$$

Bu ilişki tersten kullanılırsa: tune kayması ölçülürse $\beta_{y,j}$
(o quad'taki dikey beta fonksiyonu) belirlenir. Tüm quad'lar sırayla
kmod edilirse halka boyunca $\beta(s)$ profili çıkarılır.

**Nasıl ölçülür:** `integrator.py` üzerinden Poincaré kesiti hesaplanır,
buradan tune doğrudan okunabilir. Her kmod ölçümü bir $\beta$ ölçümü
anlamına gelir.

**Ne kadar zorlayıcı:** Kodun büyük kısmı hazır; eksik olan tune'u
her kmod sonrası raporlayan bir adım. Özellikle k-mod konfigürasyonunun
doğruluğunu kontrol etmek için yararlı bir iç tutarlılık testi.

---

### 5.4 BPM roll hatası

**Fizik:** Bir BPM kendi ekseninde $\phi$ açısıyla dönmüşse, okuduğu
"dikey" sinyal aslında $y \cos\phi + x \sin\phi$ karışımı. Saf dikey
bir orbit bozulmasında (kmod'un ürettiği gibi) o BPM yatay bileşen
de okur — ama bu gerçek çapraz sinyal değil, BPM'in kendi dönme hatası.

**Nasıl ölçülür:** Normal kmod → dikey orbit bozulması → tilt'li BPM
çapraz sinyal gösterir. Skew kmod (tilt'li quad, §5.1) → gerçek çapraz
yörünge. İkisini birbirinden ayırt etmek için:
aynı BPM'i, hem dikey hem yatay kick üreten farklı kaynaklarla ölç.
BPM roll her durumda aynı katkıyı verir; gerçek çapraz yörünge
fiziksel kaynağa göre değişir.

**Ne kadar zorlayıcı:** Tilt (§5.1) ve BPM roll birlikte fit edilmesi
birden fazla bağımsız kmod ölçümü gerektirir. Ortak fit mümkün ama
parametre sayısı artar (48 BPM roll + N quad tilt).

---

### 5.5 Sekstupol güç kalibrasyonu (feed-down yoluyla)

**Fizik:** Hizalanmamış bir sekstupol ($x$ veya $y$ yönünde kaymış),
sekstupol alanının $(x+iy)^2$ yapısı nedeniyle **dipol ve quad bileşeni**
üretir — buna feed-down denir. Quad bileşen kapalı yörüngeyi etkiler.
`params.json`'da `sextK1 = -0.015` ve `sextSwitch = 0`; sekstupol
şu an kapalı.

**Nasıl ölçülür:** Sekstupol açıldığında (`sextSwitch = 1`), kmod
tepkisi değişir çünkü sekstupol artık faz uzayında sıfırdan farklı
konumda traverse eden demet için ek bir etkin quad katkısı üretir.
Farklı sekstupol güçlerinde kmod tepkisinin değişimi, sekstupol
kalibre edilmesine olanak tanır.

**Ne kadar zorlayıcı:** Etki ikinci mertebede ($\sim K_2 \cdot x_\text{CO}$);
kapalı yörünge sıfır için bu katkı küçük. Kasıtlı bir yatay orbit
bump'ı ile sekstupol feed-down yükseltilirse ölçüm pratik hale gelir.

---

### 5.6 Linac için beam-based alignment (BBA)

Kapalı yörünge yoktur; yörünge başlangıç noktasından hedefe giden
bir trajektoridir.

**Fizik:** Bir quad'ın elektrik merkezi mekanik merkezinden $\delta$
kadar kaymışsa, quad'ın gradyanı değiştirildiğinde ($K \to K + \Delta K$)
o quad'taki kick değişir:

$$
\Delta\theta = -\Delta K \cdot \delta
$$

Bu ek kick aşağı yöndeki BPM'lerde pozisyon değişikliği üretir.
Eğer demet tam quad merkezinden geçseydi ($\delta = 0$), gradient
değişikliği yalnızca odaklama gücünü değiştirirdi — aşağıdaki pozisyon
trajektorisi *eğimi* değişirdi ama *konumu* değişmezdi.

**Nasıl ölçülür:** Quad gradient'ini değiştir; aşağıdaki BPM'lerde
pozisyon değişikliğine bak. Pozisyon değişikliği sıfır → demet quad
merkezinden geçiyor. Sıfır değilse → quad'ın gerçek merkezi
bulunana dek demeti kaydır (steering). Bu klasik linac BBA protokolü.

**Bu projeyle bağlantısı:** Aynı `integrator.cpp` mimarisi düz bir
linac geometrisine uyarlanabilir (periyodik sınır koşullarını kaldır).
Kmod altyapısı BBA için doğrudan kullanılabilir.

---

### 5.7 Uzaysal yük etkisinin ölçümü

**Fizik:** Düşük enerjili hızlandırıcılarda demet kendi uzaysal yük
alanını hisseder. Bu alan, dağılmış bir defocusing "sanal quad" gibi
davranır ve tune'u $\sim Q^2/\nu^3$ ile deprese eder (Laslett formülü).

**Nasıl ölçülür:** Kmod ölçümünü farklı demet yoğunluklarında
(farklı akımlarda) tekrarla. Orbit tepkisi $\Delta\mathbf{y}$ değişirse
(aynı $\Delta K$ için), bu fark uzaysal yük katkısından gelir.
Akım-bağımlı tune kaydı (§5.3 yöntemi) ile birleştirilirse Laslett
tune depresyonu kalibre edilebilir.

**Ne kadar zorlayıcı:** Şu an kodda tek bir demet yoğunluğu var.
pEDM hedef enerjisinde (~0.7 GeV) uzaysal yük küçük; etki düşük
enerjili enjeksiyon aşamasında daha belirgin.

---

### 5.8 Kuplaj katsayısı ölçümü

**Fizik:** Halka boyunca dağılmış skew quad veya tilt'li dipol
bileşenleri dikey ve yatay hareketleri birbirine bağlar. Global
kuplaj katsayısı $|C^-|$, dikey modun yatay moda "sızdırma" oranıdır.

**Nasıl ölçülür:** Normal (upright) quad'ı kmod et → saf dikey orbit
bozulması üretmeli. Eğer $\mathbf{\Delta x}$ da nonzero ise kuplaj
var demektir. Farklı quad'lardan gelen kmod ölçümlerinin çapraz blok
matrisinden $|C^-|$ fit edilebilir.

Bu ölçüm tilt kalibrasyonuyla (§5.1) iç içedir; ikisini ayırt etmek
için on-resonance ve off-resonance kmod frekanslarını (veya konumlarını)
değiştirmek gerekir.

---

### 5.9 Özet: Donanımın ölçüm kapasitesi

| Fiziksel büyüklük | Donanım yeterliliği | Zorluk |
|---|---|---|
| Quad dikey hizalama Fourier bileşenleri | Birden fazla kmod ile kısmi | Yüksek |
| **Quad tilt açısı** | Çapraz orbit tepkisiyle doğrudan | Orta |
| **BPM gain kalibrasyonu** | Çok kmod ölçümü yeterli | Orta |
| **Beta fonksiyonu** ($\beta_j$) | Tune shift ile her quad için | Düşük |
| BPM roll | Tilt ile ortak fit | Yüksek |
| Sekstupol güç kalibrasyonu | Orbit bump + kmod | Orta |
| Linac BBA | Aynı prensip, farklı geometri | Orta |
| Uzaysal yük tune depresyonu | Akım taraması + tune | Yüksek |
| Global kuplaj katsayısı | Çapraz orbit tepkisi | Yüksek |

En erişilebilir ölçüm **beta fonksiyonu** (tune shift yöntemi):
mevcut kodla hemen yapılabilir. En ilginç yeni yön **quad tilt**:
çapraz orbit tepkisi kodda zaten üretiliyor (`dR_dx` matrisi), sadece
tilt modeli ve çapraz fit eklenmesi gerekiyor.

---

## 6. Kısa Özet

| Alan | Durum |
|------|-------|
| Simülasyon altyapısı (GL4, BPM modeli) | **Tamamlandı** |
| Ofset-gürültü düalitesi teoremi | **Tamamlandı** |
| Drift modu 6 testle doğrulandı | **Tamamlandı** |
| Hedefli Fourier (0.02 μm, ideal senaryo) | **Tamamlandı** |
| FODO antisimetri parametrelendirmesi | **Tamamlandı** |
| Greedy/LASSO başarısızlığı analizi | **Tamamlandı** |
| Çok-konfigürasyon yığma çerçevesi | **Kısmi** |
| β-beat / model hatası etkisi (Test 8) | **Tamamlandı** (drift §, akıllı düzeltme §7) |
| Tilt taraması (Test 7) | **Tamamlandı** (kmod_drivers) |
| YSA ile karşılaştırma | **Tamamlandı** (NN=R⁻¹ ters problemde; §7.2) |
| Lineer model dışı etkiler | **Açık** |

> **NOT (2026-06):** Yukarıdaki tablo projenin *erken* (quad-hizalama geri-çatımı)
> fazını özetler. Proje sonradan asıl hedefe — **sahte-EDM** kontrolüne — kaydı.
> O arc'ın sentezi ve sıradaki plan **§7**'dedir (bkz. `akilli_duzeltme.md`).

---

## 7. Sahte-EDM ve "Akıllı Düzeltme": Sentez ve Plan (2026-06)

> Ayrıntılı kayıt: `akilli_duzeltme.md` (teknik, §1–6.9) + `akilli_duzeltme_pedagojik.md`
> (ders kitabı). Bu bölüm üst-düzey sentez ve **2/3/4. madde planıdır.**

### 7.1 Problemin doğru çerçevesi

Asıl amaç quad hizalama hatasını ölçmek **değil**, **sahte-EDM'i (geometrik/Berry
faz, dS_y/dt) sıfırlamaktır.** Kritik yapı:

- **Kapalı yörünge (COD) iki alt-uzaya ayrışır** (R'nin SVD'si, cond≈193):
  **antisimetrik** (QF/QD zıt-işaret, büyük-σ → **orbit-GÖRÜNÜR**) ve **simetrik**
  (QF/QD aynı-işaret, küçük-σ → **orbit-KÖR**, G_k=C/|Q²−k²| ile bastırılmış).
- **Antisim COD'u domine eder** (193× daha görünür) → simetriği okumak zorlaşır.
- **Sahte-EDM'i süren simetrik alt-uzaydır** ama bu, COD'da ~görünmez (10 μm simetrik
  misalignment → sahte-EDM ~247× hedef, ama COD ayak izi yalnız 1.7 μm).
- Antisim sahte-EDM GÖRÜNÜR (orbit-düzeltmeyle silinir, 7.7×); **simetrik sahte-EDM
  GÖRÜNMEZ** → asıl sınır budur (~62× hedef artık, paylaşılan taban).

### 7.2 Kapanan kapı: yörüngeden misalignment GERİ-ÇATMAK (her yöntem aynı duvar)

Simetrik misalignment'ı COD'dan geri-çatmak **bilgi-teorik olarak imkânsız**: izi
gürültü+sistematik tabanın altında. Bu **R'nin (fiziğin)** özelliği, *algoritmanın*
değil — dolayısıyla yöntem değiştirmek kurtarmaz:

| Yöntem | Neden başarısız | Belge |
|--------|-----------------|-------|
| R⁻¹ / TSVD / LASSO / CLEAN / Bozoki | simetrik yönde 1/σ_min büyütme | `false_edm_harmonic_sinir §14.5` |
| kmod (ΔR⁻¹) + SQUID-BPM | yine ΔR→misalignment ters; dağıtık-frekansta *nefes* | `squid_bpm_test §7,§8` |
| kmod + lock-in | beyaz gürültü √N ile yenilir ama simetrik <4nm ister + β-beat felaket | `squid_bpm_test §9.5` |
| **NN (misalignment↔COD)** | misalignment→COD LİNEER (NN=R); ters yine R⁻¹; NN=TSVD (sim. hata 5.6 vs 6.3 μm) | `akilli_duzeltme §6.8` |
| **kmod + LSTM** | beyaz gürültüde lock-in zaten optimal (Cramér-Rao); duvar gürültü değil ΔR-sistematiği; etiket döngüsel | `akilli_duzeltme §6.9` |
| **Omarov CR-ayrım ölçümü** (CW−CCW yörünge farkı) | ayrım da kapalı-yörünge-farkı → simetriğe tek-yön COD KADAR kör (bastırma CR 4.5× ≈ COD 3.8×, ölçüldü) | `omarov.md §9.3` |

**Tek cümle:** "Mükemmel COD okuyabilseydik bastırırdık" doğru — ama mükemmel okuma
simetrik alt-uzayda ~7 nm ister (100 μm ofset altında); hiçbir estimator bunu aşamaz.

### 7.3 Açık kalan kapı: TERS'İ ATLAMAK (ileri-harita / doğrudan spin)

Çözüm misalignment'ı geri-çatmak değil; ya sahte-EDM'i **doğrudan ölçmek** ya da
**ileri yönde öngörmek**:

- **Kol A — spin-doğrudan (KANITLI):** sahte-EDM'i spinle ölç (geometrik fazın
  doğrudan gözlenebiliri), knob'la null'la (spin ölç-trim ~6000×, §14.6). Bilgi
  tanımı gereği spinde. **NOT (Omarov düzeltmesi):** bu Omarov'un **SBA**'sı DEĞİL —
  SBA, spin okumalarıyla **E-alanı/vertical-velocity eksenini** hizalar, quad
  hizalama hatasını veya geometrik fazı düzeltmez (`omarov.md §5`). Omarov geometrik
  fazı **CW/CCW + polarite + CR-ayrım küçültme** ile kontrol eder; doğrudan spinle
  geometrik-faz null'lama, Omarov'un **vertical-polarizasyon** koluna karşılık gelir
  ve o makalede **kullanılmadı**.
- **Kol B — ileri-harita (kavramsal AÇIK, pratik HENÜZ ÇALIŞMIYOR):** COD→**skaler**
  sahte-EDM haritasını öğren (ters DEĞİL → iyi-koşullu), orbit-görünür knob'larla
  **null'la**. 4 pozitif bulgu (iyi-koşullu/öğrenilebilir/β-beat-şeffaf/no-go-dışı)
  AMA **kapalı-döngü null'lama basit orbit-düzeltmeyi geçemedi** (Plan 2c negatif) —
  harita mutlak doğruluğu yetersiz. Tam döküm §7.4'te.

### 7.4 Kol B tam bulgu dökümü (referans — kronolojik, `akilli_duzeltme.md`)

Bu tablo, ileri-harita (Kol B) hattında **denenen her şeyi ve sonucunu** kaydeder
ki sonradan referans olsun. (§ numaraları `akilli_duzeltme.md`.)

| # | Ne denendi | Sonuç | Yön |
|---|-----------|-------|-----|
| §1 | Estimator doğrulama (4D-CO+model-fit, p=2.002) | σ² geometrik faz teyit | ✓ temel |
| §2 | SVD ayrışım (cond=193; büyük-σ antisim, küçük-σ simetrik) | geometri kuruldu | ✓ |
| §3 | Dejenerasyon ölçümü (simetrik 10μm → f 247× hedef, COD izi 1.7μm) | ilk yorum "COD f'i belirlemez" | ⚠ sonra düzeltildi |
| §4 | **Koşullanma düzeltmesi:** ∂f/∂COD≈0.15 (mütevazı, 1/σ_min DEĞİL); 7nm ortalamayla ulaşılır | ileri-harita ≠ inversiyon; iyi-koşullu | ✓ POZİTİF |
| §5-6 | Öğrenilebilirlik: simetrik kanal CV R² 80→0.07, **240→0.77** (trend) | harita var, karmaşık, öğreniliyor | ✓ POZİTİF |
| §6.5 | **β-beat model-fidelity:** %1 β-beat şeffaf (transfer R² 0.62 = nominal 0.61) | sim-harita gerçeğe taşınır | ✓ POZİTİF |
| §6.6 | **Kullanım:** çıkarma ÖLÜ (%0.1 gerek); CW/CCW β-beat'i geri ALMAZ (corr −0.89, EDM-kanalı); yalnız null'lama | çıkarma dışlandı | ⚠ kısıt |
| §6.7 | **Büyük-genlik homojenlik:** simetrik kanal 1mm'e σ²-homojen (p=2.03); antisim doyar (p=0.9) | orbit-kör deseni büyüt→gerçek makinede öğren→ölçekle | ✓ strateji |
| §6.8 | **NN vs R⁻¹:** misalignment→COD lineer→NN=R; ters simetrikte aynı taban (NN 5.6 ≈ TSVD 6.3 μm) | fark algoritmada değil, yönde (ileri/ters) | ✗ ters ölü |
| §6.9 | **kmod/SQUID/LSTM = aynı ters duvar** (lock-in optimal; etiket döngüsel) | orbit-geri-çatım sınıfı kapalı | ✗ |
| §6.10 | **Plan 2c — kapalı-döngü null'lama (make-or-break):** map-null orbit-düzeltmeyi GEÇEMİYOR (map kazanç 0.2–23× vs orbit 45–1692×); hata tabanı ~300× hedef (slope 0.36) | **pratik HENÜZ çalışmıyor** | ✗ NEGATİF |
| §6.11 | **Algoritma iyileştirme (güvenli/ensemble/pessimistic):** naif model-istismarını dizginler (geomean 236→67×) ama orbit-null (2.1×) yine önde | gerekli-yetersiz | ⚠ kısmi |

**Kol B özet:** kavramsal olarak sağlam ve dört bağımsız testte pozitif (iyi-koşullu,
öğrenilebilir, β-beat-dayanıklı, no-go-dışı) — AMA mevcut 240-örnek empirik harita
null'lama için **çok kaba** (hata tabanı ~300× hedef). Güvenli optimizasyon zararı
önler ama sınırı kaldırmaz. **Tek kök çözüm: çok daha doğru harita → Plan 4 (analitik
Berry fonksiyoneli).**

**Ayrıca (Omarov):** CR-ayrım ölçümü de simetriğe kör (§7.2 tablosu; `omarov.md §9.3`)
→ projenin özgün katkısı olan "Omarov ölçüm-zincirinin simetrik körlüğü" doğrudan
gösterildi.

### 7.5 Harita ömrü ve β-beat: ne sıklıkla yeniden eğitilmeli?

**Önemli açıklık (yaygın karışıklık):** β-beat, ileri-haritada "büyük sahte-EDM"e
yol açmaz. Doğru tablo:
- β-beat sahte-EDM'i ~%18 **kaydırır** (§6.7) ve CW/CCW-tektir (§6.6). **İnversiyonu**
  felakete sürükler (1/σ_min, §9.5).
- Ama **ileri-harita için ŞEFFAFTIR** (§6.5): nominal-eğitimli harita, %1 β-beat'li
  makineyi held-out nominal kadar iyi öngörür (R² 0.62=0.61, **ek bozulma yok**).
  Sebep: haritaya **gerçek (ölçülen) yörünge** beslenir; sahte-EDM yörüngenin sabit
  fonksiyoneli olduğundan β-beat ikisini birlikte kaydırır, harita izler.

**Sonuç — harita ömrü β-beat'e bağlı DEĞİL:** Harita β-beat-şeffaf olduğundan,
β-beat sürüklendikçe **yeniden eğitmeye gerek yoktur** — anlık ölçülen yörüngeyi
beslersin, yeter. Yeniden eğitim yalnız **COD→f fonksiyonelini kökten değiştiren**
şeyler için gerekir: **tune değişimi, enerji değişimi, sekstüpol/optik yeniden-
konfigürasyon, büyük latis değişikliği.** Bunlar seyrektir (majör rekonfigürasyon
başına, sürüklenme başına değil).

**β-beat zaman ölçeği (gerçek hızlandırıcı):** güç-kaynağı sürüklenmesi (~10⁻⁴–10⁻³/
saat), termal (saat–gün), histerezis/rampa/refill'de sıçrama. LOCO tipik olarak
günlük/fill-başına yenilenir. Ama yukarıdaki gereği harita bunlardan **etkilenmez.**

**Kullanıcı önerisi (β-beat'li eğitim):** Makul bir *veri-artırma* (augmentation) —
haritayı β-beat çeşitliliğiyle eğitmek şeffaflığı **pekiştirir** (özellikle daha
büyük β-beat, ör. %5, için). Ama darboğaz bu değil (harita zaten ~şeffaf); asıl
darboğaz **mutlak doğruluk** (§6.10). Test edilmedi; ucuz-orta maliyetli robustluk
adımı.

> **Caveat:** Şeffaflık %1 β-beat'te ölçüldü; %5+ ve latis-değişikliği ayrıca
> sınanmalı. Ve tüm bu tartışma harita *yeterince doğru* olduğunda anlamlı (§6.10).

---

## 8. PLAN — akıllı düzeltme'yi ilerletmek

> **GÜNCELLEME (2026-06):** Plan 2c (kapalı-döngü null'lama) KOŞULDU → **negatif**
> (§6.10-6.11, §7.4). Mevcut empirik harita null'lama için yetersiz. Dolayısıyla
> **kritik yol artık Plan 4 (analitik Berry fonksiyoneli)** — o olmadan Kol B pratik
> değil. Ayrıca **Plan 5** (kullanıcı fikri: spin-modülasyon + gradient descent)
> eklendi; spin-tabanlı olduğundan harita-doğruluğu sınırını AŞAR (aşağıda).

### Plan 2 — İleri-harita'yı null'lama-hedefine taşımak (Kol B operasyonel)

**Amaç:** COD→sahte-EDM haritasını, simetrik sahte-EDM'i EDM-hedefine yaklaştıracak
**kapalı-döngü null'lama**da kullanılabilir doğruluğa getirmek.

- **2a — Veri-doğruluk eğrisi.** Ensemble'ı 240'tan ~1000–2000 örneğe çıkar; simetrik
  CV R²'nin (şu an 0.77'de doygun görünüyor) ve **mutlak nispi hatanın** N ile
  eğrisini çıkar. Çıktı: "X residual için Y örnek" tablosu. (C++ maliyeti ~saatler,
  arka plan.)
- **2b — Öznitelik/mimari.** Berry yönlü-alan + bilineer özniteliği ~0.32 tavanına
  takılıyor; per-hücre çiftli, öğrenilmiş gömme veya **fizik-bilgili Berry-çekirdeği**
  (Plan 4'ten) dene. Hedef: simetrik kanal mutlak hatasını düşürmek.
- **2c — KAPALI-DÖNGÜ NULL'LAMA DEMOSU (make-or-break).** Simülasyonda iterasyon kur:
  {COD ölç → harita f öngör → orbit-görünür knob'larla (corrector) öngörülen f'i
  azalt → tekrar}. Ölç: kalan gerçek sahte-EDM vs iterasyon. **Kritik soru:** kusurlu
  harita (çarpımsal hata hipotezi) ile null'lama hedefe yakınsıyor mu, yoksa toplamsal
  bir tabana mı takılıyor? Bu, tüm Kol B'nin pratik geçerliliğini belirler.
- **2d — Gerçekçi ölçüm.** BPM ofset (100 μm) + gürültü (1 μm) + ortalama (→~7 nm,
  §4) ekle; döngü hâlâ yakınsıyor mu? Gürültü-bütçesini (entegrasyon süresi vs
  kalan f) raporla.

**Teslim:** veri-doğruluk eğrisi; kapalı-döngü residual(iterasyon); gürültü-bütçesi.
**Risk:** toplamsal hata tabanı null'lamayı hedef-üstünde durdurur.
**Altyapı:** `build_response_matrix.py` (corrector tepkisi için R), `false_edm_4d.py`
(f ölçer); yeni: corrector-knob → COD + null'lama döngüsü scripti (/tmp).

### Plan 3 — Büyük-genlik gerçek-makine kalibrasyonu (σ²-ölçekleme)

**Amaç:** "orbit-kör deseni ~1 mm'e uyar → simetrik fonksiyoneli gerçek makinede
temiz öğren → σ² ile operasyona ölçekle" stratejisini uçtan uca doğrulamak.

- **3a — Harita-transferi (genlik).** Haritayı 1 mm (orbit-kör desen) verisinde eğit,
  σ²-ölçekleyip 10 μm'de test et: 1 mm-haritası 10 μm f'i öngörüyor mu? (Toplam f
  homojenliğini %5.6 gösterdik, §6.7; şimdi **harita** transferi.)
- **3b — Optimal genlik.** Eğitim genliğini tara (100 μm / 300 μm / 1 mm) vs
  ölçeklenmiş doğruluk + latis-lineerliği; tatlı nokta bul.
- **3c — Sekstüpol dayanıklılığı (kritik caveat).** `sextSwitch=1` ile sekstüpolleri
  aç, büyük genlikte homojenliği yeniden test et: simetrik kanal (küçük orbit)
  sekstüpolle homojen kalıyor mu? Antisim (cm orbit) ne zaman bozuluyor?
- **3d — Gerçek-makine protokol taslağı.** Hareket-ettiricilerle orbit-kör desen
  uygulama; f'i spinle (büyük/hızlı) etiketleme; öğren+ölçek. (Kavramsal.)

**Teslim:** genlik-transfer R²; tatlı-nokta genlik; sekstüpol-homojenlik eğrisi.
**Risk:** sekstüpol simetrik kanalı da bozar; movers temiz orbit-kör desen kuramaz.
**Altyapı:** `amp_scan.py` (genişlet), `perquad_orbit.py`, `integrator.cpp`
(`sextSwitch`, `quad_dG` mevcut — DEĞİŞTİRME).

### Plan 4 — Analitik Berry fonksiyoneli (Thomas-BMT'den türetme)

**Amaç:** COD→sahte-EDM fonksiyonelini Thomas-BMT spin denkleminden **kapalı-form**
türetip öğrenilen haritayı doğrulamak/değiştirmek (veri gerektirmeyen "altın standart").

- **4a — Türetme.** Kapalı yörünge üzerindeki parçacık için tur-başına geometrik fazı
  lider (bilineer) mertebede x_CO(s), y_CO(s) ve latis K(s) cinsinden yaz. Beklenen
  form (orbit_ileri §5): ardışık quad'ların **yönlü-alanı** Σ(x_i y_{i+1} − x_{i+1} y_i)
  tipi, yerel alan-ağırlıklı ∮ W(s)·(faz yoğunluğu) ds.
- **4b — Doğrulama.** Analitik fonksiyoneli 48-quad COD'a indirgeyip ensemble'daki
  C++ sahte-EDM ile karşılaştır (R²). Yüksekse → fonksiyonel doğrulandı; öğrenilen
  haritayı **açıklar** (neden simetrik kanal R²→0.77, antisim neden doyar).
- **4c — Kullanım.** Analitik fonksiyoneli (i) doğrudan ileri-harita olarak (sıfır
  eğitim verisi) null'lamada kullan; (ii) Plan 2b'de fizik-bilgili öznitelik olarak.

**Teslim:** kapalı-form fonksiyonel; C++ ile doğrulama R²; analitik-harita null'lama.
**Risk:** türetme zor; bilineer form yetersiz (yüksek-mertebe gerekli).
**Altyapı:** kalem-kâğıt + sembolik; doğrulama `berry_data/` ensemble'ı ile ucuz.

### Plan 5 — Spin-modülasyon + gradient descent (kullanıcı fikri, 2026-06) ⭐

**Fikir:** Quad'lar modüle edildiğinde **sahte-EDM sinyali de modüle olur.** Bunu
kullan: her düzeltme-knob'unu (quad'a bitişik dipol) modüle et, **sahte-EDM'in
tepkisini spinle ölç** → böylece $\partial f/\partial(\text{knob})$ **gradyanını
DOĞRUDAN ölçersin** → gradient descent ile sahte-EDM'i null'la. Düzeltme dipolleri:
$B_{\max}=g\cdot d = 0.2\,\text{T/m}\times 200\,\mu\text{m} = \pm40\,\mu\text{T}$
(≈±200 μm eşdeğer sapma; tipik misalignment/COD ~10–100 μm → **menzil yeterli**).

**Neden ÖNEMLİ (Kol B'nin duvarını aşar):**
- **Spin, simetrik modu GÖRÜR.** Sahte-EDM'i süren şey simetrik misalignment'tır;
  dolayısıyla quad-modülasyonuna **spin tepkisi simetrik moda duyarlıdır** — orbit
  k-mod'un (ΔR) simetriğe kör olduğu yerde spin k-mod görür. (Orbit no-go'yu atlar.)
- **Gradyanlar ÖLÇÜLÜR, öngörülmez.** Kol B'nin çöküş sebebi öğrenilmiş haritanın
  hatasıydı (§6.10-6.11, model-istismarı). Burada gradyan **gerçek** (spinle ölçülür)
  → *harita-doğruluğu darboğazı YOK*, kör-nokta yok. Bu, Kol A (spin) ailesinden ama
  **sistematik/gradyan-tabanlı** (kör spin-trim'in verimli hâli, §14.6).
- **Yakınsama:** $f$ knob'lara ~kuadratiktir (∝ orbit çarpımı); gerçek gradyanla
  descent/Gauss-Newton birkaç adımda gerçek null'a iner (menzil ve polarimetre
  gürültüsüyle sınırlı).

**Maliyet/dezavantaj:** Spin ölçümü yavaş (polarimetre istatistiği); her gradyan
$N_{\text{knob}}$ ölçüm ister. Kol A gibi spin gerektirir (orbit-tarafı ucuz çözüm
değil) — ama *çalışması beklenen* taraf bu.

**Test SONUCU (KOŞULDU → POZİTİF, `akilli_duzeltme.md §6.12`):** en-kötü orbit-kör
simetrik makine (COD 1.6 μm); 12 orbit-görünür knob; gerçek(C++) gradyan Gauss-Newton.
**orbit-null işe yaramadı (kazanç 1.0×, orbit-kör); SPİN-descent 2 adımda hedefin
altına indi (8×→~0, kazanç 726×).** Fark: spin simetriği görür + gradyan gerçek →
harita-hatası/model-istismarı YOK. Bu, tüm arc'ı doğrular: orbit-tarafı tıkalı,
spin-tarafı (Kol A/Plan 5) çalışır. Caveat: gürültüsüz gradyan (gerçekte polarimetre
gürültüsü/süre), tek makine — daha çok makine + gerçekçi gürültü sınanmalı.

---

### Önceliklendirme (güncel)

| Plan | Kazanç | Risk | Maliyet | Durum/Sıra |
|------|--------|------|---------|-----------|
| **2c** kapalı-döngü null | Kol B pratik geçerliliği | — | orta | ✗ KOŞULDU-negatif |
| **6.11** güvenli-opt | model-istismarını dizginle | — | orta | ⚠ KOŞULDU-kısmi |
| **5** spin-modülasyon descent | spin simetriği görür → duvarı aşar (Kol A verimli) | orta | orta-yüksek (spin) | **✓ KOŞULDU-POZİTİF (726×)** |
| **4** analitik fonksiyonel | Kol B'yi kurtarır (veri-bağımsız harita) | yüksek | düşük (analitik) | **2** |
| **3c** sekstüpol-homojenlik | büyük-genlik fikrini de-riske | düşük | düşük | 3 |
| **3a/3b/3d** kalibrasyon | gerçek-makine yolu | orta | orta | 4 |
