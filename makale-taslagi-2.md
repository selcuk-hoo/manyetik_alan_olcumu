# pEDM Halkasında Quad Hizalama Hatalarının BPM Tabanlı Ölçümü: Tepki-Matrisi Yöntemlerinin Sistematik Değerlendirmesi

---

## Özet

Proton EDM (pEDM) deneyinin alternating-gradient versiyonunda quad
hizalama hatalarının sürekli izlenmesi zorunludur: Omarov vd. [Omarov 2021,
PRD] bu hataların spin koherans zaman ölçeğinde $\sim 10\,\mu\text{m}$
seviyesinde bilinmesi gerektiğini göstermiştir. Bu çalışmada, kalibrasyon
referansına dayalı bir **online drift izleme yöntemi** öneriyoruz ve bu
yöntemin 48-quadrupole'lü pEDM FODO örgüsündeki başarısını sistematik
simülasyon testleriyle doğruluyoruz. Yöntem, kalibrasyon anındaki BPM
okumasını referans alarak $\widehat{\delta q}(t) = R^{-1}(\mathbf{y}(t)
-\mathbf{y}_0)$ ile hizalama driftini tahmin eder; gerçekçi BPM gürültü
ve ofset seviyelerinde 6-7 μm RMS hassasiyet elde edilir. Bu başarının
fiziksel temeli olarak, lineer iki-ölçümlü tam-ofset-iptal eden estimator
sınıfında $\|\Delta R^{-1}\|\sim\|R^{-1}\|/\varepsilon$ yapısal alt
sınırı türetilir; bu sınır k-modülasyon ruhundaki $\Delta R^{-1}$
yaklaşımlarının pEDM koşullarında ($\varepsilon\approx 0.02$, BPM gürültüsü
$\sim 1\,\mu$m) hedef hassasiyete ulaşamayacağını ($10^3\,\mu$m mertebesi)
açıklar. Açık uç: örgü modeli hatalarının (β beating, tune kayması) drift
modu performansına etkisi, yöntemin pratikte kullanılabilirliğinin
belirleyicisidir ve devam eden çalışmalarda ölçülmektedir.

---

## 1. Giriş

### 1.1 Bağlam

Proton EDM deneyinin orijinal önerileri tamamen elektrostatik ve zayıf
odaklamalı halka tasarımına dayanıyordu [pEDM proposal]. Bu tasarımda
birincil sistematik, ortalama vertikal manyetik alandı. Sonraki nesil
tasarımda hüzme dinamiğini iyileştirmek ve hızlandırıcının kararlılığını
artırmak için manyetik quadrupole'larla **alternating-gradient (FODO)**
odaklamasına geçildi. Bu değişiklikle sistematik öncelikler de değişti:
ortalama manyetik alan hâlâ önemli, ancak baskın sistematik artık **quad
hizalama hatalarıdır**.

### 1.2 Hizalama hatalarının fiziği

Hizalanmamış bir quadrupole, üzerinden geçen yüklü parçacığa Lorentz kuvveti
verir. Bu kuvvetin dikey bileşeninin halka boyunca ortalaması, EDM sinyalini
taklit eden sahte bir vertikal alana karşılık gelir.

Omarov vd. [Omarov 2021, PRD] manyetik-quad'lı pEDM tasarımında bu sahte
alanın sistematik bütçesini ayrıntılı türettiler. Hedef EDM hassasiyetini
($\sim 10^{-29}\,e\cdot\text{cm}$) bozmadan tutmak için, hizalama hataları
spin koherans zaman ölçeğinde **10 μm RMS seviyesinde bilinmek zorundadır**.
Bu sayı bu çalışmanın hedef hassasiyetini doğrudan belirler.

Halkada 48 manyetik quadrupole vardır. Her biri yatay (`dx`) ve dikey (`dy`)
eksende bağımsız kayabilir; toplam 96 bilinmeyen. Ölçüm aracı: 48 BPM (Beam
Position Monitor) çifti.

### 1.3 Konvansiyonel k-modülasyonunun zorlukları

Sabit-β depolama halkalarında klasik k-modülasyonu [Lee 2004], tek bir
quad'ın gradient'ini periyodik olarak modüle eder ve o noktadaki hüzme
tepkisinden hizalama hatasını çıkarır. Yöntem üç varsayıma dayanır:

1. β fonksiyonu yerel olarak bilinir ve sabittir,
2. Modülasyon frekansı tune'lardan iyi ayrılır,
3. Tek-quad modülasyonu örgünün geri kalanını etkilemez.

pEDM-FODO örgüsü bu üç varsayımı **birden** zorlar: 48 quad'ın her birinde
β farklıdır, halkayı fizik run'ı boyunca onlarca farklı frekansta modüle
etmek sistematik temizliğini bozar, ve manyetik quad'lar elektrostatik
deflektörlerle eşli tasarımdadır. Bu yüzden yerel k-modülasyon yerine
**küresel tepki-matrisi tabanlı yaklaşımları** araştırıyoruz.

### 1.4 Çalışmanın katkısı

Bu çalışmanın merkezi katkısı, pEDM için **sürekli online çalışabilen bir
hizalama drift izleme yöntemi** önermek ve altı sistematik simülasyon
testiyle bu yöntemin 10 μm hedefine ulaştığını doğrulamaktır. Yöntem iki
katmanlı bir operasyonel mimari içinde çalışır: yavaş bir mutlak kalibrasyon
katmanı (LOCO/BBA, saatlik-günlük) ile hızlı bir online drift katmanı.

Bu yöntemin neden çalıştığı ve rakip yaklaşımların neden yapısal sınırlara
takıldığı, aynı simülasyon altyapısı üzerinde nicelenmiştir. Bu karşılaştırma
ayrı bir katkı olarak sunulmaktadır: lineer iki-ölçümlü tam-ofset-iptal eden
estimator sınıfı için bir alt sınır türetilmiş (§2.4) ve k-modülasyon ruhlu
$\Delta R^{-1}$ yaklaşımlarının bu sınır nedeniyle hedef hassasiyete
ulaşamayacağı sayısal olarak gösterilmiştir.

---

## 2. Yöntem

### 2.1 Lineer model

BPM okumalarıyla quad hizalama hataları arasındaki ilişki lineer rejimde:

$$
\mathbf{y} = R\,\Delta q + \mathbf{b} + \boldsymbol{\eta},
$$

$\Delta q \in \mathbb{R}^{48}$ misalignment vektörü, $R \in \mathbb{R}^{48\times 48}$
tepki matrisi, $\mathbf{b}$ BPM elektronik ofseti, $\boldsymbol{\eta}$
gürültü. Yatay-dikey kuplajı ihmal edilebilir; problemi iki bağımsız 48×48
sisteme ayırıyoruz.

### 2.2 Tepki matrisinin analitik inşası

Periyodik FODO örgüsünde Courant-Snyder formalizmiyle:

$$
R_{ij} = \frac{\sqrt{\beta_i\beta_j}}{2\sin(\pi Q)}
         \cos\!\bigl(|\phi_i-\phi_j|-\pi Q\bigr) \cdot (KL)_j,
$$

burada $\beta_i, \phi_i$ Twiss parametreleri, $Q$ tune, $(KL)_j$ quad'ın
işaretli integral gücüdür. Yatay düzlemde elektrostatik ark deflektörlerinin
katkısı için $K_{x,\text{arc}}$ parametresi tune-eşleme ile kalibre edilir
(bkz. §3.3, ters-suç kontrolü). Dikey düzlemde Maxwell garantisiyle
$K_{y,\text{arc}}=0$.

### 2.3 Aday yöntemler

#### (i) Mutlak tek-gradient
$$\widehat{\Delta q} = R^{-1}\mathbf{y}.$$
$\kappa(R) \approx 160$ olduğu için gürültü iyi kontrol altında, ama
BPM ofseti $R^{-1}\mathbf{b}$ olarak tahmine sızar. Mutlak hizalama
bilgisini ofsete kurban eder.

#### (ii) İki-gradient $\Delta R^{-1}$ (klasik k-mod ruhu)
$g_2 = g_1(1+\varepsilon)$, $\varepsilon\approx 0.02$:
$$\widehat{\Delta q} = \Delta R^{-1}(\mathbf{y}_1 - \mathbf{y}_2),\quad
\Delta R = R_1 - R_2.$$
Ofset iptal olur. Kondisyon sayısı için: Courant-Snyder formülünde $R$,
$(KL)_j$'ye lineer bağlıdır ve $KL\propto g$. Dolayısıyla birinci
dereceden Taylor açılımı:
$$\Delta R = R(g_2) - R(g_1) = \varepsilon g_1\frac{\partial R}{\partial g}
+ O(\varepsilon^2) \approx \varepsilon R_1.$$
Bu ölçeklemeden $\kappa(\Delta R)\approx\kappa(R)/1 = \kappa(R)$ beklenirken,
$\varepsilon$ sadece büyüklüğü değil tekil değerlerin rölatif ayrışmasını
da etkiler. Sayısal olarak $\kappa(\Delta R)\approx 27\,000$ iken
$\kappa(R)\approx 160$ — yaklaşık $1/\varepsilon\approx 50$ oranı,
$\varepsilon$ taramasıyla Test 2'de doğrulanmaktadır.

#### (iii) $\Delta R^{-1}$ + düzenlileştirme
Tikhonov: $(\Delta R^\top \Delta R + \lambda I)^{-1}\Delta R^\top$.
TSVD: en büyük $k$ tekil değer dışındakileri sıfırla. Gürültü-bias
trade-off; en iyi durumda 50 μm RMS.

#### (iv) Drift modu (kalibrasyon-referans)
Tek gradient, iki zaman:
$$\widehat{\delta q}(t) = R^{-1}\bigl(\mathbf{y}(t)-\mathbf{y}_0\bigr).$$
Ofset zaman farkıyla iptal, $\kappa(R)\approx 160$ küçük gürültü büyütmesi,
**fakat** sadece değişim ölçer; mutlak hizalama dış kaynaktan (LOCO/BBA)
gelmek zorundadır.

### 2.4 Offset–noise duality teoremi

Bu bölümün amacı, aday yöntem (ii)'nin gürültü büyütmesinin algoritmik
değil yapısal olduğunu göstermek. Sonuç sınırlı bir estimator sınıfı için
geçerlidir.

**Estimator sınıfı $\mathcal{C}$.** Aşağıdaki dört koşulu sağlayan
estimator'lar:
1. *Lineerlik:* $\widehat{\Delta q} = A_1\mathbf{y}_1 + A_2\mathbf{y}_2$,
   $A_1, A_2$ sabit matrisler.
2. *İki-ölçüm:* iki ayrı ölçüm $\mathbf{y}_1=R_1\Delta q+\mathbf{b}+\eta_1$,
   $\mathbf{y}_2=R_2\Delta q+\mathbf{b}+\eta_2$ aynı $\Delta q$ ve aynı
   $\mathbf{b}$ ile.
3. *Ön-yargısızlık:* her $\Delta q$ için
   $\mathbb{E}[\widehat{\Delta q}]=\Delta q$.
4. *Tam ofset iptali:* her sabit $\mathbf{b}$ için tahmine sızmaz.

**İddia.** $\mathcal{C}$ içinde tek bir çözüm vardır:
$A_1=\Delta R^{-1}, A_2=-\Delta R^{-1}$.

**Türetiş.** Koşul 3: $A_1 R_1 + A_2 R_2 = I$. Koşul 4: $A_1 + A_2 = 0
\Rightarrow A_2 = -A_1$. Birleştirince $A_1(R_1-R_2) = I \Rightarrow
A_1 = \Delta R^{-1}$. ∎

**Sonuç.** $\mathcal{C}$ içindeki tek çözümün gürültü büyütme faktörü
$\|\Delta R^{-1}\|$. $\Delta R \approx \varepsilon R$ ölçeklemesiyle
yaklaşık $\|R^{-1}\|/\varepsilon$. Bu ölçekleme $\varepsilon\to 0$
sınırında patlar (§3.2'de SVD spektrumu ile sayısal olarak doğrulanır).

$\mathcal{C}$ dışında kalan yaklaşımlar (Tikhonov/TSVD, Bayesian,
Kalman, çok-epoch) bu sınıra uymak zorunda değildir ve farklı
trade-off'larla çalışır — bunlar §3.1-3.2'de incelenmektedir.

Drift modu (yöntem (iv)) da $\mathcal{C}$ dışındadır: iki ölçüm aynı anda
değil iki farklı zamanda alınır; estimator $\Delta q$'yu değil
$\delta q(t) = \Delta q(t) - \Delta q_0$'ı kestirir. Ofset zaman farkıyla
iptal olur ve $\kappa(R)\approx 160$ ile küçük gürültü büyütmesi elde
edilir. Bu sonuç Test 4 ve Test 6'da sayısal olarak doğrulanmaktadır.

---

## 3. Sayısal Deneyler

Aşağıdaki altı testin hepsi aynı simülasyon altyapısı üzerinde koşuldu:
simplektik integratör, gerçekçi BPM gürültü ve ofset modeli, opsiyonel
quad ve dipol tilt'leri. Altyapı ve veri akışının detayları Ek A'da.

### 3.1 Test 1 — Düzenlileştirme ham $\Delta R^{-1}$'i kurtarır mı?

**Yöntem.** Aynı veri üzerinde dört aday estimator: direct $R^{-1}$
(yöntem (i)), ham $\Delta R^{-1}$ (yöntem (ii)), Tikhonov (L-curve) ve
TSVD (oracle $k$, yöntem (iii)). BPM ofseti **yok**; bu test ofsetin
olmadığı ideal şartta noise floor ölçer.

**Sonuç.**

| Estimator | y-RMS | y-corr | Yorum |
|---|---|---|---|
| Direct $R^{-1}$ ortalama | 3.5 μm | 0.998 | Noise floor referansı (ofset varsayımı yok) |
| Ham $\Delta R^{-1}$ | 1865 μm | 0.085 | Yöntem (ii) — §2.4 sınırının nümerik gerçekleşmesi |
| Tikhonov | 53 μm | 0.348 | Yöntem (iii), bias-varyans dengesi |
| TSVD oracle ($k=3$) | 52 μm | 0.383 | Yöntem (iii), oracle üst-sınır |

**Önemli uyarı: Direct $R^{-1}$ operasyonel bir rakip değil.** BPM ofseti
sıfır olduğunda yöntem (i) yapısal olarak tartışılmaz ama gerçek halkada
$\mathbf{b}\neq 0$ olduğu için bu satır 200+ μm'e tırmanır (bkz. Test 4).
Tabloda yalnızca *noise floor referansı* olarak yer alır.

Düzenlileştirme ham (ii)'yi 35× iyileştiriyor ama korelasyon 0.998'den
0.35'e düşüyor — RMS azalıyor, biçim bozuluyor. Tikhonov/TSVD bu sınıfın
($\mathcal{C}$ dışında, bias kabul eden) en iyi performansını temsil
ediyor, hâlâ 10 μm hedefinin çok üstünde.

### 3.2 Test 2 — Estimator'ın uzaysal transfer fonksiyonu ve SVD spektrumu

**Yöntem.** Saf sinüsoidal misalignment paterni ($\Delta q_j = A\cos(2\pi k j/N)$)
girişine, estimator çıkışında geri kurtarılan oran. Mod-mod analiz. Ayrıca
$R$ ve $\Delta R$'nin singular value spektrumları yan yana çizilir.

**Sonuç (mod transferi).** Tikhonov/TSVD: yüksek $k$ modlarını ($k\geq 18$)
tamamen söndürüyor. 48 modun yalnızca 3-5'i geri kurtarılıyor; geri kalan
40+ mod sıfıra çekiliyor. Düzenlileştirmenin RMS azaltma maliyeti budur.

**Sonuç (SVD spektrumu).** $R$ ve $\Delta R$'nin singular değerleri yan
yana çizildiğinde §2.4'teki $\Delta R \approx \varepsilon R$ ölçeklemesi
doğrulanır: $\Delta R$ spektrumu $R$ spektrumunun yaklaşık $\varepsilon$
katı düşey kayma gösterir. $\kappa(\Delta R)/\kappa(R) \approx 1/\varepsilon
\approx 50$ ilişkisi $\varepsilon \in [0.005, 0.05]$ taramasında geçerli
kalıyor. Bu sayısal doğrulama §2.4 alt-sınır iddiasının ölçek tarafını
destekler.

### 3.3 Test 3 — Yatay model ters-suç kontrolü

**Yöntem.** $K_{x,\text{arc}}$ kalibrasyonu simülasyondan alınıyor; bu klasik
ters-suç. $K_{x,\text{arc}}(1+\delta)$ ile $\delta \in [-10\%,+10\%]$
taraması yapıldı.

**Sonuç.** $\pm 10\%$ aralığında yatay RMS 3.48-4.01 μm (0.5 μm değişim).
Dikey düzlem sabit (Maxwell). Yorum: gerçek halkada LOCO ile sağlanan
$<1\%$ doğruluk için yöntem **operasyonel olarak ters-suçtan bağımsız**.

### 3.4 Test 4 — Drift modunun kalibrasyon-referansla gösterimi

**Senaryo.** $t=0$: 100 μm RMS hizalama + 50 μm RMS BPM ofseti kaydedildi.
$t=1..10$: hizalama 10 μm RMS yavaşça kayıyor, ofset sabit.

**Sonuç.**

| Yöntem | RMS hata |
|---|---|
| Mutlak $R^{-1}\mathbf{y}(t)$ | 170-200 μm (ofset baskın) |
| Drift $R^{-1}(\mathbf{y}-\mathbf{y}_0)$ | **6.5 μm** |

Drift modu ofseti zaman farkıyla iptal ediyor; geriye yalnızca
$\sqrt{2}\sigma_n\|R^{-1}\|$ gürültü kalıyor.

### 3.5 Test 5 — BPM ofseti zamanla kayarsa?

**Senaryo.** Test 4 ofsetin sabit olduğunu varsaydı. Eğer BPM ofseti
~μm/epoch kayarsa, drift modu (A) bozulur. Alternatif olarak her epoch
yeniden iki-gradient ölçüm + 30-epoch ortalama (B).

**Sonuç.** Geçiş ~2 μm/epoch'ta. Termal-stabil bir hızlandırıcı salonunda
modern BPM elektroniği bu hızın çok altındadır (~0.1 μm/°C). A baskın,
B yedek.

### 3.6 Test 6 — Üç yöntemin adil yan yana karşılaştırması

**Yöntem.** Aynı veri, aynı hata bütçesi (50 μm ofset, 1 μm gürültü,
10 μm drift, 0.2 mrad tilt'ler) altında üç estimator:
- A: analitik $\Delta R^{-1}$ (per-epoch absolute → epoch farkı)
- B: drift modu (önerilen)
- C: sayısal $\Delta R^{-1}$ (48 finite-difference sim ile inşa, LOCO benzeri)

**Sonuç.**

| Estimator | y-RMS | y-corr | x-RMS | x-corr |
|---|---|---|---|---|
| A: analitik $\Delta R$ | 3282 μm | -0.02 | 3756 μm | 0.09 |
| **B: drift modu** | **6.25 μm** | **0.85** | **7.18 μm** | **0.85** |
| C: sayısal $\Delta R$ | 980 μm | -0.02 | 1357 μm | -0.11 |

$\kappa(\Delta R_{\text{analitik}}) = \kappa(\Delta R_{\text{sayısal}})
\approx 27\,000$. Model uyumsuzluğu $\|R_{\text{an}}-R_{\text{num}}\|
/\|R_{\text{num}}\|$ = %2.2 (y), %6.9 (x).

**Yorum.** Sayısal R, $\Delta R$ yaklaşımını kurtarmıyor —
$\kappa(\Delta R)$ R'nin nasıl inşa edildiğine değil, $\varepsilon$ ve
örgü fiziğine bağlı. Drift modu, yapısal olarak farklı bir matematiksel
problem çözdüğü için (mutlak değil, değişim) bu sınırın dışında.

### 3.7 Test özeti

| Test | Soru | Ana sayı |
|---|---|---|
| 1 | Düzenlileştirme $\Delta R$'yi kurtarır mı? | 1865→52 μm, ama direct 3.5 μm |
| 2 | Düzenlileştirme nasıl çuvallıyor? | 48 modun 43-45'i siliniyor |
| 3 | Yatay modelde ters-suç var mı? | $\pm 10\%$ → <0.5 μm değişim |
| 4 | Drift modu ofseti tolere eder mi? | Mutlak 180 μm → Drift 6 μm |
| 5 | BPM ofseti kayarsa? | A: <2 μm/epoch'a kadar üstün |
| 6 | Aday yöntemler aynı senaryoda? | $\Delta R$: 1000-3700 μm, Drift: 6-7 μm |

---

## 4. Önerilen İşletme Modu

### 4.1 İki katmanlı yapı

Tek bir yöntem her ihtiyacı karşılamıyor. Önerimiz:

**Yavaş mutlak katman (saatlik-günlük).** LOCO + BBA + survey, mutlak
hizalama $\Delta q_0$, BPM ofseti $\mathbf{b}_0$ ve örgü modeli ($\beta$,
$\phi$, $Q$, dolayısıyla $R$) için. Bu, hızlandırıcı operasyonunda zaten
varolan standart bir prosedür.

**Hızlı drift katmanı (sürekli, saniye-dakika).** Bu çalışmanın katkısı:
$\widehat{\delta q}(t) = R^{-1}(\mathbf{y}(t)-\mathbf{y}_0)$. Sürekli
çalışır, fizik veri toplamasını bozmaz, kalibrasyondan beri hizalama
değişimini takip eder.

İki katman tamamlayıcı: yavaş katman mutlak referansı verir, hızlı katman
o referansa göre değişimi izler.

### 4.2 Üç işletim modu

Bu mimari içinde üç pratik mod var:

- **Mod 1 (önerilen).** Fizik run'ı boyunca sadece drift modunu çalıştır.
- **Mod 2.** Periyodik olarak iki-gradient kontrolü yap, drift modunun
  tutarlılığını doğrula.
- **Mod 3.** Fizik dışı pencerelerde tam k-mod kalibrasyonu, $R$ ve
  $\mathbf{b}_0$ güncellemesi.

### 4.3 Diğer yaklaşımlarla ilişki

Aynı problem yapay sinir ağı tabanlı bir paralel çalışmada da ele
alınmaktadır; sonuçlarımız bu yaklaşımla karşılaştırılabilir bir referans
çerçevesi oluşturur. §2.4'teki alt sınırın kapsamı dışında kalan
estimator'lar (örn. çok-epoch'lu, lineer olmayan, sparse-prior'lı) bu
sınırın altına inebilir; karşılaştırmalı analiz başka bir çalışmaya
bırakılmıştır.

---

## 5. Tartışma ve Sonuçlar

Bu çalışmada, pEDM alternating-gradient halkasında quad hizalama driftini
sürekli izlemek için kalibrasyon-referans yöntemi önerildi ve bu yöntemin
gerçekçi BPM hata bütçesi altında (50 μm ofset, 1 μm gürültü, 0.2 mrad
tilt'ler) 6-7 μm RMS hassasiyetle çalıştığı altı sistematik test ile
doğrulandı.

Bu başarı iki katmanlı bir operasyonel mimari üzerine oturmaktadır: yavaş
bir mutlak kalibrasyon katmanı (LOCO/BBA, fizik-dışı pencereler) referans
noktasını ve $R$ matrisini sağlar; hızlı drift katmanı fizik run'ı boyunca
sürekli $\widehat{\delta q}(t) = R^{-1}(\mathbf{y}(t)-\mathbf{y}_0)$ ile
çalışır.

Bu başarının neden mümkün olduğu da gösterilmiştir: lineer iki-ölçümlü
tam-ofset-iptal eden estimator sınıfı $\mathcal{C}$ için
$\|\Delta R^{-1}\|\sim\|R^{-1}\|/\varepsilon$ yapısal alt sınırı
türetilmiştir. k-modülasyon ruhlu $\Delta R^{-1}$ bu sınıfa girmekte ve
$R$ analitik ya da sayısal inşa edilsin, gerçekçi BPM gürültüsü altında
hedef hassasiyete ulaşamamaktadır ($10^3\,\mu$m mertebesi). Drift modu
problemi yeniden tanımlayarak bu sınırın dışına çıkar.

Çalışmanın sınırları ve gelecek işler:

- **Örgü modeli hatasının drift moduna etkisi (kritik).** Drift modunun
  6-7 μm hassasiyeti tamamen $R^{-1}$'in doğruluğuna bağlıdır. $\beta$
  beating, tune kayması, faz ilerleme hatası gibi gerçekçi örgü
  bozulmalarının drift mode performansını ne kadar bozduğu, yöntemin
  pratik kullanılabilirliğinin belirleyicisidir. Test 8 (sürmekte).
- **Tilt etkilerinin sistematik taraması.** Test 6'da tilt'ler 0.2 mrad
  sabitti; bu seviyenin altında/üstünde drift mode'un kırılma noktası
  belirlenmeli (Test 7, sürmekte).
- **Hata kaynaklarının ayrıştırılması.** Test 6 kontrol senaryolarıyla
  (BPM gürültü, ofset, tilt birer birer kapatılarak) drift mode'un 6 μm
  hatasının hangi bileşenden geldiği nicelenmeli.
- **Lineer modelin ötesi.** Sonuçlar Eş.~(1) lineer modeline dayanır.
  Gerçek halkada BPM gain hataları, roll, kuplaj, sextupole feed-down,
  fringe alanlar, manyetik hysteresis ve akım dalgalanması gibi etkiler
  vardır. Bu etkilerin sistematik incelenmesi gerekir.
- **BPM ofset uzun-vadeli kararlılığı.** Drift mode'un üst sınır
  varsayımı $\mathbf{b}(t)\approx\mathbf{b}_0$ olduğundan, pEDM BPM
  donanımının saatler-günler zaman ölçeğinde ofset stabilitesi deneysel
  olarak karakterize edilmelidir.

---

## Kaynaklar

*(Taslak — referans listesi tamamlanacak.)*

- Z. Omarov vd., "Comprehensive symmetric-hybrid ring design for a
  proton EDM experiment at below $10^{-29}\,e\cdot\text{cm}$,"
  *Phys. Rev. D* (2021). [10 μm hizalama bütçesi]
- pEDM Collaboration, "Storage Ring Proton Electric Dipole Moment
  Experiment Proposal," BNL (2011).
- S. Y. Lee, *Accelerator Physics*, 3rd ed., World Scientific (2011).
  [Standart k-modülasyon ve tepki matrisi formalizmi]
- J. Safranek, "Experimental determination of storage ring optics using
  orbit response measurements," *NIM A* 388 (1997) 27. [LOCO]
- P. Castro vd., BBA literatürü.

---

## Ek A: Sayısal simülasyon altyapısı

Tüm sonuçlar Python tabanlı bir simülasyon altyapısında üretildi.
Parçacık takibi dördüncü-mertebe Runge-Kutta (RK4) entegratörüyle
yapıldı. Analitik tepki matrisi Courant-Snyder formalizmiyle inşa edildi.
Hızlandırıcı ve test parametreleri ayrı JSON dosyalarında tutulmaktadır.
Test betikleri ve ham veriler istek üzerine paylaşılmaktadır.
