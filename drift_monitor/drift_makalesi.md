# Dondurulmuş-spin proton EDM halkasında kapalı-yörünge tabanlı hizalama drift izlemesi: performans ve gözlenebilirlik sınırları

> **Durum:** Taslak (v0.1, 2026-06-19). Dil: Türkçe. Bu belge `makale-taslagi-2.md`
> ve `drift_monitor/` altındaki test sonuçlarından konsolide edilmiştir.
> Başlık geçicidir.

---

## Özet

Proton EDM (pEDM) deneyinin alternating-gradient (AG) versiyonunda, manyetik
kuadrupol hizalama hataları baskın sistematik kaynağıdır ve spin koherans
zaman ölçeğinde $\sim 10\,\mu\text{m}$ seviyesinde kontrol altında tutulmalıdır
[Omarov 2022]. Bu çalışmada, kalibrasyon anına göreli bir **kapalı-yörünge
tabanlı online hizalama drift izleme yöntemi** öneriyor ve sistematik
simülasyon testleriyle hem yeteneklerini hem de temel sınırını niceliyoruz.
Yaklaşım, problemi bilinçli olarak yeniden tanımlar: amaç **mutlak hizalama
rekonstrüksiyonu** ($\Delta q = R^{-1}\mathbf{y}$) değil, **göreli hizalama
kararlılığı izlemesi**dir. Yöntem, kalibrasyon anındaki BPM okumasını referans
alarak $\widehat{\delta q}(t) = R^{-1}\bigl(\mathbf{y}(t)-\mathbf{y}_0\bigr)$
ile hizalama driftini kestirir; sabit BPM elektronik ofseti zaman farkında
iptal olur. Gerçekçi hata bütçesinde (50 μm RMS ofset, 1 μm gürültü) yöntem
6–7 μm RMS hassasiyetle çalışır (mutlak rekonstrüksiyona göre $\sim 29\times$
iyileşme) ve LOCO kalitesinde (%1) β-beating altında dahi hedefin altında
kalır. Yöntemin temel sınırını singüler-değer (SVD) per-mod analiziyle ortaya
koyuyoruz: tepki matrisinin en kötü koşullanmış modları (%96 oranında
**simetrik** alt-uzay; hücre içi QF/QD aynı yönde) en büyük koşulluluk-kaynaklı
duyarlılık dezavantajına (en kötü modda en iyiye göre $\sim 193$ kat) maruz
kalır. Bu simetrik alt-uzay, sahte EDM'yi süren $dx\cdot dy$ geometrik-faz
kanalına ($\propto\sigma^2$) en güçlü katkıyı veren bileşenlerle örtüşür
[Omarov 2022, Fig. 9a]; dolayısıyla kapalı-yörünge izleme antisimetrik
hizalama driftini hassasça takip ederken, sahte-EDM-kritik simetrik kanalı bu
ölçüm konfigürasyonu altında gürültü sınırında bırakır. Bu sonuç,
kapalı-yörünge tabanlı izlemenin yalnızca belirli hizalama modlarına duyarlı
olduğunu ve gözlenebilirlik sınırının tepki matrisinin mod yapısından
kaynaklandığını gösterir; yöntemin kesin tanımlı bir **geçerlilik alanı** ve
kesin tanımlı bir **kör noktası** vardır.

---

## 1. Giriş

### 1.1 Bağlam ve sistematik önceliği

Proton EDM deneyinin ilk önerileri tamamen elektrostatik, zayıf-odaklamalı
halka tasarımına dayanıyordu; baskın sistematik ortalama dikey manyetik
alandı. Demet dinamiğini ve kararlılığı iyileştirmek için sonraki nesil
simetrik-hibrit tasarım, manyetik kuadrupollerle **alternating-gradient
(FODO)** odaklamaya geçti [Omarov 2022]. Bu geçişle sistematik öncelikler de
değişti: baskın sistematik artık **kuadrupol hizalama hatalarıdır**.

Hizalanmamış bir kuadrupol, üzerinden geçen demete net bir kuvvet uygular;
bu kuvvetin halka boyunca uygun bileşeni EDM sinyalini taklit eden sahte bir
dikey spin presesyonu üretir. Omarov vd. [Omarov 2022], manyetik-quad'lı pEDM
tasarımında bu sahte alanın sistematik bütçesini ayrıntılı türetmiş ve hedef
hassasiyeti ($d_p < 10^{-29}\,e\cdot\text{cm}$, eşdeğer $dS_y/dt < 1$ nrad/s)
korumak için hizalama hatalarının spin koherans zaman ölçeğinde **10 μm RMS
seviyesinde bilinmesi/kontrol edilmesi** gerektiğini göstermiştir. Bu sayı,
bu çalışmanın hedef hassasiyetini doğrudan belirler.

Halkada $2\times 24 = 48$ manyetik kuadrupol vardır; her biri yatay ($dx$) ve
dikey ($dy$) eksende bağımsız kayabilir. Ölçüm aracı 48 BPM çiftidir. Temel
zorluk şudur: BPM elektronik ofsetleri ($\sim 100\,\mu$m) ölçülmek istenen
hizalama sinyaliyle ($\sim 10\,\mu$m) **aynı büyüklüktedir**; mutlak bir
kapalı-yörünge ölçümü bu ofsete boğulur.

### 1.2 Alanın benimsediği hizalama stratejisi

Omarov vd.'nin simetrik-hibrit tasarımı, hizalamayı mekanik olarak μm
seviyesinde dayatmak yerine sistematiği hizalamaya **duyarsız** kılan üç
katmanlı bir savunmaya dayanır: (i) pasif kafes simetrisi ($\sigma^2$
bastırma), (ii) CW/CCW karşı-dönen demetler + kuadrupol polarite çevirmeyle
aktif iptal, (iii) spin-tabanlı hizalama (SBA), "yükselt-sonra-söndür"
numarasıyla. Bu çerçevede asıl hassas hizalama ölçümü tek-demet kapalı
yörüngesinden değil, karşı-dönen demet ayrımının ($\Delta y$) SQUID-tabanlı
BPM'lerle okunmasından ve spinin kalibrasyon probu olarak kullanılmasından
gelir; mekanik tolerans $\sim 100\,\mu$m'ye gevşetilir.

Dolayısıyla bu makalenin amacı klasik kapalı-yörünge/k-modülasyon yöntemini
bu yerleşik stratejinin yerine koymak **değildir**. Aksine, mevcut BPM
altyapısıyla sürekli, ucuz ve fizik-veri-toplamayı bozmadan çalışabilen bir
**tamamlayıcı online drift izleyici**nin neyi başarabileceğini ve nerede
yapısal olarak duracağını nicelemektir.

### 1.3 Çalışmanın katkısı

1. **Pozitif sonuç (yöntem).** Kalibrasyon-referanslı drift modu
   $\widehat{\delta q}(t)=R^{-1}(\mathbf{y}(t)-\mathbf{y}_0)$, sabit BPM
   ofsetini yapısal olarak iptal eder ve iyi-koşullu $R$ ($\kappa\approx 193$)
   kullanır; gerçekçi hata bütçesinde 6–7 μm RMS sağlar (§3).
2. **Temel sınır (özgün omurga — makalenin merkezi tezi).** Per-mod SVD
   analiziyle, kapalı-yörünge izlemenin kör noktasının **simetrik alt-uzay**
   olduğu ve bunun sahte-EDM'yi süren $dx\cdot dy$ kanalıyla çakıştığı
   gösterilir (§4). Bu, kapalı-yörünge yöntemlerinin sahte-EDM-kritik bilgiyi
   bu ölçüm konfigürasyonunda neden sağlayamadığının nicel bir ifadesidir;
   makalenin asıl sonucu ölçülen RMS sayısı değil, bu gözlenebilirlik
   ayrımıdır.
3. **Yapısal arka plan (destekleyici).** İki-ölçümlü, tam-ofset-iptal eden
   lineer estimatör sınıfı için bir tekillik önermesi verilir
   ($\|\Delta R^{-1}\|\sim\|R^{-1}\|/\varepsilon$); bu, "iki gradient farkıyla
   ofseti iptal et" gibi bariz alternatiflerin neden yapısal olarak
   kötü-koşullu olduğunu, drift modunun (iki *zaman*) ise neden bu sınırın
   dışında kaldığını açıklar (§2.4).

---

## 2. Model ve Yöntem

### 2.1 Lineer model

Lineer rejimde BPM okumaları ile hizalama hataları arasındaki ilişki:

$$
\mathbf{y} = R\,\Delta q + \mathbf{b} + \boldsymbol{\eta},
$$

burada $\Delta q\in\mathbb{R}^{48}$ misalignment vektörü, $R\in\mathbb{R}^{48\times48}$
tepki matrisi, $\mathbf{b}$ BPM elektronik ofseti, $\boldsymbol{\eta}$
ölçüm gürültüsüdür. Yatay-dikey kuplaj ihmal edilebilir kabul edilir; problem
iki bağımsız $48\times48$ sisteme ayrılır.

### 2.2 Tepki matrisinin analitik inşası

Periyodik FODO örgüsünde, Courant-Snyder formalizmiyle kapalı-yörünge tepki
matrisi:

$$
R_{ij} = \frac{\sqrt{\beta_i\beta_j}}{2\sin(\pi Q)}\,
         \cos\!\bigl(|\phi_i-\phi_j|-\pi Q\bigr)\,(KL)_j,
$$

$\beta_i,\phi_i$ her quad girişindeki Twiss parametreleri, $Q$ tune, $(KL)_j$
quad'ın işaretli integral gücüdür (QF/QD tip işareti ve düzlem işareti içinde
saklanır). Gürültü için bu çalışmada $\sigma_\eta = 1\,\mu$m alınır; bu, tek
BPM okuma gürültüsü değil, çok-tur ortalaması sonrası etkin yörünge belirleme
belirsizliğini temsil eden bir **model parametresidir**. Yatay düzlemde elektrostatik ark deflektörlerinin katkısı için
$K_{x,\text{arc}}$ tune-eşlemeyle kalibre edilir (§3.3, ters-suç kontrolü);
dikey düzlemde Maxwell garantisiyle ($n=1\Rightarrow E_z=0$) $K_{y,\text{arc}}=0$.
Bu inşa simülasyon izleyicisinden bağımsızdır ve `drift_monitor/fodo_lattice.py`
içinde uygulanmıştır. Varsayılan örgüde $\kappa(R)\approx 193$,
$\sigma_{\max}\approx 28.4$, $\sigma_{\min}\approx 0.147$.

### 2.3 Aday estimatörler

Aday yöntemler iki sınıfa ayrılır: ofseti tahmin etmeye/iptal etmeye çalışan
**mutlak rekonstrüksiyon** yaklaşımları (i–iii) ve ofseti bir nuisance
(rahatsızlık) parametresi gibi eleyip yalnız değişimi kestiren **göreli
kararlılık izlemesi** (iv).

**(i) Mutlak tek-gradient.** $\widehat{\Delta q}=R^{-1}\mathbf{y}$. $\kappa(R)$
küçük olduğundan gürültü iyi kontrol altındadır, ancak BPM ofseti
$R^{-1}\mathbf{b}$ olarak tahmine doğrudan sızar. Mutlak hizalama bilgisini
ofsete kurban eder.

**(ii) İki-gradient $\Delta R^{-1}$ (ofset-iptal).** $g_2=g_1(1+\varepsilon)$
ile iki ayrı gradient ayarında ölçüm alıp
$\widehat{\Delta q}=\Delta R^{-1}(\mathbf{y}_1-\mathbf{y}_2)$,
$\Delta R=R_1-R_2$. Ofset iptal olur, fakat $R\propto(KL)\propto g$ olduğundan
küçük $\varepsilon$ için $\Delta R\approx\varepsilon\,(g\,\partial R/\partial g)$
olur; bu, ofset-iptal eden estimatörün **gürültü büyütmesini**
$\|\Delta R^{-1}\| = 1/\sigma_{\min}(\Delta R)\propto 1/\varepsilon$ olarak
patlatır (asıl sorun budur). Dikkat: $1/\varepsilon$ ile ölçeklenen kondisyon
sayısı $\kappa(\Delta R)$ **değil** — o, türev matrisi $g\,\partial R/\partial g$'nin
koşulluluğu olup $\varepsilon$'dan kabaca bağımsızdır ($\sim 10^4$, $\kappa(R)\approx 193$'ün
~2 mertebe üstünde). Pratikte ($\varepsilon\approx 0.02$) gürültü büyütmesi
$\|\Delta R^{-1}\|\sim 10^4$ mertebesine çıkar (Şekil 1, Şekil 6).

**(iii) $\Delta R^{-1}$ + düzenlileştirme.** Tikhonov / TSVD. Gürültü-bias
dengesi; en iyi durumda dahi $\sim 50\,\mu$m (§3.1).

**(iv) Drift modu (kalibrasyon-referans) — önerilen.** Tek gradient, iki
*zaman*:

$$
\widehat{\delta q}(t)=R^{-1}\bigl(\mathbf{y}(t)-\mathbf{y}_0\bigr).
$$

Sabit ofset zaman farkında iptal olur; geriye yalnızca
$\sqrt{2}\,\sigma_\eta\|R^{-1}\|$ gürültüsü kalır. Yöntem mutlak hizalamayı
değil, kalibrasyondan beri **değişimi** kestirir; mutlak referans dış
kaynaktan (LOCO/BBA) gelir (§5).

### 2.4 Ofset–gürültü dualitesi: bir önerme (destekleyici)

Bu bölüm, aday (ii)'nin gürültü büyütmesinin algoritmik değil **yapısal**
olduğunu, dolayısıyla "iki gradient farkıyla ofseti iptal et" fikrinin neden
çıkmaz olduğunu gösterir. Sonuç, aşağıda açıkça tanımlanan dar bir estimatör
sınıfı için geçerli bir tekillik önermesidir (genel bir teorem iddiası
değildir).

**Estimatör sınıfı $\mathcal{C}$.** Şu dört koşulu sağlayanlar:
(1) *Lineerlik:* $\widehat{\Delta q}=A_1\mathbf{y}_1+A_2\mathbf{y}_2$;
(2) *İki-ölçüm:* aynı $\Delta q$ ve aynı $\mathbf{b}$ ile
$\mathbf{y}_a=R_a\Delta q+\mathbf{b}+\eta_a$, $a\in\{1,2\}$;
(3) *Ön-yargısızlık:* her $\Delta q$ için $\mathbb{E}[\widehat{\Delta q}]=\Delta q$;
(4) *Tam ofset iptali:* her sabit $\mathbf{b}$ tahmine sızmaz.

**İddia.** $\mathcal{C}$ içinde tek çözüm $A_1=\Delta R^{-1}$, $A_2=-\Delta R^{-1}$'dir.

**Türetiş.** Koşul (3): $A_1R_1+A_2R_2=I$. Koşul (4): $A_1+A_2=0\Rightarrow A_2=-A_1$.
Birleştirince $A_1(R_1-R_2)=I\Rightarrow A_1=\Delta R^{-1}$. $\blacksquare$

**Sonuç.** $\mathcal{C}$ içindeki tek çözümün gürültü büyütmesi
$\|\Delta R^{-1}\|$'dir; $\Delta R\approx\varepsilon R$ ile yaklaşık
$\|R^{-1}\|/\varepsilon$. Bu, $\varepsilon\to 0$ sınırında patlar (§3.2'de SVD
spektrumuyla doğrulanır). Önemli nokta: **drift modu (iv) $\mathcal{C}$
dışındadır** — iki ölçüm aynı anda değil iki farklı *zamanda* alınır ve
estimatör $\Delta q$'yu değil $\delta q(t)=\Delta q(t)-\Delta q_0$'ı kestirir.
Ofset zaman farkıyla iptal olur, kondisyon sayısı $\kappa(R)\approx 193$
kalır. $\mathcal{C}$ dışındaki diğer yaklaşımlar (Tikhonov/TSVD, Bayesçi,
Kalman, çok-epoch) bu sınıra uymak zorunda değildir; farklı trade-off'larla
çalışırlar.

---

## 3. Sayısal Deneyler

Tüm testler ortak bir altyapıda koşuldu (Ek A): semplektik izleyici, gerçekçi
BPM gürültü ve ofset modeli, opsiyonel quad/dipol tilt'leri. Hızlandırıcı
parametreleri `params.json`, test parametreleri `drift_monitor/test_params.json`
içindedir.

### 3.1 Test 1 — Düzenlileştirme ham $\Delta R^{-1}$'i kurtarır mı?

BPM ofseti **yok**; bu test ideal şartta gürültü tabanını ölçer. Aynı veride
dört estimatör:

| Estimatör | y-RMS | y-korr | Yorum |
|---|---|---|---|
| Direct $R^{-1}$ (ofsetsiz) | 3.5 μm | 0.998 | Gürültü tabanı referansı |
| Ham $\Delta R^{-1}$ | 1865 μm | 0.085 | §2.4 sınırının nümerik gerçekleşmesi |
| Tikhonov (L-curve) | 53 μm | 0.348 | bias-varyans dengesi |
| TSVD ($k=3$, oracle) | 52 μm | 0.383 | oracle üst-sınır |

**Uyarı:** Direct $R^{-1}$ operasyonel bir rakip değildir; ofset sıfır
varsayımına dayanır ve gerçek halkada ($\mathbf{b}\neq 0$) 200+ μm'e tırmanır
(Test 4). Tabloda yalnızca gürültü tabanı referansı olarak yer alır.
Düzenlileştirme ham (ii)'yi $\sim 35\times$ iyileştirir ama korelasyon
0.998'den 0.35'e düşer (RMS azalır, biçim bozulur) ve hâlâ 10 μm hedefinin
çok üstündedir.

### 3.2 Test 2 — Uzaysal transfer fonksiyonu ve SVD spektrumu

Saf sinüsoidal patern girişine ($\Delta q_j=A\cos(2\pi kj/N)$) estimatör mod
transferi: Tikhonov/TSVD yüksek-$k$ modlarını ($k\gtrsim 18$) tamamen söndürür;
48 modun yalnızca 3–5'i kurtarılır. SVD spektrumu yan yana çizildiğinde
$\Delta R\approx\varepsilon R$ bulk ölçeklemesi doğrulanır (Şekil 1):
$\Delta R$'nin büyük singüler değerleri $R$'ninkilerin yaklaşık $\varepsilon$
katıdır; en küçük modlar ise bu ölçeklemenin altına çöker, $\Delta R$ her
durumda $R$'den ~2 mertebe kötü koşulludur.

$\varepsilon$ taraması (Şekil 6) asıl ölçeklemeyi netleştirir: ofset-iptal
eden estimatörün gürültü büyütmesi $\|\Delta R^{-1}\|=1/\sigma_{\min}(\Delta R)$
tüm $\varepsilon\in[0.005,0.10]$ aralığında temiz biçimde $\propto 1/\varepsilon$
patlarken, kondisyon sayısı $\kappa(\Delta R)$ kabaca sabit ($\sim 10^4$) kalır.
Yani $\varepsilon\to 0$ sınırında yöntemi kullanılamaz kılan κ değil,
$\|\Delta R^{-1}\|$'dir — bu, §2.4'teki alt-sınır önermesinin ($\|\Delta R^{-1}\|
\sim\|R^{-1}\|/\varepsilon$) doğrudan sayısal doğrulamasıdır.

### 3.3 Test 3 — Yatay model ters-suç kontrolü

$K_{x,\text{arc}}$ kalibrasyonu simülasyondan alınır (klasik ters-suç).
$K_{x,\text{arc}}(1+\delta)$, $\delta\in[-10\%,+10\%]$ taramasında yatay RMS
3.48–4.01 μm (0.5 μm değişim); dikey sabit (Maxwell). Gerçek halkada LOCO'nun
sağladığı $<1\%$ doğrulukta yöntem **operasyonel olarak ters-suçtan bağımsız**.

### 3.4 Test 4 — Drift modunun kalibrasyon-referansla gösterimi

**Senaryo.** $t=0$: 100 μm RMS hizalama + 50 μm RMS BPM ofseti kaydedilir.
$t=1..10$: hizalama 10 μm RMS yavaşça kayar, ofset sabit.

| Yöntem | Düzlem y | Düzlem x |
|---|---|---|
| Mutlak $R^{-1}\mathbf{y}(t)$ | ~197 μm (ofset baskın) | ~185 μm |
| **Drift $R^{-1}(\mathbf{y}-\mathbf{y}_0)$** | **6.6 μm** | **6.5 μm** |
| İyileşme | 29× | 28× |

50 μm RMS ofset yöntemi etkilemez; geriye yalnızca gürültü-kaynaklı taban
kalır (Şekil 2). (`drift_monitor/drift_monitor_sim.py`)

### 3.5 Test 5 — BPM ofseti zamanla kayarsa?

Drift modu $\mathbf{b}(t)\approx\mathbf{b}_0$ varsayar; ofset kayma hızı
$\dot{\mathbf{b}}$ arttıkça kestirime $R^{-1}\dot{\mathbf{b}}\,\Delta t$ olarak
sızar. Simülasyonda drift modu ile per-epoch iki-gradient + 30-epoch ortalama
karşılaştırıldığında, drift modunun üstünlüğünü kaybettiği geçiş
$\sim 2\,\mu$m/epoch ofset kayma hızındadır. Bu eşiğin gerçek bir pEDM
BPM sisteminde sağlanıp sağlanmadığı **deneysel olarak doğrulanması gereken
bir gerekliliktir** (§6); burada bir donanım iddiası yapılmamakta, yalnızca
yöntemin geçerli kaldığı kayma-hızı bütçesi nicelenmektedir.

### 3.6 Test 6 — Üç yöntemin adil karşılaştırması

Aynı veri, aynı bütçe (50 μm ofset, 1 μm gürültü, 10 μm drift, 0.2 mrad
tilt'ler):

| Estimatör | y-RMS | y-korr | x-RMS | x-korr |
|---|---|---|---|---|
| A: analitik $\Delta R$ | 3282 μm | -0.02 | 3756 μm | 0.09 |
| **B: drift modu** | **6.25 μm** | **0.85** | **7.18 μm** | **0.85** |
| C: sayısal $\Delta R$ | 980 μm | -0.02 | 1357 μm | -0.11 |

Her iki $\Delta R$ inşası da $\kappa\sim 10^4$ mertebesinde kötü koşulludur
(düzleme bağlı $\sim 7\times10^3$–$4\times10^4$; bkz. Şekil 1).
Sayısal $R$, $\Delta R$ yaklaşımını kurtarmaz — $\kappa(\Delta R)$ matrisin
nasıl inşa edildiğine değil $\varepsilon$ ve örgü fiziğine bağlıdır. Drift
modu yapısal olarak farklı bir problem (mutlak değil, değişim) çözdüğü için
bu sınırın dışındadır.

### 3.7 Test 8 — Örgü modeli hatası altında β-beating sağlamlığı

Drift modu $R^{-1}$'in doğruluğuna bağlıdır. Gerçek halkada LOCO + BBA sonrası
$\sim\%1$–5 β-beating ve faz hatası kalır. Nominal model $R_\text{nom}$ ile
gerçek makine $R_\text{true}$ ($\beta,\phi$'de $\varepsilon_\beta$ bozunumu)
arasında kasıtlı uyumsuzluk; veri $R_\text{true}$, kestirim $R_\text{nom}^{-1}$
ile (15 tohum medyanı):

| β-beating ($\varepsilon_\beta$) | y [μm] | x [μm] | Yorum |
|---|---|---|---|
| 0% (mükemmel) | 5.98 | 5.88 | gürültü tabanı |
| 0.5% | 5.94 | 5.91 | önemsiz |
| **1% (LOCO-gerçekçi)** | **6.08** | **6.09** | **hedef altında** |
| 2% | 6.48 | 6.58 | <7 μm |
| 5% | 8.57 | 9.13 | <10 μm güvenli marj |
| 10% | 13.00 | 14.51 | sınır aşılır |

LOCO sonrası tipik $\sim\%1$ β-beating'de taban yalnızca 0.1 μm artar
(5.98→6.08 μm); hedefe 3.9 μm marj kalır. Yöntem, standart-kalite LOCO'su olan
bir hızlandırıcıda operasyonel olarak kullanılabilir.
(Şekil 3; `drift_monitor/test8_betabeat.py`)

### 3.8 Test özeti

| Test | Soru | Ana sayı |
|---|---|---|
| 1 | Düzenlileştirme $\Delta R$'yi kurtarır mı? | 1865→52 μm, ama direct 3.5 μm |
| 2 | Düzenlileştirme nasıl çuvallıyor? | 48 modun 43–45'i siliniyor |
| 3 | Yatay modelde ters-suç? | $\pm10\%$ → <0.5 μm |
| 4 | Drift modu ofseti tolere eder mi? | 197 μm → 6.6 μm (29×) |
| 5 | Ofset kayarsa? | $<2\,\mu$m/epoch'a kadar üstün |
| 6 | Aday yöntemler yan yana? | $\Delta R$: 1000–3700 μm, drift: 6–7 μm |
| 8 | β-beating? | %1→6.1 μm; %5→8.6 μm; hedef altında |

---

## 4. Gözlenebilirlik Sınırı

Bu bölüm makalenin özgün omurgasıdır: drift modunun gürültü tabanının (§3)
rastgele bir sayı olmadığını, **hangi hizalama desenlerinin** iyi/kötü
izlendiğini belirleyen yapısal bir bölünmeden kaynaklandığını gösterir.

### 4.1 Yörünge kazanç yasası

Bu kafes (24-hücreli FODO) için yapılan mod analizinde, azimutal harmonik
$k$'lı bir kick desenine kapalı-yörünge tepkisi tune'da rezonant bir kazançla
ölçeklenir:

$$
G_k = \frac{C}{\,|Q_\text{eff}^2-k^2|\,},\qquad C\approx 24.8,\;\; Q_\text{eff}^2\approx 5.03.
$$

(Sabitler bu örgüye özgüdür; aşağıdaki argüman genel kapalı-yörünge yasası
değil, bu konfigürasyonun mod yapısı üzerine kuruludur.)

Kazanç $k\approx Q\,(\approx 2.7)$ civarında tepe yapar ve $k\gg Q$ için
hızla bastırılır. Bu, kapalı yörüngenin esasen bir **rezonant alçak-geçiren
filtre** gibi davrandığı anlamına gelir: düşük-$k$ (düzgün) kick desenleri
görünür, yüksek-$k$ (alternatif) desenler bastırılır.

### 4.2 Simetrik / antisimetrik ayrışım

48 boyutlu misalignment uzayı, hücre içi QF/QD çiftinin göreli işaretine göre
iki dik alt-uzaya ayrılır:

- **Antisimetrik** (QF ve QD zıt yönde): düzgün, **düşük-$k$** kick deseni
  üretir → $G_k$ büyük → kapalı yörüngede **görünür**.
- **Simetrik** (QF ve QD aynı yönde, $\text{QF}=\text{QD}=a_c$): kuadrupol
  gradyan işareti QF/QD'de değiştiği için **alternatif, yüksek-$k$ ($k\approx 24$)**
  kick deseni üretir → $G_k$ küçük → kapalı yörüngede **neredeyse görünmez**.

### 4.3 Per-mod SVD analizi (anahtar sonuç)

$R=U\Sigma V^\top$ ayrışımında her sağ-tekil vektör $V_i$ bir misalignment
deseni, $1/\sigma_i$ ise o deseni kestirirken yaşanan gürültü büyütmesidir.
Her mod için (i) gürültü büyütmesi $1/\sigma_i$ ve (ii) simetrik alt-uzaydaki
güç oranı hesaplanır:

$$
\sigma_{\max}=28.4,\quad \sigma_{\min}=0.147,\quad \kappa(R)=193.
$$

| Mod | $\sigma$ | Gürültü büyütme ($1/\sigma$) | Simetrik güç |
|---|---|---|---|
| 0 (en iyi) | 28.4 | 0.04 | %4 |
| 2 | 10.1 | 0.10 | %6 |
| 10 | 2.0 | 0.51 | %13 |
| 20 | 0.49 | 2.03 | %42 |
| 40 | 0.16 | 6.39 | %91 |
| 47 (en kötü) | 0.147 | 6.82 | **%98** |

En iyi 8 mod ortalama **%4** simetrik (antisimetrik baskın, iyi koşullu);
en kötü 8 mod ortalama **%96** simetrik. En kötü modun ($\sigma_{\min}$)
koşulluluk-kaynaklı duyarlılığı en iyiye göre $\sim 193$ kat ($=\kappa(R)$)
daha kötüdür; bu fark tümüyle simetrik alt-uzayda yoğunlaşır. (Bu, "tüm
simetrik modlar 193 kat kötü" demek değildir; tablodaki $1/\sigma$ sütunu
mod-mod artışı gösterir.) Bu ilişki Şekil 4'te açıkça görülür: gürültü
duyarlılığı $1/\sigma$ arttıkça simetrik güç oranı %100'e tırmanır.
(`drift_monitor/permode2.py`)

### 4.4 Sahte EDM kanalıyla çakışma

Bu yapısal kör noktanın önemi, simetrik alt-uzayın fiziksel anlamından gelir.
Sahte EDM'yi süren baskın mekanizma, ardışık dönüşlerin geometrik (Berry)
fazından doğan ve $dx\cdot dy$'ye orantılı, dolayısıyla misalignment ile
**kuadratik ($\propto\sigma^2$)** ölçeklenen bir terimdir [Omarov 2022, Fig. 9a].
Simülasyonlarımızda bu kuadratik sistematik kanala en güçlü katkıyı veren
hizalama bileşenlerinin **simetrik alt-uzayda yoğunlaştığı** görülmüştür
(çok-tohum RMS taramasında $\sigma$-üssü $\approx 2.0$ ile mekanizma bağımsız
olarak doğrulanır).

Sonuç birleştiğinde: kapalı-yörünge izleme, sahte-EDM-kritik simetrik
deseni en kötü koşullanmış modlarda en iyiye göre $\sim 193$ kata varan
duyarlılık dezavantajıyla "görür". Mevcut 1 μm BPM gürültüsünde bu modlarda
etkin hassasiyet $\sim 193\,\mu$m mertebesine düşer — drift modunun $\sim 6\,\mu$m'lik RMS tabanına bu simetrik bileşen
karışır, ama **sahte-EDM açısından kritik kısım iki mertebe daha az
hassasiyetle** kestirilebilir. Bu sınır, simetrik alt-uzayın yüksek-$k$
doğasının ($G_k\propto 1/|Q^2-k^2|$, $k\approx 24$) doğrudan bir sonucudur:
*bu ölçüm konfigürasyonunda* — tek kapalı yörünge, tek tepki matrisi — daha
iyi BPM donanımı veya daha fazla veri yığılması yalnız gürültü tabanını
düşürür, simetrik modların $\sim 193\times$'lik göreli dezavantajını
gidermez. Sınırın ilkesel olmadığını, yalnız bu konfigürasyona özgü olduğunu
vurgulamak gerekir: ek gözlemlenebilirler (farklı tune/optik kombinasyonları,
faz-ilerleme modülasyonu, ya da yörünge+spin birleşik ölçümü) prensipte bu
alt-uzaya erişebilir. Pratikte simetrik kanala doğrudan duyarlı gözlemlenebilir
karşı-dönen demet ayrımı veya spin presesyonudur [Omarov 2022].

---

## 5. Önerilen İşletme Modu

Tek bir yöntem her ihtiyacı karşılamaz; iki katmanlı bir mimari öneriyoruz
(Şekil 5).

**Yavaş mutlak katman (saatlik–günlük).** LOCO + BBA + survey ile mutlak
hizalama $\Delta q_0$, BPM ofseti $\mathbf{b}_0$ ve örgü modeli ($\beta,\phi,Q$,
dolayısıyla $R$). Bu, hızlandırıcı operasyonunda zaten var olan standart
prosedürdür.

**Hızlı drift katmanı (sürekli, saniye–dakika).** Bu çalışmanın katkısı:
$\widehat{\delta q}(t)=R^{-1}(\mathbf{y}(t)-\mathbf{y}_0)$. Fizik run'ı boyunca
sürekli çalışır, veri toplamayı bozmaz, kalibrasyondan beri (antisimetrik)
hizalama değişimini izler.

İki katman tamamlayıcıdır: yavaş katman mutlak referansı ve $R$'yi sağlar,
hızlı katman o referansa göre değişimi takip eder. §4'ün sınırı gereği, her
iki katman da simetrik alt-uzayı kapalı yörüngeden kurtaramaz; bu bilgi dış
bir gözlemlenebilirden (demet ayrımı / spin) gelmelidir.

---

## 6. Tartışma ve Sonuç

Bu çalışmada pEDM AG halkasında kuadrupol hizalama driftini sürekli izlemek
için kalibrasyon-referans yöntemi önerildi ve gerçekçi BPM hata bütçesi
altında (50 μm ofset, 1 μm gürültü, 0.2 mrad tilt) 6–7 μm RMS hassasiyetle,
$\sim\%1$ β-beating'e dayanıklı şekilde çalıştığı sistematik testlerle
gösterildi. Yöntem iki katmanlı bir mimaride (yavaş LOCO/BBA + hızlı drift)
operasyonel olarak konumlandırıldı.

Daha önemlisi, yöntemin **temel sınırı** nicelendi: per-mod SVD analizi,
kapalı-yörünge izlemenin kör noktasının %96 simetrik içerikli en kötü
koşullanmış modlar olduğunu (en kötü modda en iyiye göre $\sim 193$ kat
koşulluluk-kaynaklı duyarlılık dezavantajı) ve bu modlara en güçlü katkı veren
hizalama bileşenlerinin sahte EDM'yi süren $dx\cdot dy$ geometrik-faz kanalıyla
örtüştüğünü gösterdi. Bu, kapalı-yörünge yöntemlerinin sahte-EDM-kritik bilgiyi bu ölçüm
konfigürasyonunda neden sağlayamadığının nicel bir ifadesidir: antisimetrik
drift hassasça izlenir, simetrik (kritik) kanal ise gürültü sınırında kalır ve
donanım iyileştirmesiyle kolayca giderilemez. Bu kanala erişim ek bir
gözlemlenebilir gerektirir (farklı optik kombinasyonları ya da yörünge-dışı bir
prob — demet ayrımı/spin) [Omarov 2022].

Çalışmanın kısıtları:

- **Lineer model ötesi.** Sonuçlar Eş. (1)'e dayanır; gerçek halkada BPM gain
  hataları, roll/kuplaj, sekstupol feed-down, fringe alanlar, histerezis ve
  akım dalgalanması vardır. Bunların sistematik incelenmesi gelecek çalışmadır.
- **Tek kafes.** Tüm sonuçlar 24-hücreli FODO'da. Genellenebilirlik kazanç
  yasası $G_k$ üzerinden tahmin edilebilir ama doğrulanmamıştır.
- **BPM ofset uzun-vadeli kararlılığı.** Drift modunun temel varsayımı
  $\mathbf{b}(t)\approx\mathbf{b}_0$ olduğundan, donanımın saatler–günler
  ölçeğinde ofset stabilitesi deneysel olarak karakterize edilmelidir.

---

## Kaynaklar

*(Taslak — tamamlanacak.)*

1. Z. Omarov, H. Davoudiasl, … Y. K. Semertzidis vd., "Comprehensive
   symmetric-hybrid ring design for a proton EDM experiment at below
   $10^{-29}\,e\cdot\text{cm}$," *Phys. Rev. D* **105**, 032001 (2022).
2. pEDM Collaboration, "Storage Ring Proton Electric Dipole Moment Experiment
   Proposal," BNL (2011).
3. S. Y. Lee, *Accelerator Physics*, 3rd ed., World Scientific (2011).
   [tepki matrisi formalizmi]
4. J. Safranek, "Experimental determination of storage ring optics using orbit
   response measurements," *NIM A* **388**, 27 (1997). [LOCO]

---

## Ek A: Sayısal simülasyon altyapısı

Tüm sonuçlar Python tabanlı bir altyapıda üretildi. Parçacık takibi
dördüncü-mertebe Gauss-Legendre (GL4) semplektik integratörüyle; analitik tepki
matrisi Courant-Snyder formalizmiyle (`drift_monitor/fodo_lattice.py`) inşa
edildi. Ana gösterim `drift_monitor/drift_monitor_sim.py` (Test 4), β-beating
sağlamlığı `drift_monitor/test8_betabeat.py` (Test 8), per-mod SVD analizi
`drift_monitor/permode2.py` içindedir. Hızlandırıcı parametreleri `params.json`,
test parametreleri `drift_monitor/test_params.json`'dadır.

---

## Şekiller ve Caption'lar

Tüm şekiller **PRD tek-sütun** formatında (genişlik 3.375 in / 246 pt, serif +
Computer-Modern mathtext, 600 dpi) üretilir: `drift_monitor/make_figures.py`
(Şekil 1–4, 6) ve `drift_monitor/make_fig5_architecture.py` (Şekil 5). Çok
panelli şekiller dikey istiflenir ve (a)/(b) ile etiketlenir; şekil-içi başlık
yoktur, açıklama aşağıdaki caption'lardadır (PRD konvansiyonu).

**ŞEKİL 1.** (`fig1_svd_spektrum.png`) Tepki matrisi $R$ ve iki-gradient farkı
$\Delta R=R(g_1)-R(g_1(1+\varepsilon))$ ($\varepsilon=0.02$) için singüler-değer
spektrumları; (a) dikey, (b) yatay düzlem. Kesikli gri çizgi $\varepsilon\,\sigma(R)$
beklenen bulk ölçeklemesidir. $\Delta R$'nin büyük singüler değerleri bu çizgiyi
izler; en küçük modlar daha da çöker. $R$ iyi koşulludur ($\kappa\approx 193/140$),
$\Delta R$ ise ~2 mertebe kötüdür. (§2.4, §3.2)

**ŞEKİL 2.** (`fig2_drift_izleme.png`) Test 4 kalibrasyon-referans drift izleme,
(a) dikey, (b) yatay düzlem. Gerçek drift, kestirilen drift ve naif mutlak
rekonstrüksiyon hatası epoch'a karşı (log ölçek). 50 μm BPM ofseti mutlak
rekonstrüksiyonu ~200 μm'de boğarken drift modu 6–7 μm RMS'te izler. (§3.4)

**ŞEKİL 3.** (`fig3_betabeat.png`) Test 8: örgü-modeli hatası (β-beating)
altında drift takip hatası, $\varepsilon_\beta$'ya karşı (15-tohum medyanı).
Kesik kırmızı çizgi 10 μm hedef, noktalı gri çizgi LOCO-gerçekçi %1 seviyesi.
%1'de hata 6.1 μm; %5'e kadar hedef altında. (§3.7)

**ŞEKİL 4.** (`fig4_permode_svd.png`) $R$'nin (dikey düzlem) singüler modları:
sol eksen (mavi) gürültü duyarlılığı $1/\sigma_i$, sağ eksen (kırmızı) modun
simetrik alt-uzaydaki güç oranı. En kötü koşullanmış modlar (sağ) %96–98
simetriktir; gürültü duyarlılığı arttıkça simetrik içerik %100'e tırmanır. (§4.3)

**ŞEKİL 5.** (`fig5_mimari.png`) İki-katmanlı hizalama izleme mimarisi: yavaş
mutlak katman (LOCO/BBA → $\Delta q_0,\mathbf{b}_0,R$) ve hızlı drift katmanı
($R^{-1}(\mathbf{y}(t)-\mathbf{y}_0)$). Simetrik (sahte-EDM-kritik) alt-uzay her
iki katmanda da gürültü sınırındadır. (§5)

**ŞEKİL 6.** (`fig6_epsilon_sweep.png`) $\varepsilon$ taraması: (a) ofset-iptal
eden estimatörün gürültü büyütmesi $\|\Delta R^{-1}\|=1/\sigma_{\min}(\Delta R)$,
noktalı $1/\varepsilon$ referansıyla — temiz $1/\varepsilon$ ölçeklemesi;
(b) kondisyon sayısı $\kappa(\Delta R)$, $\varepsilon$'dan kabaca bağımsız
($\sim 10^4$). $\varepsilon\to0$'da yöntemi bozan κ değil $\|\Delta R^{-1}\|$'dir.
(§2.4, §3.2)
