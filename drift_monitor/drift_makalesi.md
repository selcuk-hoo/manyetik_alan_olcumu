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
kalır. Ayrıca sahte EDM'i süren baskın kanal, yatay ve dikey kaçıklığın
**çarpımına** ($dx\cdot dy$, $\propto\sigma^2$) bağlı bilineer bir geometrik-faz
etkisidir [Omarov 2022, Fig. 9a]; lineer ve düzlem-ayrık olan drift monitör bu
çarpımı **yapısal olarak temsil edemez**. Bu iki yapısal neden — düzlem-ayrıklık
ve simetrik alt-uzay körlüğü — kapalı-yörünge izlemenin antisimetrik hizalama
driftini hassasça takip ederken, sahte-EDM-kritik bilgiyi tanımlı biçimde dış
bir gözlemlenebilire (karşı-dönen demet ayrımı / spin) bıraktığını gösterir.
Yöntemin kesin tanımlı bir **geçerlilik alanı** ve kesin tanımlı bir **kör
noktası** vardır.

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
yapısal olarak duracağını nicelemektir. Kısaca: kapalı yörüngeyi spin-tabanlı
hizalama teşhisinin yerine koymuyoruz; onun sürekli bir hizalama-drift izleyici
olarak hizmet edebileceği **gözlenebilir alt-uzayı** ve bu alt-uzayın sınırını
niceliyoruz.

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
ölçüm gürültüsüdür. **Modelleme varsayımı:** yatay–dikey kuplaj yoktur (skew
bileşeni = kuadrupol dönmesi sıfır kabul edilir), dolayısıyla problem birbirinden
bağımsız iki $48\times48$ sisteme ($dy\to$ dikey yörünge, $dx\to$ yatay yörünge)
ayrılır. Bu düzlem-ayrıklığın drift izlemenin sahte-EDM ile ilişkisi açısından
önemli bir sonucu vardır (§4.4): lineer yörünge tepkisi tek bir düzlemin yer
değiştirmesini taşır, iki düzlemin *çarpımını* değil.

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

**Analitik R'nin tam parçacık takibiyle doğrulanması.**
Analitik R, tüm sonuçların temelidir; bu yüzden onu bağımsız bir tam parçacık
izleyicisiyle (C++ GL4 semplektik, `integrator.cpp`) karşılaştırıyoruz.
İzleyici R'si, her kuadrupole küçük bir $\delta y$ (veya $\delta x$) perturbasyonu
uygulayıp kapalı yörünge tepkisini 48 BPM'de ölçerek kurulur
(`build_response_matrix.py`). Sonuç (Şekil~\ref{fig:thsim}): analitik ve
izleyici R'leri eleman-eleman korelasyon **0.9992** (dikey) / **0.9977** (yatay),
göreli fark $\|R_\text{sim}-R_\text{an}\|/\|R_\text{an}\|=\%5.3/\%7.2$, kondisyon
sayıları tutarlı ($\kappa$: 228 vs 193 dikey, 135 vs 140 yatay). Böylece
analitik R üzerine kurulan tüm gözlenebilirlik analizi (SVD mod yapısı, $\kappa$,
drift kurtarımı) tam parçacık dinamiğiyle doğrulanmış olur.
(`drift_monitor/theory_sim_validate.py`)

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

Tablodaki $x$ ve $y$ sütunları **bağımsızdır**: bu çalışmadaki lineer model
yatay ve dikey düzlemleri ayrık ele alır (skew kuplajı = kuadrupol dönmesi
sıfır), dolayısıyla bir $x\!-\!y$ çapraz (kuplaj) terimi yoktur. Her düzlem
kendi $48\times48$ sistemidir. (Sahte EDM'i süren $dx\cdot dy$ kuplajı bu lineer
yörünge modelinde değil, ikinci-derece bir spin etkisidir — bkz. §4.4.)

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

**İzleyici doğrulaması — ve focal length hatası nereye düşüyor.** Yukarıdaki
tarama β-beating'i analitik olarak ($\beta,\phi$ doğrudan bozularak), yani
*belirtiyi* modelliyordu. Bunu β-beating'in fiziksel *kaynağıyla* tekrarladık:
**kuadrupol odak-uzaklığı (focal length) hataları.** Bir kuadrupolün odak
uzaklığı $1/f = (G/B\rho)\,L$ gradyanla belirlenir; fraksiyonel bir gradyan
hatası $\delta G/G$ doğrudan bir focal-length hatasıdır. Yani focal-length
hatasını β-beating'in *içine gömmedik* — tersine, onu **kaynak** olarak verip
sonucunu izleyiciye ürettirdik. Bu yaklaşım focal hatanın *tam* etkisini yakalar
(yalnız β-fonksiyon şekil bozulması değil, tune kayması da dahil), çünkü hepsi
$R_\text{nom}$ ile gerçek $R_\text{true}$ arasındaki uyumsuzluğa yansır.

Her kuadrupole bağımsız rastgele $\delta G/G$ verip bozulmuş $R_\text{true}$
izleyiciden kuruldu; monitör nominal $R_\text{nom}^{-1}$ ile çalıştı. Gradyan
(focal) hata RMS'i 0, %2, %5 için drift takip hatası 6.17, 6.17, 6.25 μm
($\kappa$: 228→239→233). Gerçek focal-hata kaynaklı β-beating, analitik
$\beta,\phi$ taramasından bile daha sağlam çıktı — Test 8 tam parçacık
dinamiğiyle doğrulandı. (`drift_monitor/drift_betabeat_sim.py`)

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

### 4.1 Hangi kaçıklık desenleri kapalı yörüngede görünür?

Önce kurulumu somutlaştıralım. Halka 24 FODO hücresinden oluşur; her hücrede
bir **odaklayıcı** kuadrupol (QF) ve bir **dağıtıcı** kuadrupol (QD) vardır —
toplam 48 kuadrupol. Bir "hizalama hatası deseni", 48 kuadrupolün dikey
kaymalarını veren 48 sayılık bir vektördür ($\Delta q\in\mathbb{R}^{48}$). Bir
kaymış kuadrupolün kapalı yörüngeye etkisi bir **kick**tir (küçük açısal sapma)
ve büyüklüğü (gradyan $\times$ kayma) ile orantılıdır.

Bir kick dizisinin kapalı yörüngeye toplam etkisi, kick deseninin **azimutal
harmoniğine** — yani halka çevresi boyunca kaç kez tekrarladığına, $k$ — bağlıdır.
Kapalı yörünge harmonik $k$'ya bir kazançla yanıt verir:

$$
G_k = \frac{C}{\,|Q_\text{eff}^2-k^2|\,},\qquad C\approx 24.8,\;\; Q_\text{eff}^2\approx 5.03,
$$

burada $Q$ betatron tune'udur ($\approx 2.7$); sabitler bu örgüye özgüdür.
Kazanç tune'a yakın ($k\approx Q$) harmoniklerde en büyük, $k\gg Q$ harmoniklerde
küçüktür. Yani **kapalı yörünge bir mod-seçici (rezonant) filtredir**: yalnız
belirli uzaysal harmonikleri güçlü biçimde gösterir, ötekileri bastırır. (Bu
klasik bir alçak-geçiren filtre değildir — tepe $k=0$'da değil $k\approx Q$'dadır.)

### 4.2 İki tür hizalama deseni: simetrik ve antisimetrik

Hangi desenin hangi harmoniğe düştüğünü görmek için hizalama hatalarını
hücre-hücre iki gruba ayırıyoruz. Her hücredeki (QF, QD) çifti için iki uç durum:

- **Antisimetrik desen:** QF ve QD **zıt yönde** kayar (ör. QF $+a$, QD $-a$).
- **Simetrik desen:** QF ve QD **aynı yönde** kayar (QF $+a$, QD $+a$).

Her hizalama hatası bu ikisinin toplamı olarak yazılabilir; böylece 48-boyutlu
desen uzayı iki dik **alt-uzaya** ayrılır (her biri 24-boyutlu). Buradaki
"alt-uzay" basitçe *belli bir simetri kuralına uyan tüm desenlerin ailesi*
demektir: simetrik alt-uzay = hücre içi QF/QD'nin aynı yönde kaydığı tüm desenler,
antisimetrik alt-uzay = zıt yönde kaydığı tüm desenler.

Belirleyici ayrıntı **gradyan işaretidir**: QF odaklar (gradyan $+$), QD dağıtır
(gradyan $-$). Kick = gradyan $\times$ kayma olduğundan:

- **Antisimetrik** (kaymalar $+a/-a$, gradyanlar $+/-$): iki kick **aynı işarette**
  toplanır $\rightarrow$ hücreden hücreye düzgün, **düşük-$k$** desen
  $\rightarrow$ $G_k$ büyük $\rightarrow$ **yörüngede görünür**.
- **Simetrik** (kaymalar $+a/+a$, gradyanlar $+/-$): iki kick **zıt işarette**
  $\rightarrow$ hücre içinde işaret değiştiren, hızlı alternatif **yüksek-$k$
  ($k\approx 24$)** desen $\rightarrow$ $k\gg Q$ $\rightarrow$ $G_k$ küçük
  $\rightarrow$ **yörüngede neredeyse görünmez**.

Sezgi: simetrik kayma, gradyan alternasyonu yüzünden kendi kendini söndüren bir
kick örüntüsü üretir; kapalı yörünge bunu "görmez", antisimetrik kaymayı ise
güçlü görür.

### 4.3 Hangi desenler ne kadar gürültüyle kestirilir? (per-mod SVD)

§4.2'deki simetrik/antisimetrik ayrım kategorikti (iki uç durum). Tepki
matrisinin **tekil değer ayrışımı (SVD)** bunu nicel ve sürekli hale getirir.
$R=U\Sigma V^\top$ yazıldığında her sağ-tekil vektör $V_i$ bir hizalama deseni
("mod"), karşılık gelen tekil değer $\sigma_i$ ise o desenin yörüngede ne kadar
güçlü göründüğüdür. Drift monitör bir modu kestirirken gürültüyü $1/\sigma_i$
ile büyütür: büyük-$\sigma$ modlar temiz, küçük-$\sigma$ modlar gürültülü
kestirilir. Her mod için (i) gürültü büyütmesi $1/\sigma_i$ ve (ii) modun §4.2
anlamında **simetrik içeriği** (gücünün yüzde kaçı simetrik alt-uzayda)
hesaplanır:

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

### 4.4 Bunun sahte EDM ile ilişkisi — ve kesin sınır

Drift monitörün amacı hizalama driftini izlemekti; ama asıl önemsediğimiz
sahte EDM'dir. İki arasındaki bağı dürüstçe, sınırlarıyla kuruyoruz.

**Sahte EDM'in baskın kanalı bir yatay–dikey kuplajıdır.** Sahte vertikal spin
presesyonunu ($dS_y/dt$) süren baskın terim, yatay ($dx$) ve dikey ($dy$)
kaçıklığın **çarpımına** orantılı bir geometrik (Berry) faz katkısıdır;
kaçıklıkla **kuadratik** ($\propto\sigma^2$) ölçeklenir [Omarov 2022, Fig. 9a;
ayrıca bizim §13.7 spin-takip taramamız $\sigma$-üssünü $2.0$ ölçer]. Yalnız-$dy$
katkısı doğrusal ve küçüktür; yalnız-$dx$ ise $S_y$ için sıfırdır. Anahtar
sözcük **çarpım** $dx\cdot dy$'dir: etki iki düzlemin *birlikte* kaçık olmasını
gerektirir (komütatif olmayan iki eksenli spin dönmesi → net Berry fazı).

**Yapısal sınır (kesin).** Drift monitörün lineer modeli (Eş. 1, §2.1)
yatay ve dikey düzlemleri **ayrık** ele alır: $dy$ yalnız dikey yörüngeyi, $dx$
yalnız yatay yörüngeyi üretir; birinci derecede çapraz terim yoktur (çapraz terim
ancak skew kuplajı = kuadrupol dönmesiyle doğar, bu modelde sıfır). Tablo 6'nın
$x$ ve $y$ sütunlarını birbirinden bağımsız vermesinin sebebi tam da budur:
problem iki ayrık $48\times48$ sistemdir, ortak $x\!-\!y$ terimi yoktur.
Dolayısıyla monitör, sahte EDM'i süren **bilineer $dx\cdot dy$ büyüklüğünü
yapısal olarak temsil edemez ve kısıtlayamaz** — bu, mod/gözlenebilirlik
ayrıntısından bağımsız, kesin bir ifadedir. Kapalı yörünge tek düzlemin
*ortalama* yer değiştirmesini taşır; iki düzlemin *çarpımını* taşımaz.

**Düzlem-içi sınır.** Buna ek olarak, *her bir düzlemin kendi içinde* monitör
simetrik (yüksek-$k$) alt-uzaya §4.2–4.3 gereği neredeyse kördür (en kötü modda
$\sim 193$ kat gürültü dezavantajı). Yani simetrik bir dikey drift, sahte-EDM'e
katkı verse bile, monitörün en zayıf gördüğü yöndedir.

**Açıkça belirtilen sınır (dürüstlük).** Baskın $dx\cdot dy$ kanalının
ağırlığının özellikle *simetrik* alt-uzayda mı yoğunlaştığı ayrı, desen-bağımlı
bir sorudur; spin taramamız bu kanalın seed'e göre $\sim 36\times$ saçıldığını
gösterir (§13.7) ve bu ayrışım tek bir kapalı-yörünge ölçümüyle çözülemez — bu
da rapor ettiğimiz sınırın bir parçasıdır. Yalnız-$dy$ (baskın olmayan) kanalda
spin simülasyonu, gözlenebilir harmonikler çıkarıldıktan sonra artık sahte-EDM'in
küçük kaldığını gösterir; baskın kanal için kesin nicel alt-uzay ayrışımı
gelecek iştir.

**Sonuç.** Kapalı-yörünge drift monitörü sahte-EDM-kritik bilgiyi **iki bağımsız
yapısal nedenle** kısıtlayamaz: (1) lineer ve düzlem-ayrık olduğundan baskın
kanalı süren bilineer $dx\cdot dy$ büyüklüğünü ilkesel olarak göremez;
(2) gördüğü her düzlem içinde simetrik (yüksek-$k$) alt-uzayı zayıf gözler. Her
iki sınır da daha iyi BPM donanımı ya da daha fazla veriyle kapanmaz; bu kanala
erişim **farklı bir gözlemlenebilir** gerektirir — karşı-dönen (CW/CCW) demet
ayrımı veya doğrudan spin presesyonu [Omarov 2022]. Drift monitörün net rolü
budur: antisimetrik hizalama driftini ucuza ve sürekli izler, sahte-EDM-kritik
kanalı ise tanımlı şekilde dış bir gözlemlenebilire bırakır.

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

Daha önemlisi, yöntemin **temel sınırı** iki bağımsız yapısal nedenle nicelendi
(§4.4). **Birincisi (kesin):** sahte EDM'i süren baskın kanal, yatay ve dikey
kaçıklığın çarpımına ($dx\cdot dy$) bağlı bilineer bir etkidir [Omarov 2022];
lineer ve düzlem-ayrık olan monitör bu çarpımı yapısal olarak temsil edemez
(Tablo 6'da $x$/$y$ sütunlarının bağımsızlığı bunun göstergesi). **İkincisi:**
her düzlem içinde, per-mod SVD analizi monitörün kör noktasının %96 simetrik
içerikli en kötü koşullanmış modlar olduğunu gösterir (en kötü modda en iyiye
göre $\sim 193$ kat duyarlılık dezavantajı). Baskın kanalın ağırlığının
özellikle simetrik alt-uzayda mı yoğunlaştığı ise desen-bağımlı, açık bir
sorudur (§4.4). Net sonuç: antisimetrik drift hassasça izlenir; sahte-EDM-kritik
bilgi ise donanım/veriyle kapanmayan yapısal nedenlerle erişilemez ve farklı bir
gözlemlenebilir (karşı-dönen demet ayrımı / spin) gerektirir [Omarov 2022].

Çalışmanın kısıtları:

- **Lineer model ötesi.** Sonuçlar Eş. (1)'e dayanır; gerçek halkada BPM gain
  hataları, roll/kuplaj, sekstupol feed-down, fringe alanlar, histerezis ve
  akım dalgalanması vardır. Bunların sistematik incelenmesi gelecek çalışmadır.
- **Tek kafes.** Tüm sonuçlar 24-hücreli FODO'da. Genellenebilirlik kazanç
  yasası $G_k$ üzerinden tahmin edilebilir ama doğrulanmamıştır.
- **BPM ofset kararlılığı operasyonel bir gerekliliktir.** Yöntem BPM ofsetine
  *duyarsız* değildir; ofsetin *hızına* duyarsızdır. Drift modu
  $\mathbf{b}(t)\approx\mathbf{b}_0$ varsayar ve geçerliliği bir kayma-hızı
  bütçesine bağlıdır: $\dot{\mathbf{b}} \lesssim 2\,\mu$m/epoch (§3.5). Bu eşiğin
  pEDM BPM donanımında saatler–günler ölçeğinde sağlanıp sağlanmadığı deneysel
  olarak karakterize edilmelidir; yöntemin pratik kullanılabilirliğinin
  belirleyicisidir.

---

## Kaynaklar

1. **[omarov2022]** Z. Omarov, H. Davoudiasl, S. Hacıömeroğlu, V. Lebedev,
   W. M. Morse, Y. K. Semertzidis, A. J. Silenko, E. J. Stephenson, R. Suleiman,
   "Comprehensive symmetric-hybrid ring design for a proton EDM experiment at
   below $10^{-29}\,e\cdot\text{cm}$," *Phys. Rev. D* **105**, 032001 (2022).
   doi:[10.1103/PhysRevD.105.032001](https://doi.org/10.1103/PhysRevD.105.032001);
   arXiv:2007.10332.
2. **[anastassopoulos2016]** V. Anastassopoulos vd., "A storage ring experiment
   to detect a proton electric dipole moment," *Rev. Sci. Instrum.* **87**,
   115116 (2016).
   doi:[10.1063/1.4967465](https://doi.org/10.1063/1.4967465);
   arXiv:1502.04317. [pEDM deney önerisinin hakemli versiyonu]
3. **[lee2011]** S. Y. Lee, *Accelerator Physics*, 3rd ed., World Scientific
   (2011). doi:[10.1142/8335](https://doi.org/10.1142/8335).
   [tepki matrisi formalizmi]
4. **[safranek1997]** J. Safranek, "Experimental determination of storage ring
   optics using orbit response measurements," *Nucl. Instrum. Meth. A* **388**,
   27 (1997).
   doi:[10.1016/S0168-9002(97)00309-4](https://doi.org/10.1016/S0168-9002(97)00309-4).
   [LOCO]

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
(Şekil 1–4, 6), `drift_monitor/make_fig5_architecture.py` (Şekil 5) ve
`drift_monitor/theory_sim_validate.py` (Şekil 7). Çok panelli şekiller dikey
istiflenir ve (a)/(b) ile etiketlenir; şekil-içi başlık yoktur, açıklama
aşağıdaki caption'lardadır (PRD konvansiyonu).

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

**ŞEKİL 7.** (`fig7_theory_sim.png`) Analitik Courant-Snyder R (teori) ile tam
parçacık izleyicisinden (C++ GL4, `integrator.cpp`) kurulan R'nin eleman-eleman
karşılaştırması; (a) dikey, (b) yatay düzlem. Noktalar $y=x$ üzerinde:
korelasyon 0.9992/0.9977, $\kappa$ tutarlı. Drift makalesinin analitik temelini
tam parçacık dinamiğiyle doğrular. (§2.2)
