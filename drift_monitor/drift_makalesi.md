# Kapalı-yörünge tabanlı hizalama-drift tanısının gözlenebilirlik sınırı: koherent yer-hareketine yapısal körlük ve hassas (EDM) halkalardaki kör noktayla bağı

> **Durum:** Taslak (v0.3, 2026-06). Dil: Türkçe. **Literatür-bilgili dürüst
> konumlandırma:** Bu bir "yeni yöntem/algoritma" makalesi DEĞİLDİR — kullanılan
> matematik (ofset-iptalli diferansiyel orbit = SOFB; R⁻¹ ile rekonstrüksiyon =
> LOCO; SVD null-space = Chung 1993; koherent yer-hareketi → küçük COD = Rossbach
> 1989; ATL = Shiltsev; EDM sistematiği = Omarov, Hacıömeroğlu & Semertzidis)
> yerleşik prior-art'tır. **Özgün katkı bir *sentez/negatif-sonuç*tur:** bilinen
> yer-dalgası bastırma fiziğini, tanı *response*-matrisinin SVD null-space'ine
> açık projeksiyonla bir **yapısal gözlenebilirlik körlüğü** ifadesine çevirmek
> ve bu kör alt-uzayın **EDM-ilgili simetrik artıkla çakıştığını** göstermek.
> NIM-A tipi odaklı gözlenebilirlik-sınırı/negatif-sonuç makalesi hedeflenir.

---

## Özet

Depolama halkalarında mıknatıs hizalaması yer hareketi ve termal etkilerle zamanla
**kayar**; bu drifti mevcut BPM'lerle, girişimsiz, pasif izlemek doğal ve ucuz bir
fikirdir. Bu çalışmada böyle bir kapalı-yörünge tabanlı drift tanısının **temel bir
yapısal gözlenebilirlik sınırı** taşıdığını gösterir ve niceliyoruz. Tanı aracı,
kalibrasyon-anı yörüngesine göre fark alıp $\widehat{\delta q}(t) =
R^{-1}(\mathbf{y}(t)-\mathbf{y}_0)$ ile drifti kestirir; sabit BPM ofseti zaman
farkında iptal olur (bu matematik standart yavaş-orbit-geribeslemesi/LOCO ile
ortaktır — özgünlük burada değil). **Asıl sonuç gözlenebilirlik sınırıdır:**
response matrisinin singüler-değer ayrışımı, hangi *drift desenlerinin* görünür
hangilerinin **kör** olduğunu belirler. Bir FODO hücresindeki zıt-gradyanlı quad
çiftinin **simetrik** (aynı yön) yer değişimi alternatif (yüksek mekânsal-frekans)
kick üretir; bu da rezonant kazanç yasası $G_k\propto 1/|Q^2-k^2|$ ile bastırılarak
neredeyse sıfır kapalı-yörünge verir (en kötü modda $\sim 193\times$ gürültü
dezavantajı). **Özgün adım:** gerçekçi drift kaynaklarının mekânsal-korelasyon
spektrumunu (ATL yer-hareketi) bu SVD alt-uzaylarına projekte ederiz ve gösteririz
ki **koherent (uzun-korelasyonlu) yer-hareketi tam da kör simetrik alt-uzaya düşer**
— komşu mıknatısları birlikte kaydırdığından. Böylece kapalı-yörünge tanısı, bedava
ve girişimsiz olmasına karşın, **koherent drifte yapısal olarak kördür** (yalnız
korelasyonsuz, magnet-özel drifti görür). Bu kör nokta, hassas halkalarda — özellikle
depolama-halkası EDM aramalarında, sahte EDM'yi süren $dx\cdot dy$ geometrik-faz
sistematiğiyle ilişkili simetrik artıkla — çakışır; dolayısıyla orbit-tabanlı tanı
o artığı bastıramaz ve spin-tabanlı yöntemleri (CR demet ayrımı, SBA) gerektirir.
Sonuç, yer-dalgası bastırma fiziğinin (Rossbach 1989) ve SVD null-space'in
(Chung 1993) bilinen sonuçlarını, **bir tanı-gözlenebilirlik teoremine** ve onun
EDM kör-noktası sonucuna sentezler — yeni bir matematik aracı değil, yaygın bir
yöntem-sınıfının fiziksel limitinin açık ifadesidir.

---

## 1. Giriş

### 1.1 Bağlam: hizalama drifti ve mevcut araçlar

Depolama halkalarında mıknatıs konumları sabit değildir: yer hareketi (ATL
yasası), termal döngüler ve mekanik oturma, hizalamayı saatler–günler ölçeğinde
**kaydırır**. Bu drift demet kararlılığını bozar ve hizalamaya duyarlı her
uygulamada sistematik bütçeye girer. Hizalama/optik hatalarını kapalı yörüngeden
çıkarmak olgun bir alandır (LOCO, orbit-response-matrix analizi, beam-based
alignment, yavaş orbit geri-beslemesi). Bu çalışmanın hedeflediği boşluk dar ama
pratik: **çalışma sırasında, girişimsiz, BPM ofsetlerine bağışık biçimde hizalama
*driftini* izlemek** — mutlak hatayı çözmek değil, kalibrasyon-anına göre
değişimi takip etmek. Mutlak rekonstrüksiyonun temel engeli, BPM elektronik
ofsetlerinin ($\sim 100\,\mu$m) ölçülmek istenen hizalama sinyaliyle aynı
büyüklükte olmasıdır; zaman-farkı alan bir drift ölçümü bu ofseti yapısal olarak
iptal eder.

**Motivasyon örneği — pEDM.** Yöntemi, hizalama gereksinimi alışılmadık ölçüde
sıkı olan bir uygulamayla somutlaştırıyoruz. Proton EDM deneyinin simetrik-hibrit
tasarımı, demet dinamiğini iyileştirmek için manyetik kuadrupollerle
**alternating-gradient (FODO)** odaklamaya geçmiştir [Omarov 2022]; bu geçişle
baskın sistematik **kuadrupol hizalama hatalarıdır**. Hizalanmamış bir kuadrupol
demete net kuvvet uygular ve halka boyunca uygun bileşeni EDM sinyalini taklit
eden sahte dikey spin presesyonu üretir. Omarov vd. [Omarov 2022] bu sahte alanın
bütçesini türetmiş ve hedef hassasiyeti ($d_p < 10^{-29}\,e\cdot\text{cm}$,
eşdeğer $dS_y/dt < 1$ nrad/s) korumak için hizalamanın spin-koherans ölçeğinde
**$\sim$10 μm seviyesinde bilinmesi/kontrol edilmesi** gerektiğini göstermiştir —
bu, yöntemimiz için zorlu bir hedef hassasiyet sağlar. (Yöntemin pEDM
sistematiğini *çözdüğünü* iddia etmiyoruz; §4'te göstereceğimiz gibi geçerlilik
alanı sınırlıdır ve spin-tabanlı yöntemleri tamamlar — bkz. §4.4.)

Halkada $2\times 24 = 48$ manyetik kuadrupol vardır; her biri yatay ($dx$) ve
dikey ($dy$) eksende bağımsız kayabilir. Ölçüm aracı 48 BPM çiftidir.

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

### 1.3 Çalışmanın katkısı (ve ne olmadığı)

Açıkça belirtelim: bu çalışma **yeni bir ölçüm yöntemi, yeni bir orbit-düzeltme
algoritması ya da yeni bir BPM önermiyor.** Kullandığı araçlar yerleşiktir (§1.4).
Katkı bir **gözlenebilirlik-sınırı sentezi**dir:

1. **Asıl sonuç — yapısal körlük teoremi.** Gerçekçi drift kaynaklarının
   mekânsal-korelasyon spektrumunu, kapalı-yörünge tanı response-matrisinin SVD
   alt-uzaylarına projekte ederiz (§4.4). Sonuç: **koherent (uzun-korelasyonlu)
   yer-hareketi, tam da orbit-kör simetrik alt-uzaya düşer** — komşu mıknatısları
   birlikte kaydırdığı için. Yani kapalı-yörünge tanısı, bedava/girişimsiz
   olmasına karşın koherent drifte **yapısal olarak kördür**; yalnız korelasyonsuz
   magnet-özel drifti görür.
2. **EDM kör-noktası bağı.** Bu kör alt-uzay, depolama-halkası EDM aramalarında
   sahte EDM'yi süren simetrik artıkla çakışır; dolayısıyla orbit-tanı o artığı
   bastıramaz, spin-tabanlı yöntemleri (CR ayrımı, SBA) **tamamlar, yerini tutmaz**.
3. **Nicel karakterizasyon (destekleyici).** Yöntemin geçerlilik alanı
   (antisimetrik/korelasyonsuz drift, 6–7 μm RMS) ve kör noktası (simetrik/koherent,
   $\sim 193\times$ gürültü) sistematik simülasyonla ve tam parçacık+spin
   izleyicisiyle doğrulanmış analitik $R$ ile nicelenir (§2–§3).

Yani özgünlük yeni *matematikte* değil; **bilinen yer-dalgası bastırma fiziğini
(Rossbach 1989) ve SVD null-space'i (Chung 1993) bir *tanı-gözlenebilirlik körlüğü*
ifadesine sentezleyip EDM kör-noktasına bağlamakta**dır.

### 1.4 Önceki çalışmalarla ilişki (dürüst konumlandırma)

Bu çalışmanın yapı taşları yerleşik literatürdür; katkı onların **sentezi ve
yeni bir gözlenebilirlik-diline aktarımı**dır. Açıkça:

- **Ofset-iptalli diferansiyel orbit + R⁻¹ rekonstrüksiyon:** yavaş-orbit
  geribeslemesi (SOFB) ve LOCO'nun [Safranek 1997] standart matematiği; mıknatıs
  hatasını kapalı yörüngeden pasif/ML ile çıkarma da güncel [Huang 2022; Xu 2025].
  Bizim yeniden-tanımımız (mutlak değil *drift*) bir çerçeveleme farkıdır.
- **Response-matrisinin SVD'si, koşulluluk ve null-space:** Chung, Decker & Evans
  1993 (orbit düzeltme bağlamında); Mirza vd. 2019 (simetrik/circulant ORM);
  Wegscheider/Vilsmeier vd. 2023 (ORM ters-modellemede degeneracy — quadrupol
  *gücü* için). Kaçıklık→orbit matrisi, corrector→orbit matrisiyle yapısal olarak
  aynıdır; dolayısıyla SVD koşulluluk/null-space **analizi özgün değildir.**
- **Mekânsal-frekans ↔ tune rezonansı ve uzun-dalga bastırması:** Rossbach 1989,
  FODO'da yer-dalgalarının COD'unu analitik hesaplar; uzun dalga (koherent) →
  küçük COD, rezonans mN±Q. Kazanç yasası $G_k\propto 1/|Q^2-k^2|$ ders kitabı /
  [Khan 2017]. Simetrik/antisimetrik mod ayrımı FODO'da daha önce (demet zarfı
  için) [Tiefenback 1985]. Yani **bu fizik bilinmektedir.**
- **Yer-hareketi modeli (ATL):** Shiltsev 1995; Parkhomchuk vd. 1994.
- **EDM sistematiği:** sahte EDM ∝ $dx\cdot dy$ geometrik faz ve simetrik kaçıklığın
  rolü [Hacıömeroğlu & Semertzidis 2017]; simetrik-hibrit tasarım, CR ayrımı,
  SBA, SQUID-BPM [Omarov vd. 2022]. **EDM bağlantısı da kolaborasyonun mevcut
  işidir.**

**Özgün kalıntı (residue):** Rossbach'ın *bozulma-büyüklüğü* sonucu, açıkça bir
*tanı gözlenebilirliği* ifadesine çevrilmemiştir; kimse drift-kaynağı korelasyon
spektrumunu tanı matrisinin **null-space'ine projekte edip** "koherent drift
yapısal olarak ölçülemez" teoremini kurmamış, ne de bunu EDM kör-noktasıyla
açıkça eşlememiştir. Katkımız bu sentez ve negatif-sonuç ifadesidir.

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
belirsizliğini temsil eden bir **model parametresidir**.

Tek bir incelik yatay düzlemdedir. Halkadaki elektrostatik ark deflektörleri
yatay düzlemde demeti hafifçe **odaklar**; bu odaklamanın gücü $K_{x,\text{arc}}$,
$R$'yi doğru kurmak için gerekir ama temiz bir analitik kapalı formu yoktur.
Onu **tune-eşleme** ile belirleriz: analitik modelin ürettiği yatay tune $Q_x$,
tam izleyicinin verdiği $Q_x$ değerine eşitlenene kadar tek serbest parametre
$K_{x,\text{arc}}$ ayarlanır (bir kök-bulma). Dikey düzlemde bu serbestlik yoktur:
Maxwell gereği elektrostatik deflektörün orta düzlemde dikey alanı sıfırdır
($E_z=0$), yani arklar dikeyde odaklamaz ve $K_{y,\text{arc}}=0$ — kesin,
kalibrasyonsuz.

Bir model parametresini, sonra karşı sınayacağımız simülasyona uydurmak bir
**ters-suç (inverse crime)** riski taşır: uyumu yapay olarak şişirebilir. Bunu
§3.3'te açıkça denetliyoruz — $K_{x,\text{arc}}$'ı $\pm\%10$ bozup drift
kurtarımının neredeyse hiç değişmediğini ($<0.5\,\mu$m) gösteriyoruz; yani sonuç
bu kalibrasyonun tam değerine bağlı değildir.

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

**(ii) İki-gradient $\Delta R^{-1}$ (ofset-iptal).** İki farklı gradient
ayarında ölçüm alalım: $g_1$ ve $g_2=g_1(1+\varepsilon)$, tepki matrisleri
$R_1,R_2$. Aynı kaçıklık $\Delta q$ ve aynı ofset $\mathbf b$ için
$$
\mathbf y_1=R_1\Delta q+\mathbf b+\eta_1,\qquad
\mathbf y_2=R_2\Delta q+\mathbf b+\eta_2.
$$
Fark alınca ofset düşer ve
$\mathbf y_1-\mathbf y_2=\Delta R\,\Delta q+(\eta_1-\eta_2)$, $\Delta R\equiv R_1-R_2$;
böylece $\widehat{\Delta q}=\Delta R^{-1}(\mathbf y_1-\mathbf y_2)=
\Delta q+\Delta R^{-1}(\eta_1-\eta_2)$. Sorun, $\Delta R$'nin **küçük** olmasıdır:
$R\propto(KL)\propto g$ olduğundan
$$
\Delta R = R_1-R_2 \approx -\varepsilon\,g\,\frac{\partial R}{\partial g}
\equiv -\varepsilon R',
$$
yani $\Delta R$, $R$ ölçeğinin yalnız $\varepsilon\,(\approx0.02)$ katıdır.
Tersini almak gürültüyü
$$
\bigl\|\widehat{\Delta q}-\Delta q\bigr\|\approx
\sqrt2\,\sigma_\eta\,\|\Delta R^{-1}\| = \frac{\sqrt2\,\sigma_\eta}{\varepsilon}\,
\|R'^{-1}\|
$$
gibi $1/\varepsilon$ ile patlatır. (Dikkat: $1/\varepsilon$ ile büyüyen şey
kondisyon sayısı $\kappa(\Delta R)$ **değil** — o $\approx\kappa(R')$,
$\varepsilon$'dan bağımsız; bizzat gürültü büyütmesi $\|\Delta R^{-1}\|$'dir.)
Pratikte $\|\Delta R^{-1}\|\sim 10^4\times$ (Şekil 1, Şekil 6).

**(iii) $\Delta R^{-1}$ + düzenlileştirme.** (ii)'nin gürültü patlamasını
dizginlemek için düzenlileştirme denenir. **Tikhonov**, ham ters yerine
$(\Delta R^\top\Delta R+\lambda I)^{-1}\Delta R^\top$ kullanır: $\lambda$ terimi
küçük tekil değerlerin patlamasını keser. **TSVD** ise yalnız en büyük $k$ tekil
değeri tutup gerisini sıfırlar. İkisi de gürültüyü azaltır, ama bunu kestirimi
**bozarak** (bias) yapar: zayıf-gözlenen modlar kısmen/tamamen atıldığından
geri çatılan desen gerçeğinden sapar. En iyi ayarda bile $\sim50\,\mu$m'de kalır
(§3.1) — 10 μm hedefin çok üstünde.

**(iv) Drift modu (kalibrasyon-referans) — önerilen.** Tek gradient, iki
*zaman*. $t_0$'da referans $\mathbf y_0=R\Delta q_0+\mathbf b+\eta_0$ kaydedilir;
sonra
$$
\widehat{\delta q}(t)=R^{-1}\bigl(\mathbf y(t)-\mathbf y_0\bigr).
$$
Burada $\mathbf y(t)-\mathbf y_0=R(\Delta q(t)-\Delta q_0)+(\eta(t)-\eta_0)$
(sabit $\mathbf b$ düşer), dolayısıyla
$$
\widehat{\delta q}(t)=\underbrace{\Delta q(t)-\Delta q_0}_{\text{aranan drift}}
+\,R^{-1}\bigl(\eta(t)-\eta_0\bigr).
$$
İki bağımsız gürültü örneği ($\eta(t),\eta_0$; her biri $\sigma_\eta$) toplamda
$2\sigma_\eta^2$ varyans verir; kestirim hatası kovaryansı
$R^{-1}(2\sigma_\eta^2 I)R^{-\top}$, RMS mertebesi $\sqrt2\,\sigma_\eta\|R^{-1}\|$.
Kritik fark (ii) ile: burada terslenen matris **iyi-koşullu $R$** ($\kappa\approx193$),
kötü-koşullu $\Delta R$ değil — o yüzden gürültü $\sim1/\varepsilon$ değil
$\mathcal O(1)$'dir. Yöntem mutlak hizalamayı değil kalibrasyondan beri
**değişimi** verir; mutlak referans dış kaynaktan (LOCO/BBA) gelir (§5).

### 2.4 Neden (ii) çıkmaz ama (iv) çalışır?

(ii) ile (iv) **farklı sorular** çözer; aradaki fark yöntemin tüm hikâyesidir.

**Sorun.** Ofseti yok etmenin "doğal" yolu, eşzamanlı iki ölçümün farkını
almaktır (ii). Ama eşzamanlı iki ölçümde ofseti tam iptal eden *her* lineer,
ön-yargısız estimatör, kaçıklığı sonuçta $\Delta R$ üzerinden geri çatmak
zorundadır — ve $\Delta R$ yapısal olarak $\varepsilon$-küçüktür (§2.3-ii), yani
gürültü $1/\varepsilon$ patlar. Bu, algoritma seçimiyle aşılamaz: ön-yargısızlık
$A_1R_1+A_2R_2=I$ ve tam ofset iptali $A_1+A_2=0$ koşulları birlikte tek bir
çözüme zorlar, $A_1=\Delta R^{-1}$. Düzenlileştirme (iii) bu sınıfın *dışına*
bias ekleyerek çıkar, ama bedeli $\sim50\,\mu$m'dir.

**Simülasyon bunu doğruluyor:** ham $\Delta R^{-1}$ Test 1'de 1865 μm, Test 6'da
$\sim$3000 μm verir; Şekil 6 gürültünün $\propto1/\varepsilon$ patladığını
gösterir. Sınır gerçek ve yapısaldır, sayısal bir aksaklık değil.

**(iv) neden kaçar?** Drift modu ofseti *eşzamanlı* iptal etmeye çalışmaz; iki
farklı **zamanda** ölçer. Sabit ofset zaman farkında kendiliğinden düşer
(algoritmik hüner değil, fiziksel). Geriye terslenen matris **iyi-koşullu $R$**
($\kappa\approx193$), kötü-koşullu $\Delta R$ değildir — bu yüzden gürültü
$1/\varepsilon$ değil $\mathcal O(1)$'dir. Bedeli: yalnız *değişimi* verir,
mutlak hizalamayı değil; ama bizim aradığımız da zaten değişimdir.

---

## 3. Sayısal Deneyler

Tüm testler ortak bir altyapıda koşuldu (Ek A): semplektik izleyici, gerçekçi
BPM gürültü ve ofset modeli, opsiyonel quad/dipol tilt'leri. Hızlandırıcı
parametreleri `params.json`, test parametreleri `drift_monitor/test_params.json`
içindedir.

### 3.1 Test 1 — Düzenlileştirme iki-gradient yöntemini kurtarır mı?

Soru: §2.3-(iii)'teki düzenlileştirme, (ii)'nin gürültü patlamasını 10 μm
hedefine indirebilir mi? Bunu sınamak için ofseti **sıfır** alıyoruz (en lehte
durum: (ii)'nin tek derdi gürültü, ofset değil) ve aynı veride dört estimatörü
karşılaştırıyoruz (Tablo~1). "$y$-korr", geri çatılan desenin gerçek desenle
korelasyonudur (1 = biçim korunmuş, 0 = biçim bozulmuş).

| Estimatör | y-RMS | y-korr | Açıklama |
|---|---|---|---|
| Direct $R^{-1}$ (ofsetsiz) | 3.5 μm | 0.998 | yalnız *referans*: ofset=0 alındığında gürültü tabanı |
| Ham $\Delta R^{-1}$ (ii) | 1865 μm | 0.085 | düzenlileştirmesiz iki-gradient: gürültü patlar |
| Tikhonov (iii) | 53 μm | 0.348 | $\lambda$ ile dengelenmiş; biçim büyük ölçüde kayıp |
| TSVD ($k=3$, iii) | 52 μm | 0.383 | en iyi 3 mod tutulmuş ($k$ ideal seçilmiş üst-sınır) |

İki ders: (a) **düzenlileştirme gürültüyü ~35× bastırır (1865→~52 μm) ama
hedefin hâlâ 5× üstündedir** ve bunu biçimi bozarak yapar (korelasyon 0.998→0.35:
RMS düşer çünkü estimatör zayıf modları "0 tahmin et"meye kayar). (b) Direct
$R^{-1}$ (3.5 μm) burada en iyi görünür ama **operasyonel rakip değildir** —
ofset=0 varsaydığı için gerçek halkada ($\mathbf b\neq0$) 200+ μm'e fırlar
(Test 4); tabloda sadece gürültü tabanı referansı olarak var. Yani hiçbir
iki-gradient varyantı işe yaramıyor; çözüm farklı bir problem kurmaktan (drift
modu) geçiyor.

### 3.2 Test 2 — Uzaysal transfer fonksiyonu ve SVD spektrumu

Bu test, (ii)'nin neden çuvalladığını $R$ ile $\Delta R$'nin singüler-değer
spektrumlarını yan yana koyarak gösterir (Şekil 1). $\Delta R$'nin tekil
değerleri, $R$'ninkilerin kabaca **$\varepsilon$ katı** çıkar (empirik:
$\varepsilon=0.02$'de büyük modlarda oran $\sim0.02$) — §2.3-(ii)'deki
$\Delta R\approx\varepsilon R'$ beklentisiyle uyumlu.

Ama bir incelik var: **en küçük birkaç mod bu $\varepsilon$ çizgisinin de altına
"çöker".** Bunun anlamı: $\Delta R=R(g_1)-R(g_2)$ farkında, bazı modlar için
$R(g_1)$ ile $R(g_2)$ neredeyse birbirini götürür; o modların $\Delta R$ tekil
değeri $\varepsilon\,\sigma(R)$'nin bile altına iner (uniform $\varepsilon$
ölçeklemesinden daha hızlı küçülür). Sonuç: $\sigma_{\min}(\Delta R)$ ekstra
küçük → $\kappa(\Delta R)$ daha da büyür ($\sim10^4$, $R$'den $\sim$2 mertebe
kötü).

$\varepsilon$ taraması (Şekil 6) asıl ölçeklemeyi netleştirir: ofset-iptal eden
estimatörün gürültü büyütmesi $\|\Delta R^{-1}\|=1/\sigma_{\min}(\Delta R)$ tüm
$\varepsilon\in[0.005,0.10]$ aralığında temiz biçimde $\propto1/\varepsilon$
patlarken, kondisyon sayısı $\kappa(\Delta R)$ kabaca **sabit** ($\sim10^4$)
kalır. Yani $\varepsilon\to0$'da yöntemi kullanılamaz kılan $\kappa$ değil,
gürültü büyütmesi $\|\Delta R^{-1}\|$'dir — §2.4'teki argümanın sayısal
doğrulaması. (Ayrıca düzenlileştirilmiş estimatörler yüksek-$k$ modları söndürür:
48 modun yalnız 3–5'i geri çatılabilir; biçim bozulmasının kaynağı budur.)

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

**Peki gerçek quad tilt bu varsayımı kırar mı?** Kuadrupol dönmesi (tilt) gerçek
bir skew bileşeni yaratır ve $dx\to$ dikey, $dy\to$ yatay yörünge çapraz
terimlerini açar. Bunu doğrudan test ettik: rastgele quad tilt'lerle **kuplajlı**
tam tepki matrisini (dört blok $R_{yy},R_{yx},R_{xy},R_{xx}$) izleyiciden kurup,
düzlem-ayrık monitörle ($R_{yy}^{-1},R_{xx}^{-1}$) drift kurtarımı yaptık
(Tablo~7). Sonuç: gerçekçi 0.2 mrad tilt'te çapraz kuplaj diyagonal tepkinin
yalnız **%0.33'ü**; drift takip hatası **değişmez** (6.27 μm). 1 mrad'a kadar
(kuplaj %1.1) hata yine sabit. Yani çapraz kuplaj, drift modunun ölçtüğü *değişime*
$\sim$(kuplaj)$\times$(drift) $\approx 0.1\,\mu$m katkı verir — 6 μm tabanının
çok altında. **Düzlem-ayrık model gerçekçi quad tilt altında geçerlidir.**

| Quad tilt | $x\!-\!y$ kuplajı $\|R_{yx}\|/\|R_{xx}\|$ | $y$-takip | $x$-takip |
|---|---|---|---|
| 0 | %0.00 | 6.27 μm | 5.95 μm |
| **0.2 mrad** | **%0.33** | **6.27 μm** | **5.95 μm** |
| 1 mrad | %1.08 | 6.26 μm | 5.95 μm |

*(Tablo 7. İzleyiciden kurulan kuplajlı R; düzlem-ayrık drift kurtarımı,
15-tohum medyanı. `drift_monitor/drift_quadtilt_sim.py`)*

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

### 3.8 Test 9 — BPM kazanç hataları

Gerçek BPM'ler bir kazanç hatasıyla okur: $y_{\text{ölç},i}=(1+g_i)\,y_{\text{gerçek},i}
+ b_i + \text{gürültü}$, $g_i$ per-BPM kalibrasyon hatası (tipik %1–2). Drift modu
zaman farkı aldığından sabit ofset $g$'den bağımsız yine iptal olur; geriye
çarpımsal bir model uyumsuzluğu kalır: $\widehat{\delta q}=R^{-1}\,\text{diag}(1{+}g)\,
R\,\delta q = \delta q + R^{-1}\text{diag}(g)R\,\delta q$. Per-BPM $g\sim N(0,\sigma_g)$
ile tarama (Şekil 9):

| $\sigma_g$ | y-takip |
|---|---|
| 0 | 5.7 μm |
| %1 | 6.0 μm |
| **%2 (tipik)** | **6.4 μm** |
| %5 | 8.6 μm |
| %10 | 13.4 μm |

Tipik %2 kazanç hatasında taban yalnız 0.6 μm artar (5.7→6.4 μm); %5'e kadar
hedefin altında. Yöntem gerçekçi BPM kazanç hatalarına dayanıklıdır.
(`drift_monitor/drift_gain_sim.py`)

### 3.9 Test özeti

| Test | Soru | Ana sayı |
|---|---|---|
| 1 | Düzenlileştirme $\Delta R$'yi kurtarır mı? | 1865→52 μm, ama direct 3.5 μm |
| 2 | Düzenlileştirme nasıl çuvallıyor? | 48 modun 43–45'i siliniyor |
| 3 | Yatay modelde ters-suç? | $\pm10\%$ → <0.5 μm |
| 4 | Drift modu ofseti tolere eder mi? | 197 μm → 6.6 μm (29×) |
| 5 | Ofset kayarsa? | $<2\,\mu$m/epoch'a kadar üstün |
| 6 | Aday yöntemler yan yana? | $\Delta R$: 1000–3700 μm, drift: 6–7 μm |
| 8 | β-beating? (gerçek gradyan/focal hatası) | %1→6.1 μm; %5→8.6 μm |
| (tilt) | quad tilt / x–y kuplajı? | 0.2 mrad: kuplaj %0.33, takip değişmez |
| 9 | BPM kazanç hatası? | %2→6.4 μm; %5→8.6 μm; hedef altında |

---

## 4. Gözlenebilirlik Sınırı

§3'te drift modunun bir **gürültü tabanına** ($\sim6$–7 μm, Test 4) oturduğunu
gördük: 1 μm BPM gürültüsü $R^{-1}$'den geçerken $\sqrt2\,\sigma_\eta\|R^{-1}\|$
mertebesinde bir hataya büyür (§2.3-iv). Bu bölüm o tabanın **rastgele bir sayı
olmadığını**, doğrudan $R$'nin mod yapısından geldiğini gösterir: $R^{-1}$'in
gürültüyü hangi yönde ne kadar büyüttüğü, hangi hizalama desenlerinin iyi/kötü
çözüldüğünü belirler. Somut olarak, 6–7 μm'lik tabana en büyük katkı, $R$'nin en
küçük tekil değerli (en kötü koşullu) modlarından gelir — ve aşağıda bu modların
**simetrik** desenler olduğunu göreceğiz. Yani §3'teki taban ile §4'teki kör
nokta aynı olgunun iki yüzüdür: monitör simetrik driftleri zayıf çözer, bu da
hem gürültü tabanını yükseltir hem de geçerlilik alanını antisimetrik driftle
sınırlar. (Bu karakterizasyon tümüyle monitörün kendi özelliğidir; herhangi bir
fiziksel sistematik bütçesinden bağımsızdır.)

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

Hangi desenin hangi harmoniğe düştüğünü görmek için her hücredeki (QF, QD)
çiftini iki bileşene ayırırız:
$$
s_c = \tfrac12(q_{QF,c}+q_{QD,c})\ \text{(simetrik)},\qquad
d_c = \tfrac12(q_{QF,c}-q_{QD,c})\ \text{(antisimetrik)},
$$
öyle ki $q_{QF,c}=s_c+d_c$ ve $q_{QD,c}=s_c-d_c$. Her gerçek kaçıklık — genelde
ne saf simetrik ne saf antisimetrik — bu iki bileşene **tek türlü** ayrılır.
*Örnek:* bir hücrede QF $=3$, QD $=1$ ise $s=2,\ d=1$; yani kayma
$(2,2)_{\text{sim}}+(1,-1)_{\text{antisim}}$ olarak yazılır. (48-boyutlu desen
uzayı böylece 24 simetrik + 24 antisimetrik genlikten oluşan iki dik **aileye**
ayrılır; "her hata = iki *aileden* birer bileşen" — iki sabit desenin toplamı
değil.)

Bir desenin ne kadar simetrik olduğunu **tek bir sayıyla** ölçeriz:
$$
\chi \equiv \frac{\|s\|^2-\|d\|^2}{\|s\|^2+\|d\|^2}\in[-1,+1],
$$
$\chi=+1$ tümüyle simetrik, $\chi=-1$ tümüyle antisimetrik, $\chi\approx0$
karışık. (Spin $S$, singüler değer $\sigma$ veya tepki $R$ ile karışmasın diye
$\chi$ harfi.)

**Belirleyici ayrıntı gradyan işaretidir:** QF odaklar (gradyan $+$), QD dağıtır
($-$); kick $=$ gradyan $\times$ kayma olduğundan kick dizisi $k_j=g(-1)^j q_j$.
Buradan:

- **Antisimetrik bileşen** ($+a/-a$): kickler **aynı işarette** → hücreden
  hücreye düzgün, **düşük-$k$** → $G_k$ büyük → **yörüngede görünür**.
- **Simetrik bileşen** ($+a/+a$): kickler **zıt işarette** → hücre-içi alternatif,
  **yüksek-$k$** → $G_k$ küçük → **neredeyse görünmez**.

$(-1)^j$ çarpanı (Nyquist harmoniği, $k=24$) spektrumu 24 kaydırır: antisimetrik
bileşen $k\in[0,12]$, simetrik bileşen $k\in[12,24]$ bandına düşer (üniform
simetrik desen $k=24$ özel durumu). Tümü $k\gg Q$ olduğundan simetrik içerik
bastırılır. Sezgi: simetrik kayma kendi kendini söndüren bir kick örüntüsü
üretir; kapalı yörünge bunu görmez, antisimetriği güçlü görür.

### 4.3 Hangi desenler ne kadar gürültüyle kestirilir? (per-mod SVD)

§4.2'deki simetrik/antisimetrik ayrım kategorikti (iki uç durum). Tepki
matrisinin **tekil değer ayrışımı (SVD)** bunu nicel ve sürekli hale getirir.
$R=U\Sigma V^\top$ yazıldığında her sağ-tekil vektör $V_i$ bir hizalama deseni
("mod"), karşılık gelen tekil değer $\sigma_i$ ise o desenin yörüngede ne kadar
güçlü göründüğüdür. Drift monitör bir modu kestirirken gürültüyü $1/\sigma_i$
ile büyütür: büyük-$\sigma$ modlar temiz, küçük-$\sigma$ modlar gürültülü
kestirilir. Her mod için (i) gürültü büyütmesi $1/\sigma_i$ ve (ii) modun §4.2
anlamında **simetrik içeriği** ($\chi_i$; modun gücünün simetrik bileşendeki
oranı) hesaplanır:

$$
\sigma_{\max}=28.4,\quad \sigma_{\min}=0.147,\quad \kappa(R)=193.
$$

| Mod | $\sigma$ | Gürültü büyütme ($1/\sigma$) | Simetrik içerik $\chi_i$ |
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

**Bu sıralamanın fiziksel kökeni.** Modların simetrik içeriğinin $\sigma$
küçüldükçe artması tesadüf değil, latisin periyodikliğinin doğrudan sonucudur.
Halka 24 özdeş FODO hücresinden oluştuğundan $R$ (yaklaşık) periyodiktir; cebirsel
sonucu, singüler modlarının uzaysal harmonikler olmasıdır (sayısal olarak modların
$\gtrsim$%96'sı tek harmonikten ibaret). Böylece her modu bir kick-harmoniği $k$
ile etiketleyebiliriz. Kapalı yörüngenin harmonik-$k$ bir kick'e tepkisi Hill
denkleminin rezonans paydasıyla ölçeklenir, $\sigma\sim 1/|Q_{\rm eff}^2-k^2|$
(§4.2'deki kazanç yasası $G_k$ ile aynı iskelet): betatron tune'una yakın
($k\approx Q\approx2.3$) kick'ler yükseltilir (büyük $\sigma$), uzak ($k\gg Q$)
olanlar bastırılır (küçük $\sigma$). Simetri buraya şöyle bağlanır: hücre içinde
aynı yönlü (simetrik) bir kaçıklık, odaklayıcı/odaksızlaştırıcı kuadrupol
diziliminin $(-1)^j$ işaret değişimi yüzünden **alternatif** bir kick (yüksek-$k$,
$k\!\to\!24$) üretir — tune'dan uzak, dolayısıyla küçük $\sigma$ ve yüksek
gürültü. Antisimetrik kaçıklık ise düşük-$k$ (tune'a yakın) bir kick verdiğinden
büyük $\sigma$ ile iyi ölçülür. Şekil 4'teki monoton tırmanışın mekanizması budur.
(Bu ölçekleme periyodiklikten gelen yapısal bir özelliktir; mod-mod düzeyinde
$\sigma/G_k$ oranı $\sqrt\beta$ ve $KL$ ağırlıkları yüzünden $\sim$1.2–3.4
arasında değişir — kesin bir eşitlik değil, doğru bir iskelettir.)

### 4.4 Gerçek drift kaynakları hangi alt-uzaya düşer?

§4.1–4.3 gözlenebilirliği soyut kaçıklık desenleri (simetrik/antisimetrik)
cinsinden verir. Operasyonel soru şudur: **gerçek drift kaynakları bu
alt-uzaylara nasıl projekte olur?** Bir FODO hücresindeki QF ve QD fiziksel
komşudur; bir drift kaynağı bu ikisini **aynı yönde** kaydırırsa simetrik
(kör), **zıt yönde** kaydırırsa antisimetrik (görünür) içerik üretir. Mekânsal
korelasyon uzunluğunu ($L$, quad-aralığı biriminde) tarayarak rastgele drift
alanlarının simetrik güç payını ölçtük:

| Korelasyon uzunluğu $L$ | Simetrik (kör) güç |
|------|------|
| 0 (korelasyonsuz) | %51 (50/50) |
| 1 quad | %89 |
| 2 quad | %97 |
| ≥4 quad | ~%100 |

Sonuç keskin: **bir quad-aralığından uzun her mekânsal korelasyon, drifti ezici
biçimde simetrik (kör) yapar.** Bunun fiziksel okuması:

- **Koherent yer-hareketi / global termal** (korelasyon uzunluğu $\gg$ hücre):
  komşu mıknatısları birlikte kaydırır → **simetrik → monitöre büyük ölçüde
  görünmez.**
- **Magnet-özel korelasyonsuz drift** (güç kaynağı, mesnet, soğutma; ayrıca QF
  ve QD'nin farklı mıknatıs tipleri olması): quad-quad bağımsız → **antisimetrik
  içerik → görünür.**

Dolayısıyla yöntemin etkin kapsamı **drift kaynağı karışımına** bağlıdır:
korelasyonsuz/magnet-özel drifti izler, koherent yer-hareketine kördür. (Not:
sahte-EDM açısından görünür antisimetrik kısım birim başına daha çok katkı
verir; ancak rutin orbit düzeltmesi açıkken antisimetrik drift zaten düzelticiler
tarafından telafi edilir ve kalan sahte-EDM driftini kör simetrik kısım sürer —
bu ayrım §4.5 ve ayrı çalışmaya bırakılır.) Bu, yöntemin **geçerlilik alanını**
dürüstçe sınırlar: bir tesis, drift kaynağı spektrumunu bilerek yöntemin o
ortamda ne kadar işe yarayacağını öngörebilir.

### 4.5 Kör noktanın anlamı ve sistematik bütçeyle ilişkisi (ileri bakış)

§4.1–4.3'ün sonucu **yöntem düzeyinde** kesin bir ifadedir: kapalı-yörünge
drift monitörü antisimetrik hizalama driftini güçlü, simetrik driftini ise
$\sim 193$ kat daha gürültülü çözer. Bu, monitörün geçerlilik alanını
(antisimetrik) ve kör noktasını (simetrik) tanımlar ve daha iyi BPM donanımı
ya da daha fazla veriyle kapanmaz; bu alt-uzaya erişim ilkesel olarak **farklı
bir gözlemlenebilir** (ör. karşı-dönen CW/CCW demet ayrımı veya doğrudan spin
presesyonu [Omarov 2022]) gerektirir.

Bu kör noktanın belirli bir **sistematik bütçe** için ne kadar önemli olduğu
ayrı bir sorudur. pEDM bağlamında ilgili sistematik, kaçıklığın ürettiği sahte
EDM'dir; baskın kanalı yatay×dikey kaçıklığın çarpımına ($dx\cdot dy$) bağlı,
misalignment'ta ikinci-dereceden bir geometrik-faz etkisidir [Omarov 2022].
Bu kanalın ağırlığının monitörün zayıf-gözlediği simetrik alt-uzaya ne kadar
düştüğü desen-bağımlı, inceliklidir ve tek bir kapalı-yörünge ölçümüyle
çözülemez. Bu nedenle sahte-EDM ↔ gözlenebilirlik bağını niceleme işini —
tam parçacık + spin izleyicisiyle, orbit-düzeltme öncesi/sonrası ayrımıyla —
**bu çalışmanın kapsamı dışında, ayrı bir incelemeye** bırakıyoruz (ön
sonuçlar `drift_monitor/` altındaki keşif betiklerinde). Bu makalenin iddiası
yöntem düzeyinde kalır: monitör, antisimetrik hizalama driftini ucuza ve
sürekli izleyen, geçerlilik alanı ve kör noktası net tanımlı bir araçtır.

---

## 5. Önerilen İşletme Modu

Tek bir yöntem her ihtiyacı karşılamaz; iki katmanlı bir mimari öneriyoruz
(Şekil 5).

**Yavaş mutlak katman (saatlik–günlük).** LOCO + BBA + survey ile mutlak
hizalama $\Delta q_0$, BPM ofseti $\mathbf{b}_0$ ve örgü modeli ($\beta,\phi,Q$,
dolayısıyla $R$). Bu, hızlandırıcı operasyonunda zaten var olan standart
prosedürdür.

**Hızlı drift katmanı (sürekli, saniye–dakika).**
$\widehat{\delta q}(t)=R^{-1}(\mathbf{y}(t)-\mathbf{y}_0)$ — matematiği standart
SOFB/LOCO ile ortaktır (özgünlük burada değil; §1.4). Fizik run'ı boyunca sürekli
çalışır, veri toplamayı bozmaz, kalibrasyondan beri **antisimetrik** hizalama
değişimini izler; ama §4'ün gözlenebilirlik sınırı gereği koherent/simetrik drifti
göremez.

İki katman tamamlayıcıdır: yavaş katman mutlak referansı ve $R$'yi sağlar,
hızlı katman o referansa göre değişimi takip eder. §4'ün sınırı gereği, her
iki katman da simetrik alt-uzayı kapalı yörüngeden kurtaramaz; bu bilgi dış
bir gözlemlenebilirden (demet ayrımı / spin) gelmelidir.

---

## 6. Tartışma ve Sonuç

**Asıl sonuç bir gözlenebilirlik sınırıdır, yeni bir yöntem değil.** Kapalı-yörünge
tabanlı hizalama-drift tanısının — mevcut BPM'lerle, girişimsiz, pasif — yapısal
bir kör noktası olduğunu gösterdik ve niceledik: tanı response-matrisinin SVD
alt-uzayları, hangi drift desenlerinin görünür hangilerinin kör olduğunu belirler;
ve **gerçekçi drift kaynaklarının mekânsal-korelasyon spektrumu bu alt-uzaylara
projekte edildiğinde, koherent (uzun-korelasyonlu) yer-hareketi tam da kör simetrik
alt-uzaya düşer** (§4.4). Yani tanı, korelasyonsuz magnet-özel drifti hassasça
izler (geçerlilik alanı; §3'te 50 μm ofset + %1 β-beating altında 6–7 μm RMS,
tam parçacık izleyicisiyle doğrulanmış $R$ ile), ama koherent yer-hareketine
**yapısal olarak kördür** — bu körlük donanım çözünürlüğüyle (daha iyi BPM) değil,
ancak farklı bir gözlemlenebilirle aşılır.

**Önceki çalışmalarla dürüst ilişki (bkz. §1.4).** Bu sentezin yapı taşları
yerleşiktir ve özgün değildir: ofset-iptalli diferansiyel orbit + $R^{-1}$
rekonstrüksiyon SOFB/LOCO matematiğidir [Safranek 1997; Huang 2022]; response
matrisinin SVD'si, koşulluluğu ve null-space'i 1990'lardan beri kanoniktir
[Chung 1993; Mirza 2019; Wegscheider 2023]; mekânsal-frekansın tune ile rezonansı
ve uzun-dalga (koherent) bastırması analitik olarak Rossbach 1989'da, kazanç yasası
$G_k\propto 1/|Q^2-k^2|$ ders kitabında [Khan 2017] vardır; ATL yer-hareketi modeli
[Shiltsev 1995] ve simetrik/antisimetrik FODO mod ayrımı [Tiefenback 1985] de
bilinmektedir. **Özgün kalıntı:** Rossbach'ın *bozulma-büyüklüğü* sonucunu, drift-
kaynağı korelasyon spektrumunun tanı matrisinin null-space'ine açık projeksiyonuyla
bir *tanı-gözlenebilirlik körlüğü* teoremine çevirmek ve bu kör alt-uzayın hassas
(EDM) halkalardaki sistematik kör noktayla çakıştığını göstermek.

**EDM bağlamı.** Depolama-halkası EDM aramalarında sahte EDM, kaçıklığın
$dx\cdot dy$ geometrik-faz kanalından gelir [Hacıömeroğlu & Semertzidis 2017;
Omarov 2022] ve düzeltme-sonrası simetrik artıkla ilişkilidir. Tanının kör alt-uzayı
tam da bu artıkla çakıştığından, orbit-tabanlı izleme onu bastıramaz; bu, srEDM
kolaborasyonunun CR demet ayrımı, spin-tabanlı hizalama ve SQUID-BPM'e yönelmesinin
zaten bilinen gerekçesidir [Omarov 2022; Anastassopoulos 2016]. Çalışmamız bu kör
noktaya *yörünge tarafından* kesin bir gözlenebilirlik sınırı koyar.
(İleri yönde açık bir kanal: ayrı bir incelemede [`berry.md`], sahte EDM'nin
*ölçülen yörüngenin* alan-ağırlıklı bilineer bir fonksiyoneli olarak — inversiyon
gerektirmeden — öngörülebildiğine dair ön kanıt bulduk; bu, kör alt-uzayı *ileri-
ölçümle* kısmen geri kazanma ihtimalini açar ama analitik türetme gerektirir.)

Sağlamlık, başlıca lineer-olmayan/model hata kanalları için açıkça test edildi:
BPM kazanç hataları (Test 9, %2→6.4 μm), kuadrupol tilt'inin yarattığı x–y skew
kuplajı (Tablo 7, 0.2 mrad'da etkisiz) ve β-beating/focal-uzunluk hataları
(Test 8). Üçünde de tanı geçerlilik alanı içinde hedefin altında kalır.

Çalışmanın kalan kısıtları:

- **Model ötesi diğer etkiler.** Sonuçlar Eş. (1) lineer modeline dayanır;
  test edilmeyen ikincil kanallar arasında sekstupol feed-down, fringe alanlar,
  manyetik histerezis ve akım dalgalanması var. Bunlar gelecek çalışmadır, ama
  test edilenler (gain, skew-kuplaj, β-beating) bu sınıfın baskın üyeleridir.
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
   [LOCO; parametre null-space]
5. **[chung1993]** Y. Chung, G. Decker, K. Evans Jr., "Closed orbit correction
   using singular value decomposition of the response matrix," *Proc. PAC 1993*,
   p. 2263. [SVD orbit-düzeltme; mode-space, decoupled/gözlenemez modlar]
6. **[mirza2019]** S. H. Mirza, R. Singh, H. Klingbeil, P. Forck, "Closed orbit
   correction at synchrotrons for symmetric and near-symmetric lattices,"
   *Phys. Rev. Accel. Beams* **22**, 072804 (2019). arXiv:1902.08683.
   [circulant ORM, DFT/Fourier mode-space]
7. **[wegscheider2023]** D. Vilsmeier, R. Singh, M. Bai, "Inverse modeling of
   circular lattices via orbit response measurements in the presence of
   degeneracy," *Phys. Rev. Accel. Beams* **26**, 032803 (2023).
   [ORM ters-modelleme degeneracy/null-space — quadrupol gücü]
8. **[rossbach1989]** J. Rossbach, "Closed-orbit distortions of periodic FODO
   lattices due to plane ground waves," *Particle Accelerators* **23**, 121
   (1989). [yer-dalgası → COD; uzun dalga bastırma, rezonans mN±Q]
9. **[shiltsev1995]** V. Shiltsev, "Space-time ground diffusion: the ATL law for
   accelerators," *Proc. IWAA 1995* / SSCL-Preprint-470. [ATL yasası]
10. **[parkhomchuk1994]** V. Parkhomchuk, V. Shiltsev, G. Stupakov, "Slow ground
    motion and operation of large colliders," *Phys. Rev. E* **49**, 3980 (1994).
    [korelasyon → COD; tam-korele hareket sıfır bozulma]
11. **[khan2017]** S. A. Khan vd., harmonik/DFT kapalı-yörünge analizi (SIS18),
    $\sigma_k = Q/(\pi|Q^2-k^2|)$ kazanç yasası. [gain yasası]
12. **[tiefenback1985]** M. G. Tiefenback, FODO örgüsünde simetrik/antisimetrik
    mod ayrışımı (demet zarfı), LBL preprint (1985).
13. **[huang2022]** X. Huang vd., "Simultaneous beam-based alignment measurement
    for multiple magnets," arXiv:2203.14869 (2022). [modülasyonsuz çoklu-quad BBA]
14. **[haciomeroglu2017]** S. Hacıömeroğlu, Y. K. Semertzidis, "Systematic errors
    related to quadrupole misplacement in an all-electric storage ring for proton
    EDM experiment," arXiv:1709.01208 (2017). [sahte EDM ∝ dx·dy; simetrik kaçıklık]

> **Kaynak notu:** Tam künyeler/sayfa numaraları yayın öncesi doğrulanacak;
> [khan2017], [tiefenback1985] deep-research ile bulundu, birincil kaynaktan
> teyit edilmeli. İlgili PDF tam metinleri `literatur/` altında.

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
(Şekil 1–4, 6, 8), `drift_monitor/make_fig5_architecture.py` (Şekil 5),
`drift_monitor/theory_sim_validate.py` (Şekil 7) ve
`drift_monitor/drift_gain_sim.py` (Şekil 9). Çok panelli şekiller dikey
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

**ŞEKİL 9.** (`fig9_bpm_gain.png`) Test 9: BPM kazanç hatası RMS $\sigma_g$
altında drift takip hatası (30-tohum medyanı, analitik $R$). Kesik kırmızı
10 μm hedef, noktalı gri tipik %2 seviyesi. Sabit ofset gain'den bağımsız iptal
olduğundan etki yalnız çarpımsal model uyumsuzluğudur; %5'e kadar hedef altında.
(§3.8)
