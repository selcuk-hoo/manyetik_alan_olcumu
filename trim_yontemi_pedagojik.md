# Sahte EDM'i Kaynağında Bastırmak — Ders Kitabı Tarzı Anlatım

Bu belge, proton EDM (pEDM) deneyinde kuadrupol hizalama hatalarının yarattığı
**sahte EDM** sinyalini nasıl ölçüp bastırdığımızı sıfırdan anlatır. Hiçbir ön
bilgi varsaymaz; her kavram ilk kullanıldığında tanımlanır. Teknik analiz günlüğü
`false_edm_harmonic_sinir.md`'de (§1–14), makale dili `makale_trim_tr.tex`'tedir.

İçindekiler:
1. Sahte EDM nedir ve asıl kaynağı neden **dx·dy**'dir?
2. Hata desenini iki şekilde ayırmak: Fourier modları + simetrik/antisimetrik
3. İki gözlenebilir: yörünge (BPM) ve spin — hangisi neyi görür?
4. Sahte EDM'i temiz ölçmek: dört simetrik parçacık
5. Ölç–trimle döngüsü: tek bir sayıyı sıfırlamak
6. Birinci kademe — yörünge (COD) trimi: ne kadar gider, neden durur?
7. İkinci kademe — spin trimi: yörüngenin göremediğini temizler
8. Büyük resim: iki kademe + CW/CCW iptal zinciri
9. Sık sorulan sorular

---

## 1. Sahte EDM nedir ve asıl kaynağı neden dx·dy'dir?

Proton EDM deneyi, protonun spininin yatay düzlemden düşeye doğru **çok yavaş
dönmesini** arar. Gerçek bir elektrik dipol momenti bu dönmeyi yaratır; ölçtüğümüz
büyüklük dikey spin bileşeninin birikme hızıdır:
$$f \equiv \frac{dS_y}{dt}.$$
Sorun şu: **mıknatıs hizalama hataları da $f\neq0$ üretir** ve bunu gerçek
EDM'den ayırt etmek imkânsızdır. Buna **sahte EDM** denir; deneyin en tehlikeli
sistematiğidir.

### Mekanizma: iki düzlem bir arada

Bir kuadrupol ideal konumundan kayarsa, içinden geçen demete sapmış bir alan
gösterir. Hangi yönde kaydığı kritik:

- **Düşey kaçıklık $dy$** → demet kuadrupolün **radyal** manyetik alanını ($B_x$)
  görür → Thomas–BMT denklemi gereği spin **x-ekseni** etrafında döner. Bu, $S_y$
  ile $S_z$'yi karıştırır.
- **Yatay kaçıklık $dx$** → demet **dikey** alanı ($B_y$) görür → spin **y-ekseni**
  etrafında döner. Bu, $S_x$ ile $S_z$'yi karıştırır ama **$S_y$'ye dokunmaz**.

Buradan ilk sürpriz çıkar: **tek başına yatay kaçıklık dikey spini hiç
değiştirmez** (simülasyonda dx-only $\Rightarrow$ $S_y$ tam sıfır). Tek başına
düşey kaçıklık ise küçük bir etki verir — üstelik halka çevresinde büyük ölçüde
birbirini götürür (birinci derecede iptal).

Asıl büyük sahte EDM, **iki kaçıklık birlikte varken** doğar. Sebebi geometriktir:
x-ekseni etrafında bir dönme ile y-ekseni etrafında bir dönme **sıra
değiştirilemez** (komütatif değildir). Ardışık iki farklı-eksen dönmesi, net bir
üçüncü-eksen dönmesi bırakır — bu, kuantum/klasik mekanikteki **geometrik (Berry)
fazıdır**. Sonuç:
$$f \;\propto\; dx \cdot dy.$$

Pratik anlamı büyüktür. $dx$ ve $dy$ ikisi de tipik RMS $\sigma$ mertebesindeyse,
çarpım $\propto\sigma^2$ — yani sahte EDM hizalama hatasının **karesiyle**
ölçeklenir. (Çok-tohumlu RMS ölçümünde üs $\approx 2.0$ çıkar; Omarov vd.'nin
bildirdiği kuadratik davranışla uyumlu.) Demek ki kontrol etmemiz gereken
sistematik, dikey kaçıklığın tek başına değil, **yatay × dikey** içeriğinin
çarpımıdır.

### Sayısal ölçek

200 μm RMS mekanik hizalamada (her iki düzlem) sahte EDM $\sim 9\times10^{-4}$
rad/s'dir. Hizalama 10 μm'ye indirilirse — kuadratik ölçek gereği — $\sim10^{-4}$
kat düşer. Nihai hedef, CW/CCW + quad-flip iptal zincirinin (§8) giriş gereksinimi
olan $\sim10^{-5}$ rad/s'nin altına inmektir; oradan zincir $10^{-9}$'un altını
halleder.

---

## 2. Hata desenini ayırmanın iki yolu

48 kuadrupolün her birinin yatay ve dikey kayması ayrı birer sayıdır: düzlem
başına 48 bilinmeyen. Bunlarla tek tek uğraşmak yerine deseni **iki tamamlayıcı
biçimde** ayrıştırırız.

### 2a. Fourier modları (halka çevresinde dalga sayısı)

Kayma desenini halka çevresince $k$ kez dalgalanan modlara açarız:
$$\Delta_j \;=\; \sum_k A_k\,\cos\!\Big(\tfrac{2\pi k\,n_j}{N} - \varphi_k\Big),
\qquad N=24\ \text{FODO hücresi}.$$
$k=1$ tek dalga, $k=2$ iki tepe-iki çukur, vb. Bu yararlıdır çünkü halka her moda
eşit tepki vermez (§6'da göreceğiz: dikey betatron tonu $Q_y\approx2.68$'e yakın
modlar yörüngeyi en çok bozar).

### 2b. Simetrik / antisimetrik ayrışım (QF ve QD'nin işareti)

Her FODO hücresinde bir odaklayıcı (QF, gradyanı +) ve bir dağıtıcı (QD,
gradyanı −) kuadrupol vardır. Bir kaçıklık deseni, hücre içindeki QF ve QD'nin
**aynı mı zıt mı** yönde kaydığına göre ikiye ayrılır:

- **Antisimetrik** (QF ve QD zıt yönde): gradyan işaretleri de zıt olduğundan
  iki tekme **üst üste biner** → büyük yörünge bozulması. 25 boyutlu alt-uzay.
- **Simetrik** (QF ve QD aynı yönde): tekmeler **birbirini söndürmeye** çalışır
  → küçük yörünge. 23 boyutlu alt-uzay.

Bu ayrım, belgenin kalbidir: birazdan göreceğimiz gibi **yörünge yalnız
antisimetrik yarıyı görür; simetrik yarıya kördür** — ve dx·dy sahte EDM'i büyük
ölçüde bu görünmez simetrik yarıda yaşar.

---

## 3. İki gözlenebilir: yörünge ve spin

Hizalamayı düzeltmek için iki farklı ölçüm aracımız var; her biri farklı şey görür.

**Yörünge (BPM okuması).** Kaçık kuadrupoller kapalı yörüngeyi bozar; bunu ışın
konum monitörleri (BPM) ölçer. Avantajı: gerçek EDM kapalı yörüngeyi **hiç
bozmaz**, dolayısıyla BPM ölçümü gerçek sinyale yapısal olarak kördür —
yanlışlıkla EDM'i "düzeltip" yok etme riski yoktur. Dezavantajı: BPM'lerin
$\sim100$ μm'lik **statik elektronik ofseti** vardır ve yörünge yalnız
antisimetrik, yüksek-kazançlı modları ofset gürültüsünün üstünde görebilir.

**Spin ($f=dS_y/dt$).** Sahte EDM'in kendisini doğrudan ölçer. Avantajı: **tüm 48
boyutu görür** — simetrik içerik dahil (kuplaj zayıf ama sıfır değil). Dezavantajı:
hem sahte hem gerçek EDM'e duyarlıdır; gerçek sinyali yutmamak için CW/CCW fark
kanalı gerekir (§8), ve ölçüm polarimetre istatistiğiyle sınırlıdır (yavaş).

İki aracın **birbirini tamamlaması** bu belgenin ana mesajıdır: yörünge hızlı ve
EDM-kör ama yarım-kör; spin yavaş ama eksiksiz.

---

## 4. Sahte EDM'i temiz ölçmek: dört simetrik parçacık

$f$'i ölçmenin bir tuzağı var. Tek bir parçacığı halkanın eksenine fırlatırsak,
kaçıklıkların kaydırdığı kapalı yörünge etrafında büyük (mm mertebesinde)
**betatron salınımı** yapar. Bu salınımın $S_y$'ye karışması, aradığımız minik
seküler sürüklenmeyi boğar. Eskiden çözüm parçacığı kapalı yörüngeye "oturtmaktı";
ama buna gerek yok.

Bunun yerine **dört parçacık** göndeririz; eksen etrafındaki transvers ofsetleri
işaretlerin dört kombinasyonudur:
$$(\pm\Delta x,\ \pm\Delta y)\quad\text{(dört bağımsız işaret)},$$
spin hepsinde boylamsal. Dördünün $S_y(t)$'sini ortalarız. Neden işe yarar?
Kafesin yansıma simetrisini örneğe **zorlarız**:

- Betatron salınımı ve istenmeyen $\langle\Delta x\,\Delta y\rangle$ örnekleme
  karışması işaret çevirmeleri altında **tektir** (odd); dört terimde
  $(+,-,-,+)$ toplanıp **tam sıfırlanır**.
- Gerçek sahte-EDM sürüklenmesi (kapalı yörünge çarpımı $x_{\rm CO}y_{\rm CO}$'dan
  gelir) işaret çevirmeleri altında **çifttir** (even); dördünde aynıdır,
  ortalamada **korunur**.

Böylece kapalı yörüngeyi hiç aramadan temiz bir $f$ elde ederiz. Kalan yavaş
salınım, model fitiyle (seküler doğru + sinüzoit bileşenleri) çıkarılır. *(Eski
analizlerde kullanılan "tek parçacık + düz doğrusal fit" yöntemi bu salınımı
seküler sanıp $f$'i mertebelerce şişiriyordu; bu belgedeki tüm güncel sayılar
dört-parçacık + model fit ile alınmıştır — bkz. `false_edm_harmonic_sinir.md`
§13.)*

---

## 5. Ölç–trimle döngüsü

Sahte EDM'i bastırma fikri üç cümlede özetlenir:

1. **Ölç:** $f$'i ölç (yukarıdaki dört-parçacık yöntemiyle).
2. **Trimle:** Kuadrupolleri kaydırıcılarla bilinçli, küçük bir desende oynat;
   genliği $A_{\rm trim} = -f/c$. Burada $c$, o trim modunun **kaldıraç koludur**
   (birim trim başına kaç rad/s ürettiği). Bu, ölçülen sinyali tam götürecek
   "karşı-kirliliktir".
3. **Tekrarla:** Kalanı ölç, gerekirse küçük bir düzeltme daha.

**Neden tek bir mod yeter?** $f$ tek bir **sayıdır** (skaler). Kaldıraç kolu $c$
bilinen tek bir düğmeyi $A_{\rm trim}=-f/c$ kadar çevirmek, $f$'i doğrusal
mertebede **tam sıfırlar** — kalibre bir kadranla bir göstergeyi sıfırlamak gibi.
Hangi kuadrupolün ne kadar kaydığını bilmemize gerek yoktur; sadece ibre
okuması ($f$) ve bir düğmenin kalibrasyonu yeter. Geriye kalan (doğrusal-ötesi
artık + öteki modlar + ölçüm gürültüsü) bir-iki iterasyonda ölçüm tabanına iner.

> **Kalibrasyon hakkında bir not.** $c$ kaldıraç kolu, modu küçük bir bilinen
> genlikle uygulayıp $f$ değişimini ölçerek bulunur; iki kuadratür (cos ve sin)
> ölçülürse modun hem gücü $|c|$ hem etkin fazı $\psi=\mathrm{atan2}(c^{\sin},
> c^{\cos})$ belirlenir ve "ölü faza" (etkisiz yön) düşme riski kalmaz. Bu
> simülasyonda $c$'ler analitik olarak da hesaplanabilir ve küçük kaçıklıklarda
> desenden bağımsızdır; gerçek deneyde de kalibrasyon rutin bir adımdır. Bu
> yüzden burada kalibrasyonu bir **ön hazırlık** olarak anıyoruz, yöntemin özü
> olarak değil — özü, skaler $f$'i $-f/c$ ile nullamaktır.

---

## 6. Birinci kademe: yörünge (COD) trimi

İlk kademe hızlı ve EDM-kördür: BPM yörüngesinden hizalama modlarını kestirip
kaydırıcılarla geri besler, spini hiç okumaz. **Her iki düzlemde** çalışır —
dikey yörüngeden $dy$ modları, yatay yörüngeden $dx$ modları kestirilir (iki
bağımsız 48-boyutlu problem, skew kuplaj ihmal edilir).

### 6a. Hangi modları görebiliriz? Kazanç hiyerarşisi

Bir kuadrupol modunun yörüngeyi ne kadar bozduğu, halka boyunca **betatron
rezonansıyla** belirlenir. Sezgi: her kaçık kuadrupol demete küçük bir tekme
verir; bu tekmeler halka çevresinde dolaşırken üst üste biner. Eğer modun uzaysal
frekansı betatron tonuna ($Q_y\approx2.68$) yakınsa, tekmeler **rezonansla**
büyür — tıpkı bir salıncağı doğru ritimde itmek gibi. Uzaksa, tekmeler birbirini
söndürür.

Sonuç bir **kazanç hiyerarşisidir** $G_k$ (birim mod genliği başına RMS yörünge):

| $k$ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| $G_k$ | 6.2 | **24.1** | 6.3 | 2.3 | 1.2 | 0.8 |

$k=2$, $Q_y$'ye en yakın olduğundan zirvedir. Bu kazançlar analitik bir yasaya
oturur: $G_k = C/|Q_{\rm eff}^2-k^2|$ (nominal kafeste $C=24.8$,
$Q_{\rm eff}^2=5.03$, fit artığı %0.6) — yani bir uydurma değil, kapalı-yörünge
dinamiğinin parametrizasyonudur.

**Kazanç neden fit edilebilirliği belirler?** BPM'lerin $\sigma_b\approx100$ μm
statik ofseti vardır. Bir modu yörüngeden kestirirken bu ofset, mod kestirimine
bir **yanlılık** olarak sızar; büyüklüğü kabaca $\sigma_b/G_k$'dır (kazanç böler).
Yüksek-kazançlı $k=2$ için yanlılık $\sim$μm-altı — zararsız. Düşük-kazançlı
$k\geq6$ için yanlılık, modun gerçek içeriğinden büyük olur; o modu fit etmek
düzeltmek yerine **ofset hayaletini sisteme enjekte eder**. Kapalı-form eşik:
$$G_k > \frac{\sigma_b}{\sigma_q}\quad\Longleftrightarrow\quad
k_{\max}^2 < Q_{\rm eff}^2 + C\,\frac{\sigma_q}{\sigma_b}.$$
$\sigma_b=\sigma_q$ için $k_{\max}\approx5.5$: $k\leq4$ güvenle fit edilir, $k=5$
sınırda, $k\geq6$ zararlı. Yani **kazanç hiyerarşisi doğal bir düzenleyicidir**:
fazla mod fit etmeye çalışmak sonucu kötüleştirir. (Sayısal teyit: $k\leq4$ fit
eden varyant en iyi; $k\leq6$ fit eden, düşük-kazançlı modlara ofset enjekte
ederek daha kötü sonuç verir.)

### 6b. Yörünge tek başına çözer mi? Hayır — simetrik taban

Burada belgenin ana fizik sonucu gelir. dx·dy sahte EDM'i büyük ölçüde **her iki
düzlemin simetrik (QF/QD aynı-yönlü) parçalarının çarpımındadır** (§2b). Ve
simetrik alt-uzay, tanımı gereği **düşük kazançlıdır** (tekmeler söndüğü için): bu
modlar yörüngeyi $\sim$3 kat zayıf sürer ($G\sim3$), ofset gürültüsünün eşiğine
yakın ya da altında kalır.

Sayısal olarak (ortogonal ayrıştırma, `test_symm_vs_antisym` mantığı):

| parça | desen RMS | COD RMS | $f$ [rad/s] |
|---|---|---|---|
| antisimetrik (25 boyut) | 70 μm | 550 μm | büyük |
| **simetrik** (23 boyut) | 55 μm | 165 μm | **kalanın çoğu** |

Yörünge trimi antisimetrik parçayı temizler — ama dx·dy kanalında bu küçük bir
paydır. Simetrik parça **yerinde durur**, çünkü yörünge onu gürültü tabanının
altında göremez (sinyal/gürültü $\approx0.3<1$).

**Sonuç (her iki düzlemde COD trimi, doğru estimator, 200 μm):** kademe sahte
EDM'i yalnız $\sim$2–5× düşürür ve $\sim3\times10^{-4}$ rad/s tabanına çarpar.
Bu, $10^{-5}$ hedefinin çok üstündedir.

### 6c. Rekonstrüksiyon metodu tabanı değiştirmez

"Belki daha iyi bir geri-çatım metodu simetrik tabanı aşar" diye altı metot aynı
veride denendi (her iki düzlem, 200 μm dx+dy):

| metot | bastırma |
|---|---|
| R-LS (FODO-antisim, $k\leq4$) | **4.7× (en iyi)** |
| TSVD | 1.9× |
| CLEAN | 1.1× |
| Bozoki (azimut baz) | 0.9× |
| R-LS ($k\leq7$) | 0.8× |
| tam $R^{-1}$ | 0.3× |

İki ders: (1) en iyi metot **dar, düşük-modlu** R-LS'tir (kazanç hiyerarşisi
düzenleyici); agresif metotlar (CLEAN, tam $R^{-1}$, geniş R-LS) düşük-kazançlı
modlara ofset enjekte edip kötüleşir. (2) **Hiçbir metot $\sim10^{-4}$ tabanının
altına inmez** — bu bir **gözlenebilirlik sınırıdır**, metot kusuru değil.
Simetrik alt-uzay yörüngeye yapısal olarak kapalıdır; onu yalnız spin görür.

---

## 7. İkinci kademe: spin trimi

Yörüngenin bıraktığı simetrik tabanı **spin** temizler — çünkü spin tüm modlara
bağlanır (kuplaj $c_k$ hiçbir modda sıfır değildir; simetrik içerikte
$\sim$12 kat zayıf ama sonlu). Burada §3'teki avantaj devreye girer: spin simetrik
alt-uzayda iyi bir sinyal/gürültüye sahiptir (polarimetre tabanına karşı
SNR$\sim$17), oysa yörünge için SNR$\sim$0.3 idi.

### Uçtan uca zincir, sayılarla (200 μm → hedef altı)

| Aşama | Ne yapılır | $f$ [rad/s] |
|---|---|---|
| Başlangıç | 200 μm dx+dy | $\sim 9\times10^{-4}$ |
| **Stage 1 — yörünge** | her iki düzlemde antisim $k\leq4$ trimi | $\sim 9\times10^{-4}$ (çok az düşer) |
| **Stage 2 — spin** | simetrik modun $c$'sini kalibre et, ölç–trimle | $1.6\times10^{-7}$ |

Stage 1'in neredeyse hiç düşürmemesi (§6b) beklenendir: kanal simetrik-domine.
Stage 2'de simetrik mod $S_2$'nin spin kuplajı ölçülür ($|c|\approx12.6$ rad/s/m —
zayıf ama sonlu), §5'teki ölç–trimle döngüsü işletilir. Dört adımda
$$9.4\times10^{-4}\to2.2\times10^{-6}\to\dots\to1.6\times10^{-7}\quad(\sim6000\times),$$
75 μm trim bütçesiyle. Bu, CW/CCW iptal zincirinin giriş hedefi $10^{-5}$'in
$\sim$60 kat altındadır. Tek bir simetrik mod skaler $f$'i nullamaya yeter (§5);
iterasyonlar ölçüm tabanına yakınsar.

**EDM güvenliği.** Spin hem sahte hem gerçek EDM'i görür; körlemesine $f\to0$
yapmak gerçek sinyali de yutabilir. Çözüm: hata sinyalini CW (saat yönü) ve CCW
(ters yön) saklamaların **farkı** üzerinden kurmak. Gerçek EDM yön değişiminde
işaret değiştirmez (toplamda korunur), sahte EDM değiştirir (farkta kalır). Böylece
spin kademesi de yapısal olarak EDM-kör hale gelir.

---

## 8. Büyük resim: iki kademe + CW/CCW zinciri

Güncel verilerle tutarlı tam tablo:

| Kademe | Araç | Gördüğü | 200 μm'den $f$ |
|---|---|---|---|
| 0 | Ham | — | $\sim9\times10^{-4}$ |
| **1** | Yörünge (COD) trimi, EDM-kör | antisim, yüksek-kazanç | $\sim9\times10^{-4}$ (zayıf düşüş; dx·dy simetrik-domine) |
| **2** | Spin ölç–trimle | tüm modlar, simetrik dahil | $\sim1.6\times10^{-7}$ |
| 3 | CW/CCW + quad-flip | yön/gradyan simetrisi | $<10^{-9}$ |

Roller net: **yörünge kademesi** hızlıdır, EDM'e dokunmaz, ama yapısal olarak
simetrik alt-uzaya kördür — antisimetrik, yüksek-kazançlı içeriği saniyeler
içinde temizler. **Spin kademesi** yavaştır (polarimetre istatistiği) ama
eksiksizdir; yörüngenin göremediği simetrik tabanı temizleyip sinyali CW/CCW
zincirinin giriş seviyesinin altına indirir. **CW/CCW + quad-flip** kalanı yön ve
gradyan simetrileriyle $10^{-9}$'un altına taşır. Üç kademe **tamamlayıcıdır**;
hiçbiri tek başına hedefe ulaşmaz.

*(Önemli düzeltme: bu belgenin eski sürümünde dx·dy yerine yalnız dikey kaçıklık
vurgulanıyor ve eski "tek parçacık + düz fit" estimator'ından gelen abartılı
bastırma rakamları kullanılıyordu. Yukarıdaki tüm sayılar dört-parçacık + model
fit ile, dx·dy kanalında alınmıştır; ayrıntı `false_edm_harmonic_sinir.md` §13–14.)*

---

## 9. Sık sorulan sorular

**S1 — Sahte EDM'in asıl kaynağı tek başına dikey kaçıklık değil mi?**
Hayır. Tek başına $dy$ küçük ve büyük ölçüde iptal olan bir etki verir; tek başına
$dx$ dikey spini hiç etkilemez. Asıl büyük sinyal **dx·dy geometrik-faz
çarpımından** gelir ve $\sigma^2$ ile ölçeklenir (§1).

**S2 — Yörünge düzeltmesini hem x hem y'de yaparsak yeter mi?**
Hayır. Her iki düzlemde COD trimi yapılır (zaten öyle yapıyoruz), ama dx·dy
kanalı **simetrik alt-uzayda** domine ve yörünge bu alt-uzaya kördür; sonuç
$\sim$2–5×'te durur. Denenen altı rekonstrüksiyon metodu da aynı tabana çarpar
(§6c). COD tek başına çözmez.

**S3 — O zaman spin neden görüyor da yörünge görmüyor?**
İki farklı fizik. Yörünge kazancı **rezonans** kökenlidir ($G_k\propto
1/|Q_{\rm eff}^2-k^2|$); simetrik/yüksek modlar düşük kazançlı olup BPM ofseti
gürültüsünün altında kalır (SNR$<$1). Spin kuplajı $c_k$ ise **geometrik bir yol
integralidir**, rezonans gerektirmez; tüm modlara bağlanır (simetrik içerikte
zayıf ama sıfır değil) ve polarimetre tabanına karşı SNR$\sim$17 verir.

**S4 — Tek bir trim modu nasıl tüm sinyali sıfırlıyor?**
Çünkü $f$ tek bir skalerdir; kalibre bir düğmeyi $-f/c$ kadar çevirmek onu
doğrusal mertebede tam götürür (§5). Deseni çözmek gerekmez; ibreyi sıfırlamak
yeter.

**S5 — Spin trimi gerçek EDM'i yutmaz mı?**
Yutabilirdi; bu yüzden hata sinyali CW$-$CCW farkı üzerinden kurulur. Gerçek EDM
yön değişiminde işaret değiştirmez, sahte EDM değiştirir → fark yalnız
sistematiği görür (§7).

**Açık başlıklar:** RF ve sekstüpoller açıkken doğrusallığın teyidi; quad tilt
(skew) çapraz terimleri; çok-tohumlu RMS ile zincirin istatistiksel sağlamlığı;
zamanla sürüklenen hizalamada sürekli izleme kipi.

---

*Sayısal kanıtlar ve reprodüksiyon: `false_edm_harmonic_sinir.md` §13–14
(dx·dy kuadratiği, demet=ideal, rekonstrüksiyon karşılaştırması, COD+spin zinciri).*
