# pEDM Halkasında Manyetik Alan Ölçümü

> **Okuyucu varsayımı.** Tune, beta fonksiyonu, kapalı yörünge,
> FODO lattice gibi temel hızlandırıcı kavramları biliniyor.
> Matris koşullanması, SVD, Fourier analizi gibi sayısal yöntemler
> yabancı ya da soluk kalmış. Bu belge her kavramı ilk kullandığında
> kısa bir sezgiyle açıklar — ve önce hikayeyi anlatır, sonra teknik
> ayrıntılara geçer.

---

## İçindekiler

1. [Sahne ve bağlam](#1-sahne-ve-bağlam)
2. [Simülasyon altyapısı: halka nasıl modelleniyor?](#2-simülasyon-altyapısı)
3. [Problem: 10 μm'yi neden ölçemiyoruz?](#3-problem)
4. [İlk fikir: k-modülasyon](#4-i̇lk-fikir-k-modülasyon)
5. [Uniform kmod başarılı — ama pratik değil](#5-uniform-kmod)
6. [Drift modu: farklı bir soruyu çözmek](#6-drift-modu)
7. [Tek ve iki-quad kmod denemeleri](#7-tek-ve-i̇ki-quad-kmod)
8. [Fourier fikri: bilinmeyenleri azalt](#8-fourier-fikri)
9. [Hedefli Fourier: büyük başarı, ama idealize](#9-hedefli-fourier)
10. [Ne zaman çalışmadı ve neden?](#10-ne-zaman-çalışmadı-ve-neden)
11. [Greedy ve LASSO denemeleri](#11-greedy-ve-lasso)
12. [Çok-konfigürasyon yığma: rank sorununa çözüm](#12-çok-konfigürasyon-yığma)
13. [Nerede duruyoruz? Fiziksel bir değerlendirme](#13-nerede-duruyoruz)
14. [Bu donanımla başka ne ölçülebilir?](#14-bu-donanımla-başka-ne-ölçülebilir)
15. [Depo yapısı ve hızlı başlangıç](#15-depo-yapısı-ve-hızlı-başlangıç)
16. [params.json — parametre referansı](#16-paramsjson--parametre-referansı)
17. [Bilinen tuzaklar](#17-bilinen-tuzaklar)
18. [Açık konular](#18-açık-konular)

---

## 1. Sahne ve Bağlam

pEDM (proton Electric Dipole Moment) deneyi, protonun elektrik dipol
momentini ölçmeyi hedefler. Deney, **frozen-spin** yöntemiyle çalışır:
protonlar özel tasarlanmış bir depolama halkasında dönerken spinin
momentumla hizalı kalmaya zorlandığı bir konfigürasyon kurulur. Gerçek
bir EDM varsa spin yavaşça orbitalden sapacaktır; bu sapma ölçülür.

Deneyin en büyük teknik zorluğu şudur: eğer halkadaki bir kuadrupol
mıknatıs ideal konumundan küçük bir $\Delta y$ kadar kaymışsa, o
quad'dan geçen demet ekstra bir dikey kick alır. Bu kick halka boyunca
birikerek kapalı yörüngeyi bozar. Bozulan yörünge, demeti başka
mıknatıslardaki yanlış konumlarda gezdirerek falso bir EDM sinyali üretir.

Hesaplamalar bu sahteciliği önlemek için her quad'ın dikey konumunun
**~10 μm** hassasiyetle bilinmesi gerektiğini gösteriyor.
Mekanik ölçüm bu hassasiyete ulaşamaz. Çözüm: demetin kendi yörüngesini
kullanarak quad konumlarını ölçmek.

Halka 24 FODO hücresi × 2 quad = **48 kuadrupol** içeriyor. Her
quad'ın yanında bir BPM (Beam Position Monitor) var. Problem şu: BPM
okumaları ~300 μm elektronik ofset taşıyor ve aradığımız sinyal yalnızca
~10 μm. Sinyal arka planın 30 katı altında gömülü.

Bu çalışma, bu problemi çözmek için bir dizi yöntemi sistematik biçimde
deniyor. Hangisinin neden çalıştığını, hangisinin neden çalışmadığını
anlıyor ve 10 μm hedefinin gerçekçi sınırlarını belirliyor.

---

## 2. Simülasyon Altyapısı

Gerçek bir halkada bu teknikleri doğrulamak için önce güvenilir bir
simülasyon gerekiyor. Kod iki katmandan oluşuyor.

### integrator.cpp: parçacık fiziksel simülasyonu

`integrator.cpp`, C++ ile yazılmış bir parçacık takip kodudur. Proton,
FODO hücresindeki her elemandan (drift, quad, yay deflektörü) sırayla
geçer; her element parçacığın 6 koordinatını ($x, x', y, y', t, \delta$)
günceller.

**Neden standart Runge-Kutta değil?** Newton denklemlerini sayısal
çözen standart yöntemler (Euler, RK4) faz uzayı hacmini yavaşça
değiştirir — parçacık sanki enerji kazanıyor ya da kaybediyormuş gibi
davranır. Hamiltonian mekaniğinde bu hacim korunmalıdır (Liouville
teoremi). Uzun simülasyonlarda bu hata birikir ve fiziksel olmayan
sonuçlar verir.

**GL4** (4. mertebe Gauss-Legendre simplektik entegratör) faz uzayı
hacmini makine hassasiyetine kadar tam koruyan bir yöntemdir. Bir
depolama halkası simülasyonu için doğal seçimdir.

Quad hizalama hatası $\Delta y_j$ varsa, parçacık o quad'ın merkezinden
$\Delta y_j$ uzakta geçer ve Lorentz kuvvetinden ekstra bir kick alır.
Simülasyon bu etkiyi mikron altı hassasiyetle modelleyebilir.

### integrator.py: Python köprüsü

Python kodu `integrator.py` üzerinden `ctypes` arayüzüyle C++
kütüphanesini çağırır. Tepki matrisi hesaplaması için her quad sırayla
küçük bir miktar kaydırılır, bir tam tur sonrası 48 BPM'deki orbit
değişimi ölçülür. 48 quad × 48 simülasyon → tam 48×48 tepki matrisi.

### build_response_matrix.py ve test_kmod_reconstruction.py

`build_response_matrix.py`, nominal ve perturbe gradyan ayarlarında
tepki matrislerini hesaplar. `test_kmod_reconstruction.py`, bilinen
bir $\Delta q$ desenini simüle edip çeşitli yöntemlerle geri çatmayı
dener ve hata ile korelasyonu raporlar.

---

## 3. Problem: 10 μm'yi Neden Ölçemiyoruz?

BPM okumaları:

$$\mathbf{y} = R\,\Delta q + \mathbf{b} + \boldsymbol{\eta}$$

$R$ tepki matrisi bilinebilir (simülasyon veya ölçümden). BPM gürültüsü
$\boldsymbol{\eta} \sim 1$ μm ihmal edilebilir. Sorun $\mathbf{b}$:
her BPM'in elektronik sıfır noktasının mekanik merkezden ~300 μm
sapması. Bu ofset **saatler-günler boyunca sabit kalır, ama bilinmez.**

Doğrudan $\Delta q = R^{-1}(\mathbf{y} - \mathbf{b})$ çözemeyiz çünkü
$\mathbf{b}$ bilinmiyor. $R^{-1}\mathbf{y}$ deneseydik, sonuç
$R^{-1}\mathbf{b}$ kirliliği taşırdı ve bu kirliliğin büyüklüğü
$\|\Delta q\|_\text{aranan}$'dan çok daha büyük.

$R$'nin kondisyon sayısı ~160'tır. "Kondisyon sayısı" sezgisel olarak
şu anlama gelir: girişteki küçük bir belirsizlik çıkışta kaç kat
büyütülür? 1 μm BPM gürültüsü → ~14 μm tahmin hatası. Bu kabul
edilebilir. Ama 300 μm BPM ofseti → ~4 mm kirliliği: tamamen kabul
edilemez.

**Anahtar soru:** BPM ofsetini iptal edecek bir ölçüm tasarımı mümkün mü?

---

## 4. İlk Fikir: K-Modülasyon

BPM ofseti $\mathbf{b}$ sabit — her ölçümde aynı. O zaman iki farklı
gradyan ayarında ölçüp fark alırsak:

$$\Delta\mathbf{y} = \mathbf{y}_2 - \mathbf{y}_1
= (R_2 - R_1)\,\Delta q + \underbrace{(\mathbf{b} - \mathbf{b})}_0
+ (\boldsymbol{\eta}_2 - \boldsymbol{\eta}_1)$$

$$\Delta\mathbf{y} = \Delta R\,\Delta q + \text{küçük gürültü}$$

$\mathbf{b}$ iptal oldu. Şimdi $\Delta R\,\Delta q = \Delta\mathbf{y}$
denklemini çözmek yeterli. Teoride bu basit. Pratikte asıl sorun
$\Delta R$'nin ne kadar iyi koşullanmış olduğuna bağlı.

---

## 5. Uniform Kmod Başarılı — Ama Pratik Değil

**İlk deneme:** Tüm 48 quad'ın gradyanı aynı anda %2 artırılır.
Bu durumda $\Delta R \approx 0.02 \times R$, dolayısıyla
$\kappa(\Delta R) \approx \kappa(R) \approx 160$. Kondisyon sayısı
$R$ kadar iyi, gürültü büyütmesi küçük.

**Sonuç:** Hata ~6.6 μm RMS, korelasyon ≈ 1.000. 10 μm hedefine
oldukça yakın, temiz bir başarı.

**Neden pratik değil?** Gerçek bir halkada 48 bağımsız güç kaynağını
eşzamanlı, hassas biçimde modüle etmek çok zordur. Güç kaynakları
arasındaki küçük farklar (biri %2.05 değişirken diğeri %1.97 değişirse)
analizi bozar. Daha gerçekçi bir senaryo: yalnız 1 veya 2 quad
modüle edilir, diğerleri sabit kalır.

---

## 6. Drift Modu: Farklı Bir Soruyu Çözmek

Uniform kmod dışında başka bir yöntem daha var: **drift modu.**
Fikir farklı — gradyan değiştirme değil, zaman farkı almak:

$$\delta q(t) = R^{-1}\bigl(\mathbf{y}(t) - \mathbf{y}_0\bigr)$$

Referans orbit $\mathbf{y}_0$ sabit bir anda ölçülür. Sonraki her
ölçümde BPM ofseti aynı kaldığından fark alındığında iptal olur.
$R$'nin kondisyon sayısı ~160 olduğundan gürültü büyütmesi küçük.

**Sonuç:** Hata ~6.5 μm RMS, korelasyon ≈ 1.000. Uniform kmod
kadar iyi ve çok daha kolay uygulanabilir.

**Kritik sınır:** Drift modu, halkanın kurulumundan bu yana var olan
**kalıcı** hizalama hatalarını göremez. $\mathbf{y}_0$ bu hataları
zaten içerir ve onları "sıfır" sayar; yalnızca $t_0$'dan sonra oluşan
**değişimleri** ölçer. Kurulumda var olan 100 μm'lik bir sistematik
sapma drift modunda görünmez.

Bu çalışmanın bir parçası: uniform kmod ile drift modu benzer hassasiyete
ulaşıyor ama **farklı soruları çözüyorlar.** Drift modu zaman içindeki
değişimleri, uniform kmod anlık mutlak konumları ölçüyor.

---

## 7. Tek ve İki-Quad Kmod Denemeleri

Uniform kmodun pratik sınırını gören sonraki adım: **yalnız 1 ya da 2
quad modüle edilirse ne olur?**

### Tek-quad kmod başarısız

Yalnız bir quad'ın ($j_1$) gradyanı değiştirilince $\Delta R$'nin yalnız
$j_1$. sütunu (ve dolaylı olarak tüm örgünün optiği) değişir. Etkin rank ~1.
Kondisyon sayısı ~4×10⁸. 1 μm gürültü → tahmin hatası yüzlerce mm.
Tamamen anlamsız.

### İki-quad kmod: rank-2 ve yüksek kondisyon sayısı

İki quad ($j_1$, $j_2$) birlikte modüle edilince etkin rank ~2.
Kondisyon sayısı ~10⁶. Doğrudan $\Delta R^{-1}$ uygulanırsa:
hata 107 μm, korelasyon 0.03 — başarısız.

TSVD (truncated SVD: yalnız büyük tekil değerleri tut, küçükleri at):
hata 78 μm, korelasyon 0.16 — küçük bir iz kalıyor ama hâlâ çok kötü.

Açıklama: matriste **gerçek bilgi var** ama çok az boyutta (rank ~2).
Doğrudan ters çevirmek 46 boyutluk "gürültü uzayını" da bir çözüme
dahil ediyor. Bunu engellemek için problemi yeniden çerçevelemek gerekiyor.

---

## 8. Fourier Fikri: Bilinmeyenleri Azalt

Bu noktada kritik bir sezgi: quad hizalama hataları fiziksel olarak
**uzun dalgalı** bozulmalardan kaynaklanır (yerçekimi, zemin hareketi,
montaj sistematikleri). Tünel yavaşça eğilir; 48. quad'ın 1. quad'dan
tamamen bağımsız, rastgele bir konumda olması beklenmez.

Bu da şu anlama gelir: $\Delta q$'nun **Fourier içeriği düşük
frekanslarda yoğunlaşmalı.** 48 bağımsız sayı yerine birkaç Fourier
katsayısı yeterli olabilir.

Matematiksel çerçeve:

$$\Delta q_j \approx a_0
+ a_{2c}\cos\!\left(\tfrac{2\pi \cdot 2 \cdot j}{48}\right)
+ a_{2s}\sin\!\left(\tfrac{2\pi \cdot 2 \cdot j}{48}\right)
+ \ldots$$

Matris biçiminde $\Delta q = F\hat{a}$, k-mod denklemine koyunca:

$$\Delta\mathbf{y} = \Delta R\,F\,\hat{a} \equiv M\,\hat{a}$$

$M = \Delta R \cdot F$ matrisi 48 × $n_\text{baz}$ boyutlu. $n_\text{baz} = 3$
seçilirse: 48 denklem, 3 bilinmeyen → aşırı belirlenmiş sistem → en
küçük kareler çözümü.

**Neden bu yardımcı olur?** $\Delta R$'nin "güçlü yönleri" (büyük tekil
değerleri), tune frekansına yakın düşük frekanslı modlara karşılık gelir.
Fourier bazını bu güçlü yönlerle hizalı seçersek, $M = \Delta R \cdot F$
yalnız güçlü yönleri taşır ve kondisyon sayısı dramatik biçimde düşer.

### FODO antisimetrisi

Aynı FODO hücresindeki QF ve QD quad'ları zıt işarette kayarsa birbirlerinin
kick'ini güçlendirir ve halkada belirgin bir COD oluşturur. Aynı yönde
kayarlarsa birbirini neredeyse iptal eder. Bu nedenle Fourier baz
fonksiyonu QF/QD ayrımını dikkate alır:

$$F_k[j] = (-1)^j \cdot \cos\!\left(\frac{2\pi k \lfloor j/2 \rfloor}{24}\right)$$

$(-1)^j$ faktörü, QF ve QD'nin zıt işaretli katkısını modelliyor.

---

## 9. Hedefli Fourier: Büyük Başarı, Ama İdealize

Fourier fikrinin ne kadar güçlü olduğu baz seçimiyle ortaya çıkıyor.
Aşağıdaki tablo, gerçek $\Delta q$'nun yalnız $k=2$ ve $k=4$
harmoniklerinden oluştuğu **idealize test senaryosunda** farklı bazların
karşılaştırmasını gösteriyor:

| Baz | $\kappa(M)$ | Hata RMS | Korelasyon |
|-----|-------------|----------|------------|
| Direkt $\Delta R^{-1}$ (baz yok) | ~10⁶ | 107 μm | 0.03 |
| Geniş: $k = 1, 2, 3, 4$ | 13.000 | 35 μm | 0.88 |
| Yanlış: $\{k=4\}$ tek | 14 | 1466 μm | −0.38 |
| **Tam doğru: $\{k=2, k=4\}$** | **186** | **0.02 μm** | **1.000** |

Tam doğru bazla geniş baza kıyasla **2000 kat fark.** Bu farkın kaynağı
kondisyon sayısı: 186 ile 13.000 arasındaki ~70 katlık artış, gürültü
büyütmeyi doğrudan 70 kata çıkarıyor.

**Baz nasıl seçilmeli?** İdeal durumda fizikten belli: uzun dalgalı
bozulmalar k=0 (DC) ve k=1, 2 gibi düşük harmonikler. Yanlış harmonik
eklenmesi kondisyon sayısını şişirir, eksik harmonik ise sızıntı hatası
üretir. Tek bir yanlış harmonik tüm çözümü patlatabiliyor (tablodaki
$\{k=4\}$ satırı: doğru harmoniği içermiyor ama en küçük kareler
$k=2$'yi $k=4$ cinsinden açıklamaya çalışıyor ve katsayı absürd büyüyor).

---

## 10. Ne Zaman Çalışmadı ve Neden?

Hedefli Fourier'ın 0.02 μm başarısı gerçek — ama çok özel koşullar
altında. Gerçekçi senaryolarda iki ayrı sorunla karşılaşıldı.

### Sorun 1: Rank yetersizliği (sayım problemi)

$k=0$ ve $k=2$ harmoniklerini aynı anda kestirmek için 3 bilinmeyen var:
$a_0$ (DC), $a_{2c}$ (cos₂), $a_{2s}$ (sin₂). Ama iki-quad kmod yalnız
rank ~2 bağımsız bilgi üretiyor.

3 bilinmeyen, 2 bağımsız denklem → **yetersiz belirlenmiş sistem.**
En küçük kareler sonsuz sayıda çözüm arasından en küçük norma sahip
olanı döndürür — bu rastgele bir seçim ve $a_0$ ile $a_{2c}$'yi
güvenilir biçimde birbirinden ayırt edemez.

### Sorun 2: Tune ve Fourier frekanslarının uyumsuzluğu

Tek bir quad modüle edilince BPM'lerde oluşan orbit deseni
şu biçimdedir:

$$\Delta y_i \propto \sqrt{\beta_i}\,\cos\!\bigl(\phi_i - \phi_{j_1} - \pi Q\bigr)$$

Bu desen **tune frekansında** ($Q \approx 2.68$) titreşiyor — irrasyonel
bir sayı. Ölçmek istediğimiz Fourier harmonikleri ise tam sayı $k$'da
titreşiyor.

$M = \Delta R \cdot F$ matrisi, "tune-frekanslı ölçüm → Fourier katsayısı"
dönüşümünü yapmak zorunda. Bu dil çevirisi ne kadar zorsa kondisyon
sayısı o kadar büyür. Hedefli Fourier'ın 0.02 μm başarısı, test
verisindeki $k=2, 4$ harmoniklerinin tune frekansıyla *tesadüfen* iyi
örtüşmesinden kaynaklanıyor.

### Sorun 3: 100 μm rastgele arka plan SNR'ı boğuyor

Gerçek $\Delta q$ iki bileşenden oluşur:

$$\Delta q = \underbrace{\Delta q_\text{smooth}}_{\sim 10\;\mu\text{m}}
+ \underbrace{\Delta q_\text{random}}_{\sim 100\;\mu\text{m RMS}}$$

$\Delta R$ üzerinden ölçüme karışırlarsa:

$$\Delta\mathbf{y} = \underbrace{\Delta R\,\Delta q_\text{smooth}}_{\text{aranan}}
                  + \underbrace{\Delta R\,\Delta q_\text{random}}_{\sim 10\times\text{sinyal}}$$

Random katkısı sinyalden 10 kat büyük ve $\Delta\mathbf{y}$'nin içinde
gerçek bir **sinyal** olarak görünüyor — gürültü gibi değil. Fit
bunu göremez; sadece $\Delta\mathbf{y} = M\hat{a}$'yı minimize etmeye
çalışır ve random katkı sonuca karışır.

**Bu, BPM elektronik gürültüsünden tamamen farklı bir sorundur.**
BPM gürültüsü (~1 μm) ölçüm cihazından kaynaklanır ve aşırı
belirlenmiş sistemde bastırılabilir. $\Delta R\,\Delta q_\text{random}$
ise quad kayma alanının kendisinden kaynaklanır — daha fazla ölçüm
yapmak bunu azaltmaz.

100 μm RMS senaryosundaki testler bu yüzden kötü sonuç vermiştir
ve bu beklenen bir sonuçtur.

---

## 11. Greedy ve LASSO

Harmonikleri fizikten tahmin etmek yerine **veriden otomatik tespit**
eden iki yöntem denendi. İkisi de rank-2 sistemde başarısız oldu.

### Greedy matching pursuit

Her adımda rezidüeli en çok düşüren Fourier harmonik $k$'yı seç;
seçim kazancı yeterince büyük değilse dur.

**Neden başarısız?** $\Delta R$'nin SVD ayrışımından iki baskın yön
($v_1, v_2$) var. Greedy bu iki yönde "görebilir" — ama bu yönler
tune frekansında titreşiyor (bkz. §10, Sorun 2). FODO-Fourier harmonikleri
bu yönlerle tam hizalı değil. Sonuç: greedy rezidüeli düşürür ama
**fiziksel olmayan harmonikler** seçer.

Test: gerçek $k=0, 2, 4$ yerine greedy $k=1, 8, 7$ seçti.

### LASSO (L1 cezalı rekonstrüksiyon)

Tüm olası harmonikleri aday koy, L1 ceza gereksizleri sıfıra çeksin.

**Neden başarısız?** $M$ rank-2 ve ~25 sütunlu → 23 boyutlu null uzayı.
Sistem bu null uzayda enerjiyi tüm katsayılara eşit dağıtıyor: her
katsayı $\sim |\text{sinyal}|/25 \approx 0.4$ μm alıyor. Bu eşik
altında — LASSO tümünü sıfıra itiyor. **Hiçbir $\lambda$ değeri
bu durumu kurtaramaz:** $\lambda$ küçültülürse ayrımcılık kaybedilir,
büyütülürse sinyal de silinir.

Her iki yöntemin başarısızlığının ortak nedeni aynı: **rank yetersizliği.**
Rank ≥ 3 olduğunda (§12) greedy daha güvenilir hale gelir.

---

## 12. Çok-Konfigürasyon Yığma: Rank Sorununa Çözüm

### Fikir

3 bilinmeyen için 3 bağımsız denklem gerekiyor. Bunu sağlamak için
**üç farklı tekil-quad ölçümünü dikey yığ:**

| Ölçüm | Modüle edilen quad | Rank katkısı |
|-------|-------------------|--------------|
| Konfig 0 | j = 3 | ~1 |
| Konfig 1 | j = 9 | ~1 |
| Konfig 2 | j = 1 | ~1 |
| **Yığılmış** | — | **~3** |

Yığılmış sistem: 3×48 = 144 denklem, 3 bilinmeyen → aşırı belirlenmiş,
rank ~3.

### Neden j = 1, 3, 9 seçildi?

Bu seçimin arkasında bir fizik var. Her quad'ın $k=2$ moduna projeksiyonu
FODO hücresindeki konumuna göre değişiyor. j=3, 9, 1 için bu projeksiyonlar
+1.00, −0.50, +0.87 — iki pozitif, bir negatif. Bu $\cos_2$ ve $\sin_2$
katkılarını birbirinden ayırmak için yeterli doğrusal bağımsızlığı sağlıyor.

Hepsi QD tipi olduğu için `integrator.cpp`'deki bilinen j=0 bug'undan
da etkilenmiyorlar.

### Sınır

Bu yığma **rank sorununu** çözüyor. §10'daki SNR sorununu çözmüyor.
100 μm random arka plan varlığında her yeni kmod ölçümü de aynı
$\Delta q_\text{random}$'ı taşıdığından, daha fazla konfigürasyon
eklemek bu paraziti azaltmıyor.

---

## 13. Nerede Duruyoruz? Fiziksel Bir Değerlendirme

### Sonuçların özeti

| Yöntem | Hata | Koşul | Not |
|--------|------|-------|-----|
| Uniform kmod | ~6.6 μm | Tüm 48 quad eşzamanlı | Pratik uygulama zor |
| Drift modu | ~6.5 μm | Herhangi orbit ölçümü | Yalnız zamansal değişimler |
| Hedefli Fourier (idealize) | ~0.02 μm | Baz tam doğru, random yok | Gerçekçi değil |
| Hedefli Fourier (100 μm random) | yüksek | — | SNR sorunu |

### Falso EDM sinyali ve beta fonksiyonu

Bu çalışmada ölçülen büyüklük quad kayma profili $\Delta y_j$. Ama
pEDM için asıl önemli büyüklük falso EDM sinyali:

$$\Phi_\text{spin} \propto \sum_j K_j L_j \bigl(y_\text{CO}(s_j) - \Delta y_j\bigr)$$

Kapalı yörünge $y_\text{CO}(s_j) = \sum_k R_{jk}\,\Delta y_k$ ve
$R_{jk} \propto \sqrt{\beta_j \beta_k}$ olduğundan bu toplam
**beta fonksiyonuyla ağırlıklandırılmış** bir büyüklük. Büyük
$\beta$'lı quad'ların katkısı orantısız büyük; basit ortalama
$\langle \Delta y \rangle$ ile falso EDM sinyali arasındaki
korelasyon zayıf.

Bu çalışma $\Delta y_j$'yi ölçüyor; falso EDM sinyalinin kendisini
doğrudan hedeflemek farklı bir ölçüm tasarımı gerektirir (bkz. §14).

### K-modülasyonun kör olduğu şeyler

K-modülasyon yalnız **gradyan değişimiyle birlikte değişen** etkileri
görür. Tilted dipol, stray manyetik alan, elektrostatik asimetri gibi
**gradyan-bağımsız sabit alanlar** $\Delta R$'ye yansımaz; k-modülasyon
bunlara kördür. Bu alanlar da kapalı yörüngeyi bozar ve BPM'ler bazen
görür — ama BPM ofsetiyle karışık olduğu için k-modülasyon bile ayırt edemez.

---

## 14. Bu Donanımla Başka Ne Ölçülebilir?

1-2 modüle edilebilir quad + 48 BPM; quad hizalama dışında ne ölçebilir?

### Beta fonksiyonu (en temiz uygulama)

Bir quad'ın gradyanı $\Delta K$ kadar değiştirilince tune kayması:

$$\Delta Q_y = \frac{\beta_j \cdot \Delta K \cdot L}{4\pi}$$

Bu denklemden quad $j$'nin bulunduğu noktadaki $\beta_j$ doğrudan
ölçülür. BPM bilgisi gerekmez, kondisyon sayısı sorunu yok. Bu
çalışmada kullanılan donanımın en doğal ve temiz uygulaması.

### Falso EDM katkısının doğrudan hedeflenmesi

§13'te görüldüğü gibi falso EDM sinyali $w^T \Delta q$ formunda
bir doğrusal fonksiyonel. $w$ vektörü $K_j$, $\beta_j$, $R$'den
önceden hesaplanabilir. Hangi quad'ın kmod sondaj yönü $v_j$ bu
$w$'ya en iyi hizalanırsa, **o tek quad** modüle edilerek falso EDM
katkısı doğrudan ölçülebilir. 48 komponentin hepsini ölçmek yerine
tek sayı hedefleniyor — rank sorunu bu özel problem için uygun.

### Quad tilt (skew kuplaj)

Bir quad dikey eksen etrafında tilt edilirse $x$-$y$ kuplaj üretir:
dikey kick yatay orbit değişikliğine yol açar. Yatay BPM okumaları
dikey kmod ölçümüyle birleştirilirse tilt miktarı kesilebilir.

### BPM gain kalibrasyonu (LOCO benzeri)

Bilinen orbit tepkisine karşı BPM okumaları karşılaştırılırsa
her BPM'in kazanç hatası tahmin edilebilir. Gradyan modülasyonu
"bilinen uyarı" sağlar.

---

## 15. Depo Yapısı ve Hızlı Başlangıç

### Dosya haritası

| Dosya | Rolü |
|-------|------|
| `integrator.cpp` | GL4 simplektik entegratör (C++) |
| `integrator.py` | Python/ctypes sarmalayıcı |
| `build_response_matrix.py` | R₁, R₂ ve ΔR matrislerini hesaplar |
| `test_kmod_reconstruction.py` | Simülasyon + rekonstrüksiyon karşılaştırması |
| `reconstruction.py` | Hedefli, Greedy, LASSO, çok-konfig rekonstrüksiyon |
| `scan_j2.py` | En iyi j₂ quad çiftini tara |
| `show_response.py` | Tepki matrisi görselleştirme |
| `FOURIER_REKONSTRUKSIYON.md` | Fourier yönteminin pedagojik derinlemesine anlatımı |
| `PROJE_ANALIZI_VE_ONERILER.md` | Analiz ve beyin fırtınası |
| `YÖNTEM.md` | Detaylı matematiksel türetimler |

### Kütüphaneyi derle

```bash
# Linux
g++ -O2 -shared -fPIC -o lib_integrator.so integrator.cpp -std=c++17
# macOS
clang++ -O2 -shared -fPIC -o integrator.dylib integrator.cpp -std=c++17
```

### Uniform kmod testi (temel doğrulama)

```bash
# params.json: kmod_quad1_index=-1, kmod_quad2_index=-1
python3 build_response_matrix.py
python3 test_kmod_reconstruction.py
```
Beklenen: hata ~6.6 μm, korelasyon ≈ 1.

### İki-quad hedefli Fourier (idealize)

```bash
# params.json: kmod_quad1_index=3, kmod_quad2_index=9
# dy_harmonics: yalnız k=2 ve k=4, dy_random_RMS=0
python3 build_response_matrix.py
python3 reconstruction.py
```
Beklenen: ~0.02 μm, korelasyon ≈ 1.

### Çok-konfigürasyon yığma

```bash
# dy_harmonics: k=0 ve k=2, dy_random_RMS=0
for n in 0 1 2; do python3 build_response_matrix.py --config $n; done
for n in 0 1 2; do python3 test_kmod_reconstruction.py --config $n; done
python3 reconstruction.py
```
Beklenen: yığılmış sistem rank ~3, k=0+k=2 birlikte ayrıştırılır.

---

## 16. params.json — Parametre Referansı

### Lattice ve simülasyon

| Anahtar | Anlam |
|---------|-------|
| `R0` | Halka yarıçapı (m) |
| `nFODO` | FODO hücresi sayısı (24 → 48 quad) |
| `quadLen`, `driftLen` | Element uzunlukları (m) |
| `g0`, `g1`, `g2` | Nominal, modüle-1 ve modüle-2 quad gradyenleri |
| `dt`, `t2`, `t_pr` | Zaman adımı, toplam süre, periyot |
| `quadSwitch`, `sextSwitch`, `EDMSwitch`, `rfSwitch` | Element açma/kapama |

### K-modülasyon konfigürasyonu

| Anahtar | Anlam |
|---------|-------|
| `kmod_quad1_index` | Modüle edilen 1. quad. −1: uniform mod |
| `kmod_quad2_index` | Modüle edilen 2. quad. −1: tek-quad mod |
| `kmod_configs` | Çok-konfig listesi: `[{"j1":3,"j2":-1}, ...]` |

**Mod seçim mantığı:**
- `j1=j2=−1` → uniform (tüm 48 quad)
- `j2=−1` → tek-quad
- `j2≥0` → iki-quad

### Hizalama hatası üretimi

| Anahtar | Anlam |
|---------|-------|
| `dy_harmonics` | `[{k, amp_cos, amp_sin}, ...]` — smooth Fourier bileşenleri |
| `dy_random_RMS` | Random arka plan gürültüsü RMS (m) |
| `smooth_antisym_fodo` | `true`: FODO antisimetrik (önerilen, fiziksel) |

### Rekonstrüksiyon algoritması

| Anahtar | Anlam |
|---------|-------|
| `k_search_max` | Greedy aramasının üst sınırı (Nyquist = 12) |
| `greedy_residual_threshold` | Minimum oransal rezidüel düşüşü (tipik: 0.01–0.05) |
| `max_harmonics` | Greedy maksimum harmonik sayısı |
| `lasso_lambda` | LASSO L1 ceza katsayısı |

### Ek hata kaynakları

| Anahtar | Anlam |
|---------|-------|
| `dipole_random_tilt_max` | Dipol tilt dağılımı üst sınırı (rad) |
| `quad_random_tilt_max` | Quad tilt — x-y kuplajı üretir |
| `bpm_noise_sigma` | BPM elektronik gürültüsü σ (m) |
| `bpm_offset_sigma` | BPM statik ofseti σ (m) |

---

## 17. Bilinen Tuzaklar

### `kmod_quad1_index = 0` kullanma

`integrator.cpp:541`'de `current_fodo == 0` için özel "QUAD_F_MOD"
tipi tetiklenir ve `quad_dG` modülasyon vektörü yok sayılır → $\Delta R = 0$.
Her zaman `j₁ ≥ 1` kullan.

### lib_integrator.so güncelliği

`quad_dG` desteği `commit 2e44a72` ile eklendi. Eski kütüphane bu
argümanı yok sayar → $R_1 \equiv R_2$, $\Delta R = 0$. Şüphe varsa
yeniden derle.

### FODO antisimetri kapatılırsa

`smooth_antisym_fodo: false` ile simetrik desen üretilirse QF ve QD
kick'leri birbirini söndürür, COD sinyali neredeyse sıfır olur. Bu
mod yalnız "çok zor durum" testi için.

### Greedy eşiği

`greedy_residual_threshold` çok küçükse sahte harmonikler eklenir.
Çok büyükse gerçek harmonikler atlanır. Tipik: 0.01–0.05.

### 100 μm random senaryosu

`dy_random_RMS = 1e-4` testleri kötü sonuç verir. Bu beklenen:
random bileşen sinyalden ~10× büyük, Fourier fit ikisini ayırt edemez.
Bu bir yazılım hatası değil, yöntemin fiziksel sınırı.

---

## 18. Açık Konular

- **Çok-konfig yığma doğrulaması:** 3 konfig → rank ~3 elde ediliyor mu?
  k=0+k=2 katsayıları doğru çıkıyor mu? Kod hazır, sayısal sonuç
  raporlanmadı.

- **Beta-beat etkisi:** Gerçek halkada $R$ model $R$'den sapınca
  ne oluyor? %1 beta-beat → kaç μm ek hata? (Test 8, sürmekte.)

- **Falso EDM hedef vektörü:** $w = (w_1, \ldots, w_{48})$ vektörünü
  hesapla ($K_j$, $\beta_j$, $R$ kullanarak), hangi quad en iyi hizalanıyor?
  Tek ölçümle falso EDM katkısını doğrudan hedefleme denenebilir.

- **Greedy + çok-konfig:** Rank ≥ 3 ile greedy harmonikleri doğru
  seçiyor mu? Sistematik test yapılmadı.

- **Bootstrap hata çubukları:** Katsayı belirsizliği nasıl ölçülür?
