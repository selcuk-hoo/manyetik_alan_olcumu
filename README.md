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
12b. [Gürültü altındaki zayıf harmoniği ayıklamak: CLEAN](#12b-gürültü-altındaki-zayıf-harmoniği-ayıklamak-neler-i̇şe-yaramaz-clean-ne-yapar)
12c. [Tek orbit, R-tabanlı CLEAN: kmod'suz BPM ofseti aşımı](#12c-tek-orbit-r-tabanlı-clean)
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

$R$'nin kondisyon sayısı ~249'dur. "Kondisyon sayısı" sezgisel olarak
şu anlama gelir: girişteki küçük bir belirsizlik çıkışta kaç kat
büyütülür? 1 μm BPM gürültüsü → ~4 μm tahmin hatası. Bu kabul
edilebilir. Ama 300 μm BPM ofseti → ~1.2 mm kirliliği: tamamen kabul
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
$\kappa(\Delta R) \approx \kappa(R) \approx 249$. Kondisyon sayısı
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
$R$'nin kondisyon sayısı ~249 olduğundan gürültü büyütmesi küçük.

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

### Peki 3-4 quad modüle edilirse?

Tek-quad (rank ~1) ile iki-quad (rank ~2) arasındaki sıçrama, doğal
soruyu doğuruyor: **3 ya da 4 quad modüle edilirse sonuç düzelir mi?**

Evet, ve monoton biçimde. Her bağımsız quad modülasyonu rank'a ~1
katkı yapıyor:

| Modüle edilen quad | Etkin rank | κ(ΔR) eğilimi |
|--------------------|------------|----------------|
| 1 | ~1 | ~4×10⁸ |
| 2 | ~2 | ~10⁶ |
| 3 | ~3 | daha düşük |
| 4 | ~4 | daha da düşük |
| … | … | … |
| 48 (uniform) | ~48 | ~249 |

Kondisyon sayısı, quad sayısı arttıkça iki-quad'ın ~10⁶'sından uniform
kmod'un ~249'una doğru iniyor. Tam değer konfigürasyona (hangi quad'lar,
hangi FODO fazlarında) bağlı ama eğilim açık: **daha çok bağımsız ölçüm
= daha yüksek rank = daha düşük κ = daha çözülebilir sistem.**

### Temel teorem: rank'ı belirleyen quad sayısıdır, gradyen değil

Birinci mertebede $\Delta R$:

$$\delta R = -R\,\mathrm{diag}(\delta K)\,R$$

Buradan doğrudan şu sonuç çıkar:

$$\mathrm{rank}(\delta R) \leq \mathrm{rank}\bigl(\mathrm{diag}(\delta K)\bigr)
= \text{modüle edilen quad sayısı}$$

**Yani gradyen değişimi ne kadar büyük olursa olsun rank artmaz.**
2 quad'ı %10 değiştirmek → rank ≤ 2. 48 quad'ı %2 değiştirmek → rank ≤ 48.

Bu, sayısal olarak doğrulandı: **2 quad %10 ile 48 quad %2 karşılaştırılınca,
48 quad %2 çok daha etkili** sonuç verdi. Genlik rank satın almaz — farklı
quad konumları (fazları) satın alır.

Pratik sonuç: lokalize birkaç quad'ı büyük miktarda modüle etmek yerine
**çok sayıda farklı quad'ı küçük miktarlarda** modüle etmek her zaman daha
iyidir. Bu sezgi, uniform kmod'un (%2, tüm 48 quad) neden ~6.6 μm hassasiyet
verirken 2-quad kmod'un (%10 bile olsa) neden başarısız olduğunu açıklar.

Önemli bir nüans: "3 quad'ı aynı anda modüle et" ile "3 ayrı tek-quad
ölçümü yap ve yığ" matematiksel olarak (quad konumları ölçümler arasında
değişmiyorsa) aynı rank-3 sistemi üretir. Eşzamanlı modülasyonun avantajı
hız (tek ölçümde rank-3 bilgi); dezavantajı hangi quad'ın ne kattığını
çözmenin biraz daha karmaşık olması. Bu "yığma" stratejisi §12'nin konusu.

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
altında. Gerçekçi senaryolarda iki ayrı sorunla karşılaşıldı: rank
yetersizliği ve SNR (sinyal/gürültü) problemi. Bir de tune frekansının
tam sayı olmamasının yarattığı, sanıldığından **daha hafif** bir etki var.

### Sorun 1: Rank yetersizliği (sayım problemi)

$k=0$ ve $k=2$ harmoniklerini aynı anda kestirmek için 3 bilinmeyen var:
$a_0$ (DC), $a_{2c}$ (cos₂), $a_{2s}$ (sin₂). Ama iki-quad kmod yalnız
rank ~2 bağımsız bilgi üretiyor.

3 bilinmeyen, 2 bağımsız denklem → **yetersiz belirlenmiş sistem.**
En küçük kareler sonsuz sayıda çözüm arasından en küçük norma sahip
olanı döndürür — bu rastgele bir seçim ve $a_0$ ile $a_{2c}$'yi
güvenilir biçimde birbirinden ayırt edemez. **Çözümü §12'de.**

### Sorun 2: Tune'un tam sayı olmaması — bir engel değil, yalnızca verim kaybı

Önceki bir taslakta bu etki abartılmıştı; burada doğrusunu kaydediyoruz.

Tek bir quad modüle edilince BPM'lerde oluşan orbit deseni:

$$\Delta y_i \propto \sqrt{\beta_i}\,\cos\!\bigl(|\phi_i - \phi_{j_1}| - \pi Q\bigr)$$

Bu desen tune frekansında ($Q \approx 2.68$) salınıyor. **Ama BPM ölçümleri
hâlâ periyodik** — 48 nokta, halka boyunca tekrar ediyor. Periyodik bir
sinyalin her Fourier bileşeni (k=2 dahil) tam sayı $k$ ile pekâlâ
hesaplanabilir ve **sıfır değildir.** Yani tune irrasyonel diye k=2
ölçülemez değil — ölçülebilir.

Tam sayı olmamanın gerçek etkisi şu: response sütununun Fourier gücü tek
bir $k$'da değil, $k \approx Q = 2.68$ çevresinde birkaç komşu moda
**yayılıyor.** Sonuçları:

- **İyi taraf:** k=2 bileşeni sıfırdan farklı, ölçülebilir. Nitekim
  $\kappa(M) \approx 186$ bunu kanıtlıyor — sistem çözülebilir.
- **Kötü taraf:** Response k=2'ye tam hizalı değil, biraz yayılmış.
  Eğer tune tam 2 olsaydı response saf k=2 olurdu ve $\kappa \approx 1$
  çıkardı. $Q=2.68$'de $\kappa \approx 186$, bu yayılmanın bedeli —
  ölçüm mümkün ama optimal değil.

**Sezgi (k=0 → ∞ limiti):** k=2'yi tek başına (k=0 yani DC karşısında)
çözmek matematiksel olarak en kolayı. Baza komşu harmonikler
($k=4, 6, \ldots$) eklendikçe çözüm zorlaşır ve doğruluk azalır, ama
hâlâ kabul edilebilir bir sonuç çıkabilir — çünkü ölçüm periyodik ve
modlar (yayılmış da olsa) birbirinden ayrışabiliyor. Başarısızlığın
**asıl** nedeni bu değil; Sorun 1 (rank) ve Sorun 3 (SNR).

### Sorun 3: Büyük arka plan harmonikleri SNR'ı boğuyor

Gerçek $\Delta q$ tek bir temiz harmonikten ibaret değil; ölçmek istediğimiz
küçük k=2 sinyalinin yanında çok daha büyük başka bileşenler var:

$$\Delta q = \underbrace{\Delta q_{k=2}}_{\sim 10\;\mu\text{m, aranan}}
+ \underbrace{\Delta q_\text{diğer}}_{\sim 100\text{–}300\;\mu\text{m}}$$

Her ikisi de $\Delta R$ üzerinden ölçüme karışır:

$$\Delta\mathbf{y} = \underbrace{\Delta R\,\Delta q_{k=2}}_{\text{aranan}}
                  + \underbrace{\Delta R\,\Delta q_\text{diğer}}_{\gg\,\text{sinyal}}$$

Arka plan katkısı sinyalden ~10× büyük ve $\Delta\mathbf{y}$'nin içinde
gerçek bir **sinyal** olarak görünüyor — gürültü gibi değil. Fit bunu
göremez; sadece $\Delta\mathbf{y} = M\hat{a}$'yı minimize eder ve arka
plan k=2 tahminine sızabilir.

**Bu, BPM elektronik gürültüsünden tamamen farklı bir sorundur.** BPM
gürültüsü (~1 μm) ölçüm cihazından kaynaklanır, zamanda rastgeledir ve
ortalama alarak (ya da aşırı belirlenmiş sistemle) bastırılabilir.
$\Delta R\,\Delta q_\text{diğer}$ ise quad kayma alanının **uzaysal**
yapısından kaynaklanır — daha fazla ölçüm yapmak bunu azaltmaz.

#### Lock-in neden burada doğrudan işe yaramaz?

Doğal bir mühendislik refleksi: "büyük gürültü altındaki küçük sinyal →
lock-in (faz-kilitli) amplifikatör." Lock-in, sinyali bilinen bir
frekansta modüle edip yanıtı o frekansta süzer; **zamanda** rastgele
gürültüyü bastırır. K-modülasyon bir anlamda bunu zaten yapıyor: gradyanı
değiştirip fark alarak DC'yi (BPM ofseti) ve zaman gürültüsünü atıyor.

Ama buradaki arka plan zamanda rastgele değil — quad konumlarının
**uzayda sabit** dağılımı. Ne zaman ölçersen ölç, $\Delta R\,\Delta q_\text{diğer}$
terimi orada duruyor. Bu yüzden klasik lock-in bu terimi süzemez.

#### Kritik gözlem: "rastgele" gürültünün içinde k=2 gömülü olabilir

Önceki testlerde arka plan **beyaz Gaussian gürültü** (100 μm RMS) olarak
verilmişti. Beyaz gürültünün gücü tüm Fourier modlarına eşit dağılır;
k=2 bileşeni $\approx 100/\sqrt{48} \approx 14$ μm. Bu, ölçmeye
çalıştığımız 10 μm'lik "asıl" k=2 sinyalinden **büyük.**

Yani fit başarısız olduğunda sandığımız şey: "100 μm gürültü altında
10 μm k=2 bulunamıyor." Gerçekte olan: fit hem 10 μm amaçlı hem de
14 μm rastgele k=2 içeriğini **birlikte** görüyor ve ikisini ayırt
edemiyor. Belki de zaten ölçmek istediğimiz büyüklüğü ölçüyorduk —
ama "doğru cevap" ile "kirlilik" karışmış durumda.

Bu nedenle daha temiz ve fiziksel bir test kurgusu: beyaz gürültü
yerine **yapılandırılmış harmonikler** ver. Örneğin k=2'yi küçük
(10 μm) tut, k=4, 6, 8'i büyük (200–300 μm) yap:

```json
"dy_harmonics": [
    {"k": 2, "amp_cos": 1e-5, "amp_sin": 0.0},
    {"k": 4, "amp_cos": 3e-4, "amp_sin": 0.0},
    {"k": 6, "amp_cos": 3e-4, "amp_sin": 0.0},
    {"k": 8, "amp_cos": 2e-4, "amp_sin": 0.0}
]
```

Bu kurguda k=2, diğer modlardan **frekansta temiz biçimde ayrı.**
Soru netleşiyor: hedefli baz {k=2} ile, veride 20–30× daha büyük
k=4,6,8 varken küçük k=2 çekilebilir mi? Bu, `reconstruction.py`'de
`recon_k_list_dy` anahtarıyla bazı truth'tan ayırarak test edilebilir
(bkz. §15). Mevcut `params.json` bu yapılandırılmış senaryoyu kullanıyor.

#### Sayısal sonuç: sızıntı testi (gerçek veri)

`recon_k_list_dy = [2]` ile baz yalnız {k=2} (baz boyutu = 2),
gerçek ise {k=2, 4, 6, 8}; k=2 = 10 μm @ φ=0.64, k=4,6,8 = 200–300 μm.
3-konfig yığma ile çalıştırınca:

```
baz ≠ gerçek (sızıntı)   3 konfig   TAM BELİRLİ   κ = 1.43
k=2:  19.5 μm @ ∠0.53   |   gerçek 10.0 μm @ ∠0.64
      %95 genlik hatası, faz Δ = 0.11 rad (iyi!)
Profil: RMS hata = 79.7 μm   korelasyon = 0.159
```

İki ders bir arada:

- **Genlik kirleniyor (%95 şişme):** k=4,6,8 katkıları sızarak k=2
  genliğini ~2× büyütüyor. Matematik:
  $$\hat{a}_\text{fit} = a_{2,\text{gerçek}} + \underbrace{(M_2^T M_2)^{-1} M_2^T M_{468}\,a_{468}}_{\text{k=4,6,8 → k=2 sızıntısı}}$$
  Kondisyon (κ=1.43) ve rank mükemmel olsa da bu sızıntı durdurulamıyor.

- **Faz şaşırtıcı biçimde iyi (Δ=0.11 rad):** Bu, önceki bir taslakta
  belirtilen "faz tamamen yanlış" iddiasını **düzeltir.** O sonuç,
  gerçek fazın tam 0 olduğu (saf cos) **dejenere** bir kurguda çıkmıştı:
  φ=0 noktasında en ufak sızıntı fazı uçurur. Gerçek faz 0.64 olunca
  sızıntı kabaca aynı fazda eklendiği için faz korunuyor, yalnız genlik
  şişiyor. **k=2 fazı gerçekten ölçülebilir bir büyüklük.**

**Ders:** İyi kondisyon + tam rank doğruluk için **gerekli ama yeterli
değil** — yanlış baz genliği kirletir. Ama faz, bazdan görece bağımsız
olarak çoğu kurguda korunur.

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

Daha çok harmonik çözmek istenirse (örn. k=2,4,6,8 → cos+sin = 8 katsayı)
daha çok bağımsız konfig gerekir: kabaca **bağımsız konfig sayısı ≥
çözülecek katsayı sayısı.** 8 katsayı için ≥8 tek-quad ölçümü (ya da
≥4 iki-quad). Bu, §7'deki "3-4 quad rank'ı artırır" gözleminin aynısı.

### Sayısal sonuçlar: 3-konfig, k=2,4,6,8

3 konfigürasyonu yığarak k=2,4,6,8 için (cos+sin = 8 katsayı)
çözüm denendi. Karşılaştırma için önce tek-konfig:

```
Tek konfig (baz = gerçek = k=2,4,6,8):
  rank = 2 / 8 → yetersiz belirlenmiş
  k=2:  ~2307 μm (gerçek 10 μm)   korelasyon ≈ −0.04
```

3-konfig yığma:

```
Yığılmış rank = 4 / 8 → hâlâ yetersiz belirlenmiş
k=6:  %21 genlik hatası   (300 μm sinyal, en büyük → en iyi)
k=2:  %511 genlik hatası  (10 μm sinyal, en küçük  → en kötü)
Profil: RMS hata = 378 μm   korelasyon = −0.105
```

Rank 4'e çıktı ama 8 katsayı için yetmiyor. K=2,4,6,8'in hepsini
tam çözmek için ≥8 bağımsız konfig gerekiyor. Dikkat çekici:
en büyük sinyal (k=6, 300 μm) rank yetersizliğinde bile en iyi
sonucu veriyor; en küçük sinyal (k=2, 10 μm) en kötüsünü.

### Sınır: rank'ı çözer, SNR'ı çözmez

Bu yığma **rank sorununu** çözüyor. §10'daki SNR sorununu (Sorun 3)
çözmüyor. Büyük arka plan harmonikleri varlığında her yeni kmod ölçümü
de aynı $\Delta q_\text{diğer}$'i taşıdığından, daha fazla konfigürasyon
eklemek bu paraziti azaltmıyor. Rank ve SNR iki ayrı sorundur:

| Sorun | Kaynak | Çözüm |
|-------|--------|-------|
| Rank yetersizliği | denklem < bilinmeyen | Çok-konfig yığma (bu bölüm) |
| SNR yetersizliği | arka plan ≫ sinyal | Çok-konfig işe yaramaz |

---

## 12b. Gürültü Altındaki Zayıf Harmoniği Ayıklamak: Neler İşe Yaramaz, CLEAN Ne Yapar?

§10–11'de gördük: büyük k=4,6,8 harmonikleri varken küçük k=2'yi
ölçmek SNR sorunu yaratıyor. Doğal bir refleks: "sinyal işleme
hilelerinden biri bunu kurtarabilir mi?" Önce neden bazı bariz
yöntemlerin **işe yaramadığını**, sonra neyin denenmeye değer olduğunu
açıklayalım.

### Neden doğrudan FFT çalışmaz?

Elimizdeki ölçüm $\Delta y$ değil, $\Delta R \cdot \Delta y$'dir:

$$\Delta\mathbf{y}_\text{ölçülen} = \Delta R \, \Delta q$$

$\Delta R$, Fourier modlarını **birbirine karıştıran** bir matris.
$\Delta q$ saf k=2 olsa bile $\Delta R\,\Delta q$ tek bir Fourier moduna
karşılık gelmez — tune frekansı çevresine yayılmış bir desendir (§10
Sorun 2). Dolayısıyla $\Delta\mathbf{y}$'nin FFT'si $\Delta q$'nun
FFT'si **değil**; araya $\Delta R$ girmiş. $\Delta R$ birim matris
olsaydı FFT mükemmel olurdu — ama değil.

### Neden sinyali katlayıp (folding) ortalama almak çalışmaz?

Periyodik katlama, **farklı periyottaki** rassal gürültüyü bastırmak
için kullanılır. Ama k=4, 6, 8 hepsi **k=2'nin tam katı** harmonikleri:
periyotları sırasıyla k=2'nin 1/2, 1/3, 1/4'ü. k=2 periyodunda katlarsan
bu harmonikler tam sayıda dönem tamamlar → **iptal olmaz, güçlenerek
kalır.** Katlama kendi harmoniklerini süzemez.

### Neden MUSIC / ESPRIT gibi yüksek-çözünürlük yöntemleri doğrudan çalışmaz?

Bu yöntemler veri $z = \sum_k A_k e^{i 2\pi f_k t}$ gibi **saf frekans
bileşenleri** içerdiğinde güçlüdür. Bizim $\Delta\mathbf{y}$'miz bu
form değil — $\Delta R$ bir konvolüsyon değil, tam matris çarpımı.
MUSIC $\Delta\mathbf{y}$'deki baskın yapıyı bulur ama bu $\Delta R$'nin
kendi tekil vektörlerine karşılık gelir, $\Delta q$'daki k=2'ye değil.

### CLEAN: dominant kaynağı soy, kalanı ölç

Radyo astronomisindeki **CLEAN** algoritmasının fiziği burada anlamlı:
en parlak kaynağı bul, çıkar, tekrarla. Bizim problemde:

```
artık r = Δy
döngü (her tur):
  her aday k için: r'yi yalnız k ile fit et, ne kadar düşürür?
  en çok düşüreni seç (örn. ilk turda büyük k=6)
  r ← r − gain · ΔR·F_k·â_k          (kesirli çıkarım, gain<1)
  biriktir: â_toplam[k] += gain · â_k
büyük harmonikler soyulunca artıkta zayıf k=2 ortaya çıkar
```

**CLEAN ≠ greedy.** Greedy bir harmoniği seçip onu tam taahhüt eder.
CLEAN ise **loop gain < 1** (tipik 0.1–0.3) ile her adımda dominant
harmoniğin yalnız bir kesrini çıkarır. Böylece bir moda erkenden tam
bağlanmaz; sonraki turlarda geri dönüp düzeltebilir. Mode-mixing olan
$\Delta R$ için bu daha sağlamdır.

### Gerçek veri sonucu: CLEAN k=2 için en iyisi

3-konfig yığma verisiyle (k=2 = 10 μm @ φ=0.64, k=4,6,8 = 200–300 μm)
üç yöntemin **k=2** kestirimi:

| Yöntem | k=2 genlik | k=2 faz | Profil kor. |
|--------|-----------|---------|-------------|
| Joint lstsq (baz=truth, 8 bilinmeyen) | 61.1 μm (%511) | ∠1.00 (Δ0.36) | −0.105 |
| Sızıntı (baz={2}, tam belirli) | 19.5 μm (%95) | ∠0.53 (Δ0.11) | 0.159 |
| **CLEAN** (gain=0.2) | **14.3 μm (%43)** | ∠0.47 (Δ0.18) | **0.260** |

Gerçek k=2 = 10 μm @ ∠0.64. CLEAN k=2 genliğini en yakın (%43 hata,
joint lstsq'nin %511'ine karşı) ve profil korelasyonunu en yüksek
(0.26) veriyor. Neden? CLEAN büyük harmonikleri (kusurlu da olsa)
soğurmaya çalıştığı için tüm sızıntıyı k=2'ye yıkmıyor — saf {k=2}
bazının yaptığının aksine.

### Önemli dürüstlük notu: CLEAN rank EKLEMEZ

CLEAN bir sihir değil. Ölçümün taşımadığı bilgiyi yaratamaz. Sınırlar:

- **Tam rankta** ($\Delta R$ tüm modları görüyorsa): CLEAN k=2'yi
  mükemmel ayıklar (sentetik kontrol testinde 9.98 μm / gerçek 10 μm).
- **Rank yetersizken** (3-konfig → rank 4, 8 bilinmeyen): büyük
  harmonikleri (k=4,6,8) düzgün bulamaz — gerçek veride k=6 için
  112 μm verdi (gerçek 300). Hiçbir algoritma 4 bağımsız denklemle
  8 bilinmeyeni tam çözemez.

Ama bizim **hedefimiz k=2** ve onu CLEAN makul ölçüyor: büyükleri
imperfect soğurması bile k=2 sızıntısını joint lstsq'ye göre büyük
ölçüde azaltıyor. Yani CLEAN'in faydası tam da bu senaryoda ortaya
çıkıyor.

> **Sonuç:** CLEAN, "önce büyük katkıları modelle-çıkar, sonra zayıf
> sinyali ölç" stratejisinin disiplinli halidir. Gerçek veride k=2
> için en iyi sonucu verdi (%43 genlik, 0.18 rad faz). Büyük
> harmonikleri tam çözmek hâlâ rank gerektiriyor (§12 çok-konfig),
> ama hedef yalnız k=2 ise CLEAN pratik bir kazanç sağlıyor.

### Yalnız k=2 isteniyorsa: null-steering ve garantili çözüm

Çoğu zaman k=4,6,8'i umursamıyoruz — tek istediğimiz **k=2'yi**, üstelik
diğer harmoniklerin bilinmeyen sin/cos değerlerinden **bağımsız** ölçmek.
Bunun teorik olarak doğru aracı **null-steering** (dizi işlemedeki MVDR)
estimatörüdür:

$$
\hat{a}_2 = w^T \Delta\mathbf{y}, \qquad
\begin{cases}
w^T (\Delta R\,F_2) = 1 & \text{(k=2'ye birim tepki)} \\
w^T (\Delta R\,F_k) = 0,\ k=4,6,8 & \text{(kontaminantlara sıfır tepki)}
\end{cases}
$$

k=4,6,8'in cos+sin = 6 bileşenini **null'lamak** + k=2'nin 2 bileşenini
tutmak = **8 kısıt.** Bunu sağlamak için yığılmış $\Delta R$'nin bu 8
boyutu gerçekten görmesi, yani **rank ≥ 8** olması gerekir.

Pratikte bu, "diğer harmonikleri baza dahil et, birlikte fit et, ama
yalnız k=2 raporla" demektir. Sağlamlığın ölçüsü **çözünürlük matrisi**
$R = M^+ M$'in k=2 satırındaki nuisance girdileridir (sızıntı):

| Rank | Nuisance sızıntısı | k=2 sonucu (sentetik test) |
|------|--------------------|-----------------------------|
| 8/8 (≥4 iki-quad konfig) | **0.000** | **%0 hata, faz tam** — kontaminanttan BAĞIMSIZ |
| 6/8 | 0.501 | %208 hata — hâlâ fazlara bağımlı |
| 4/8 (mevcut 3-konfig) | büyük | sızıntı baskın |

**Sonuç:** k=2'yi kontaminant fazlarından tam bağımsız ölçmenin
**garantili** yolu rank'ı ≥ 8'e çıkarmaktır — yani ≥8 bağımsız tek-quad
(ya da ≥4 iki-quad) kmod konfigürasyonu. O noktada null-steering k=2'yi,
k=4,6,8 ne olursa olsun, tam verir. Rank yetersizken (mevcut durum) en
iyi yaklaşık CLEAN'dir. `fourier_reconstruct.py` her iki yöntemi de basar
ve sızıntı metriğiyle ne kadar uzakta olduğunu gösterir.

### Tasarım kuralı: kaç quad gerekli?

Ulaşılabilir rank, **modüle edilebilen farklı quad sayısına** eşittir
(aynı quad'ı farklı gradyenle çalıştırmak rank eklemez — bilgi quad'ın
faz konumundan gelir, gradyen değerinden değil). Buradan basit bir
tasarım kuralı çıkar:

$$
\boxed{\text{gereken rank} = \underbrace{2}_{k=2\ (\cos,\sin)}
+ 2 \times (\text{bastırılacak kontaminant harmonik sayısı})}
$$

| Bastırılacak kontaminant | Gereken rank | Gereken farklı quad |
|--------------------------|--------------|---------------------|
| Yok (kontaminant ihmal edilebilir) | 2 | 2 |
| Yalnız k=4 | 4 | 4 |
| k=4, k=6 | 6 | 6 |
| k=4, k=6, k=8 | 8 | 8 |

Bunun pratik sonucu, az sayıda quad varken serttir. Aşağıdaki tablo,
kontaminant fazları **keyfi** (saf cos değil, rastgele) iken k=2
kestiriminin rank ile nasıl değiştiğini gösterir (sentetik, 40 rastgele
geometri ortalaması; gerçek k=2 = 10 μm @ 0.64):

| Rank ≈ quad | Sızıntı | k=2 genlik | Hata | Faz hatası |
|-------------|---------|-----------|------|------------|
| 2 | 0.51 | 76 μm | %663 | 1.40 rad |
| 3 | 0.58 | 95 μm | %848 | 1.45 rad |
| 4 | 0.60 | 84 μm | %739 | 1.48 rad |
| 6 | 0.51 | 79 μm | %692 | 1.51 rad |
| **8** | **0.00** | **10.0 μm** | **%0** | **0.00 rad** |

Dikkat: rank 8'in altında kademeli bir iyileşme **yok** — keyfi
kontaminant fazlarında k=2 ya tam çözülür (rank 8) ya da güvenilmezdir.
Daha önce sızıntı testinde fazı iyi (Δ0.11 rad) bulmamız, kontaminantların
o senaryoda saf cos (hizalı faz) olmasındandı; rastgele fazda bu şans yok.

**Az quad varsa ne yapılabilir:**
- **2 quad:** k=2'yi yalnızca *tespit* eder (var/yok), kaba bir genlik
  üst-sınırı verir. Hassas ölçüm değil.
- **3 quad:** Tek bir kontaminant harmoniğini bile (2 boyut ister) tam
  null'layamaz; 2 quad'dan kayda değer fark yok.
- **Tek kaçış — dış bilgi:** k=4,6,8 başka bir ölçüm/modelden bilinirse
  katkıları çıkarılıp 2 quad ile k=2 kurtarılabilir, ama bu "yalnız
  veriden" çözüm olmaktan çıkar.

---

## 12c. Tek Orbit, R-Tabanlı CLEAN

> **Bağlam.** §4–12b'deki k-modülasyon (kmod) yaklaşımı, gradyan
> değiştirip fark alarak BPM ofsetini iptal ediyordu. Bu bölüm **kmod'u
> tamamen terk ediyor** ve tek bir orbit ölçümüyle, BPM ofsetiyle birlikte
> çalışıyor. Fiziksel temel: k=2 harmonik tune rezonansı nedeniyle orbit
> yanıtında ~34× güçlenirken BPM ofseti böyle bir güçlenme almaz — bu
> asimetri k=2'nin doğrudan çekilmesini mümkün kılar.

### Fikir: y = R·Δq + b'yi olduğu gibi fit et

Kmod yaklaşımında b'yi sıfırlamak için Δy = ΔR·Δq denklemine geçiliyordu.
Yeni yaklaşımda b iptal edilmiyor — direkt fit ediliyor:

$$\mathbf{y} = R\,\Delta q + \mathbf{b}$$

$\Delta q = F\hat{a}$ Fourier parametreleştirmesiyle:

$$\mathbf{y} = \underbrace{R\,F}_{M}\,\hat{a} + \mathbf{b}$$

BPM ofseti $\mathbf{b}$ beyaz bir vektör (her BPM'e bağımsız ~300 μm),
$M = R\cdot F$ ise büyük tekil değerler taşıyan yapılı bir matris.
**Fark:** $\mathbf{b}$ rastgele iken $M\hat{a}$ tutarlı bir desen —
en küçük kareler $\hat{a}$'yı $\mathbf{b}$'den doğal olarak ayrıştırabilir,
eğer $M$'nin büyük tekil değerleri $\|\mathbf{b}\|$'den çok büyükse.

İşte tune rezonansı burada devreye girer.

### Tune rezonansı: k=2'nin beklenmedik avantajı

Bir quad $j$'yi 1 m kaydırdığımızda tüm BPM'lerdeki orbit yanıtı:

$$R_{ij} \propto \frac{\sqrt{\beta_i\beta_j}}{2\sin(\pi Q)}\cos(|\phi_i - \phi_j| - \pi Q)$$

$Q \approx 2.68$ için $\sin(\pi \cdot 2.68) = \sin(0.68\pi) \approx 0.891$;
ama k=2 Fourier modu quad fazları boyunca $2\pi\cdot2\cdot j/48$'de salınıyor —
tune Q=2.68'e yakın. Bu yakınlık **rezonant güçlenme** üretir:

| Büyüklük | Değer |
|----------|-------|
| k=2 misalignment | 10 μm |
| k=2'nin yarattığı orbit normu ‖R·Δq(k=2)‖ | 1668 μm |
| Tüm harmoniklerin (k=2,4,6,8) orbit normu ‖R·Δq‖ | 5268 μm |
| BPM ofseti normu ‖b‖ (σ=100 μm) | 611 μm |
| k=2 güçlenme faktörü (tune yakınlığı) | **~34×** |

Karşılaştırma: k=2 modu 10 μm misalignment'tan 1668 μm orbit sinyali üretiyor.
BPM ofseti 100 μm'den yalnızca 611 μm orbit etkisi yapıyor — ve bu etki
**tüm frekanslara yayılmış** beyaz bir yapı. k=2'nin yarattığı 1668 μm,
b'nin k=2 bileşeninden (~0.7 μm) ~2400× büyük.

**Sonuç:** kmod'a gerek yok. R·Δq ile b yapısal olarak ayrışıyor.

### Mekanizma: üç etki birlikte çalışıyor

k=2'nin, kendisinden 20–30× büyük k=4,6,8 ve 100 μm BPM ofseti yanında
%0.6 hatayla çözülmesi tek bir hileden değil, üç bağımsız mekanizmadan
geliyor. Hepsi doğrudan R'den hesaplanabilir (ayrıntılı türetim:
`FOURIER_REKONSTRUKSIYON.md` §13c).

**1. Tune rezonansı modları seçici güçlendiriyor.** Her modun R kazancı
$\|RF_k\|$ tune yakınlığıyla belirleniyor:

| k | 1 | **2** | 3 | 4 | 6 | 8 |
|---|---|-------|---|---|---|---|
| R kazancı | 8.8 | **34.1** | 8.9 | 3.2 | 1.1 | 0.59 |

k=2, k=4'ten 10×, k=8'den 58× güçlü orbit üretir.

**2. Güçlenme genlik dezavantajını siliyor.** Misalignment'ta k=4,6,8
30× büyük olsa da orbite çevrilince k=2 baskın hale gelir:

| k | misalignment | → orbit normu |
|---|--------------|---------------|
| 2 | 10 μm | 1669 μm |
| 4 | 300 μm | 4707 μm |
| 6 | 300 μm | 1651 μm |
| 8 | 200 μm | 577 μm |

k=8'in 20× büyük misalignment'ı k=2'den küçük orbit veriyor.

**3. Modlar R altında dik → karışmıyorlar.** Çapraz korelasyon
$|\langle RF_2, RF_{4,6,8}\rangle| \approx 0.01$. En küçük kareler k=2'yi
diğerlerinden temiz ayırır; büyük k'lar k=2 kestirimine sızmaz. (kmod'daki
sızıntının tersi: orada ΔR karıştırır, burada R ayırır.)

### Birim hatası: kritik bir bulgu

Bu analiz yapılırken `build_response_matrix.py`'de kritik bir birim hatası
keşfedildi:

- `integrator.cpp` satır 533–534: orbit verisini **milimetre** olarak yazar
  (`x_mm`, `y_mm` başlıklı `cod_data.txt`; ×1000 çarpımı)
- `read_cod_quads` fonksiyonu bu mm değerlerini **metre** olarak okuyordu

Sonuç: R = orbit[mm]/misalign[m] → **R 1000× şişmiş halde.**

```
Hata öncesi:   max|R| ≈ 1950, σ_max(R) ≈ 34729
Hata sonrası:  max|R| ≈ 1.95,  σ_max(R) ≈ 34.73   κ = 249 (değişmez)
```

κ ve rank ölçek-bağımsız olduğundan kmod sonuçları etkilenmedi
(1000× hem payda hem paydada iptal olur). Ama BPM ofseti testleri
tamamen yanlış ölçekte çalışıyordu:

| | Hata öncesi | Hata sonrası |
|--|-------------|--------------|
| ‖R·Δq‖ | ~5.27 m | 5.27 mm |
| ‖b‖ (σ=100 μm) | 611 μm | 611 μm |
| Sinyal/ofset oranı | **8600×** (anlamsız) | **8.6×** (gerçekçi) |

**Düzeltme:** `build_response_matrix.py` içindeki `read_cod_quads` fonksiyonuna
`cd[:, 1:3] *= 1e-3` satırı eklendi. Doküman başlığı ve türetilmiş matrisler
(R_dy_1.npy vb.) yeniden hesaplandı.

### CLEAN uygulaması: oracle olmadan harmonik tespiti

kmod testlerinden farklı olarak, CLEAN'e **hangi k'ların gerçek olduğu
söylenmiyor.** Aday küme: k=1,2,...,12 (tamamı). CLEAN bunların arasından
dominant harmonikleri sırayla buluyor ve çıkarıyor.

Test kurgusu:
- **Gerçek misalignment:** k=2 (10 μm) + k=4,6,8 (200–300 μm) — antisimetrik FODO
- **BPM ofseti:** her BPM'e bağımsız Gaussian, σ = 100 μm
- **Aday kümesi:** k=1..12 (oracle bilgisi yok)
- **50 Monte Carlo deneyi**

#### TANI 1: sinyal ve ofset güçleri

$$\|R\cdot\Delta q_\text{gerçek}\| = 5268\;\mu\text{m}, \quad
\|b\| = 611\;\mu\text{m}, \quad
\|R\cdot\Delta q_{k=2}\| = 1668\;\mu\text{m}$$

k=2 bileşeni bile tek başına BPM ofsetinden ~2.7× büyük.

#### TANI 2: b'den gelen sahte harmonik

Saf b (Δq = 0) verilince CLEAN ne buluyor?

$$\text{Sahte } k{=}2 \text{ genliği} = 0.722 \pm 0.410\;\mu\text{m}$$

b gerçekten sisteme sızıyor ama küçük düzeyde. 10 μm gerçek sinyale
karşı 0.7 μm sahte → **SNR ≈ 14** (yeterli).

#### Ana sonuç: k=2 kestirimi

```
k=2: 9.992 ± 0.578 μm   (gerçek: 10.000 μm)
     %0.08 genlik hatası   0.055 rad faz hatası
```

100 μm BPM ofseti altında, oracle bilgisi olmadan, tek orbit ölçümüyle:
**10 μm hedefe ulaşıldı.**

#### TANI 3: σ_b taraması

| σ_b | k=2 hatası | Sahte k=2 |
|-----|-----------|-----------|
| 0 μm | ~0 | ~0 |
| 50 μm | ~0.3 μm | ~0.4 μm |
| 100 μm | ~0.6 μm | ~0.7 μm |
| 200 μm | ~1.2 μm | ~1.5 μm |
| 300 μm | ~1.8 μm | ~2.2 μm |

Ölçek doğrusal — b sızıntısı σ_b ile orantılı. Mekanik olarak beklenen
~300 μm BPM ofseti için k=2 hatası ~2 μm — 10 μm hedefinin altında.

### Önemli sınır: sahte harmonikler problemi

CLEAN k=1..12 aday kümesiyle çalışırken, BPM ofseti tüm frekanslara
eşit güç dağıtır. Bu nedenle CLEAN **her aday k'yı gerçekmiş gibi görür:**

```
Bulunan k'lar (>1 μm): k=1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
```

k=2, 4, 6, 8 gerçek harmonikler — ama CLEAN'in k=1, 3, 5, 7, 9, 10, 11, 12
için de >1 μm bulması, bunların b gürültüsünden gelen sahte harmonikler
olduğunu gösteriyor.

**Sorun:** Oracle bilgisi olmadan gerçek k=2'yi sahte bir k=3'ten
nasıl ayırt edebiliriz?

- **Fiziksel ön bilgi:** Quad hizalama hataları uzun dalgalı olduğundan
  düşük k dominant beklenir. Bu k=2'yi önceliklendirir ama kesin değil.
- **TANI 2 testi:** Aynı ölçüm geometrisinde saf b'den gelen sahte
  harmonik genliği kalibre edilirse eşik belirlenebilir.
- **Çok-orbit ortalama:** Farklı zamanlarda ölçülen orbitlerden b
  zamansal değişim gösterirse, ortalama b'yi bastırır ama Δq sabit kalır.

Bu açık problem §18'de kaydedildi.

### Model hatası etkisi

R matrisi simülasyondan geliyor; gerçek halkada gradyan hataları
(β-beat) R'yi bozabilir. Gradyan hatasının R kestirimine etkisi:

$$R_\text{model} = R_\text{gerçek}\cdot\mathrm{diag}(1 + \varepsilon), \quad
\varepsilon \sim \mathcal{N}(0, \sigma_\text{model})$$

| δK/K | k=2 per-quad hatası |
|------|---------------------|
| 0% | ~0.1 μm |
| 1% | ~0.5 μm |
| 2% | ~1.5 μm |
| 3% | ~3.5 μm |
| **4%** | **~10 μm** (sınır) |
| 5% | ~15 μm |

Pratik sonuç: R modelinin gradyan doğruluğu **δK/K ≲ 3–4%** olduğunda
10 μm per-quad hedefi karşılanabilir. Bu, β-beat ölçümüyle gerçekleştirilebilir
bir gereksinim.

### Kmod ile karşılaştırma

| Özellik | kmod (ΔR-tabanlı) | R-tabanlı CLEAN (bu bölüm) |
|---------|------------------|-----------------------------|
| BPM ofseti | Tamamen iptal (fark) | Ayrışıyor (tune güçlenmesiyle) |
| Orbit sayısı | 2 (nominal + pert.) | **1** |
| Gradyan değişimi | Gerekli | **Yok** |
| k=2 hatası | ~%43 (§12b CLEAN) | **%0.08** |
| Sahte harmonik | k seçimiyle kontrol edilebilir | **Tüm k=1..12 görünüyor** |
| Rank sorunları | Her kmod konfigürasyonu | Yok (R tam rank ~249 κ) |

---

## 13. Nerede Duruyoruz? Fiziksel Bir Değerlendirme

### Sonuçların özeti

| Yöntem | k=2 hatası | Koşul | Not |
|--------|------|-------|-----|
| Uniform kmod | ~6.6 μm | Tüm 48 quad eşzamanlı | Pratik uygulama zor |
| Drift modu | ~6.5 μm | Herhangi orbit ölçümü | Yalnız zamansal değişimler |
| Hedefli Fourier (idealize) | ~0.02 μm | Baz=gerçek={k=2,4}, sin=0 | Gerçekçi değil |
| 3-konfig joint lstsq (baz=truth, 8 bilinmeyen) | %511 | rank=4/8, underdetermined | Bilgiyi 8 katsayıya saçıyor |
| 3-konfig sızıntı (baz={k=2}) | %95 genlik, Δφ=0.11 | κ=1.43, tam belirli | Genlik şişer, faz iyi |
| **3-konfig CLEAN** | **%43 genlik, Δφ=0.18** | gain=0.2 | k=2 için en iyi (kmod yöntemi) |
| **R-tabanlı CLEAN (tek orbit)** | **%0.08, Δφ=0.055 rad** | σ_b=100 μm, k=1..12 aday | Kmod yok; sahte harmonik sorunu var |

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
| `reconstruction.py` | Hedefli, Greedy, LASSO, çok-konfig rekonstrüksiyon (tam çıktı) |
| `fourier_reconstruct.py` | Temiz Fourier kalite raporu: hedefli fit + CLEAN (LASSO/greedy yok) |
| `scan_j2.py` | En iyi j₂ quad çiftini tara |
| `show_response.py` | Tepki matrisi görselleştirme |
| `bpm_offset_test.py` | R-tabanlı CLEAN testi: BPM ofseti altında tek orbit ölçümü (§12c) |
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

### Yapılandırılmış arka plan + sızıntı testi

Beyaz gürültü yerine, k=2 küçük (10 μm), k=4,6,8 büyük (200–300 μm).
Soru: hedefli baz {k=2} ile küçük k=2 çekilebilir mi?

```bash
# params.json: dy_harmonics = k=2 (1e-5), k=4,6,8 (büyük); dy_random_RMS=0
python3 build_response_matrix.py
python3 test_kmod_reconstruction.py
python3 reconstruction.py        # baz = truth (k=2,4,6,8) → 8 katsayı, rank-2'de underdetermined
```

Bazı truth'tan ayırıp yalnız k=2 ile rekonstrüksiyon (sızıntı/kontaminasyon
testi) için `params.json`'a şu satırı ekle:

```json
"recon_k_list_dy": [2]
```

Bu, veride k=4,6,8 olsa bile rekonstrüksiyon bazını yalnız k=2'den kurar.
`reconstruction.py` çıktısında "UYARI: baz ≠ gerçek harmonikler" görünür
ve k=2 katsayısının büyük arka plandan ne kadar etkilendiği raporlanır.

### Temiz kalite raporu + CLEAN

Karmaşık `reconstruction.py` çıktısı yerine sade bir genlik/faz/hata
tablosu için:

```bash
python3 fourier_reconstruct.py
```

Bu script LASSO/greedy basmaz; yalnız (1) hedefli Fourier fit, (2) varsa
sızıntı testi, (3) CLEAN hiyerarşik çıkarımı tablolar. Her k için
tahmin genliği/fazı, gerçek değer, %hata ve faz farkını gösterir.

CLEAN parametreleri (`params.json`, opsiyonel):

```json
"clean_gain": 0.2,
"clean_max_iter": 200,
"clean_candidates_dy": [2, 4, 6, 8]
```

`clean_gain` loop gain (0.1–0.3 tipik). `clean_candidates_dy` verilmezse
gerçek harmonikler aday kümesi olur.

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

### Hizalama hatası üretimi (truth)

| Anahtar | Anlam |
|---------|-------|
| `dy_harmonics` | `[{k, amp_cos, amp_sin}, ...]` — üretilen gerçek hata bileşenleri |
| `dy_random_RMS` | Beyaz Gaussian arka plan RMS (m). Yapılandırılmış test için 0 |
| `smooth_antisym_fodo` | `true`: FODO antisimetrik (önerilen, fiziksel) |

> Mevcut varsayılan: `dy_harmonics` = k=2 (10 μm) + k=4,6,8 (200–300 μm),
> `dy_random_RMS` = 0. Bu, beyaz gürültü yerine yapılandırılmış arka plan
> testidir (§10, Sorun 3).

### Rekonstrüksiyon algoritması

| Anahtar | Anlam |
|---------|-------|
| `recon_k_list_dy` | (Opsiyonel) Hedefli fit bazını truth'tan ayırır. Örn. `[2]` → veride k=4,6,8 olsa bile baz yalnız k=2. Yoksa baz = truth harmonikleri |
| `recon_k_list_dx` | Aynısı yatay için |
| `k_search_max` | Greedy aramasının üst sınırı (Nyquist = 12) |
| `greedy_residual_threshold` | Minimum oransal rezidüel düşüşü (tipik: 0.01–0.05) |
| `max_harmonics` | Greedy maksimum harmonik sayısı |
| `lasso_lambda` | LASSO L1 ceza katsayısı |
| `clean_gain` | (`fourier_reconstruct.py`) CLEAN loop gain, tipik 0.1–0.3 |
| `clean_max_iter` | CLEAN maksimum iterasyon sayısı |
| `clean_candidates_dy` | CLEAN aday harmonik kümesi; yoksa = truth harmonikleri |

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

- **k=2,4,6,8 için tam belirlenmiş sistem:** 3-konfig → rank=4/8,
  k=2 için %511 hata. Tam çözüm için ≥8 bağımsız tek-quad konfig
  gerekiyor. Ancak sızıntı testi gösterdi ki rank=baz olsa bile büyük
  arka plan harmonikleri k=2 doğruluğunu bozuyor — rank yeterli koşul
  değil.

- **Beta-beat etkisi:** Gerçek halkada $R$ model $R$'den sapınca
  ne oluyor? %1 beta-beat → kaç μm ek hata? (Test 8, sürmekte.)

- **Falso EDM hedef vektörü:** $w = (w_1, \ldots, w_{48})$ vektörünü
  hesapla ($K_j$, $\beta_j$, $R$ kullanarak), hangi quad en iyi hizalanıyor?
  Tek ölçümle falso EDM katkısını doğrudan hedefleme denenebilir.

- **Greedy + çok-konfig:** Rank ≥ 3 ile greedy harmonikleri doğru
  seçiyor mu? Sistematik test yapılmadı.

- **Bootstrap hata çubukları:** Katsayı belirsizliği nasıl ölçülür?

- **R-tabanlı CLEAN sahte harmonik ayrımı (§12c):** CLEAN k=1..12 aday
  kümesiyle çalışırken b gürültüsü tüm frekanslara sahte harmonik sızıntısı
  yapıyor. Gerçek (k=2,4,6,8) ile sahte (b'den gelen k=1,3,5,...) arasında
  oracle bilgisi olmadan güvenilir bir ayrım yapılabilir mi? TANI 2 testi
  (saf b altında sahte genlik kalibrasyonu) bir eşik belirleme yolu sunuyor
  ama sistematik doğrulama yapılmadı.

- **Çok-orbit R-tabanlı yöntem:** Farklı zamanlarda ölçülen birden çok
  orbit verisini birleştirmek b'nin zamansal değişimini kullanarak ofseti
  bastırabilir. Δq sabit kalırken b değişirse, ortalama b etkisini azaltır.
  Bu yaklaşım §12c CLEAN ile birleştirilmemiş.

- **R model doğruluğu alt sınırı:** §12c'de δK/K ≲ 3–4% eşiği saptandı.
  Gerçek halkada β-beat ölçümünden R'yi ne kadar iyi kalibrasyon yapılabilir?
  LOCO benzeri yöntemlerle R kalibrasyonu henüz uygulanmadı.
