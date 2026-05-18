# Metot: pEDM Halkasında Quad Hizalama Hatalarının BPM Tabanlı Ölçümü

Bu doküman, makalenin 3., 4. ve 5. bölümlerinin pedagojik açıklamasıdır.
Amaç: konuyu ilk defa duyan birinin de takip edebilmesi. Önce motivasyon,
sonra aday yöntemler, sonra sistematik testler, sonra önerilen işletme modu.

**Çalışmanın çerçevesi.** Bu çalışma yeni bir yöntem icadı değildir; pEDM
halkasının özel koşullarında klasik tepki-matrisi yöntemlerinin **hangi
sınırlarda 10 μm hizalama hassasiyetine ulaşabildiğini** sistematik olarak
inceleyen bir mühendislik değerlendirmesidir. Paralel bir çalışmada aynı
problem yapay sinir ağı yaklaşımıyla da ele alınmaktadır; iki yaklaşımı
adil biçimde karşılaştırmak için ortak bir referans çerçevesi gereklidir.
Bu doküman o çerçeveyi kuruyor.

---

## 1. Problem motivasyonu

### 1.1 pEDM'de hizalama neden kritik?

Orijinal proton EDM (pEDM) önerilerinde halka tamamen elektrostatik ve
**zayıf odaklamalı** idi. Bu tasarımda hassas sistematik olarak ortalama
manyetik alan ($\langle B\rangle$) hedef alınmıştı. Sonraki nesil tasarımda,
hüzme dinamiğini iyileştirmek için **manyetik quadrupole'larla alternating
gradient** (FODO) odaklamasına geçildi. Bu değişiklikle:

- Ortalama manyetik alan hâlâ bir sistematik kaynağıdır, ancak
- Asıl ölçülmesi gereken sistematik artık **quad hizalama hataları**dır.

Çünkü hizalanmamış bir quad, EDM ölçümünü taklit edebilecek vertikal ortalama
alan üretir. pEDM'in hedef hassasiyeti, **hizalama hatalarının 10 μm
seviyesinde bilinmesini** gerektirir.

### 1.2 Problem matematiği

Halkada 48 quadrupole var. Her biri ideal konumundan yatay (`dx`) ve dikey
(`dy`) ekseninde mikronlar mertebesinde kaymış olabilir. Bu kaymalar hüzmenin
kapalı yörüngesini bozar; bozulma BPM'lerle ölçülür. Lineer rejimde:

$$ \mathbf{y} = R\,\Delta q + \mathbf{b} + \boldsymbol{\eta}. $$

- $\Delta q$ ∈ ℝ⁴⁸: misalignment vektörü (bilinmeyen).
- $R$: yanıt matrisi — örgü optiğinden Courant-Snyder formülüyle hesaplanır.
- $\mathbf{b}$: BPM elektronik ofseti — sensör başına on mikronlar mertebesinde,
  zaman içinde yavaş değişen.
- $\boldsymbol{\eta}$: ölçüm gürültüsü — tek-okuma RMS ~1 μm.

Yatay ve dikey düzlemler arasındaki kuplaj ihmal edilebilir olduğu için
problemi iki bağımsız 48×48 sisteme ayırıyoruz.

### 1.3 Niye konvansiyonel k-modülasyon doğrudan uygulanamıyor?

Sabit-β depolama halkalarında klasik k-modülasyonu (Lee 2004, vd.) tek bir
quad'ın gradient'ini titretip o quad'taki hüzmenin tepkisinden hizalama hatasını
bulur. Bu yöntem şu varsayımlara dayanır:

1. $\beta$ fonksiyonu quad konumunda yerel olarak bilinir ve sabittir.
2. Modülasyon frekansı tek bir tune'dan iyi ayrılır.
3. Tek-quad modülasyonu örgünün geri kalanını etkilemez.

pEDM örgüsü bu üç varsayımı **birden** zorlar:
- 48 quad'ın her birinde β farklıdır (FODO ile değişken),
- Halkayı işletim sırasında onlarca farklı frekansta modüle etmek
  sistematik temizliğini bozar,
- Manyetik quad'lar birbirlerine eşli (coupled) tasarımdadır.

Bu yüzden klasik yerel k-modülasyon yerine **küresel tepki matrisi tabanlı
yaklaşımları** araştırıyoruz. Aşağıda bu yaklaşımların aday formlarını
listeleyip her birini sistematik olarak ölçüyoruz.

### 1.4 Aday yöntemler

Tepki matrisi formalizmiyle 96 bilinmeyenli sistemi çözmenin akla gelen
yolları:

| # | Yöntem | Çözdüğü |
|---|---|---|
| (i) | Tek-gradient direct: $\widehat{\Delta q} = R^{-1}\mathbf{y}$ | Mutlak hizalama |
| (ii) | İki-gradient $\Delta R$: $\widehat{\Delta q} = \Delta R^{-1}(\mathbf{y}_1-\mathbf{y}_2)$ | Mutlak hizalama (k-mod ruhu) |
| (iii) | $\Delta R$ + düzenlileştirme (Tikhonov, TSVD) | Mutlak, gürültü dengeli |
| (iv) | Drift modu: $\widehat{\delta q}(t) = R^{-1}(\mathbf{y}(t)-\mathbf{y}_0)$ | Kalibrasyondan beri **değişim** |
| (v) | Yapay sinir ağı (paralel çalışma) | Mutlak/drift |

Her yöntem farklı bir hata kaynağına (BPM ofseti, gürültü, model uyumsuzluğu,
tilt'ler) karşı farklı davranıyor. Geri kalan bölümler bu davranışı niceliyor.

### 1.5 Bir özet beklentisi

Aşağıdaki bölümlerin sonunda göstereceğimiz tablo şu olacak:

| Yöntem | Mutlak hassasiyet | Drift hassasiyeti | Online uygulanabilir? |
|---|---|---|---|
| (i) Direct $R^{-1}$ | Ofset baskın → 100+ μm | — | Evet |
| (ii) $\Delta R^{-1}$ direkt | ~6 μm (gürültüsüz) / ~1900 μm (gürültülü) | ~1000 μm | Hayır (fizik bozar) |
| (iii) Düzenlileştirme | ~50 μm (bias) | ~50 μm | Kısmen |
| (iv) Drift modu | Uygulanamaz | **~6 μm** | **Evet** |

10 μm hedefine ulaşan tek pratik klasik yol drift modu görünüyor. Bu
çalışmanın esas katkısı, bu sonucu **sistematik olarak göstermek** ve
neden böyle olduğunu yapısal nedenlerle açıklamak.

---

## 2. Yöntemleri sınırlayan yapısal kısıt: ofset–gürültü ikilemi

Bu bölümün amacı, yukarıdaki aday yöntemlerin (ii) ve (iii) seçeneklerinin
neden **belli bir taban hata seviyesinin altına inemediğini** matematiksel
olarak göstermek. Bu sınır, herhangi bir spesifik algoritmaya değil,
problemin yapısına özgü.

### 2.1 Sezgi: niye $\Delta R$'yi terslemek tehlikeli?

$R_1$ ve $R_2$ arasındaki fark ne kadar küçükse, $\Delta R$ o kadar "yassı"
olur — yani çoğu yönde küçük sayıların matrisidir. Küçük sayıların matrisini
terslemek, küçük sayılarla bölmek demektir; bu da gürültüyü büyütür.

Niceleyim. Eğer $\varepsilon \to 0$ ise $\Delta R \to 0$. O zaman $\Delta R^{-1}$
patlar. Pratikte $\varepsilon \approx 0.02$ alıyoruz (gradient'i %2 değiştiriyoruz);
bu durumda

$$ \|\Delta R^{-1}\| \sim \frac{1}{\varepsilon}\,\|R^{-1}\|. $$

Yani gürültü, klasik $R^{-1}$ tersine göre **~50 kat** büyür. Bu çok büyük bir
bedel.

### 2.2 Şu gözlemi tartışılmaz biçimde söyleyelim

İddia: BPM ofsetini iptal eden, ön-yargısız (unbiased), lineer her estimator,
mutlaka $O(1/\varepsilon)$ gürültü büyütür.

Bunu hızlıca türetelim. Genel lineer estimator şu biçimde olmalı:
$\widehat{\Delta q} = A_1\mathbf{y}_1 + A_2\mathbf{y}_2$.

**Ön-yargısızlık koşulu.** Her $\Delta q$ için doğru cevabı versin:
$A_1 R_1 + A_2 R_2 = I$.

**Ofset iptali koşulu.** Hiçbir $\mathbf{b}$ tahmine sızmasın:
$A_1 + A_2 = 0 \implies A_2 = -A_1$.

İki koşulu birleştirin: $A_1(R_1-R_2) = I$, dolayısıyla $A_1 = \Delta R^{-1}$.
Yani ofseti iptal eden ön-yargısız tek bir estimator var ve o da $\Delta R^{-1}$
estimatörüdür. Bu da $\sim 1/\varepsilon$ kat gürültü büyütür.

**Bu sonucun adı: ofset–gürültü düalitesi.** Ofset iptali ve gürültü sadakati
lineer cebir düzeyinde birbirine karşıttır. Birinden ödün vermeden öbürünü
elde edemezsiniz.

### 2.3 Düzenlileştirme (regularization) bir kaçış yolu mu?

İlk akla gelen kurtuluş: $\Delta R^{-1}$ doğrudan yerine "yumuşatılmış" terslerini
kullanmak. İki popüler yöntem:

- **Tikhonov:** $(\Delta R^\top \Delta R + \lambda I)^{-1}\Delta R^\top$
- **TSVD:** SVD'nin sadece en büyük $k$ tekil değerini tutmak

Bunlar gürültüyü gerçekten azaltır. Ama bedava değil: $\Delta R$'nin küçük
tekil değerleri, misalignment'in **yüksek frekanslı uzaysal modlarına** karşılık
gelir. Onları söndürdüğünüzde, gerçek sinyalden de o modlar gider.

Sezgi: pürüzlü bir resmi tersliyorsunuz; gürültüyü atmak için yüksek frekansları
filtreliyorsunuz; ama o yüksek frekanslarda gerçek detay da kayboluyor. Sonuç:
"net" ama "donuk" bir tahmin.

Yani düzenlileştirme **gürültü problemini bir uzaysal bant-genişliği problemine
dönüştürür.** Bunu §3'te (Test 2) deneyle göstereceğiz.

### 2.4 İkilemden kaçış: soruyu değiştirmek

Yukarıdaki imkânsızlık sonucu bir varsayıma dayanıyor: estimator iki
ölçümünü **aynı zamanda farklı gradient'lerden** alıyor. Bu varsayım
mutlak hizalama tahmini için kaçınılmaz.

Ama pratikte sistematik kontrolü için **mutlak** hizalamayı bilmek
zorunda mıyız? pEDM'in sistematik bütçesinde önemli olan, kalibrasyon
anından bu yana hizalamanın **ne kadar değiştiği** — bu da çok daha kolay
bir soru.

Eğer iki ölçümü iki **gradient**te değil, iki **zaman**da alırsak, ofset zaten
sabittir ve çıkarınca gider:

$$ \mathbf{y}(t) - \mathbf{y}_0 = R\bigl(\Delta q(t) - \Delta q_0\bigr)
   + \underbrace{(\mathbf{b}(t)-\mathbf{b}_0)}_{\approx 0} + \boldsymbol{\eta}. $$

Sonra **iyi-koşullu** $R^{-1}$ tersini kullanabilirim:

$$ \widehat{\delta q}(t) = R^{-1}\bigl(\mathbf{y}(t)-\mathbf{y}_0\bigr). $$

Ofset gradient farkıyla değil, **zaman farkıyla** iptal oluyor. Bu kavramsal
olarak yeni bir matematiksel formülasyon değil; literatürdeki orbit-bumps
ve slow-drift correction tekniklerinin bir uygulaması. Yenilik iddiası bunda
değil. Bu çalışmanın katkısı: pEDM koşullarında bu modun (ii) ve (iii)
seçeneklerine kıyasla **niceliksel üstünlüğünü göstermek** ve mutlak
mod ile drift modu arasındaki seçimi sistematik bir karara dönüştürmek.

Bedeli: mutlak hizalama bilgisinin kaybı. Onu ayrı bir yavaş kalibrasyon
katmanı (LOCO, BBA, survey) verecek; drift modu bu katmanın üstünde sürekli
çalışır. Bu iki katmanlı yapı §4'te ayrıntılı.

---

## 3. Sayısal Deneyler (makalenin §4'ü)

Bu bölümde beş kontrollü simülasyon ile yukarıdaki teorik iddiaları
**sayılarla** doğrulayacağız. Her test bir sorunun cevabıdır.

### 3.1 Test 1 — Düzenlileştirme gerçekten kurtarır mı?

**Sorulan soru.** Ham $\Delta R^{-1}$ tahmini $\sim 1900\,\mu$m hata veriyor.
Bu, "$\Delta R$ yöntemi kötü" demek midir, yoksa Tikhonov/TSVD ile bu
düzeltilebilir mi?

**Yöntem.** Aynı simülasyon verisi üzerinde altı estimator karşılaştırılır:
| # | Estimator | Ne yapıyor |
|---|---|---|
| 1 | $R_1^{-1}\mathbf{y}_1$ | Direct, ofset varsayımı yok |
| 2 | $R_2^{-1}\mathbf{y}_2$ | Aynı, ikinci gradient |
| 3 | $(v_1+v_2)/2$ | Direct ortalama |
| 4 | Ham $\Delta R^{-1}$ | Klasik k-mod |
| 5 | Tikhonov $\Delta R$, en iyi $\lambda$ | L-curve köşesi |
| 6 | TSVD $\Delta R$, en iyi $k$ | "Oracle" — gerçek cevabı bilen seçim |

**Sonuç.**

| Estimator | y-RMS | y-korelasyon |
|---|---|---|
| Direct $(v_1+v_2)/2$ | **3.5 μm** | **0.998** |
| Ham $\Delta R^{-1}$ | 1865 μm | 0.085 |
| Tikhonov (L-curve) | 53 μm | 0.348 |
| TSVD (oracle, $k=3$) | 52 μm | 0.383 |

İki ders var:
1. Düzenlileştirme ham yöntemi **35×** iyileştiriyor (1865 → 52 μm), demek
   ki ham sonuç doğru worst-case'dir.
2. Buna rağmen direct estimator **15×** daha iyi (3.5 vs 52 μm).
3. Daha şaşırtıcı olan: korelasyon 0.998'den 0.35'e çöküyor. Yani düzenlileştirme
   RMS'i azaltıyor ama **biçimi de bozuyor**. Niye? Çünkü 48 modun sadece
   3-5'ini kurtarıyor (TSVD oracle $k=3$). Geri kalan modların hepsi feda
   ediliyor.

Bu son cümleyi Test 2 görsel olarak ispatlayacak.

### 3.2 Test 2 — Estimator'ın "uzaysal transfer fonksiyonu"

**Sorulan soru.** Tikhonov ve TSVD, hangi uzaysal frekanslarda iyi çalışıyor,
hangilerinde çuvallıyor?

**Anahtar fikir.** Her estimator'ı bir "filtre" olarak düşünelim. Girişine
saf bir sinüsoidal misalignment paterni veririm:

$$ \Delta q_j^{(k)} = A\cos(2\pi k j/N + \varphi),\quad k=0,1,\ldots,N/2. $$

Çıkışında ne kadarını geri kurtarıyor? Oran 1.0 ise mükemmel, 0 ise tamamen
bastırılmış. Bu fonksiyonun grafiği bana o estimator'ı tam tanımlar.

**Sonuç (gürültüsüz senaryo).**

| $k$ | Direct | Ham $\Delta R^{-1}$ | Tikhonov | TSVD |
|---|---|---|---|---|
| 1 | 1.000 | 1.000 | 0.00 | 0.00 |
| 4 | 1.000 | 1.000 | 0.00 | 0.00 |
| 12 | 1.000 | 1.000 | 0.00 | 0.00 |
| 20 | 1.000 | 1.000 | 0.3 | 0.4 |
| 24 | 1.000 | 1.000 | 0.5 | 0.5 |

**Burada gizlenmiş büyük bir gerçek var.** Ham $\Delta R^{-1}$ aslında
**ön-yargısız**: her modu mükemmel kurtarıyor. Sorunu sadece **varyans**:
gerçek deneyde gürültü çarpı $1/\varepsilon$. Yani 1865 μm'lik hata gürültü
amplifikasyonu, sinyal bozulması değil.

Buna karşılık Tikhonov/TSVD **ön-yargılı**: 48 modun 40+ tanesini sıfıra
çekiyor. Onların verdiği 50 μm'lik hata, **gürültüden değil, gerçek sinyali
silmekten** geliyor.

**Bu bölümün kalbi:** Düzenlileştirme, gürültü problemini ön-yargı problemine
çevirir. Trade-off yok olmaz, biçim değiştirir.

İşletmesel sonuç: Ham $\Delta R^{-1}$ pratikte ölü gibi görünüyor ama eğer
**zaman içinde ortalama alabilirsek**, gürültü $\sqrt{T}$ ile azalır (varyans
$1/T$). Yeterince uzun süre beklersek, ham $\Delta R$ tahmini geri canlanır.
Bunu Test 5'te test edeceğiz.

### 3.3 Test 3 — Yatay-düzlem ters-suç kontrolü

**Arkaplan.** $R_x$ matrisini kurarken bir parametremiz var: yatay ark
odaklaması $K_{x,\text{arc}}$. Bunu, simülasyondan ölçülen $Q_x = 2.6824$
tune'unu hedef alarak hesaplıyoruz. Yani: model parametresini, ileri
simülasyondan alıyoruz. Bu klasik **ters-suç** (inverse crime) durumudur.
Aynı simülasyona göre ayarladığım modelle, o simülasyonu çözüyorsam, sonuç
yapay olarak iyi olabilir.

**Soru.** $K_{x,\text{arc}}$'da %10 hata olsaydı, rekonstrüksiyon ne kadar
kötüleşirdi? Eğer cevap "epeyce", o zaman gerçekten ters-suç vardır. Cevap
"çok az" ise, model parametresinde küçük hatalara karşı yöntem dayanıklıdır.

**Yöntem.** İleri simülasyon **gerçek** $K_{x,\text{arc}}$ ile koşulur. Ters
rekonstrüksiyon ise $K_{x,\text{arc}}\cdot(1+\delta)$ ile, $\delta \in [-10\%,+10\%]$
aralığında 21 farklı değerle. Kontrol olarak dikey düzlem ($K_{y,\text{arc}}=0$
Maxwell garantisi) hiç değişmeden devam etmeli.

**Sonuç.**
- Tüm $\pm 10\%$ aralığında yatay RMS 3.48–4.01 μm arasında oynuyor. Yani
  yarım mikronluk değişim.
- Dikey RMS sabit 3.489 μm. Maxwell tahmini doğrulandı.

Yorum: LOCO-tipi kalibrasyon gerçek halkalarda $<1\%$ doğruluk verir;
biz $10\%$ hatada bile yarım mikron kaybediyoruz. Yani ters-suç **operasyonel
anlamda yok**.

### 3.4 Test 4 — Drift takibinin somut gösterimi

**Senaryo.**
- $t=0$ kalibrasyon: 100 μm RMS misalignment + **50 μm RMS BPM ofseti** kaydedildi.
- $t=1\ldots 10$ epochs: misalignment yavaşça 10 μm RMS toplam kayıyor,
  ofset sabit.
- Drift tahmini: $\widehat{\delta q}(t) = R^{-1}(\mathbf{y}(t)-\mathbf{y}_0)$.
- Kontrol: naif mutlak rekonstrüksiyon $\widehat{\Delta q}(t) = R^{-1}\mathbf{y}(t)$.

**Sonuç.**

| Yöntem | Ortalama RMS |
|---|---|
| Drift mode $R^{-1}(\mathbf{y}-\mathbf{y}_0)$ | **6.5 μm** |
| Mutlak $R^{-1}\mathbf{y}$ | 170–200 μm |

Drift modu mutlak moddan **~28 kat** iyi. Niye? Çünkü $R^{-1}\mathbf{b}_0$
terimi her iki ölçümde de aynı; çıkarıldığında siliniyor. 50 μm'lik ofset
göründüğü gibi büyük bir sorun değil — **kalibrasyon ile ölçüm arasında
sabit kaldığı sürece** önemsiz.

Geriye kalan 6.5 μm hata da ofsetten değil, BPM gürültüsünün
$\sqrt{2}\sigma_n\|R^{-1}\|$ ile propagasyonundan geliyor.

### 3.5 Test 5 — BPM ofseti gerçekten "sabit" kalmazsa?

**Endişe.** Test 4 ofsetin sabit olduğunu varsaydı. Ama BPM elektroniği
zamanla kayar (termal, voltaj, eskime). O zaman direct estimator A bozulur.

**Alternatif: Estimator B.** Her epoch'ta yeni bir iki-gradient ölçüm yap,
$\Delta R^{-1}$ ile o anlık ofseti iptal et, ama tek-epoch tahmin gürültülü
olduğu için ($\kappa(\Delta R)\approx 27\,000$) bir kayan pencere üzerinden
30 epoch'luk ortalama al. Bu estimator BPM driftine **yapısal olarak duyarsız**.

**İki estimator karşılaştırması.**

| BPM drift hızı [μm/epoch] | A: direct | B: $\Delta R$ avg-30 |
|---|---|---|
| 0 | **5.6** μm | 170 μm |
| 0.5 | 83 | 190 |
| 2.0 | 335 | **184** |
| 5.0 | 917 | **210** |

Geçiş ~2 μm/epoch. Modern BPM elektroniklerinin termal katsayısı ~0.1 μm/°C
seviyesindedir — kayma hızının 2 μm/epoch'a ulaşması için her ölçüm aralığında
1 °C civarında değişim gerekir. Termal-stabil bir hızlandırıcı salonunda bu
gerçekçi değil.

**Ders.** Hızlı, hassas izleme için A; uzun vadeli kontrolde B. İkisi
**tamamlayıcı**, rakip değil.

### 3.6 Test 6 — Üç yöntemin adil yan yana karşılaştırması

**Sorulan soru.** Test 4'te drift modu çalıştı, Test 1'de $\Delta R^{-1}$
çuvalladı. Ama bunlar farklı senaryolardı. Tam aynı koşulda, aynı veri,
aynı hata kaynakları altında yan yana koysak ne çıkar?

**Üç estimator.**
- A: Analitik $\Delta R^{-1}$, her epoch için mutlak rekonstrüksiyon, sonra
  epoch farkı → kavramsal olarak "v2.7 yaklaşımının analitik karşılığı".
- B: Analitik R, tek-gradient drift modu (önerilen).
- C: Sayısal $\Delta R^{-1}$ (48 finite-difference simülasyondan inşa),
  per-epoch mutlak, sonra epoch farkı → "v2.7 yaklaşımının orijinali".

**Senaryo.** Tüm estimatörlere aynı veri sunulur:
- Statik hizalama hataları: params.json'dan (~58 μm RMS)
- Drift: ~10 μm RMS rastgele
- BPM ofset: 50 μm RMS sabit
- BPM gürültü: 1 μm RMS bağımsız
- Quad ve dipol tilt'i: 0.2 mrad max (modelde **yok** — model uyumsuzluğu)

**Sonuç.**

| Estimator | y-RMS | y-corr | x-RMS | x-corr |
|---|---|---|---|---|
| A: Analitik $\Delta R$ | **3282 μm** | -0.02 | **3756 μm** | 0.09 |
| B: Drift modu | **6.25 μm** | 0.85 | **7.18 μm** | 0.85 |
| C: Sayısal $\Delta R$ | **980 μm** | -0.02 | **1357 μm** | -0.11 |

**Önemli sayılar.**
- $\kappa(\Delta R_{\text{analitik}}) \approx 27\,000$
- $\kappa(\Delta R_{\text{sayısal}}) \approx 27\,000$
- $\|R_{\text{an}}-R_{\text{num}}\|/\|R_{\text{num}}\|$: %2.2 (y), %6.9 (x)
- $\kappa(R) \approx 160$

**Yorum.**
1. **Sayısal R, $\Delta R$ tabanlı yöntemi kurtarmıyor.** Modeli tamamen
   simülatörden inşa etsek bile $\kappa(\Delta R)$ değişmiyor; gürültü
   büyütmesi yapısal.
2. **v2.7'nin 6 μm sonucu farklı bir soruydu.** Orada **tek-epoch mutlak**
   rekonstrüksiyon ölçülmüştü ve gürültü ya yoktu ya da ihmal edilebilirdi.
   Drift tahmini için ($\delta q$, kalibrasyona göre değişim) iki epoch
   farkı gerekir, $\kappa(\Delta R)$ iki kez devreye girer.
3. **Drift modu mütevazı model uyumsuzluğunu (%2-7) tolere ediyor.** 6-7 μm
   sonucu BPM gürültüsünden geliyor, $\sqrt{2}\sigma_n\|R^{-1}\|$ ile
   tutarlı.

Bu test, §2'de ispatlanan ofset-gürültü ikilemini somutlaştırıyor: aynı
problem, aynı hata kaynakları, üç farklı estimator, üç farklı sonuç.
Klasik $\Delta R^{-1}$ yaklaşımının numerik ya da analitik versiyonu,
realistik gürültü altında 10 μm hedefine yaklaşamıyor; drift modu
yaklaşıyor.

### 3.7 Altı testin özeti

| Test | Soru | Ana sayı |
|---|---|---|
| 1 | Düzenlileştirme ham $\Delta R$'yi kurtarır mı? | 1865→52 μm, ama direct 3.5 μm |
| 2 | Düzenlileştirme nasıl çuvallıyor? | 48 modun 43-45'i siliniyor |
| 3 | Yatay modelde ters-suç var mı? | $\pm 10\%$ → <0.5 μm değişim |
| 4 | Drift modu büyük ofseti tolere eder mi? | Mutlak 180 μm → Drift 6 μm |
| 5 | BPM ofseti kayarsa? | A: <2 μm/epoch'a kadar üstün |
| 6 | Aday yöntemler aynı senaryoda nasıl? | $\Delta R$: 1000-3700 μm, Drift: 6-7 μm |

---

## 4. Kalibre-referans Drift İzleme Mimarisi (makalenin §5'i)

### 4.1 Resmi büyütelim

Sorduğumuz şu: pEDM deneyinde **operasyonel olarak** nasıl bir hizalama
izleme sistemi kurmak gerekir?

Klasik düşünce: "elimde bir denklem, çözeyim ve bitsin." Ama gerçek hayatta
problem zaman içinde değişiyor. Mıknatıslar bir kerede yerine yerleştirilmez
ve sonsuza dek o konumda kalmaz. Onlar:
- termal genleşme ile mikronlarca kayar,
- elektrik deflektörlerinin yüksek voltaj döngüleriyle titrer,
- yere oturmuş destekleri zamanla "akar",
- elektrostatik yük asimetrileri çekme kuvveti üretir.

Bütün bu mekanizmalar dakikalar ila saatler zaman ölçeğinde çalışır. **Bir
fiziksel run sırasında** quadrupole'lar mikronlarca kayar. EDM ölçümünün
sistematik özeti bu kayma sırasında değişir. Bu değişimi bilmiyorsanız,
neyle savaştığınızı da bilmiyorsunuz.

İşte burada drift izleyici devreye girer.

### 4.2 İki katmanlı mimari

Tek bir yöntemin tüm yükü taşımasına gerek yok. İki katman tasarlıyoruz:

**Yavaş Mutlak Katman: LOCO + BBA**
- Sıklık: çalışma dönemi başına bir kez (saatler–günler)
- Çıktı: mutlak misalignment $\Delta q_0$, BPM ofseti $\mathbf{b}_0$, optik
  model ($\beta$, $\phi$, $Q$ → $R$)
- Bunlar zaten varolan, hızlandırıcı operasyonunda kullanılan araçlar.
- Hassasiyet: 10-50 μm seviyesinde mutlak.

**Hızlı Drift Katmanı: Bu çalışma**
- Sıklık: sürekli (saniyeler-dakikalar)
- Çıktı: kalibrasyona göre drift vektörü $\delta q(t)$
- Araç: $R^{-1}(\mathbf{y}(t)-\mathbf{y}_0)$ direct inversion
- Hassasiyet: epoch başına 1-5 μm

İkisi **birbirinin yerine geçmiyor**; **birbirini tamamlıyor**. LOCO bana
$R$ matrisini ve referans noktasını veriyor. Sonra ben o referansı kullanarak
hızlı drift'i takip ediyorum.

### 4.3 Üç operasyon modu

Bu mimariyi kurduktan sonra üç olası işletme modu var:

**Mod 1 — Tek-gradient drift modu (önerilen).** Fizik run'ı boyunca gradient'i
hiç değiştirme. Sadece $\mathbf{y}(t)$ oku, $\mathbf{y}_0$'dan farkını al,
$R^{-1}$ uygula. Fizik veri toplamayı hiç bozmadığı için tercih edilen mod.

**Mod 2 — Aralıklı iki-gradient modu.** Saatte bir kez gradient'i %2 oynatıp
$v_1$ ile $v_2$ tahminlerini karşılaştır. Bunlar uyuşmuyorsa, ya BPM gain'i
kaymış ya da model bozulmuş — yeniden kalibrasyon zamanı.

**Mod 3 — Adanmış kalibrasyon pencereleri.** Fizik dışı zaman dilimlerinde
tam iki-gradient k-mod sürecini koşup $R$ ve $\mathbf{b}_0$'ı yenile.

### 4.4 Niye bu çerçeve "ofset–gürültü düalitesinin" dışına çıkar?

§2'de ispatladık ki ofseti iptal eden ön-yargısız her lineer estimator
gürültüyü $1/\varepsilon$ kat büyütür. Bu teoremin bir varsayımı var: estimator
**aynı epoch'ta** iki ölçümden besleniyor.

Drift modu bu varsayımı kırar. Estimator iki **gradient**ten değil, iki
**zaman**dan besleniyor. Ofset iptali gradient farkıyla ($\varepsilon$ ile
azalan), zaman farkıyla (BPM kararlılığı varsayımıyla) gerçekleşiyor. Yani
**farklı bir matematiksel problem çözüyoruz** — fakat fizik açısından doğru
soru bu zaten.

Bunun bedeli: mutlak hizalama bilgisini kaybediyoruz. Sadece "kalibrasyondan
beri ne değişti" sorusunu cevaplıyoruz. Ama pEDM için doğru soru budur:
sistematik kontrolünün ihtiyacı **değişimi** bilmektir, mutlak değeri değil.

### 4.5 Bir resimle özetlersek

```
            FİZİK RUN'I (sürekli)
   ┌─────────────────────────────────────┐
   │  BPM y(t) ───┐                      │
   │              ▼                      │
   │       y(t) − y₀ ──► R⁻¹ ──► δq(t)   │
   │              ▲                      │
   │              │                      │
   └──────────────┼──────────────────────┘
                  │
        y₀, R, b₀ (saatlik–günlük güncellenir)
                  │
   ┌──────────────┴──────────────────────┐
   │   YAVAŞ KATMAN: LOCO / BBA / Survey │
   │   (fizik-dışı zaman pencereleri)    │
   └─────────────────────────────────────┘
```

Üst çerçeve sürekli koşar, mikrosaniye-mertebesinde değil ama saniye-dakika
mertebesinde tahmin verir. Alt çerçeve saatlik-günlük periyotta referansı
yeniler. Bu mimari, klasik k-mod yönteminin gradient modülasyonunu **fizik
run'ından çıkartıp kalibrasyon penceresine taşır**. Fizik veri toplaması
hiçbir şekilde rahatsız edilmez.

---

## 5. Sıkça sorulan sorular

**S1: Neden direct $R^{-1}$ ofsete duyarlı, ama drift modu değil?**

Direct uygulandığında: $\widehat{\Delta q} = R^{-1}(R\Delta q + \mathbf{b}) =
\Delta q + R^{-1}\mathbf{b}$. İkinci terim sizin ofsetinizdir, atılamaz.
Drift modunda: $R^{-1}((R\Delta q(t)+\mathbf{b}) - (R\Delta q_0+\mathbf{b}))
= \Delta q(t)-\Delta q_0$. Ofset zaten silindi.

**S2: $\Delta R$ yöntemini tamamen mi terk ediyoruz?**

Hayır. Üç işi var: (1) ilk kalibrasyonda $R$ matrisini test eder; (2)
periyodik kontroller için $v_1$ ile $v_2$ tutarlılığını doğrular; (3) BPM
gain'inin uzun vadeli kayması varsa Estimator B olarak yedektir.

**S3: $K_{x,\text{arc}}$ kalibrasyonunu nasıl yaparım?**

Modeli simülasyondan veya LOCO fit'inden alın. Test 3 gösteriyor ki
%10'a kadar hatalar tolerans dahilinde — bu, gerçek halkada LOCO ile rahatça
sağlanan bir doğruluktur.

**S4: BPM gürültüsü ne kadar olabilir?**

800-tur ortalama sonrası tek okumadaki gürültü $\sim 1\,\mu$m alınmıştır.
Daha gürültülü BPM'ler için ortalama penceresi büyütülür, hata $\sqrt{N}$
ile azalır.

**S5: Neden 48 mıknatıs 96 bilinmeyenli bir problem?**

Her mıknatıs hem yatay hem dikey eksende ayrı ayrı kayabilir. Yatay-dikey
kuplajı (skew coupling) önemsiz mertebede olduğu için problemi iki bağımsız
$48\times 48$ problemine ayırıyoruz: biri $\Delta q_y$ için, biri $\Delta q_x$
için.

**S6: Sinüs paterni yerine rastgele misalignment'la transfer fonksiyonu
ölçemez miydim?**

Ölçebilirsiniz, ama rastgele patern tüm modları karıştırır; estimator'ın
hangi modda iyi/kötü olduğunu ayıramazsınız. Sinüs paternleri bir "Fourier
mikroskobu" gibi davranır: her modu tek tek incelemenize izin verir.

---

## 6. Hangi betik hangi testi koşar

| Test | Betik | Çıktı |
|---|---|---|
| 1 | `compare_regularization.py` | `test1_regularization.png` |
| 2 | `mode_transfer.py` | `test2_mode_transfer.png` |
| 3 | `kxarc_sensitivity.py` | `test3_kxarc_sensitivity.png` |
| 4 | `drift_monitor_sim.py` | `test4_drift_monitor.png` |
| 5 | `bpm_offset_drift_sim.py` | `test5_bpm_offset_drift.png` |
| 6 | `test6_fair_comparison.py` | `test6_fair_comparison.png` |

Tüm hızlandırıcı parametreleri `params.json`, tüm test parametreleri
`test_params.json` üzerinden kontrol edilir.
