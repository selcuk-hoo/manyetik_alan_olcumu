# pEDM Kuadrupol Hizalama Projesi — Bütüncül ve Pedagojik Sentez

> **Bu belge ne için?** Projenin baştan sona hikâyesini, hiçbir ön bilgi
> varsaymadan ve her kavramı ilk kullandığında tanımlayarak anlatır. Amaç, dağınık
> teknik günlükleri (`akilli_duzeltme.md`, `omarov.md`, `squid_bpm_test.md`,
> `orbit_ileri_olcum.md`, `false_edm_harmonic_sinir.md`, `makale-taslagi-2.md`)
> okumadan da projenin **ne aradığını, ne bulduğunu ve neden orada durduğunu**
> anlaşılır kılmaktır. Buradaki her sayı gerçek C++ izleyiciyle
> (`integrator.cpp` — GL4 semplektik parçacık + Thomas-BMT spin) üretildi; keşif
> kodu proje konvansiyonu gereği `/tmp/akilli_duzeltme/` altında tutulur ve ilgili
> `.md` günlüklerinden referanslanır. Ayrıntı isteyen okuyucu §8'deki belge
> haritasını izlesin.

---

## 0. Bir sayfada: baştan sona ne oldu?

Proton EDM (pEDM) deneyi protonun **elektrik dipol momentini** ölçmeye çalışır. Bu
mümkün olursa evrende madde-antimadde dengesizliğine dair temel bir ipucu verir.
Deneyin kalbindeki teknik zorluk şudur: halkadaki mıknatıslar ideal yerlerinden
birkaç mikron kaymışsa, bu kayma **gerçek EDM'i birebir taklit eden** sahte bir
sinyal üretir. Buna **sahte-EDM** denir ve deneyin en tehlikeli sistematiğidir.

Proje iki bağımsız iş kolu üretti:

- **Hat A — hizalama İZLEME (bitmiş, pozitif katkı).** "Mıknatıslar zamanla ne
  kadar kaydı?" sorusunu, ışının kendi yörüngesini kullanarak yanıtlar. Sonuç
  olumlu ve yayınlanabilir: 50 μm'lik ölçüm-cihazı hatası (BPM ofseti) altında bile
  kayma **6.5 μm** hassasiyetle izlenebiliyor (bir **drift-gözcüsü**). Ayrıntı
  `makale-taslagi-2.md`.

- **Hat B — sahte-EDM'i KONTROL etme (asıl entelektüel sonuç: bir sınır teoremi).**
  "Sahte-EDM'i aktif olarak ölçüp sıfırlayabilir miyiz?" sorusunu araştırdık. Cevap,
  şaşırtıcı biçimde ama sağlam biçimde, **hayır**: sahte-EDM'i süren fiziksel
  büyüklük, ışının hızlı gözlemine görünmez; onu gören tek gözlem (spin) ise o kadar
  yavaştır ki hassas ölçüm yıllar-onyıllar alır. Bu ikilemi **birbirinden bağımsız
  sekiz yöntemle** doğruladık.

**Tek cümlelik özet.** *Hızlı olan gözlem (kapalı yörünge) sahte-EDM'in bilgisini
taşımıyor; bilgiyi taşıyan gözlem (spin) hızlı değil.* Dolayısıyla aktif ölçüp-null
yapmanın makul-süreli hiçbir yolu yok; sahte-EDM ancak **önceden sınırlanarak**
(yapım toleransı + alan kalitesi + karşı-dönen demet iptali) kontrol edilebilir. Bu
sonuç, alanın referans makalesinin (Omarov ve ark., PRD 105, 032001) açık bıraktığı
bir boşluğu kapatır.

Aşağıdaki bölümler bu iki cümleye nasıl vardığımızı adım adım kurar.

---

## 1. Sahne: pEDM deneyi ve 10 mikron problemi

Proton EDM deneyi **frozen-spin** (donmuş-spin) yöntemiyle çalışır. Protonlar özel
bir depolama halkasında dönerken, spinleri momentum yönüne kilitlenecek biçimde
ayarlanır. Eğer protonun gerçek bir elektrik dipol momenti varsa, spin bu kilitten
**çok yavaşça** düşey yönde kayar. Ölçtüğümüz büyüklük tam olarak bu kayma hızıdır:

$$f \equiv \frac{dS_y}{dt}.$$

Gerçek EDM'in (10⁻²⁹ e·cm) ürettiği kayma hızı **1 nrad/s** mertebesindedir — bunu
projede "hedef" olarak anıyoruz. Bu inanılmaz derecede küçük bir sayıdır; spinin
düşey bileşeninin saniyede milyarda bir radyan dönmesi demektir.

**Sorun.** Halkadaki bir kuadrupol mıknatıs ideal konumundan birkaç mikron kaymışsa,
o da tam olarak aynı türden bir $dS_y/dt$ üretir — ve bunu gerçek EDM'den ayırt
etmek fizik olarak imkânsızdır (ikisi de aynı gözlemlenebilirde görünür). 10 μm'lik
tipik bir kayma, hedefin **~1000 katı** sahte-EDM üretir. Yani gerçek sinyal, sahte
sinyalin bin kat altında gömülüdür. Hesaplar, sahte-EDM'i hedefe indirmek için her
mıknatısın konumunun **~10 μm** hassasiyetle kontrol edilmesi gerektiğini söyler.

Halka **24 FODO hücresi × 2 kuadrupol = 48 kuadrupol** içerir. (FODO, odaklayan bir
kuadrupol — QF — ile dağıtan bir kuadrupolün — QD — ardışık dizilişidir; bir ışın
demetini yatayda ve düşeyde sırayla sıkıştırıp gevşeterek halka boyunca odaklı
tutar.) Her mıknatısın yanında bir BPM (Beam Position Monitor, ışın konum
monitörü) vardır.

Bu belgenin geri kalanı iki soruyu birbirinden ayırır ve ayrı ayrı yanıtlar:

1. **Hat A:** Mıknatıslar zamanla ne kadar kaydı? (İzleme — çözülebilir.)
2. **Hat B:** Sahte-EDM'i aktif olarak sıfırlayabilir miyiz? (Kontrol — sınırlı.)

---

## 2. Sahte-EDM neden **dx·dy**'den doğar? (Mekanizma)

Bu, tüm projenin fizik çekirdeğidir; birkaç dakikada sezgisel olarak anlaşılır.

Bir kuadrupol ideal konumundan kayarsa, içinden geçen ışına sapmış bir alan
gösterir. **Hangi yönde kaydığı kritiktir:**

- **Düşey kaçıklık $dy$** → ışın kuadrupolün **radyal** alanını ($B_x$) görür →
  Thomas-BMT denklemi gereği spin **x-ekseni** etrafında döner.
- **Yatay kaçıklık $dx$** → ışın **düşey** alanı ($B_y$) görür → spin **y-ekseni**
  etrafında döner.

İlk sürpriz: tek başına $dx$ ya da tek başına $dy$ neredeyse hiç sahte-EDM üretmez.
Tek $dx$ düşey spine (S_y) dokunmaz; tek $dy$ küçük bir etki verir ama halka
çevresinde büyük ölçüde birbirini götürür. **Asıl büyük sahte-EDM iki kaçıklık
birlikte varken doğar** ve sebebi tamamen geometriktir:

> Bir eksende dönme ile başka bir eksende dönme **sıra değiştirilemez**
> (komütatif değildir). Önce x sonra y ekrafında döndürmek ile tersini yapmak farklı
> sonuç verir; aradaki fark, üçüncü eksende net bir dönme bırakır. Bu, klasik ve
> kuantum mekaniğinde bilinen **geometrik (Berry) fazıdır**.

Sonuç:

$$f \;\propto\; dx\cdot dy \;\;\Longrightarrow\;\; f \propto \sigma^2,$$

çünkü $dx$ ve $dy$ ikisi de tipik RMS $\sigma$ mertebesindeyse çarpımları
$\sigma^2$ ile ölçeklenir. Bunu simülasyonda doğruladık: $\sigma = 10\to5\to2.5$ μm
taramasında üs **p = 2.00 ± 0.01** çıkıyor (Omarov Fig. 9a ile birebir; lineer
kaçak yok). Bu σ² yasası ileride önemli olacak: hizalamayı 10 kat iyileştirmek
sahte-EDM'i **100 kat** düşürür.

**Doğru ölçüm reçetesi (bir tuzak).** $f$'i naifçe ölçmeye kalkarsak, tek
parçacığın kapalı yörünge etrafında yaptığı büyük betatron salınımı aradığımız
minik seküler kaymayı boğar. Doğru yöntem: ya **dört simetrik parçacık**
$(\pm dx, \pm dy)$ ortalaması (betatron karışmasını simetriyle söndürür) ya da
**tek ideal parçacığı 4D kapalı yörüngeye oturtup model-fit** ile yalnız seküler
eğimi çekmek. Düz doğrusal fit KULLANILMAZ — salınımı sinyal sanıp $f$'i şişirir.
(Ayrıntı `false_edm_harmonic_sinir.md §13`; doğrulanmış estimator
`berry_data/false_edm_4d.py`.)

---

## 3. Asıl zorluk: yörüngenin göremediği "simetrik" alt-uzay

Şimdi projenin neden zor olduğunu anlıyoruz. 48 kuadrupolün kaçıklık desenini iki
tamamlayıcı biçimde ayırmak, tüm hikâyenin anahtarıdır.

**Antisimetrik vs simetrik.** Her FODO hücresinde QF (gradyanı +) ve QD (gradyanı
−) vardır. Bir kaçıklık deseni, hücre içindeki QF ile QD'nin **aynı mı, zıt mı**
yönde kaydığına göre ikiye ayrılır:

- **Antisimetrik** (QF ve QD zıt yönde): gradyan işaretleri de zıt olduğu için iki
  kuadrupolün ışına verdiği tekmeler **üst üste biner** → **büyük** kapalı yörünge
  bozulması. Bu desen düzgün, düşük-dalga-sayılı (düşük-$k$) bir tekme üretir.
- **Simetrik** (QF ve QD aynı yönde): tekmeler **birbirini söndürmeye** çalışır →
  **küçük** yörünge. Bu desen hücre-içinde işaret değiştiren, yüksek-dalga-sayılı
  ($k \approx 24$) bir tekme üretir.

**Yörünge neden simetriğe kör?** Bir kuadrupol tekmesinin yörüngeyi ne kadar
bozduğu, halka boyunca **betatron rezonansıyla** belirlenir. Sezgi: her kaçık
kuadrupol küçük bir tekme verir; bu tekmeler halka çevresinde dolaşırken üst üste
biner. Modun uzaysal frekansı betatron tonuna ($Q \approx 2.7$) yakınsa tekmeler
**rezonansla** büyür (salıncağı doğru ritimde itmek gibi); uzaksa söner. Kapalı-form
kazanç yasası:

$$G_k = \frac{C}{|Q^2 - k^2|}, \qquad C \approx 24.8,\; Q^2 \approx 5.$$

Antisimetrik desen düşük-$k$, $Q$'ya yakın → **büyük kazanç, görünür**. Simetrik
desen $k\approx24 \gg Q$ → kazanç ezilir → **görünmez**.

**Ve işte kritik nokta:** sahte-EDM'i (dx·dy) büyük ölçüde **simetrik alt-uzay
sürer**, ama bu alt-uzay kapalı yörüngede neredeyse hiç iz bırakmaz. Somut sayılar:

| kaçıklık parçası | desen RMS | yörünge (COD) izi | sahte-EDM |
|---|---|---|---|
| antisimetrik (25 boyut) | ~70 μm | ~550 μm (**büyük, görünür**) | büyük ama düzeltilebilir |
| **simetrik (23 boyut)** | ~55 μm | ~165 μm (**kazanç 193× daha zayıf**) | **kalanın çoğu, görünmez** |

Tepki matrisi $R$'nin kondisyon sayısı (en büyük/en küçük tekil değer oranı)
**~193**'tür; bu, iki alt-uzay arasındaki görünürlük uçurumunu tek sayıda özetler.

Bunun deneysel sonucu: kapalı yörüngeyi düzeltince (antisimetrik kısmı silince)
sahte-EDM 7.7× düşer, ama geriye **simetrik, orbit-kör bir artık kalır (~62× hedef)**
— ve bu artık yörüngeye görünmez olduğu için orada takılır. Hat B'nin tüm hikâyesi,
bu 62×'lik simetrik artığı aşmaya çalışıp neden aşılamadığını anlatır.

**İnce ama hayati bir düzeltme (sürücü büyüklük neyin ofseti?).** Sahte-EDM, ışının
kuadrupolün **manyetik merkezine** göre ofsetine bağlıdır — mekanik/geometrik
merkezine değil:

$$\text{manyetik merkez} = \text{mekanik merkez} + \frac{B_{\text{dipol-hata}}}{g}.$$

Kuadrupoldeki ufak akım/alan hataları rahatça ~20 μT'lik parazit bir dipol alan
yaratır; bu da ~10–100 μm'lik eşdeğer bir manyetik-merkez kayması demektir. Bu
nokta ileride mekanik-metroloji kaçışını (§5.5) kapatan şeydir: mıknatısı 1 μm'ye
mekanik hizalasan bile manyetik merkez 10 μm kaymış olabilir ve bunu **yalnızca ışın
görür.**

---

## 4. Hat A — hizalama izleme (bitmiş, pozitif katkı)

Sahte-EDM kontrolüne geçmeden önce, projenin **başarıyla kapadığı** kolu anlatalım.
Buradaki soru daha alçakgönüllü ama tam olarak çözülüyor: *mıknatıslar kurulumdan
bu yana zamanla ne kadar kaydı?*

**Neden zor?** BPM okuması şudur:

$$\mathbf{y} = R\,\Delta q + \mathbf{b} + \boldsymbol{\eta}.$$

Burada $R\,\Delta q$ kaçıklığın yörünge izi (istediğimiz), $\boldsymbol{\eta}\sim1$
μm elektronik gürültü (önemsiz), ve $\mathbf{b}$ her BPM'in **~100 μm'lik statik
elektronik ofseti**dir. Ofset, saatler-günler boyunca sabit ama **bilinmez**;
aradığımız sinyal (~10 μm) onun 10 kat altında. Doğrudan
$\Delta q = R^{-1}\mathbf{y}$ çözersek $R^{-1}\mathbf{b}$ kirliliği sinyali gömer.

**Çözüm — drift modu (Hat A'nın en güçlü sonucu).** Soruyu yeniden tanımla: mutlak
kaçıklık yerine **iki zaman arasındaki değişimi** ölç.

$$\delta q(t) = R^{-1}\bigl(\mathbf{y}(t) - \mathbf{y}_0\bigr).$$

BPM ofseti $\mathbf{b}$ iki ölçüm arasında sabit olduğu için **farkta tam iptal
olur**. Geriye yalnız $R^{-1}(\text{gürültü})$ kalır; $R$'nin kondisyon sayısı
küçük olduğundan bu sınırlıdır. Sonuç: **50 μm BPM ofseti altında 6.5 μm RMS
hassasiyet** (mutlak referansa göre 197 μm; yani 29× iyileştirme). β-beating
(optik hataları) %1 iken 6.1 μm, %5 iken 8.6 μm — LOCO kalitesinde bir makinede
operasyonel.

**Diğer Hat A sonuçları (kısaca).** (i) Bir **ofset-gürültü düalite teoremi**:
ofseti iptal eden, ön-yargısız, lineer her estimator sınıfında tek çözüm ΔR⁻¹'dir
ve gürültü büyütmesi $\|R^{-1}\|/\varepsilon$ ile alttan sınırlıdır — anlık k-mod
estimatoru bunun altına inemez. (ii) **Hedefli Fourier**: harmonik içerik önceden
biliniyorsa kondisyon 13000→186'ya iner; keyfi dağılımda greedy/LASSO
rank-yoksulluğundan çöker.

**Kavramsal olarak kritik:** Hat A'nın tüm başarısı **antisimetrik (orbit-görünür)**
alt-uzaydadır. Simetrik alt-uzaya Hat A da kördür — ki bu bizi doğrudan Hat B'nin
duvarına götürür. Hat A pozitif ve bitmiş bir katkıdır (bir "drift-gözcüsü"); Omarov'un
spin-tabanlı kalibrasyonuyla **yarışmaz, tamamlar** — veri alımı sırasında
girişimsiz, sürekli bir izleme kanalı.

---

## 5. Hat B — sahte-EDM'i sıfırlama girişimi ve neden başarısız olduğu

Şimdi asıl soru. Amaç kuadrupol hizalamasını ölçmek **değil** (o bir inversiyon,
bilinen bir no-go); amaç **doğrudan sahte-EDM'i (asıl zararlı skaler) sıfırlamak**.
Denenen her yolu ve düştüğü noktayı kaydediyoruz. Önce hepsini birleştiren tek
kavramı kuralım.

### 5.1 Anahtar kavram: TERS ölçüm mü, İLERİ ölçüm mü?

Yörüngeyi iki şekilde kullanabiliriz ve bunları karıştırmak tüm kafa karışıklığının
kaynağıdır:

- **TERS (inversiyon):** yörüngeden 48 kaçıklığı geri-çöz, sonra sahte-EDM'i hesapla.
  Bu, simetrik yönde $1/\sigma_{\min}$ ile **patlar** (~10⁴ büyütme). Kötü-koşulluluk
  **fiziğin (R'nin)** özelliğidir, kullandığın algoritmanın değil.
- **İLERİ (forward):** yörüngeden doğrudan **tek skaler** $f$'i öngör. Duyarlılık
  $\partial f/\partial \text{COD} \approx 0.15$ — mütevazı, $1/\sigma_{\min}$ değil.
  Yani **iyi-koşullu.**

> **Analoji.** Bir pastanın "çok mu tatlı" olduğunu (1 sayı) tahmin etmek kolaydır;
> ama pastadan "tam tarifi" (48 malzeme miktarı) geri-çıkarmak imkânsızdır — o bilgi
> pastada yok. Sahte-EDM tek bir sayıdır (tatlılık); 48 kaçıklık ise tariftir.

Bu ayrım, projenin en önemli kavramsal kazanımıdır: bilinen no-go **inversiyon**
hakkındadır; ileri-öngörü o sınıfa girmez ve bu yüzden umut vericiydi.

### 5.2 TERS sınıfı — hepsi aynı duvara çarpar

Simetrik bilgi, gürültü/sistematik tabanının altında olduğu için, kaçıklığı
geri-çözmeye çalışan **her** yöntem aynı yerde durur. Bunu bağımsız yollarla
sınadık:

| Yöntem | Neden düşer | Belge |
|--------|-------------|-------|
| R⁻¹ / TSVD / LASSO / CLEAN / Bozoki | simetrik yönde 1/σ_min büyütmesi; **altı yöntem aynı tabana çarpar** | `false_edm_harmonic_sinir.md §14.5` |
| k-mod (ΔR⁻¹) + SQUID-BPM | ΔR→kaçıklık yine ters; dağıtık-frekansta *optik-nefes* etkisiyle ölü | `squid_bpm_test.md §7,§8` |
| k-mod + lock-in | beyaz gürültüyü √N ile yener ama simetrik için <4 nm ister + β-beat felaketi | `squid_bpm_test.md §9.5` |
| **NN (kaçıklık↔COD)** | kaçıklık→COD haritası **LİNEER**, dolayısıyla NN = R; tersi yine R⁻¹ (NN 5.6 μm ≈ TSVD 6.3 μm). Fark algoritmada değil, problemin yönünde | `akilli_duzeltme.md §6.8` |
| **k-mod + LSTM** | beyaz gürültüde lock-in zaten Cramér-Rao-optimal; duvar gürültü değil, sistematik+koşullanma; etiket döngüsel | `akilli_duzeltme.md §6.9` |
| **Omarov CR-ayrım** (CW−CCW yörünge farkı) | ayrım da bir kapalı-yörünge-farkıdır → simetriğe tek-yön COD KADAR kör (doğrudan ölçüldü: bastırma CR 4.5× ≈ COD 3.8×) | `omarov.md §9.3` |

Son satır özellikle değerli: Omarov'un önerdiği "karşı-dönen demetler arasındaki
mesafeyi ölç" fikri de bir yörünge-farkı olduğundan **aynı simetrik körlüğü
paylaşır**. Bunu doğrudan ölçerek Omarov'un açık bıraktığı §9 boşluğunu kapattık.

### 5.3 İLERİ harita (Kol B) — iyi-koşullu ama pratikte yetersiz

Umut buradaydı: yörüngeden doğrudan skaler $f$'i öngören öğrenilmiş bir harita (bir
sinir ağı). Dört testte **pozitif**, ama "ya-hep-ya-hiç" testinde **negatif** çıktı:

| Test | Bulgu | Sonuç |
|------|-------|-------|
| İyi-koşullu mu? | $\partial f/\partial\text{COD}\approx0.15$; gereken ~7 nm COD **ortalamayla** (~21 s) ulaşılabilir; yönlü-alan özniteliği BPM ofsetine değişmez | ✓ |
| Öğrenilebilir mi? | simetrik-kanal çapraz-doğrulama R²: 80 örnekte ~0 → **240 örnekte +0.77** (harita karmaşık ama var) | ✓ |
| β-beat şeffaf mı? | %1 β-beat'li makineye transfer R² 0.62 = held-out nominal 0.61 (ek bozulma yok) | ✓ |
| No-go dışı mı? | evet; birleşik no-go kaçıklık geri-çatımı içindir, ileri-harita o sınıfta değil | ✓ |
| **Çıkarma işe yarar mı?** | sahte-EDM ~1000× sinyal → çıkarma ~%0.1 mutlak doğruluk ister; harita %22 hata → ~220× artık | ✗ |
| **Null'lama işe yarar mı? (make-or-break)** | kapalı-döngü null'lama, basit orbit-düzeltmeyi **geçemiyor**; harita hata tabanı ~300× hedef (240 örnek yetersiz) | ✗ |
| Güvenli-optimizasyon | ensemble + kötümser tahmin model-istismarını dizginler (geomean 236×→67×) ama orbit-null (2.1×) yine önde | ⚠ kısmi |

**Neden "make-or-break negatif"?** Öğrenilmiş bir haritaya karşı optimizasyon
yaparsan, optimize edici haritanın **kör noktalarını** bulur (modelin yanıldığı
yönde "sahte kazanç" gösterir). Harita mutlak olarak ~%0.1 doğru olmadıkça bu
kaçınılmazdır ve empirik yolla o doğruluğa ulaşmak mertebelerce daha fazla veri
ister — pratik değil. **Kol B kavramsal olarak sağlam ama pratikte yetersiz;** eksik
olan tek şey **mutlak harita doğruluğu.**

> **Sık sorulan: "NN ile haritalamada sorun neydi, neden vazgeçtik?"** Kısa cevap:
> harita, beklenen ~10 μm kaçıklık rejiminde çok karmaşık ve az-örnekle
> pinlenemiyor; ona karşı optimize etmek model-istismarına düşüyor. β-beat de
> haritanın karakterini değiştirdiği için (bir simülasyonda %1 β-beat → sahte-EDM'de
> ~%40 hata, `akilli_duzeltme.md §6.5`) haritanın ömrü kısa ve yeniden-kalibrasyon
> maliyetli. Kavram ölmedi, ama analitik bir tutamak olmadan pratik değil.

### 5.4 Spin (Kol A / Plan 5) — doğru ama istatistik-yasak

Bilgi, tanımı gereği spindedir; spin simetrik alt-uzayı **görür**. İki gerçekleme
denedik: **kör spin-trim** (simülasyonda ~6000× temizler) ve **spin-gradient
descent** (kuadrupolü modüle et → sahte-EDM'in tepkisini spinle ölç → gradyanla
null'la). Gürültüsüz demo **çalıştı**: orbit-kör makinede orbit-null işe yaramaz
(1.0×), spin-descent 2 adımda hedefin altına iner (726×).

**Ama istatistik-yasak (kesin sonuç).** Polarimetre $dS_y/dt$'yi bir dolumda
(~1000 s) σ ≈ 900× hedef, **bir yılda** σ ≈ 7× hedef hassasiyetle ölçebilir
(`cosy_polarimeter.md §4`). σ ∝ 1/√T olduğundan 1 nrad/s'e inmek **~50 yıl**
sürer. Descent ~40–100 ölçüm gerektirir ve son ölçümler nrad/s mertebesinde
olduğundan **her biri yıllar → toplam onyıllar-yüzyıllar**. İteratif spin-null
fiilen **imkânsızdır**. (Simülasyondaki ~6000× rakamı "ücretsiz ölçüm"
varsayar; gerçek zaman bütçesiyle geçersiz.)

> **Not — Omarov'un "yükselt-söndür" numarası bunu atlatır mı?** Omarov spini
> minik artığı doğrudan ölçmek için değil, bilinen büyük bir düğmeyle (B_x) bir
> alan-harmoniğini **yükseltip** hızlı ölçmek, sonra söndürmek için kullanır — ve
> bu, **elektrik alanı / dikey hızı** hizalar, geometrik-faz kaçıklığını değil.
> Simetrik geometrik-faz artığı hâlâ görünmezdir (§5.2 son satır). Omarov'un
> **SBA'sı (spin-based alignment) kuadrupol hizalamasını düzeltmez**, E-alanını
> düzeltir — bu önemli bir ayrımdır.

### 5.5 Analitik fonksiyonel ve mekanik metroloji — ikisi de kapalı

İki "son çare" kaldı; ikisini de fizik kapatıyor.

**Analitik fonksiyonel (Plan 4).** COD→f'i Thomas-BMT'den kapalı-form türetmek. İlke
olarak ileri-harita inversiyona girmez, ama: (i) empirik arama 40 config ile
yakınsamadı (en tutarlı aday, Berry'nin doğal büyüklüğü olan **yönlü-alan**
$\Sigma(x_1y_2 - x_2y_1)$, ancak ~−0.5 korelasyon veriyor; `orbit_ileri_olcum.md`).
(ii) Daha da önemlisi: **gerçek makine teoriden çok farklı** olacak (bilinmeyen
çok-kutuplar, fringe alanlar, alan hataları). Sabit bir analitik form da öğrenilmiş
harita gibi model-uyum sorununa düşer. Kullanıcı kararı: teorik türetmeye girmek
verimsiz.

**Mekanik metroloji.** "Simetrik mod ışına görünmez ama survey'e görünür; f∝σ²
olduğundan ~1.3 μm mekanik hizalama artığı hedefe indirir" diye düşünülebilir.
**Ama sürücü büyüklük manyetik-merkez ofsetidir** (§3 sonu): ufak akım/alan
hataları ~20 μT → ~10–100 μm eşdeğer ofset yaratır ve bunu **mekanik survey göremez**
(geometriyi ölçer, alanı değil). Manyetik merkezi gören tek şey **ışındır** — ve
ışın ikilemi (§6) baki.

---

## 6. Sonuç: iki-taraflı ikilem (sınır teoremi)

Yukarıdaki tüm düşüşler tek bir yapıya oturur. Bu, projenin asıl entelektüel
ürünüdür:

> **Sahte-EDM'i süren büyüklük (orbit-kör simetrik alt-uzaydaki manyetik-merkez
> ofseti) yalnızca ışın-tanımlıdır. Ama ışın onu iki uçlu bir ikilemle verir:**
>
> - **Kapalı yörünge (hızlı — saniyeler):** ama simetrik iz, gürültü/sistematik
>   tabanının altında → **görünmez.**
> - **Spin (doğru — bilgi burada):** ama istatistik-sınırlı → hassas ölçüm
>   **yıllar.**
>
> **Hızlı gözlem bilgiyi taşımaz; bilgiyi taşıyan gözlem hızlı değildir.** Bu
> yüzden aktif ölçüp-null'lamanın hiçbir yolu makul sürede çalışmaz.

Denenen sekiz bağımsız kanalın tek tabloda özeti:

| Tutamak | Sınıf | Neden düşer |
|---------|-------|-------------|
| R⁻¹ / k-mod / SQUID / NN / LSTM / CR-ayrım | orbit-TERS | simetrik iz tabanın altında |
| Kol B ileri-harita | orbit-İLERİ | harita kaba (~300×) + model-uyum |
| analitik fonksiyonel | model | gerçek makine ≠ teori |
| spin-descent / spin-trim | spin | istatistik → her ölçüm yıllar |
| mekanik survey | metroloji | manyetik ≠ mekanik merkez (alan/akım ~10 μm, yalnız ışın-görülür) |

**Tek kontrol yolu — a-priori sınırlama (null değil).** Gerçek deney (Omarov)
iteratif ölçüp-null yapmaz. Bunun yerine sahte-EDM'i **önceden bağlar**:
**CW/CCW eşzamanlı demetler** (sahte-EDM'in çoğu farkta iptal olur; tek başına
3.4×) + **a-priori orbit kontrolü** (CR-ayrımını küçültme; hızlı ama simetriğe kör)
+ **yapım toleransı** (mekanik + alan kalitesi). Bunlar artığı ~62× hedefte
**bağlar**, sonra 1-yıllık kampanyada bir kez ölçülür.

---

## 7. Bu ne anlama geliyor? (Konumlandırma)

- **Bu bir başarısızlık değil, bir teoremdir.** Omarov ve ark. geometrik-faz
  kontrolünün *fiziğini* kanıtlar, ama ölçüm-zincirinin simetrik-artığa körlüğünü
  (§9 boşluğu) açık bırakır. Biz o körlüğü **sekiz bağımsız kanaldan** nicelleştirdik:
  simetrik geometrik-faz sistematiği **aktif yöntemlerle indirgenemez**. Bu, pozitif
  bir "kurtarma" değil; **kesin ve özgün bir sınırdır.**
- **Hat A (drift monitör)** ayrı, pozitif, bitmiş bir katkıdır — ikinci makale adayı.
- **Deney bu sonuçla yaşar:** kontrol, aktif null'lama ile değil, a-priori sınırlama
  ile (tolerans + alan kalitesi + CW/CCW) sağlanır. Bizim katkımız *neden daha
  iyisinin mümkün olmadığını* kesin biçimde göstermektir.

**Dürüst açık uç.** İkilemi kıracak, denenmemiş bir ışın-fiziği tutamağı şu an
görünmüyor; muhtemelen olmaması doğru cevap. Tek teorik çıkış, latis/tune'u yeniden
tasarlayıp simetrik alt-uzayı *orbit-görünür* yapmaktır ($G_k$ bastırmasını kaldırıp
$Q$'yu simetrik harmoniğe yaklaştırmak) — ama bu $Q \approx 24$ gerektirir,
gerçekçi değil. Ertelendi.

---

## 8. Belge ve reprodüksiyon haritası

Bu belge üst-düzey sentezdir; her iddianın arkasındaki sayı ve kod şu günlüklerdedir:

| Konu | Ana belge | Kod |
|------|-----------|-----|
| Sahte-EDM mekanizması + estimator (p=2.00) | `false_edm_harmonic_sinir.md §13` | `berry_data/false_edm_4d.py` |
| İki-kademe (orbit+spin) pedagojik anlatım | `trim_yontemi_pedagojik.md` | — |
| Kol B tam yolculuk (§5.1–5.5) | `akilli_duzeltme.md` (+`_pedagojik`) | `/tmp/akilli_duzeltme/*` |
| Ham=antisim / artık=sim ayrımı, ileri-ölçüm | `orbit_ileri_olcum.md` | `/tmp/spin_meas/*` |
| Omarov okuma + CR-ayrım körlüğü (§9.3) | `omarov.md`, `omarov_symmetric_hybrid.md` | `/tmp/akilli_duzeltme/cr_separation.py` |
| k-mod / SQUID / lock-in dalları | `squid_bpm_test.md` | `/tmp/kmod_recover/*` |
| Hat A: düalite + drift + hedefli Fourier | `makale-taslagi-2.md`, `README.md` | `drift_monitor/` |
| Polarimetre zaman bütçesi | `cosy_polarimeter.md` | — |

**Hat B çekirdeğini yeniden üretmek:** `bash build_integrator.sh`, sonra
`/tmp/akilli_duzeltme/` altındaki scriptler (`surrogate` → `gen_patterns` →
`measure_f` → `analyze`; `gen_ensemble` → `fit_forward`; `closed_loop` /
`improved_null`; `cr_separation`; `spin_descent`). `integrator.cpp` **hiç
değiştirilmedi**; β-beat için per-quad `quad_dG`, CR-ayrım için `direction = ±1`
zaten mevcut.

---

## 9. Yan not: bu donanımla ölçülebilecek başka büyüklükler

k-mod + BPM altyapısı, sahte-EDM'den bağımsız olarak tanı/kalibrasyon değeri taşıyan
başka nicelikleri de ölçebilir (ayrıntı git geçmişinde): **beta fonksiyonu**
(tune-shift, en düşük eşik), **quad tilt** (çapraz orbit tepkisi, `R_dx` mevcut),
**BPM gain/roll** (çok-k-mod LOCO), **sekstüpol feed-down**, **linac BBA**. Bunlar
projenin ana sonucu (Hat B no-go) için merkezî değildir ama gerçek bir makinede
altyapının ek değerini gösterir.
