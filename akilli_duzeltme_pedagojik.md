# Akıllı Düzeltme: Sahte EDM'i Misalignment'ı Bilmeden Sıfırlamak — Ders Kitabı Tarzı

Bu belge, proton EDM (pEDM) deneyinde bir **fikri** sıfırdan anlatır: kuadrupol
hizalama hatalarını (misalignment) tek tek ölçüp düzeltmeye çalışmak yerine, asıl
zararlı büyüklüğü — **sahte EDM** sinyalini — *doğrudan* sıfırlamak. Bu fikre
"**akıllı düzeltme**" diyoruz. Belge hiçbir ön bilgi varsaymaz; her kavram ilk
kullanıldığında tanımlanır. Teknik/terse kayıt `akilli_duzeltme.md`'de,
sahte-EDM'in temel mekanizması `trim_yontemi_pedagojik.md`'de, ölçüm-zinciri
boşlukları `squid_bpm_test.md` ve `orbit_ileri_olcum.md`'dedir.

**Sonucu baştan söyleyelim (spoiler):** Fikrin "yörünge tarafı" kolu **çalışmıyor**
ve *neden* çalışmadığı çok öğreticidir. Bu, projedeki üç ayrı "no-go"yu (imkânsızlık
sonucunu) tek bir cümlede birleştiren temiz bir **birleşik sınır teoremine** götürür.

İçindekiler:
1. Hatırlatma: sahte EDM nedir, neden tehlikeli?
2. Fikir: misalignment'ı değil, sahte-EDM'i hedeflemek
3. İki kol: spin-gözlemli (A) vs yörüngeden-öğrenilmiş (B)
4. Kol B neden *makul* görünüyordu? (ileri-harita ≠ inversiyon)
5. Anahtar geometri: yörünge neyi görür, neyi göremez?
6. Karar-verici deney: "ikiz makineler"
7. Sonucu okumak: 7 nanometre duvarı
8. Neden Kol A çalışır da Kol B çalışmaz? (yerel ofset hilesi)
9. Büyük resim: birleşik no-go (üç kol, tek duvar)
10. Empirik pekiştirme: gerçek bir sinir ağı da öğrenemez
11. Sık sorulan sorular

---

## 1. Hatırlatma: sahte EDM nedir, neden tehlikeli?

Proton EDM deneyi, protonun spininin (küçük bir pusula iğnesi gibi düşünün) yatay
düzlemden **düşeye doğru çok yavaş dönmesini** arar. Gerçek bir elektrik dipol
momenti (EDM) bu dönmeyi yaratır. Ölçtüğümüz büyüklük, dikey spin bileşeninin
zamanla birikme hızıdır:
$$f \equiv \frac{dS_y}{dt}.$$

Hedef öyle küçük ki, anlamlı bir EDM ölçümü için $f \sim 1\ \text{nrad/s}$
($10^{-9}$ rad/s) seviyesini ölçmek gerekir. Sorun: **mıknatıs hizalama hataları
da $f \neq 0$ üretir** ve bu, gerçek EDM'le birebir aynı görünür. Buna **sahte
EDM** denir.

Sahte EDM'in mekanizması (ayrıntısı `trim_yontemi_pedagojik.md §1`): bir kuadrupol
yatayda $dx$, düşeyde $dy$ kadar kayarsa, demet hem radyal hem dikey sapmış alan
görür. Spinin x-ekseni etrafındaki dönmesiyle y-ekseni etrafındaki dönmesi **sıra
değiştirilemez** (komütatif değil); ardışık iki dönme net bir üçüncü-eksen dönmesi
bırakır — bu **geometrik (Berry) fazıdır.** Sonuç:
$$f \;\propto\; dx \cdot dy \quad\Rightarrow\quad f \propto \sigma^2,$$
yani sahte EDM, hizalama hatasının RMS'inin **karesiyle** büyür. (Bu $\sigma^2$
yasası bu projede defalarca doğrulandı; bu oturumda da: üs $p = 2.002$.)

---

## 2. Fikir: misalignment'ı değil, sahte-EDM'i hedeflemek

Önceki çalışmalarda doğal yaklaşım şuydu: **48 kuadrupolün her birinin kaçıklığını
ölç, sonra düzelt.** Kaçıklığı ölçmek için kapalı yörüngeyi (closed orbit, COD —
demetin halka çevresinde izlediği ortalama yol; BPM'lerle, yani konum ölçen
dedektörlerle okunur) kullanırız.

Ama bu yaklaşımın bir **no-go**'su (kanıtlanmış imkânsızlığı) var. Kaçıklığı
yörüngeden geri-çatmak bir **inversiyon** problemidir (yörünge → kaçıklık) ve bu
inversiyon, sahte-EDM'i süren kısımda **patolojik derecede kötü koşulludur**
(matrisin koşul sayısı çok büyük; gürültüyü ~$10^4$ kat büyütür). Bu yüzden
kaçıklığı tam ölçmek pratikte imkânsız (`squid_bpm_test.md §8`).

**Kullanıcının fikri (akıllı düzeltme):** Ama bizim asıl amacımız kaçıklığı ölçmek
değil ki! Asıl amaç **sahte-EDM'i sıfırlamak.** O hâlde neden kaçıklığı tam
çözmeye uğraşalım? Belki sahte-EDM'i **doğrudan** sıfırlayan bir düzeltme buluruz,
kaçıklığın kendisini hiç bilmeden.

Bir benzetme: Bir radyoda cızırtıyı (gürültüyü) gidermek istiyorsunuz. İki yol var:
(A) cızırtının tam kaynağını (hangi devre, hangi direnç) teşhis edip onarmak —
zor; (B) sadece cızırtıyı dinleyip, ses sıfırlanana dek bir düğmeyi çevirmek —
kaynağı hiç bilmeden. Akıllı düzeltme, (B) yaklaşımının pEDM'deki karşılığıdır.

---

## 3. İki kol: spin-gözlemli (A) vs yörüngeden-öğrenilmiş (B)

"Sahte-EDM'i doğrudan sıfırla" fikrini iki şekilde gerçekleştirebiliriz. Fark,
**neyi ölçtüğümüzde** yatar.

**Kol A — spin-gözlemli.** Sahte-EDM'i ($f = dS_y/dt$) **spinle doğrudan ölç**,
sonra bir düzeltici-mıknatıs (corrector) düğmesini $f$ sıfırlanana dek çevir.
Tıpkı radyo örneğindeki (B) yaklaşımı: dinle ve düğmeyi ayarla. Bu, projede
"spin ölç-trim" olarak zaten biliniyor ve **çalışıyor** (simetrik artığı ~6000×
temizler — `false_edm_harmonic_sinir.md §14.6`).

**Kol B — yörüngeden-öğrenilmiş ileri-harita.** Burada hile: spin ölçümü pahalı ve
yavaş (kutuplanma ölçmek dakikalar alır). **Ya sadece yörüngeye (BPM'lere) bakarak**
sahte-EDM'i öngörebilseydik? O zaman ucuz, hızlı, sürekli bir düzeltme yapardık.
Plan:
- Bir **sinir ağına (NN)** simülasyon verisiyle **yörünge → sahte-EDM** haritasını
  öğret (buna **ileri-harita** denir).
- NN'e "sahte-EDM'i sıfırlamak için yörüngeyi nasıl değiştirmeliyim" sorusunu
  sordur; orbit-görünür düğmelerle (corrector) düzelt.

Bu oturumun **açık sorusu** Kol B'ydi: *Yörüngeden öğrenilmiş bir ileri-harita,
inversiyon no-go'sunu atlatabilir mi?*

---

## 4. Kol B neden *makul* görünüyordu? (ileri-harita ≠ inversiyon)

Hipotez şuna dayanıyordu: **No-go bir *inversiyon* sınırıdır.** İnversiyon
(yörünge → kaçıklık) kötü koşulludur. Ama Kol B inversiyon yapmıyor; o bir
**ileri-harita** (yörünge → sahte-EDM). İleri yön genelde iyi-huyludur. Ayrıca
düzeltmeyi **orbit-görünür** düğmelerle, doğrudan **EDM'i hedefleyerek** yapıyoruz —
görünmez kaçıklığı geri-çatmaya hiç çalışmıyoruz.

Kulağa mantıklı geliyor. Ama bir varsayım gizli: **"sahte-EDM, yörüngenin bir
fonksiyonudur."** Yani aynı yörüngeye her zaman aynı sahte-EDM karşılık gelmeli.
Eğer bu doğruysa, NN o fonksiyonu öğrenebilir. Eğer **yanlışsa** — aynı yörünge
farklı sahte-EDM'lere karşılık gelebiliyorsa — hiçbir NN (ne kadar büyük olursa
olsun) bu haritayı öğrenemez, çünkü **öğrenilecek tek-değerli bir fonksiyon
yoktur.** İşte testimiz tam olarak bunu sınadı.

---

## 5. Anahtar geometri: yörünge neyi görür, neyi göremez?

Burası fikrin kalbi. Önce iki kavram:

**Simetrik vs antisimetrik kaçıklık.** Halka 24 hücreden oluşur; her hücrede bir
"odaklayan" (QF) ve bir "odaksızlaştıran" (QD) kuadrupol var. Bir hücredeki QF ve
QD kaçıklıklarını **zıt işaretli** kılarsanız buna *antisimetrik*, **aynı işaretli**
kılarsanız *simetrik* desen denir. Bu ayrım kritik çünkü:

- **Antisimetrik kaçıklık → büyük kapalı yörünge** (BPM'ler net görür).
- **Simetrik kaçıklık → minik kapalı yörünge** (BPM'ler neredeyse göremez).

Neden? Simetrik desen, yörüngeyi "yüksek frekanslı" (yüksek-$k$ harmonik) bir
biçimde dürter; halkanın yörünge tepkisi $G_k = C/|Q^2 - k^2|$ yasasıyla yüksek-$k$'da
**bastırılır** ($Q \approx 2.3 \ll k$). Yani simetrik kaçıklık yörüngede zar zor
iz bırakır — ona **"orbit-kör"** (orbit-blind) deriz.

**Acı gerçek:** Sahte-EDM'i asıl süren şey, tam da bu **orbit-kör simetrik**
alt-uzaydır. Yörünge düzeltmesi antisimetrik (görünür) kısmı temizler, ama geriye
sahte-EDM'i süren simetrik artık kalır (`orbit_ileri_olcum.md §6`).

**Bunu sayıyla görmek (SVD).** Yörünge tepki matrisi $R$'yi (kaçıklık → yörünge)
tekil-değer ayrışımıyla (SVD) parçalarız: $R = U\Sigma V^\top$. Her tekil değer
$\sigma$, bir kaçıklık-deseninin yörüngede ne kadar görünür olduğunu söyler. Bu
oturumda hesapladık:

| | tekil değer $\sigma$ | simetri | karakter |
|---|---|---|---|
| en büyük modlar | 28.4 … 8.6 | antisimetrik | **orbit-GÖRÜNÜR** |
| en küçük modlar | 0.16 … **0.147** | simetrik | **orbit-KÖR** |

Koşul sayısı $\sigma_\text{max}/\sigma_\text{min} = 193$. Yani **simetrik bir
kaçıklık, aynı büyüklükteki antisimetrik kaçıklığa göre yörüngede 193 kat daha az
iz bırakır.** Bu sayı, projedeki bağımsız bir analizle ($\kappa = 193$,
`drift_monitor/permode2.py`) **birebir** aynı çıktı — yani geometri doğru.

---

## 6. Karar-verici deney: "ikiz makineler"

Şimdi testin fikri çok basit hâle geliyor. **İki depolama halkası** hayal edin:

- **Makine A:** belli bir kaçıklık deseni.
- **Makine B:** Makine A'nın deseni + üzerine eklenmiş bir **simetrik (orbit-kör)
  pertürbasyon** (RMS 10 μm — yani tabanla aynı büyüklükte gerçek bir kaçıklık).

Simetrik pertürbasyon orbit-kör olduğu için, **iki makinenin BPM'lerle okunan
yörüngesi neredeyse aynıdır.** Soru: sahte-EDM'leri de aynı mı?

Bunu gerçek C++ spin izleyicisiyle (4D kapalı yörünge + Thomas-BMT, $p=2.00$
doğrulanmış estimator) ölçtük. **Sonuç çarpıcı:**

| sınıf | yörünge ayak izi (ΔCOD) | sahte-EDM değişimi \|Δf\| | hedefe göre |
|-------|------------------------:|--------------------------:|------------:|
| Antisimetrik (görünür) pertürbasyon | 115 μm | $9.3\times10^{-6}$ | ~9270× |
| **Simetrik (kör) pertürbasyon** | **1.7 μm** | $2.5\times10^{-7}$ | **~247×** |

İkinci satırı okuyun: **yörüngeyi yalnızca 1.7 μm değiştiren** (toplam yörünge
~97 μm; ve 100 μm'lik BPM ofset gürültüsünün *altında*) bir kaçıklık,
**sahte-EDM'i EDM-hedefinin 247 katı** kadar oynatıyor.

Başka bir deyişle: **BPM'leriyle ayırt edilemeyen iki makine, sahte-EDM'de
yüzlerce kat hedef farkı taşıyabilir.** Yani:

> **Sahte-EDM, kapalı yörüngenin tek-değerli bir fonksiyonu DEĞİLDİR** (kullanışlı
> hassasiyette). Aynı yörünge → farklı sahte-EDM.

Bu, §4'teki gizli varsayımı **çürütür.** Öğrenilecek bir fonksiyon yoksa, hiçbir
sinir ağı onu öğrenemez. **Kol B kavramsal olarak ölüdür** — NN'in büyüklüğünden
veya eğitim verisinden bağımsız.

![dejenerasyon figürü](/tmp/akilli_duzeltme/fig_kolb_dejenerasyon.png)

*Figür: Kırmızı üçgenler (simetrik) minik yörünge ayak izinde (~1.7 μm) ama büyük
sahte-EDM değişiminde; mavi daireler (antisimetrik) yörüngeyi izler. Kırmızı
gölgeli bölge BPM-ofset tabanının altıdır: simetrik kanal oraya düşer.*

---

## 7. Sonucu okumak: 7 nanometre duvarı

Sonucu nicel bir "gereksinim"e çevirelim. Kol B'nin simetrik sahte-EDM katkısını
(10 μm kör kaçıklık → $|\Delta f| \approx 2.5\times10^{-7}$) EDM hedefine
($10^{-9}$) çekebilmesi için, yörüngeyi hangi *doğrulukta* ölçmesi gerekir?

$$
\text{gereken BPM doğruluğu} \;\approx\; \underbrace{1.7\ \mu\text{m}}_{\text{COD ayak izi}}
\times \frac{\overbrace{10^{-9}}^{\text{hedef}}}{\underbrace{2.5\times10^{-7}}_{|\Delta f|}}
\;\approx\; 6.9\ \text{nm}.
$$

**Yaklaşık 7 nanometre.** Üstelik bu, ~100 μm'lik BPM ofsetlerinin *altında*
sağlanması gerekiyor — yani ofsetten 14000 kat daha ince bir sinyali çıkarmak.

Şimdi sürpriz: bu 7 nm sayısı, projede **tamamen farklı bir yöntemle** bulunan
inversiyon-no-go sınırıyla (**< 4 nm**, `squid_bpm_test.md §9.5`) **aynı mertebede.**
Yani:

> **İleri-harita (Kol B), inversiyon (geri-çatım) ile FARKLI bir yol değildir.**
> Her ikisi de simetrik (orbit-kör) sahte-EDM bilgisini ~nanometre-seviyesi
> yörünge sinyalinden çıkarmak zorundadır; o bilgi BPM sistematik tabanının
> altındadır. Hipotez ("ileri-harita inversiyonu atlatır") **çürütüldü.**

---

## 8. Neden Kol A çalışır da Kol B çalışmaz? (yerel ofset hilesi)

Bu en derin ve en öğretici nokta. Sahte-EDM (Berry fazı) neyi "görür"? **Demetin
kuadrupol merkezine göre yerel ofsetini.** Kuadrupol $i$'de demetin gördüğü yatay
ofset:
$$x_i^\text{yerel} = x_{\text{CO},i} - dx_i,$$
yani *(kapalı yörünge pozisyonu)* eksi *(kuadrupolün kendi kaçıklığı)*.

Şimdi sihir: simetrik kaçıklık için kapalı yörünge $x_{\text{CO}} \approx 0$
(orbit-kör!). Ama yerel ofset:
$$x_i^\text{yerel} \approx 0 - dx_i = -dx_i,$$
yani **kaçıklığın kendisi** — kaybolmaz! Berry fazı simetrik kaçıklığı **yerel
olarak tam görür**, ama BPM (sabit bir konumda $x_{\text{CO}}$ okur) **göremez.**

İşte iki kolun kaderini ayıran tek cümle:

- **Kol B**'nin girdisi BPM-yörüngesidir → simetrik yerel ofset girdide **yoktur**
  → öğrenecek bilgi yoktur. (Bölüm 6-7.)
- **Kol A**'nın girdisi spindir → spin, Berry fazının **doğrudan** gözlenebiliridir
  → simetrik bilgi girdide **vardır** → geri-besleme (düğmeyi $f$ sıfırlanana dek
  çevirme) çalışır.

Bir benzetme: İki hediye kutusu **dıştan birebir aynı** (aynı yörünge), ama
içleri farklı (farklı sahte-EDM). Kutuyu hafifçe sallamak (BPM'e bakmak) içeriği
söylemez. Ama içine bir sensör (spin) koyarsanız, ne olduğunu doğrudan görürsünüz.

> **Önemli incelik:** Kol A bile kaçıklığı *onarmaz*; sahte-EDM'i sıfırlamak için
> orbit-görünür düğmelerle **kasıtlı bir telafi-yörüngesi** ekler ($f=0$ yeter,
> optiği eski hâline getirmek gerekmez). Bilgi spinden geldiği için bu, yörünge
> tabanına çarpmaz. Ama spin gerektirir — yani Omarov'un SBA/spin-trim bölgesidir,
> "ucuz yörünge tarafı" değil.

---

## 9. Büyük resim: birleşik no-go (üç kol, tek duvar)

Bu oturumun asıl kazanımı, üç ayrı çabanın **aynı** duvara çarptığını göstermektir:

| Yöntem | Ne yapmaya çalışır | Simetrik kanalda sonuç |
|--------|--------------------|------------------------|
| Orbit-inversiyon ($R^{-1}$) | yörünge → kaçıklık geri-çat | koşul sayısı patlar, < 4 nm gerek |
| Orbit-lock-in (zaman-ortalama) | gürültüyü $\sqrt{N}$ ile yen | beyaz gürültüyü yener, **simetriği yenemez** |
| **Orbit-ileri-harita (Kol B)** | yörünge → sahte-EDM öğren | **~7 nm gerek; bilgi tabanın altında** |

Üçü de aynı fiziksel gerçeğe çarpar: **sahte-EDM'i süren simetrik kaçıklık,
yörüngede neredeyse iz bırakmaz.** Yörünge hangi akıllı işleme tabi tutulursa
tutulsun (ters çevir, ortala, sinir ağıyla öğren) — bilgi orada olmadığından
çıkarılamaz. Bu, **gözlenebilirlik (observability) tabanlı bir no-go**'dur:
*ölçemediğin şeyi hiçbir algoritma kurtaramaz.*

Tek kaçış: **gözlenebiliri değiştirmek** — yani spine bakmak (Kol A). Spin,
geometrik fazın doğrudan gözlenebiliridir. Bu da bizi Omarov ve ark.'nın
(PRD 105, 032001) spin-tabanlı yöntemlerine geri götürür (`omarov.md`).

---

## 10. Empirik pekiştirme: gerçek bir harita da öğrenemez

§6'daki karar **modelden bağımsızdır** (ikiz makineler herhangi bir yörünge→sahte-EDM
haritasını çürütür). Yine de somut olması için, jenerik rastgele kaçıklıklarla
**gerçek bir regresyon** eğittik: 80 örnek, simetri ağırlığı $w$ tam-antisimetrikten
($w=0$) tam-simetriğe ($w=1$) taranıyor; girdi = 48-BPM yörünge, çıktı = C++ ile
ölçülen sahte-EDM. Üç bulgu, §6-9'u pekiştirir — ve bir **incelikle** öğreticidir.

**Bulgu 1 — Orbit-kör konfigler "temiz" değildir.** Ortalama sahte-EDM, simetri
arttıkça düşer ama **sıfırlanmaz:**

| $w$ | karakter | ⟨\|sahte-EDM\|⟩ | hedefe göre |
|-----|----------|----------------:|------------:|
| 0.00 | tam antisimetrik (görünür) | $3.3\times10^{-6}$ | ~3300× |
| 1.00 | **tam simetrik (orbit-kör)** | $1.8\times10^{-7}$ | **~180×** |

Yani yörüngede neredeyse hiç iz bırakmayan kaçıklıklar bile sahte-EDM'i hedefin
**180 katına** taşıyor. Görünmezlik, masumiyet değildir.

**Bulgu 2 — Harita yalnız "kolay" (görünür) kısmı öğrenir.** Temiz yörüngeyle
eğitilen basit bir model (Ridge), sahte-EDM varyansının ~%32'sini açıklar. Ama bu
%32, tamamen **orbit-görünür (antisimetrik)** paydır — yani **orbit-düzeltmenin
zaten temizlediği**, bizim ihtiyaç duymadığımız kısım.

> **Öğretici incelik:** "Peki 100 μm'lik BPM ofseti haritayı bozmaz mı?" diye
> sorabilirsiniz. Şaşırtıcı cevap: *akıllı bir öznitelik için bozmaz.* Yönlü-alan
> $\sum_i(x_iy_{i+1}-x_{i+1}y_i)$, kapalı bir halkada **sabit ofsete değişmezdir**
> (ofset terimleri toplamda iptal olur). Eklenen 100 μm ofsetle bile model yine
> ~%32'de kalır. Demek ki **duvar, BPM ofsetinin kendisi değil** — duvar, simetrik
> kanalın yörüngede *ilkesel* görünmezliğidir. Ofseti aşan akıllı öznitelik bile
> simetrik kanala **ulaşamaz**, çünkü orada okunacak yörünge zaten yok.

**Bulgu 3 (karar-verici) — harita simetrik kanala taşınmıyor.** Modeli yalnız
**antisimetrik** (görünür) konfiglerle eğitip **simetrik** (kör) konfiglerde test
edersek: $R^2 = -134$ — yani ortalamayı tahmin etmekten ~135 kat **daha kötü.**
Görünür rejimde öğrenilen şey, sahte-EDM'i süren kör rejime **hiç transfer
olmuyor.** Bu, "ikiz makine" sonucunun istatistiksel kardeşidir: bilgi girdide
yoksa, model onu icat edemez.

(Ek olarak: Berry yönlü-alan proxy'sinin korelasyonu $w$'ye göre tutarsız
seyreder — $-0.53, +0.02, -0.55, -0.08, -0.36$ — yani tek bir temiz kapalı-form
fonksiyonel ampirik olarak pinlenmiyor; `orbit_ileri_olcum.md §3` ile birebir.)

**Özet:** Yörüngeden öğrenilen harita, sahte-EDM'in **görünür payını** yakalar
(gerekmeyen kısım) ama **orbit-kör simetrik kanala — işin tüm bel kemiğine —
taşınamaz.** Sinir ağı, olmayan bilgiyi icat edemez.

---

## 11. Sık sorulan sorular

**S: "Sahte-EDM yörüngenin fonksiyonelidir" demiştiniz; şimdi "fonksiyonu değil"
diyorsunuz. Çelişki mi?**
C: Hayır. Matematiksel olarak yörünge → kaçıklık tersi vardır (matris tam-ranklı),
dolayısıyla *sonsuz hassasiyette* sahte-EDM yörüngeden belirlenir. Ama *kullanışlı*
(deneysel) hassasiyette belirlenmez: gerekli yörünge doğruluğu ~7 nm, gerçek
BPM ofsetleri ~100 μm. "Fonksiyonel ama pratikte tek-değerli değil" — fark budur.

**S: Daha büyük/derin bir sinir ağı veya milyonlarca örnek yardımcı olmaz mı?**
C: Hayır. Sorun NN'in kapasitesi veya veri miktarı değil; **bilginin girdide
olmaması.** Simetrik sahte-EDM yörünge girdisinde (BPM ofset tabanının altında)
yok. Hiçbir model, girdisinde bulunmayan bir ayrımı öğrenemez.

**S: BPM'leri 7 nm doğruluğa çıkaramaz mıyız (örn. SQUID-BPM)?**
C: SQUID-BPM'in tek-atış gürültüsü düşüktür, ama buradaki engel gürültü değil
**sistematik ofset** (~100 μm). 7 nm'lik bir sinyali 100 μm ofsetin altından
çekmek, ofseti 7 nm'de bilmeyi gerektirir — bu da inversiyon no-go'sunun ta
kendisi. (SQUID'in nerede *işe yaradığı* için `squid_bpm_test.md §9.4`.)

**S: O zaman akıllı düzeltme tümüyle ölü mü?**
C: *Yörünge tarafı* (Kol B) ölü. *Spin tarafı* (Kol A) çalışır — ama o, yörünge
gözlemine dayanmaz; spin ölçer ve bu zaten Omarov/spin-trim bölgesidir (özgün
katkısı dar). Yani "akıllı düzeltme orbit-kör simetrik kısmı ucuz yörünge gözlemiyle
kapatır" umudu kapandı.

**S: Bu negatif sonuç neden değerli?**
C: Çünkü üç bağımsız yöntemi (inversiyon, lock-in, ileri-harita) tek bir
gözlenebilirlik sınırında birleştirir. pEDM literatüründe (Omarov dahil) geometrik-faz
kontrolü "yörünge/ayrım ölçümüne" dayandırılır ama bu ölçümün **simetrik artığa
körlüğü** nicel olarak gösterilmemiştir. Bizim katkımız tam burada: kesin bir
sınır teoremi (`omarov.md §9-10`, `akilli_duzeltme.md §7`).

---

> **Kayıt notu (2026-06-29):** Bu belge `akilli_duzeltme.md`'nin (terse teknik
> kayıt) pedagojik kardeşidir. Sayılar gerçek C++ izleyiciyle ($p=2.002$
> doğrulanmış estimator) üretildi; keşif kodu `/tmp/akilli_duzeltme/` altında
> (proje konvansiyonu), reprodüksiyon yolları `akilli_duzeltme.md §8`'de.
> `integrator.cpp` değiştirilmedi.
