# Klasik BBA Neden Çalıştı? — Öğretici Açıklama

> ## ⚠️ ÖNEMLİ DÜZELTME (2026-07): β-beat, ham yörüngede BBA'yı bozuyor (§3–5 fazla iddialıydı)
>
> Bu belge klasik BBA'nın **temiz optikte** çalıştığını doğru anlatır (C++:
> 356× → 1.62× hedef, sub-μm merkezler). **Ama** §3–§5'teki "null model-
> bağımsız, β-beat şeffaf" iddiası **fazla güçlüydü ve C++ tarafından
> düzeltildi.** Gerçek durum:
>
> - β-beat %1 + **düzeltilmemiş 0.37 mm yörünge** altında BBA çöküyor
>   (bastırma ~1×). Sebep FİZİK: modülasyon tüm optiği esnetir, diğer kaçık
>   quad'ların kurduğu **büyük yörüngeyi** yeniden taşır ("optik-nefes"), ve
>   β-beat bu nefesin desenini feed-down'dan saptırınca her BPM'in null'u ayrı
>   yere kayar (~2 μm). Yani null, katıksız model-bağımsız DEĞİL — **nefes
>   terimi** kadar model-duyarlı.
> - **AMA çözülebilir:** izole test kesin gösterdi ki bias **diğer quad'ların
>   yörüngesinden** gelir (o yörünge sıfırlanınca bias 2 μm → 0.02 μm). Nefes
>   ∝ (düzeltilmemiş yörünge × β-beat); yörüngeyi domine eden antisimetrik
>   kısım kolay-düzeltilebilir → **BBA'dan önce orbit düzelt / iterasyon** ile
>   bias küçülmeli. (Standart BBA pratiği zaten "önce orbit düzelt"tir; ilk
>   kodum bunu atlamıştı.)
>
> **Okuma kuralı:** Aşağıda §3–§5'te "β-beat şeffaf / null tamamen model-
> bağımsız" diyen cümleleri şöyle oku: *bu, yörünge küçükken (nefes ihmal
> edilebilirken) doğrudur; ham büyük yörüngede nefes terimi devreye girer ve
> β-beat'e duyarlılık doğar.* İteratif çözümün C++ testi sürüyor; sonuç
> `separation_bba_testleri.md §5.2`'ye. (Analitik prototipin "şeffaf" öngörüsü
> yörünge nefesini eksik modelledi; kural gereği C++ esas.)

> **Bu belge ne için?** Aylarca "bu sinyal demetten ölçülemez" diye
> düşündüğümüz bir problemi, hızlandırıcı dünyasının standart bir tekniği
> (klasik demet-tabanlı hizalama, BBA) çözdü. Bu şaşırtıcı bir sonuç ve
> "neden?" sorusunu hak ediyor. Burada hiçbir ön bilgi varsaymadan, sezgiyle
> kurarak anlatıyorum: sinyal nedir, neden üç farklı ölçüm yolu var, neden
> ikisi çöküp üçüncüsü çalıştı, simülasyonu nasıl kurdum ve yol boyunca
> hangi hatalardan ne öğrendim.
>
> **Bir uyarı, en baştan:** Aşağıdaki analitik (kâğıt-kalem) hesaplar yalnızca
> *nereye bakacağımızı* söyler. Nihai hüküm her zaman tam simülasyona
> (`integrator.cpp` — gerçek parçacık + spin izleyici) aittir. Sayıların
> hangisinin C++, hangisinin analitik-kılavuz olduğunu her yerde belirtiyorum.

---

## 0. Bulmaca

Proton EDM deneyinde en tehlikeli sistematik, halkadaki kuadrupol mıknatısların
birkaç mikronluk kaçıklığının ürettiği **sahte EDM** sinyalidir. Bu projede
uzun süre şu sonuca varılmıştı:

> "Sahte EDM'i süren kaçıklık bileşeni demetin yörüngesine neredeyse görünmez;
> dolayısıyla onu yörüngeden ölçüp düzeltmek mümkün değil."

Bu hüküm boşuna değildi — arkasında altı ayrı yöntemin başarısızlığı vardı
(tepki matrisi ters çevirme, SVD, sinir ağı, LOCO + drift izleme, K-modülasyon
genlik okuma, ...). Sonra bir işbirliği üyesinin yorumu üzerine, o güne kadar
hiç denenmemiş bir yöntemi — **standart kuadrupol BBA'sını** — simüle ettik.
Çalıştı: sahte EDM'i hedefin ~200 katından hedefin eşiğine indirdi.

Nasıl oldu da "imkânsız" denen şey, aslında hızlandırıcı laboratuvarlarında
her gün yapılan bir ölçümle çözülüyordu? Cevap, "yörüngeden ölçmek" diye tek
bir şey olmadığında gizli. Onu göreceğiz.

---

## 1. Sahne: kaçık bir kuadrupol demete ne yapar?

Bir kuadrupol mıknatısın işi demeti odaklamaktır. Tam merkezinden geçen bir
demet hiçbir sapma görmez — merkezde alan sıfırdır. Ama demet merkezden **d**
kadar uzaktan geçerse, kaçıklıkla orantılı bir yana-itme (kick) yer:

$$\theta \;=\; K L \cdot d,$$

burada $KL$ mıknatısın gücü, $d$ ise **demetin mıknatıs merkezine göre
uzaklığıdır.** Dikkat: itmeyi belirleyen şey demetin mutlak konumu değil,
**merkeze göreli** konumudur. Bu ayrım tüm hikâyenin kalbi olacak.

Şimdi merkez neden kaçık olabilir? İki sebep: (a) mıknatıs mekanik olarak
yerinden oynamış olabilir; (b) daha sinsisi, mıknatısın içindeki küçük alan
kusurları "manyetik merkezi" geometrik merkezden kaydırır. İkinci durumu
yalnızca demet görür, hiçbir cetvel görmez. Ölçmek istediğimiz $m$ (her
kuadrupolün merkez kaçıklığı) işte budur.

**Sahte EDM neden buna bağlı?** Demet kaçık bir kuadrupolden geçerken gördüğü
alan spini biraz döndürür. Yatay kaçıklık spini bir eksen etrafında, dikey
kaçıklık başka eksen etrafında çevirir. Farklı eksenlerdeki iki dönüş
**sırası değiştirilemez** (önce sağa sonra yukarı bakmak ile tersi, kafanı
farklı yöne çevirir); aradaki fark her turda spinin dikeyde küçük bir kalıcı
kaymasıdır — tam da gerçek EDM'in görüneceği yerde. Bu, "geometrik faz"dır ve
büyüklüğü $\propto d_x \cdot d_y$'dir. Yani sahte EDM, merkez kaçıklıklarının
**çarpımıyla** ölçeklenir. Merkezleri yeterince iyi ölçüp demeti onlardan
geçirebilirsek ($d \to 0$), sahte EDM söner. **Bütün mesele $m$'i ölçebilmek.**

---

## 2. Aynı sinyali kullanmanın üç yolu

Her kuadrupolün merkez kaçıklığı $m_i$ hakkındaki bilgi tek bir fiziksel
büyüklükte saklı: kuadrupolün gradyanını azıcık "titretirsek" (gücünü
$KL \to KL(1+\varepsilon)$ yaparsak), demete verdiği ekstra itme

$$\Delta\theta_i \;=\; \varepsilon \, KL \cdot \big(y(i) - m_i\big)$$

olur — yani **demetin o noktadaki konumu ile merkez arasındaki farkla**
orantılı. Bu ifade sıfır olduğunda ($y(i) = m_i$) demet tam merkezden geçiyor
demektir. İşte üç yöntemin hepsi bu $\Delta\theta$'yı kullanır; fark, onu
**nasıl** kullandıklarındadır. Üçünü sırayla, birer benzetmeyle anlatayım.

### Yol 1 — "Herkesi aynı anda tart": global inversiyon
Bütün kaçıklıkları tek bir denklem sisteminde birlikte çözmeye çalışırız:
48 BPM'in (demet konum monitörü) okuduğu yörüngeyi al, tepki matrisini ters
çevir, 48 kaçıklığı birden çıkar.

**Benzetme:** 48 kişinin tek tek kilolarını, hepsini farklı gruplar hâlinde
aynı bandüle çıkarıp toplam okuyarak çözmeye çalışmak. Bazı kombinasyonlar
(özellikle "simetrik" olanlar — herkesin aynı yöne eşit kayması gibi) baskülü
neredeyse hiç oynatmaz. O kombinasyonu geri hesaplamak için baskülün minik
titreşimini devasa bir sayıya bölmeniz gerekir → baskülün ufacık hatası, o
kişilerin kilosunda kilometrelerce yanlışa dönüşür. Matematikte buna "kötü
koşullanma" denir; bizim halkamızda simetrik yönde bu büyütme ~10 000 kat.
Beyaz gürültüyü uzun ölçümle yenebilirsiniz, ama optik model hatası
(β-beat: gerçek makinenin odaklamasının modelden %0.5–1 sapması) **ortalanmaz**
— ve bu büyütmeyle çarpılınca felaket olur (simetrik hata sinyalin 7 katı).

### Yol 2 — "Kalabalığın salınımına bakıp tek kişiyi çıkarmaya çalışmak": genlik okuma
Her kuadrupolü ayrı bir frekansta titretir, BPM sinyalinde o frekansın
**genliğini** okur, kalibrasyona böleriz. Umut: genlik $\propto (y(i)-m_i)$,
merkez kaçıklığını verir.

**Sorun — "optik-nefes":** Bir kuadrupolü titrettiğinizde yalnız onun kendi
itmesi değişmez; tüm halkanın odaklaması azıcık esner (β fonksiyonları
"nefes alır"). Halkada zaten 48 kaçıklığın birlikte kurduğu **büyük** bir
yörünge var (~0.37 mm); bu koca yörünge esneyen optikten yeniden geçince
BPM'de birkaç mikronluk bir salınım yapar. Aradığımız sinyal ise ~0.9 μm.
Yani ölçtüğünüz genlik, titrettiğiniz kuadrupolün kendi merkezini değil,
**komşularının kurduğu yörüngenin sallanmasını** yansıtır (etki 265 kat daha
büyük).

**Benzetme:** Kalabalık bir salıncaklı köprüde tek bir kişiyi dürtüp,
köprünün sallanmasından o kişinin ağırlığını okumaya çalışmak. Köprü zaten
herkesin yüküyle sallanıyor; sizin dürtüşünüzün yarattığı fark, o genel
sallanmanın içinde kaybolur. Üstelik bu sallanma rastgele gürültü değil,
**tutarlı (koherent)** bir bozulma — daha çok BPM koyup ortalama almak onu
söndürmez. Bu dal bu yüzden öldü.

### Yol 3 — "Terazinin dengelendiği yeri ara": yerel null (çalışan yol)
Kuadrupolü titretirken, demeti o kuadrupolün içinde **yavaşça kaydırırız**
(yerel bir "bump" ile tararız) ve titretme tepkisinin **tam sıfırlandığı**
demet konumunu buluruz. O sıfır noktası, tanımı gereği, kuadrupolün merkezidir.
48 kuadrupolü **tek tek, birbirinden bağımsız** böyle ölçeriz.

**Benzetme:** Bir tahterevallinin dengelendiği noktayı bulmak. Tahterevalliyi
tartmanıza (genliği okumanıza) veya odadaki başka her şeyi bilmenize gerek
yok — sadece düzleşene kadar oynatırsınız. Denge noktası nerede durduğunuzdur;
tahterevallinin ağırlığından (Yol 2'nin çıkmazı) da, odadaki diğer eşyalardan
(Yol 1'in çıkmazı) da bağımsızdır.

Bu üçüncü yol, projede hiç denenmemişti. Şimdi neden çalıştığını görelim.

---

## 3. İşin kalbi: sıfır aramak neden temelden farklı?

Yol 3'ün sırrı tek cümlede: **bir eğrinin sıfırı nerede olduğu, eğrinin
dikliğinden bağımsızdır.**

Titretme tepkisini demet konumuna karşı çizin. Elde ettiğiniz şey bir doğru:
$A = s \cdot (y - m_i)$. Burada $s$ eğim (tepkinin "gücü"), $m_i$ ise doğrunun
ekseni kestiği yer — merkez. Şimdi düşünün:

- **Yol 2 (genlik okuma)** doğrunun bir noktadaki *yüksekliğini* okur ve $s$'e
  böler. $s$'i bilmek zorundadır; ve $s$'i bozan her şey (nefes, kalibrasyon)
  cevabı bozar.
- **Yol 3 (null)** doğrunun *sıfırı kestiği yeri* okur. Bu yer, $s$ ister
  büyük ister küçük olsun, ister gürültüyle azıcık kaysın, **aynı yerdedir.**
  Eğimi bilmeye gerek yok; onu ölçmeye bile gerek yok.

Bunun neden bu kadar güçlü olduğunu görmek için, eski yöntemleri öldüren
duvarları teker teker bu doğrunun üstüne koyalım:

- **β-beat (optik model hatası):** eğimi $s$ değiştirir — doğru daha dik ya da
  daha yatık olur. Ama **sıfırı kestiği noktayı kaydırmaz.** Yol 1'i %0.5
  β-beat'te öldüren şey (283 μm hata), buraya hiç dokunmaz. Bu, çalışan yolun
  belki de en önemli özelliği.
- **Optik-nefes:** nefes, demetin o kuadrupoldeki konumunu ($y$'yi) büyütür —
  ama biz zaten $y$'yi tarama ekseni olarak *ölçüyoruz*, varsaymıyoruz. Nefes
  büyük $y$ demek, o da taramada daha geniş bir eksen demek; sıfırın yeri yine
  $y = m_i$. Yol 2'nin katili, Yol 3'te sadece bir eksen etiketine dönüşür.
- **Kötü koşullanma (1/σ_min):** bu, matris ters çevirmenin hastalığıydı.
  Yol 3 hiçbir matris ters çevirmez — her kuadrupolün merkezi ayrı bir
  skalerin sıfırıdır. "Simetrik mod" kavramı, 48 kuadrupolün *kolektif*
  yörünge tepkisinin bir özelliğiydi; her merkezi tek tek ölçtüğünüzde o
  kolektif özellik devreye bile girmez.
- **BPM ofseti:** BBA'da bütün büyüklükler *farktır* (titretme açık eksi
  kapalı). Statik ofset farkta düşer.

Yani üç yol da aynı sinyali görür, ama Yol 3 onu, eski duvarların hiçbirinin
tutunamayacağı bir soruya çevirir: "bu tek sayı nerede sıfır oluyor?"

---

## 4. Aynı fikrin titiz hâli: rank-1 yerellik

Yukarıdaki sezgi, matematikte kesin bir ifadeye oturur ve bu, "belki bir
mertebe yaklaşımıdır, tam makinede bozulur" endişesini kapatır.

Bir kuadrupolün gücünü $\delta k$ kadar değiştirmenin kapalı yörüngeye etkisi,
yörünge Green fonksiyonunun (bir kick'in nereye nasıl yayıldığını söyleyen
$G$ matrisinin) bir güncellemesidir. Bu güncelleme **rank-1**'dir:

$$\delta G \;=\; G[:,i]\,\big(\delta k\big)\,G[i,:].$$

Sadeleştirilmiş sonucu şu: quad $i$'yi titretmenin bütün BPM'lerdeki tepkisi
**tek bir ortak çarpanın** katlarıdır:

$$A(\text{tüm BPM'ler}) \;=\; G[:,i]\,\cdot\,\varepsilon KL\,\cdot\,\big(y(i)-m_i\big).$$

Buradan üç şey doğrudan çıkar:
1. **Bütün BPM'ler tam aynı noktada sıfırlanır:** $y(i) = m_i$. Yani null,
   48 boyutlu bir sistemin değil, tek bir skalerin sıfırıdır — ve 48 BPM'in
   hepsini kullanmak yalnızca o tek sayıyı daha hassas belirler.
2. **Bu, bir yaklaşım değil, kesin:** rank-1 yapı, bump dâhil bütün kaynaklar
   için geçerlidir; "birinci mertebe, sonra bozulur" diye bir madde yok.
3. **Simetrik/antisimetrik ayrımı buraya girmez:** o ayrım $G$'yi (kolektif
   haritayı) ters çevirdiğinizde ortaya çıkan bir şeydi. Burada $G$'yi ters
   çevirmiyoruz; onu yalnızca "hangi BPM daha çok sinyal görüyor" diye ağırlık
   olarak kullanıyoruz.

---

## 5. Eski duvarlar neden bu yolda yok — özet tablo

| Duvar (eski yöntemleri öldüren) | Yol 1–2'de etkisi | Yol 3'te (BBA) neden yok |
|---|---|---|
| Kötü koşullanma, $1/\sigma_{\min}$ | inversiyonda hata ×10⁴ | inversiyon yok; per-quad tek skalerin sıfırı |
| β-beat (model hatası) | simetrik hata 283 μm @ %0.5 | null model-bağımsız: eğim değişir, sıfırın yeri sabit |
| Optik-nefes | genlik komşu-yörünge-domine | nefes = tarama ekseni; sıfıra etkisiz |
| BPM ofseti | mutlak okumayı kirletir | her şey fark; ofset düşer, düzeltmede golden-orbit ile geri alınır |
| BPM gürültüsü | simetrikte <4 nm ister | yerel ölçüm; düz $1/\sqrt{N}$ ile ortalanır, büyütme yok |

---

## 6. Simülasyonu nasıl kurdum

Amaç, bu üçüncü yolu **gerçek demet + spin dinamiğiyle** (analitik kısayolla
değil) uçtan uca sınamak: BBA ile 48 merkezi ölç, düzelt, kalan sahte EDM'i
doğrudan spin izleyicisiyle ölç. Adımlar:

1. **Titretme.** Her kuadrupolün gradyanını `quad_dG[i] = +0.02` ile %2
   büyütürüm (bu düğme C++'ta zaten vardı; `integrator.cpp`'ye dokunmadım).
   *Tek istisna:* hücre-0'ın QF kuadrupolü özel bir eleman tipi ve bu düğmeyi
   okumaz; o yüzden sim'de o kuadrupolü ölçemiyorum, kaçıklığını sıfır alıp
   bunu açıkça raporluyorum. (Gerçek makinede o kuadrupol başka yoldan modüle
   edilir.)

2. **Bump = komşu kuadrupola verilen küçük merkez kayması.** Demeti bir
   kuadrupolde kaydırmak için komşusuna bir "düzeltici" uygularım. Burada
   önemli bir fizik özdeşliği var: bir kuadrupola bindirilen dipol düzeltici
   bobin, alanı $g(y-m) + \Delta B = g\big(y - (m - \Delta B/g)\big)$ yapar —
   yani **etkin merkezi kaydırır.** Dolayısıyla düzelticiyi modellemek,
   kuadrupolün `quad_dy/dx` kaçıklığına bir terim eklemekle *birebir aynıdır*
   (hem yörünge hem spin için). *(Yan not: `dipole_tilt` düğmesini kullanmam;
   o, elektrik deflektörün tiltini değil "eşdeğer manyetik" bir radyal alan
   enjeksiyonunu uygular ve spin için doğrudan bir EDM taklidi üretir — bu
   işte yanıltıcı olurdu.)*

3. **Hangi komşu, ne kadar?** Bump için komşu kuadrupol adaylarından (±1, ±2)
   **ilgili düzlemin** analitik tepki matrisinde en güçlü olanı seçilir.
   Dikey düzlemde bu ±1, yatayda ±2 çıkıyor — çünkü QF yatayı odaklarken
   dikeyi dağıtır, iki düzlemin yapısı farklıdır. (İlk koşumda yatay bump'ı
   yanlışlıkla *dikey* matrisle boyutlamıştım; bu, yatay merkez hatasını 4 kat
   şişirdi. Düzelttim — küçük ama öğretici bir hata.)

4. **Tarama.** Demeti ±150 μm iki noktada kaydırıp tepkiyi ölçerim. Tepki
   lineer olduğundan iki nokta doğruyu (ve sıfırını) tanımlar. **Kritik:**
   tarama ekseni, o kuadrupoldeki BPM'in kendi okumasıdır — yani $y(i)$'yi
   *ölçerim*, modelden *varsaymam*. Bump'ın modelden kurulmuş olması bu yüzden
   sonucu yanlılaştırmaz.

5. **Yörünge okuma — pahalıya öğrenilen ders.** Demeti ideal eksenden fırlatıp
   zaman ortalamasıyla okumak YANLIŞTIR: bu, ~0.2 μm'lik bir betatron salınım
   artığı bırakır ve o artık, null hesabında eğime bölününce ~25 kat büyüyüp
   ~5 μm'lik sahte bir hataya döner. İlk C++ koşumum tam bunu gösterdi: üç
   kuadrupolde null'lar merkezlerden ortalama 5.8 μm kaymıştı. Çözüm:
   parçacığı önce **4D kapalı yörüngeye oturtup** ($find\_co\_4d$) sonra
   okumak. Bunu yapınca bias 5.8 μm → 0.13 μm'e düştü (46 kat). Gerçek
   makinedeki karşılığı: okuma zaten çok-turluk ortalamayla betatrondan
   arınmış olur.

6. **Null hesabı.** Her BPM için "tepki = eğim × konum + sabit" doğrusunu
   kurar, ortak sıfırı eğimi büyük BPM'lere daha çok ağırlık vererek (en
   küçük kareler) bulurum. 48 BPM'in hepsi katkı verir.

7. **Düzeltme ve nihai ölçüm.** Kestirilen merkezleri kaçıklıklardan düşerim
   (madde 2'deki özdeşlik gereği bu, gerçek düzeltici bobinle aynı işlem).
   Sonra kalan sahte EDM'i **spin izleyicisiyle doğrudan ölçerim**
   (`fast_est.fast_measure`: 4D kapalı yörünge + model-fit; $\sigma^2$
   testinde $p=2.00$ ile doğrulanmış estimatör). "$f = A\sigma^2$" gibi bir
   formül tahmini sonuç olarak asla kullanılmaz.

**Sistematikler** (yazım sırasında süren koşum, `--bbeat 0.01
--bpm-offset 100e-6 --bpm-noise 1e-6`): β-beat, her kuadrupola statik bir
gradyan hatası olarak **C++ dinamiğine gömülür** (titretmenin üstüne biner —
yani BBA'yı gerçek, kusurlu makinede yaparız). BPM ofseti okuma katmanındadır
ve golden-orbit'e sürülünce iptal olur (sentetik testte tam 0 nm). Gürültü de
okuma katmanındadır; farklı ortalama sayılarında ($N_{\rm avg}$) merkez hatası
ölçülür ve her seviyede kalan sahte EDM yine **C++ spin izleyicisiyle**
çapalanır.

---

## 7. Şimdiye kadarki C++ sonuçları

| Adım | Sonuç |
|---|---|
| Null doğrulaması (3 quad, kapalı-yörünge okumalı, gürültüsüz) | merkez biası ortalama **0.13 μm** (ideal-eksen okumada 5.8 μm idi) |
| Uçtan uca ilk geçiş (47 quad × 2 düzlem; yatay bump henüz yanlış ölçekli) | merkez hataları dikey 0.19 μm, yatay 0.83 μm; **ham sahte EDM hedefin 356 katı → düzeltme sonrası 1.62 katı (220× bastırma)** |
| Yatay bump düzeltmesi + β-beat + ofset + gürültü | *(koşum bitmek üzere; sonuç `separation_bba_testleri.md §5.2`'ye)* |

1.62× hedef değeri, **düzeltilmemiş yatay bump'lı, iterasyonsuz tek geçişin**
sonucudur — yani muhtemelen daha da iyileşecek. Sahte EDM $\propto d_x d_y$
olduğundan, yatay hatayı (0.49 μm) dikey seviyesine (0.16 μm) indirmek kalan
sinyali hedefin altına taşımalı; bu bir *beklenti*dir, yerini biten koşumun
ölçümüne bırakacak.

---

## 8. Analitik kılavuz (sonuç değil, sadece nereye bakacağımızı söyler)

Hızlı analitik prototip (`classic_bba_sim.py`; per-quad Twiss, nefes dâhil)
şunları öngördü ve C++ koşumlarında aranacak yeri işaretledi:
- Simetrik ve antisimetrik yönlerde **eşit** hassasiyet (körlük yok).
- β-beat'e **şeffaflık** (%1 β-beat merkez hatasını 0.32 → 0.31 μm, yani
  neredeyse hiç değiştirmiyor).
- Hata saf $1/\sqrt{N_{\rm avg}}$ ile düşüyor; ~0.1 μm'e kadar bir taban yok.
- Kaba kural: null hassasiyeti $\approx$ (BPM okuma gürültüsü) $\times$ 25
  (titretme derinliği %2 iken eğimin tersi).

Bu öngörüler C++ ile teyit edilene kadar *hipotez*tir.

---

## 9. Peki bunu neden daha önce görmedik? (dürüst muhasebe)

Bu, teknik olduğu kadar yöntemsel bir ders olduğu için ayrı yazıyorum.

1. **Negatif sonuçlar sınıf-özgüydü ama sınıf-bağımsız cümlelerle
   kaydedildi.** "Simetrik mod orbit'ten ölçülemez" hükmünün *kanıtı* yalnız
   Yol 1 (inversiyon) ve Yol 2 (genlik) içindi. Yol 3 hiç koşulmadı. Bir
   negatif sonucu, hangi ölçüm sınıfını kapsadığını yazmadan genellemek, kör
   nokta yaratır.
2. **Bir dal erken kapandı.** Per-quad K-modülasyon fikri "operasyonel olarak
   ağır" diye işaretlenmiş ve onun *genlik-okuyan* çeşidi nefesle ölünce, tüm
   per-quad ailesi ölü sayılmıştı. Oysa nefes, ailenin yalnız genlik okuyan
   üyesini öldürüyordu; null-arayan üyesine dokunmuyordu.
3. **Çıkışı iki şey sağladı:** bir dış itiraz (işbirliği üyesinin K-mod + BPM
   + HLS planı) ve "lafla cevap verme, simülasyonla göster" ilkesi. Analitik
   prototip yönü gösterdi; hükmü C++ verdi.
4. **Alınacak ders:** her negatif sonucu "hangi ölçüm sınıfı için" etiketiyle
   kaydet. Bir sınıfın ölümü, aileyi öldürmez.

---

## 10. Sınırlar ve dürüst açık uçlar

- **En önemli açık kalem — donanım (simülasyon dışı):** BBA, kuadrupolün
  gradyanını titretirken ölçüm yapar. Gerçek bir mıknatısta gradyanı
  değiştirmek, manyetik merkezi de biraz oynatabilir (histerezis, ısınma).
  Bu ~μm mertebesinde olabilir ve öyleyse yöntemin *gerçek* tabanını o
  belirler. Simülasyonumuz ideal mıknatıs varsayar; bu kalemi ancak makine
  ölçümü ya da mıknatıs modeli kapatır.
- Tek seed (tek rastgele kaçıklık deseni); çok-seed sağlamlaştırma sırada.
- Kuadrupol tilt'i (dönme) ayrı bir sahte-EDM kanalıdır (~0.3 mrad tolerans),
  bu ölçümün dışında.
- Model lineer (sekstüpol yok); büyük yörüngeli rejimlerde gerçek makine daha
  erken bozulur, ama BBA yerel/küçük-sinyalli çalıştığı için görece korunur.

## 11. Bu makaleye ne yapar?

Makalenin şu anki ana iddiası ("simetrik bileşen orbit'ten hiçbir standart
teknikle gerekli seviyede ölçülemez") **revize edilmeli** (kullanıcı onayıyla).
Doğru ve daha güçlü ifade şu: *global/tek-atış ölçüm sınıfları (inversiyon,
genlik okuma, ayrım-sürme) simetrik bileşene kördür — ve bunu nicel duvarlarla
gösteriyoruz; ama yerel null-arayan sınıf (klasik BBA) onu görür ve sahte
EDM'i hedefe taşır.* Böylece makale tek bir olumsuz sınırdan çıkıp, "hangi
ölçüm işe yarar, hangisi yaramaz ve neden" diyen çok daha yararlı bir yapıya
kavuşur. Omarov'un (PRD 105, 032001) açık bıraktığı ölçüm-zinciri boşluğu da
iki yönden kapanır: karşı-dönen demet ayrımını sürmek tek başına yetmez
(simetriğe kör, doğrudan ölçtük), ama aynı donanımla (K-modülasyon + BPM)
yapılan null-BBA yeter.

---

## 12. Reprodüksiyon

```bash
python3 classic_bba_sim.py --seeds 5        # analitik kılavuz (hızlı)
python3 classic_bba_cpp_check.py -w 4       # C++ null doğrulaması (~25 dk)
python3 classic_bba_full.py -w 4            # uçtan uca + sistematikler (~3 saat)
```
Sonuçlar `kmod_drivers/paper_runs_results.json` içinde
(`bba_cpp_check`, `bba_full`, `bba_full_syst`). İlgili günlük:
`separation_bba_testleri.md` (T1–T5 programı ve sonuçları).
