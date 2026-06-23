# Bölüm 4 — Kapalı Yörünge İzlemenin Gözlenebilirlik Sınırı
### (pedagojik anlatım)

> Bu bölüm, makaledeki §4'ün sıfırdan, bir lisans öğrencisinin takip
> edebileceği şekilde anlatılmış halidir. Hiçbir terimi tanımsız bırakmıyoruz;
> her adımda "neden" sorusunu yanıtlıyoruz. Sonunda makaledeki kısa/teknik
> ifadelerin neyi kastettiği netleşmiş olacak.

---

## 4.0 Bu bölümde hangi soruyu soruyoruz?

Önceki bölümlerde şunu gösterdik: **drift monitör**, kalibrasyon anına göre
kuadrupol hizalama hatalarındaki *değişimi* (drift) yaklaşık **6 μm**
duyarlılıkla izleyebiliyor. Güzel. Ama unutmayalım: bizim asıl derdimiz
hizalama hatasının kendisi değil, onun ürettiği **sahte EDM** sinyalidir
(EDM deneyini taklit eden, kaçıklıktan doğan sahte bir dikey spin presesyonu).

O hâlde dürüst soru şu:

> **Drift monitör, sahte EDM'i kontrol etmemize yetecek bilgiyi sağlıyor mu?**

Bu bölüm bu soruyu yanıtlıyor. Cevap iki parçalı:
1. Kapalı yörünge, hizalama hatalarının **hangi türünü** iyi "görür", hangisini
   göremez? (§4.1–4.4)
2. Göremediği tür ile sahte EDM'i süren tür **örtüşüyor mu?** (§4.5)

---

## 4.1 Önce sözlük: quad, kick, kapalı yörünge, tepki matrisi

Halkamız **24 FODO hücresinden** oluşuyor. Her hücrede iki kuadrupol mıknatıs
var:
- **QF** (focusing): yatay düzlemde demeti odaklayan kuadrupol,
- **QD** (defocusing): yatay düzlemde demeti dağıtan kuadrupol.

(Dikey düzlemde roller terstir — bir kuadrupol bir düzlemde odaklarsa diğerinde
dağıtır; FODO'nun "F" ve "D"si buradan gelir.) Toplam: $24\times 2 = 48$
kuadrupol.

**Kuadrupol nedir?** Demete, eksenden uzaklığıyla orantılı bir geri-çağırıcı
kuvvet uygulayan manyetik bir "mercek". İdeal halde her kuadrupol tam
eksende durur.

**Hizalama hatası (misalignment).** Gerçekte her kuadrupol ideal yerinden biraz
kaymıştır. Bir kuadrupolün dikey kayması $dy$, yatay kayması $dx$ ile
gösterilir. 48 kuadrupolün dikey kaymalarını bir araya getirirsek **48 sayılık
bir liste** elde ederiz:
$$
\Delta q = (dy_1,\ dy_2,\ \dots,\ dy_{48}).
$$
Bunu 48-boyutlu bir **vektör** gibi düşünün: her bileşeni bir kuadrupolün ne
kadar kaydığını söyler.

**Kick.** Kaymış bir kuadrupolün içinden geçen demet, eksende değil de kenarda
geçtiği için yana doğru küçük bir **itme (kick)** alır. Bu kickin büyüklüğü
basittir:
$$
\text{kick} \;\propto\; (\text{gradyan}) \times (\text{kayma}).
$$
Burada "gradyan", kuadrupolün ne kadar güçlü odakladığının ölçüsüdür. **Önemli:**
QF ile QD'nin gradyanları **zıt işaretlidir** (biri odaklar `+`, öteki dağıtır
`−`). Bu işaret farkı, bölümün kalbinde yatıyor (§4.3).

**Kapalı yörünge (closed orbit).** Demet halkada milyonlarca tur atar. Eğer her
tur aynı yolu izliyorsa, bu yola "kapalı yörünge" denir. İdeal (hatasız) halkada
kapalı yörünge tam merkezdedir (sapma sıfır). Kuadrupoller kaymışsa, kicklerin
toplam etkisiyle kapalı yörünge merkezden **saparak** yeni, bozulmuş bir kapalı
yörüngeye oturur. BPM'ler (Beam Position Monitor) bu sapmayı 48 noktada ölçer.

**Tepki matrisi $R$.** Kaçıklıklar küçük olduğu sürece, kapalı yörünge sapması
kaçıklıklarla **doğrusal** (lineer) ilişkilidir:
$$
\boxed{\ \mathbf{y} = R\,\Delta q\ }
$$
Burada $\mathbf{y}$ = 48 BPM'deki yörünge sapması (vektör), $\Delta q$ =
kaçıklık vektörü, $R$ = $48\times 48$'lik **tepki matrisi**. $R$ matrisinin
$ij$ elemanı şunu söyler: "*$j$. kuadrupolü 1 birim kaydırırsam, $i$. BPM'deki
yörünge ne kadar değişir?*"

> **Drift monitörün tüm fikri tek satır:** $\mathbf{y}$'yi ölç, $R$'yi bil,
> ters çevir, kaçıklığı geri çat:
> $$\widehat{\Delta q} = R^{-1}\,\mathbf{y}.$$
> (Drift modunda iki *zaman* arasındaki farkı alırız ki BPM ofseti iptal olsun;
> ama matematiksel iskelet budur.)

---

## 4.2 Kapalı yörünge bir "mod-seçici filtre" gibi davranır

Şimdi kilit fiziksel olguyu görelim. Kapalı yörünge, kendisine uygulanan kick
desenine **her harmonikte aynı şiddette tepki vermez**.

**Azimutal harmonik $k$ nedir?** Bir kick desenini halka çevresi boyunca
çizdiğinizi düşünün. Desen çevre boyunca **kaç kez** tekrarlıyorsa, o sayı
harmonik $k$'dır. $k=1$ → çevrede bir tam dalga; $k=24$ → çevrede 24 hızlı
salınım (komşu hücreler arasında işaret değiştiren bir desen).

Bir kuadrupol halkasının kapalı yörüngesi, $k$ harmonikli bir kicke şu
**kazançla** yanıt verir:
$$
G_k = \frac{C}{\,\lvert Q_{\text{eff}}^2 - k^2\rvert\,},
\qquad C\approx 24.8,\quad Q_{\text{eff}}^2\approx 5.03 .
$$
Burada $Q$ halkanın **betatron tune**'udur — demetin bir turda kaç betatron
salınımı yaptığı ($Q\approx 2.7$).

**Bu formül ne diyor?** Payda $\lvert Q^2 - k^2\rvert$, $k$ tune'a yaklaştıkça
küçülür → kazanç **büyür** (rezonans!). $k$ tune'dan uzaklaştıkça payda büyür →
kazanç **küçülür**.

> **Sezgi (salıncak analojisi):** Bir salıncağı doğru ritimde (rezonans
> frekansında) iterseniz büyük genlikle sallanır; yanlış ritimde iterseniz
> neredeyse hiç tepki alamazsınız. Kapalı yörünge de böyle: tune'a yakın ($k
> \approx Q$) "ritimdeki" kick desenlerini güçlü gösterir, tune'dan çok uzak
> ($k \gg Q$) desenleri yutar.

Yani **kapalı yörünge bir mod-seçici (rezonant) filtredir.** (Dikkat: bu klasik
bir "alçak-geçiren" filtre değildir — en güçlü tepki $k=0$'da değil, $k\approx
Q$'dadır. Ama bizim ilgilendiğimiz aralıkta — $k$'nın 2 mi yoksa 24 mü olduğu —
düşük-$k$ güçlü, yüksek-$k$ zayıf gösterilir.)

---

## 4.3 İki tür kaçıklık deseni: simetrik ve antisimetrik

Şimdi en önemli kavramsal adıma geldik: **hangi kaçıklık deseni hangi
harmoniğe düşer?** Bunu görmek için kaçıklıkları hücre-hücre **iki gruba**
ayırıyoruz. Her hücredeki (QF, QD) çiftine bakın:

| Desen türü | QF kayması | QD kayması | Sözle |
|---|---|---|---|
| **Antisimetrik** | $+a$ | $-a$ | çift, **zıt yönde** kayar |
| **Simetrik**     | $+a$ | $+a$ | çift, **aynı yönde** kayar |

Herhangi bir kaçıklık deseni bu iki "saf" türün toplamı olarak yazılabilir.
Böylece 48-boyutlu kaçıklık uzayı, birbirine **dik** iki **alt-uzaya** ayrılır
(her biri 24-boyutlu).

> **"Alt-uzay" korkutmasın.** Burada alt-uzay sadece *belli bir kurala uyan tüm
> desenlerin ailesi* demek. "Simetrik alt-uzay" = hücre içinde QF ve QD'nin hep
> aynı yönde kaydığı bütün desenler. "Antisimetrik alt-uzay" = hep zıt yönde
> kaydığı bütün desenler. Tıpkı bir vektörü "yatay bileşen + dikey bileşen"
> diye ayırmak gibi; burada bileşenler "simetrik kısım + antisimetrik kısım".

**Şimdi kilit nokta — gradyan işareti.** §4.1'de QF (`+`) ve QD (`−`)
gradyanlarının zıt olduğunu söylemiştik. Kick = gradyan × kayma olduğuna göre,
iki desenin ürettiği kickleri yazalım:

**Antisimetrik desen** (kaymalar $+a/-a$, gradyanlar $+/-$):
$$
\text{QF kicki} \propto (+)\cdot(+a) = +,\qquad
\text{QD kicki} \propto (-)\cdot(-a) = + .
$$
İki kick **aynı işarette**! Hücreden hücreye düzgün ilerleyen, **düşük-$k$** bir
kick deseni doğar. Düşük-$k$ → $G_k$ büyük → **kapalı yörüngede güçlü görünür.**

**Simetrik desen** (kaymalar $+a/+a$, gradyanlar $+/-$):
$$
\text{QF kicki} \propto (+)\cdot(+a) = +,\qquad
\text{QD kicki} \propto (-)\cdot(+a) = - .
$$
İki kick **zıt işarette**! Hücre içinde işaret değiştiren, **yüksek-$k$** bir
desen doğar. Yüksek-$k$ → $k \gg Q$ → $G_k$ küçük → **kapalı yörüngede
neredeyse hiç görünmez.**

> **Hangi $k$ tam olarak? (incelik)** Kick = gradyan × kayma ve gradyan QF/QD'de
> işaret değiştirdiğinden, kick dizisi $k_j = g\,(-1)^j\,dq_j$ biçimindedir.
> Buradaki $(-1)^j$ çarpanı halkanın en hızlı titreşimidir (Nyquist, $k=24$) ve
> bir diziyi onunla çarpmak spektrumu **24 kadar kaydırır**. Bu yüzden:
> antisimetrik kaçıklık → kick düşük banda ($k\in[0,12]$) düşer (görünür);
> simetrik kaçıklık → yüksek banda ($k\in[12,24]$) kayar (bastırılır). Yani
> "simetrik = tek bir $k=24$ modu" **değildir**; $k=24$ yalnız hücreden hücreye
> *üniform* simetrik desenin özel durumudur. Simetrik genlik hücreyle değişirse
> kick $k=24-k'$ ($k'=0\dots12$) bandına yayılır — hepsi $k\gg Q$ olduğundan
> hepsi bastırılır.

> **Özetle sezgi:** Simetrik kayma, gradyan alternasyonu yüzünden kendi kendini
> söndüren hızlı titreşen bir kick örüntüsü yaratır; kapalı yörünge buna
> (salıncak analojisi: çok yüksek frekanslı itme) tepki vermez. Antisimetrik
> kayma düzgün, "ritimde" bir örüntü yaratır ve yörüngeyi belirgin biçimde büker.

**Tek bir sayıyla: simetri parametresi $\hat S$.** İki kutuya ayırmak yerine,
bir desenin ne kadar simetrik olduğunu tek bir sayıyla ölçebiliriz. Hücre $c$
için ortalama $s_c=\tfrac12(q_{QF,c}+q_{QD,c})$ ve fark
$d_c=\tfrac12(q_{QF,c}-q_{QD,c})$ tanımlayıp, hücre-içi çarpımları toplarsak
şu hoş özdeşlik çıkar:
$$
\sum_c q_{QF,c}\,q_{QD,c} = \sum_c (s_c^2 - d_c^2) = \|s\|^2 - \|d\|^2,
\qquad
\hat S \equiv \frac{\|s\|^2-\|d\|^2}{\|s\|^2+\|d\|^2}\in[-1,+1].
$$
$\hat S=+1$ tümüyle simetrik, $\hat S=-1$ tümüyle antisimetrik desendir. ($\hat S$
kaba bir özettir; tam gözlenebilirlik resmi §4.4'teki SVD'dir, çünkü simetrik
alt-uzayın kendi içinde de $k'$ değişir.)

**Sonuç:** Antisimetrik kaçıklık → görünür; simetrik kaçıklık → görünmez. Bu,
bölümün omurgası. Ama "görünür/görünmez" kategorik bir ifade; bir sonraki adımda
bunu **sürekli ve nicel** hale getireceğiz.

---

## 4.4 "Ne kadar iyi görünür?" — SVD ve koşulluluk sayısı

§4.3 desenleri iki kutuya ayırdı. Gerçekte bir geçiş bölgesi var, ve onu görmek
için lineer cebrin güzel bir aracını kullanıyoruz: **tekil değer ayrışımı (SVD,
singular value decomposition).**

**SVD nedir, sezgisel olarak?** Her matris $R$, şöyle yazılabilir:
$$
R = U\,\Sigma\,V^{\top}.
$$
Bunun bizim için anlamı şu: $R$ matrisinin "doğal modları" vardır. Her **mod**
bir kaçıklık desenidir ($V$'nin bir sütunu, $V_i$) ve her moda bir **tekil
değer** ($\sigma_i$; İng. *singular value*) eşlik eder. Matematiksel tanım:
$\sigma_i=\sqrt{\lambda_i(R^\top R)}$, yani $\sigma_i^2$, $R^\top R$ matrisinin
özdeğeridir; $R V_i=\sigma_i U_i$. $\sigma_i$, o desenin kapalı yörüngede
**ne kadar güçlü göründüğünün** ölçüsüdür:
- Büyük $\sigma_i$ → desen yörüngede güçlü iz bırakır (kolay görülür).
- Küçük $\sigma_i$ → desen yörüngede zayıf iz bırakır (zor görülür).

**Gürültü neden önemli?** Monitör kaçıklığı geri çatarken $R$'yi ters çeviriyor.
Ters çevirmek, her modu **$\sigma_i$'ye bölmek** demektir. Ölçümde küçük bir
gürültü varsa, küçük $\sigma_i$'li bir modu geri çatarken bu gürültü
$1/\sigma_i$ kat **büyür**. Yani:
$$
\text{bir moddaki gürültü büyütmesi} = \frac{1}{\sigma_i}.
$$
En iyi mod (en büyük $\sigma$) ile en kötü mod (en küçük $\sigma$) arasındaki
oran, matrisin **koşulluluk sayısıdır**:
$$
\kappa(R) = \frac{\sigma_{\max}}{\sigma_{\min}}.
$$

Bizim halkamız için (dikey düzlem) sayılar:
$$
\sigma_{\max} \approx 28.4,\qquad \sigma_{\min}\approx 0.147,\qquad
\kappa(R)\approx 193 .
$$

Şimdi her modu inceleyelim: (i) gürültü büyütmesi $1/\sigma_i$ ve (ii) modun ne
kadarının **simetrik** alt-uzayda olduğu (§4.3 anlamında):

| Mod | $\sigma_i$ | gürültü büyütme $1/\sigma_i$ | modun simetrik içeriği |
|---|---|---|---|
| 0 (en iyi)   | 28.4  | 0.04 | %4  |
| 2            | 10.1  | 0.10 | %6  |
| 10           | 2.0   | 0.51 | %13 |
| 20           | 0.49  | 2.03 | %42 |
| 40           | 0.16  | 6.39 | %91 |
| 47 (en kötü) | 0.147 | 6.82 | **%98** |

**Tablonun okunuşu:** En iyi görülen modlar neredeyse tümüyle **antisimetrik**
(ortalama %4 simetrik). En kötü görülen modlar neredeyse tümüyle **simetrik**
(ortalama %96). Yani §4.3'ün kategorik ifadesi ("simetrik görünmez") SVD'de
**sürekli bir eğim** olarak doğrulanır: *bir mod ne kadar simetrikse, o kadar
gürültülü kestirilir.* En kötü mod, en iyiye göre $\kappa\approx 193$ kat daha
fazla gürültü büyütmesine maruz kalır.

> **Çok önemli incelik (yanlış anlamayı önlemek için):** "$193$ kat" demek
> *bütün* simetrik modlar 193 kat kötü demek **değildir**. Bu, en kötü tek mod
> ile en iyi tek mod arasındaki orandır. Tablodaki $1/\sigma$ sütunu, modlar
> arası kademeli artışı gösterir.

Bu ilişki makaledeki **Şekil 4**'te tek bakışta görülür: yatayda mod indeksi,
solda gürültü büyütmesi $1/\sigma$, sağda simetrik içerik %; iki eğri birlikte
yükselir.

---

## 4.5 Peki bu, sahte EDM ile nasıl bağlanıyor? (ve nerede dururuz)

Şimdi bölümün asıl sorusuna geldik. Burada **çok dikkatli** olacağız, çünkü
kolayca yanlış bir argümana kayılıyor (aşağıda o tuzağı da açıkça gösteriyoruz).

### Sahte EDM hangi kaçıklığa bağlı?

Sahte EDM'i (sahte dikey spin presesyonu $dS_y/dt$) süren **baskın** mekanizma,
yatay kaçıklık $dx$ ile dikey kaçıklık $dy$'nin **çarpımına** orantılı bir
geometrik (Berry) faz katkısıdır:
$$
\frac{dS_y}{dt}\ \Big|_{\text{baskın}}\ \propto\ \sum_j (\text{ağırlık}_j)\,
dx_j\, dy_j .
$$
Bunun iki önemli özelliği var:
1. Bir **çarpımdır** ($dx\cdot dy$) → kaçıklıkta **ikinci dereceden**
   (bilineer/kuadratik). Tek başına $dy$ doğrusal ve küçük; tek başına $dx$
   sıfır. Etki, iki düzlemin **birlikte** kaçık olmasını ister.
2. Kaçıklık genel ölçeği $\sigma$ ile $\propto \sigma^2$ büyür. Bunu spin
   takibiyle bağımsızca doğruladık (ölçülen üs $\approx 2.0$) ve Omarov vd.\
   (PRD 105, 032001) ile mertebe uyumludur.

### ⚠️ Kolay ama YANLIŞ argüman (ve neden yanlış)

İlk bakışta şöyle demek isteyebilirsiniz:

> *"Kapalı yörünge kaçıklıkta lineer (Eş. $\mathbf{y}=R\,\Delta q$), sahte EDM
> ise bilineer (dx·dy). Lineer bir ölçüm bilineer bir şeyi göremez. Bitti."*

**Bu argüman yanlıştır.** Neden? Çünkü monitör ham yörüngeyi değil, ondan geri
çattığı **kaçıklık vektörü $\widehat{\Delta q}$'yu** verir. $R$ tam-rütbeli ve
tersinir olduğundan (sıfır-uzayı yok), gürültüsüz limitte kapalı yörünge
$\Delta q$'yu **tek türlü belirler** — hem bütün $dx$'leri hem bütün $dy$'leri.
Ve $\Delta q$'yu bildiyseniz, ondan **istediğiniz her şeyi**, dx·dy çarpımını
bile, hesaplayabilirsiniz. Yani ilkesel olarak mükemmel bir kapalı-yörünge
ölçümü sahte EDM'i de belirler. Lineerlik tek başına bir engel **değildir**.

> Bu inceliği vurguluyoruz çünkü taslağın bir ara sürümünde tam bu hatayı
> yaptık; doğrusu aşağıdadır.

### ✅ Gerçek engel: koşulluluk (gözlenebilirlik)

Tek gerçek engel §4.4'teki **gürültü büyütmesidir.** Monitör $\Delta q$'yu geri
çatar, ama **simetrik bileşenleri** $\sim 193$ kat gürültüyle çatar (çünkü
simetrik kaçıklık yüksek-$k$ kick üretir, yörünge ona zar zor tepki verir —
§4.3). Yani:
- $\Delta q$'nun **antisimetrik** kısmı: temiz, güvenilir bilinir.
- $\Delta q$'nun **simetrik** kısmı: gürültüde boğulur, güvenilmez.

Şimdi mantık zinciri:
1. Sahte EDM, $\Delta q$'nun (kuadratik) bir fonksiyonudur.
2. **Eğer** sahte EDM ağırlıklı olarak $\Delta q$'nun **simetrik** (kötü
   bilinen) bileşenine bağlıysa → monitör onu kısıtlayamaz.
3. **Eğer** ağırlıklı olarak antisimetrik (iyi bilinen) bileşene bağlıysa →
   monitör onu da kısıtlayabilir.

Hangisi doğru? İşte burası **dürüstçe açık bıraktığımız** nokta. Baskın dx·dy
kanalının ağırlığının özellikle simetrik alt-uzayda mı yoğunlaştığı,
**desen-bağımlı** bir sorudur; spin-takip taramamız bu kanalın rastgele desenden
desene $\sim 36$ kat saçıldığını gösteriyor. Bu ayrışımı tek bir kapalı-yörünge
ölçümünden kesin çıkarmak mümkün değil — ve bu belirsizliğin kendisi,
rapor ettiğimiz sınırın bir parçasıdır.

### Önizleme: spin simülasyonu ne diyor? (ve neden ayrı bir çalışma)

Bu bağı kapalı bir biçimde keşfetmeye başladık (spin izleyicisiyle, 4D kapalı
yörüngede sahte EDM ölçümü). İlk bulgu **öğretici ve ilk sezgiyi tersine
çeviriyor:** σ=10μm kaçıklıkta, sahte EDM **antisimetrik** ($\hat S=-1$)
desenlerde *yüksek* (~3600 nrad/s medyan), **simetrik** ($\hat S=+1$) desenlerde
*düşük* (~66 nrad/s) çıkıyor — yani görünür desen daha çok sahte EDM üretiyor!
Sebebi temiz: antisimetrik kaçıklık büyük kapalı yörünge yapar → demet hem $x$
hem $y$'de büyük salınır → büyük $x_{\rm co}\!\cdot\!y_{\rm co}$ geometrik fazı.

**Ama kritik uyarı:** bu ölçüm **orbit-düzeltmesizdir.** Antisimetriğin ürettiği
büyük sahte EDM, görünür (düzeltilebilir) yörünge salınımından gelir;
korrektörlerle (ki monitör tam bunu görür) söndürülebilir. **Düzeltme-sonrası
indirgenemez taban** ise simetrik kısımdır (görünmez). Yani ham vs
düzeltme-sonrası ayrımı kritiktir ve doğru köprü ancak bu ayrımla kurulur.

Bu, kendi başına bir araştırma (orbit-düzeltme öncesi/sonrası, desen-bağımlılık,
σ² ölçekleme, tilt kanalı) — **bu metodoloji makalesinin kapsamı dışında.** Bu
yüzden burada yalnız *yöntem düzeyindeki* kesin ifadeyle yetiniyoruz:

> Kapalı-yörünge drift monitörü **antisimetrik** hizalama driftini güvenle ve
> ucuza izler; **simetrik** kısmı kötü belirler ($\sim 193$ kat gürültü). Bu
> kör noktanın sahte-EDM bütçesi için önemi makineye bağlı ve ayrı bir
> çalışmanın konusudur; o alt-uzaya doğrudan erişim **farklı bir
> gözlemlenebilir** (karşı-dönen CW/CCW demet ayrımı ya da spin presesyonu)
> gerektirir.

---

## 4.6 Bir endişe: ya düzlemler birbirine karışıyorsa? (quad tilt / skew)

Yukarıda $dy$ ile $dx$'i sanki bağımsızmış gibi konuştuk. Gerçek bir halkada
kuadrupoller sadece kaymaz, biraz da **dönmüş (tilt)** olabilir. Dönmüş bir
kuadrupol bir **skew (eğik) bileşeni** yaratır ve düzlemleri birbirine bağlar:
dikey bir kayma artık biraz **yatay** yörünge de üretir, ya da tersi.

Burada iki kavramı **kesinlikle** karıştırmamak gerekiyor:

| | Ne demek | Hangi mertebe |
|---|---|---|
| **Skew kuplajı** (quad tilt) | $dx \to$ dikey yörünge (ya da $dy\to$ yatay) | **Lineer** — yörünge tek bir kaçıklığa ($dx$ *veya* $dy$) birinci derecede bağlı; ama katsayı tilt $\theta$ ile orantılı ($R_{yx}\propto\theta$, tilt yoksa sıfır) |
| **Sahte EDM $dx\cdot dy$** | iki kaçıklığın **çarpımı** | **Bilineer** — ikinci derece, ikisi *birlikte* |

> **Kuplajın kaynağı tilt'tir, kaçıklık değil.** Sıradan (dönmemiş) bir
> kuadrupolün $dx$ kayması yalnız *yatay* yörünge üretir; çapraz (dikeye) bir
> etki için kuadrupolün **dönmüş** olması gerekir. Tilt'li quad'ın skew bileşeni,
> yatay kaçıklık $dx$'i gördüğünde dikey bir kick verir: $R_{yx}\propto\theta\,dx$
> — yani $dx$'te lineer, ama katsayısı $\theta$'ya orantılı; $\theta=0$ iken tam
> sıfır (nitekim ölçtük). 
>
> **Sık yapılan hata:** "Skew kuplajı x ile y'yi karıştırıyor, demek ki bu
> dx·dy etkisidir." **Hayır.** Skew kuplajı dikey yörüngeyi $dx$'e **lineer**
> bağlar (tek başına $dx$, $\propto\theta$). Sahte EDM ise $dx$ *çarpı* $dy$ —
> ikinci-derece, ikisi birlikte. Biri "çapraz ama birinci derece", öteki
> "çarpım, ikinci derece". (x-y kuplajı nonlineerlik **gerektirmez**; skew
> kuadrupol standart bir *lineer* kuplaj elemanıdır — hareket denklemleri lineer
> kalır, yalnız transfer matrisinin köşegen-dışı blokları dolar.)

**Peki bu skew kuplajı düzlem-ayrık varsayımımızı bozar mı?** Doğrudan test
ettik. "Düzlem-ayrık monitör" derken şunu kastediyoruz: monitör $R_{yy}$
(dikey→dikey) ve $R_{xx}$ (yatay→yatay) bloklarını ayrı ayrı ters çevirir;
skew çapraz bloklarını ($R_{yx}, R_{xy}$) **ihmal eder**. Rastgele quad
tilt'lerle **kuplajlı** tam tepki matrisini izleyiciden kurup, bu düzlem-ayrık
monitörle drift kurtarımı yaptık:

| Quad tilt | x-y kuplajı $\lVert R_{yx}\rVert/\lVert R_{xx}\rVert$ | dikey takip hatası |
|---|---|---|
| 0          | %0.00 | 6.27 μm |
| **0.2 mrad** | **%0.33** | **6.27 μm** |
| 1 mrad     | %1.08 | 6.26 μm |

**Sonuç:** Gerçekçi bir 0.2 mrad tilt'te kuplaj, ana tepkinin yalnız %0.33'ü
kadar; drift takip hatası **hiç değişmiyor**. Yani düzlem-ayrık yaklaşım gerçek
quad tilt altında geçerli — bu bir **sağlamlık kontrolüdür** (tıpkı β-beating
testi gibi) ve §4.5'teki sahte-EDM tartışmasından bağımsızdır. (Skew kuplajı
küçük olduğu için $\Delta q$'yu yine de doğru çatarız; çapraz katkı, drift
modunun ölçtüğü değişime $\sim$kuplaj$\times$drift $\approx 0.1$ μm ekler — 6 μm
tabanının çok altında.)

---

## 4.7 Bölümün özeti (üç cümle)

1. **Kapalı yörünge bir mod-seçici filtredir:** antisimetrik (QF/QD zıt yönde)
   kaçıklığı güçlü görür, simetrik (QF/QD aynı yönde) kaçıklığı neredeyse hiç
   görmez — çünkü simetrik kayma, gradyan alternasyonu yüzünden yüksek-$k$ bir
   kick üretir ve yörünge buna rezonant olarak tepkisizdir (§4.2–4.3).

2. **Bu, SVD'de bir koşulluluk olgusudur:** en kötü gözlenen modlar %96 simetrik
   ve $\kappa\approx 193$ kat gürültü büyütmesine uğrar; yani monitör kaçıklığın
   simetrik kısmını güvenilmez kestirir (§4.4).

3. **Sahte EDM ile bağ ayrı bir çalışmanın konusudur.** Engel "lineerlik"
   değildir (monitör ilkesel olarak $\Delta q$'yu çatıp dx·dy'yi hesaplayabilir);
   engel, kaçıklığın simetrik kısmının kötü bilinmesidir. Spin simülasyonu
   önizlemesi inceliği gösteriyor: ham (orbit-düzeltmesiz) sahte EDM aslında
   *antisimetrik* desende yüksek (büyük yörünge salınımı), simetrikte düşük;
   indirgenemez taban ise düzeltme-sonrası simetrik kısımdır. Bu ham-vs-düzeltme
   ayrımı, desen-bağımlılık ve tilt kanalı kendi başına bir araştırma →
   **bu metodoloji makalesinin dışında.** Bu makale yöntem düzeyinde durur:
   monitör antisimetrik drifti güvenle izler, simetrik kör noktayı tanımlı
   biçimde dış bir gözlemlenebilire bırakır. (Quad tilt'in yarattığı lineer,
   küçük skew kuplajı bunu değiştirmez; §4.6.)

---

*Bu pedagojik anlatım, makaledeki §4'ün (drift\_makalesi.md / .tex) genişletilmiş
halidir. Sayılar ve şekiller orada; buradaki amaç kavramların ilk kez görüldüğünde
de anlaşılmasıdır.*
