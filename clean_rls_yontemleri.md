# Kuadrupol Hizalama Hatalarının Orbit Tabanlı Ölçümü:
# R-Matris En Küçük Kareler ve CLEAN Yöntemleri

---

## Önsöz

Bu belge, pEDM halkasında kuadrupol hizalama hatalarını orbit ölçümlerinden
geri çatmak için geliştirilen iki yöntemi — R-matris en küçük kareler
(R-LS) ve CLEAN — anlatıyor. Fizik altyapısı olan ama bu yöntemlere
yabancı olan bir okuyucu hedefleniyor: formüller türetimsiz sunulmayacak,
her adımda *neden bu yolu seçtik* sorusunun cevabı verilecek.

Yapının tamamı, temel bir gözlemden inşa ediliyor:

> **Bir kuadrupol kaçık durduğunda demet onu tam merkezden göremez;
> bunun yarattığı orbit sapması, geometrinin bize verdiği en değerli
> ölçüm verisidir.**

---

## 1. Problem: Ne Ölçmek İstiyoruz?

pEDM halkası 48 kuadrupol mıknatıs içeriyor; bunlar 24 FODO hücresine
birer odaklayıcı (QF) ve birer saçtırıcı (QD) olarak dizilmiş.
İdeal durumda her kuadrupol manyetik merkezi üzerinden geçer;
gerçekte her biri dikeyde $dy_j$ kadar kaçıktır.

Mekanik anket bu kaçıklıkları ~100 μm hassasiyetle ölçebilir.
Hedefimiz ise **~10 μm mertebesinde** bir Fourier harmoniğini
(k=2, halka boyunca iki tam sinüs dalgası) demet ölçümlerinden
geri çatmak. Mekanik yeterli değil; demet tabanlı yönteme ihtiyaç var.

### 1.1 Kaçık Kuadropolün Orbit Üzerindeki Etkisi

Gradyent alanlı bir kuadrupol, kendi ekseninden $dy_j$ uzaklıkta geçen
bir demet parçacığına $\Delta y'_j = -K_j \, dy_j$ büyüklüğünde
dikey bir açısal sapma (kick) uygular; burada

$$K_j = \frac{1}{B\rho}\,\frac{\partial B_x}{\partial y}\,L_q$$

integre kick gücüdür ($B\rho$ rijidite, $L_q$ uzunluk).
QF'de $K_j > 0$, QD'de $K_j < 0$'dır — bu, FODO yapısının temel
işaret dönüşümüdür.

Tüm 48 quad'ın bu küçük kick'leri birikir ve halkada kalıcı bir
orbit sapması (kapalı yörünge bozulması, COD) yaratır.
BPM $i$'deki sapma Courant–Snyder formülüyle verilir:

$$y_i^{\rm CO} =
  \frac{\sqrt{\beta_i}}{2\sin\pi\nu}
  \sum_j \sqrt{\beta_j}\,
  \cos\!\bigl(\pi\nu - |\phi_i - \phi_j|\bigr)\cdot \Delta y'_j$$

Bu denklem lineerdir — tüm $dy_j$'ler $y_i$'ye doğrusal katkı verir.
Matris formuna geçersek:

$$\boxed{\;y \;=\; R\,dy\;}$$

$R \in \mathbb{R}^{48\times48}$ **tepki matrisi** (orbit response matrix,
ORM), $R_{ij} = K_j\sqrt{\beta_i\beta_j}\,\cos(\pi\nu-|\phi_i-\phi_j|)/(2\sin\pi\nu)$.

### 1.2 Neden Doğrudan $R^{-1}y$ Yapmıyoruz?

İki nedeni var.

**Birincisi, BPM sistematik ofsetleri.** Her BPM'in elektronik sıfır
noktası ideal konumdan kaçıktır; bunu $b_i$ ile gösterelim.
Gerçek ölçüm:

$$y_{\rm ölç} = R\,dy + b + \varepsilon$$

$\sigma_b \approx 100\,\mu\text{m}$ iken hedef sinyal sadece 10 μm'dir.
$R^{-1}y_{\rm ölç}$ hesaplarsak çözümü $b$ baskılar;
asıl $dy$'yi göremeyiz.

**İkincisi, 48 quad'ı tek tek kestirmek istiyorsak çok
parametre var.** Ölçüm gürültüsü ve koşulsallık (condition number)
sorunları doğrudan tersini almayı güvensiz kılar.

Bu iki sorunu aşmak için iki farklı fikir kullanıyoruz: *parametreyi
azalt* (Fourier harmonik modeli) ve *ölçümü değiştir* (k-modülasyon farkı).

---

## 2. Fourier Modeli: 48 Parametreyi Kaça İndir?

### 2.1 Fikir

Fizikte genellikle bilinmeyeni *parçalara ayırmak* yerine
*uygun bir baza açmak* problemi basitleştirir.
Kuadrupol hizalama hatalarını şöyle yazalım:

$$dy_j = \sum_k \bigl[\,a_k^c\,F_{j,k}^{\rm cos}
               + a_k^s\,F_{j,k}^{\rm sin}\bigr]$$

$F_{j,k}$ baz fonksiyonları, $a_k$ ise bilinmeyen katsayılar.
Birkaç harmoniği modellemek yeteriyse (örn. k = 2, 4, 6, 8),
48 bilinmeyenden 8'e iniyoruz. Hem istatistiksel güç artar,
hem de BPM offset sorunuyla daha kolay başa çıkılır.

Ama hangi baz fonksiyonları?

### 2.2 Azimutal Baz ve FODO Yapısının Çelişkisi

Sezgisel tercih, "halkayı $\theta$ ile dolanıp
$\cos(k\theta)$, $\sin(k\theta)$ yazmak" olurdu.
Bu, Bozoki (1989)'nun BNL-NSLS makinesinde kullandığı yaklaşımdır
ve o makinede işe yaramıştır.

Neden işe yaradığını anlamak için NSLS ile pEDM arasındaki farka
bakmak gerekir.

NSLS'te hedef harmonik k=1'di: halkayı bir kez dolanan, yani
dalga boyu $\lambda = 2\pi R \approx 100\,\text{m}$ olan bir sinüs.
Makinenin FODO hücresi uzunluğu ~3 m'dir. Yani hücre boyu,
k=1 dalgasının yüzde üçü — FODO'nun QF/QD işaret dönüşümü,
bu uzun dalga üzerinde küçük bir titreşim bırakır.
Azimutal $\cos(\theta)$ bu titreşimi görmezden gelebilir,
k=1'i makul biçimde tanımlar.

pEDM'de ise 24 FODO hücresi var ve hedef k=2.
Bir k=2 harmoniğinin yarım periyodu $\lambda/2 = \pi R / 2 \approx 75\,\text{m}$;
bu, tam olarak 12 FODO hücresi uzunluğu.
Yani **k=2 dalgasının bir tam periyodu, iki FODO hücresini kapsıyor.**
QF ve QD'nin işaret farkı artık küçük bir pertürbasyon değil;
k=2 modunun yapısını tanımlayan temel özellik.

Bu farkı görmek için bir örnek verelim.
Azimutal k=2 baz vektörü:

$$f_j^{\rm azim} = \cos\!\left(\frac{2\pi \cdot 2 \cdot j}{48}\right)
= \cos\!\left(\frac{\pi j}{12}\right)$$

Bu vektörde quad j=0 ve j=1 (aynı hücrenin QF ve QD'si)
farklı değerler alır: $f_0 = 1$, $f_1 = \cos(\pi/12) \approx 0{,}97$.
Aralarında *neredeyse aynı işaret* var; hücreler farklılaşmıyor.

FODO-antisimetrik k=2 baz vektörü:

$$F_{j,2}^{\rm cos} = (-1)^j\,\cos\!\left(\frac{2\pi \cdot 2 \cdot \lfloor j/2\rfloor}{24}\right)$$

Burada j=0 için $(-1)^0 = +1$, j=1 için $(-1)^1 = -1$:
**QF ve QD ters işaret taşıyor.** Bu, FODO gradyanının
($K_{\rm QF} = -K_{\rm QD}$) yarattığı fiziksel gerçeği yansıtıyor.

Sonuç: eğer gerçek hizalama hatası FODO-antisimetrik yapıda ise,
onu azimutal baza sığdırmaya çalışmak sistematik bir önyargı üretir.
Simülasyonda bunu ölçtük: gürültüsüz ve parazitsiz ortamda bile
azimutal tabanlı Bozoki yöntemi k=2 için ~%312 hata veriyor.

### 2.3 FODO-Antisimetrik Fourier Bazı

Doğru baz şöyle tanımlanıyor:

$$\boxed{
F_{j,k}^{\rm cos} = (-1)^j\,\cos\!\!\left(\frac{2\pi k\,\lfloor j/2\rfloor}{N}\right),
\qquad
F_{j,k}^{\rm sin} = (-1)^j\,\sin\!\!\left(\frac{2\pi k\,\lfloor j/2\rfloor}{N}\right)
}$$

$j = 0,\ldots,47$ quad indeksi, $N = 24$ FODO hücresi sayısı,
$\lfloor j/2 \rfloor$ quad'ın hangi hücreye ait olduğu.

**Fiziksel okunuşu:** Her hücre bir "harmonik genlik" alıyor
($\cos$ veya $\sin$ ile belirlenen), ama o hücre içindeki QF ve QD
ters işaretli. Bu tam olarak FODO'nun alternating gradient yapısını
yansıtıyor.

---

## 3. R-Matris En Küçük Kareler (R-LS)

### 3.1 Ölçüm Operatörü

k harmoniği için $dy^{(k)} = F_k\,a_k$ yazarsak
($F_k \in \mathbb{R}^{48\times2}$, $a_k = [a_c,\,a_s]^\top$)
ve bunu tepki denklemine koyarsak:

$$y = R\,F_k\,a_k + \cdots = M_k\,a_k + \cdots$$

$M_k = R\,F_k \in \mathbb{R}^{48\times2}$ **ölçüm operatörü**dür:
"k harmoniğindeki bir hizalama hatasının ölçüm uzayındaki
izini" temsil eder.

### 3.2 Geometrik Yorum

$M_k$'yı bir mercek gibi düşünebiliriz: $dy^{(k)}$ uzayı
(2-boyutlu, $a_c$ ve $a_s$ ile parametrelenmiş) bu mercekten
geçince 48-boyutlu ölçüm uzayında 2-boyutlu bir *iz* bırakıyor.
R-LS, ölçülen $y$'yi bu ize en yakın noktaya projekte ediyor:

$$\hat{a}_k = M_k^\dagger\,y
= (M_k^\top M_k)^{-1}\,M_k^\top\,y$$

Genlik ve faz:
$$\hat{A}_k = \|\hat{a}_k\|_2,
\qquad
\hat{\phi}_k = \arctan\!\left(\hat{a}_s / \hat{a}_c\right)$$

### 3.3 Ölçüm Gücü: Harmonikler Eşit Değil

Tepki matrisi $R$ her harmoniği farklı kuvvetle yükseltir.
$\|M_k\|_F$ normu bize k harmoniğinin ölçüm uzayındaki "büyüklüğünü"
veriyor:

| k | $\|M_k\|_F$ |
|---|-------------|
| 2 | 236,0 |
| 4 | 22,2 |
| 6 | 7,8 |
| 8 | 4,1 |

k=2 sinyali, k=8'e göre 60 kat daha güçlü yanıt veriyor.
Bu asimetri CLEAN için de önemli olacak.

### 3.4 R-Uzayında Ortonormallik ve Sabit Kontaminasyon

Ölçüm uzayında iki harmonik birbirine ne kadar benzer?
Bunu şu normalize iç çarpımla ölçüyoruz:

$$\rho(k, m) = \frac{\langle M_k,\,M_m\rangle_F}{\|M_k\|_F\,\|M_m\|_F}
= \frac{\operatorname{tr}(M_k^\top M_m)}{\|M_k\|_F\,\|M_m\|_F}$$

Bu, iki matrisi birer 96-boyutlu vektör gibi düşünüp aralarındaki
açının kosinüsü. Eğer $\rho = 0$ ise, k ve m harmonikleri ölçüm
uzayında ortogonal — birinin katsayısı diğerini etkilemiyor.
Eğer $\rho = 1$ ise tam örtüşme, ayırt etmek imkânsız.

Sayısal sonuçlar:

| Harmonik çifti | $\rho$ |
|----------------|--------|
| k=2 vs k=4 | −0,0073 |
| k=2 vs k=6 | −0,0102 |
| k=2 vs k=8 | −0,0122 |

Bu değerler çok küçük — harmonikler R-uzayında neredeyse ortogonal.
Bu FODO-antisimetrik bazın güzel bir özelliği: doğru baz seçildiğinde
harmonikler iyi ayrışıyor.

Ama "neredeyse" kelimesi önemli. $\rho = -0{,}012$ sıfır değil.
Bu küçük örtüşme, arka plan harmoniklerinin k=2 kesitimine sızdığı
**sabit kontaminasyona** (sistematik önyargıya) yol açıyor:

$$\text{bias}_{k=2} = M_2^\dagger \sum_{m \neq 2} M_m\,a_m^{\rm true}$$

params.json senaryosu (k=4@300 μm, k=6@300 μm, k=8@200 μm):

| Kaynak | k=2'ye sızdırdığı |
|--------|-------------------|
| k=4 @ 300 μm | 0,191 μm |
| k=6 @ 300 μm | 0,119 μm |
| k=8 @ 200 μm | 0,071 μm |
| **Toplam** | **0,381 μm** |

Bu sabit kontaminasyon, k=2 sinyali ne kadar küçük olursa o kadar
baskın: $A_2 = 10\,\mu$m'de %3,8 hata iken, $A_2 = 3\,\mu$m'de %13
olur. R-LS bu tabanı tek başına aşamaz.

---

## 4. CLEAN: İteratif Harmonik Soyma

### 4.1 Radyo Astronomi'den Gelen Fikir

CLEAN algoritması, 1974'te Jan Högbom'un radyo astronomi
görüntülemesi için geliştirdiği bir yöntemdir. Problem benzer:
teleskop, gerçek gökyüzü görüntüsünü "bulanık" bir biçimde ölçer;
dominant kaynakların izini adım adım çıkararak zayıf kaynakları
görünür kılmak gerekir.

Bizdeki amaç: ölçüm sinyalinden dominant harmoniklerin (k=6, k=4)
izini adım adım ayıklamak, zayıf hedef harmoniğin (k=2) temiz bir
artıkta kestirilebilmesini sağlamak.

### 4.2 Neden Tek Adımlı Fit Yetmiyor?

Yalnız k=2'yi fit edersek, $M_2^\dagger y$ hesaplıyoruz; bu,
k=6 @ 300 μm'den gelen sinyalin k=2 "görüntüsünü" de kapsamak
zorunda — işte sabit kontaminasyon budur.

Alternatif olarak tüm harmonikleri (k=2,4,6,8) aynı anda fit edebiliriz.
Bu da çalışır, ama pratikte k=6 ve k=8 sinyalleri çok zayıf
kesitimlere ($\|M_6\| = 7{,}8$, $\|M_8\| = 4{,}1$) düşüyor;
tek adımlı çözüm bu zayıf sütunlara güvenmek zorunda,
gürültüye karşı hassas hale geliyor.

CLEAN ise *büyük olanı önce, küçük olanı sonra* felsefesini izler.

### 4.3 Algoritmanın Adım Adım Okunuşu

Başlangıç olarak $r = y$ (artık = tam ölçüm).
Biriken katsayılar $\text{acc}[k] = 0$ her $k$ için.

```
döngü t = 1, 2, ..., max_iter:

    (1) Her aday k için artığı en çok azaltacak katsayıyı bul:
          â_k = M_k⁺ r          (en küçük kareler adımı)

    (2) En iyi harmoniği seç:
          k* = argmax_k [ ‖r‖² − ‖r − M_k â_k‖² ]

    (3) Artıktan bu harmoniğin küçük bir payını çıkar:
          r ← r − g · M_{k*} â_{k*}

    (4) Biriktir:
          acc[k*] ← acc[k*] + g · â_{k*}

    (5) Dur: ‖r‖/‖r₀‖ < ε ise ya da max_iter'e ulaşıldıysa
```

Burada $g = 0{,}2$ **döngü kazancı** (loop gain).

### 4.4 Döngü Kazancı Neden 1 Değil?

$g = 1$ olsaydı, her adımda seçilen harmoniğin *tamamını* çıkarırdık.
Ama harmonikler tam ortogonal değil ($\rho \neq 0$); bir harmoniği
tam çıkarmak, bir sonraki adımda başka bir harmoniği seçimi
bozabilir. Küçük bir pay ($g = 0{,}2$) çıkarmak, algoritmanın
birkaç turda aynı harmoniğe dönmesine izin verir ve bu salınım
zamanla hepsinin doğru tahminini ortaya çıkarır.

Radyo astronomi deneyimi $g \in [0{,}1, 0{,}3]$ aralığının
iyi çalıştığını gösteriyor; biz $g = 0{,}2$ kullanıyoruz.

### 4.5 Seçim Sırası: Güçlüden Zayıfa

İlk yinelemede hangi harmonik seçilir?
Artık başlangıçta tüm harmonikleri barındırıyor.
k=6 @ 300 μm, $\|M_6\,a_6\|$ açısından büyük bir sinyal,
ama $\|M_6\| = 7{,}8$ küçük. k=4 @ 300 μm, $\|M_4\| = 22{,}2$
ile çok daha güçlü.

Algoritma artığı en çok azaltan harmoniği seçiyor.
Gerçek seçim sırasını hesaplarsak:

```
iter  1 → k=6  (artık oran: 0,68)
iter  2 → k=4  (artık oran: 0,42)
iter  5 → k=8  (artık oran: 0,21)
iter 12 → k=2  (artık oran: 0,08)
iter 40 → ...  (artık oran: 0,004)  → yakınsama
```

k=6 ve k=4 temizlendikten sonra k=2, artık içinde görünür hale gelir
ve seçilmeye başlar. Sonuçta k=2 katsayısının tahmini, arka plan
harmoniklerinin sızdırdığı 0,38 μm'den belirgin biçimde küçük
bir hataya ulaşır.

---

## 5. k-Modülasyon: BPM Ofseti Nasıl İptal Olur?

### 5.1 Temel Gözlem

BPM ofseti $b$ hem R-LS hem CLEAN'in önündeki en büyük engel.
$\sigma_b \approx 100\,\mu\text{m}$ iken hedef sinyal 10 μm —
ofset sinyalden 10 kat büyük.

Tek ölçümde $y = R\,dy + b + \varepsilon$; $b$ ve $R\,dy$'yi
birbirinden ayırmak mümkün değil. Ama şunu fark edelim:
BPM'nin elektroniği fiziksel konumundan gelmiyor, ölçüm
referans noktasından geliyor. Bu referans, gradient değişince
değişmiyor. Yani **iki farklı gradient ayarında ölçüm yaparsak**:

$$y_1 = R_1\,dy + b + \varepsilon_1$$
$$y_2 = R_2\,dy + b + \varepsilon_2$$

$$\Delta y = y_2 - y_1 = (R_2 - R_1)\,dy + (\varepsilon_2 - \varepsilon_1)$$

$b$ tamamen iptal oldu.

### 5.2 Tüm Quadlara Tek Tip Modülasyon

$g_1 \to g_1(1+\varepsilon)$ şeklinde tüm quadlara %2 gradient
değişimi uygularsak:

$$R_2 \approx (1 + \varepsilon)\,R_1
\quad\Rightarrow\quad
\Delta R = R_2 - R_1 \approx \varepsilon\,R_1$$

$R_1$ tam rankı 48 olduğundan $\Delta R = \varepsilon R_1$ de tam rankı
koruyor. CLEAN ve R-LS ölçüm matrisleri $M_k^{\rm kmod} = \Delta R \cdot F_k$
hesaplanabilir, harmonik ayrışma bozulmuyor.

### 5.3 Tek-Quad Kmod Neden Başarısız?

Eğer yalnız 1–2 quad modüle edilirse, $\Delta R = R_2 - R_1$ yalnız
o quad'ların sütunlarından oluşuyor. Bu matrisin rankı sadece 2
(ya da modüle edilen quad sayısı kadar). Tekil değerlere bakarsak:

$$\sigma_1 = 3{,}73,\quad \sigma_2 = 3{,}31,\quad
\sigma_3 = 0{,}021,\quad \sigma_4 = 0{,}015,\,\ldots$$

Üçüncü tekil değer, ikincinin yüzde birinden küçük — etkin rank 2.
Bu 2-boyutlu bir uzaydan 48 harmoniği ayrıştırmak imkânsız.
Bu durumda $\Delta R$ yerine $\varepsilon\,R_1$ kullanmak gerekir:
fiziksel karşılığı uniform modülasyon, matematiksel avantajı
tam rank korunması.

### 5.4 CLEAN + kmod Birleşimi

$r \leftarrow \Delta y$, $M_k \leftarrow \varepsilon\,R_1\,F_k$
başlangıcıyla standart CLEAN çalıştırılır.
BPM ofsetinden kaynaklanan her türlü sistematik hata iptal olmuştur;
yalnız gürültü $\varepsilon_{\rm diff} = \varepsilon_2 - \varepsilon_1$
(tek ölçüm gürültüsünden $\sqrt{2}$ kat büyük) kalmaktadır.

---

## 6. Dört Yöntemi Karşılaştırmak

### 6.1 Yöntemlerin Özellik Tablosu

| Yöntem | Baz | BPM ofseti | Sabit kontam. |
|--------|-----|------------|---------------|
| Bozoki LS | Azimutal cos/sin | var | ~%312 intrinsik |
| R-matris LS | FODO-antisim. | var | 0,38 μm |
| CLEAN (kmodsuz) | FODO-antisim. | var | azaltılır |
| CLEAN + kmod | FODO-antisim. | **iptal** | azaltılır |

### 6.2 $A_2 = 10\,\mu$m'de Beklenen Hata

Sayısal benzetimden (params.json arka planı, 80 Monte Carlo):

| Yöntem | Ortalama hata |
|--------|---------------|
| Bozoki LS | ~%305 |
| R-matris LS | ~%4 |
| CLEAN (kmodsuz) | ~%3,5 |
| CLEAN + kmod | ~%0 |

### 6.3 Hangi Yöntem Ne Zaman?

**Bozoki yöntemi** tarihsel önem taşır; uzun dalga boylu
(küçük k) harmoniklerde ve büyük SNR'de makul çalışır.
pEDM'in FODO yapısı ve k=2 hedefinde yetersiz.

**R-matris LS** tek ölçümle hızlı sonuç. BPM ofseti sorunuyla
başa çıkılamıyor ama FODO-antisimetrik baz sayesinde Bozoki'nin
önyargısından kurtarılmış. Sabit kontaminasyon tabanı: 0,38 μm.

**CLEAN (kmodsuz)** arka plan harmonikleri dominant olduğunda
R-LS'ye küçük bir avantaj sağlar; $A_2 < 5\,\mu$m bölgesinde
kontaminasyon tabanını kısmen giderir.

**CLEAN + kmod** maksimum doğruluk için. BPM ofseti tamamen
iptal; arka plan harmonikleri iteratif ayıklanmış.
Dezavantajı: iki orbit ölçümü gerekiyor ve fark sinyalinin
gürültüsü $\sqrt{2}$ kat artar.

### 6.4 Kontaminasyon Tabanı ve Hata Sınırı

R-LS'nin 0,38 μm sabit kontaminasyonu, $A_2$'nin göreli
hatasına şöyle yansır:

| $A_2$ | Beklenen R-LS hatası |
|--------|----------------------|
| 100 μm | %0,38 |
| 30 μm | %1,3 |
| 10 μm | %3,8 |
| 3 μm | %12,7 |
| 1 μm | %38 |

%10 altında kalmak için R-LS ile $A_2 \gtrsim 4\,\mu$m gerekiyor.
CLEAN bu sınırı ~2–3 kat düşürebilir; kmod ise BPM
ofsetini tamamen kaldırdığı için bu kısıtlamayı sadece
gürültü sınırına taşır.

---

## 7. Özet

İki yöntem de aynı temel fikri paylaşıyor: FODO-antisimetrik Fourier
baz fonksiyonları, 48 kuadropolün hizalama hatasını fiziğe uygun
biçimde parametreliyor ve tepki matrisi $R$ aracılığıyla bu
parametrelerden ölçüm uzayına doğrusal bir dönüşüm kuruyor.

**R-matris LS**, bu doğrusal sistemi tek adımda çözer.
Hızlı, yorumlanması kolay, ancak arka plandan gelen sabit
kontaminasyon tabanı var.

**CLEAN**, aynı doğrusal sistemi iteratif biçimde çözer.
Her adımda dominant harmoniği kısmen ayıklayarak zayıf
hedef harmoniği daha temiz bir artıkta ölçer.
Contaminasyon tabanını kısmen giderir; kmod ile birleşince
BPM ofseti de iptal olur.

Her iki yöntemin başarısı, doğru baza dayanıyor. Baz yanlış seçilirse
— örneğin FODO'nun işaret yapısını görmezden gelen azimutal harmonikler —
hiçbir istatistiksel teknik sistematik önyargıyı gidemez.

---

*Simülasyon parametreleri: $N_Q = 48$, $n_{\rm FODO} = 24$,
$R_0 = 95{,}49\,\text{m}$, $g_1 = 0{,}21\,\text{T/m}$,
$L_q = 0{,}4\,\text{m}$.
Referans dosyalar: `params.json`, `fourier_reconstruct.py`,
`R_dy_1.npy`.*
