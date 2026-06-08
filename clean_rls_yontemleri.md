# Hizalama Hatalarının Orbit'ten Geri Çatımı:
# Bozoki, R-Matris LS ve CLEAN

---

## Önsöz

Bir hızlandırıcı halkasında kuadrupol mıknatıslar ideal eksenlerinden
küçük miktarlarda kaçıktır. Bu kaçıklıklar demetin kapalı yörüngesini
(closed orbit) bozar. Ters problem şudur: **ölçülen orbit'ten, onu
yaratan hizalama hatalarını geri çatabilir miyiz?**

Bu belge üç yöntemi anlatıyor ve hepsini aynı somut senaryoda
çalıştırıp karşılaştırıyor:

- **Bozoki (1989):** klasik, model-bağımsız azimutal harmonik fiti.
- **R-matris en küçük kareler (R-LS):** tepki matrisini kullanan
  tek-adım çözüm.
- **CLEAN:** dominant bileşenleri sırayla soyan iteratif çözüm.

Hedef okuyucu, hızlandırıcı fiziğinin temellerini bilen ama bu
diagnostik yöntemlere yabancı olan kişidir. Her adımda *neden bu yol*
sorusunu da yanıtlamaya çalışacağız.

---

## 1. Problemin Kuruluşu

### 1.1 Kaçık Kuadrupol Orbit'i Nasıl Bozar?

48 kuadrupollü, 24 FODO hücreli örnek bir halka düşünelim. Her hücrede
bir odaklayıcı (QF) ve bir saçtırıcı (QD) kuadrupol vardır.
Kuadrupol $j$, ekseninden $dy_j$ kadar kaçıksa, oradan geçen demet
ek bir dipol-benzeri açısal sapma (kick) alır:

$$\Delta y'_j = -K_j \, dy_j$$

Buradaki $K_j$ kuadrupolün integre odaklama gücüdür. **Çok önemli bir
ayrıntı:** QF'lerde $K_j > 0$, QD'lerde $K_j < 0$. Yani art arda gelen
kuadrupollerde $K_j$ işaret değiştirir — kabaca $K_j \propto (-1)^j$.
Bu, FODO yapısının (alternating gradient) tanımıdır ve birazdan
yöntemleri şekillendirecek.

Tüm quad'ların kick'leri birikip halkada kalıcı bir orbit bozulması
yaratır. BPM $i$'deki sapma klasik Courant–Snyder formülüyle:

$$y_i = \frac{\sqrt{\beta_i}}{2\sin\pi\nu}
        \sum_j \sqrt{\beta_j}\,
        \cos\!\bigl(\pi\nu - |\phi_i - \phi_j|\bigr)\cdot \Delta y'_j$$

Doğrusal olduğu için matris biçiminde yazılır:

$$\boxed{\;y = R\,dy\;}$$

$R \in \mathbb{R}^{48\times48}$ **tepki matrisi** (orbit response matrix).
Sütun $j$, "tek başına $j$ kuadrupolünü 1 birim oynatınca tüm BPM'lerde
oluşan orbit"tir.

### 1.2 Hedef: Harmonik İçerik

48 kaçıklığı tek tek $dy = R^{-1}y$ ile çözmek mümkün ama gürültüye
aşırı duyarlıdır; ayrıca genellikle 48 ayrı sayı değil, kaçıklıkların
**harmonik içeriği** ilgilendirir bizi. Kaçıklıkları birkaç Fourier
harmoniğiyle ifade edip parametre sayısını düşürürüz. Soru şu:
hangi baz fonksiyonları?

---

## 2. Doğru Bazı Seçmek: Neden FODO-Antisimetrik?

### 2.1 İki Aday Baz

**Azimutal baz** (Bozoki'nin kullandığı): halkayı açı $\theta$ ile
dolanıp düz sinüzoidler.

$$\cos(k\theta_j),\ \sin(k\theta_j),\qquad \theta_j = 2\pi j/48$$

**FODO-antisimetrik baz:** her sinüzoide $(-1)^j$ çarpanı ekli.

$$F_{j,k}^{\rm cos} = (-1)^j\cos\!\left(\tfrac{2\pi k\lfloor j/2\rfloor}{N}\right),
\quad
F_{j,k}^{\rm sin} = (-1)^j\sin\!\left(\tfrac{2\pi k\lfloor j/2\rfloor}{N}\right)$$

Hangisi doğru? Bir deneyle görelim.

### 2.2 Kritik Deney: Hangi Kaçıklık Büyük Orbit Üretir?

Aynı normda iki kaçıklık deseni alıp ürettikleri orbit'i ölçelim:

| Kaçıklık deseni | $\|dy\|$ | Ürettiği $\|y\|$ |
|-----------------|----------|------------------|
| Azimutal $\cos(2\theta_j)$ | 4,90 | **24,5** |
| FODO-antisimetrik $(-1)^j\cos(\cdots)$ | 4,90 | **166,9** |

Aynı büyüklükteki kaçıklık, FODO-antisimetrik desende **~7 kat** daha
büyük orbit üretiyor. Halka, azimutal desene neredeyse kör.
Niçin?

### 2.3 Açıklama: Kick'in İşareti Zaten Alternatif

Kick $\Delta y'_j = -K_j\,dy_j$ ve $K_j \propto (-1)^j$. Kick deseni,
kaçıklık ile $K_j$'nin çarpımıdır:

- **Azimutal (pürüzsüz) kaçıklık:** kick
  $\propto (-1)^j\times(\text{pürüzsüz})$ → **dişli/alternatif** desen.
  Baskın Fourier bileşeni yüksek harmonikte (48 üzerinden ~22.),
  betatron tune'undan ($Q_y\approx1{,}73$) çok uzak. Kapalı yörünge
  tune'a yakın harmonikleri rezonant yükseltir ($\propto1/(\nu^2-p^2)$),
  uzakları bastırır → küçücük orbit.

- **FODO-antisimetrik kaçıklık** $(-1)^j\times(\text{pürüzsüz})$: kick
  $\propto (-1)^j(-1)^j(\text{pürüzsüz}) = (\text{pürüzsüz})$. İki
  alternasyon birbirini götürür. Baskın bileşen alçak harmonikte
  (2.), tam tune'un yanında → rezonant yükseltme → büyük orbit.

Doğrulama (kick'in baskın laboratuvar harmoniği):

| Kaçıklık | Kick'in baskın harmoniği | Tune'a göre |
|----------|--------------------------|-------------|
| Azimutal | 22 | uzak → bastırılır |
| FODO-antisim. | 2 | $Q_y\approx1{,}73$'e yakın → yükseltilir |

**Sonuç:** Orbit'ten geri çatabildiğimiz kaçıklıklar FODO-antisimetrik
olanlardır. Bu, dalga boyu ya da hücre sayısıyla ilgili değil; tamamen
kick'teki $(-1)^j$ işaretinin nasıl iptal olduğuyla ilgilidir.
Doğru baz $F_k$'dir. Kaçıklıkları $dy=\sum_k F_k\,a_k$ ile yazıyoruz
($F_k\in\mathbb{R}^{48\times2}$, $a_k=[a_k^c,a_k^s]^\top$).

### 2.4 "Bu Desen Yapay Değil mi?" — SVD Kanıtı

Haklı bir itiraz: simülasyonda biz **bilerek** aynı FODO'daki QF ve
QD'yi zıt yönde oynatıyoruz ($(-1)^j$ çarpanı). O hâlde "kick işaret
iptali" açıklaması, kendi kurduğumuz deseni tarif etmekten ibaret
değil mi? Gerçek bir makinede kaçıklıklar rasgeledir, antisimetrik
değil.

Cevap, tepki matrisinin **tekil değer ayrışımında** (SVD) gizli.
$R = U\Sigma V^\top$ yazalım. Sağ singular vektörler $v_i$, kaçıklık
desenleridir; karşılık gelen $\sigma_i$ ne kadar büyükse o desen
orbit'i o kadar çok üretir (o kadar **gözlenebilir**). Her desen için
"antisimetri skoru" tanımlayalım: komşu QF/QD ne kadar zıt hareket
ediyor (+1 = tam zıt/antisim, −1 = aynı yön/pürüzsüz).

| Mod grubu | tekil değer | antisimetri skoru |
|-----------|-------------|--------------------|
| En gözlenebilir 12 mod | 34,7 … 8,9 | **+0,90** |
| En az gözlenebilir 12 mod | ≈ 0,14 | **−0,90** |

Tekil değer oranı ≈ **250 kat**. Yani:

- R'nin **en gözlenebilir modları kendiliğinden FODO-antisimetriktir** —
  bunu biz dayatmadık, lattice'in alternating-gradient yapısı dayatıyor.
- Pürüzsüz (azimutal) kaçıklıklar R'nin **en az gözlenebilir** modlarıdır;
  orbit'e 250 kat daha az yansırlar, pratikte gürültüye gömülürler.

Dolayısıyla simülasyonda antisimetrik desen enjekte etmemizin sebebi
"k=2'yi vurgulamak" değil — **orbit'ten geri çatılabilen tek alt-uzay
budur.** Gerçek bir makinedeki rasgele kaçıklığın da yalnızca
antisimetrik bileşeni ölçülebilir; pürüzsüz bileşeni neredeyse
görünmezdir. "Kick işaret iptali", bu gözlenebilirlik farkının
**mekanizmasıdır**: antisimetrik kaçıklık → pürüzsüz kick → tune'a
yakın → büyük tekil değer.

---

## 3. Çalışacağımız Senaryo

- **Sinyal:** alçak modlar $k=1,2,3$, her biri **5 μm** — ölçmek
  istediğimiz hizalama hataları.
- **Gürültü/arka plan:** yüksek modlar $k=4,5,6,7,8$, her biri rasgele
  fazlı **~100 μm** — büyük ama ilgilenmediğimiz yapısal kaçıklıklar.

Sinyal/gürültü oranı bu senaryoda zorlu: arka plan modları sinyalin
20 katı genlikte.

Kritik kısıt: **Ölçüm anında hangi modlarda hata olduğunu bilmiyoruz.**
Bu yüzden aday olarak $k=1$'den $k=10$'a tüm modları hesaba katıp,
hangilerinin gerçekten var olduğunu yöntemin çözmesini bekliyoruz.

### 3.1 Mod Yanıt Güçleri ($\|M_k\|=\|R F_k\|$)

| $k$ | $\|M_k\|$ | | $k$ | $\|M_k\|$ |
|-----|-----------|---|-----|-----------|
| 1 | 60,7 | | 6 | 7,8 |
| 2 | **236,0** | | 7 | 5,5 |
| 3 | 61,6 | | 8 | 4,1 |
| 4 | 22,2 | | 9 | 3,2 |
| 5 | 12,1 | | 10 | 2,7 |

$k=2$ en güçlü (laboratuvar harmoniği tam tune'un yanına düşer).
Alçak modlar (1,2,3) yüksek modlardan çok daha güçlü yanıt verir.

### 3.2 Önemli Sonuç: Genlik ≠ Görünürlük

Orbit'e gerçek katkı **genlik × yanıt gücü**dür:

| Mod | Genlik | $\|M_k\|$ | Orbit katkısı |
|-----|--------|-----------|---------------|
| $k=4$ (gürültü) | 100 μm | 22,2 | **0,0022** |
| $k=2$ (sinyal) | 5 μm | 236,0 | 0,0012 |
| $k=6$ (gürültü) | 100 μm | 7,8 | 0,0008 |
| $k=1$ (sinyal) | 5 μm | 60,7 | 0,0003 |

Çarpıcı olan: $k=2$ sinyali, genliği 20 kat küçük olmasına rağmen
güçlü yanıtı sayesinde orbit'e $k=6$ gürültüsünden bile fazla katkı
yapıyor. "Büyük genlikli mod = orbit'e en çok katkı yapan mod" demek
değildir. Ayrıca dikkat: 10 μm yerine 5 μm sinyalde $k=2$'nin katkısı
artık $k=4$'ün altına düştü — bu, CLEAN'in seçim sırasını
değiştirecek (Bölüm 6.4).

---

## 4. Yöntem 1 — Bozoki (1989)

### 4.1 Fikir

Bozoki'nin yöntemi **model-bağımsızdır**: tepki matrisi $R$'yi
kullanmaz. Orbit'i $\beta$ fonksiyonuna göre normalize edip
($\eta_i = y_i/\sqrt{\beta_i}$) doğrudan azimutal harmoniklere fit eder:

$$\eta_i \approx \sum_k \bigl[a_k\cos(k\theta_i) + b_k\sin(k\theta_i)\bigr]$$

NSLS gibi makinelerde, tune'a yakın **tek bir dominant harmonik**
için bu yeterliydi — orbit harmoniği doğrudan okunabiliyordu.

### 4.2 Senaryoda Bozoki

Bozoki'yi diğerleriyle aynı birime (kaçıklık μm) getirmek için tek
farkı baz olacak şekilde kuralım: aynı tepki matrisi $R$, ama azimutal
bazla forward model ($R\cdot G_k\cdot a = y$, $G_k$ azimutal):

| Mod | Gerçek | Azimutal-baz LS | Hata |
|-----|--------|-----------------|------|
| $k=1$ | 5 μm | 31,4 μm | **529 %** |
| $k=2$ | 5 μm | 25,2 μm | **404 %** |
| $k=3$ | 5 μm | 19,6 μm | **293 %** |

Yalnızca bazı değiştirdik (FODO-antisim. yerine azimutal); hata
%300–500'e fırladı. Hata yüzdesi sinyal genliğinden bağımsız (10 μm'de
de aynıydı) — çünkü baz uyuşmazlığı **sistematik**tir, sinyalle orantılı
büyür. Bu, baz seçiminin tek başına ne kadar belirleyici olduğunun en
net kanıtıdır.

---

## 5. Yöntem 2 — R-Matris En Küçük Kareler (R-LS)

### 5.1 Fikir

$k$ harmoniği için $dy^{(k)}=F_k a_k$ koyunca:

$$y = R\,dy = \sum_k \underbrace{R F_k}_{M_k} a_k$$

İlgilendiğimiz modları bir araya koyup tek doğrusal sistem kurarız ve
en küçük kareyle çözeriz:

$$\hat{a} = M^\dagger y,\qquad M=[M_1\,M_2\,M_3],\quad
\hat{A}_k=\|\hat{a}_k\|_2$$

### 5.2 Senaryoda R-LS

Yalnız sinyal modlarını ($k=1,2,3$) modelleyip gürültü modlarını
yok sayarak:

| Mod | Gerçek | R-LS | Hata |
|-----|--------|------|------|
| $k=1$ | 5,0 μm | 4,96 μm | 0,8 % |
| $k=2$ | 5,0 μm | 4,95 μm | 0,9 % |
| $k=3$ | 5,0 μm | 4,93 μm | 1,3 % |

Gürültü modlarını hiç modellemediğimiz halde hata ~%1 —
çünkü FODO-antisimetrik harmonikler R-uzayında neredeyse **ortogonal**;
$k=4..8$ gürültüsünün $k=1,2,3$'e sızıntısı küçük. (10 μm sinyalde hata
~%0,5'ti; sinyal yarıya inince sabit sızıntının göreli payı arttığı için
biraz büyüdü.)

### 5.3 Sınırı

R-LS hızlı ve basit (tek matris çözümü). Ama **hangi modları fit
edeceğini önceden bilmen gerekir.** Yukarıda "sinyal $k=1,2,3$" dedik;
gerçekte bunu bilmeyiz. Yanlış mod kümesi → bozuk kestirim. CLEAN bu
boşluğu doldurur: hangi modların var olduğunu kendisi keşfeder.

---

## 6. Yöntem 3 — CLEAN

### 6.1 Fikir ve Kökeni

CLEAN, 1974'te radyo astronomide (Högbom) bulanık görüntülerden gerçek
kaynakları çıkarmak için geliştirildi. Mantığı:

> Orbit'e en çok katkı yapan modu bul, küçük bir kesrini çıkar,
> kalanı tekrar incele. Yeterince tekrarla.

Hangi modların gerçek olduğunu önceden bilmesi gerekmez.

### 6.2 Algoritma

Başlangıç: artık $r=y$, biriken katsayılar $\text{acc}[k]=0$.

```
döngü t = 1, 2, ..., max_iter:
    (1) her aday k için:  â_k = M_k⁺ r
        azalma_k = ‖r‖² − ‖r − M_k â_k‖²
    (2) en çok azaltanı seç:  k* = argmax_k azalma_k
    (3) küçük payını çıkar:   r ← r − g · M_{k*} â_{k*}
    (4) biriktir:             acc[k*] ← acc[k*] + g · â_{k*}
    (5) dur: ‖r‖/‖r₀‖ < ε  ya da  max_iter
```

$g$ **döngü kazancı** (loop gain), tipik $0{,}1$–$0{,}3$; biz $0{,}2$.

### 6.3 Neden Tüm Modu Birden Çıkarmıyoruz? ($g<1$)

$g=1$ olsa seçilen modun tamamını çıkarırdık. Ama modlar tam ortogonal
değil; bir modu tam çıkarmak sonraki adımda başka modun seçimini
bozabilir. Küçük pay, algoritmaya aynı moda dönüp tahmini kademeli
düzeltme şansı verir.

### 6.4 Somut Çalışma: İlk Sekiz İterasyon

| iter | seçilen $k$ | kalan $\|r\|/\|r_0\|$ |
|------|-------------|------------------------|
| 1 | **4** | 0,897 |
| 2 | **4** | 0,825 |
| 3 | **4** | 0,775 |
| 4 | 5 | 0,736 |
| 5 | 2 | 0,697 |
| 6 | 4 | 0,659 |
| 7 | 5 | 0,630 |
| 8 | 2 | 0,601 |

**Dikkat:** CLEAN "$k=10$'dan $k=1$'e" sırayla inmiyor; **orbit'e en
çok katkı yapan modu** seçiyor. Üstelik bu sıra senaryoya bağlı:
sinyal 10 μm iken ilk seçilen mod $k=2$ idi; ama **5 μm'e indirince
ilk seçilen $k=4$ oldu** (Bölüm 3.2'deki katkı tablosu: $k=4$ gürültüsü
0,0022 > $k=2$ sinyali 0,0012). $k=2$ ancak 5. iterasyonda gelebiliyor.
"Genlik ≠ görünürlük" ilkesinin canlı bir örneği — sinyali yarıya
indirince görünürlük sırası değişti.

### 6.5 Sonuç (316 iterasyonda yakınsar)

| $k$ | gerçek | CLEAN | hata | tür |
|-----|--------|-------|------|-----|
| 1 | 5,0 μm | 5,00 μm | 0,0 % | sinyal |
| 2 | 5,0 μm | 5,00 μm | 0,0 % | sinyal |
| 3 | 5,0 μm | 5,00 μm | 0,0 % | sinyal |
| 4 | 100,0 μm | 100,0 μm | 0,0 % | gürültü |
| 5 | 100,0 μm | 100,0 μm | 0,0 % | gürültü |
| 6 | 100,0 μm | 100,0 μm | 0,0 % | gürültü |
| 7 | 100,0 μm | 100,0 μm | 0,0 % | gürültü |
| 8 | 100,0 μm | 100,0 μm | 0,0 % | gürültü |
| 9 | — | 0,0 μm | — | yok |
| 10 | — | 0,0 μm | — | yok |

CLEAN tüm modları doğru çatıyor — sinyal, gürültü ve var olmayan
$k=9,10$ dahil. Üstelik **hangi modların gerçek olduğunu önceden
söylemeden**; algoritma keşfetti.

---

## 7. Üç Yöntemin Karşılaştırması

Aynı senaryo (sinyal $k=1,2,3$ @ 5 μm), $k=1$ kaçıklığını geri çatma:

| Yöntem | Baz | $R$ kullanır mı? | $k=1$ hatası |
|--------|-----|------------------|--------------|
| Bozoki | azimutal | hayır (model-free) | ~529 % |
| R-matris LS | FODO-antisim. | evet | 0,8 % |
| CLEAN | FODO-antisim. | evet | 0,0 % |

| Özellik | Bozoki | R-LS | CLEAN |
|---------|--------|------|-------|
| Çözüm tarzı | tek-adım fit | tek-adım | iteratif |
| Doğru baz mı? | **hayır** | evet | evet |
| Mod kümesi | sabit | önceden verilir | keşfedilir |
| Hız | hızlı | hızlı | yavaş (yüzlerce iter) |
| Tipik hata | %300–500 | <%1 | ~%0 |

### Okunuşu

- **Bozoki** tarihsel önem taşır ve tune'a yakın **tek** dominant
  harmonik için tasarlandı. Çoklu-harmonik, FODO-antisimetrik yapıda
  azimutal bazı yüzünden başarısız: harmonikler birbirine karışır.

- **R-LS**, doğru bazı tepki matrisiyle birleştirir; hangi modları
  çözeceğini *sen söylersin*, tek adımda yüksek doğrulukla çözer.

- **CLEAN**, aynı doğru bazı iteratif kullanır; hangi modların var
  olduğunu *kendisi keşfeder*. Mod kümesini bilmediğimiz ya da gürültü
  sızıntısının önemli olduğu durumlarda en güvenli seçenektir.

---

## 8. Özet

Üç yöntem de aynı ters problemi çözmeye çalışır: orbit $y$'den
kaçıklık $dy$'yi geri çatmak. Belirleyici iki unsur:

1. **Tepki matrisi** $R$: kaçıklıkları orbit'e bağlayan doğrusal harita.
   Bozoki bunu kullanmaz (ve bu onun zayıflığıdır); R-LS ve CLEAN kullanır.

2. **FODO-antisimetrik baz** $F_k$: kick'teki $(-1)^j$ işaret yapısını
   nötrleyerek, halkanın gerçekten duyarlı olduğu kaçıklık modlarını
   yakalar. Azimutal baz (Bozoki) bunu kaçırır ve %300+ hata üretir.

Doğru baz + tepki matrisi seçildiğinde hem R-LS hem CLEAN hizalama
hatalarını yüksek doğrulukla geri çatar. İkisi arasında tercih,
mod kümesini önceden bilip bilmediğimize bağlıdır: biliyorsak R-LS'in
hızı, bilmiyorsak CLEAN'in keşif gücü öne çıkar.

---

*Örnek halka: 48 kuadrupol, 24 FODO hücresi, $Q_y \approx 1{,}73$.
Tüm sayısal sonuçlar `R_dy_1.npy` tepki matrisi ve
`fourier_reconstruct.py` (FODO baz + CLEAN) ile üretilmiştir.*
