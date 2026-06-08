# CLEAN ve R-Matris En Küçük Kareler Yöntemleri

> pEDM halkasında kuadrupol hizalama hatalarının Fourier harmonik
> analizi ile ölçülmesi için geliştirilen iki yöntemin ilkeli karşılaştırması.

---

## İçindekiler

1. [Problem Tanımı](#1-problem-tanımı)
2. [Ölçüm Modeli](#2-ölçüm-modeli)
3. [FODO-Antisimetrik Fourier Bazı](#3-fodo-antisimetrik-fourier-bazı)
4. [R-Matris En Küçük Kareler (R-LS)](#4-r-matris-en-küçük-kareler-r-ls)
5. [CLEAN Algoritması](#5-clean-algoritması)
6. [k-Modülasyon ile BPM Ofset İptali](#6-k-modülasyon-ile-bpm-ofset-iptali)
7. [Yöntem Karşılaştırması](#7-yöntem-karşılaştırması)
8. [Özet](#8-özet)

---

## 1. Problem Tanımı

pEDM halkasında $N_Q = 48$ kuadrupol mıknatıs, iki birbiri ardına gelen
odaklayıcı–saçtırıcı (QF–QD) çifti olan 24 FODO hücresine dizilmiştir
($R_0 = 95{,}49$ m, $g_1 = 0{,}21$ T/m, $L_q = 0{,}4$ m).

Her kuadrupol ideal ekseninden dikey yönde $dy_j$ kadar kaçık durmaktadır.
Kaçık duran bir kuadrupol, ek bir dipol-benzeri kick üretir:

$$\Delta y'_j = -K_j \cdot dy_j$$

burada $K_j = \frac{1}{B\rho} \frac{\partial B}{\partial x} L_q$ integre kick
gücüdür. Bu kick'ler birikir ve halkadaki **kapalı yörünge** (closed-orbit,
CO) ideal eksenden sapar. BPM $i$'deki sapma:

$$y_i^{\rm CO} = \frac{\sqrt{\beta_i}}{2\sin\pi\nu}
\sum_j \sqrt{\beta_j}\,
\cos\!\bigl(\pi\nu - |\phi_i - \phi_j|\bigr)\cdot \Delta y'_j$$

### Neden önemli?

Proton EDM ölçümünde hassasiyet hedefi ~10 μm'dir. Hedef harmonik
k=2'dir (halka çevresinde iki tam dalgalı CO büküntüsü); bu moda karşılık
gelen hizalama hatası 10 μm iken, arka planda **k=4 @ 300 μm**,
**k=6 @ 300 μm**, **k=8 @ 200 μm** büyüklüğünde parazit harmonikler
ve $\sigma_b = 100\,\mu\text{m}$ gürültü-düzeyinde BPM sistematik ofsetleri
bulunmaktadır. Dolayısıyla sinyalin arka plana oranı (SNR) ~ 1/30'dan
küçüktür; yöntem seçimi kritiktir.

---

## 2. Ölçüm Modeli

### Tek-yörünge ölçümü

48 BPM'li bir halka için ölçüm denklemi matris biçiminde:

$$\boxed{y = R\,dy + b + \varepsilon}$$

| Sembol | Boyut | Açıklama |
|--------|-------|----------|
| $y$ | $48\times 1$ | BPM ölçüm vektörü |
| $R$ | $48\times 48$ | Nominal tepki matrisi (ORM) |
| $dy$ | $48\times 1$ | Kuadrupol dikey hizalama hataları |
| $b$ | $48\times 1$ | BPM sistematik ofset vektörü ($\sigma_b \approx 100\,\mu$m) |
| $\varepsilon$ | $48\times 1$ | Ölçüm gürültüsü |

$R$ tam ranklıdır ($\text{rank}=48$, $\sigma_1 = 34{,}7$), ancak
$b$ bilinmemektedir. Bu nedenle **tek yörünge ölçümü** ile $dy$'yi
dolaysız çözülemez; BPM ofsetleri her şeyi baskılar.

### k-Modülasyon farkı

İki farklı gradient ayarında ($g_1 \to g_1 + \Delta g$) iki yörünge
ölçülürse:

$$\Delta y = y_2 - y_1 = (R_2 - R_1)\,dy + \underbrace{(b_2 - b_1)}_{\approx\,0}
+ \varepsilon_{\rm diff}$$

BPM ofseti **her iki ölçümde aynı** olduğundan ($b_2 = b_1 = b$) fark
denkleminde iptal olur. Tüm quadlara tek tip modülasyon uygulandığında
$R_2 - R_1 \approx \varepsilon_{\rm kmod}\,R_1$ (tam rankı korur,
$\varepsilon_{\rm kmod} \approx 0{,}02$).

> **Uyarı:** Yalnızca 1–2 quad modüle edildiğinde $\Delta R = R_2 - R_1$
> etkin rank 2'ye düşer (tekil değerler: 3,73 / 3,31 / 0,021 / …),
> harmonik ayrıştırma başarısız olur.

---

## 3. FODO-Antisimetrik Fourier Bazı

### Klasik azimutal baz ile farkı

Naif yaklaşım $\cos(k\theta_i)$, $\sin(k\theta_i)$ gibi düz azimutal
harmonikler kullanmaktır (Bozoki 1989). Ancak FODO yapısında QF ve QD
katsayıları birbirinin karşıtıdır: QF'de $+K$, QD'de $-K$. Bu
**işaret dönüşümü**, yörünge moduna doğal bir $(-1)^j$ faktörü
katar. Azimutal baz bu yapıyı yakalamaz; ~%312 sistemli önyargıya
(intrinsic bias) yol açar.

### FODO-antisimetrik baz

$$\boxed{F_{j,k}^{\rm cos} = (-1)^j \cos\!\left(\frac{2\pi k \lfloor j/2\rfloor}{N}\right),
\qquad
F_{j,k}^{\rm sin} = (-1)^j \sin\!\left(\frac{2\pi k \lfloor j/2\rfloor}{N}\right)}$$

burada $j = 0,1,\ldots,47$ quad indeksi, $N = N_Q/2 = 24$ FODO çifti
sayısıdır. $(-1)^j$ çarpanı QF–QD işaret dönüşümünü açıkça modeller.

### R-uzayında ortonormallik

R-uzayı sütunları $M_k = R_1 \cdot F_k$ alınarak normalizasyon yapılırsa,
farklı harmonikler arasındaki çapraz korelasyon son derece küçüktür:

| Harmonik çifti | Çapraz korelasyon |
|----------------|-------------------|
| k=2 vs k=4 | −0,0073 |
| k=2 vs k=6 | −0,0102 |
| k=2 vs k=8 | −0,0122 |

Bu yaklaşık ortonormallik, hem R-LS hem CLEAN yöntemlerinin teorik
temelini oluşturur: harmonikler birbirini karıştırmaz, ayrı ayrı
kestirilebilir.

---

## 4. R-Matris En Küçük Kareler (R-LS)

### İleri model

k harmoniğine ait hizalama hatası vektörü şöyle parametrelenir:

$$dy^{(k)} = F_k \cdot a_k, \qquad a_k = [a_c,\, a_s]^\top$$

Bütün katkı yörüngesi:

$$y = R\,dy \approx \sum_k R\,F_k\,a_k = \sum_k M_k\,a_k$$

burada $M_k = R\,F_k \in \mathbb{R}^{48\times 2}$ **ölçüm matrisi**dir.

### Çözüm

Yalnız k=2'yi kestireceksek:

$$\hat{a}_2 = M_2^\dagger\,y, \qquad
\hat{A}_2 = \|\hat{a}_2\|, \qquad
\hat{\phi}_2 = \arctan(\hat{a}_{2,s}/\hat{a}_{2,c})$$

Tekil değer dağılımı ($\sigma_1 = 34{,}7$, $\sigma_{48}=0{,}44$)
gösteriyor ki $M_2$ tam rankı destekler; çözüm iyi koşullanmıştır.

### Sabit kontaminasyon (bias)

Harmonikler tam ortonormal değildir ($|{corr}| \le 0{,}012$). Bu nedenle
k=2 kestiriminde arka plan harmoniklerinden sabit bir önyargı kalır:

$$\text{bias}_{k=2} = M_2^\dagger \sum_{k\ne 2} M_k\,a_k^{\rm true}$$

Sayısal hesap (params.json arka planı: k=4@300 μm, k=6@300 μm, k=8@200 μm):

| Kaynak | k=2'ye sızıntı |
|--------|----------------|
| k=4 @ 300 μm | 0,191 μm |
| k=6 @ 300 μm | 0,119 μm |
| k=8 @ 200 μm | 0,071 μm |
| **Toplam** | **0,381 μm** |

Bu sabit kontaminasyon, $A_2 < 4\,\mu$m bölgesinde %10'u aşan sistematik
hataya dönüşür. R-LS tek başına bu tabanı aşamaz.

### R-LS'nin gücü ve sınırı

- **Gücü:** FODO bazının doğru kullanımı sayesinde azimutal bazın
  %312 önyargısını önler; tek-yörünge yöntemiyle %3–5 doğruluk.
- **Sınırı:** BPM ofsetleri baskın ise (σ_b ≫ sinyal) doğrudan
  uygulanamazlıklı; ayrıca 0,38 μm sabit kontaminasyon tabanı vardır.

---

## 5. CLEAN Algoritması

### Motivasyon

R-LS yalnız hedef harmoniği (k=2) içeren tek bir adım çözer. Arka plan
harmonikleri (k=4,6,8) yüksek amplitüdlü olmakla birlikte, yaklaşık
ortonormallik sayesinde k=2'ye sızıntıları küçüktür — ama sıfır değildir.
CLEAN, sıralı "soyma" ile bu küçük ama sabit kontaminasyonu da ortadan
kaldırır.

### Algoritma adımları

Girdi: ölçüm vektörü $r = y$ (ya da $r = \Delta y$ kmod durumunda),
aday harmonik kümesi $\mathcal{K}$, döngü kazancı $g$, tolerans $\varepsilon$.

```
r ← ölçüm vektörü                       # artık (residual)
accum[k] ← 0   ∀k ∈ 𝒦                  # biriken katsayılar

döngü t = 1, 2, …, max_iter:
    best_k ← argmax_{k ∈ 𝒦} [ ||r||² − ||r − M_k M_k⁺ r||² ]
    â_k    ← M_k⁺ r                     # best harmoniği fit et
    r      ← r − g · M_k â_k            # kesirli çıkar (loop gain)
    accum[best_k] += g · â_k            # biriktir
    eğer ||r||/||r₀|| < ε: dur
```

**Loop kazancı** $g = 0{,}2$: her adımda artığın %20'si çıkarılır.
Büyük $g$ yakınsama hızını artırır ama dengesizleştirebilir;
küçük $g$ tutarlı ama yavaştır. $g = 0{,}2$ radyo-astronomi pratiğinden
alınan standart değerdir.

### Neden CLEAN işe yarar?

1. **Büyük harmonikler önce seçilir.** k=6 @ 300 μm, $\|M_6 a_6\|$ büyük
   olduğundan ilk yinelemede seçilir ve artıktan çıkarılır. k=2 @ 10 μm
   arka planda gizlenmiş olsa bile temizlenmiş artıkta görünür hale gelir.

2. **Kontaminasyon azalır.** R-LS'de sabit olan 0,38 μm bias,
   CLEAN'de arka plan harmoniklerinin adım adım çıkarılmasıyla
   küçülür.

3. **İteratif iyileştirme.** Ortonormallik tam olmadığından tek adım
   mükemmel değildir; ancak çok sayıda iterasyonla artık hata
   $\mathcal{O}(g^t)$ üssel hızla geriler.

### Yakınsama ve durma koşulu

Artık normu $\|r\| / \|r_0\| < 10^{-4}$ düzeyine indiğinde ya da
`max_iter = 300` adıma ulaşıldığında algoritma durur. Tipik yörünge
(params.json arka planı, gürültüsüz):

```
iter  1: k=6 seçildi   |r|/|r₀| = 0.68
iter  2: k=4 seçildi   |r|/|r₀| = 0.42
iter  5: k=8 seçildi   |r|/|r₀| = 0.21
iter 12: k=2 seçildi   |r|/|r₀| = 0.08
iter 40: k=2 seçildi   |r|/|r₀| = 0.004  → yakınsadı
```

k=6 ve k=4 önce ayıklanır; k=2 ancak bunlar temizlendikten sonra
güvenilir biçimde seçilir.

### CLEAN sonuç katsayıları

Sonunda her k için biriken amplitüd:

$$\hat{A}_k = \|\text{accum}[k]\|, \qquad
\hat{\phi}_k = \arctan\!\left(\frac{\text{accum}[k]_s}{\text{accum}[k]_c}\right)$$

---

## 6. k-Modülasyon ile BPM Ofset İptali

### BPM ofseti neden önemli?

$\sigma_b = 100\,\mu$m iken k=2 sinyali yalnız 10 μm'dir; SNR ~ 0,1.
Tek yörünge ölçümüyle (R-LS veya CLEAN) BPM ofseti $b$'yi bilinmediği
için k=2'den ayırt etmek imkânsızdır.

### Fark ölçümü

İki yörünge farkı:

$$\Delta y = R_2\,dy - R_1\,dy = (R_2 - R_1)\,dy$$

$b_2 = b_1$ olduğundan ofset tamamen iptâl olur. Tüm quadlara tek tip
modülasyon uygulandığında:

$$\Delta R \approx \varepsilon_{\rm kmod}\,R_1, \qquad \varepsilon_{\rm kmod} = 0{,}02$$

Bu durumda hem R-LS hem CLEAN için ölçüm matrisi:

$$M_k^{\rm kmod} = \Delta R \cdot F_k \approx \varepsilon_{\rm kmod}\,R_1\,F_k
= \varepsilon_{\rm kmod}\,M_k^{\rm nominal}$$

Oran küçüldü ama rank korundu; harmonik ayrıştırma çalışır.

### Tek-quad kmod neden yetersiz?

Yalnız 1–2 quad modüle edilirse $\Delta R = R_2 - R_1$ efektif rank 2'ye
iner (tekil değerler: 3,73 → 3,31 → 0,021 → …, faktör ~100 düşüş).
48'li harmonik setini 2-boyutlu bir uzayda çözmeye kalkmak rank
yetersizliğine ve %100+ hataya yol açar. Bu durumda CLEAN ya da R-LS
için $\Delta R$ yerine $\varepsilon_{\rm kmod} \cdot R_1$ kullanılmalıdır.

### CLEAN + kmod birleşimi

$r \leftarrow \Delta y$, $M_k \leftarrow \varepsilon_{\rm kmod} R_1 F_k$
başlangıcıyla standart CLEAN çalıştırılır. BPM ofsetinden kaynaklanan
bütün sistematik hata iptal olmuştur; yalnız 0,38 μm sabit kontaminasyon
tabanı kalır — ve CLEAN bunu da iteratif biçimde küçültür.

---

## 7. Yöntem Karşılaştırması

### Ölçüm matrisi güç sıralaması

Tepki matrisi R-uzayında her harmonik için ölçüm matrisi normu:

$$\|M_k\| = \|R_1 F_k\|_F$$

| k | $\|M_k\|$ | Anlamı |
|---|-----------|--------|
| 2 | 236,0 | Güçlü, tam rankta kestirilebilir |
| 4 | 22,2 | Orta güçte |
| 6 | 7,8 | Zayıf |
| 8 | 4,1 | Çok zayıf |

k=2 sinyali diğerlerine göre ~10 kat daha güçlü yanıt verir; bu nedenle
R-uzayında dominant harmoniktir.

### Dört yöntem karşılaştırması

| Yöntem | Bazı | BPM ofset | Sabit kontam. | $A_2 = 10\,\mu$m hatası |
|--------|------|-----------|--------------|--------------------------|
| Bozoki LS | Azimutal cos/sin | Var (dominas) | ~%312 | ~%305 |
| R-matris LS | FODO-antisim. | Var | 0,38 μm | ~%4 |
| CLEAN (kmodsuz) | FODO-antisim. | Var | azalır | ~%3,5 |
| CLEAN + kmod | FODO-antisim. | **İptal** | azalır | ~%0 |

- **Bozoki yöntemi:** FODO yapısını görmezden gelen azimutal baz
  kullandığı için ~%312 intrinsik önyargı içerir; gürültüsüz, parazitsiz
  durumda dahi bu hata ortadan kalkmaz.

- **R-matris LS:** Doğru FODO-antisimetrik bazı kullanır; tek-yörünge
  yöntemleri arasında en iyisidir ama 0,38 μm sabit kontaminasyon tabanı
  ve BPM ofset baskısı ile sınırlıdır.

- **CLEAN (kmodsuz):** R-LS ile aynı ileri modeli, ancak iteratif soyma
  sayesinde sabit kontaminasyonu kısmen giderir. $A_2 < 5\,\mu$m
  bölgesinde R-LS'den belirgin üstündür; BPM ofset hâlâ kısıtlayıcıdır.

- **CLEAN + kmod:** BPM ofset tamamen iptal edilmiş, arka plan harmonikleri
  iteratif ayıklanmış; pratikte sıfır sistematik hata.

### Kontaminasyon tabanı ve hata yüzdesi

Sabit kontaminasyon 0,38 μm'dir; $A_2$ cinsinden göreli hatası:

$$\text{hata}(\%) = \frac{0{,}38\,\mu\text{m}}{A_2} \times 100$$

| $A_2$ | R-LS tahmini hatası |
|--------|---------------------|
| 100 μm | %0,38 |
| 10 μm | %3,8 |
| 3 μm | %12,7 |
| 1 μm | %38 |

CLEAN bu tabanı yaklaşık 2–3 kat azaltabilir; ancak mükemmel
ayıklama için kmod kaçınılmazdır.

---

## 8. Özet

### R-matris LS nedir?

Kuadrupol hizalama hatalarının FODO-antisimetrik Fourier parametrelemesi
altında, tepki matrisi $R$ üzerinden **tek adımda** en küçük kareler
çözümüdür. Formül:

$$\hat{a}_k = (R\,F_k)^\dagger\,y$$

Doğru ileri modeli kullandığı için Bozoki'nin %300+ önyargısını önler;
ancak BPM ofsetlerinin varlığında tek-yörünge uygulanabilirliği kısıtlıdır.

### CLEAN nedir?

Radyo-astronominin **CLEAN** algoritmasından uyarlanmış, iteratif ve
açgözlü (greedy) harmonik soyma yöntemidir. Her adımda artık sinyali en
çok azaltan harmoniğin küçük bir kesrini ($g=0{,}2$) çıkarır.
Böylece dominant harmonikler (k=6,4) önce, zayıf hedef (k=2) sonra
temiz bir artıkta kestirilebilir.

### Ne zaman hangisi?

| Koşul | Önerilen yöntem |
|-------|----------------|
| Hızlı tek ölçüm, yüksek SNR | R-matris LS |
| Arka plan harmonikleri baskın | CLEAN (kmodsuz) |
| BPM ofset ~ sinyal büyüklüğü | kmod + R-LS veya kmod + CLEAN |
| Maksimum doğruluk gerekli | kmod + CLEAN |

Her iki yöntemin temel avantajı, **FODO-antisimetrik Fourier bazının**
halkanın fiziksel yapısıyla örtüşmesidir. Bu baz kullanılmadan (azimutal
Fourier bazıyla) hiçbir istatistiksel teknik sistematik önyargıyı gideremez.

---

*Simülasyon parametreleri: N_Q = 48, nFODO = 24, R_0 = 95,49 m,
g₁ = 0,21 T/m. Referans dosyalar: `params.json`, `fourier_reconstruct.py`,
`R_dy_1.npy`.*
