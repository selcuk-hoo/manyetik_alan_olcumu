# k=1..3 Harmonik Düzeltmesinin Sınırları: Betatron Kirlenmesi ve Eşit-Katkı

> **Referans not** — bu belge `test_false_edm_modes.py` taramalarından elde edilen
> bulgular ışığında neden k=1,2,3 modlarını yörüngeden temizlemenin sahte EDM
> sinyalini temizlemediğini açıklar.  Önceki `false_edm_yontemi.md` belgesi
> CO=True (kapalı yörünge fırlatması) rejimini anlatır; bu belge ise CO=False
> (gerçekçi fırlatma) rejimindeki mekanizmayı açıklar ve iki rejim arasındaki
> farkın fiziksel kaynağını türetir.

---

## İçindekiler

1. [İki Rejim: CO=True ve CO=False](#1-iki-rejim)
2. [Neden Birinci-Dereceden Etki Sıfır?](#2-birinci-derece-sifir)
3. [Betatron Kirlenmesi Mekanizması](#3-betatron-kirlenmesi)
4. [Tüm Modların Eşit Katkısı](#4-esit-katki)
5. [k=3'ün Hafif Baskınlığı: Qy Yakın Rezonansı](#5-k3-baskini)
6. [Side Band Hipotezi ve Çürütülmesi](#6-side-band)
7. [k=1,2,3 Düzeltmesinin %30 Etkinliği — Sayısal Kanıt](#7-yuzde30)
8. [Yatay-Dikey Çapraz Terim (Yeni Bulgu)](#8-capraz-terim)
9. [Makale Revizyonu İçin Çıkarımlar](#9-makale)
10. [Üç Açık Sorunun Sayısal Yanıtı](#10-uc-soru)
11. [Test Sonuçları: Kick Düzeltmesi ve Harmonik İptal](#11-test-sonuclari)

---

## 1. İki Rejim: CO=True ve CO=False <a name="1-iki-rejim"></a>

Sahte EDM simülasyonunda parçacığın başlangıç koşulu iki farklı şekilde
belirlenebilir:

| Parametre | CO=True | CO=False |
|---|---|---|
| `use_closed_orbit` | `true` | `false` |
| Başlangıç noktası | Kapalı yörünge üzerinde | $y=\delta_0$, $y'=0$ (ideal $y=0$) |
| Betatron genliği | $J_y \approx 0$ | $J_y \sim \delta_0$ (tam betatron) |
| Sinyal seviyesi | $\sim 10^{-8}$–$10^{-9}$ rad/s | $\sim 10^{-4}$–$10^{-5}$ rad/s |
| Baskın mod | $k=2$ (Q_y rezonansı) | $k=3$ (hafifçe) + tümü eşit |
| Dominant mekanizma | Kapalı-yörünge spin kumu | **Betatron kirlenmesi** |

CO=True, deneyin hedeflendiği "ideal" koşuldur ve gerçek false EDM
sinyalini verir. CO=False, başlangıç koşulunu yörünge üzerine oturtamadan
yapılan ölçümü temsil eder; betatron hareketi baskın olmaktadır.

Önemli: hangi rejimde çalışıldığını sinyalin büyüklüğünden anlamak
mümkündür. $\sim 10^{-4}$ mertebesindeki sinyal CO=False rejimini gösterir.

---

## 2. Neden Birinci-Dereceden Etki Sıfır? <a name="2-birinci-derece-sifir"></a>

Antisimetrik FODO bazında $k$-modundaki kuadrupol kayması:
$$\Delta y_j = A \cdot (-1)^j \cos\!\Bigl(\frac{2\pi k \lfloor j/2\rfloor}{N}\Bigr),
\quad j = 0,\ldots,47,\; N=24.$$

Bu kayma kapalı yörünge bozunumunu (KYB) $y_{\text{CO},j}$ oluşturur.
Spin'in bir tur sonundaki dikey değişimi:
$$\Delta S_y \propto \sum_j G_j \cdot y_{\text{CO},j}.$$

FODO örgüsünde $G_j = (-1)^j G_0$ (F quadlar $+G_0$, D quadlar $-G_0$):
$$\sum_j G_j \cdot y_{\text{CO},j}
= G_0 \sum_j (-1)^j \cdot (-1)^j \cos(\cdots)
= G_0 \sum_j \cos\!\Bigl(\frac{2\pi k m_j}{N}\Bigr) = 0, \quad k \ge 1.$$

İki $(-1)^j$ çarpımı birbirini tam olarak iptal eder. Yani FODO
antisimetrik $k \ge 1$ modlarının **tamamı birinci dereceden sıfır
false EDM** üretir; döngüsel ortalama tam olarak kapanır.

**Sonuç:** CO=True koşulunda çok küçük olan sinyal ($\sim 10^{-9}$ rad/s),
ikinci-derece ve rezonans katkılarından kaynaklanır; baskın mod k=2'dir.

---

## 3. Betatron Kirlenmesi Mekanizması <a name="3-betatron-kirlenmesi"></a>

CO=False'ta parçacık kapalı yörünge etrafında betatron hareketi yapar:
$$y_j(n) = y_{\text{CO},j} + \sqrt{\frac{\beta_j}{\beta_0}}\,\delta_0
\cos(2\pi Q_y n + \psi_j).$$

Stroboskopik örneklemede ölçülen $S_y(n)$'nin doğrusal fit eğimi:
$$\left.\frac{dS_y}{dt}\right|_{\text{ölçüm}} = \underbrace{\frac{dS_y}{dt}\bigg|_{\text{gerçek}}}_{\approx 0} + \underbrace{\text{betatron kirlenmesi}}_{\text{baskın}}.$$

Birinci-derece etki sıfır olduğundan ölçüm **tamamen** betatron
kirlenme katkısından oluşur.

Betatron kirlenmesinin kaynağı: $Q_y = 67/25 = 2.68$ tam rasyonel.
$T = 219$ dönüşte beklenen $T \cdot Q_y = 219 \times 2.68 = 586.92$, yani
$219 = 8 \times 25 + 19$ — her 25 dönüşten sonra 19 turn artık kalır.
Bu artık betatron fazından, lineer fit kaçınılmaz biçimde etkilenir.

---

## 4. Tüm Modların Eşit Katkısı <a name="4-esit-katki"></a>

Her $k$ modu için betatron kirlenmesi:
$$\text{kirlenme}(k) \propto \frac{|\text{DFT}_n[\text{spin kick}]|}{T \cdot |\sin(\pi \cdot \text{beat}(k))|},$$
burada $\text{beat}(k) = |n_k - Q_y|$ stroboskopik frekans ve
$\text{DFT}_n[\text{spin kick}]$, spin kick profilinin $n=k$ harmonik
bileşenidir.

Spin kick profili $G_j \cdot y_j = G_0 \cos(2\pi k m_j / N)$:
* $k=1$: DFT$[n=1] = 0.499$, beat $= 1.68$
* $k=2$: DFT$[n=2] = 0.496$, beat $= 0.68$
* $k=3$: DFT$[n=3] = 0.490$, beat $= 0.32$
* $k=4$: DFT$[n=4] = 0.483$, beat $= 1.32$
* ...

Her modun stroboskopik kirlenme katkısı hesaplandığında (tablo):

| $k$ | DFT amp | beat | Yakl. kirlenme | Toplama oranı |
|-----|---------|------|----------------|---------------|
| 1 | 0.499 | 1.68 | ≈7.2×10⁻⁵ | %10.9 |
| 2 | 0.496 | 0.68 | ≈7.1×10⁻⁵ | %10.8 |
| 3 | 0.490 | 0.32 | ≈7.0×10⁻⁵ | %10.7 |
| 4 | 0.483 | 1.32 | ≈6.9×10⁻⁵ | %10.5 |
| 5 | 0.473 | 2.32 | ≈6.8×10⁻⁵ | %10.3 |
| 6 | 0.462 | 3.32 | ≈6.6×10⁻⁵ | %10.0 |
| 7 | 0.448 | 4.32 | ≈6.4×10⁻⁵ | %9.8 |
| 8 | 0.433 | 5.32 | ≈6.2×10⁻⁵ | %9.4 |
| 9 | 0.416 | 6.32 | ≈5.9×10⁻⁵ | %9.0 |
| 10 | 0.397 | 7.32 | ≈5.7×10⁻⁵ | %8.6 |

**Her mod yaklaşık %10 katkı verir.** Bu eşitliğin kökü şudur: DFT
genliği $k$ arttıkça azalırken ($\sim \cos(\pi k/48)$), beat frekansı
da artar ve bunlar $|\sin(\pi \cdot \text{beat})|$'te kısmen dengelenir.
Net etki: $k=1$'den $k=10$'a gidişte katkı ancak %10.9'dan %8.6'ya
düşer — temelde eşit dağılım.

---

## 5. k=3'ün Hafif Baskınlığı: Qy Yakın Rezonansı <a name="5-k3-baskini"></a>

Simülasyonda $k=3$ modu hafifçe baskın çıkar (1.146×10⁻⁴ rad/s), $k=2$
ise daha küçük (6.435×10⁻⁵ rad/s). Nedeni:

Spin kick profili için $k=3$ modunun devrim harmonik DFT'si $n=3$'te
yoğunlaşır ve $|3 - Q_y| = |3 - 2.68| = 0.32$ en küçük beat frekansıdır.
Beat küçük → stroboskopik ortalama en yavaş kapanır → eğim kirlenmesi
diğer modlardan görece büyük.

Ancak fark küçüktür (%10.7 vs %10.9). **k=3 baskın görünmesi, k=1,2
önemsiz anlamına gelmez**; 7 mod olan k=4..10 toplamı k=1..3 toplamından
yaklaşık 2 kat büyüktür.

---

## 6. Side Band Hipotezi ve Çürütülmesi <a name="6-side-band"></a>

Gözlemden yola çıkarak şu hipotez kurulabilir: "FODO modu $k=2$ devrim
harmonik uzayında sadece $n=2$ değil, büyük genlikli $n=22$ ve $n=26$
bileşenleri de üretir; bunlar ('side band'lar) baskın katkıyı verir ve
$k=1,2,3$ düzeltmesi bunları temizleyemez."

Bu hipotez **kısmen doğru ama sonuç yanlıştır:**

**Yörünge DFT'si** (deplasman $y_j$): gerçekten $k=2$ modunda $n=22$
bileşeni büyüktür (|A|=0.496), $n=2$ küçüktür (|A|=0.065) — faktör ~7.6.

**Ama spin kick DFT'si** ($G_j \cdot y_j = G_0 \cos(\cdots)$) **tersine
döner**: $n=2$ büyüktür (|A|=0.496), $n=22$ küçüktür (|A|=0.065).
$G_j \propto (-1)^j$ ile $y_j \propto (-1)^j$ çarpımı $(-1)^{2j}=1$
verir, yörüngedeki $(-1)^j$ modulasyonu spin kick'ten silinir.

Dolayısıyla **side band n=22,26 yörüngede büyük görünür ama spin tepkisinde
küçüktür**. Spin $n=k$ harmonik üzerinden etkilenir, $n=N-k$ üzerinden değil.
$k=2$ düzeltmesi spin kick'in $n=2$ bileşenini kapatır; side band
ayrı ve bağımsız bir sinyal değildir.

Bunun üstüne: FODO antisimetrik bazında aliasing, $k=22 \equiv k=2$,
$k=26 \equiv k=2$ özdeşliğini verir (sayısal doğrulama: max|mod(22) - mod(2)| = 1.8×10⁻¹⁴).
Ayrı bir "side band düzeltmesi" yapmak imkânsızdır, çünkü side band'lar
zaten aynı fiziksel dizi.

---

## 7. k=1,2,3 Düzeltmesinin %30 Etkinliği — Sayısal Kanıt <a name="7-yuzde30"></a>

$k=1..24$ tek-mod taramasından (her mod 100 μm cos, t2=1 ms, CO=False):

```
k=1:  ~6e-05 rad/s        k=13: = k=11 (aliasing)
k=2:  6.435e-05 rad/s     k=22: = k=2  (aliasing, doğrulandı)
k=3:  1.146e-04 rad/s
k=4:  ...                 k=12 dahil k=4..12 her biri katkı verir
...
```

Çoklu-mod senaryosunda (k=1..10 her biri 100 μm, eşit faz):
- **Sadece k=1,2,3 sıfırlanınca** kalan sinyal: ~%68 orijinal
- k=1..3 düzeltme etkinliği: **~%32**

Bu teorik tahminle (%32.3 = k=1+2+3 oranları toplamı) birebir örtüşür.

**Neden %32 ve daha fazla değil?** k=4..10 arası 7 mod, her biri ~%9-10
katkıyla, toplamda %66 verir. Üç mod silince 7 mod kalır; sayısal üstünlük
k=4..10 tarafındadır.

---

## 8. Yatay-Dikey Çapraz Terim (Yeni Bulgu) <a name="8-capraz-terim"></a>

Saf $dy$ (dikey) + saf $dx$ (yatay) birlikteliğinde ikinci-derece bir
çapraz terim ortaya çıkar:

$$\left.\frac{dS_y}{dt}\right|_{dx+dy} =
\left.\frac{dS_y}{dt}\right|_{dy} +
\left.\frac{dS_y}{dt}\right|_{dx} +
\underbrace{\varepsilon_{\times} \cdot x_0 \cdot (\text{dy genliği})}_{\text{çapraz terim}}.$$

Simülasyon sonuçları (k=3 dy=100μm, farklı $x_0$):

| Kutuplanma | Çapraz/sinyal | $x_0$ ölçeklemesi |
|---|---|---|
| Boyuna (Sz=1) | ~%0.2 | doğrusal (onaylandı) |
| **Radyal (Sx=1)** | **~%18** | doğrusal ($\varepsilon_\times/x_0 = -4.22$ s⁻¹/m) |

Mekanizma (radyal kutuplanma için):
1. $x$ yörünge → $B_y = G \cdot x$ → $dS_z/dt \propto G \cdot x \cdot S_x$ (Sx=1 → ilk-dereceden Sz büyümesi)
2. $\Delta S_z$ → $B_x = G \cdot y$ (dy ofseti) → $dS_y/dt \propto G \cdot y \cdot \Delta S_z$

İki adım bilineeri: $\varepsilon_\times \propto x \cdot y \cdot G^2$.

**Boyuna demet için %0.2, radyal demet için %18** çıkmasının nedeni:
adım 1'deki kazanç $S_x$ ile orantılıdır — boyuna demette $S_x \approx 0$
(birinci adım çalışmaz), radyal demette $S_x = 1$ (birinci adım
ilk-dereceden aktif).

Bu çapraz terim, BPM'den ölçülen $\langle x \cdot y \rangle$ harmonik
korelasyonuyla ilişkilendirilebilir. Gerçek halkada $dx$ misalignment
veya başka yatay bozunumlar varsa bu kanal sahte EDM'e katkıda bulunur
ve radyal demet ölçümü ile kalibre edilebilir. Bu mekanizmanın
simülasyonla sistematik olarak karakterize edilmesi devam etmektedir.

---

## 9. Makale Revizyonu İçin Çıkarımlar <a name="9-makale"></a>

`makale_tr.tex` şu anda şu iddiayı taşıyor:

> "yalnızca baskın sahte-EDM harmonik bileşenini ($k=2$) tahmin etmeyi
> hedefleyen ... $k=1,2,3$ modları ... kurtarılmaktadır"

Yeni bulgular bu çerçeveyi iki düzeyde değiştiriyor:

### 9.1 k=2 baskın mı?

CO=True rejiminde evet: kapalı yörünge kazancı k=2'de zirve yapar
($Q_y \approx 2.68$ rezonansına en yakın), gerçek sahte EDM
$\sim 10^{-9}$ rad/s seviyesinde k=2 baskındır.

CO=False rejiminde hayır: betatron kirlenmesi dominanttır, her mod
yaklaşık eşit katkı verir, k=3 marjinal olarak öne çıkar.

**Makaleye eklenmesi gereken:** hangi rejimin incelendiği ve iki
rejim arasındaki sinyal mertebesi farkı (10⁻⁹ vs 10⁻⁴) açıkça
belirtilmelidir. Makale CO=True'yu ele alıyorsa (önerilir) bu
belirtilmeli; CO=False'ta hangi modun baskın olduğu farklıdır.

### 9.2 k=1,2,3 kurtarılınca false EDM gideriliyor mu?

CO=True rejiminde: k=2 hakikaten baskındır; k=1,2,3 kurtarılırsa
false EDM bütçesinin büyük bölümü kapatılır. Bu çerçevede makale iddiası
tutarlıdır.

CO=False rejiminde: k=1,2,3 düzeltmesi false EDM'i yalnızca %30 azaltır;
k=4..10 modları baskın olmaya devam eder. Makale bu rejimi hedef
almıyorsa bunu açıkça sınırlamak gerekmektedir.

### 9.3 Önerilen revizyon adımları

1. Soyutta "baskın harmonik k=2" ifadesini "CO=True rejiminde baskın
   harmonik k=2" biçiminde sınırlandır.
2. Giriş bölümüne CO=True / CO=False ayrımını ve sinyal mertebesini ekle.
3. Simülasyon bölümüne §2 türetmesini (birinci-derece sıfır) ve betatron
   kirlenmesi mekanizmasını kısaca ekle (neden CO=True kritik).
4. Tartışma bölümüne: "CO=True sağlanamıyorsa k=1,2,3 düzeltmesi yeterli
   değildir; tüm modların eşit katkısı betatron kirlenmesinden kaynaklanır"
   notunu ekle.
5. §8'deki yatay-dikey çapraz terimi, gelecek çalışma olarak not et.

---

## 10. Üç Açık Sorunun Sayısal Yanıtı <a name="10-uc-soru"></a>

> Tüm sonuçlar `test_cross_correlation.py` çıktısından alınmıştır.
> CO=True simülasyonları: `A = 10 μm`, `t2 = 0.5 ms`, `co_turns = 24`.

### 10.1 CO=True Geçerliliği: Tek Kick Yeterli mi?

| Parametre | CO=True | CO=False |
|---|---|---|
| k=2, A=10μm | +1.384e-9 rad/s | +1.458e-4 rad/s |
| Oran | 1 | **1.1 × 10⁵ ×** |

**CO=True'nun gerçek hızlandırıcıda karşılığı:** Ring boyunca dağıtık BPM
ölçümü + harmonik analiz + korrektör sistemi. Mevcut yörünge,
korrektörler aracılığıyla kapalı yörüngeye yaklaştırılır → CO=True koşulu
pratik olarak sağlanır.

**Tek noktada kick:** Tek bir korrektör kick'i, yalnızca bir harmonik
bileşeni kısmen değiştirir. Ring'in geri kalanındaki betatron amplitüdü
azalmaz. CO=True rejimi elde etmek için **en az `2N` korrektör** (N =
düzeltilecek mod sayısı) gerekir. g-2 deneyindeki "CBO kick" analog değildir:
buradaki bozunum statik kapalı yörünge sapmasıdır (serbest salınım değil).

### 10.2 Çapraz Korelasyon Matrisi: İşaret Bulgusu

3×3 ölçüm matrisi (k=2, 3, 4 cos bileşenleri, `A = 10 μm`):

```
         k=2           k=3           k=4
k=2  +1.384e+01  -3.158e+00  -7.888e-01
k=3  -3.158e+00  -6.160e+00  -9.115e-01
k=4  -7.888e-01  -9.115e-01  -1.926e+00
```

**Kritik bulgu:** köşegenler `c_33 = -6.16`, `c_44 = -1.93` **negatif**.
n_iter=2, co_turns=36 ile doğrulanan değerler:

| Mod | dSy/dt (A=10μm) | CO ofseti | İşaret |
|---|---|---|---|
| k=2 | +1.401e-9 rad/s | 0.198 mm | **Pozitif** |
| k=3 | -6.320e-10 rad/s | 0.088 mm | **Negatif** |
| k=4 | -1.951e-10 rad/s | 0.028 mm | **Negatif** |

**Fiziksel açıklama:** Betatron tune'u Q_y ≈ 2.68'dir.

- k=2: Q_y > 2 → rezonansın **altında** → pozitif spin döndürme
- k=3: Q_y < 3 → rezonansın **üstünde** → negatif spin döndürme
- k=4: Q_y < 4 → rezonansın üstünde → negatif, daha küçük

Bu, klasik rezonans geçiş işareti dönüşümüdür.
k = Q_y ≈ 2.68'in altındaki modlar pozitif, üstündekiler negatif false
EDM üretir.

**Rassal hizalama neden küçük?** k=2 (pozitif) ve k=3, k=4 (negatif)
katkıları kısmen birbirini götürür:

$$\langle dS_y/dt \rangle_{\text{rassal}} \approx \frac{1}{24}\left[c_2 + c_3 + c_4 + \ldots\right] \cdot A^2 \ll c_2 \cdot A^2$$

Yani rassal hizalamada ~65× bastırım, çoğunlukla modlar arası işaret
iptali ile açıklanır; bu da k=2 tek başına baskın (koheren) durumun
özellikle tehlikeli olduğunu doğrular.

**Genlik skalama testi** (§10.3): `dSy/dt ∝ A^1` — lineer.

| Mod | A=5μm | A=10μm | A=20μm | F(2A)/F(A) |
|---|---|---|---|---|
| k=2 | 6.32e-10 | 1.39e-9 | 3.09e-9 | **2.21** |
| k=3 | -2.96e-10 | -6.23e-10 | -1.23e-9 | **2.10, 1.97** |

Lineer ölçek = 2.0, kuadratik = 4.0; ölçülen ≈ 2.1 → **lineer baskın.**

> **DÜZELTME (bkz. §13):** Bu doğrusal baskınlık önceden *kalın-lens
> (thick-lens) birinci-derece iptali hafifçe bozuyor* diye açıklanmıştı.
> **Bu açıklama §13.2'de ÇÜRÜTÜLDÜ:** quad uzunluğu 8× inceltilip integral
> güç $K=G\cdot L$ sabit tutulduğunda α **değişmedi** (1.24→1.19). Yani
> doğrusal terimin kaynağı kalın-lens DEĞİLDİR. Gerçek kaynak: §2'deki
> ideal birinci-derece iptal, gerçek kapalı yörüngenin betatron-faz
> transferiyle *yayılmış* olmasından ötürü tam kapanmaz; dy-only için
> bu doğrusal terim gerçek ve indirgenemezdir. Ayrıca asıl Omarov
> kuadratiği bu (dy-only) kanalda değil, **dx·dy geometrik-faz çapraz
> kanalındadır** (§13.3b, α=2.00). Aşağıdaki $c^{(1)}\gg c^{(2)}A$ ifadesi
> yalnızca dy-only kanalın küçük-genlik davranışını betimler:
$$\frac{dS_y}{dt}\bigg|_{dy\text{-only}} \approx c_k^{(1)} A_k + c_k^{(2)} A_k^2,
\quad c^{(1)} \gg c^{(2)} A \;\; (A \lesssim 20\,\mu\text{m})$$

### 10.3 Optimizasyon: Minimum False EDM Konfigürasyonu

Özdeğer ayrışımı:

| Özdeğer λ | Özvektör (k2, k3, k4) | Fiziksel anlam |
|---|---|---|
| **-6.86** s⁻¹m⁻² | [+0.16, **+0.97**, +0.20] | k=3 baskın → negatif |
| -1.74 s⁻¹m⁻² | [+0.01, -0.21, +0.98] | k=4 baskın → negatif |
| **+14.35** s⁻¹m⁻² | [-0.99, +0.15, +0.04] | k=2 baskın → pozitif |

Doğrulama simülasyonu (A=10μm total):
- Optimal kombinasyon (λ_min özvektörü): ölçülen **-4.14e-10 rad/s**
  (teorik: -6.86e-10 rad/s) → k=2 tek modundan **3.3× bastırım**
- En kötü kombinasyon (λ_max): **-1.45e-9 rad/s**

**Pratik optimizasyon stratejisi:**

$dS_y/dt$ fonksiyonu mod katsayılarında yaklaşık LİNEER:
$$\frac{dS_y}{dt} \approx \sum_k c_k A_k, \quad c_2 > 0,\; c_3 < 0,\; c_4 < 0.$$

Spin ölçümü ile optimizasyon:
1. Mevcut hizalama deseni için dSy/dt'yi ölç (baseline)
2. Orbit korrektörüyle k=2 bileşenini tarar: $A_2 \to A_2 - \delta$
3. dSy/dt'nin sıfır geçişini yakala → $\delta^* = A_2 + (c_3 A_3 + c_4 A_4)/c_2$
4. Bu noktada $\sum c_k A_k = 0$ → false EDM sıfırlanmış

**Bu k=2'yi mükemmel kaldırmaktan daha iyi bir hedeftir:** hedef k=2
orbit sıfırı değil, dSy/dt sıfırıdır.

Gerekli spin ölçümü sayısı: k=2 taraması için 5-7 ölçüm yeterli
(doğrusal fiti bulmak için). Pratikte bu, ring'in bir deney çalışmasına
karşılık gelir.

---

## 11. Test Sonuçları: Kick Düzeltmesi ve Harmonik İptal <a name="11-test-sonuclari"></a>

### 11.1 Test A — Kick Sayısı vs False EDM (`test_kick_correction.py`)

**Yöntem:** k=2 cos hizalama (A=10μm) için N equidistant QF korrektör ile
analitik Courant-Snyder Green fonksiyonu üzerinden en küçük kareler orbit
düzeltmesi; ardından tam simülasyon ile dSy/dt ölçümü.

**Sayısal sonuçlar (CO=True, T2=5×10⁻⁴ s):**

| N_corr | dSy/dt [rad/s] | Bastırma | CO artık RMS [μm] |
|---|---|---|---|
| 0 | 1.38×10⁻⁹ | 1.00 | 9.13 |
| 2 | 1.18×10⁻⁹ | 0.85 | 5.80 |
| 4 | 1.16×10⁻⁹ | 0.84 | 4.68 |
| 8 | 1.21×10⁻⁹ | 0.87 | 1.29 |
| 12 | 1.16×10⁻⁹ | 0.84 | 0.66 |
| 24 | 1.18×10⁻⁹ | 0.85 | 0.46 |
| 48 | ~0 | ~0 | ~0 |

**Ana bulgu:** N=2..24 korrektör orbiti %50-95 azaltır ama false EDM yalnızca
~%15 düşer. N=48 (tüm quadlar korrektör, yani tam orbit iptali) → sıfır
false EDM.

**Fiziksel yorum:** Orbit minimizasyonu ve dSy/dt minimizasyonu farklı
optimizasyon hedefleridir. Orbit düzeltmesi ile elde edilen artık orbit
$y_{\text{CO}}^{\text{corr}}$ hâlâ k=2 Fourier bileşeni içerir çünkü
equidistant QF korrektörler k=2 cos ve sin bileşenlerini aynı anda bastıracak
şekilde konumlandırılmamıştır (N=8 için cos bileşeni erişilebilir, N=4 için
yalnızca tek bileşen). False EDM için asıl önemli olan kapalı-yörüngenin
k=2 Fourier spektrum bileşenidir, toplam amplitude değil.

**Sonuç:** Yörünge tabanlı düzeltme yeterli değildir; spesifik harmonik
hedefleme (Test B, harmonic cancellation) veya tam yörünge iptali (N=48)
gereklidir.

### 11.2 Test B — Harmonik Kombinasyon ile False EDM İptali
**Dosya:** `test_harmonic_cancellation.py` (henüz çalıştırılmadı)

k=3 cos modunu A₃ ≈ 22.5 μm ile ekleyerek k=2 kaynaklı false EDM'yi iptal
etmek. Test A'nın aksine bu yöntem orbiti büyütür ama dSy/dt'yi hedefler.
Beklenen A₃ taraması: A₂=10μm sabit, A₃ ∈ [−20, +30] μm, sıfır geçişi
~22.5 μm'de beklenir.

### 11.3 İki Yaklaşımın Karşılaştırması

| Özellik | Kick düzeltmesi (Test A) | Harmonik iptal (Test B) |
|---|---|---|
| Hedef | Orbit → 0 | dSy/dt → 0 |
| Orbit büyüklüğü sonrası | Azalır (N×) | Büyüyebilir |
| False EDM bastırımı (az korrektörle) | Zayıf (%15) | Kuvvetli (teorik %100) |
| Tam düzeltme (N=48) | Evet | — |
| Gürültü sağlamlığı | Yüksek (LSQ) | A₃ hassasiyetine bağlı |
| Pratik uygulanabilirlik | Karmaşık (çok korrektör) | Basit (1 mod ekle) |

**Önerilen strateji:** Önce harmonik iptal (Test B) ile dSy/dt ≈ 0 noktasını
bul; ardından yeterli korrektör sayısını Test A sonuçlarından seç.

---

## 12. Test B Sonuçları: Ortak-Mod Arama ve Birleşik Bastırma <a name="12-test-b"></a>

**Dosyalar:** `test_b_partner_search.py`, `test_b_combined.py`,
`test_b_ck_table.json` (ölçülen katsayılar)
**Kafes:** üniform 0.2 T/m (g0=g1=0.2), CO=True, T2=0.5 ms

### 12.1 c_k tablosu k=1..10 (ilk geniş tarama)

| k | c_k [rad/s/m] | işaret |
|---|---|---|
| 1 | +4.566×10⁻⁵ | + |
| 2 | **+1.974×10⁻⁴** | + (baskın) |
| 3 | −4.796×10⁻⁵ | − |
| 4 | −1.629×10⁻⁵ | − |
| 5 | −9.10×10⁻⁶ | − |
| 6 | −5.65×10⁻⁶ | − |
| 7 | −3.99×10⁻⁶ | − |
| 8 | −3.08×10⁻⁶ | − |
| 9 | −2.45×10⁻⁶ | − |
| 10 | −2.09×10⁻⁶ | − |

İşaret kuralı doğrulandı: k < Q_y ≈ 2.68 → pozitif; k > Q_y → negatif.
k ≥ 3 için |c_k| kabaca 1/k² ile azalır (rezonans paydası 1/(k²−Q_y²)).

### 12.2 k=2 ortağı: tekil mod telafisi verimsiz

| ortak k' | A* (lineer) | gerçek artık | bastırma |
|---|---|---|---|
| 3 | 41.2 μm | −1.6×10⁻¹⁰ | 12× |
| **4** | **121.2 μm** | **+8.3×10⁻¹¹** | **24×** |
| 5..10 | 217–945 μm | büyür | 1–6× |

İnce tarama (k'=4): gerçek sıfır geçişi 125.0 μm — lineer tahminden yalnızca
%3.1 sapma → lineer model güvenilir. Ancak **c₂ baskın olduğundan k=2'yi tek
zayıf modla silmek 4–12× büyük telafi genliği ister** — hizalama bütçesi
açısından pratik değil. Ters yön ucuz: k=3'teki 10 μm, sadece 2.4 μm'lik
k=2 ile silinir.

### 12.3 k=1 ve k=3 ortakları: dengeli çiftler pratik

| hedef | ortak k' | A* | artık | bastırma |
|---|---|---|---|---|
| k=1 (10μm) | 3 | +9.5 μm | +1.7×10⁻¹¹ | 27× |
| k=1 (10μm) | 4 | +28.0 μm | −7.3×10⁻¹² | **63×** |
| k=3 (10μm) | 1 | +10.5 μm | +1.8×10⁻¹¹ | 27× |
| k=3 (10μm) | 2 | +2.4 μm | −3.4×10⁻¹¹ | 14× |

|c| büyüklükleri dengeli çiftlerde (k=1↔k=3) telafi genliği hedef
genlikle aynı mertebede → uygulanabilir.

### 12.4 Birleşik bastırma: a₁=a₂=a₃=10 μm

| durum | dSy/dt [rad/s] | bastırma |
|---|---|---|
| lineer tahmin Σc_k·a_k | +1.951×10⁻⁹ | — |
| (a) telafisiz (gerçek) | +2.050×10⁻⁹ | — |
| (b) k=3 trim (+40.7 μm) | −2.9×10⁻¹¹ | 70× |
| (c) k=3 + k=4 bölüşmüş (+30.4 μm her biri) | **−4.4×10⁻¹²** | **468×** |

Bulgular:

1. **Lineerlik:** tahmin/gerçek farkı %5 — Σc_k·a_k modeli birleşik desende
   de geçerli.
2. **Tek serbest mod matematiksel olarak yeterli** (tek kısıt denklemi),
   70× bastırma sağlar.
3. **İki moda bölüştürme hem genlik bütçesini korur hem bastırmayı
   derinleştirir** (468×): küçük tekil genlikler lineer bölgede kalır,
   lineer-ötesi artıklar küçülür.

**Sonraki adım:** rastgele desenlerde (ölçülen â_k projeksiyonlarından
hesaplanan trim ile) evrensellik testi; ardından k_targets={1,2,3} +
bölüşmüş trim stratejisinin BPM gürültüsü altındaki sağlamlığı.

### 12.5 Mod haritası ve evrensellik: c_k (k=1..24) üç arka planda

**Dosyalar:** `test_b_mode_map.py`, `test_b_mode_map.json`,
`test_b_mode_map.png`

Tasarım: lineer sistemde 24×24 telafi haritası M[k,k'] = −c_k/c_k'
**rank-1**'dir — tüm harita tek c_k vektöründen türetilir. Bu yüzden
evrensellik, c_k vektörünü üç arka planda ölçüp karşılaştırarak test edildi
(576 çift simülasyonu yerine 64 simülasyon):

- boş arka plan (c_bare),
- seed-7 rastgele 10 μm RMS hizalama üzerinde (c_eff^A),
- seed-21 üzerinde (c_eff^B).

Prob: her k için +10 μm cos modu; c_k^eff = [f(P + prob) − f(P)]/A.

**Bulgu 1 — Aliasing simetrisi (yeni):** c_k = c_{24−k} **tam olarak**
(c₁=c₂₃, c₂=c₂₂=+1.974×10⁻⁴, c₃=c₂₁, …). Nedeni: N=24 FODO hücresinde
cos(2πkn/24) = cos(2π(24−k)n/24) — k ve 24−k modları quad konumlarında
**aynı fiziksel desendir**. Bağımsız mod sayısı 24 değil **12'dir**
(k=1..12; k=24 antisimetrik-DC eşleniği, c₂₄=+1.8×10⁻⁵). Spektrum k=12
etrafında simetrik bir "V" çizer: |c_k| minimumu k≈12'de (~1.8×10⁻⁶).

**Bulgu 2 — Evrensellik:**

| metrik | değer |
|---|---|
| korelasyon c_bare ↔ c_eff(A) | 0.9999 |
| korelasyon c_eff(A) ↔ c_eff(B) | 0.9984 |
| bağıl RMS sapma (A−B) | %7.2 |

Baskın modlarda (k=1..4, sahte EDM'nin ~%95'ini taşıyanlar) sapma %3–10
bandında. Büyük bağıl sapmalar yalnızca |c_k| ~ 3×10⁻⁶ olan zayıf modlarda
(k=8,9,15,16) görülür — bu modlarda prob yanıtı (~3×10⁻¹¹ rad/s) taban
çıkarma/eğim-fit gürültüsü mertebesindedir; sapma fizik değil ölçüm tabanıdır.

**Sonuçlar:**

1. Telafi haritası **seed'den bağımsızdır**: boş arka planda ölçülen c_k
   vektörü, herhangi bir rastgele hizalama deseninin üzerinde %3–10
   doğrulukla geçerlidir → trim reçetesi makine konfigürasyonuna değil
   kafese (tune'a) aittir; bir kez kalibre edilir.
2. %3–10'luk katsayı belirsizliği tek atışta bastırmayı ~10–30× ile
   sınırlar; daha derini (468× gibi) için ölçtükten sonra **iteratif trim**
   gerekir (ilk trim → kalan dSy/dt ölç → ikinci küçük trim).
3. Harita çalışmalarında k>12 kullanmaya gerek yok — alias. Telafi bütçesi
   k=3..12 negatif bandına dağıtılabilir; k=12 civarı en zayıf kaldıraç.

### 12.6 İteratif ölç-trimle döngüsü: desen bilgisi olmadan ~1000× bastırma

**Dosyalar:** `test_b_iterative_trim.py`, `test_b_iterative_trim.png`

Deneysel olarak gerçekçi şema: hizalama deseni **hiç bilinmeden**, yalnızca
ölçülen dSy/dt ve bir kez kalibre edilmiş c_k ile trim genliği hesaplanır:
A_trim = −f_ölçülen/c_trim. Taze desen (seed 99, haritanın hiç görmediği
konfigürasyon, RMS 10 μm), iki kol: (A) trim k=3 tek mod, (B) k=3+k=4
bölüşmüş.

| adım | kol A (k=3) | bastırma | kol B (k=3+4) | bastırma |
|---|---|---|---|---|
| 0 (trimsiz) | −2.405×10⁻¹⁰ | 1× | −2.405×10⁻¹⁰ | 1× |
| 1 | −1.09×10⁻¹¹ | 22× | −7.87×10⁻¹² | 31× |
| 2 | +2.5×10⁻¹⁴ | **9574×** | +3.3×10⁻¹³ | 719× |
| 3 | −2.8×10⁻¹³ | 872× | −1.7×10⁻¹³ | 1412× |

Bulgular:

1. **1. atış öngörüyle birebir uyumlu:** 22–31× — c_k'nin %3–10 evrensellik
   belirsizliği sınırı (§12.5 öngörüsü 10–30×).
2. **2. atış ölçüm tabanına iner:** kalan |f| ~ 10⁻¹³–10⁻¹⁴; bu seviye eğim
   fit gürültüsü tabanıdır (3. adım trimleri ~0.01 μm — artık gürültü
   trimleniyor, değerler taban etrafında salınıyor). Pratik bastırma sınırı
   bu konfigürasyonda ~**10³×**, yöntem değil ölçüm tabanı belirliyor.
3. **Genlik bütçesi ihmal edilebilir:** toplam trim 5–8 μm — hizalama
   bütçesi mertebesinde, onu aşmıyor.

**Strateji özeti:** c_k vektörünü bir kez kalibre et (24 simülasyon /
deneyde 12 prob ölçümü); her yeni makine konfigürasyonunda 2 ölçüm-trim
turu → sahte EDM ~1000× bastırılır, desen bilgisi ve yörünge
rekonstrüksiyonu gerekmeden. Daha derine inmek için tek gereken daha uzun
ölçüm (daha düşük dSy/dt tabanı).

### 12.7 Gerçekçi koşul: COD'ye oturtma olmadan (CO=False) — kesin sonuçlar

**Dosyalar:** `test_b_trim_realistic.py`, `test_b_mode_map_cofalse.py`,
`test_b_ck_cofalse.json`, `test_b_mode_map_cofalse.json`

Kick ile COD'ye oturtmanın pratik olmadığına karar verildiğinden
(injection_kick_raporu.md), tüm trim makinesi gerçekçi koşula taşındı:
parçacık **eksenden fırlatılır**, kapalı yörünge hiç aranmaz. Betatron
salınımı ölçümün doğal parçasıdır. Sonuçlar idealize koşuldan **daha iyi**:

**c_k tablosu (CO=False, k=1..12):** işaret yapısı aynen korunur
(k=1,2 pozitif; k≥3 negatif), değerler ~6 mertebe büyüktür çünkü gözlenebilir
artık betatron×misalignment kuplajının sürdüğü toplam dSy/dt'dir:

| k | c_k [rad/s/m] | | k | c_k [rad/s/m] |
|---|---|---|---|---|
| 1 | +23.46 | | 7 | −1.477 |
| 2 | +88.80 | | 8 | −0.983 |
| 3 | −22.54 | | 9 | −0.683 |
| 4 | −7.703 | | 10 | −0.502 |
| 5 | −3.937 | | 11 | −0.405 |
| 6 | −2.324 | | 12 | −0.374 |

**Tam doğrusallık:** k=2'de A=5/10/20 μm → f/A yayılımı **%0.0**.
Boş kafes tabanı tam 0 (eksen = ideal kafesin kapalı yörüngesi).

**Tam evrensellik:** c_k üç arka planda (boş / seed-7 / seed-21) ölçüldü:
tüm k'larda oran **1.000**, korelasyonlar **1.0000**, bağıl RMS sapma
**%0.00**. CO=True'da görülen %3–10 sapmanın kaynağı fizik değil,
Newton kapalı-yörünge bulucusunun sayısal gürültüsüymüş. CO=False
gözlenebiliri dy'de **tam lineerdir** → süperpozisyon kesin geçerli.

**Trim döngüsü (taze desen seed-99, f0 = −1.508×10⁻⁴):**

| adım | kol A (k=3) | bastırma | kol B (k=3+4) | bastırma |
|---|---|---|---|---|
| 1 | −7.7×10⁻¹² | **2.0×10⁷×** | −8.0×10⁻¹² | 1.9×10⁷× |
| 2 | +3.6×10⁻¹³ | 4.2×10⁸× | −9.5×10⁻¹³ | 1.6×10⁸× |
| 3 | +1.2×10⁻¹² | taban | +1.1×10⁻¹² | taban |

**Tek atışta 2×10⁷× bastırma** (CO=True'da 22–31× idi); 2. adım eğim-fit
tabanına (~10⁻¹²) iner. Trim bütçesi 6.7–10 μm.

**Açık soru:** trim, tek bir fırlatma koşulunun (eksen) dSy/dt'sini
sıfırlar. Farklı başlangıç koşullu parçacıklar için c_k(fırlatma)
katsayıları farklı olabilir → demet ortalamasında artık ne kadar?
(Sonraki test adayı.)

### 12.8 BPM ofseti/gürültüsünün trim döngüsüne etkisi: sınırlamaz

**Dosyalar:** `test_b_trim_bpm.py`, `test_b_trim_bpm.png` (CO=False)

Döngünün tek girdisi spin ölçümü → BPM verisi doğrudan kullanılmaz;
gerçekçi koşulda oturtma adımı da olmadığından "yanlış yörüngeye oturtma"
kanalı tanım gereği yok. Kalan iki dolaylı kanal test edildi:

**Kanal A — dSy/dt ölçüm gürültüsü** (σ_ε = 0.1·|f₀| = 1.5×10⁻⁵):
adım 1 ve 2 artıklarının σ_ε'a oranı tam **1.000** — döngü tabanı = σ_ε,
gürültü birikmez (her adım son gerçekleşmeyi trimler). Nihai derinlik
polarimetre istatistiğiyle ölçeklenir.

**Kanal B — statik aktüasyon hatası** (trim BPM-referanslı tümsekle
uygulanırsa; b₃ = 20μm ≈ σ_b=100μm'nin k=3 bileşeni):

| adım | dSy/dt [rad/s] | not |
|---|---|---|
| 0 | −1.508×10⁻⁴ | trimsiz |
| 1 | −4.507×10⁻⁴ | 3.0× kötüleşme (= c₃·b₃ tam) |
| 2 | +2.55×10⁻¹¹ | 5.9×10⁶× bastırma (diferansiyel) |
| 3 | +2.85×10⁻¹² | 5.3×10⁷× bastırma |

Statik hata kendini ele verir: döngü büyüyen sinyali ölçüp artımsal
trimle temizler (artımda statik ofset iptal). Maliyet: 1 ek tur.

**Sonuç:** BPM ofseti/gürültüsü trim yöntemini sınırlamaz — yörünge
rekonstrüksiyonunun aksine. İki yöntem tamamlayıcı: rekonstrüksiyon
hizalama hatasının *tanısı*, trim döngüsü BPM'lerden bağımsız *tedavisi*.

---

### 12.9 c_k'nin fırlatma koşuluna bağımlılığı: demet ortalaması geçerliliği

**Dosyalar:** `test_b_trim_launch_dep.py`, `test_b_trim_launch_dep.png`,
`test_b_launch_dep.json` (CO=False, k=2, A=10μm, t2=1ms)

**Soru:** Trim kalibrasyonu eksen fırlatmasında (y=py=0) yapılıyor. Gerçek
demetteki parçacıklar farklı başlangıç koşullarında (y₀, py₀). c_k bu fırlatma
koşuluna bağlı mı? Eksen-kalibre trim, demet ortalamasında artık bırakır mı?

---

**Bölüm 1 — Başlangıç konumu taraması** (py₀=0):

| y₀ [μm] | c_k/c_k(0) |
|---|---|
| 0 | 1.00000 |
| 100 | 1.00065 |
| 200 | 1.00130 |
| 500 | 1.00325 |
| 1000 | 1.00648 |
| 2000 | 1.01286 |

Maks sapma: **%1.29** (y₀=2mm'de). Lineer bağımlılık: ~%0.065/100μm.
Tipik demet boyutu σ_y~0.5mm için etki ~%0.33 → önemsiz.

---

**Bölüm 2 — Başlangıç açısı taraması** (y₀=0):

| α=py₀/pz₀ [mrad] | c_k/c_k(0) |
|---|---|
| 0.0 | 1.00000 |
| 0.1 | 1.03002 |
| 0.2 | 1.05777 |
| 0.5 | 1.12693 |
| 1.0 | 1.19216 |

Maks sapma: **%19.2** (α=1mrad). Lineer rejim: ~%3/0.1mrad.

**Fizik yorumu:** Bu etki fiziksel bir c_k değişimi değil, sonlu-t₂ ölçüm
yapaylığıdır. Büyük başlangıç açısı → büyük betatron genliği A_β → polyfit
eğiminde betatron salınımı kontaminasyonu. Gerçek trim, a_k → 0 olan
kafese uygulanır; tüm parçacıklar için seküler sürüş = 0. Ayrıca: demet
ortalamasında rastgele fazlar bu kontaminasyonu kısmen yok eder.

---

**Bölüm 3 — Demet ortalaması benzetimi** (N=30, σ_y=0.5mm, σ_α=0.2mrad):

| Büyüklük | Değer |
|---|---|
| Eksen c_k | +88.804 rad/s/m |
| Demet ortalama c_k | +88.390 rad/s/m |
| c_k std (parçacık başına) | 4.81 rad/s/m (%5.4) |
| Eksen−demet farkı | −%0.47 |
| Trim sonrası artık (eksen kalibre) | 4.14×10⁻⁶ rad/s |
| Bastırma (eksen kalibre) | **213×** |

Demet-ortalaması, eksen ölçümünden **%0.47 düşük** çıkıyor; bu fark esas
olarak büyük açılı parçacıkların negatif betatron katkısından kaynaklanıyor
(c_k(−α) < c_k(0) etkisi, c_k(+α) artışından baskın çıkıyor).

**Bastırma 213× ← alt sınır.** Bu rakam "eksen sinyalini her parçacıktan
çıkar" yaklaşımından geliyor. Gerçek trim kafes modifikasyonu yaptığından
(a_k → 0), tüm parçacıkların seküler sürüşü sıfırlanır; artık yalnızca
betatron kontaminasyonu (<1e-6 rad/s) kalır → **gerçek bastırma çok daha büyük**.

**Demet-ortalaması kalibrasyonu:** ⟨f⟩_demet ile c_k ölçülürse trim miktarı
tam olarak belirlenir → artık = 0. Eksen yerine demet ortalaması kullanmak
%0.47 sapmasını ortadan kaldırır.

**Sonuç:** c_k fiziksel olarak evrenseldir (tüm başlangıç koşulları için aynı).
Görünen fırlatma bağımlılığı sonlu-t₂ polyfit yapaylığı olup demet
ortalamasında kısmen iptal olur. Eksen kalibrasyonu %0.47 doğrulukla demet
ortalamasını temsil eder; daha soğuk demette bu doğruluk daha yüksek.
Trim yöntemi demet büyüklüğünden bağımsız geçerliliğini korur.

---

### 12.10 Rastgele desen + fazlı çok-modlu trim: faz problemi çözümü

**Dosyalar:** `test_b_random_trim.py`, `test_b_random_trim.png`,
`test_b_random_trim.json` (CO=False, t2=1ms)

**Senaryo:** Gerçek deneye en yakın durum — 48 kuadrupolde tam rastgele
hizalama hatası (seed=123, RMS=10μm). Desendeki her k modunun iki
kuadratürü var: Δy = Σ A_k·cos(2πkn/N − φ_k), φ_k fazları rastgele.
Şimdiye dek yalnız cos fazı (φ=0) kalibre edilmişti.

---

**Çift kuadratür kalibrasyon** (k=1..6, A=10μm):

| k | c_k^cos | c_k^sin | \|c_k\| | ψ_k [°] |
|---|---|---|---|---|
| 1 | +23.46 | −2.41 | 23.58 | −5.87 |
| 2 | +88.80 | −18.38 | 90.69 | −11.70 |
| 3 | −22.54 | +7.06 | 23.62 | 162.60 |
| 4 | −7.70 | +3.26 | 8.36 | 157.09 |
| 5 | −3.94 | +2.10 | 4.46 | 151.89 |
| 6 | −2.32 | +1.50 | 2.77 | 147.16 |

**Lineer faz rampası keşfi:** ψ_k ≈ −5.85°·k (mod 180°): −5.87, −11.70,
180−17.4, 180−22.9, 180−28.1, 180−32.8. Mod başına sabit faz eğimi,
gözlenebilirin sabit bir azimut referans noktasına (fırlatma/gözlem
azimutu civarı) ağırlıklandığını gösterir. 180° sıçraması k>Q_y işaret
değişiminin faza yansımasıdır.

**Faz modeli doğrulaması:** c₂(φ) = |c₂|·cos(φ−ψ₂) modeli, φ=45° ve 135°
ara ölçümlerinde **%0.000** sapmayla doğrulandı — tam sinüzoid, sistem
faz uzayında da tamamen lineer.

---

**Rastgele desen spektrumu ve f₀ tahmini:**

Desen k=1..12 modlarına ayrıştırıldı (A_k = 1–4 μm aralığında, fazlar
rastgele). Ölçülen f₀ = −2.503×10⁻⁴ rad/s. Kalibre k=1..6 katsayılarıyla
tahmin: −2.694×10⁻⁴ (fark %7.6 — kalibre edilmeyen k≥7 katkısı).
Baskın katkı k=2'den: −2.56×10⁻⁴ (desenin A₂=2.82μm @ 170.5° içeriği).

---

**Üç trim stratejisi yarışması** (ölç-trimle, 3 iterasyon):

| Strateji | Trim | Adım 1 | Adım 2–3 tabanı | Bütçe |
|---|---|---|---|---|
| A | k=2 @ ψ₂=−11.7° | 3.2×10⁷× | ~10⁻¹² rad/s | 2.76 μm |
| B | k=2+k=3 @ ψ'ler, bölüşmüş | 7.8×10⁷× | **~10⁻¹⁵ rad/s (10¹¹×)** | 4.38 μm |
| C | k=2 @ φ=0 (faz-cahil) | 1.6×10⁷× | ~4×10⁻¹⁵ rad/s | 2.82 μm |

**Bulgular:**

1. **Tek atış yeter:** Üç strateji de ilk adımda ≥10⁷× bastırır — sistem
   tam lineer olduğundan ölçülen skalerin iptali kesindir.
2. **Faz, skaler iptal için kritik DEĞİL:** Faz-cahil strateji C de
   çalışır; tek bedel bütçenin 1/cos(φ−ψ₂) = 1.02 katına çıkması
   (ψ₂ küçük olduğu için önemsiz). Faz yalnız ψ_k±90°'ye yaklaşınca
   tehlikeli olur (trim etkisizleşir, bütçe ıraksar).
3. **Çift kuadratür kalibrasyonun değeri:** ψ_k'yi bilmek (i) bütçeyi
   minimize eder, (ii) ölü fazdan kaçınmayı garantiler, (iii) desenin
   gerçek mod içeriğiyle karşılaştırma sağlar: strateji A trimi
   (2.76μm @ −11.7°) desenin k=2 içeriğinin (2.82μm @ 170.5°) neredeyse
   tam anti-paralelidir — trim fiilen k=2 kirliliğini fiziksel olarak
   söküyor.
4. **Bölüşmüş trim (B) en derine iner:** 2 mod kullanmak hem bütçe/mod'u
   düşürür hem ilk adımda en iyi bastırmayı verir.

**Sonuç:** Faz problemi pratikte iki kuadratür kalibrasyonla (mod başına
2 ölçüm) tamamen çözülür. Rastgele, çok-modlu, rastgele-fazlı gerçekçi
hizalama hatası tek ölç-trimle adımında ≥10⁷×, ikinci adımda ~10¹¹×
bastırılır; trim bütçesi 3–4.4 μm.

---

### 12.11 Yörünge-sürülü (BPM-referanslı) trim: kaba kademe, k-mod'suz

**Dosyalar:** `test_orbit_trim.py`, `test_orbit_trim.png`,
`test_orbit_trim.json` (CO=False, t2=1ms)

**Neden yörünge, neden spin değil?** Spin-sürülü trim toplam
dS_y/dt → 0 hedefler. Ama gerçek EDM de dS_y/dt içindedir; eğer trim
onu bastırırsa bilim sinyali gider. Yörünge ise EDM'ye **doğası gereği
kördür**: gerçek bir elektrik dipol momenti, kapalı yörüngeyi
milimetre de kıpırdatmaz. Dolayısıyla yörüngeyi rehber alarak yapılan
trim, EDM sinyaline dokunamaz. Ek avantaj: hızlıdır (BPM ölçümü
saniyeler sürer, polarimetre günler ister) ve vektöreldir (48 BPM
okuması mod fazını da verir, ayrıca faz kalibrasyonu gerekmez).

#### Senaryo

48 quad rastgele dikey kaçıklık, RMS = **100 μm** (seed=321). 48 BPM
statik ofset, RMS = **100 μm** (seed=777). k-modülasyon YOK; tek optik
konfigürasyon. Başlangıç spin hızı: **f₀ = −1.623×10⁻³ rad/s**.

Ölçülen mod genlik spektrumu (trim öncesi, μm cinsinden):

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Aₖ | 37.6 | 16.2 | 40.9 | **45.0** | 17.0 | 6.1 | 26.8 | 31.9 |

#### Yöntem

1. **Kalibrasyon:** Her k=1..6 modu için cos ve sin kuadratür (12 mod
   düğmesi toplamda), A=50μm genlikle uygulanır, 48 BPM'den tur-ortalamalı
   kapalı yörünge okunur → tepki matrisi O [48×12]. Bu ölçüm diferansiyel
   olduğundan statik BPM ofseti yok olur; sadece uygulanan mod değişimine
   yanıt görülür.

2. **Kestirim:** Gerçek desenin BPM okumasi y_ölç = y_gerçek + b (b: statik
   ofsetler). Least-squares fit O·â ≈ y_ölç → mod kestirimleri â. BPM
   ofseti b bilinmediği için â içinde sistematik yanlılık taşır; bu yanlılık
   kazancı küçük modlarda büyür (aşağıya bakınız).

3. **Trim:** Tüm quad kaçıklıklarına Δp = −Σₖ âₖ·mod_k eklenir. Ardından
   spin takibiyle doğrulama yapılır.

#### Yörünge kazancı neden rezonansla ilgilidir?

Bir mod-k kaçıklığı, kafesten geçerken transfer matrisinin rezonans
büyütme faktörüyle çarpılır. Bu faktör, betatron tununa (Q_y ≈ 2.68)
yakınlıkla belirlenir. k=2, Q_y'ye en yakın → 24.1× kazanç. k değeri
Q_y'den uzaklaştıkça kazanç düşer:

| k | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Yörünge kazancı | 6.2 | **24.1** | 6.3 | 2.26 | 1.24 | 0.79 |

100 μm BPM ofseti altında kestirim yanlılığı ε ≈ σ_b·√(2/N)/kazanç
(N=48 BPM): k=2'de **~0.9 μm**, k=4'te **~9 μm**, k=5'te **~16 μm**,
k=6'da **~26 μm**. Ölçülen değerler (O⁺b, JSON): k=2: 0.1–0.5 μm,
k=4: 0.2–1.0 μm, k=5: 6.7–7.6 μm, k=6: **~28 μm** — formülle uyumlu.
Bu yanlılıklar, hangi modların güvenle kestirilebileceğini doğrudan
belirler: yanlılık, desenin gerçek mod içeriğine (per kuadratür
~σ_q·√(2/N) ≈ 20 μm) yaklaştığında fit zarar vermeye başlar.

#### Dört varyant: adım adım spin değerleri

| Varyant | Fit | f sonrası | Bastırma |
|---|---|---|---|
| A | k=1..3 | **−6.72×10⁻⁵** rad/s | 24.2× |
| **C** | **k=1..4** | **+1.61×10⁻⁵** rad/s | **100.8×** ★ |
| D | k=1..5 | +1.37×10⁻⁴ rad/s | 11.9× |
| B | k=1..6 | +1.07×10⁻⁴ rad/s | 15.2× |

**Başlangıç:** f₀ = −1.623×10⁻³ rad/s.

**A trimi (k=1..3):** k=1,2,3 içerikleri 37.6/16.2/40.9 μm'den
3.8/7.0/9.2 μm'e düşer (5–11× azalma). Ancak k=4 = 45.0 μm **hiç
dokunulmaz**; trim bunu hedeflemez. Bunun sonucu: f = −6.72×10⁻⁵ rad/s.
Bu artık neredeyse tamamen k=4 katkısından gelir (c₄ × A₄ =
8.36 × 45×10⁻⁶ ≈ 3.8×10⁻⁴ rad/s magnitude, ancak faz ile ölçeklenmiş
katkı −6.7×10⁻⁵ olarak ölçülür).

**C trimi (k=1..4):** A'nın yaptığına ek olarak k=4'ü de hedefler.
k=4 içeriği 45.0 μm'den 6.0 μm'e düşer. Ortaya çıkan f = +1.61×10⁻⁵ rad/s.
Bu artık **k=5 ve üzeri** içerikten gelir; bunlar C tarafından hiç
dokunulmamıştır:

| k | Trim öncesi | C sonrası | Dokunuldu mu? |
|---|---|---|---|
| 1 | 37.6 μm | **3.8 μm** | ✓ ~10× azaldı |
| 2 | 16.2 μm | **6.9 μm** | ✓ ~2× azaldı |
| 3 | 40.9 μm | **9.2 μm** | ✓ ~4× azaldı |
| 4 | 45.0 μm | **6.0 μm** | ✓ ~7× azaldı |
| 5 | 17.0 μm | 17.0 μm | ✗ değişmedi |
| 6 | 6.1 μm | 6.1 μm | ✗ değişmedi |
| 7 | 26.8 μm | 26.8 μm | ✗ değişmedi |
| 8 | 31.9 μm | 31.9 μm | ✗ değişmedi |

k=2'nin diğerlerinden daha az düzeltilmesinin nedeni: kazancı çok yüksek
olduğundan BPM ofseti kestirime neredeyse hiç sızmaz — ancak BPM ofseti
mesafeyi değil yönü bozar, dolayısıyla kısmi silinme kaçınılmaz.

**D trimi (k=1..5):** k=5 kazancı = 1.24 → yanlılık ≈ 20.4/1.24 ≈ 16 μm
(ölçülen: kuadratür başına 7–12 μm hata). Gerçek A₅ = 17 μm ile aynı
mertebede: kestirim yarı yarıya ofset ürünü, trim k=5'i kısmen yanlış
yönde oynatır ve c₅=4.5 rad/s/m bu hatayı spine taşır. Sonuç:
f = +1.37×10⁻⁴ rad/s — C'den 8.5× DAHA KÖTÜ.

**B trimi (k=1..6):** k=6 kazancı = 0.79 < 1 → yanlılık ≈ 20.4/0.79 ≈
26 μm (ölçülen: cos −24.7, sin −26.9 μm; bileşke 36.5 μm). Gerçek A₆ =
6.1 μm'nin 4–6 katı: trim, ofset hayaletini hizalama hatası sanıp
sisteme enjekte eder; k=6 içeriği **6.1 μm'den 36.5 μm'e ÇIKAR**.
f = +1.07×10⁻⁴ rad/s.

**Sonuç:** C'nin sihiri, k=4'ün kazancının (2.26×) hâlâ yararlı eşiğin
üzerinde olmasından; k=5'inki (1.24×) ise gürültüyü bastırmak için
yeterli değildir. Kesim noktası kazanç ≈ 1.5–2 civarındadır.

#### Neden yörünge k≥7'ye kör ama spin etkileniyor?

Bu iki farklı fiziksel mekanizmadan kaynaklanır ve asla çelişkili değildir:

**Yörünge kazancı rezonans gerektiriyor.** Transfer matrisi yalnızca Q_y'ye
yakın modları büyütür. k=7 için tahmini kazanç ~0.45×; 100 μm BPM gürültüsü
ile bu mod tamamen gürültü içinde boğulur. BPM okuyamazsanız trim
yapamazsınız.

**c_k (spin kuplajı) rezonans gerektirmiyor.** dS_y/dt = Σₖ cₖ·Aₖ
formülündeki cₖ, halka boyunca radyal B alanının geometrik yol
integralidir. Q_y bu integrale girmez. Ölçülen değerler:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7* | 8* |
|---|---|---|---|---|---|---|---|---|
| |cₖ| (rad/s/m) | 23.6 | **90.7** | 23.6 | 8.4 | 4.5 | 2.8 | ~1.96 | ~1.52 |
| Yörünge kazancı | 6.2 | 24.1 | 6.3 | 2.26 | 1.24 | 0.79 | ~0.45 | ~0.35 |

(*k=7,8 için cₖ, k=5..6 eğrisinden ekstrapolasyonla elde edildi.)

**Sayısal kanıt:** C trimi sonrası k=7 = 26.8 μm, k=8 = 31.9 μm değişmedi.
Bunların spin katkısı:
- k=7: 1.96 × 2.68×10⁻⁵ ≤ **5.3×10⁻⁵ rad/s** (faz bağımlı üst sınır)
- k=8: 1.52 × 3.19×10⁻⁵ ≤ **4.8×10⁻⁵ rad/s**

Rastgele fazların kısmi iptali sonucu net C-artığı = +1.61×10⁻⁵ rad/s
mertebesine düşer. **Yörünge triminin tavanı, onun göremediği k≥7
içeriği tarafından belirlenir.** Bu, yörünge kademedinin sınırını kanıtlar
ve neden bir sonraki adım gerektiğini sayısal olarak açıklar.

#### Statik taban kanıtı

2. iterasyonda orbit güncellemesi: tam **0.0000 μm**. Aynı statik BPM
ofseti aynı yanlı kestirimi verir; yörünge döngüsü ilk adımda kendi
tabanına oturur, tekrar etmek hiçbir şeyi değiştirmez.

#### Büyük resim: kademeli mimari

| Kademe | Mekanizma | Bastırma | f sonrası |
|---|---|---|---|
| 0 (ham) | — | — | −1.623×10⁻³ rad/s |
| **1 Yörünge** | BPM-referanslı, EDM-kör, k=1..4 | **~100×** | **+1.61×10⁻⁵ rad/s** |
| 2 CW/CCW | Simetrik iptal | ~10³–10⁸× | ~10⁻⁸–10⁻¹³ rad/s |
| 3 Spin (son) | Yalnız-sistematik gözlenebilir | polarimetre sınırı | EDM tabanı |

Yörünge kademesi hem en hızlı hem de EDM açısından en güvenlidir.
Kalan ~10⁻⁵ rad/s, k≥7 içeriğinden kaynaklandığı için CW/CCW yöntemi
tarafından doğal olarak iptal edilir (her iki ışın da aynı hizalama
hatasını görür).

> **⚠ Evrensellik uyarısı (bkz. §12.12):** Bu bölümdeki bastırma
> oranları (A=24×, C=100.8×) seed=321/777 çiftine aittir. 4 yeni seed
> çiftiyle yapılan tarama, 100× değerinin **faz şansı** olduğunu ve
> A/C/D/B sıralamasının seed'den seed'e değiştiğini gösterdi. Evrensel
> olan: k=1..4 yörünge içeriğinin birkaç μm'e temizlenmesi ve analitik
> kazanç yasası. Spin artığı ise fit bazının DIŞINDA kalan içerikle
> belirlenir (~2.5×10⁻⁴ rad/s taban, 5-seed RMS).

### 12.12 Evrensellik testi: genişlik haritası seed'e bağlı mı?

**Dosyalar:** `test_orbit_trim_seeds.py`, `test_orbit_trim_seeds.json`,
`test_orbit_trim_seeds.png` (4 yeni seed çifti, kalibrasyon O matrisi
§12.11'den yeniden kullanıldı — kafese aittir, desene değil)

**Soru:** "C (k=1..4) optimal, k≥5 fit zararlı" bulgusu evrensel mi?

**Sonuç tablosu (mutlak artıklar, rad/s):**

| seed | f₀ | A (k≤3) | C (k≤4) | D (k≤5) | B (k≤6) |
|---|---|---|---|---|---|
| 101/201 | +1.28×10⁻³ | −4.3×10⁻⁴ | −4.4×10⁻⁴ | −3.8×10⁻⁴ | −3.8×10⁻⁴ |
| 102/202 | −2.0×10⁻⁴ | −1.6×10⁻⁶ | +1.4×10⁻⁴ | +2.8×10⁻⁴ | +1.9×10⁻⁴ |
| 103/203 | +1.44×10⁻³ | +6.5×10⁻⁵ | −2.0×10⁻⁴ | −1.5×10⁻⁴ | −2.4×10⁻⁴ |
| 104/204 | −9.1×10⁻⁴ | +4.0×10⁻⁴ | +3.7×10⁻⁴ | +2.1×10⁻⁴ | +3.1×10⁻⁴ |
| 321/777 (orij.) | −1.62×10⁻³ | −6.7×10⁻⁵ | +1.6×10⁻⁵ | +1.4×10⁻⁴ | +1.1×10⁻⁴ |
| **RMS (5 seed)** | | **2.6×10⁻⁴** | **2.8×10⁻⁴** | **2.5×10⁻⁴** | **2.6×10⁻⁴** |

**Bulgu 1 — Sıralama evrensel DEĞİL:** A/C/D/B artıkları istatistiksel
olarak ayırt edilemez (hepsi ~2.5×10⁻⁴ RMS). Seed 321'deki "C=100.8×"
sonucu faz şansıydı. Tipik (medyan) bastırma ~5×; oran 1.4×–123×
arasında savruluyor çünkü payda (f₀) seed'e göre 8× değişiyor.

**Bulgu 2 — Artığın gerçek kaynağı fit bazının DIŞI:** Her seed için
artık ≈ (f₀ − antisym k≤6 tahmini) eşitliği sağlanıyor (örn. seed 101:
artık −4.4×10⁻⁴, baz-dışı katkı −5.1×10⁻⁴). 48 quad'lık desenin yalnızca
25 serbestlik derecesi antisym k=0..12 bazında; geri kalan 23 boyut
(simetrik QF/QD kombinasyonları, 57–90 μm RMS taşıyor) hem yörünge
fit'ine hem spektrum analizine görünmezdir ama spine bağlanır. Seed
321'de bu baz-dışı katkı tesadüfen küçüktü.

**Bulgu 3 — k=5 fit'inin kararsızlığı analitik eşiğin kendisidir:**
k=5 kazancı (1.24), eşik değeri σ_b/σ_q = 1.0'ın hemen üstünde —
sınırda olan mod, seed'e göre bazen kazandırır bazen kaybettirir
(D, 4 seed'in 2'sinde C'den iyi, 2'sinde kötü). k≤4 (G≥2.26) her
seed'de güvenli, k=6 (G=0.79) hiçbir seed'de belirgin kazandırmıyor.

#### Analitik kazanç yasası ve eşik formülü

Ölçülen 6 kazanç, klasik düz-yaklaşım kapalı yörünge harmonik yanıtına
%0.6 RMS sapmayla oturur:

$$G_k = \frac{C}{|Q_\mathrm{eff}^2 - k^2|}, \qquad C = 24.8,\;
Q_\mathrm{eff}^2 = 5.03\;(Q_\mathrm{eff}=2.243)$$

| k | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Ölçülen | 6.20 | 24.08 | 6.29 | 2.26 | 1.24 | 0.794 |
| Model | 6.16 | 24.09 | 6.25 | 2.26 | 1.24 | 0.801 |

(Not: Q_eff = 2.24, FFT ile ölçülen betatron tunu Q_y = 2.68'den
farklıdır — formülün biçimi düz-yaklaşım COD yanıtıdır; etkin Q, gerçek
kafesin β-modülasyonunu soğurur.)

Mod k'yı fit etmenin kazandırma koşulu: ofset yanlılığı < gerçek içerik:

$$\varepsilon_k = \frac{\sigma_b\sqrt{2/N}}{G_k} < A_k \approx
\sigma_q\sqrt{2/N} \;\Longrightarrow\; \boxed{G_k > \sigma_b/\sigma_q}$$

√(2/N) çarpanları sadeleşir — **eşik yalnız σ_b/σ_q oranına bağlıdır.**
Kazanç yasasıyla birleşince: k_max² < Q_eff² + C·(σ_q/σ_b).

| σ_b/σ_q | 2.0 | 1.0 | 0.5 | 0.1 |
|---|---|---|---|---|
| k_max | 4.2 | 5.5 | 7.4 | 15.9 |

σ_b = σ_q = 100 μm'de k_max = 5.5: k=4 güvenli, k=5 sınırda, k=6 zararlı
— simülasyonun bulduğuyla birebir.

**Gerçek hızlandırıcıda bağlayıcılık:** Simülasyondaki sayılar değil,
prosedür bağlayıcıdır. G_k devreye almada aynı diferansiyel kalibrasyonla
yerinde ölçülür; eşik o makinenin ölçülen G_k ve σ_b değerleriyle
yeniden hesaplanır. Beam-based alignment ile σ_b 10 μm'e inerse
k_max ≈ 16 olur — tüm antisym bazı güvenle fit edilebilir.

#### 10⁻⁵ hedefi için sonuç

Deney gereksinimi (CW/CCW + quad-flip ~10⁴× ek bastırma ile toplam
<10⁻⁹): yörünge kademesinin ~10⁻⁵ rad/s'ye inmesi yeterli. Bu testin
gösterdiği: σ_b = 100 μm ile 12-düğmeli antisym baz bunu **garanti
edemez** (taban 2.5×10⁻⁴). Kapanması gereken açık iki yönlüdür:
1. **σ_b'yi düşür** (beam-based alignment) → k_max büyür,
2. **Bazı genişlet** — antisym k≤12'nin ötesinde, baz-dışı 23 boyutu
   (simetrik kombinasyonlar) gören düğmeler eklenmelidir; aksi hâlde
   57–90 μm'lik görünmez içerik spin artığını domine etmeye devam eder.

### 12.13 Radyal başlangıç polarizasyonu: hassasiyet artar mı?

**Dosyalar:** `test_radial_spin.py`, `test_radial_spin.json`,
`test_radial_spin.png` (CO=False, t2=1ms)

**Soru:** Spin-sürülü trimde başlangıç polarizasyonu boylamsal yerine
radyal olsa sistem daha hassas olur mu?

**Teorik beklenti:** Hem gerçek EDM dönüşü (radyal E) hem sahte EDM
dönüşü (dikey COD × quad gradyanı → radyal B_x) radyal eksen
etrafındadır. Thomas-BMT'de dS_y/dt = Ω_z·S_x − Ω_x·S_z: boylamsal spin
(S_z=1) Ω_x'i tam görür; radyal spin (S_x=1, S_z=0) Ω_x'e birinci
mertebede kördür, ama Ω_z'yi (boylamsal presesyon ekseni) görür.

**Ölçüm (4 koşum):**

| Koşum | Spin | Kaynak | dS_y/dt [rad/s] |
|---|---|---|---|
| 1 | boylamsal | kaçıklık (100μm) | −1.62×10⁻³ |
| 2 | radyal | kaçıklık (100μm) | **+8.2×10⁻⁵** |
| 3 | boylamsal | saf EDM (η=1.88×10⁻¹⁵) | −9.66×10⁻¹⁰ |
| 4 | radyal | saf EDM | **−2.8×10⁻¹⁶** |

**Yanıt: hassasiyet ARTMAZ — EDM sinyali 3.5 milyon kat bastırılır**
(oran 2.9×10⁻⁷). Radyal demet EDM'ye fiilen kördür.

**Ama beklenmedik bir armağan var:** Kaçıklık sinyali sıfır ÇIKMADI —
boylamsalın %5'i (+8.2×10⁻⁵) kaldı. Mekanizma: dikey betatron/yörünge
hızı (β_y) × radyal E alanı → boylamsal Ω_z bileşeni → dS_y/dt = Ω_z·S_x.
Yani radyal demet, **EDM'den arındırılmış bir hizalama-sistematiği
kanalıdır**: kademeli mimarinin 3. basamağında aranan "yalnız-sistematik
gözlenebilir"in somut bir adayı. Spin-sürülü trim radyal demetle
yapılırsa EDM sinyalini silme riski olmaz (duyarlılık haritası farklı
olduğundan kalibrasyonu ayrıca yapılmalıdır).

### 12.14 Mod korelasyonu: fit neden 4'te kesiliyor — korelasyon mu, eşik mi?

**Dosyalar:** `test_orbit_mode_correlation.py`,
`test_orbit_mode_correlation.json`, `test_orbit_mode_correlation.png`
(kalibrasyon k=7..12'ye genişletildi: 13 ek yörünge koşumu)

**Soru:** Modların yörünge uzayındaki korelasyonu ölçülebilir mi? Fit'in
k=4'te kesilmesinin nedeni korelasyon mu?

**1. Gram matrisi — modlar yörüngede neredeyse diktir:** 24 düğmenin
(k=1..12 × cos/sin) normalize yörünge parmak izleri arasında
korelasyon C_ij = ô_i·ô_j ölçüldü. k≤6 bloğunda max |köşegen-dışı| =
**0.011** — yani %1. LSQ bu yüzden modları tek atışta temiz ayırır;
**fit kesiminin nedeni modlar arası korelasyon DEĞİLDİR.** (Not: k=12
sin düğmesi Nyquist'te özdeş sıfırdır — 24 hücreli kafeste
sin(πn) = 0.)

**2. Sızıntı — fit dışı antisym modların etkisi ihmal edilebilir:**
L = O_C⁺·O_dışı (fit dışı k=5..12 modlarının C-fit kestirimlerine
izdüşümü): max |L| = **0.002**. Fit dışı 30μm'lik bir mod, kestirime
en fazla 0.06μm sızar — ölçülen kestirim hatalarını (0.4–8μm)
açıklayamaz.

**3. Hata muhasebesi — kapanmıyor, ve bu bilgilendirici:**

| düğme | ölçülen hata | ofset O⁺b | sızıntı L·a |
|---|---|---|---|
| k2 sin | 6.86 μm | −0.51 | +0.01 |
| k3 sin | 7.93 μm | −1.41 | +0.02 |
| k4 sin | 4.58 μm | −0.96 | +0.03 |

Ofset + sızıntı, ölçülen hatanın küçük bir kısmı. Kalan baskın bileşen:
desenin **antisym baz DIŞINDAKİ** (simetrik QF/QD) içeriğinin yörünge
izdüşümü — §12.12'deki baz-dışı bulgusunun kestirim-uzayındaki
görüntüsü. Yani kestirim hatası bir korelasyon problemi değil, bir
**baz eksikliği** problemi.

**4. Kazanç yasası k=7..12'de ÖNGÖRÜ olarak doğrulandı:** Yasa
(G_k = 24.8/|5.03−k²|) k≤6'ya oturtulmuştu; k=7..12 ölçümleri saf
öngörü testidir:

| k | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|
| Ölçülen | 0.558 | 0.417 | 0.328 | 0.273 | 0.242 | 0.164 |
| Yasa (öngörü) | 0.564 | 0.421 | 0.326 | 0.261 | 0.214 | 0.178 |

k≤9'da sapma ≤%1; k=10..12'de %4–12 (kazançlar sayısal tabana yakın).

**Sonuç:** Fit'in 4'te kesilmesi tamamen **eşik** meselesidir
(G_k > σ_b/σ_q, §12.12), korelasyon meselesi değildir. Spin
gözlenebiliriyle kontrast: spin tek skalerdir — orada "Gram matrisi"
rank-1'dir, bütün modlar tek sayıya çöker ve korelasyon tamdır. Tek
başına k=2'yi spin geri-beslemesiyle bastırmanın işe yaramamasının
(hatta |f|'yi artırabilmesinin) nedeni budur: f₀ = Σc_k·a_k toplamında
k=2 terimi diğerleriyle kısmen iptal hâlindeyse, yalnız k=2'yi çıkarmak
kalan toplamı büyütebilir. Yörünge fit'i ise 48 bileşenli vektörde
çalıştığından modları aynı anda ve neredeyse dik biçimde ayırır.

---

### 12.15 Spin tabanının kaynağı kanıtlandı: simetrik QF/QD içeriği

**Dosyalar:** `test_symm_vs_antisym.py` (ayrıştırma),
`test_symm_basis_fit.py` (genişletilmiş baz denemesi)

**Ayrıştırma testi:** Seed=321 deseninin 48 bileşeni, antisimetrik baz
(k=0..12, 25 boyut) üzerine dik izdüşümle iki parçaya ayrıldı
(iç çarpım ~10⁻²⁴, ayrıştırma temiz):

| desen | desen RMS [μm] | COD RMS [μm] | dSy/dt [rad/s] |
|---|---|---|---|
| tam | 89.5 | 628 | −1.62×10⁻³ |
| antisim (25 boy.) | 69.8 | 550 | −1.52×10⁻³ |
| **simetrik (23 boy.)** | 54.9 | **165** | **−1.02×10⁻⁴** |

İki düzeltme ve bir kanıt:

1. **Simetrik içerik yörüngeye kör DEĞİL:** COD kazancı ~3.0
   (antisimin ~1/2.6'sı ama sıfır değil). QF/QD tekme iptali kısmidir
   çünkü iki quad arasındaki betatron faz ilerlemesi tam iptale izin
   vermez.
2. **Spin, simetrik içeriği ~12× bastırılmış görür:** birim kaçıklık
   başına kuplaj antisimde 2.2×10⁻⁵, simetrikte 1.9×10⁻⁶ rad/s/μm.
   Kalıntıyı deflektör arclarındaki geometrik spin fazı
   (aγ·Φ_def ≈ 0.23 rad/yarı-hücre) bırakır.
3. **Floor kanıtı:** simetrik parçanın tek başına ürettiği
   1.0×10⁻⁴ rad/s, §12.12'deki 5-seed artığının (~2.5×10⁻⁴ RMS) ana
   bileşenidir.

**Genişletilmiş baz denemesi — başarısız ve öğretici:** Madem simetrik
içerik yörüngede görünüyor (kazanç ~3), simetrik k=1..4 modları
kalibrasyona eklenip 16-sütunlu fit denendi. Sonuç (seed=321):

| koşul | \|dSy/dt\| [rad/s] | bastırma |
|---|---|---|
| trim öncesi | 1.62×10⁻³ | — |
| yalnız antisim C (k=1..4) | 1.61×10⁻⁵ | 101× |
| **antisim + simetrik (k=1..4)** | **1.42×10⁻⁴** | **11× (9× KÖTÜ)** |

Neden: simetrik mod kazançları (k=1: 0.98, k=2: 4.73, k=3: 1.58,
k=4: 0.72) eşik G_k > σ_b/σ_q = 1.0'ın altında veya hemen üstünde.
Eşik altı sütun eklemek fit'e BPM ofsetini enjekte eder: kestirilen
simetrik genliklerin RMS'i 136 μm — gerçek içerik 55 μm. Fit gürültüyü
"düzeltir", spin artığı büyür. §12.12'deki eşik yasası burada da
bağlayıcıdır: **hangi modu fit edebileceğini korelasyon değil, kazanç
belirler.**

**Pratik sonuç:** σ_b = 100 μm BPM ofsetiyle BPM-tabanlı trim
2.5×10⁻⁴ rad/s tabanına demirler. Omarov zinciri (CW+CCW+quad-flip)
10⁻⁵ → 10⁻⁹ götürdüğünden zincire teslim hedefi 10⁻⁵'tir — aradaki
~25× boşluk yalnız yörüngeyle kapanmaz. Kapatma adayları:
(a) boylamsal spin-trim son kademesi (tüm 48 DOF'u görür, c_k hiçbir
modda sıfır değil), (b) BBA ile σ_b'yi düşürmek, (c) kafes tasarımını
değiştirmek (§12.16).

---

### 12.16 Kafes topolojisi denemeleri: iptal mekanizmasını kaynağında kırmak

**Dosyalar:** `integrator2.cpp`, `integrator2.py`, `lib_integrator2.so`
(bağımsız ikinci motor — eski kod dokunulmadı), `test_new_topology.py`,
`find_stable_gradient.py`

**Fikir 1 — deflektörü quad çiftinin dışına almak:** Mevcut hücrede
(ARC–d–QF–d–ARC–d–QD–d) QF ile QD arasında deflektör vardır; spin QF
tekmesinden sonra aγ·Φ_def kadar döner, QD tekmesi döndürülmüş çerçevede
gelir → simetrik iptal bozulur. Yeni hücre QF–d–QD–d–ARC(2Φ_def)–d(2d)
spin iptalini QF-QD arasında saf drift ile korur. 6-elemanlı hücre
`integrator2.cpp`'ye kodlandı (hücre uzunluğu ve çevre aynı).

**Sonuç — mevcut gradyanla kararsız:** g₀=0.2 T/m ile yörünge patlar
(kazançlar ~10⁷, f₀ ≈ −441 rad/s). İnce mercek tek-hücre transfer
matrisi analizi (`find_stable_gradient.py`) nedeni netleştirdi:

| topoloji | kararlı g aralığı | Q_y≈2.68 için g |
|---|---|---|
| orijinal FODO | 0.01–0.70 T/m | 0.241 T/m |
| yeni (QF-d-QD-d-ARC-d) | 0.01–**1.45** T/m | **0.498 T/m** |

Yeni topolojinin kararlılık bölgesi aslında daha geniş; ancak aynı tonu
vermek ~2× güçlü quad ister ve fırlatma/kapalı-yörünge koşulları yeni
kafese göre yeniden türetilmelidir. Bu tam bir kafes yeniden tasarımı
problemi — şimdilik rafa kaldırıldı.

**Fikir 2 — tek tip quad (yalnız QD): reddedildi.** Gradyan işareti
alternasyonu kaldırılırsa simetrik/antisim iptal ayrımı kökten yok olur
ve 48 DOF'un tamamı yörüngede görünür hale gelirdi. Ancak quad-flip
tekniği tam da bu alternasyona dayanır (flip'te sahte EDM işaret
değiştirir, gerçek EDM değişmez); tek tip quad ile flip simetrisi
ortadan kalkar ve 10⁻⁵→10⁻⁹ kademesinin en güçlü aracı kaybedilir.
Kazanılandan fazlası kaybedilir — bu yol kapalı.

---

## 13. dy-only Doğrusallığının Kaynağı ve Omarov Kuadratiğinin Yeri (nihai tanı) <a name="13-nihai-tani"></a>

> **Bağlam:** "Omarov makalesinde sahte EDM quad misalignment'ın *karesiyle*
> değişir; bizde ise doğrusal terim baskın — kuadratik nerede gizli?" sorusu
> sistematik olarak araştırıldı. Sonuç: kuadratik *gizlenmemiş*; yanlış kanal
> uyarılıyordu. Aşağıda dört yanlış hipotezin elenmesi ve doğru cevap.
> Ölçümler `claude/awesome-babbage-nmi6w9` oturumunda, **CO=True + madde 2
> model fit** (`measure_dSy_dt_model`, `false_edm_mode_scan.py`) ve **antitetik
> demet-ortalaması** ile yapıldı; t2=0.5 ms, k=2 FODO-antisim mod ya da rastgele
> desen.

### 13.1 Ölçüm artefaktı: betatron sızıntısı (gerçek ama AYRI sorun)

CO=False tek parçacıkta stroboskopik $S_y(n)$'in düz `polyfit(...,1)` eğimi
seküler değildir; içine **betatron salınımı** sızar. Ölçülen yapı (k=2, 10μm):

- Baskın frekans **0.30/tur** (periyot 3.3 tur) = dikey betatron tune
  $Q_y=2.68$'in stroboskopik alias'ı; genlik $\mathrm{std}(S_y)\approx1.6\times10^{-5}$
  rad, betatron genliğiyle (∝$A_\beta$) orantılı.
- Düz fit bu salınımın kapanmamış kuyruğunu seküler eğim sanır → yapay
  ∝misalignment terim. t2 ile küçülür: 1ms→1.9×10⁻⁴, 3ms→3.8×10⁻⁵,
  10ms→1.0×10⁻⁷ (640×, 370× düşüş — gerçek seküler t2-bağımsız olurdu).

**Spin tune DEĞİL:** ölçüldü — $S_x$ 1 ms'de yalnız ~0.7 µrad hareket eder,
$S_z=-1.0\ldots-0.9999999991$ (donuk, $1-|S_z|\sim10^{-9}$). Spin-tune faz
ilerlemesi mrad-altı → seküler eğime katkısı ihmal edilebilir. *(Bu belgenin
erken taslaklarındaki "yavaş spin-tune dalgası" yorumu yanlıştı; kontaminasyon
hızlı betatrondur — §3'teki betatron kirlenmesinin ta kendisi.)*

**Çözüm:** madde 2 model fit salınımı açıkça çıkarır (CO=False'ta t2 ile
iyileşir: 0.5ms'de 5.5×, 3ms'de 84×); CO=True salınımı kaynağında öldürür;
demet-ortalaması faz iptaliyle yok eder.

### 13.2 İdeal parçacık (CO=True) GERÇEKTEN doğrusal — thick-lens DEĞİL

Betatron yokken (CO=True, salınım yok, model=düz fit aynı) ölçülen seküler
eğim temizdir ve doğrusaldır:

| A [μm] (k=2) | dSy/dt [rad/s] | yerel α |
|---|---|---|
| 1→2 | 1.58→3.12 ×10⁻¹⁰ | 0.98 |
| 5→10 | 0.76→1.71 ×10⁻⁹ | 1.16 |
| 10→20 | →4.28×10⁻⁹ | 1.32 |
| 20→50 | →2.64×10⁻⁸ | 1.98 |
| 50→100 | →1.65×10⁻⁷ | 2.64 |

Küçük/gerçekçi genlikte (≲20μm) **α≈1 (doğrusal)**; kuadratik ancak büyük
misalignment'ta (kesişim $A^*\approx20\mu$m) devreye girer. Küçük-A doğrusal
katsayısı $c_1\approx+1.6\times10^{-4}$ rad/s/m.

**Elenen hipotezler:**

- **Thick-lens (kalın mercek): ÇÜRÜTÜLDÜ.** $K=G\cdot L$ ve cell uzunluğu sabit
  tutulup $L$ 0.40→0.05 m (8×) inceltildiğinde α **sabit** (1.24,1.18,1.19,1.19),
  genlik neredeyse değişmedi. Doğrusal terim sonlu quad uzunluğundan gelmiyor.
- **Coherent-vs-rastgele: ÇÜRÜTÜLDÜ.** Rastgele desen (CO=True, σ=2..40μm,
  5 tohum) da **α≈1.03** (medyan). İşaret-iptali kuadratiği baskın kılmıyor.

**Gerçek kaynak:** §2'deki $\sum_j G_j y_{\mathrm{CO},j}=0$ iptali, $y_{\mathrm{CO}}$'nun
misalignment ile *aynı* $(-1)^j\cos$ desenine sahip olduğunu varsayar; oysa
gerçek kapalı yörünge kick'lerin $(I-M)^{-1}$ ile **betatron-fazı boyunca
yayılmış** hâlidir → desen bozulur, iptal birinci derecede kapanmaz →
indirgenemez $c_1 A$ terimi (rastgele desende: net radyal alan
$\sum_j(-1)^j dy_j\neq0$).

### 13.3 İki BÜYÜK ikinci-derece kanal — dy-only'nin 100–2300×'i

dy-only ideal seküler (~10⁻⁹) baskın değildir. Asıl fizik iki kanalda:

**(a) Emittans kanalı (demet-bağımlı).** Antitetik-fazlı demet
(σ_y=0.5mm, σ_y'=0.2mrad) ile $\langle S_y\rangle$'den ölçülen seküler:

| | dSy/dt [rad/s] |
|---|---|
| CO=True ideal $s_0$ | 1.71×10⁻⁹ |
| Demet ort. $s_{\rm beam}$ | **3.77×10⁻⁷** (≈**220×**) |

Betatron genliğiyle ∝$A_\beta^2$ (ölçülen üs 2.19); misalignment'ta ∝$A^{1.0}$.
CO=True bu kanalı tamamen kaçırır — bkz. §13.5/açık sorun.

**(b) dx·dy geometrik-faz çapraz kanalı (Omarov σ²).** İki düzlem birlikte
kaçıkken (rastgele dx ve dy, rms σ; CO=True, model fit):

| tohum | 2μm | 10μm | 20μm | α |
|---|---|---|---|---|
| 0 | 1.17e-8 | 2.92e-7 | 1.17e-6 | **2.00** |
| 1 | 2.28e-9 | 5.56e-8 | 2.16e-7 | 1.98 |
| 2 | 1.85e-8 | 4.61e-7 | 1.85e-6 | 2.00 |

**Medyan α = 2.00 — tam kuadratik.** Kontrast (σ=10μm): dy-only −1.3×10⁻¹⁰,
**dx-only = +0.000 (tam sıfır)**, dx+dy = +2.9×10⁻⁷ (dy-only'nin ~2300×'i),
çapraz = both−dy−dx = +2.9×10⁻⁷ (sinyalin tamamı).

Mekanizma: $dy\to B_x$ (spin x-ekseni etrafında döner), $dx\to B_y$
(spin y-ekseni etrafında döner). Tek başına $dx$ dikey spini etkilemez
(düzlemler ayrık). İki farklı eksen etrafındaki **komütatif olmayan** dönmeler
→ geometrik (Berry) faz → net $S_y\propto dx\cdot dy$; $dx\sim dy\sim\sigma$
ise **∝σ²**. Bu, §8'deki bilinear $x\cdot y$ çapraz teriminin σ-ölçeklenmiş
hâlidir (§8 sabit-dy/değişken-x_0 ile doğrusal görmüştü; iki düzlem birlikte
ölçeklenince kuadratik).

### 13.4 Omarov kuadratiğinin yeri — nihai cevap

Omarov'un misalignment-kuadratik sahte EDM'i **dx·dy geometrik-faz
kanalıdır** ve onu görmek için **her iki düzlemin aynı anda kaçık** olması
gerekir. Proje şimdiye dek **yalnız dy** test ettiği için bu kanalı hiç
uyarmadı; ölçtüğü dy-only sinyali baskın olmayan, birinci-derece radyal-alan
kanalıdır (doğrusal, ~10⁻⁹). **Kuadratik gizlenmemişti; dikey-tek testlerin
hiç dokunmadığı dx·dy kanalında duruyordu.** dy-only doğrusallığı Omarov ile
çelişmez.

**Dört hipotezin özeti:**

| Hipotez | Sonuç |
|---|---|
| Fit artefaktı (betatron) tek başına açıklar | Kısmen — gerçek ama ayrı; CO=True'da yok, yine doğrusal |
| Thick-lens | Çürütüldü (8× inceltmede α sabit) |
| Coherent vs rastgele | Çürütüldü (rastgele de α≈1) |
| Emittans kanalı misalignment-kuadratik | Çürütüldü (∝A¹) |
| **dx·dy geometrik-faz** | **Doğrulandı (α=2.00)** |

### 13.5 Çıkarımlar ve açık sorunlar

1. **dy-only çatı dominant fiziği kaçırıyor.** `false_edm_mode_scan.py`,
   `c_k` tabloları (§12), trim döngüsü — hepsi dy-only. Gerçek sahte EDM
   iki büyük kanalca (dx·dy ∝σ², emittans ∝$A_\beta^2$) yönetiliyor, her ikisi
   de ~10⁻⁷. Analiz **dx+dy birlikte** ve **demet-ortalamalı** olacak şekilde
   genişletilmelidir.
2. **CO=True yetersiz:** yalnız ideal parçacığın $s_0$'ını verir; ideal olmayan
   (sonlu betatron) parçacıkların emittans kanalını (220×) kaçırır. Fiziksel
   gözlenebilir $d\langle S_y\rangle/dt$ = demet ortalamasıdır.
3. **Omarov mutlak değer teyidi açık:** buradaki α üsleri ve oranlar Omarov'un
   *mertebeleriyle* (~10⁻⁵/10⁻⁹) ölçek uyumunu doğrulamak için makale
   parametreleriyle (kafes, σ, emittans, t2) birebir eşleme gerekir
   (YAPILACAKLAR; kütüphane erişimine bağlı).

---

### 13.6 dx·dy kanalında demet: ideal'e eşit + örnekleme artığı dersi <a name="13-6"></a>

**Soru:** dx·dy α=2.00 (§13.3b) **CO=True** ile ölçüldü — yani tek ideal parçacık.
Gerçek demette (sonlu betatron genlikli ideal-olmayan parçacıklar) bu kanal
büyür mü? (dy-only'de emittans kanalı CO=True'yu 220× büyütüyordu, §13.3a.)

**Yöntem — 4D kapalı yörünge + antitetik demet:** dx hem yatay hem (skew yokken)
yalnız yatay kapalı yörünge üretir, dy yalnız dikey. Düzlemler orbit düzeyinde
ayrık olduğundan yatay CO (dx'ten) ve dikey CO (dy'den) bağımsız bulunur; parçacık
4D kapalı yörüngede $(x_{\rm CO},x'_{\rm CO},y_{\rm CO},y'_{\rm CO})$ fırlatılır.
Demet, bu merkez etrafında rastgele $(\Delta x,\Delta x',\Delta y,\Delta y')$
örnekleriyle temsil edilir; $\langle S_y\rangle$ alınıp madde 2 ile seküler eğim
çıkarılır.

**İLK SONUÇ YANILTICIYDI — kritik tuzak:** 2-katlı antitetik demet (her çekiliş
için $+v$ ve $-v$) **σ^0.17 düz, ~1.5×10⁻⁵** verdi. Büyüklük Omarov'la örtüşüyordu
ama **misalignment'tan bağımsız** olması Omarov'un kuadratiğiyle çelişir → şüpheli.

**TANI:** Düz sinyal, sonlu-N **$\langle\Delta x\cdot\Delta y\rangle$ örnekleme
artığıdır.** Gerçek korelasyonsuz demette $\langle\Delta x\Delta y\rangle=0$, ama
N örnekte örnek ortalaması $\sim\sigma_x\sigma_y/\sqrt N\neq0$. Bu kalıntı,
betatron'un kendi geometrik fazını ($\propto\Delta x\Delta y$, misalignment'tan
bağımsız) besler. **2-katlı antitetik bunu iptal etmez**: $\Delta x\Delta y$
çifttir, $(-\Delta x)(-\Delta y)=\Delta x\Delta y$.

**ÇÖZÜM — 4-katlı çift-düzlem simetri:** x-düzlemi ve y-düzlemi işaretleri
**bağımsız** çevrilir. Her çekiliş için 4 parçacık (spin hepsinde AYNI,
boylamsal $(0,0,-1)$ — spin simetriye sokulMAZ):

| # | x | x' | y | y' |
|---|---|---|---|---|
| 1 | $x_{\rm CO}{+}\Delta x$ | $x'_{\rm CO}{+}\Delta x'$ | $y_{\rm CO}{+}\Delta y$ | $y'_{\rm CO}{+}\Delta y'$ |
| 2 | $x_{\rm CO}{+}\Delta x$ | $x'_{\rm CO}{+}\Delta x'$ | $y_{\rm CO}{-}\Delta y$ | $y'_{\rm CO}{-}\Delta y'$ |
| 3 | $x_{\rm CO}{-}\Delta x$ | $x'_{\rm CO}{-}\Delta x'$ | $y_{\rm CO}{+}\Delta y$ | $y'_{\rm CO}{+}\Delta y'$ |
| 4 | $x_{\rm CO}{-}\Delta x$ | $x'_{\rm CO}{-}\Delta x'$ | $y_{\rm CO}{-}\Delta y$ | $y'_{\rm CO}{-}\Delta y'$ |

Yatay sapma (Δx, Δx' birlikte) bir bütün olarak ±; dikey (Δy, Δy' birlikte) bağımsız
±. Δx·Δy işaretleri (+,−,−,+) → ortalamada **tam 0**. İptal olan: bir düzlemde tek
her terim ($\langle\Delta x\Delta y\rangle$, $x_{\rm CO}\Delta y$, $y_{\rm CO}\Delta x$).
Hayatta kalan: $x_{\rm CO} y_{\rm CO}$ (ideal/kapalı-yörünge) ve her iki düzlemde çift
emittans terimleri ($\langle\Delta x^2\rangle,\langle\Delta y^2\rangle$). Özünde
kafesin $x\to-x,\,y\to-y$ yansıma simetrisi örneğe zorla dayatılır.

**4-KATLI SONUÇ:** $s_{\rm beam}=s_{\rm ideal}$ (**1.0×** tüm σ'da), σ^1.97.
**Yani dx·dy kanalında demet ideal'e EŞİTtir — büyütmez.**

**Açı vs ofset:** "ideal-olmayan parçacıkları ofset yerine giriş açısıyla göndersek?"
sorusu test edildi (saf açı, $\Delta x'=\Delta y'\neq0$, pozisyon ofseti 0, yine
4-katlı): yine **1.0×**. Betatron'u nasıl uyarırsan (ofset ya da açı) demet=ideal.

**Asimetrinin sebebi (dy-only 220× ama dx·dy 1.0×):**
- dy-only emittans terimi $\propto\langle\Delta y^2\rangle$ — Δy'de **çift**, faz
  ortalamasında hayatta kalır → gerçek büyütme.
- dx·dy baskın geometrik fazı $x_{\rm CO}\cdot y_{\rm CO}$ kapalı-yörünge çarpımıdır;
  betatron katkısı $\langle\Delta x\Delta y\rangle$ üzerinden gelir, korelasyonsuz
  demette 0 → büyütme yok. **dx·dy için CO=True temsilcidir** (dy-only'nin aksine).

### 13.7 Çok-seed RMS ve Omarov nicel karşılaştırması <a name="13-7"></a>

dx·dy çok desen-bağımlı (rastgele desende $\sum$ işaret iptalleri); doğru istatistik
**desen ensemble'ı üzerinden RMS**. 10 rastgele desen (dx ve dy bağımsız $N(0,\sigma)$),
CO=True 4D, tam dt=1e-11 (beam=ideal olduğundan tek ideal parçacık yeterli):

| σ (dx+dy rms) | RMS | medyan | min | maks |
|---|---|---|---|---|
| 5 μm | 8.25×10⁻⁷ | 3.7×10⁻⁷ | 6.1×10⁻⁸ | 2.2×10⁻⁶ |
| **10 μm** | **3.32×10⁻⁶** | 1.5×10⁻⁶ | 2.4×10⁻⁷ | 8.7×10⁻⁶ |
| 20 μm | 1.34×10⁻⁵ | 6.0×10⁻⁶ | 9.0×10⁻⁷ | 3.5×10⁻⁵ |

- **RMS ~ σ^2.01 — kusursuz kuadratik** (Omarov σ²).
- **10μm RMS = 3.3×10⁻⁶ vs Omarov ~10⁻⁵ → oran 3.0×.** Bazı desenler (en kötü:
  8.7×10⁻⁶) neredeyse tam Omarov'da.
- Seed saçılması min→maks ~36× → sahte EDM güçlü desen-bağımlı; tek seed (seed-0:
  1.1×10⁻⁶) yanıltıcı düşüktü, RMS doğru karşılaştırma.
- Kalan **3× faktör** Omarov'un tam parametreleriyle (σ tanımı per-plane/toplam,
  tune/β fonksiyonları, t2, halka detayları) eşlemediğimiz için beklenen belirsizlik
  içinde. **Mekanizma + kuadratik ölçek + mertebe oturuyor.**

**Omarov bağlamı:** İlgili makale Omarov vd. PRD 105, 032001 (2022) "symmetric-**hybrid**
ring" — elektrik saptırıcı + manyetik quad, bu projenin kafesiyle aynı tip (kullanıcı
teyit etti: Omarov bu testte elektrik quad değil, tam bizim örgüyü kullanıyor, dx-dy
10μm rms → ~10⁻⁵, misalignment'ta kuadratik). Eski tamamen-elektrik makale
(Hacıömeroğlu & Semertzidis arXiv:1709.01208) farklı kafestir.

### 13.8 Yöntem kaydı: ne, nasıl, NEDEN (tekrar bu yollardan geçmemek için) <a name="13-8"></a>

| Teknik | Ne yapar | Neden gerekli |
|---|---|---|
| **Model fit** (madde 2, `measure_dSy_dt_model`) | $S_y(n)=a+bn+\sum_k[c_k\cos+d_k\sin]$ fit eder, yalnız seküler $b$ döner | Düz `polyfit(...,1)` betatron salınımını (0.30/tur, ∝$A_\beta$) seküler sanar; model fit onu açıkça çıkarır. **Sınır:** periyodu pencereden uzun salınım çözülemez (yavaş dalga değil, çözünürlük) |
| **CO=True (kapalı yörüngede fırlatma)** | Betatronu kaynağında öldürür → temiz seküler | Tek parçacık betatron sızıntısı t2 ile küçülür ama ∝misalignment kalır, fitle inmez. **Uyarı:** dy-only'de emittans kanalını kaçırır (220×); dx·dy'de temsilci |
| **4D CO (yatay+dikey)** | dx+dy için tam kapalı yörünge | dx yatay orbit üretir; yalnız dikey CO bulup x=0 fırlatmak yatay betatron sokar, değeri değiştirir |
| **Antitetik (2-katlı, $\pm v$)** | $v$'de tek terimleri (doğrusal betatron) iptal | dy-only emittans ölçümü için yeterli (Δy² çift, korunur) |
| **4-katlı çift-düzlem** | x,y işaretleri bağımsız → $\langle\Delta x\Delta y\rangle=0$ tam | dx·dy demetinde 2-katlı yetmez; sonlu-N $\langle\Delta x\Delta y\rangle$ artığı sahte düz ~1.5e-5 üretir |
| **CO doğrusal ölçekleme** | CO'yu bir σ'da bul, $CO(\sigma)=\frac{\sigma}{\sigma_0}CO(\sigma_0)$ | Sextupol yok → kapalı yörünge misalignment'ta tam doğrusal; σ-taramasında CO aramasını 3× azaltır, **yaklaşıklık değil** |
| **Tam dt=1e-11 her yerde** | CO arama + spin izi aynı adım | Adım büyütmek (GL4 4. derece) yörünge/spin truncation hatasını bozar; hız için CO ölçekleme kullanılır, adım değil |

**Çürütülen hipotezler (bir daha denenmesin):** (1) doğrusallık sadece fit artefaktı
— hayır, CO=True temiz de doğrusal; (2) thick-lens — hayır, 8× inceltmede α sabit;
(3) coherent-vs-rastgele — hayır, rastgele de α≈1; (4) emittans kanalı
misalignment-kuadratik — hayır ∝A¹; (5) demet dx·dy'yi büyütür — hayır, =ideal;
(6) açı enjeksiyonu dx·dy'yi büyütür — hayır, ofsetle aynı; (7) yavaş spin-tune
dalgası — hayır, spin donuk, kontaminasyon hızlı betatron.

**Reprodüksiyon:** `test_dxdy_geometric_phase.py` (bu oturumda eklendi) — dy/dx/dx+dy
kontrastı, CO=True α taraması ve çok-seed RMS'i tek koşuda üretir.

---

## 14. Yörünge-trim'in §13 estimator'ıyla yeniden değerlendirmesi (makale_trim_tr.tex) <a name="14"></a>

> §13 estimator düzeltmesinden sonra, `makale_trim_tr.tex`'in yörünge-trim
> kademesinin gerçek tabanı ve "yörünge-trim tek başına yeter mi" sorusu
> 200 μm mekanik hizalama senaryosunda yeniden ölçüldü.

### 14.1 200 μm baseline: "1e-3 CW" uzlaşması

200 μm RMS dx+dy rastgele desende sahte EDM (simetrik-4-parçacık + model fit,
3 seed): seed 0/1/2 = +9.89×10⁻⁴ / −4.27×10⁻⁴ / −1.05×10⁻³, **RMS ~8.7×10⁻⁴**.
Bu, §13.7'deki 10 μm RMS değerinin (2.67×10⁻⁶) σ² ölçeğiyle tam tutarlı:
2.67×10⁻⁶ × (200/10)² = 1.07×10⁻³. **Sonuç:** baştan beri "CW için ~1e-3"
denen değer betatron artefaktı DEĞİL — **200 μm'deki gerçek dx·dy geometrik-faz
sahte EDM'idir.** Hizalama 10 μm'ye inerse sinyal ~3×10⁻⁶'ya (σ² ile 400×) düşer.

### 14.2 Basit ölçüm reçetesi: CO=True gereksiz

Sahte EDM'i ölçmek için kapalı-yörünge bulmaya (CO=True) gerek yok. Yeterli:
dx,dy ver, **4-katlı simetrik 4 parçacık** (eksen etrafında ±Δx, ±Δy bağımsız,
spin hepsinde boylamsal) gönder, $\langle S_y\rangle$'i **model fit** et.
§13.6'da kanıtlandığı gibi 4-katlı simetri betatron + ⟨ΔxΔy⟩ artığını temizler;
sonuç CO=True ile aynı (4-katlı ortalama tek-parçacık ham değerinden hem tutarlı
hem temiz: 200 μm'de RMS 8.7×10⁻⁴ vs tek-parçacık dağınık 0.6–3×10⁻³).

### 14.3 Yörünge-trim (k≤4) tek başına YETMİYOR

Yörünge-trim: kafese özgü O_x,O_y mod parmak izleri (antisim k=1..4 cos/sin) bir
kez kalibre; pattern orbiti BPM ofseti (100 μm) ile ölçülür; düz LSQ fit → trim
(k-mod/LASSO YOK). Kalan misalignment §14.2 reçetesiyle ölçüldü:

| seed | f (200μm) | f (trim sonrası) | bastırma |
|---|---|---|---|
| 0 | +9.89×10⁻⁴ | +2.34×10⁻⁴ | 4.2× |
| 1 | −4.27×10⁻⁴ | +2.40×10⁻⁴ | 1.8× |
| 2 | −1.05×10⁻³ | +4.42×10⁻⁴ | 2.4× |

**f_sonra RMS = 3.2×10⁻⁴** (~120 μm eşdeğer). Yörünge-trim dx·dy kanalını yalnız
~2.7× düşürüyor — CW/CCW girişi hedefinin (~10⁻⁵) **~30× üstünde**.

### 14.4 Simetrik-alt-uzay tabanı §13 estimator'ıyla da GERÇEK

Kalan 3.2×10⁻⁴, trim makalesinin simetrik-alt-uzay tabanıyla (~2.5×10⁻⁴) uyumlu.
**Kritik:** bu taban §13'ün doğru estimator'ıyla (model fit, CO yok, dx·dy) ÖLÇÜLDÜ
ve hâlâ gerçek — yani trim makalesinin "yörünge tek başına yetmez, simetrik
alt-uzay orbit'e kapalı" sonucu betatron artefaktına dayanmıyor, **sağlam**.
dy-only'de trim 101× alabiliyordu; dx·dy çok daha zayıf çünkü çapraz kanal büyük
ölçüde simetrik/orbit-görünmez alt-uzayda. **İki kademeli mimari (yörünge +
spin) ayakta; spin kademesi hâlâ gerekli.**

*(Reprodüksiyon: `/tmp/orbit_trim_dxdy.py` mantığı — kafes O kalibrasyonu +
k≤4 LSQ trim + simetrik-4-parçacık model-fit doğrulaması; bu oturumda koşuldu.)*

### 14.5 Rekonstrüksiyon metotları karşılaştırması — taban metot-bağımsız

`makale_trim_tr.tex`'in orbit-trim tabanının (§taban/§erisim/§genis-baz) bir
rekonstrüksiyon metodu kusuru DEĞİL, gözlenebilirlik sınırı olduğunu doğrulamak
için altı metot aynı 200μm dx+dy verisinde (tek yörünge, ofset 100μm, gürültü
1μm, 3 seed) denendi; her birinin trim sonrası kalan sahte EDM'i simetrik-4-
parçacık + model fit ile ölçüldü:

| metot | RMS [rad/s] | bastırma |
|---|---|---|
| baseline 200μm | 8.67×10⁻⁴ | 1× |
| **R-LS k≤4 (FODO-antisim)** | **1.84×10⁻⁴** | **4.7× (en iyi)** |
| TSVD r=8 | 4.53×10⁻⁴ | 1.9× |
| CLEAN (k=0..8) | 7.81×10⁻⁴ | 1.1× |
| Bozoki (azimut k≤4) | 9.79×10⁻⁴ | 0.9× |
| R-LS k≤7 | 1.06×10⁻³ | 0.8× |
| R⁻¹ direct (48-dim) | 2.68×10⁻³ | 0.3× |

**Bulgular — makalenin taban tezinin metot-bağımsız teyidi:**
1. **En iyi R-LS k≤4** (dar, FODO-antisim). Bu, §kazanç/§esik'teki "kazanç
   hiyerarşisi doğal düzenleyicidir" ilkesidir: yüksek-kazançlı k≤4 modları
   ofsete dayanıklı.
2. **Agresif metotlar (R-LS k≤7, CLEAN, TSVD, R⁻¹) k≤4'ten KÖTÜ.** Düşük-kazançlı/
   yüksek modları fit etmek BPM ofsetini o modlara enjekte eder — tam olarak
   **§genis-baz mekanizması** (eşik-altı sütunlara ofset enjeksiyonu, orada 9×
   kötüleşme). R⁻¹ direct en kötü: ofset R⁻¹b olarak tam sızar (κ(R)≈249×100μm).
3. **Bozoki (azimut baz) zayıf** — yanlış baz (clean_rls_yontemleri.md §4 ile uyumlu).
4. **Hiçbir orbit metodu ~1.8×10⁻⁴ altına inmiyor** — bu plato §erisim'in simetrik
   alt-uzay SNR≈0.3<1 gözlenebilirlik sınırıdır. Rekonstrüksiyon metodu seçimi
   tabanı değiştirmez → **spin kademesi metot-bağımsız olarak gereklidir.**

*(Reprodüksiyon: `/tmp/recon_compare.py`; R_dy_1.npy/R_dx_1.npy
`build_response_matrix.py` ile üretilir.)*

### 14.6 Stage 2 — spin ölç-trim zinciri tamamlıyor (COD + spin uçtan uca)

Orbit-trim'in çarptığı simetrik-alt-uzay tabanı (orbit-görünmez) spin kademesiyle
temizlendi. Senaryo: 200μm dx+dy (seed 0), orbit-trim sonrası artık ≈ simetrik kısım
(QF/QD hücre-içi ortalaması, RMS ~141μm), f₀ ölçüldü; simetrik mod S₂'nin spin
kuplajı çift-kuadratürle kalibre edildi; §12.6 ölç-trim döngüsü (`A_trim=−f/|c|`,
faz ψ yönünde) uygulandı. f, simetrik-4-parçacık + model fit (§13, CO yok) ile ölçüldü.

| adım | f [rad/s] | bastırma |
|---|---|---|
| orbit-trim sonrası (simetrik artık) | +9.44×10⁻⁴ | 1× |
| spin iter 1 | −2.19×10⁻⁶ | 431× |
| spin iter 2 | −9.12×10⁻⁷ | 1035× |
| spin iter 3 | −3.80×10⁻⁷ | 2485× |
| spin iter 4 | −1.58×10⁻⁷ | **5964×** |

**Bulgular:**
- Simetrik S₂ spin kuplajı **|c|=12.6 rad/s/m** — orbit göremez (kazanç G~3,
  eşik altı) ama spin görür; §taban/§erisim'in (SNR_spin~17) somut teyidi.
- Spin ölç-trim simetrik artığı **~6000× düşürüp 1.6×10⁻⁷'ye** indiriyor —
  **CW/CCW girişi hedefinin (~10⁻⁵) ~60× altında.** Trim bütçesi 75μm (çoğu ilk adım).
- Tek simetrik mod (S₂) skaler f'i nullamaya yeterli; iterasyonlar ölçüm tabanına yakınsar.

**Büyük resim:** dx·dy kanalında sahte EDM **simetrik alt-uzay tarafından domine
edilir** (orbit-görünmez); orbit-trim tek başına ~2-4× (§14.3-14.5), ama **spin
kademesi baskın simetrik kısmı 6000× temizler** → hedef altı. İki-kademeli mimari
(COD + spin), §13 doğru estimator'ı ve dx·dy kanalıyla **uçtan uca doğrulandı**.

*(Reprodüksiyon: `/tmp/spin_trim_chain.py`; estimator git geçmişinden
4cb9cc6:false_edm_mode_scan.py.)*

---

*Son güncelleme: oturum `claude/awesome-babbage-nmi6w9`, tarih 2026-06-16.
§13 tam: dy-only doğrusallığı (§13.1-2), iki ikinci-derece kanal (§13.3),
Omarov'un yeri (§13.4), demet=ideal + 4-katlı simetri (§13.6), çok-seed RMS +
Omarov 3× uyum (§13.7), yöntem kaydı (§13.8). §10.2 thick-lens argümanı düzeltildi.
§14: yörünge-trim §13 estimator'ıyla yeniden değerlendirildi — 200μm baseline
~1e-3 (dx·dy gerçek), orbit-trim k≤4 tek başına ~2.7× (yetmez), simetrik taban
~3×10⁻⁴ gerçek → iki kademeli mimari sağlam.*
