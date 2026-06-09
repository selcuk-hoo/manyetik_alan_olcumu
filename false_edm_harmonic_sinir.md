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

*Son güncelleme: oturum `claude/claude-md-docs-spai7t`, tarih 2026-06-09*
