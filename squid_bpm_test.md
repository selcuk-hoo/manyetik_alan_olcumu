# squid_bpm_test.md — Tek BPM ile per-quad K-modülasyon: neden çalışmadı (pedagojik)

> **Durum (2026-06).** Bu belge, "tüm quad'ları (farklı frekansta) modüle edip
> **tek bir BPM** ile 48 quad'ın 100 μm rastgele hizalama hatasını ölçebilir
> miyiz?" sorusunun ilk testinin **dürüst** kaydıdır.
>
> **Kısa cevap: HAYIR.** Sinyali, modülasyonun yarattığı **optik-nefes** (tüm
> halkanın β/φ'sinin değişmesi) boğuyor; **çok BPM de kurtarmıyor** (48 BPM'de
> bile korelasyon ≈ 0). Bu sonuç, bu oturumda ürettiğim `ac_bba_observability.py`
> "simetrik modu görür, corr=0.997" iddiasını **YANLIŞLIYOR** — o iddia,
> optik-nefesi ihmal eden idealize bir modele dayanıyordu.
>
> **Bu nefes etkisi gerçek C++ izleyiciyle bağımsız doğrulandı (§5.5):** analitik
> bir artefakt değil. Dolayısıyla **"farklı-frekans per-quad modülasyon + SQUID
> BPM ile genlik okuma" dalı bu belgede kesin olarak kapatılır (§7).** Modülasyon
> yöntemini kurtarabilecek iki *ayrı* yol açık kalır (sinir ağı; aynı-frekans
> tüm-quad modülasyonu, v2.7) — bunlar bu testin kapsamı dışındadır, §8'de
> ertelenmiş olarak listelenir.
>
> Reprodüksiyon: `/tmp/kmod_recover/single_bpm_test.py` (analitik; C++ gerekmez),
> `/tmp/kmod_recover/breathing_cpp.py` (C++ doğrulama).

---

## 1. Soru ve umut

Sahte EDM'i süren **simetrik** kaçıklık alt-uzayı kapalı-yörüngeye neredeyse
görünmezdir (no-go, `README §19.2`). Bunu görmenin önerilen yollarından biri
**per-quad K-modülasyonu**: her quad'ı ayrı bir frekansta hafifçe "titret"
(gradyanını $g_i \to g_i(1+\varepsilon\sin 2\pi f_i t)$ yap), BPM sinyalini FFT'le
ve $f_i$ frekansındaki **genliği** çek. Umut: bu genlik, o quad'ın demet-merkez
ofsetiyle orantılıdır → her quad'ın hizalama hatasını **tek tek** okuruz, global
bir matris ters-çevirmesi yapmadan (yani no-go'yu atlatarak).

Bu testin amacı bu umudu **en sade** halinde sınamak: tek BPM, 48 quad.

---

## 2. Modülasyonu nasıl simüle ettim? (kritik — çünkü zaman-takibi imkânsız)

Bu kısım kafa karıştırıcı olduğu için ayrıntılı açıklıyorum. (Not: bu **FDTD**
değil; "finite-difference" = **sonlu fark**, ama zamanda değil, **gradyanda**.)

### 2.1 Neden zaman-domeninde FFT yapamayız?
İzleyicinin adımı $dt = 10^{-11}$ s, tur süresi $T_{\rm rev}\approx 4.5\,\mu$s.
Modülasyon frekansı ~1–10 kHz → bir modülasyon periyodu ~0.1–1 ms. FFT için
onlarca periyot izlemek gerekir → **$10^9$–$10^{10}$ adım**. Pratikte imkânsız.
**İlk çalışmalarda iki gradyan değeri ($g_1$, $g_2$) kullanılmasının sebebi tam
da buydu** — zaman-modülasyonundan kaçınmak.

### 2.2 Adyabatik eşdeğer (numaranın özü)
Modülasyon frekansı (1–10 kHz), betatron frekansından ($Q\,f_{\rm rev}\approx
2.3\times224\,{\rm kHz}\approx 515$ kHz) **çok yavaştır**. Bu **adyabatik** limitte
demet, her an o anki gradyanın **kapalı yörüngesinde** oturur. Dolayısıyla
BPM sinyalindeki $f_i$ frekansının genliği, basitçe kapalı yörüngenin gradyana
göre **statik türevidir**:

$$
A_i \;=\; \frac{\partial\, y_{\rm BPM}}{\partial g_i}\,\times\,(\text{modülasyon derinliği}).
$$

Yani **zaman-takibi hiç yapmıyoruz.** Genlik = statik bir duyarlılık.

### 2.3 Türevi sonlu-farkla hesaplama
Bu türevi, quad $i$'nin gradyanını $\varepsilon$ kadar değiştirip **iki statik
kapalı yörünge** alıp farkla buluyoruz:

$$
A_i \;=\; y_{\rm BPM}\big(g_i(1+\varepsilon)\big) \;-\; y_{\rm BPM}(g_i).
$$

48 quad için ~49 statik kapalı-yörünge hesabı. **Bu, $g_1/g_2$ hilesinin per-quad
versiyonudur.** "Farklı frekans / FFT" yalnızca *gerçek deneyde* 48 türevi tek
ölçümde ayırmanın yöntemidir; simülasyonda her birini sonlu-farkla ayrı buluruz —
sonuç birebir aynıdır (adyabatik limitte exact).

---

## 3. Kapalı yörüngeyi nasıl hesapladım? (optik-nefes neden kendiliğinden var)

C++ izleyici yerine **analitik kapalı-yörünge çözücüsü** kullandım (hızlı + temiz;
sıfırdan fırlatmanın getirdiği betatron kirlenmesi yok):

1. Halkayı eleman-eleman kur: 24 hücre × [QF, drift, QD, drift]. **Her quad'a
   kendi $K$'sı** (per-quad; **uniform değil**).
2. Bir-tur matrisi $M$ = tüm eleman 2×2 matrislerinin çarpımı.
3. Kaçık quad'ın feed-down kick'i $\theta_j = K_j L\, dy_j$.
4. Kapalı yörünge: $X_0 = (I-M)^{-1} D$ ($D$ = kicklerin başa taşınmış toplamı);
   sonra $X_0$'dan başlayıp her BPM'de $y$ oku.

**Kritik nokta:** Bir quad'ın $K$'sını değiştirince **bir-tur matrisi $M$ değişir**
→ $\beta(s)$, $\phi(s)$, tune $Q$ **tüm halkada** değişir. Yani modülasyonun
"**optik-nefes**" terimi (bir quad'ı kıpırdatınca tüm optiğin nefes alması) bu
modelde **kendiliğinden** vardır. *(Eski `ac_bba_observability.py`'de bu YOKTU:
orada sabit optikte yalnızca feed-down terimi vardı — işte hatanın kaynağı bu.)*

---

## 4. Geri-çatım yöntemi

- Kalibrasyon $C_i$: yalnız quad $i$ birim kaçıkken aynı modülasyon genliği
  (diagonal duyarlılık).
- **Tek BPM:** $\hat{dy}_i = A_i / C_i$.
- **Çok BPM:** per-quad projeksiyon $\hat{dy}_j = \sum_i C_{ij}A_{ij}/\sum_i C_{ij}^2$.

---

## 5. Sonuçlar (dikey düzlem, ±100 μm, depth %2, seed 0)

Sağlık kontrolü: 100 μm kaçıklık → kapalı yörünge **0.37 mm RMS** (makul, ~3.7×
büyütme).

| Geri-çatım | gerçek dy ile korelasyon |
|---|---|
| **[1] Tek BPM, optik-nefes DAHİL** (gerçekçi) | **+0.07** (≈ 0) |
| [2] Tek BPM, optik-nefes YOK (saf feed-down) | +1.000 (kurgu gereği önemsiz) |
| **[3] Çok BPM (2→48), optik-nefes DAHİL** | **her sayıda ≈ −0.03; 48 BPM bile** |

Ek: kör nokta **9/48** (faz-ilerlemesi $\cos(|\phi_{\rm BPM}-\phi_i|-\pi Q)=0$
sıfırları), diagonal duyarlılık $|C_i|$ yayılımı **1187×**.

---

## 5.5 C++ ile bağımsız doğrulama (analitik artefakt değil)

"Hem analitik hem simülasyon aynı çıkıyor; ortak bir hata olabilir mi?" sorusu
haklı bir şüphedir — iki kod aynı yanlışı yapıyorsa uyuşmaları bizi aldatabilir.
Bunu sınamak için aynı per-quad duyarlılığı $A_i=\partial y_{\rm BPM}/\partial
g_i$'yi **gerçek C++ izleyiciyle** (GL4 semplektik, deflektör dahil tam alan
integratörü) `quad_dG` ile yeniden hesapladım — `integrator.cpp` değiştirilmeden,
çünkü per-quad statik kesirsel gradyan (`quad_dG`) zaten mevcuttu.

Kritik nokta: bu iki kod **neredeyse hiçbir şey paylaşmaz.** Analitik çözücü
kafesi 2×2 matrislerle kurar ve deflektörü dikeyde *saf drift* sayar; C++ ise
gerçek alan integratörüdür (tamamen farklı uygulama). Ortak olan tek şey
`params.json` (K değerleri, uzunluklar). İki **farklı kafes temsili** %1'de
uyuşuyorsa, ortak bir *modelleme* hatasının ikisini birden aynı yönde yanıltma
olasılığı çok düşüktür.

| quad | $A_{\rm C++}$ [μm] | $A_{\rm analitik}$ [μm] | oran |
|---|---|---|---|
| 0  | 0.000  | 3.447  | 0.00 ⚠️ |
| 9  | −2.062 | −2.059 | 1.00 |
| 18 | 2.114  | 2.106  | 1.00 |
| 28 | 2.521  | 2.507  | 1.01 |
| 37 | −9.848 | −9.827 | 1.00 |
| 47 | −5.909 | −5.929 | 1.00 |

corr$(A_{\rm C++}, A_{\rm analitik}) = 0.966$ (tek aykırı nokta quad 0 olmasaydı
≈ 1.00). **6 quad'ın 5'i %1 içinde birebir.** Maliyet ~40 s/kapalı-yörünge.

> **quad 0 aykırılığı ($A_{\rm C++}=0$):** BPM/fırlatma referans noktası tam
> quad 0 girişindedir; okumanın yapıldığı quad'ın *kendi* gradyanı kick'ten önce
> okunduğu için C++ orada yerel olarak kördür. Tek nokta, fiziği değiştirmez.

**Sonuç:** optik-nefes gerçek bir fizik etkisidir, analitik çözücünün bir
artefaktı değil.

---

## 6. Teşhis — neden başarısız oldu?

**Suçlu optik-nefes** (kör noktalar ikincil):

- Modülasyon quad'ı titretince, **mevcut 0.37 mm'lik kapalı yörüngenin tamamı**
  nefes alır.
- Büyüklük tahmini: %2 gradyan bump → tune kayması ~%0.04, ama kapalı-yörünge
  ölçeği $\propto 1/(2\sin\pi Q)$ değişir → BPM'de birkaç μm.
- Doğrudan per-quad feed-down sinyali: $\sim K\varepsilon L\,dy \times G \approx
  0.9\,\mu$m.
- **Nefes (~birkaç μm) > doğrudan sinyal (~0.9 μm), birkaç kat.** Genliği nefes
  domine ediyor → aradığımız quad ofseti gürültüye gömülüyor.

**[2] (nefessiz) corr=1.000 ama [1] (nefesli) corr=0 olması, suçlunun kesin
olarak optik-nefes olduğunu kanıtlar.**

### 6.1 "Kaldıraç" ne demek? (asıl mekanizma)

Buradaki sezgiye aykırı nokta şudur: **ufacık (%1–2) bir modülasyon nasıl
~10 μm'lik bir BPM salınımı yaratıyor?** Cevap, modülasyonun *neyi* oynattığında
gizli. Bir quad'ı modüle etmenin iki etkisi var:

1. **Kendi feed-down kick'i** değişir: $\Delta\theta_i = K\varepsilon L\,dy_i$.
   Bu, 48 katkıdan **yalnızca biri** ve aradığımız sinyal budur. Küçük (~0.9 μm).
2. **Tüm optiği** ($M$, dolayısıyla $\beta$, $\phi$, $Q$) değiştirir. Bu değişmiş
   optikten **48 quad'ın HEPSİNİN feed-down kick'i** yeniden taşınır. Yani
   modülasyon, **48 quad'ın birlikte kurduğu 0.37 mm'lik kapalı yörüngenin
   tamamına** etki eder.

**"Kaldıraç" tam olarak budur:** modülasyonun tutup oynattığı şey, o quad'ın
*kendi* küçük ofseti değil, **önceden var olan tüm yörünge**. Tek bir quad'ı
%2 kıpırdatmak kendi katkısını zar zor değiştirir; ama 0.37 mm'lik koca yörüngenin
geçtiği optiği %birkaç oynatmak, doğal olarak birkaç-on μm verir. Kaldıracın ucu
küçük (bir quad), ama kaldıraç kolu uzun (tüm yörünge).

**Ayrıştırma ile kanıt** (quad 37, %2, BPM 0):

| Senaryo | bump yanıtı |
|---|---|
| **Tüm dy** deseni varken | **−9.83 μm** |
| **Yalnız dy[37]** varken (diğer 47 quad ideal) | **−0.037 μm** (265× küçük!) |

Quad 37'nin kendi ofseti ($dy_{37}$) tek başına yalnız 0.037 μm verir; ölçülen
9.83 μm'nin neredeyse tamamı **diğer 47 quad'ın kurduğu yörüngenin yeniden
taşınmasından** gelir. Yani genlik, modüle edilen quad'ın ofsetiyle değil,
**komşularının kurduğu yörüngeyle** orantılı → kalibrasyona bölmek anlamsız.

### 6.2 Ortak hata değil — dört kontrol

Sürprizin yapay (kod hatası) olmadığını dört bağımsız sağlama gösterir:

1. **Tune patolojik değil.** Kesirsel $Q = 0.3032$, $\sin\pi Q = 0.815$ →
   yörünge büyütme $1/(2\sin\pi Q) = 0.614$. Tamsayı/yarım-tamsayı rezonansından
   uzak; "(I−M)⁻¹ neredeyse tekil" gibi yapay bir şişme **yok**.
2. **Tüm yörünge oynuyor, lokal artefakt değil.** Quad 37'yi %2 bump'layınca
   *tüm halka* yörüngesi 11.9 μm RMS kayıyor (367→359 μm). Etki global.
3. **Kaldıraç izolasyonu** (§6.1 tablosu): nefes, modüle edilen quad'ın kendi
   ofsetiyle değil tüm yörüngeyle ölçekleniyor — fiziksel olarak şeffaf.
4. **Doğrusal, büyük-eps artefaktı değil.** Minik türev ($\delta\varepsilon=
   10^{-5}$) %2'ye ekstrapole edilince −10.0 μm; doğrudan ölçülen −9.83 μm.
   Ayrıca derinlik taraması (%0.5→%4): hem sinyal hem nefes derinlikle *doğrusal*
   ölçeklenir, oranları **sabit ~0.14** kalır. Yani sonlu-fark adımı doğrusal
   olmayan bir bölgeyi yoklamıyor; modülasyonu küçültmek de oranı düzeltmiyor.

**Sürprizin gerçek kaynağı yukarıda:** "küçük modülasyon" anormal bir şey
yapmıyor. Anormal görünen sayı, **±100 μm rastgele kaçıklığın ~3.7× büyütülerek
0.37 mm'lik bir kapalı yörünge kurması.** Bu büyük yörüngenin optiğini %2 oynatmak
doğal olarak birkaç-on μm verir — yörüngeye *göre* küçük (%2–3), sadece ölçmek
istediğimiz ~0.9 μm'lik feed-down'a *göre* büyük.

### 6.3 Çok BPM neden kurtarmıyor?

Nefes, **koherent** bir kontaminasyondur (≈ mevcut yörünge × tek bir skaler),
rastgele gürültü değil. Bu yüzden BPM sayısı arttıkça **ortalamayla sönmez**;
per-quad projeksiyon her quad'da **yanlı** kalır → 48 BPM'de bile corr ≈ 0.

> **İnce ayrım:** BPM sayısını artırmak **kör noktaları** (9/48; $\cos(\cdot)=0$
> sıfırları) giderir — farklı BPM'ler farklı quad'larda kördür, birlikte hepsini
> kaplarlar. Ama kör noktalar zaten sınırlayıcı etken değildi; kör noktalar tümüyle
> kapansa bile **nefes** corr ≈ 0 tutar. Kör-nokta körlüğü ile nefes körlüğü ayrı
> şeylerdir; ilki BPM ekleyerek çözülür, ikincisi çözülmez (lineer harita çöküyor).

---

## 7. Kesin kapatma: "farklı-frekans per-quad + SQUID BPM" dalı ölü

Bu test, belirli bir öneriyi **kesin olarak** kapatır ki ileride "acaba şöyle mi"
diye geri dönmek gerekmesin:

> **Öneri:** her quad'ı *farklı* bir frekansta modüle et (frekans-çoğullama);
> böylece tek ölçümde her quad'ın genliği $A_i$ ayrı ayrı çözülür. Gürültü
> sorununu da yüksek-çözünürlüklü **SQUID BPM**'lerle aş.

**Neden ölü:**

- Farklı-frekans olayının *tek* kazancı, 48 quad'ın genliğini tek ölçümde
  **ayırmaktır** ($f_i$ demodülasyonu). Ama ayrılan o genlik $A_i$, §5.5/§6'da
  gösterildiği gibi **nefes-domine** (S/B ≈ 0.14) ve modüle edilen quad'ın kendi
  ofsetiyle değil **komşularının yörüngesiyle** orantılı. Genliği temiz ayırmak,
  yanlış büyüklüğü temiz ayırmaktan başka bir şey değil.
- **SQUID BPM ne katar?** Daha iyi çözünürlük ve/veya daha çok BPM. Ama nefes
  **rastgele gürültü değil, koherent bir sistematiktir**; çözünürlük artırmak veya
  BPM eklemek koherent sistematiği **ortalamayla söndürmez** (sonuç [3]:
  2→48 BPM'de corr ≈ −0.03). SQUID'in tek üstünlüğü (düşük gürültü), nefesin
  problem olduğu yerde işe yaramaz.
- Kör noktalar (SQUID için öne sürülen ikinci gerekçe) BPM ekleyerek zaten
  giderilir (§6.3) — ama bu da sınırlayıcı etken değildi.

Dolayısıyla **`ac_bba_observability.py` (simetrik corr=0.997)** iddiası ve ona
dayalı linchpin/bütçe sayıları **YANLIŞLANDI** (o model nefessiz [2] idi). "Genlik
oku, kalibrasyona böl" + SQUID dalı **kapalıdır.** Bu, geçmiş kayıtla tutarlı:
`README §19.2` per-quad k-mod'u "operasyonel olarak ağır" diye işaretlemişti.

**Fork** (pozitif yöntem mi / negatif teorem mi) bu dal için **negatif** sonuçlandı.

---

## 8. Hayatta kalan iki yol (bu belgenin KAPSAMI DIŞINDA — sonraki adım)

Bu belge yalnız **farklı-frekans + genlik-okuma + SQUID** dalını kapatır.
Modülasyon yöntemini kurtarabilecek, mekanik olarak *farklı* iki yol **açık**
kalır. Bunlara geçmeden önce yukarıdaki dalı kesin kapatmak, "acaba şöyle mi,
böyle mi" döngüsüne girmemek içindir.

1. **Sinir ağı (nonlineer ters-harita).** Lineer "genlik ÷ kalibrasyon" çöküyor,
   çünkü nefes haritayı nonlineer/non-invertible yapıyor. Girişi (modülasyondaki
   quad K değerleri + BPM ölçümleri), çıkışı (quad hizalama hataları) olan bir NN
   simülasyon verisinden eğitilebilir. Nefes **deterministik** olduğu için NN onu
   prensipte *öğrenip ayıklayabilir*.
   > **Ama deneysel bağlayıcılık şüphesi (önemli):** (a) NN, eğitildiği ileri-modelin
   > sadakati kadar iyidir — modellenmemiş β-beat, multipoller, BPM nonlineerliği
   > gerçek makinede NN'i yanlı kılar. (b) Daha derin sorun: NN **no-go
   > gözlenebilirlik tabanını yenemez**. Simetrik (orbit-görünmez) alt-uzay BPM'de
   > alt-gürültü imza bırakıyorsa, NN o bilgiyi *yoktan var edemez* — eğitim-kümesi
   > önceline (ortalamaya) regresyon yapar, ki bu gerçek makinede **bağlayıcı
   > değildir**. Yani NN nefesi ayıklayıp **antisimetrik** kısmı iyileştirebilir
   > (ki o zaten ölçülebiliyordu), ama asıl problem olan **simetrik körlüğü**
   > açmaz. → Ayrı bir test gerektirir; umut sınırlı.
2. **Aynı-frekans tüm-quad modülasyonu (v2.7) — İNCELENDİ (2026-06-29).** Tüm
   quad'lar **aynı** frekansta modüle edilir; iki-gradyan farkı $\Delta y =
   y(g{\times}1.02)-y(g)$ alınır (BPM ofseti ortak-mod iptal) ve **tam 48×48
   $\Delta R$ matrisi** ters çevrilir (`np.linalg.solve`). Dağıtık-frekanstan farkı:
   burada nefes **$\Delta R$'nin içinde** (köşegen-dışı kuplaj dahil) → kirlilik
   değil. Sonuç:
   - **Temiz limitte exact:** 48-BPM inversiyonu corr = 1.000000 (hata ~10⁻¹² μm).
     **Nefes engel değil** — v2.7'nin "çalışıyor" iddiası temiz limitte doğru.
   - **Ama no-go duvarı:** cond$(\Delta R)$ = 3.7×10⁴; antisim/sim yön kazanç oranı
     ~1393×. Gerçekçi BPM gürültüsü (σ=0.1/1/10 μm) → corr 0.67/0.07/0.00.
   - **Verdikt:** v2.7 *yeni bir kapı değil*; bilinen **no-go inversiyon sınırının**
     başka yüzü. **Nefes ≠ no-go** (ikisi ayrı; v2.7'de suçlu no-go, dağıtık-frekansta
     suçlu nefes). Repro: `/tmp/kmod_recover/v27_recheck.py`.

---

## 9. Reprodüksiyon

```bash
python3 /tmp/kmod_recover/single_bpm_test.py    # analitik [1]/[2]/[3] + kör-nokta
python3 /tmp/kmod_recover/breathing_cpp.py --nq 6   # C++ doğrulama (§5.5)
```
Analitik çözücü (`analytic_kmod.py` yapı taşları), `params.json`,
$\varepsilon=0.02$, seed=0, dikey düzlem. C++ doğrulama `quad_dG` kullanır
(`integrator.cpp` değiştirilmez); ~40 s/kapalı-yörünge.
