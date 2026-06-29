# squid_bpm_test.md — K-modülasyonla quad hizalama ölçümü: toplu pedagojik kayıt

> **Kapsam.** Bu belge, "K-modülasyon + BPM ile quad hizalama hatalarını
> ölçebilir miyiz?" sorusunun **tüm dallarının tek kaydıdır** (notlar dağılmasın
> diye burada toplandı):
> - **§1–7 Dağıtık-frekans (her quad ayrı $f_i$) + genlik okuma:** ÖLÜ — sinyali
>   **optik-nefes** boğuyor (koherent; çok-BPM/SQUID söndürmez).
> - **§8 Tek-frekans tüm-quad $\Delta R$ (v2.7) + tam matris inversiyonu:** nefes
>   *engel değil* (matrisin içinde), ama tek-atış BPM gürültüsünde **no-go** duvarı
>   (cond $\sim$10⁴).
> - **§9 Tek-frekans + lock-in (1 kHz mod + ortalama):** BPM gürültüsü **zamanla
>   √N ile yenilir** → **antisimetrik (orbit-görünür)** misalignment'lar iyi
>   kurtarılır (SQUID tek-frekansla yeniden anlamlı, §9.4). **AMA sahte-EDM'i süren
>   simetrik (orbit-görünmez) kısım lock-in tabanında BİLE kurtarılamaz (§9.5,
>   no-go); ΔR'de %0.5 β-beat felakettir.** corr metriği bu simetrik felaketi gizler.
> - **§10 Kalan yollar:** NN ileri-harita + "akıllı düzeltme" (bkz. `YAPILACAKLAR.md §4`).

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
> bir artefakt değil. Dolayısıyla **"farklı-frekans per-quad modülasyon + genlik
> okuma" dalı (ve SQUID'in onu kurtaracağı iddiası) §7'de kesin kapatılır.** Buna
> karşılık tek-frekans v2.7 yolu (§8) lock-in ile (§9) **canlanır.**
>
> Reprodüksiyon: `/tmp/kmod_recover/{single_bpm_test,breathing_cpp,v27_recheck,v27_lockin}.py`.

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
  2→48 BPM'de corr ≈ −0.03). SQUID'in tek üstünlüğü (düşük gürültü), *bu dalda*
  (nefes problemi) işe yaramaz.
  > **Not:** Bu, "SQUID işe yaramaz" demek değildir — yalnız **dağıtık-frekans +
  > genlik-okuma** dalında işe yaramaz. SQUID'in düşük gürültüsü, gürültünün
  > gerçekten sınırlayıcı olduğu **tek-frekans + lock-in** bağlamında (§9.4)
  > yeniden anlamlıdır.
- Kör noktalar (SQUID için öne sürülen ikinci gerekçe) BPM ekleyerek zaten
  giderilir (§6.3) — ama bu da sınırlayıcı etken değildi.

Dolayısıyla **`ac_bba_observability.py` (simetrik corr=0.997)** iddiası ve ona
dayalı linchpin/bütçe sayıları **YANLIŞLANDI** (o model nefessiz [2] idi). "Genlik
oku, kalibrasyona böl" + SQUID dalı **kapalıdır.** Bu, geçmiş kayıtla tutarlı:
`README §19.2` per-quad k-mod'u "operasyonel olarak ağır" diye işaretlemişti.

**Fork** (pozitif yöntem mi / negatif teorem mi) bu dal için **negatif** sonuçlandı.

---

## 8. Tek-frekans tüm-quad $\Delta R$ (v2.7): nefes engel değil, ama no-go duvarı

Dağıtık-frekanstan **mekanik olarak farklı** bir yol: tüm quad'ları **aynı**
frekansta (aynı fazda) modüle et. Bu, projenin eski $g_1/g_2$ K-modülasyon =
$\Delta R$ yöntemidir (v2.7 tag'i, `test_kmod_reconstruction.py`).

**Yöntem:** iki gradyan konfigürasyonu ($g$ ve $g\times1.02$) arasında kapalı
yörünge farkı alınır:
$$\Delta y \;=\; y(g{\times}1.02) - y(g) \;=\; \Delta R\,\cdot\,dy,$$
sonra **tam 48×48 $\Delta R$ matrisi** ters çevrilir: $\widehat{dy} =
\Delta R^{-1}\Delta y$ (`np.linalg.solve`, 48 BPM).

**Dağıtık-frekanstan kritik fark — nefes neden artık engel değil:** Burada genliği
köşegen kalibrasyona *bölmüyoruz*; **tüm matrisi** ters çeviriyoruz. Optik-nefes
(bir quad'ı kıpırdatınca tüm optiğin değişmesi) $\Delta R$'nin **köşegen-dışı
yapısında zaten kayıtlıdır**. Yani nefes bir *kirlilik* değil, ters-çevrilen
haritanın *parçasıdır*. Sonuç (analitik $\Delta R$, C++ ile §5.5'te doğrulanmış):

| Durum | corr | Yorum |
|---|---|---|
| **Temiz 48-BPM inversiyon** | **1.000000** (hata ~10⁻¹² μm) | Nefes engel DEĞİL — tam matriste |
| BPM gürültü σ=0.1 μm (tek-atış) | 0.665 | koşul sayısı gürültüyü büyütüyor |
| σ=1 μm | 0.075 | çöküş |
| σ=10 μm | ≈0 | ölü |

**Engel artık nefes değil, $\Delta R$'nin koşul sayısı:** cond$(\Delta R)=
3.7\times10^4$; antisim/sim yön kazanç oranı ~**1393×** (no-go tablosuyla aynı
mertebe). Simetrik (orbit-görünmez) alt-uzay $\Delta R$'nin **küçük tekil
değerlerine** düşer → tek-atış BPM gürültüsü ters-çevrimde ~$1/\sigma_{\min}$ ile
büyür.

> **Kritik kavramsal ayrım — "nefes ≠ no-go":** Dağıtık-frekansta suçlu **nefes**
> (koherent sistematik, köşegen okumayı kirletir). v2.7'de nefes çözülmüş; suçlu
> **no-go** (kötü koşullu matrisin ters-çevriminde **rastgele** gürültünün
> büyümesi). İkisi farklı problemlerdir — ve §9'da göreceğimiz gibi, ikincisi
> (rastgele) *zamanla* yenilebilir, birincisi (koherent) yenilemez.

Repro: `/tmp/kmod_recover/v27_recheck.py`.

---

## 9. Tek-frekans + lock-in: BPM gürültüsünü *zamanla* yenmek (v2.7 canlanıyor)

§8'deki "σ=10 μm → corr≈0" sonucu **tek-atış bağımsız gürültü** varsayar (iki
konfig ayrı ölçülür → farkta $\sqrt2\,\sigma$). Ama modülasyon gerçekte **~1 kHz**
hızla yapılır. Bu, gürültü modelini kökten değiştirir.

### 9.1 Üç gürültü bileşeni, üç farklı davranış
1. **Statik ofset (~100 μm):** iki fazda aynı → **farkta tam iptal.** (K-mod'un
   var olma sebebi; §8 zaten bunu yapıyor.)
2. **Yavaş drift (termal, ms'ten yavaş):** 1 kHz'de iki faz ~0.5 ms arayla okunur;
   drift bu arada değişmez → **iptal.** *(Kullanıcının kilit gözlemi: hızlı
   modülasyon, $\Delta R$'nin hesaplandığı iki an arasında BPM ofsetinin/driftinin
   değişmemesini garanti eder.)*
3. **Beyaz/hızlı elektronik gürültü:** iptal olmaz — ama **lock-in (senkron
   demodülasyon)** ile yalnız $f_{\rm mod}$ çevresindeki dar banttaki gürültü kalır;
   $N$ modülasyon çevriminde **$\sqrt N$ ile bastırılır**:
   $$\sigma_{\rm eff} \;=\; \frac{\sqrt2\,\sigma_{\rm tek-atış}}{\sqrt N},
   \qquad N = T_{\rm ölçüm}\times f_{\rm mod}.$$

### 9.2 Neden lock-in v2.7'ye yarar ama dağıtık-frekansa yaramaz
Lock-in **rastgele** gürültüyü ortalar, **koherent** sistematiği değil.
- v2.7'nin düşmanı: beyaz gürültü × kötü koşul → **rastgele** → lock-in **yener**.
- Dağıtık-frekansın düşmanı: **nefes** → **koherent, deterministik** → lock-in
  *yenemez* (her çevrimde aynı yanlı katkı; ortalaması sıfır değil).

Bu, §8'deki "nefes ≠ no-go" ayrımının operasyonel sonucudur ve v2.7'nin neden
doğru yol olduğunu gösterir.

### 9.3 Sonuçlar ($f_{\rm mod}=1$ kHz, analitik $\Delta R$, 40 seed ort.)

**σ = 10 μm tek-atış BPM gürültüsü:**

| $T_{\rm ölçüm}$ | $N$ çevrim | $\sigma_{\rm eff}$ | corr | SNR>1 mod |
|---|---|---|---|---|
| 1 ms (ortalama yok) | 1 | 14 μm | 0.001 | 5/48 |
| 10 s | 10⁴ | 141 nm | 0.651 | 31/48 |
| 100 s | 10⁵ | 45 nm | 0.930 | 36/48 |
| 1000 s (~17 dk) | 10⁶ | 14 nm | **0.992** | 43/48 |

**σ = 1 μm:** 10 s → corr 0.992; 100 s → 0.999 (46/48).

**Okuma:** lock-in beyaz gürültüyü $\sqrt N$ ile yener; corr 0.99'a çıkar.

> **⚠️ corr bir TUZAK — simetrik felaketi gizler (§9.5'te niceliği).** corr 0.99,
> 48 modun çoğunu oluşturan *kolay antisimetrik* modlarca domine olur ve ölçek/
> ofset-değişmez olduğu için **üniform (simetrik) hatayı görmez.** §9.5 gösteriyor
> ki simetrik (orbit-görünmez, sahte-EDM'i süren) alt-uzay, **lock-in tabanında
> bile** kurtarılamıyor. "corr 0.99 → çözüldü" çıkarımı **yanlıştır.** Doğru metrik:
> simetrik-bileşen hatası (mean-çıkarmadan, gerçek RMS).

### 9.4 SQUID BPM'in (yeniden anlam kazanan) rolü
§7'de SQUID *dağıtık-frekans + genlik-okuma* için işe yaramıyordu (orada sorun
gürültü değil nefesti). Burada sorun **gerçekten gürültü** — ve $\sigma_{\rm eff}
\propto \sigma_{\rm tek-atış}$. SQUID'in düşük tek-atış gürültüsü, aynı corr'a
**çok daha kısa entegrasyonda** ulaşmayı sağlar (σ: 10→1 μm, aynı corr ~100× daha
kısa $T$). Yani **SQUID, tek bir modülasyon frekansıyla da değerlidir** —
modülasyonu farklı frekanslara bölmek şart değil.

### 9.5 Gerçek taban: $\Delta R$ sistematik doğruluğu — ve simetrik no-go (ölçüldü)

Beyaz gürültü zamanla yenildiğine göre, v2.7'nin gerçek sınırı **$\sqrt N$ ile
düşmeyen** şeylerdir. En önemlisi **$\Delta R$ model/kalibrasyon hatası** (β-beat,
gradyan kalibrasyonu): bir *sistematik*tir, hiç ortalanmaz. Modeli: gerçek latiste
per-quad gradyan hatası $\delta$ var ($\Delta R_{\rm true}$), inversiyon nominal
$\Delta R_{\rm model}$ ile yapılır; ölçüm $\Delta y = \Delta R_{\rm true}\,dy +$
(lock-in sonrası 10 nm artık). Sonuç (`v27_syst2.py`, 30 seed; $dy$ RMS=57.7 μm,
sim-bileşen 41 μm, anti-bileşen 60 μm):

| $\sigma_g$ | corr | gerçek hata RMS | **sim-bileşen hata** | anti-bileşen hata |
|---|---|---|---|---|
| %0 (lock-in tabanı, 10 nm) | 0.996 | 15 μm | **96 μm** | 0.08 μm |
| %0.5 | 0.703 | 291 μm | 1931 μm | 11 μm |
| %1.0 | 0.485 | 581 μm | 3848 μm | 22 μm |
| %5.0 | 0.125 | 3971 μm | 26368 μm | 155 μm |

**İki kritik bulgu:**

1. **Simetrik alt-uzay lock-in tabanında BİLE kurtarılamıyor (gürültüden bağımsız
   no-go).** $\sigma_g=0$, model mükemmel, yalnız 10 nm artık beyaz gürültü olsa
   bile **sim-bileşen hatası 96 μm** — sinyalin kendisinden (41 μm) büyük. Sebep:
   $\sigma_{\min}(\Delta R)\approx10^{-4}$; 10 nm ölçüm hatası bu yönde
   $10\,{\rm nm}/10^{-4}\approx100\,\mu$m geri-çatım hatası verir. Simetriği
   kurtarmak için $<4$ nm efektif gürültü gerekir → pratik dışı uzun entegrasyon.
   **Antisimetrik kısım ise mükemmel (0.08 μm).**
2. **$\Delta R$ sistematik hatası felaketi büyütür.** %0.5–1 gradyan/β-beat hatası:
   toplam hata sinyalin 5–10 katı, corr 0.70/0.49. Antisimetrik kısım görece dayanır
   (11–22 μm), ama simetrik yön cond=$3.7\times10^4$ ile patlar (1931–3848 μm).

**Sonuç — v2.7+lock-in'in dürüst sınırı:** Lock-in BPM *gürültüsünü* yener ve
**antisimetrik (orbit-görünür)** misalignment'ları iyi kurtarır (drift makalesinin
kullandığı kısım), %1 model hatasına makul dayanıklı. Ama **sahte-EDM'i süren
simetrik (orbit-görünmez) kısmı kurtaramaz** — ne lock-in tabanında, ne %0.5 model
hatasıyla. Bu, simetrik alt-uzayın neden **spin (Omarov) veya "akıllı düzeltme"
(§10)** gerektirdiğinin orbit-tarafı kanıtıdır.

> **Diğer ortalanmayan sınırlayıcılar (tasarımda dikkat):** modülasyonla-korelasyonlu
> pickup (quad güç kaynakları 1 kHz'de BPM'e sızarsa → koherent → $\sqrt N$ düşmez);
> 1/f BPM gürültüsü (1 kHz'de genelde önemsiz ama kontrol edilmeli).

---

## 10. Kalan yol: NN ileri-harita + "akıllı düzeltme"

Lineer inversiyon (v2.7) en iyi durumda $\Delta R$ sistematiğiyle sınırlı. Ondan
*kavramsal olarak* farklı bir yol — misalignment'ı geri-çatmak yerine doğrudan
**sahte-EDM'yi hedeflemek**:

- NN ile **COD → sahte-EDM ileri-haritası** öğren; sıfır sahte-EDM için kapalı
  yörüngenin nasıl modifiye edileceğini öğret; orbit-görünür knob'larla (quad/
  corrector) sahte-EDM'yi null'la — simetrik misalignment'ı *bilmeden*.
- **Neden no-go'yu atlayabilir (hipotez):** ileri-harita (inversiyon değil) +
  orbit-görünür kollarla **EDM-hedefli** düzeltme.
- **Şüphe:** NN gözlenebilirlik tabanını yenemez; tilt/β-beat dayanıklılığı kritik.

Ayrıntı ve plan: **`YAPILACAKLAR.md §4`** ("akıllı düzeltme" notu). Bu, bu belgenin
kapsamı dışında, sonraki çalışma kalemidir.

---

## 11. Reprodüksiyon

```bash
python3 /tmp/kmod_recover/single_bpm_test.py     # §1-7: dağıtık-frekans, nefes [1]/[2]/[3]
python3 /tmp/kmod_recover/breathing_cpp.py --nq 6   # §5.5: nefesin C++ doğrulaması
python3 /tmp/kmod_recover/v27_recheck.py         # §8: tek-frekans ΔR, no-go (cond, corr)
python3 /tmp/kmod_recover/v27_lockin.py          # §9: lock-in (T vs corr, σ_eff)
python3 /tmp/kmod_recover/v27_syst2.py           # §9.5: ΔR sistematik taban + simetrik no-go
```
Tümü analitik çözücüye dayanır (`analytic_kmod.py` yapı taşları + `single_bpm_test.
closed_orbit_at_quads`), `params.json`, $\varepsilon=0.02$, seed=0, dikey düzlem.
`breathing_cpp.py` gerçek C++ izleyiciyi `quad_dG` ile kullanır (`integrator.cpp`
**değiştirilmez**); ~40 s/kapalı-yörünge. Diğerleri saniyeler içinde koşar.
