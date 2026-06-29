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
> Reprodüksiyon: `/tmp/kmod_recover/single_bpm_test.py` (analitik; C++ gerekmez).

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

## 6. Teşhis — neden başarısız oldu?

**Suçlu optik-nefes** (kör noktalar ikincil):

- Modülasyon quad'ı titretince, **mevcut 0.37 mm'lik kapalı yörüngenin tamamı**
  nefes alır.
- Büyüklük tahmini: %2 gradyan bump → tune kayması ~%0.04, ama kapalı-yörünge
  ölçeği $\propto 1/(2\sin\pi Q)$ ~%0.66 değişir → $0.37\,{\rm mm}\times0.66\%
  \approx 2.4\,\mu$m BPM'de.
- Doğrudan per-quad feed-down sinyali: $\sim K\varepsilon L\,dy \times G \approx
  0.9\,\mu$m.
- **Nefes (~2.4 μm) > doğrudan sinyal (~0.9 μm), ~3×.** Genliği nefes domine
  ediyor → aradığımız quad ofseti gürültüye gömülüyor.

**[2] (nefessiz) corr=1.000 ama [1] (nefesli) corr=0 olması, suçlunun kesin
olarak optik-nefes olduğunu kanıtlar.**

**Çok BPM neden kurtarmıyor?** Nefes, **koherent** bir kontaminasyondur (≈ mevcut
yörünge × tek bir skaler), rastgele gürültü değil. Bu yüzden BPM sayısı arttıkça
**ortalamayla sönmez**; per-quad projeksiyon her quad'da **yanlı** kalır → 48
BPM'de bile corr ≈ 0.

---

## 7. Dürüst çıkarım

- Bu oturumdaki **`ac_bba_observability.py` (simetrik corr=0.997)** tam olarak
  **[2] (nefessiz)** modelini kullanıyordu. Gerçek fizikte (nefes dahil) o sonuç
  **çöküyor**. Dolayısıyla "per-quad AC-BBA genlik-okumayla simetrik modu görür"
  iddiası ve ona dayalı linchpin/bütçe sayıları **YANLIŞLANDI**.
- Bu, geçmiş kayıtla **tutarlı**: `README §19.2` per-quad k-mod'u "operasyonel
  olarak ağır" diye işaretliyordu. Gerçek BBA tekniği **genlik okuma değil**,
  per-quad **nulling**'tir (modülasyon yanıtı sıfırlanana dek demeti yönlendir →
  demet quad merkezinden geçiyordur). Tek-atış "genliği oku, kalibrasyona böl"
  yaklaşımı, büyük mevcut yörünge nefes alırken çalışmıyor.
- **Fork** (pozitif yöntem mi / negatif teorem mi) bu testle **negatif tarafa**
  ağırlık kazandı: orbit-tarafı genlik-okuma, nefes/inversiyon problemini atlamıyor.

---

## 8. Açık sorular (belgeyi okuduktan sonra irdelenecek)

1. **Modelde hata var mı?** Özellikle: geri-çatımı "genlik ÷ kalibrasyon" yerine
   **nulling** (per-quad yönlendirme ile yanıtı sıfırlama) olarak kurmak gerekir
   mi? Nulling, nefesi (koherent fon) yener mi?
2. Nefes ve feed-down farklı parametrik desende; **ortak fit** (nefes + per-quad)
   ikisini ayırabilir mi — yoksa bu yine bir inversiyon → no-go'ya mı döner?
3. Adyabatik sonlu-fark eşdeğeri kabul edilebilir mi, yoksa gerçek AC yanıtının
   (sonlu frekans transfer fonksiyonu) bir etkisi var mı?

---

## 9. Reprodüksiyon

```bash
python3 /tmp/kmod_recover/single_bpm_test.py
```
Analitik çözücü (`analytic_kmod.py` yapı taşları), `params.json`, $\varepsilon=0.02$,
seed=0, dikey düzlem. [1]/[2]/[3] ve kör-nokta sayısını basar.
