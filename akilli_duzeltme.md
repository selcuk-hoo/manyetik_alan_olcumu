# akilli_duzeltme.md — "Akıllı düzeltme": sahte-EDM'i null'lamak (misalignment'ı DEĞİL)

> **Durum (2026-06-29, DÜZELTİLDİ).** Bu belge, `SONRAKI_OTURUM_PROMPT.md`'de
> tanımlanan "akıllı düzeltme" fikrinin testidir. Fikir: quad misalignment'ı
> geri-çatmak yerine (bu *inversiyon* no-go'su), sahte-EDM'i **doğrudan** null'lamak.
> İki kol:
> - **Kol A (spin-gözlemli):** dS_y/dt'yi spinle ölç → corrector knob'larıyla null'la
>   (spin-trim'in genellemesi; `false_edm_harmonic_sinir.md §14.6` ~6000×). **Çalışır.**
> - **Kol B (NN ileri-harita):** 48-BPM kapalı yörüngeden (COD) sahte-EDM'in
>   **ileri-haritasını** öğren → orbit-görünür knob'larla EDM-hedefli düzelt.
>
> **ÖNEMLİ DÜZELTME (kullanıcı itirazı, 2026-06-29):** Bu belgenin ilk sürümü
> "Kol B ölü; inversiyon no-go'su ile aynı duvar" diyordu. **Bu YANLIŞTI.**
> Kullanıcının haklı itirazı: *simetrik alt-uzay kapalı yörüngeyi az dahi olsa
> değiştirir; dejenerasyon tam değildir; harita VARDIR, ama basit değildir.*
> Yeniden türetince:
>
> - **İleri-haritanın duyarlılığı ∂f/∂COD ≈ 0.15 rad/s/m'dir — MÜTEVAZI ve her iki
>   alt-uzayda ~eşit.** Skaler f'i COD'dan öngörmek, 48 misalignment'ı geri-çatmak
>   (inversiyon) gibi **1/σ_min ile büyütülmez.** Yani Kol B inversiyon DEĞİLDİR ve
>   onun duvarına çarpmaz. (§4)
> - f'i hedefe (1 nrad/s) çekmek için gereken COD doğruluğu ~7 nm; ama bu **basit
>   ortalama** ile ulaşılır (σ_BPM/√N: 1μm tek-atıştan ~21 s @1kHz). Yani **ölçüm
>   gürültüsü Kol B'yi dışlamaz.** (§4)
>
> **Doğru durum: Kol B AÇIK + POZİTİF (dört bulgu).** Gerçek engel *gözlenebilirlik*
> değil, fonksiyonelin karmaşıklığı — ve o fonksiyonel **öğrenilebilir + dayanıklı
> çıktı:** (1) ileri-harita iyi koşullu (∂f/∂COD≈0.15, inversiyon değil, §4);
> (2) orbit-kör simetrik kanal temiz COD'dan öğreniliyor (CV R² 80 örnekte ~0 →
> 240'da **+0.77**, §6); (3) **β-beat model-fidelity ŞEFFAF** — %1 β-beat'li
> makineye sim-eğitimli harita held-out nominal kadar taşınıyor (R² 0.62 vs 0.61,
> §6.5); (4) birleşik no-go ileri-haritayı kapsamaz. "Harita vardır, basit değildir"
> tam doğru; inversiyon no-go'su Kol B'yi **bağlamaz.**
>
> **AMA pratik sınav (Plan 2c, §6.10) NEGATİF:** kapalı-döngü null'lama, basit
> orbit-düzeltmeyi geçemiyor — 240-örnek haritanın mutlak hata tabanı (~300× hedef)
> çok kaba, corrector'ları yanlış yönlendiriyor. Yani 4 pozitif bulgu *gerekli ama
> yeterli değil.* **Eksik: mutlak harita doğruluğu** → Plan 4 (analitik Berry
> fonksiyoneli, exact) veya çok daha fazla veri. **Kol B kavramsal açık, pratik
> HENÜZ çalışmıyor;** "çözüldü" değil. `orbit_ileri_olcum.md §7` açık problemi
> pozitif ilerledi ama kapanmadı.
>
> **Kol A** değişmedi: çalışır (bilgi spinde), ama orbit-tarafı değil — bizim
> spin ölç-trim'imiz (§14.6). **NOT:** bu Omarov'un SBA'sı DEĞİL; SBA E-alan/
> vertical-velocity eksenini hizalar, geometrik fazı doğrudan düzeltmez (`omarov.md
> §5`). Geometrik fazı spinle doğrudan ölçüp null'lama, Omarov'da *vertical-
> polarizasyon* koluna karşılık gelir — ki o makalede **kullanılmadı**.

---

## 0. Okunması gereken SOLID temel

`omarov.md §5,§9,§10`; `orbit_ileri_olcum.md §2-3,§5,§6,§7`;
`squid_bpm_test.md §8,§9,§9.5,§10`; `false_edm_harmonic_sinir.md §14.6`;
`YAPILACAKLAR.md §4`; `CLAUDE.md`.

**Kesinleşmiş zemin:** sahte-EDM = geometrik (Berry) faz, σ²-ölçeklenir (p=2.00; bu
oturumda p=2.002 yeniden doğrulandı); kaynağı dx·dy. Sahte-EDM'i süren şey
**simetrik (QF/QD aynı-işaret, orbit-kör)** misalignment alt-uzayıdır.

---

## 1. Yöntem ve doğrulama

**Sahte-EDM ölçer (C++, doğrulandı):** `berry_data/false_edm_4d.py` /
`kmod_drivers/fast_est.fast_measure` (4D-CO + model-fit). Bu oturumda yeniden
doğrulandı: σ=10→5μm'de **p = 2.002** (saf σ² geometrik faz, lineer kaçak yok).
Tek koşum ~171 s (4 çekirdek).

**Hızlı COD yüzeyi (analitik):** misalignment → 48-BPM COD, `analytic_kmod.
build_R_analytic`. C++ ile %1 içinde doğrulanmış (`squid_bpm_test.md §5.5`).

---

## 2. Geometri: orbit-görünür ↔ orbit-kör SVD ayrışımı

R = U Σ Vᵀ (Q=2.303):

| | σ | Ŝ | karakter |
|---|---|---|---|
| en büyük (mod 0–5) | 28.4 … 8.6 | −0.92 … −0.95 | orbit-GÖRÜNÜR (antisim) |
| en küçük (mod 38–47) | 0.16 … **0.147** | +0.74 … +0.96 | orbit-KÖR (simetrik) |

cond(R) = 193 (← `drift_monitor/permode2.py` κ=193 ile birebir). Büyük-σ antisim;
küçük-σ simetrik. **Önemli:** küçük-σ ≠ sıfır-σ. Simetrik kaçıklık COD'u 193 kat
**bastırır ama yok etmez** — bu, Kol B için kritik (bilgi orada, sadece küçük).

---

## 3. Dejenerasyon ölçümü (ham veri — yorumu §4'te DÜZELTİLDİ)

Sabit orbit-görünür taban A (10 μm) + iki sınıf pertürbasyon (RMS 10 μm):
simetrik (en küçük-16 σ) ve antisimetrik (en büyük-16 σ). Her biri gerçek C++ ile
ölçüldü (taban A: |f|=5.58e-7).

| sınıf | ort ΔCOD_RMS | ort \|Δf\| | f-kaldıracı \|Δf\|/ΔCOD |
|-------|-------------:|----------:|------------------------:|
| Antisim (görünür) | 114.7 μm | 9.27e-6 | 0.081 rad/s/m |
| Simetrik (kör) | **1.69 μm** | 2.47e-7 | **0.146 rad/s/m** |

**Ham gözlem:** simetrik 10 μm kaçıklık f'i ~247× hedef oynatır ama COD'da yalnız
1.7 μm iz bırakır. **İlk (yanlış) yorum:** "COD f'i belirlemez, Kol B ölü."
**Doğru yorum (§4):** 1.7 μm iz **sıfır değil** ve f-kaldıracı (0.146) ona göre
**mütevazıdır** → bilgi COD'da var ve okunabilir.

![dejenerasyon](/tmp/akilli_duzeltme/fig_kolb_dejenerasyon.png)

---

## 4. DÜZELTME: Kol B inversiyon DEĞİLDİR (koşullanma doğru analizi)

İlk sürümdeki hata: "simetrik f'i hedefe çekmek 7 nm BPM ister = §9.5 inversiyon
duvarı." Bu, **ileri-harita ile inversiyonu karıştırıyordu.** Doğrusu:

**(a) İleri-harita iyi koşulludur.** f'in COD'a duyarlılığı ölçüldü:
$$\frac{\partial f}{\partial \text{COD}}\Big|_\text{sim} = 0.146,\quad
\frac{\partial f}{\partial \text{COD}}\Big|_\text{anti} = 0.081\ \text{rad/s/m}.$$
**İkisi ~eşit ve mütevazı.** İnversiyon (COD→48 misalignment) simetrik yönde
hatayı **1/σ_min ≈ 7×** (tam matris ΔR'de ~10⁴×) büyütür; ama **skaler f'i**
doğrudan öngörmek bu büyütmeyi içermez. Çünkü simetrik misalignment'ın *hem*
COD'u (σ_min) *hem* f-üretimi (∂f/∂mis) küçüktür; oranları (kaldıraç) normal kalır.

**(b) Gereken hassasiyet ulaşılabilir.** f'i 1 nrad/s'e öngörmek için COD
f-yönünde ~7 nm bilinmeli:
$$\delta\text{COD} = \frac{10^{-9}}{0.146} \approx 6.9\ \text{nm}.$$
Bu **inversiyonun <4 nm'siyle sayısal olarak benzer ama anlamı tamamen farklı:**
inversiyonda <4 nm, 1/σ_min büyütmesinden *sonra* gerekir (β-beat sistematiğiyle
felakete döner, §9.5); ileri-haritada 7 nm **doğrudan ölçüm gürültüsüdür** ve
basit ortalama ile yenilir:
$$\sigma_\text{eff} = \sigma_\text{BPM}/\sqrt{N};\quad
1\,\mu\text{m}/\sqrt{2\times10^4} \approx 7\,\text{nm}\ (\sim21\ \text{s @1 kHz}).$$
Üstelik Berry yönlü-alan özniteliği BPM **ofsetine değişmezdir** (kapalı halkada
sabit ofset toplamda iptal) → 100 μm ofset bile duvar değil (§6).

> **Sonuç:** "Kol B = inversiyon no-go, aynı duvar" iddiası **GERİ ÇEKİLDİ.**
> İleri-harita iyi koşulludur ve ölçüm gürültüsüyle dışlanmaz. **Birleşik no-go,
> Kol B'yi KAPSAMAZ** — yalnız orbit-inversiyon + orbit-lock-in (misalignment
> geri-çatımı) için geçerlidir.

---

## 5. Gerçek engel: fonksiyonelin karmaşıklığı (orbit_ileri §3,§7)

Kol B'nin zorluğu gözlenebilirlik değil, **doğru ileri-haritayı bulmak.** f, COD'un
**doğrusal-olmayan, yapısal** bir fonksiyonelidir (Berry yönlü-alan tipi); basit
⟨x·y⟩ veya tek-mod proxy yanlış (`orbit_ileri_olcum.md §2-3`). Empirik pinleme
40–80 config ile yakınsamıyor (§6); ya **çok daha fazla veri** ya da **analitik
türetme** (`orbit_ileri_olcum.md §5`: Thomas-BMT'den Berry fazı) gerek.

Yani Kol B iki şeye bağlı, **ikisi de açık:**
1. **Öğrenilebilirlik:** karmaşık simetrik fonksiyonel yeterli veriyle pinlenir mi?
2. **Model-fidelity:** sim'de öğrenilen harita, β-beat/bilinmeyen sistematik altında
   gerçek makineye taşınır mı? (İnversiyon burada 1/σ_min ile ölür; ileri-harita
   ∂f/∂COD≈0.15 ile *muhtemelen daha dayanıklı* — ama **test edilmedi.**)

---

## 6. Empirik durum: ileri-harita ÖĞRENİLEBİLİR (240 örnek, POZİTİF)

Kontrollü simetri ağırlığı w∈{0,…,1} (w=1 tam orbit-kör), girdi=analitik temiz COD,
çıktı=C++ f. (`gen_ensemble.py` 240 örnek + GradBoost ham-COD+bilineer öznitelik.)

**(a) Orbit-kör konfigler sahte-EDM'siz DEĞİL:** ⟨|f|⟩ w ile düşer ama sıfırlanmaz —
tam simetrik (w=1) bile **182× hedef** taşır → simetrik kanal gerçek, kapatılmalı.

**(b) KARARLI BULGU — simetrik kanal yeterli veriyle ÖĞRENİLİYOR.** 5-kat CV R²,
**simetrik alt-küme** (w≥0.75, orbit-kör — asıl iş) üzerinde, örnek sayısıyla:

| N | genel R² | **simetrik-kanal R²** |
|---|---------:|----------------------:|
| 60 | +0.36 | **−0.52** (az veri, öğrenilemiyor) |
| 100 | +0.53 | **+0.64** |
| 180 | +0.58 | **+0.74** |
| 240 | +0.62 | **+0.77** |

Orbit-KÖR simetrik kanal (sahte-EDM'i süren), temiz kapalı yörüngeden **öğreniliyor**
(R² −0.5 → 0.77, doygunluk ~0.77). Yalnız-simetrik eğitim (w≥0.5) bile CV R²=+0.40.
→ **Harita VAR, karmaşık (80 örnek yetersizdi), yeterli veriyle pinleniyor.** Bu,
kullanıcının teşhisinin ("harita vardır, basit değildir") doğrudan doğrulamasıdır;
`orbit_ileri_olcum.md §3`'ün "40 config ile yakınsamıyor"unu da açıklar (veri azdı).

![öğrenilebilirlik](/tmp/akilli_duzeltme/fig_kolb_ogrenilebilirlik.png)

**(c) Ofset incelik:** yönlü-alan özniteliği BPM-ofsetine değişmez → +100μm ofset
R²'yi düşürmez. Ofset duvar değil.

---

## 6.5 Model-fidelity: β-beat altında harita taşınır mı? (POZİTİF)

Kritik soru: ileri-harita **simülasyondan** (nominal optik) öğrenilir; gerçek makinede
β-beat (per-quad gradyan hatası) var. Sim'de öğrenilen harita gerçeğe taşınır mı?
(İnversiyon burada 1/σ_min ile ölür, §9.5.) Test: %1 per-quad β-beat (LOCO seviyesi,
tek makine realizasyonu, `quad_dG` ile C++); harita 200 nominal config'de eğitilir
(test config'leri **dışlanır**, sızıntı yok), 40 β-beat config'de test edilir.
COD_bb, `perquad_orbit.py` (β-beat kapalı yörünge, build_R ile %0.2 doğrulanmış).

| ölçüm | R² |
|-------|---:|
| held-out NOMİNAL (referans generalleme) | +0.61 |
| **β-beat transfer: F(COD_bb) vs f_bb** | **+0.62** |
| β-beat, simetrik alt-küme (w≥0.75) | **+0.83** |

**Bulgu: β-beat ŞEFFAF.** %1 β-beat f'i **ortalama ~%18** oynatır (COD'u 3.2μm), ama
nominal-eğitimli harita β-beat makineyi **held-out nominal kadar iyi** öngörür (0.62
vs 0.61 — **ek bozulma YOK**). Harita, β-beat kaymasını COD üzerinden kısmen izler
(corr(Δpred,Δf)=+0.35). Yani **f, yörüngenin ~sabit fonksiyonelidir; β-beat haritayı
kırmaz** (inversiyonun aksine). Kalan %52 nispi hata, β-beat değil **veri-sınırlı
generalleme** hatasıdır (N ile düşer; §6 trendi).

> **β-beat kayma dağılımı (config-bağımlı, güçlü kuyruk):** ortalama %18 ama ortanca
> yalnız %11; std %22; **max %96** ve config'lerin **~%12'si %40'ı aşıyor.** Yani
> bazı desenler β-beat'e çok duyarlı (%40+), bazıları neredeyse duyarsız — etki
> desene göre değişken. (Bu kuyruk ileri-haritayı yine **bozmaz**; harita gerçek
> yörüngeyi gördüğünden kaymış f'i izler. β-beat'in *felaket* yarattığı yer
> inversiyondur: %0.5 β-beat → 1931 μm, `squid_bpm_test §9.5`.)

![β-beat fidelity](/tmp/akilli_duzeltme/fig_kolb_bbeat.png)

**Açık kalan tek şey:** mutlak doğruluk (şu an N=200'de ~%52 nispi → tek-atış
null'lama ~2×; daha çok veri/öznitelik/analitik form + iterasyon ile hedefe) ve
gerçekçi-ölçüm gürültü bütçesi (§4: 7nm ortalama). Model-fidelity artık **engel
değil.**

---

## 6.6 Harita NASIL kullanılır? Çıkarma DEĞİL, null'lama (kritik ayrım)

**Kullanıcı itirazı (haklı):** Haritayı "öngör + polarimetreden ÇIKAR" diye
kullanırsak işe yaramaz. Sahte-EDM ~1000× sinyal; çıkarmanın hedefin altında artık
bırakması için haritanın **mutlak** doğruluğu ~**%0.1** olmalı (= hedef/sahte-EDM).
Harita %22–52'de → çıkarma sonrası artık ~%22×1000 = **220× sinyal**, hâlâ gömülü.
**Çıkarma ÖLÜ.** Üstelik bu, β-beat değil **sonlu-veri** doğruluğuyla sınırlı.

**CW/CCW + quad-flip β-beat belirsizliğini geri alır mı? — HAYIR (ölçüldü).**
Umut: gerçek EDM=(S_CW−S_CCW)/2; harita hatası (paylaşılan optik) CW/CCW'de ortak-mod
ise farkta iptal olur. Test (12 config × CW/CCW × nom/β-beat, signed C++):

| ölçüm | sonuç |
|-------|-------|
| CW/CCW sahte-EDM iptali \|f_CW\|/\|D\| | ~2.2× (Omarov 3.4× ile uyumlu, zayıf) |
| β-beat ortak-mod corr(δ_CW, δ_CCW) | **−0.89 (ANTİ-korele!)** |
| ortak-mod iptal kazancı | **0.7× (iptal YOK)** |
| β-beat'in EDM-kanalı D'ye etkisi | **70× hedef** (D_nom'un kendisi kadar) |

**Bulgu: β-beat sahte-EDM kayması CW/CCW-TEK'tir** (δ_CW≈−δ_CCW) → tam da gerçek
EDM gibi (CW−CCW)/2'de **hayatta kalır**, iptal olmaz. Yani CW/CCW+flip β-beat
belirsizliğini **geri ALMAZ**; aksine β-beat doğrudan EDM-mimik kanalına düşer.

![β-beat CW/CCW](/tmp/akilli_duzeltme/fig_kolb_cwccw.png)

**Doğru kullanım — NULL'LAMA (çıkarma değil):** Haritayı, sahte-EDM'i orbit-görünür
knob'larla **sıfıra sürmek** için kullan (predicted f → 0). Bu, mutlak %0.1 doğruluk
İSTEMEZ — harita yalnız *yönü* gösterir; null'a yaklaştıkça gerçek f de küçülür
(çarpımsal hata). β-beat burada **§6.5'teki gibi şeffaftır** çünkü haritaya **gerçek
(ölçülen) yörünge** beslenir — nominal değil. Sınır: harita doğruluğu + iterasyon
(veriyle iyileşir) + gürültü bütçesi. **Çıkarma yolu ölü; null'lama yolu canlı ama
mutlak hedefe ulaşması veri/iterasyona bağlı.**

---

## 6.7 Büyük genlikte eğitim (1 mm) — homojenlik kanala göre değişir (KISMEN POZİTİF)

**Kullanıcı fikri:** NN'i beklenen (~10 μm) yerine **1 mm** mertebesi hatalarla eğit.
Mantık: büyük genlikte sinyaller temiz (COD, f büyük → BPM/estimator gürültüsü
önemsiz); **gerçek makinede** öğren (sim-gerçek farkı yok); sonra σ² yasasıyla
operasyon genliğine ölçekle. Bu, ancak fonksiyonel **ölçek-değişmez (f∝σ²)** ise işe
yarar — yoksa 1 mm'de öğrenilen harita doyma/yüksek-mertebe ile kirlenir.

**Ölçtük** (aynı desen, σ=10→1000 μm, gerçek C++):

| genlik | antisim f/σ² | simetrik f/σ² | simetrik p |
|--------|-------------:|--------------:|-----------:|
| 10 μm | 2705 | 461 | — |
| 100 μm | 2419 (p=1.93) | 464 | 2.00 |
| 1 mm | 488 (p=0.93) | 486 | **2.03** |
| **1mm→10μm ölçekleme hatası** | **−82% (5.5× yanlış)** | **+5.6% (homojen)** |

**Bulgu — homojenlik KANALA bağlı:**
- **Antisim (orbit-görünür) DOYAR:** 1 mm misalignment → orbit ~cm → büyük spin
  dönüşleri → geometrik faz σ²'den sapar (1 mm'de p≈0.9). 1 mm'de eğitip ölçeklemek
  **5.5× yanlış.**
- **Simetrik (orbit-KÖR, ASIL KANAL) HOMOJEN:** 1 mm simetrik misalignment → orbit
  yalnız ~170 μm (σ_min ile bastırılmış) → spin dönüşleri küçük kalır → **σ² 1 mm'e
  korunur** (p=2.03, ölçekleme hatası %5.6).

![homojenlik](/tmp/akilli_duzeltme/fig_kolb_homojenlik.png)

**Sonuç (zarif sentez):** İhtiyaç duyduğumuz kanal (simetrik, orbit-kör) **tam da
büyük genlikte güvenle uyarılabilen** kanaldır — çünkü orbit-körlük sayesinde 1 mm
misalignment'ta bile orbit küçük (~170 μm, lineer) kalır, doymaz. Yani:
**orbit-kör alt-uzayda büyük (≈1 mm) misalignment desenleri uygula → simetrik
fonksiyoneli gerçek makinede temiz öğren (f spinle hızlı/büyük, COD 170 μm net,
β-beat içinde) → σ² ile operasyon genliğine ölçekle.** Antisim'i büyütme (cm orbit,
doyar+kararsız) — zaten orbit-düzeltmesi onu halleder.

**Caveat:** Sim'imiz lineer-latis (sekstüpol yok). Gerçek latiste sekstüpol/açıklık,
*büyük orbit* veren antisim'i daha da erken bozar; ama simetrik kanal küçük orbit
verdiği için görece korunur. Yani fikir **doğru yönde**, ama "1 mm rastgele (antisim
dahil)" değil, **"1 mm orbit-kör desen"** olarak uygulanmalı. Deployment'taki 7 nm
okuma problemi (§4) ayrı kalır (ortalama ile).

---

## 6.8 NN ile misalignment geri-çatımı R⁻¹'den iyi mi? — HAYIR (aynı taban)

**Kullanıcı sorusu:** Tepki matrisi (R⁻¹) yerine NN ile misalignment↔COD haritası
kurup, COD'den misalignment hesaplasak daha mı başarılı olur?

**Kritik gözlem:** misalignment→COD **tam lineerdir** (COD = R·mis). NN bunu öğrenince
yalnızca **R'yi** öğrenir (sömürülecek doğrusal-olmayan yapı yok). Ters yön
(COD→misalignment) = **R⁻¹**, simetrik alt-uzayda 1/σ_min ile **kötü koşulludur.**
Bu kötü-koşulluluk **R'nin (fiziğin)** özelliğidir, *algoritmanın* değil → hiçbir
yöntem (NN, TSVD, LASSO, CLEAN) gürültü tabanının altındaki simetrik bilgiyi
kurtaramaz.

**Sayıyla** (`nn_vs_Rinv.py`; 10μm misalignment, 1μm BPM gürültü, NN'e 4000 örnek):

| yöntem | antisim geri-çatım hatası | **simetrik geri-çatım hatası** |
|--------|--------------------------:|-------------------------------:|
| sinyal büyüklüğü | 10.0 μm | 9.9 μm |
| TSVD (klasik R⁺) | 0.63 μm | **6.3 μm** |
| NN (128,128) | 0.71 μm | **5.6 μm** |

İki yöntem de antisim'i iyi kurtarır (~0.7μm); **simetrik kanalda ikisi de aynı
tabana çarpar** (~6μm, sinyalin %60'ı kurtulamıyor). NN, TSVD'den iyi değil —
çünkü sınır **bilgi-teorik** (simetrik misalignment'ın COD izi gürültünün altında →
veride yok → hiçbir estimator çıkaramaz). Bu, projedeki 6-yöntem-aynı-taban
sonucunun (`false_edm_harmonic_sinir.md §14.5`) NN'le 7.'sidir.

**Ders (Kol B'nin neden farklı olduğu):** NN, *ters* problemi (COD→48 misalignment,
kötü-koşullu) çözmek için kullanılırsa R⁻¹ kadar başarısız. Kol B'nin işe yaramasının
sebebi, NN'i **ileri** problem için (COD→**skaler sahte-EDM**, iyi-koşullu
∂f/∂COD≈0.15) kullanmasıdır. **Fark algoritmada (NN vs matris) değil, problemin
yönünde:** misalignment'ı geri-çatma (kötü) vs sahte-EDM'i ileri öngör (iyi).
"Mükemmel COD okuyabilseydik bastırırdık" doğru — ama mükemmel okuma simetrik
alt-uzayda ~7nm ister; bu yüzden geri-çatım değil, **ileri-harita + null'lama** yolu.

---

## 6.9 kmod/SQUID/lock-in/LSTM hepsi AYNI ters-problem duvarı; çözüm yönleri

**Soru:** kmod + SQUID-BPM simetriği geri çatamıyordu (gürültüye gömülü); kmod+LSTM
o gürültüyü saf dışı bırakır mı? Başka çözüm?

**Önce kapatma — hepsi aynı INVERSİYON duvarı:**
- kmod = ΔR ölçer; **ΔR→misalignment** yine *ters* problem. Simetrik alt-uzayda
  σ_min(ΔR)≈10⁻⁴ → 1/σ_min büyütmesi (`squid_bpm_test §8`).
- SQUID düşük gürültü verir; lock-in beyaz gürültüyü √N ile yener — **ama simetrik
  yön <4nm efektif gürültü ister; lock-in tabanı bile yetmez ve %0.5 β-beat
  felaket** (`§9.5`). Dağıtık-frekans ayrıca *nefes* (koherent) ile ölür (`§7`).
- **LSTM/NN bunu yenemez.** İki sebep: (i) beyaz gürültü için **lock-in zaten
  optimal** (eşleştirilmiş süzgeç, Cramér-Rao); öğrenilmiş model √N'i geçemez.
  (ii) Asıl duvar gürültü değil **ΔR sistematiği (β-beat = MODEL hatası)** ve
  **kötü-koşulluluk (fizik)**; estimator değiştirmek bunları düzeltmez. Gerçek
  makinede LSTM eğitmek için *etiket* (gerçek misalignment) gerekir — ki ölçmeye
  çalıştığımız şey o (döngüsel). → §6.8'in zaman-serisi versiyonu, aynı sonuç.
- **Omarov CR-ayrım ölçümü de aynı duvar (DOĞRUDAN ÖLÇÜLDÜ).** CR-ayrım = CW−CCW
  kapalı yörünge farkı; bu da bir kapalı-yörünge-farkı → simetriğe **tek-yön COD
  kadar kör** (bastırma CR 4.5× ≈ COD 3.8×, gerçek C++; `omarov.md §9.3`). Ayrımı
  ölçüp küçültmek simetrik geometrik fazı bırakır. Omarov'un §9 boşluğu kapatıldı.

**Görünen çözüm yönleri (hepsi: TERS problemi BIRAK):**
1. **Spin-doğrudan (Kol A):** sahte-EDM'i spinle ÖLÇ (geometrik fazın doğrudan
   gözlenebiliri), knob'la null'la. Bizim spin ölç-trim'imiz, kanıtlı (~6000×,
   §14.6). Bilgi tanımı gereği spinde. Maliyet: spin ölçümü. **Omarov ile ilişki:**
   bu SBA DEĞİL (SBA E-alan hizalar); Omarov geometrik fazı CW/CCW+polarite+CR-ayrım
   ile kontrol eder, doğrudan spin-null'lama onun *vertical-pol* kolu = kullanılmadı.
2. **İleri-harita (Kol B):** COD→skaler sahte-EDM (iyi-koşullu, öğrenilebilir,
   β-beat-şeffaf) + null'lama. Orbit-tarafı umut; sınır: mutlak doğruluk + 7nm okuma.
3. **Büyük-genlik kalibrasyon (§6.7):** orbit-kör deseni ~1mm'e uyar, simetrik
   fonksiyoneli gerçek makinede temiz öğren (σ²-homojen), ölçekle.
4. **Analitik Berry fonksiyoneli (`orbit_ileri §5`):** COD→sahte-EDM'i Thomas-BMT'den
   kapalı-form türet → öğrenilen haritayı doğrula/değiştir (veri gerektirmez).
5. **(Teorik, pratik değil) Latis/tune mühendisliği:** G_k=C/|Q²−k²| bastırması
   Q'ya bağlı; Q'yu simetrik harmoniğe (k≈24) yaklaştırmak simetriği orbit-GÖRÜNÜR
   yapar — ama Q≈24 gerekir (gerçekçi değil; ertelenen latis-redesign).

**Tek cümle:** Simetrik bilgiyi *yörüngeden geri-çatmak* (kmod/SQUID/NN/LSTM)
fizik gereği imkânsız (bilgi gürültü+sistematik tabanın altında); çözüm ya **spinle
doğrudan ölçmek** ya da **ileri-harita ile öngörüp null'lamak** (ters'i atlamak).

---

## 6.10 KAPALI-DÖNGÜ NULL'LAMA (Plan 2c) — make-or-break: NEGATİF (mevcut harita)

Tüm pozitif bulguların (iyi-koşullu, öğrenilebilir, β-beat-şeffaf) **pratik sınavı:**
harita gerçekten sahte-EDM'i null'layabiliyor mu? İki test:

**(a) Yakınsama mı, taban mı? (proxy, `nulling_floor.py`)** Null'lama durduğunda gerçek
f = haritanın o konfigdeki hatası. Hata ÇARPIMSAL ise (f→0'da kaybolur) yakınsar;
TOPLAMSAL ise takılır. 240-ensemble + CV-harita: **log|hata| vs log f eğimi = 0.36**
(1=çarpımsal, 0=toplamsal → çoğunlukla toplamsal). En küçük-f konfiglerde bile
|hata| ≈ **3×10⁻⁷ = 300× hedef** (f'in kendisinden büyük). → Null'lama ~300× hedef
tabanına takılır.

**(b) Gerçek kapalı-döngü (`closed_loop.py`):** knob = orbit-görünür (büyük-σ) alt-uzayda
quad-mover (corrector eşdeğeri; COD=R·mis → harita dağılım-içi). Harita predicted |f|'i
knob'larla minimize → optimumda GERÇEK f'i C++ ölç. Karşılaştır: orbit-null (||COD||→0).

| makine | f0 (×hedef) | orbit-null | **MAP-null** | map kazanç |
|--------|------------:|-----------:|-------------:|-----------:|
| w0.5_0 | 364 | **0** | 1580 | 0.2× (KÖTÜ) |
| w0.5_1 | 152 | 2 | 785 | 0.2× (KÖTÜ) |
| w0.8_0 | 246 | 5 | 25 | 9.9× |
| w0.8_1 | 2266 | 6 | 100 | 22.7× |

**Sonuç: harita-güdümlü null'lama, basit orbit-düzeltmeyi GEÇEMİYOR** (hatta sıkça
*kötüleştiriyor*). Sebep: haritanın mutlak hata tabanı (~300× hedef) orbit-null'un
kalanından (0–6× hedef) **büyük** → harita corrector'ları yanlış yönlendiriyor.

![null'lama](/tmp/akilli_duzeltme/fig_kolb_nulling.png)

**Dürüst çıkarım:** Kol B *kavramsal olarak* sağlam (iyi-koşullu, öğrenilebilir,
β-beat-dayanıklı) AMA *pratikte* mevcut 240-örnek harita ile **çalışmıyor** —
null'lama için gereken mutlak doğruluğa (orbit-null kalanının altı) ulaşmıyor.
Bunu açmak için harita **çok daha doğru** olmalı: **Plan 4 (analitik Berry
fonksiyoneli, exact)** veya mertebelerce daha fazla veri. Empirik harita yetersiz.
*(Caveat: test makineleri jenerik; orbit-null onları zaten iyi düzeltti. En-kötü
simetrik artıkta harita daha da elverişsiz olurdu — sonuç güçlenir.)*

---

## 6.11 Algoritma iyileştirmesi: güvenli (belirsizlik-cezalı) null'lama

§6.10'daki naif null'lama, optimize ediciyi haritanın iyimser hatalarına sürüyordu
(model-istismarı). Üç sağlamlaştırma (`improved_null.py`): (1) **ensemble** (8 bootstrap
harita → ortalama + belirsizlik σ); (2) **güvenli/pessimistic** (minimize μ+β·σ →
kör-noktalardan kaç); (3) **regülarize** (μ+λ‖Δq‖).

Sonuç (gerçek C++, 4 makine, geomean ×hedef):

| yöntem | geomean | 
|--------|--------:|
| naif null (tek harita) | 236 |
| ensemble ortalama | 130 |
| **güvenli (ens+pessimistic)** | **67** (naif'ten 3.5×; bir makinede 1×) |
| regülarize | 188 |
| basit orbit-düzeltme | **2.1** (yine önde) |

**İki çıkarım:** (1) İyileştirme gerçek — güvenli optimizasyon model-istismarını
dizginler (naif 236×→67×), artık zararlı değil. (2) Ama **sınır aşılmıyor:** güvenli
null bile orbit-düzeltmeyi (2.1×) geçemez ve tutarsız (1×–726×). Kök neden §6.10'daki
duvar: **harita doğruluk tabanı.** Kör haritadan keskin bilgi çıkmaz. → Algoritma
iyileştirmesi gerekli ama yeterli değil; kök çözüm **daha doğru harita (Plan 4
analitik)**. ![](/tmp/akilli_duzeltme/fig_kolb_improved.png)

---

## 6.12 Plan 5: SPİN-modülasyon + gradient descent — POZİTİF (Kol B'nin çözümü)

**Kullanıcı fikri:** Quad'lar modüle edilince sahte-EDM de modüle olur → her knob'un
$\partial f/\partial\text{knob}$ gradyanını **spinle ölç** → gradient descent ile null'la.
Kol B'yi öldüren şey öğrenilmiş haritanın hatasıydı; burada gradyan **ÖLÇÜLÜR**
(harita yok, kör-nokta yok, model-istismarı yok). Ve spin simetrik modu **gördüğü**
için orbit-görünür knob'larla bile null'lanır.

**Test (`spin_descent.py`):** en-kötü **orbit-KÖR simetrik** makine (COD yalnız
1.6 μm → orbit-null işe yaramaz); 12 orbit-görünür knob; **gerçek (C++) gradyan** ile
Gauss-Newton descent.

| | kalan sahte-EDM (×hedef) | kazanç |
|--|------------------------:|-------:|
| f0 (düzeltmesiz) | 8 | — |
| **orbit-null** (‖COD‖→0) | **8** | **1.0× (İŞE YARAMAZ — orbit-kör)** |
| **SPİN-descent** | **~0** (8→1→0.003→0.01) | **726×** |

**Bulgu: SPİN-descent, orbit-null'un çuvalladığı yerde null'lar.** Orbit-düzeltme
orbit-kör simetriğe dokunamaz (kazanç 1.0×); spin-gradient descent aynı sahte-EDM'i
**2 adımda hedefin altına** indirir (726×). Fark net: (i) spin simetriği görür,
(ii) gradyan gerçek → §6.10-6.11'in harita-hatası/model-istismarı darboğazı **YOK**.

![spin-descent](/tmp/akilli_duzeltme/fig_kolb_spin_descent.png)

**Bu, tüm arc'ın doğrulaması:** orbit-tarafı (Kol B) harita-doğruluğuyla tıkalı;
**spin-tarafı (Kol A / Plan 5) çalışır** — ve modülasyon-gradyanıyla *sistematik*
(kör spin-trim'in verimli hâli). Menzil de yeterli (±40 μT = ±200 μm eşdeğer).

**Dürüst caveat'lar:** (i) tek makine, f0=8× (mütevazı); (ii) bu **spin gerektirir**.

> **⚠️ KRİTİK DÜZELTME (kullanıcı, 2026-06): Plan 5 pratikte ZAMAN-YASAK.** Gürültüsüz
> demo, sahte-EDM'i *ücretsiz/anında* ölçebildiğimizi varsaydı. Gerçekte polarimetre
> **istatistikle sınırlı** (`cosy_polarimeter §4`): bir dolum (1000 s) → σ≈900× hedef;
> **bir yıllık** kampanya → σ≈7× hedef; $\sigma\propto1/\sqrt T$ → **1 nrad/s'e inmek
> ~50 yıl.** Descent ~40–100 ölçüm ister ve EDM-altına sürmek için son ölçümler
> ~nrad/s hassasiyette olmalı → **her biri ~yıllar → toplam onyıllar-yüzyıllar.
> İteratif spin-null'lama fiilen İMKÂNSIZ.** Aynı sınır §14.6 "spin ölç-trim
> ~6000×" için de geçerli (o da simülasyonda ücretsiz-ölçüm varsayar).
>
> **Asıl bağ:** orbit-tabanlı kontrol **hızlı ama tıkalı** (simetrik-kör/harita-kaba);
> spin-tabanlı kontrol **doğru ama istatistik-yasak** (yıllar/ölçüm). Her iki kaçış
> da kapalı → simetrik geometrik-faz sistematiği **gerçekten zor.** Gerçek deney
> (Omarov) iteratif spin-null YAPMAZ; CW/CCW iptal + a-priori orbit-kontrol (CR-ayrım,
> hızlı) + kalanı 1-yıllık kampanyada bir kez ölçüp **sınırlar** — ama CR-ayrım
> simetriğe kör (§9.3) → simetrik artık ancak **tasarım toleransıyla** (μm-misalignment
> → ~62× hedef) sınırlanır, aktif null'lanamaz.

---

## 7. Sonuç ve fork (orbit-tarafı tıkalı, SPİN çalışır — Plan 5 doğruladı)

- **Kol B KAPATILMADI; DÖRT bağımsız bulgu destekliyor.** İlk "ölü/aynı-duvar"
  sonucu, ileri-harita ile inversiyonu karıştıran bir koşullanma hatasıydı (geri
  çekildi).
  1. **İyi koşullu:** ∂f/∂COD≈0.15 (1/σ_min DEĞİL); 7 nm ortalamayla ulaşılır;
     ofset yönlü-alanla değişmez (§4).
  2. **Öğrenilebilir:** orbit-kör simetrik kanal temiz COD'dan öğreniliyor
     (CV R² 240 örnekte +0.77; trend −0.5→0.77) (§6).
  3. **Model-fidelity:** %1 β-beat ŞEFFAF — nominal-eğitimli harita β-beat makineyi
     held-out nominal kadar iyi öngörür (R² 0.62 vs 0.61, ek bozulma yok) (§6.5).
  4. **Birleşik no-go Kol B'yi KAPSAMAZ** — o yalnız misalignment *geri-çatımı*
     (inversiyon+lock-in) içindir; ileri-harita f-öngörüsü o sınıfta değil.
- **KULLANIM kısıtı (§6.6, kritik):** Harita **çıkarma** için kullanılamaz (sahte-EDM
  ~1000× sinyal; çıkarma ~%0.1 mutlak doğruluk ister, harita %22). **CW/CCW+flip bu
  belirsizliği geri ALMAZ** — β-beat kayması CW/CCW-tek (corr −0.89), EDM-kanalında
  hayatta kalır. Harita ancak **null'lama** (predicted f→0, gerçek yörünge beslenir,
  β-beat şeffaf) için kullanılabilir.
- **AMA pratikte HENÜZ ÇALIŞMIYOR (Plan 2c, §6.10):** kapalı-döngü null'lama, basit
  orbit-düzeltmeyi GEÇEMİYOR (hatta sıkça kötüleştiriyor). Sebep: haritanın mutlak
  hata tabanı (~300× hedef, slope 0.36 → çoğunlukla toplamsal) orbit-null kalanından
  (0–6× hedef) büyük → corrector'ları yanlış yönlendiriyor. **4 pozitif bulgu
  *gerekli ama yeterli değil*; eksik olan MUTLAK harita doğruluğu.**
- **Açık (kritik, tek kalan):** mutlak doğruluğu hedefe indirmek — **Plan 4
  (`orbit_ileri §5` analitik Berry fonksiyoneli, exact)** ya da mertebelerce daha
  fazla veri. Empirik 240-örnek harita YETERSİZ. Fizik no-go'su değil ama küçük de
  bir engel değil — "çözüldü" kesinlikle değil.
- **Kol A (spin)** çalışır, orbit-tarafı değil — bizim spin ölç-trim'imiz (§14.6);
  Omarov SBA'sı DEĞİL (SBA E-alan hizalar, `omarov.md §5`).

> **Düzeltilmiş strateji:** Orbit-tarafı sahte-EDM null'lama umudu **kapanmadı,
> aksine güçlendi.** İleri-harita iyi-koşullu VE öğrenilebilir olduğundan, COD→f
> haritası + ortalama ile simetrik kanal **prensipte kapatılabilir.** Bu artık bir
> gözlenebilirlik no-go'su değil, **mühendislik/doğrulama problemi** (gürültü bütçesi,
> model-fidelity). `orbit_ileri_olcum.md §7`'nin "açık problem"i bu oturumda
> **pozitif yönde ilerledi.**

---

## 8. Reprodüksiyon

Keşif kodu `/tmp/akilli_duzeltme/`:

```bash
python3 /tmp/akilli_duzeltme/surrogate.py        # SVD: cond=193, sym=küçük-σ
python3 /tmp/akilli_duzeltme/gen_patterns.py     # karar desenleri + COD öngörüsü
python3 /tmp/akilli_duzeltme/measure_f.py 4      # dejenerasyon f'leri (C++, ~12dk)
python3 /tmp/akilli_duzeltme/analyze.py          # kaldıraç 0.146, koşullanma
python3 /tmp/akilli_duzeltme/fig_degeneracy.py   # figür
python3 kmod_drivers/fast_est.py calib -w 4 --seeds 3   # p=2.002
python3 /tmp/akilli_duzeltme/gen_ensemble.py 4 48      # 240 örnek ensemble (~2 saat)
python3 /tmp/akilli_duzeltme/fit_forward.py           # ileri-harita CV R²
python3 /tmp/akilli_duzeltme/fig_learnability.py      # öğrenilebilirlik trendi (R²→0.77)
python3 /tmp/akilli_duzeltme/perquad_orbit.py         # β-beat CO çözücü (build_R ile %0.2)
python3 /tmp/akilli_duzeltme/measure_bbeat.py 4 40    # β-beat f (C++ quad_dG, ~30dk)
python3 /tmp/akilli_duzeltme/bbeat_analyze.py         # model-fidelity transfer R²
python3 /tmp/akilli_duzeltme/fig_bbeat.py             # β-beat fidelity figürü
python3 /tmp/akilli_duzeltme/measure_cwccw.py 4 12    # CW/CCW×nom/bb signed (~34dk)
python3 /tmp/akilli_duzeltme/cwccw_analyze.py         # ortak-mod testi (corr −0.89)
python3 /tmp/akilli_duzeltme/fig_cwccw.py             # CW/CCW ortak-mod figürü
python3 /tmp/akilli_duzeltme/amp_scan.py 4            # genlik-homojenlik (σ² → 1mm)
python3 /tmp/akilli_duzeltme/fig_homogeneity.py      # homojenlik figürü
python3 /tmp/akilli_duzeltme/cr_separation.py 4 5    # Omarov CR-ayrım körlüğü
python3 /tmp/akilli_duzeltme/nn_vs_Rinv.py           # NN=R⁻¹ (ters problem)
python3 /tmp/akilli_duzeltme/nulling_floor.py        # Plan 2c proxy (hata tabanı)
python3 /tmp/akilli_duzeltme/closed_loop.py          # Plan 2c kapalı-döngü null'lama
python3 /tmp/akilli_duzeltme/improved_null.py        # §6.11 güvenli/ensemble null
python3 /tmp/akilli_duzeltme/spin_descent.py         # Plan 5 spin-gradient descent (POZİTİF)
```

Çekirdek estimator `berry_data/false_edm_4d.py` (4D-CO + model-fit, p=2.00).
Analitik COD `analytic_kmod.build_R_analytic`. `integrator.cpp` değiştirilmedi.
β-beat için per-quad `quad_dG` mevcut (C++ destekli).
