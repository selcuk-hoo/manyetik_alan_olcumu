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
> tam doğru; inversiyon no-go'su Kol B'yi **bağlamaz.** `orbit_ileri_olcum.md §7`
> açık problemi pozitif ilerledi. **Tek kalan:** mutlak doğruluk (veri/analitik +
> iterasyon) ve gürültü bütçesi — mühendislik, fizik no-go'su değil.
>
> **Kol A** değişmedi: çalışır (bilgi spinde), ama orbit-tarafı değil (Omarov/spin-trim).

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

**Bulgu: β-beat ŞEFFAF.** %1 β-beat f'i ~%18 oynatır (COD'u 3.2μm), ama nominal-
eğitimli harita β-beat makineyi **held-out nominal kadar iyi** öngörür (0.62 vs
0.61 — **ek bozulma YOK**). Harita, β-beat kaymasını COD üzerinden kısmen izler
(corr(Δpred,Δf)=+0.35). Yani **f, yörüngenin ~sabit fonksiyonelidir; β-beat haritayı
kırmaz** (inversiyonun aksine). Kalan %52 nispi hata, β-beat değil **veri-sınırlı
generalleme** hatasıdır (N ile düşer; §6 trendi).

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

## 7. Sonuç ve fork (DÜZELTİLDİ → POZİTİF eğilim)

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
- **Açık (tek kalan):** null'lamanın mutlak hedefe ulaşması — harita doğruluğu
  (veri/öznitelik/`orbit_ileri §5` analitik form) + iterasyon + gürültü bütçesi.
  **Mühendislik/veri** problemi, fizik no-go'su değil — ama "çözüldü" de değil.
- **Kol A (spin)** çalışır, orbit-tarafı değil (Omarov/spin-trim).

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
```

Çekirdek estimator `berry_data/false_edm_4d.py` (4D-CO + model-fit, p=2.00).
Analitik COD `analytic_kmod.build_R_analytic`. `integrator.cpp` değiştirilmedi.
β-beat için per-quad `quad_dG` mevcut (C++ destekli).
