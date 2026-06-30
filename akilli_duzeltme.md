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
> **Doğru durum: Kol B AÇIK + POZİTİF kanıt.** Gerçek engel *gözlenebilirlik*
> değil, fonksiyonelin karmaşıklığı. Ve o fonksiyonel **öğrenilebilir çıktı:**
> orbit-kör simetrik kanal (sahte-EDM'i süren) temiz COD'dan öğreniliyor —
> CV R² 80 örnekte ~0, ama 240 örnekte **+0.77** (trend −0.5→0.77, §6). Yani
> "harita vardır, basit değildir" tam doğru: harita karmaşık ama yeterli veriyle
> pinleniyor, ve ileri-harita iyi-koşullu olduğundan inversiyon no-go'sunu
> **gerçekten atlar.** `orbit_ileri_olcum.md §7` açık problemi pozitif ilerledi.
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

**Açık kalan:** (i) gürültülü/gerçekçi-ölçüm COD'unda (§4'teki 7nm ortalama) trend
korunur mu; (ii) model-fidelity (β-beat: sim-eğitimli harita gerçeğe taşınır mı,
`quad_dG` ile test edilebilir).

---

## 7. Sonuç ve fork (DÜZELTİLDİ → POZİTİF eğilim)

- **Kol B KAPATILMADI; üstelik POZİTİF kanıt var.** İlk "ölü/aynı-duvar" sonucu,
  ileri-harita ile inversiyonu karıştıran bir koşullanma hatasıydı (geri çekildi).
  Üç bağımsız bulgu Kol B'yi destekliyor:
  1. **İyi koşullu:** ∂f/∂COD≈0.15 (1/σ_min DEĞİL); 7 nm ortalamayla ulaşılır;
     ofset yönlü-alanla değişmez (§4).
  2. **Öğrenilebilir:** orbit-kör simetrik kanal temiz COD'dan öğreniliyor
     (CV R² 240 örnekte +0.77; trend −0.5→0.77) (§6).
  3. **Birleşik no-go Kol B'yi KAPSAMAZ** — o yalnız misalignment *geri-çatımı*
     (inversiyon+lock-in) içindir; ileri-harita f-öngörüsü o sınıfta değil.
- **Açık (test edilecek):** (i) gerçekçi-ölçüm (gürültü+ortalama, 7nm) altında
  trend; (ii) β-beat model-fidelity (`quad_dG`); (iii) `orbit_ileri §5` analitik
  Berry fonksiyoneli (öğrenilen haritayı doğrular/sadeleştirir).
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
```

Çekirdek estimator `berry_data/false_edm_4d.py` (4D-CO + model-fit, p=2.00).
Analitik COD `analytic_kmod.build_R_analytic`. `integrator.cpp` değiştirilmedi.
β-beat için per-quad `quad_dG` mevcut (C++ destekli).
