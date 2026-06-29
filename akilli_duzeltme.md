# akilli_duzeltme.md — "Akıllı düzeltme": sahte-EDM'i null'lamak (misalignment'ı DEĞİL)

> **Durum (2026-06-29).** Bu belge, `SONRAKI_OTURUM_PROMPT.md`'de tanımlanan
> "akıllı düzeltme" fikrinin **karar-verici testinin** dürüst kaydıdır. Fikir:
> quad misalignment'ı geri-çatmak yerine (bu *inversiyon* no-go'su), sahte-EDM'i
> **doğrudan null'lamak.** İki kol:
> - **Kol A (spin-gözlemli):** dS_y/dt'yi spin ile ölç → corrector/quad knob'larıyla
>   null'la (spin-trim'in genellemesi; Omarov tarafında karşılığı var).
> - **Kol B (NN ileri-harita):** kapalı yörüngeden (48-BPM COD) sahte-EDM'in
>   **ileri-haritasını** öğren → orbit-görünür knob'larla EDM-hedefli düzelt.
>   Hipotez: ileri-harita (inversiyon değil) + EDM-hedefli olduğu için no-go'yu
>   atlayabilir.
>
> **KISA CEVAP:** **Kol B ölü** — sahte-EDM kapalı yörüngeyle *kullanışlı
> hassasiyette belirlenmez.* Karar-verici ölçüm: COD'u 1.7 μm içinde uyuşan iki
> makinenin sahte-EDM'i **~250× EDM-hedefi** kadar farklılaşır. Kol B'nin simetrik
> kanalı hedefe çekmesi için gereken BPM doğruluğu **≈ 7 nm** (100 μm ofset
> altında) — bu, `squid_bpm_test.md §9.5` inversiyon-no-go'sunun (<4 nm) **ileri-
> harita karşılığıdır: aynı duvar.** NN kapasitesi yardım etmez; bilgi BPM
> sistematik tabanının **altındadır.**
>
> **Kol A** ise çalışır (spin, geometrik fazın **doğrudan** gözlenebiliridir;
> `false_edm_harmonic_sinir.md §14.6` spin ölç-trim ~6000×), ama orbit-tarafı
> değildir → özgün katkı dar (Omarov SBA/spin-trim bölgesi).
>
> **Bu, projenin "akıllı düzeltme" açık problemini kapatır: BİRLEŞİK NO-GO** —
> orbit-inversiyon (no-go), orbit-lock-in (§9.5) ve orbit-ileri-harita (Kol B)
> üçü de **aynı** simetrik (orbit-kör) alt-uzay duvarına çarpar. Orbit-tarafı
> tükendi; kalan pozitif tek yol **spin** (Kol A = Omarov).

---

## 0. Okunması gereken SOLID temel

Bu belge şunların üzerine kurulur (negatif sonuçlarına güvenilir):
`omarov.md §5,§9,§10`; `orbit_ileri_olcum.md §2-3,§6,§9`;
`squid_bpm_test.md §8,§9,§9.5,§10`; `false_edm_harmonic_sinir.md §14.6`;
`YAPILACAKLAR.md §4`; `CLAUDE.md` (estimator reçetesi).

**Kesinleşmiş zemin:** sahte-EDM = geometrik (Berry) faz, σ²-ölçeklenir (p=2.00);
kaynağı dx·dy. Sahte-EDM'i süren şey **simetrik (QF/QD aynı-işaret, orbit-kör)**
misalignment alt-uzayıdır; bu alt-uzay yüksek-k kick → G_k=C/|Q²−k²| ile bastırılır.

---

## 1. Yöntem ve doğrulama

**Sahte-EDM ölçer (C++, doğrulandı):** `berry_data/false_edm_4d.py`
(`measure_false_edm`; `kmod_drivers/fast_est.fast_measure` azaltılmış ayar).
4D kapalı yörüngede tek ideal parçacık + model-fit seküler eğim. Bu oturumda
**yeniden doğrulandı:**

```
σ=10.0μm: |f|=9.115e-07 ± 6.4e-07 rad/s (n=3)
σ= 5.0μm: |f|=2.275e-07 ± 1.6e-07 rad/s (n=3)
p (10→5μm) = 2.002   (geometrik faz beklenen 2.00)   ← SAF KUADRATİK, lineer kaçak YOK
```

Tek koşum ~171 s (4 çekirdek, azaltılmış ayar). Düz polyfit DEĞİL; 4D-CO + model-fit.

**Hızlı COD (kapalı yörünge) yüzeyi (analitik, doğrulandı):** misalignment
(dx,dy) → 48-BPM COD haritası, FODO Twiss'ten analitik R matrisleriyle
(`analytic_kmod.build_R_analytic`). Bu R, `squid_bpm_test.md §5.5`'te gerçek C++
izleyiciyle **%1 içinde** doğrulanmıştır (corr 0.966). COD ucuz; f pahalı (C++).

---

## 2. Geometri: orbit-görünür ↔ orbit-kör SVD ayrışımı

R = U Σ Vᵀ. Tekil değer spektrumu (her iki düzlem de aynı, Q=2.303):

| | σ | Ŝ (simetri) | karakter |
|---|---|---|---|
| en büyük (mod 0–5) | 28.4 … 8.6 | **−0.92 … −0.95** | **orbit-GÖRÜNÜR (antisim)** |
| en küçük (mod 38–47) | 0.16 … **0.147** | **+0.74 … +0.96** | **orbit-KÖR (simetrik)** |

- **cond(R) = σ_max/σ_min = 193** (← `drift_monitor/permode2.py` κ=193, §3.8 ile birebir).
- **Büyük-σ yönleri antisimetrik; küçük-σ yönleri simetrik.** Yani sahte-EDM'i
  süren simetrik alt-uzay, tam da R'nin **göremediği** (küçük-σ) alt-uzaydır.
- σ<0.3 (orbit-kör) mod sayısı: **24/48.**

Bu, Kol-B testinin geometrisidir: aynı COD'a karşılık gelen farklı misalignment'lar
(→ farklı f) inşa etmek için küçük-σ (simetrik) yön kullanılır.

---

## 3. KARAR-VERİCİ TEST: COD→sahte-EDM tek-değerli mi?

**Tasarım.** Sabit orbit-görünür taban A (rastgele 10 μm). Sonra iki sınıf
pertürbasyon (her biri RMS 10 μm, yani tabanla aynı büyüklük):
- **Simetrik (orbit-kör):** en küçük-16 σ alt-uzayında → COD'u neredeyse değiştirmez.
- **Antisimetrik (orbit-görünür):** en büyük-16 σ alt-uzayında → COD'u çok değiştirir.

Her desen için **gerçek C++ ile f ölçüldü** (13 koşum). Analitik COD ayak izi
(taban'a göre ΔCOD_RMS) öngörüldü.

**Sonuç (taban A: |f|=5.58e-7 rad/s):**

| sınıf | ort ΔCOD_RMS | ort \|Δf\| | \|Δf\|/hedef | f-kaldıracı (\|Δf\|/ΔCOD) |
|-------|-------------:|----------:|-----------:|-------------------------:|
| **Antisim (görünür)** | **114.7 μm** | 9.27e-6 | ~9270× | 0.081 rad/s/m |
| **Simetrik (kör)** | **1.69 μm** | 2.47e-7 | **~247×** | 0.146 rad/s/m |

**İki kritik okuma:**

1. **f, COD'u izleyince izler (antisim):** büyük COD değişimi → büyük f değişimi.
   Bu, orbit-görünür kısımdır; orbit-düzeltmenin zaten 7.7× temizlediği yer
   (`omarov.md §10`).
2. **Simetrik kanal dejenere:** 10 μm orbit-kör misalignment sahte-EDM'i
   **247× hedef** kadar oynatır, ama COD ayak izi **yalnız 1.69 μm** (taban
   COD ~97 μm; **100 μm BPM ofseti altında görünmez**). f-kaldıracı (birim COD
   başına f) simetrikte antisimden bile **biraz daha büyük** (G_deg=1.80) — yani
   simetrik misalignment birim COD başına en az antisim kadar f üretir, ama o
   COD'u σ_min/σ_max=1/193 kat **bastırılmış** gösterir.

**Sabit-COD f saçılımı:** COD'ları ~1.7 μm içinde uyuşan makineler f'te
**258× hedef** kadar farklılaşır → **COD→f haritası tek-değerli DEĞİL** (kullanışlı
hassasiyette). Bu, herhangi bir COD→f haritasının (NN dahil) **modelden bağımsız**
tabanıdır.

![Kol-B dejenerasyon](/tmp/akilli_duzeltme/fig_kolb_dejenerasyon.png)

---

## 4. BİRLEŞİK NO-GO: Kol B = inversiyon-no-go ile aynı duvar

Kol-B'nin simetrik f-katkısını (10 μm kör misalignment → |Δf|≈2.5e-7) EDM hedefine
(1e-9) çekmesi için COD'un hangi doğrulukta ölçülmesi gerektiği:

```
gereken BPM doğruluğu ≈ ΔCOD_sym × (hedef / |Δf_sym|)
                      = 1.69 μm × (1e-9 / 2.47e-7) ≈ 6.9 nm
```

**6.9 nm**, `squid_bpm_test.md §9.5`'in inversiyon-no-go'su (**<4 nm**; σ_min(ΔR)≈1e-4,
simetrik kısmı kurtarmak için <4 nm efektif gürültü) ile **aynı mertebede**.
Yani:

> **Kol B (ileri-harita), Kol-inversiyon (geri-çatım) ile FARKLI bir yol DEĞİLDİR.**
> Her ikisi de simetrik (orbit-kör) f-bilgisini ~nm-seviyesi COD'dan çıkarmak
> zorundadır; bu bilgi 100 μm BPM ofseti / μm gürültü tabanının altındadır. NN'in
> öğrenebileceği bir şey yoktur — *fonksiyon değeri girdide yok.* Hipotez
> ("ileri-harita inversiyona girmez") **çürütüldü:** ileri-harita da aynı
> gözlenebilirlik tabanına çarpar.

**Üç kol, tek duvar (BİRLEŞİK NO-GO):**

| Yol | Mekanizma | Simetrik kanalda sonuç | Belge |
|-----|-----------|------------------------|-------|
| Orbit-inversiyon (R⁻¹/ΔR⁻¹) | COD→misalignment geri-çat | cond≈193–3.7e4, <4nm gerek | §8, §9.5 |
| Orbit-lock-in (v2.7) | √N gürültü yen | beyaz gürültüyü yener, **simetriği yenemez** | §9.5 |
| **Orbit-ileri-harita (Kol B)** | **COD→f öğren** | **~7nm gerek; bilgi tabanın altında** | **bu belge §3-4** |

---

## 5. Kol A neden çalışır, Kol B neden çalışmaz (kavramsal kapanış)

Sahte-EDM (Berry fazı) **quad'daki yerel demet-merkez ofsetini** (x_CO,i − dx_i)
görür. Simetrik misalignment için kapalı yörünge x_CO ≈ 0 (orbit-kör), ama yerel
ofset ≈ −dx_i (misalignment'ın kendisi) **kaybolmaz** → Berry fazı simetrik
misalignment'ı yerel olarak görür, BPM (sabit konumda x_CO ölçer) görmez.

- **Kol B** girdisi BPM-COD'dur → simetrik yerel ofset girdide **yok** →
  öğrenecek bir şey yok. (§3-4.)
- **Kol A** girdisi spindir → spin, Berry fazının **doğrudan** gözlenebiliridir →
  bilgi girdide **var** → geri-besleme (corrector knob'larıyla null'lama) çalışır.
  Bu, `false_edm_harmonic_sinir.md §14.6` spin ölç-trim'in (~6000×) ve Omarov'un
  vertical-pol/SBA bölgesinin ta kendisidir.

**Önemli ayrım:** Kol A misalignment'ı *restore etmez*; sahte-EDM'i null'lamak için
orbit-görünür knob'larla **kasıtlı bir orbit/geometrik-faz bump'ı** ekler (f=0
yeterli, optik restorasyonu gerekmez). Bilgi spinden geldiği için bu **gözlenebilirlik
tabanına çarpmaz** — ama spin ister, orbit tarafı değildir.

---

## 6. Empirik doğrulama: NN ileri-haritası (ensemble, 80 örnek)

§3-5 karar-verici testi **modelden bağımsızdır**; bu bölüm onu jenerik rastgele
desenler + gerçek regresyonla pekiştirir. 80 örnek (kontrollü simetri ağırlığı
w∈{0,0.25,0.5,0.75,1}, her biri 16 seed; w=0 antisim, w=1 simetrik). Girdi =
analitik 48-BPM COD; çıktı = C++ sahte-EDM. (`gen_ensemble.py` + `fit_forward.py`.)

**(a) Orbit-kör konfigler sahte-EDM'siz DEĞİL.** ⟨|f|⟩, simetri arttıkça düşer
ama **sıfırlanmaz:**

| w (Ŝ) | ⟨\|f\|⟩ rad/s | std |
|------|-------------:|----:|
| 0.00 (−1.0, antisim) | 3.34e-6 | 2.46e-6 |
| 0.50 ( 0.0) | 1.48e-6 | 1.91e-6 |
| 1.00 (+1.0, **tam orbit-kör**) | **1.82e-7** | 1.81e-7 |

Tam simetrik (orbit-kör) konfig bile **182× EDM hedefi** sahte-EDM taşır →
orbit-körlük ≠ sahte-EDM'siz. (Ham antisim/sim oranı ~18×; `orbit_ileri_olcum.md
§6`'nın ~37×'iyle aynı mertebe.)

**(b) İleri-harita yalnız orbit-GÖRÜNÜR payı öğrenir (gerekmeyen kısım).**
5-kat CV R² (temiz COD, fiziksel öznitelikler: Berry yönlü-alan + bilineer özetler):

| model | R² | yorum |
|-------|---:|-------|
| Ridge | **+0.32** | orbit-görünür (antisim) pay öngörülebilir — orbit-düzeltmenin zaten yaptığı |
| MLP(64,64) | −0.88 | 80 örnekte aşırı-uyum (Ridge adil) |

Tavan ~0.32; bu **antisim varyansıdır.** Önemli incelik: Berry yönlü-alan özniteliği
Σ(xᵢyᵢ₊₁−xᵢ₊₁yᵢ) kapalı halkada **BPM ofsetine değişmezdir** (sabit ofset toplamda
iptal) → +100μm ofset/1μm gürültüde R² **yine +0.32.** Yani *ofset, duvarın kendisi
değil*; akıllı öznitelik ofseti aşar — ama yine de simetrik kanala **ulaşamaz.**

**(c) KARAR-VERİCİ: dağılım-kayması — harita simetrik kanala taşınmıyor.**
Antisim (w≤0.25) üzerinde eğit → simetrik (w≥0.75) üzerinde test:
**R² = −134** (ortalamadan ~135× kötü). İleri-harita orbit-görünür rejimde
öğrendiğini orbit-kör rejime **taşıyamaz** — tam da sahte-EDM'i süren kanal.

**(d) Tutarlı fonksiyonel yok.** Berry proxy korelasyonu w'ye göre tutarsız
(−0.53/+0.02/−0.55/−0.08/−0.36) → tek bir kapalı-form fonksiyonel empirik
pinlenmiyor (`orbit_ileri_olcum.md §3` ile birebir).

**Empirik özet:** İleri-harita sahte-EDM'in **orbit-görünür (antisim) payını**
yakalar (orbit-düzeltmenin zaten yaptığı, gerekmeyen kısım); **orbit-kör simetrik
kanala — sahte-EDM'i süren ve düzeltme-sonrası kalan kısma — taşınamaz** (dağılım-
kayması R²≪0). §3-4'ün modelden-bağımsız ~7 nm tabanını doğrular.

---

## 7. Sonuç ve fork

**Fork (pozitif yöntem mi / birleşik no-go mu) → NEGATİF (orbit), POZİTİF yalnız spin.**

- **Kol B (orbit-ileri-harita) reddedildi.** Sahte-EDM kapalı yörüngeyle kullanışlı
  hassasiyette belirlenmez; gereken BPM doğruluğu ~7 nm (= inversiyon-no-go duvarı).
  Bu, projenin "ileri-ölçüm no-go'yu atlar mı?" (`orbit_ileri_olcum.md §5,§7`) açık
  problemine **net cevaptır: HAYIR** (kullanışlı hassasiyette) — analitik fonksiyonel
  pinlenememesinin nedeni *fonksiyon değerinin orbitin orbit-kör alt-uzayında
  gizli* olmasıdır.
- **Kol A (spin) çalışır** ama orbit-tarafı katkı değil (Omarov/spin-trim).
- **Özgün katkı yeri (değişmedi):** Omarov'un §9'da açık bıraktığı boşlukları
  nicelliyoruz — orbit/ayrım ölçümünün alt-uzay yapısı (antisim görünür 7.7×,
  simetrik kör) ve simetrik artığın **paylaşılan sınırı.** Bu oturum o sınıra
  **üçüncü bağımsız kanıtı** (ileri-harita) ekler: birleşik no-go.

> **Strateji:** "Akıllı düzeltme orbit-tarafında simetrik kısmı kapatır" umudu
> **kapandı.** Orbit tarafındaki özgün katkı, **kesin sınır teoremidir** (üç
> bağımsız kol → tek gözlenebilirlik tabanı), pozitif bir kurtarma değil.

---

## 8. Reprodüksiyon

Keşif kodu `/tmp/akilli_duzeltme/` (proje konvansiyonu: kalıcı repoda değil):

```bash
# COD yüzeyi + SVD (cond=193, sym=küçük-σ)
python3 /tmp/akilli_duzeltme/surrogate.py
# karar-verici desenler + analitik COD öngörüsü
python3 /tmp/akilli_duzeltme/gen_patterns.py
# desenlerin sahte-EDM'i (GERÇEK C++; ~12 dk, 4 çekirdek)
python3 /tmp/akilli_duzeltme/measure_f.py 4
# dejenerasyon analizi (G_deg, 7nm birleşik no-go)
python3 /tmp/akilli_duzeltme/analyze.py
python3 /tmp/akilli_duzeltme/fig_degeneracy.py     # figür
# estimator p=2.002 doğrulaması
python3 kmod_drivers/fast_est.py calib -w 4 --seeds 3
# empirik NN ileri-harita (ensemble; ~1 saat)
python3 /tmp/akilli_duzeltme/gen_ensemble.py 4 16
python3 /tmp/akilli_duzeltme/fit_forward.py
```

Çekirdek estimator `berry_data/false_edm_4d.py` (4D-CO + model-fit, p=2.00
doğrulanmış). Analitik COD `analytic_kmod.build_R_analytic` (C++ ile %1, §5.5).
`integrator.cpp` **değiştirilmedi.**
