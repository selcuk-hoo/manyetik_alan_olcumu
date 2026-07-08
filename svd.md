# svd.md — N=2 stray manyetik alanın yörünge harmoniğinden ölçümü: harmonik-fit vs SVD

> **Bir cümlede:** Bir N=2 azimutal radyal stray manyetik alan (B_r = A_r·cos 2θ),
> depolama halkasında dikey bir k=2 kapalı-yörünge deseni yaratır; bu deseni 48 BPM'den
> okuyup alanı tahmin etmeye çalışırız. İki yol var — (a) yörüngeyi tek bir k=2 Fourier
> katsayısına indiren **harmonik fit**, (b) tam tepki-matrisini ters-çevirip alanı
> kaçıklıktan ayıran **SVD geri-çatım**. Bu belge ikisini gerçek C++ demet dinamiğiyle
> karşılaştırır, sınırlayıcıları (dejenerasyon, BPM ofseti, koşullanma, orbit-görünmez
> null mod) ayrıştırır ve ölçülebilecek en düşük alanı verir.

> **Durum notu:** Bu çalışma bir günlük yoğun keşif+hesabın kaydıdır; ileride bir
> makaleye temel olması için yeterli detay ve yönlendirmeyle yazıldı. Sayılar C++
> (`integrator.cpp`) demet dinamiğine dayanır; keşif kodu `/tmp/akilli_duzeltme/field_n2_*.py`.

---

## 1. Amaç ve fiziksel kurulum

**Neden.** Proton EDM deneyinde net dikey spin presesyonunu (EDM sinyalini) taklit
eden sistematiklerden biri, halka çevresinde azimutal olarak değişen bir **radyal
stray manyetik alan**dır. En basit modeli tek bir azimut harmoniğidir:

$$B_r(\theta) = A_r \cos(N\theta), \qquad N = 2.$$

**Alandan yörüngeye.** Radyal alan, boyuna hızla `v×B` çarpımı üzerinden **dikey**
bir tekme verir. Lineer örgüde harmonik korunur: N=2 alan, **dikey kapalı yörüngede
k=2** bir bozulma sürer. (Kodda: `FieldParams.B0rad_harm_amp = A_r` [T],
`B0rad_harm_N = N`; deflektörlerde uygulanır, `integrator.cpp` §"Harmonic radial
magnetic field".)

**Ölçüm hedefi.** 48 quad konumundaki BPM'lerden dikey yörüngeyi okuyup **A_r**'yi
(nT mertebesinde) tahmin etmek. Zorluk: aynı k=2 yörüngeyi **quad hizalama hataları**
da üretir → alan ile kaçıklık **dejeneredir**; ayrıca BPM ofsetleri ve gürültü vardır.

---

## 2. Orbit yanıtının kalibrasyonu (C++ demet dinamiği)

Bilinen bir alan koyup (A_r = 100 nT) gerçek C++ kapalı yörüngeyi bulup dikey CO'yu
48 azimutta ölçtük. Sonuç (birim genlik/nT):

| bileşen | genlik | not |
|---|---|---|
| **k=2** (asıl imza) | **1.36 μm/nT** | rezonant-yakını (Q≈2.24, \|Q²−4\|≈1) → büyük |
| **n=0 (DC)** | **1.01 μm/nT** | alana özgü net dikey kayma (§7) |
| max\|CO\| | 2.73 μm/nT | ; RMS(AC) 0.97 μm/nT |

**Kontrol:** RMS(AC) = k2/√2 = 1.36/√2 = 0.96 ✓ → AC yörünge **neredeyse saf k=2**
(betatron çınlaması n=2/n=3 ≈ 24× n=2 lehine; Q=2.24 tam sayı 2'ye yakın). Yani
"orbit temiz k=2 değil, Q'da çınlar" itirazı gerçektir ama **etkisi küçüktür** —
AC bileşen pratikte k=2'dir.

---

## 3. Harmonik fit yöntemi (tek modu okuma, ters-çevirme YOK)

**Fikir.** Alanın imzası dikey yörüngede saf bir k=2 desenidir. 48-BPM okumasını
`cos 2θ` **ve** `sin 2θ`'ya izdüşür — yani k=2'nin hem **genliğini hem fazını**
(kompleks katsayı). Dosya adının "harmonic fit" olmasının nedeni budur: tek harmoniğin
iki kadratürünü fitleriz.

```python
theta = 2*np.pi*np.arange(48)/48
c = 2*np.mean(y_bpm * np.cos(2*theta))     # k=2 kosinüs (in-phase)
s = 2*np.mean(y_bpm * np.sin(2*theta))     # k=2 sinüs   (quadrature)
A_r = np.hypot(c, s) / A0_k2               # alan genliği [nT], A0_k2 = 1.36 μm/nT
```

**Neden ters-çevirme değil / neden iyi-koşullu.** Bu, yörüngeyi **tek bir bilinen
ortonormal vektöre** izdüşümdür (cond=1). Tepki-matrisini ters çevirmez. Üstelik k=2,
tepki-matrisinin **büyük tekil değerli** (rezonant) modudur → gürültü/ofset bu modda
büyütülmez.

**Ama fit YANLIDIR.** İzdüşüm, k=2 katsayısına düşen **her şeyi** toplar:

$$\text{ölçülen } k{=}2 = \underbrace{A_0 A_r}_{\text{alan}} + \underbrace{M_0 D_{k2}}_{\text{kaçıklık}} + \underbrace{O_{k2}}_{\text{ofset}} + \underbrace{n_{k2}}_{\text{gürültü}}.$$

Alanın k=2'si ile kaçıklığın k=2'si **ayrılamaz** (dejenere). Tek sayı, dört katkıyı
çözemez → tahmin **kaçıklık+ofsetle yanlı**.

---

## 4. SVD geri-çatım yöntemi (tam-şekil, ters-çevirme VAR)

**Fikir.** 48-BPM yörüngesini yalnız k=2'ye değil, **tam şekle** göre çöz. Tepki
matrisi `R_dy` (48×48, her quad'ın dy'sinden dikey yörünge) ve alan sütunu `A_field`
(48, alandan yörünge) ile birleşik sistem:

$$\mathbf{y} = R_{dy}\,\mathbf{dy} + A_{field}\,A_r + \mathbf{ofset} + \mathbf{gürültü},$$

`[R_dy | A_field]`'i ters-çevirip **A_r'yi ayrı** çek. R_dy, alanın şeklinden
(özellikle §7'deki DC + n≥3 farkları) yararlanarak alanı kaçıklıktan **ayırabilir** —
harmonik fit'in attığı bilgiyi kullanır. **R_dy TEMİZ C++ ile kuruldu** (48 quad'a tek
tek dy perturbasyonu → C++ kapalı yörünge → 48-BPM sütunu; analitik Twiss DEĞİL).

**Kritik: R_dy korkunç ill-conditioned.** Tekil değer spektrumu:
`28.1, 27.8, 9.96, ... , 0.043, 0.030, 2.4×10⁻⁷`. **Son mod ~sıfır** — bu, çok az
kapalı yörünge üreten **orbit-görünmez** mod (false-EDM çalışmasındaki simetrik/yüksek-k
alt-uzay). cond(R_dy) ≈ 1.2×10⁸, tamamı bu **tek null moddan**; onu dışlarsak etkin
cond ≈ 1160.

**Sonuç: regularizasyon ŞART.**
- **Naif ters-çevirme** (`lstsq`, tüm modlar): null mod ofseti ~4×10⁶ büyütür →
  A_r = −773 ± **41000 nT** (tamamen kullanılamaz). ← SVD'yi bu şekilde kullanmak yanıltıcı.
- **TSVD/Tikhonov** (null mod atılmış): aşağıdaki tablo.

---

## 5. Doğrudan karşılaştırma (temiz C++, gerçek A_r = 1 nT, 10 μm rastgele kaçıklık)

| BPM ofset | Harmonik fit A_r | SVD-TSVD A_r (regularize) |
|---|---|---|
| 100 μm | 56.6 ± 29.5 nT | **0.4 ± 8.5 nT** |
| 30 μm  | 52.5 ± 27.8 nT | **0.5 ± 5.4 nT** |
| 0 (ideal) | 52.3 ± 26.6 nT | **0.4 ± 4.9 nT** |

**Okuma:**
- **Harmonik fit ~53 nT YANLI** (ortalama 53, gerçek 1) — alanı 10 μm kaçıklığın k=2
  bileşeninden ayıramıyor; ±30 nT saçılım kaçıklık gerçeklemesinden. Stabil (ofsetle
  değişmiyor) ama **işe yaramaz**: 1 nT alanı 53 nT arka planda göremezsin.
- **SVD-TSVD ~5–8 nT** — alanı kaçıklıktan **ayırıyor** (ortalama ≈ gerçek); ofset-sınırlı
  (8.5→4.9). Regularizasyon bir ölçek-yanlılığı getirir (kalibre edilebilir).
- **SVD ~6–10× daha iyi**, AMA yalnız (a) tam-şekil kullanırsan (b) null modu regularize
  edersen. Aksi halde harmonikten çok daha kötü.

> ⚠️ **KRİTİK ÇEKİNCE (§7):** Bu SVD-TSVD karşılaştırması, alan sütunu `A_field`'i
> **artefakt bir DC bileşeniyle** kullandı (§7'de gösterildi: DC N'den bağımsız,
> uniform-tepkiye eşit → implementasyon artefaktı, "N=2 imzası" değil). Gerçek makinede
> DC olmayacağından alan kaçıklıkla daha dejeneredir → **~5-8 nT iyimserdir**. SVD'nin
> harmonik-fit'i geçen kısmı yalnız gerçek n≥3 şekil farkına dayanmalı; bu tek başına
> ne kadar ayırır — **artefakt-DC çıkarılıp yeniden ölçülmeli** (açık iş).

**Önemli metodolojik ders:** Aynı problemi **analitik Twiss R** ile çözdüğümüzde
cond=193 (gerçek 1.2×10⁸'in çok altında) çıkıp SVD'yi yanıltıcı biçimde ~71 nT gösterdi.
**Tepki matrisi mutlaka gerçek demet dinamiğiyle kurulmalı**; analitik kısayol
koşullanmayı ve null modu gizliyor.

---

## 6. BPM ofsetinin rolü ve ileri/ters dualitesi

Ölçüm yapısı: `y = R·(kick'ler) + ofset`. **Kaçıklık kick'leri R'den geçer, ofset
geçmez.** R harmonik uzayda ~köşegen: R_k = G_k = C/\|Q²−k²\| (k=2 için rezonant, büyük).

| yön | işlem | ofsete ne olur |
|---|---|---|
| **Harmonik okuma** (ileri, k=2) | R'nin çıktısını *oku* (×G_k) | ofset ×1, sinyal ×G₂ → ofset **görece bastırılır** |
| **Naif SVD** (ters, tüm modlar) | R'yi *ters çevir* (÷G_k) | küçük-G_k/null modda 1/G_k dev → ofset **patlar** |
| **SVD-TSVD** (ters, regularize) | null mod atılmış ters | ofset ılımlı büyür (100μm→~7 nT) |

Yani "BPM ofseti geri-çatımın katilidir" sezgisi **naif** ters-çevirme için doğrudur;
proper TSVD onu ehlileştirir (100μm ofset → ~7 nT katkı, felaket değil). Harmonik
okumada ofset zaten büyütülmez ama fit dejenerasyonla yanlı olduğundan bu avantaj
kurtarmaz.

---

## 7. DC (n=0) imzası: doğrulandı → **İMPLEMENTASYON ARTEFAKTI** (ayrım kanalı DEĞİL)

Taslağın erken sürümünde, alanın yörüngesinin kaçıklığınkinden farklı bir **DC (n=0)
dikey kayma** (~1.01 μm/nT) ürettiğini gözledik ve bunu SVD'nin alanı kaçıklıktan
ayırmasının bir nedeni saydık. **Bu yanlıştı.** N-taraması + fizik argümanı ile
doğrudan test ettik (`field_n2_dcmech.py`, `integrator.cpp` DEĞİŞTİRİLMEDİ):

| kaynak | DC (μm/nT) | k=N (μm/nT) | ⟨cos Nθ⟩_eff = DC_harm/DC_uniform |
|---|---|---|---|
| **uniform B0rad** | −0.999 | — | (referans) |
| harmonik N=1 | +1.003 | 0.413 | **−1.00** |
| harmonik N=2 | +1.009 | 1.359 | **−1.01** |
| harmonik N=3 | +0.967 | 0.487 | −0.97 |
| harmonik N=4 | +0.952 | 0.172 | −0.95 |
| harmonik N=6 | +0.899 | 0.061 | −0.90 |
| harmonik N=8 | +0.826 | 0.032 | −0.83 |

**İki kanıt, kesin sonuç:**

1. **DC, N'den ~bağımsız ve uniform-alan tepkisine ~eşit** (⟨cos Nθ⟩_eff ≈ −1.0, N=1..8).
   Yani harmonik alan, DC bakımından neredeyse **uniform radyal alan gibi** davranıyor —
   "N=2'ye özgü bir imza" DEĞİL. (Buna karşılık k=N sütunu **gerçek**: G_k=C/\|Q²−N²\| ile
   N=2'de rezonant tepe, doğru fizik.)

2. **Fizik:** gerçek bir cos(Nθ) radyal alanın dikey kicki ∝ B_r = A_r cos(Nθ); azimutal
   ortalaması ∫cos(Nθ)dθ = 0 (her N≥1). Yani **dikey DC teorik olarak SIFIR olmalı.**
   Simülasyonun ~1 μm/nT DC'si fiziksel olarak imkânsız → **artefakt.**

**Kök neden** (`integrator.cpp` §"Harmonic radial magnetic field", satır ~257-268):
deflektör dalı alanı sürekli φ=atan2(Y,X) ile radyal yöne **projekte eder**
(`B[0]+=B_r0·cosφ; B[1]+=B_r0·sinφ`) → temiz k=N, DC yok. Düz-bölüm dalı ise
projeksiyonsuz `B[0]+=B_r0` ekler; düz bölümün yerel-radyal ekseni lab-çerçevesinde
dönmediğinden, halka boyunca **iptal olması gereken** dikey kick iptal olmaz → sahte
net DC (≈ uniform tepkisi, N'den bağımsız).

**Sonuç (kritik, yayın için):** DC bir **ayrım kanalı değil**; svd.md'nin özgün
"iki-kanallı imza" iddiası **düşer**. §4–5'teki SVD-TSVD, `A_field` sütununu bu artefakt
DC ile birlikte kullandı → **~5-8 nT rakamı iyimser** (gerçek makinede, DC olmadan,
alan kaçıklıkla daha dejenere olur → ayrım zayıflar). SVD'nin harmonik-fit'i geçen
kısmı yalnız gerçek **n≥3 şekil farkına** dayanmalı; onun tek başına ne kadar ayırdığı
**artefakt-DC çıkarılıp yeniden ölçülmeli** (açık iş). Bu bulgu, dejenerasyonun
svd.md'nin sonucundan **daha temel** olduğunu gösteriyor.

---

## 8. Ölçülebilecek en düşük alan (eşik) ve ölçeklemeler

**Harmonik fit (yanlı):** eşik ≈ kaçıklık arka planı. Doğrudan okuma sahte-alan tabanı:
- quad kaçıklığı: **~0.6 nT/μm** (per-quad RMS; rezonant-yakını G₂ ile büyütülmüş).
- BPM ofseti: **~0.15 nT/μm** (R'den geçmez, büyütülmez).
- Örnek: 10 μm kaçıklık + 100 μm ofset → ~53 nT; 3 μm + 30 μm → ~5 nT.

**SVD-TSVD (yansız):** ofset-sınırlı gürültü tabanı:
- 100 μm ofset → ~8.5 nT ; 30 μm → ~5.4 nT ; ofset giderilmiş → ~4.9 nT.
- + regularizasyon ölçek-yanlılığı (kalibre edilebilir) + null-mod farkındalığı şart.
- Model kalitesi (β-beat, tilt): **doğrudan C++ ile ölçüldü → tabanı YÜKSELTMİYOR** (§8.5);
  asıl belirleyici kaçıklık RMS'inin kendisi (ofset + truncation kaçağı).

**Özet eşik:** Bu koşullar altında (100 μm ofset, 1 μm gürültü, 10 μm kaçıklık) N=2 alan
pratik tabanı **~5–8 nT (SVD-TSVD)**; harmonik fit ~53 nT'de yanlı. Sub-nT için hem
**daha iyi hizalama** (μm) hem **ofset giderme/kalibrasyon** hem **doğru R (demet
dinamiği)** gerekir.

---

## 8.5. β-beat, quad tilt ve hizalama-ölçekleme sistematikleri (doğrudan C++)

§8'de "model hatası tabanı yükseltir" dedik; bunu **doğrudan demet dinamiğiyle** ölçtük
(analitik-Twiss/Test-8 analojisi YOK). Yöntem: geri-çatım **nominal R** ile yapılır ama
ölçümler **β-beat/tilt'li GERÇEK makineden** C++ ile üretilir → model hatası (R_gerçek ≠
R_nominal) doğrudan A_r tabanına yansır. İki kaçıklık seviyesinde (rcond=1e-2, 100μm ofset,
1μm gürültü; NSAMP=8 makine gerçeklemesi):

| senaryo | **10 μm** SVD-TSVD | **100 μm** SVD-TSVD | **100 μm** harmonik |
|---|---|---|---|
| analitik baz (model-hatasız) | ±8.6 nT | **±49.8 nT** | — |
| C++ baz (β-beat/tilt YOK) | ±9.4 | ±58.5 | 707 ± 358 |
| β-beat %1 (quad_dG) | ±8.4 | ±57.5 | 419 ± 221 |
| β-beat %2 | ±8.1 | ±44.7 | 542 ± 249 |
| tilt 1 mrad (+ dx) | ±9.1 (dx=10μm) | **+45 ± 71** (dx=99μm) | 385 ± 314 |

**Üç bulgu:**

1. **Asıl kaldıraç kaçıklık RMS'inin KENDİSİ.** SVD tabanı ~lineer ölçekleniyor
   (10μm→~9 nT, 100μm→~55 nT). Model-hatasız (analitik) baz bile 100μm'de ±50 nT: bu
   **TSVD-truncation kaçağı** — büyük dy'nin, koşullanma için atılan küçük-tekil-değerli
   modlara düşen kısmı geri-çatılamaz, kalan orbit A_r'ye sızar. Sonuç: ~1 nT hedefe
   **BBA ile ~10 μm hizalama şart**; hizalama β-beat/tilt'ten çok daha belirleyici.

2. **β-beat (≤%2): ölçülebilir etki YOK.** Her iki seviyede de taban gürültü içinde
   sabit (10μm: 9→8-9; 100μm: 58→45-57). Neden: rcond=1e-2 regularizasyonu zaten
   ofset+kaçıklık-sınırlı tabana oturuyor, %birkaç R-hatası bu tabanın altında kalıyor;
   kalibrasyon kayması (§8'deki ~−2/−4%) de A_r≈1 nT üzerinde ~0.02 nT → 9-50 nT tabanda
   görünmez. (Not: bu, §8.5-öncesi taslaktaki "Test 8 → %1 β-beat 6μm" **analojisini
   geçersiz kılar**; doğrudan ölçüm o dolaylı kıyası gereksiz kılıyor.)

3. **quad tilt: iyi hizalamada önemsiz, kötü hizalamada gerçek.** 10μm dx'te SVD'ye
   etkisiz (skew arka planı ~0.3 nT, tabanın altında). Ama **99μm dx'te +45 nT sistematik
   yanlılık**: modellenmemiş skew-kuplaj (büyük yatay kapalı yörünge → dikeye kaçak),
   ill-conditioned ters-çevirmede büyütülür. Yani tilt yalnız **yatay kaçıklık büyükse**
   sorun → yine BBA'ya (küçük dx) işaret eder.

**Sonuç:** "Hiçbiri sonucu ciddi bozmuyor" ifadesi **iyi hizalanmış makinede (≤10 μm)
doğru** — β-beat ve tilt ihmal edilebilir; tabanı hizalama + BPM ofseti belirler.
Kötü hizalamada (100 μm) SVD tabanı zaten ~55 nT'ye çıkar ve tilt +45 nT ekler; ama
o rejimde harmonik (~500 nT) de SVD de kullanılamaz. Reprodüksiyon:
`field_n2_betabeat_tilt.py` (kalibrasyon+arka plan), `field_n2_svdfloor.py`
(SVD tabanı; `DMIS_UM=100` ile 100μm).

---

## 9. Eski denemelerle tutarlılık

- **v4.0 / v2.1 (git tag'leri):** tek/birkaç-quad k-mod ile k=2 geri-çatımı rank çöküşüyle
  başarısızdı (%4307). Bu, buradaki "naif ters-çevirme null modda patlar" ile aynı fizik
  (ill-conditioned inversion). Fark: burada **global tam-R + TSVD** kullanıp regularize
  edince ~5-8 nT'ye iniyoruz; tek-quad k-mod rank-1 olduğundan oraya hiç ulaşamıyordu.
- **false-EDM / squid_bpm:** orbit-görünmez simetrik mod = R_dy'nin null modu (S=2.4×10⁻⁷);
  BPM ofsetinin ters-problemdeki katil rolü aynı modda yaşıyor.

---

## 10. Sonuç ve açık sorular

**Sonuç.**
1. N=2 alan → dikey k=2 kapalı yörünge; alan ile kaçıklık k=2'de **dejenere** (aynı imza).
2. **Harmonik fit** basit ve ofsete-dayanıklı ama **yanlı** (~53 nT/10μm) — ayıramaz.
3. **SVD-TSVD** alanı kaçıklıktan görünürde **ayırır** (~5-8 nT) — ama (a) tepki matrisi
   **gerçek demet dinamiğiyle** kurulmalı, (b) **null mod regularize** edilmeli, (c) BPM
   ofseti ılımlı büyür. **ÇEKİNCE:** bu ayrım kısmen §7 artefakt-DC'sine dayandı →
   ~5-8 nT **iyimser**; gerçek n≥3-şekil ayrımı ayrıca ölçülmeli.
4. **BPM ofseti** naif ters-çevirmenin katili; TSVD ehlileştirir; harmonikte büyütülmez.
5. **DC imzası ARTEFAKT** (§7): N'den bağımsız, uniform-tepkiye eşit, fiziksel DC=0
   olmalı → düz-bölüm alan-uygulamasının çerçeve hatası. **Ayrım kanalı değil.**

**Açık sorular (makale öncesi).**
- ~~DC mekanizması gerçek mi implementasyon mu~~ **YANITLANDI (§7): implementasyon
  artefaktı** (N-bağımsız + fiziksel DC=0). → SVD ayrımı artefakt-DC çıkarılıp
  **yeniden ölçülmeli**; gerçek n≥3-şekil ne kadar ayırıyor?
- Rezonant-yakınlığın (Q≈2.24) rolü: N=3,4 için eşik nasıl değişir (G_k düşer)?
- ~~Model hatası (β-beat) SVD-TSVD tabanını ne kadar yükseltir~~ **YANITLANDI (§8.5):**
  β-beat ≤%2 tabanı yükseltmiyor; asıl belirleyici kaçıklık RMS'i (10μm→9, 100μm→55 nT)
  + BPM ofseti; tilt yalnız büyük dx'te (+45 nT @99μm) devreye giriyor.
- Gerçekçi çok-mod, rastgele-faz kaçıklıkta SVD-TSVD tabanı (burada tek-seviye Gauss).

---

## 11. Reprodüksiyon

Keşif kodu `/tmp/akilli_duzeltme/`:

```bash
python3 field_n2_calib.py       # alan→CO kalibrasyonu (A0 k2=1.36, C++)
python3 field_n2_shape.py       # alan vs N=2-kaçıklık azimut spektrumu + korelasyon
python3 field_n2_dc.py          # DC doğrulama: A_r taraması (lineer, işaret-çeviren)
python3 field_n2_clean.py       # FAZ A: temiz C++ R_dy (48×48) + A_field (~28 dk, 49 koşu)
python3 field_n2_clean.py B     # FAZ B: cond, DC/k2, harmonik vs naif-SVD
python3 field_n2_dcmech.py      # §7: DC ARTEFAKT testi (uniform vs harmonik N-tarama)
python3 field_n2_betabeat_tilt.py            # §8.5: β-beat/tilt kalibrasyon+arka plan
python3 field_n2_svdfloor.py                 # §8.5: SVD tabanı (β-beat/tilt, 10μm)
DMIS_UM=100 python3 field_n2_svdfloor.py     # §8.5: 100μm kaçıklık varyantı
# TSVD adil karşılaştırma + SV spektrumu: field_n2_clean.npz üzerinden inline (bkz. §5,§4)
```

Çekirdek: `integrator.cpp` (GL4 semplektik izleyici, `B0rad_harm_amp/N` radyal harmonik
alan), `false_edm_4d.find_co_4d` (4D kapalı yörünge). `integrator.cpp` DEĞİŞTİRİLMEDİ.
