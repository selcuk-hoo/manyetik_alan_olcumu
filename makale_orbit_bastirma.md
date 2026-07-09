# Makale Taslağı — Proton EDM Deneyinde Sahte EDM Sinyalinin Yörünge Düzeltmesiyle Bastırılması ve Sınırları

> **Durum (2026-07): İSKELET TASLAK — tartışma/iterasyon için.** Bölüm yapısı ve
> her bölüme girecek sonuçlar (sayılar + kaynak belge) burada; düz metin (prose)
> henüz yazılmadı. Tüm sayılar repodaki doğrulanmış günlüklerden alınmıştır; her
> maddenin yanında kaynak belge/script referansı vardır. Dil şimdilik Türkçe;
> İngilizce çekirdek özet `PROJE_ANALIZI_VE_ONERILER.md §6.1`'de hazır.
>
> **Kapsam kararları (kullanıcı, 2026-07):**
> - Pozitif çerçeve: "kaç e·cm'e kadar bastırılabilir" belgelenir; no-go, eğrinin
>   doyma gerekçesi olarak içeride kalır.
> - **Spin-tabanlı null'lama ve NN ileri-harita (Kol B) KAPSAM DIŞI.** Spin için
>   gerekçe fiziksel: sahte/gerçek EDM aynı gözlenebilirde (dS_y/dt) dejenere —
>   sahte'yi null'lamak gerçeği de null'lar (`omarov.md §10`, `svd.md §9.1`).
>   İleri-harita için gerekçe editoryal: ayrı, olgunlaşmamış açık problem
>   (`akilli_duzeltme.md §6.10`); Conclusions'ta tek cümleyle "kapsam dışı" denir.
> - Off-momentum spin-sensing Results sonunda KISA alt bölüm (yörünge-dışı kaçış).
> - Metod'a giren sistematikler: **β-beat, quad tilt, BPM ofset/kazanç/gürültü**
>   (nihai sayılarda belirleyici olanlar).

---

## 0. Özet (Abstract) — yazılacak

Çekirdek iddia (İngilizcesi `PROJE_ANALIZI §6.1`'de):
- Yörünge-tabanlı hizalama kontrolü, quad-kaynaklı sahte-EDM'i σ=10 μm toleransta
  **~6×10⁻²⁸ e·cm**'e (62× hedef) kadar bastırır; taban **σ² ile ölçeklenir**.
- Bu taban **BPM teknolojisinden bağımsızdır**: sınır beyaz gürültü değil,
  ortalamayla sönmeyen koherent sistematikler (optik-nefes; β-beat × koşullanma).
- 10⁻²⁹ e·cm'e inmek, yörüngeyle **doğrulanamayan** ≲1.3 μm simetrik
  manyetik-merkez hizalaması gerektirir (manyetik ≠ mekanik merkez).

---

## 1. Giriş (Introduction)

1.1 **pEDM ve frozen-spin:** hedef 10⁻²⁹ e·cm ↔ dS_y/dt ≈ 1 nrad/s; MDM≫EDM
kuplajı → katı hizalama gereksinimi. Ana referans: Omarov ve ark., PRD 105,
032001 (2022) — simetrik-hibrit halka, CW/CCW, polarite-switch, CR-ayrım.

1.2 **Sahte-EDM mekanizması:** quad kaçıklığı → geometrik (Berry) faz;
f ∝ dx·dy → σ² (Omarov Fig. 9a; bizim p=2.00 doğrulaması Metod'da).
Öncül: Hacıömeroğlu & Semertzidis, arXiv:1709.01208 (all-elektrik halkada
misplacement→sahte-EDM ileri hesabı).

1.3 **Literatür incelemesi (çözülmüş problemler / belirlenmiş etkiler),
`literatur/` tam-metin karşılaştırmalı:**
- SVD-gözlenebilirlik aygıtı: Chung–Decker–Evans, PAC 1993 s.2263 (tekil-değer
  kanalları, decoupled modlar, ε-truncation, harmonik≈tune) → bizim
  G_k=C/|Q²−k²| bunun harmonik-limiti. **Aygıt 1993'lük; özgünlük uygulamada.**
- Dejenerasyon → yanlı estimator (MSE=Var+Bias²): Wegscheider–Vilsmeier,
  PRAB 26, 032803 (2023) — onlar gradyan/optik, biz transvers kaçıklık.
- BBA prior-art: Martí ve ark., PRAB 23, 012802 (fast AC-BBA); Huang,
  arXiv:2203.14869 (simultaneous BBA). **Bizim özgünlük teknik değil:**
  bu yöntemlerin sahte-EDM'i süren simetrik alt-uzaydaki performansı.
- Komşu: Mirza (PRAB 22, 072804, circulant ORM), Ziemann & Ziemann
  (arXiv:2104.05300, online ORM), Rossbach (Particle Accel. 23, 1989,
  koherent yer-hareketi → COD).
- Polarimetre/zaman bütçesi: arXiv:2010.13536 (`cosy_polarimeter.md`).

1.4 **Omarov'un açık bıraktığı boşluk (`omarov.md §9`):** geometrik-faz kontrolü
CR-ayrımını ölçüp küçültmeye dayanır; ama (i) ölçüm zinciri (48-BPM/SQUID +
K-mod geri-çatım) test edilmemiş, (ii) simetrik (orbit-kör) bileşene körlük
ihtimali ele alınmamış, (iii) polarite-switch'in CW/CCW'ye dejenere olup
olmadığı nicelenmemiş. **Bu makale üçünü de nicel kapatır.**

1.5 **Katkı cümlesi:** "Yörünge-tabanlı ölçüm+düzeltme zincirinin ulaşabileceği
sahte-EDM tabanını (σ'ya bağlı eğri) belirliyoruz; tabanın BPM teknolojisiyle
değil, iki koherent sistematikle ve latis yapısının kendisiyle (AG-alternasyonu)
belirlendiğini gösteriyoruz."

---

## 2. Metod

2.1 **Simülasyon altyapısı:**
- C++ izleyici (`integrator.cpp`): GL4 semplektik parçacık + Thomas-BMT spin;
  elektrik defleksiyon + manyetik FODO; `EDMSwitch` ile gerçek EDM açılır
  (η=1.88×10⁻¹⁵ → 9.81×10⁻¹⁰ rad/s, Eq. C1 ile birebir).
- Latis: 24 FODO, 48 quad, R0=95.49 m, g=0.2 T/m, Q_y≈2.3 (`params.json` tek kaynak).
- Analitik Twiss/R çözücüsü (`analytic_kmod.py`): C++ ile **%1 içinde**
  çapraz-doğrulanmış (`squid_bpm_test.md §5.5`) — hızlı taramalar analitik,
  tüm kritik sonuçlar C++ ile teyit.
- **Metodolojik uyarı (svd.md dersi):** tepki matrisi analitik-Twiss ile değil
  **gerçek demet dinamiğiyle** kurulmalı; aksi hâlde null-mod/koşullanma gizlenir
  (cond 193 vs gerçek 1.2×10⁸, `svd.md §5`).

2.2 **Sahte-EDM estimatörü (kritik reçete):**
- 4D kapalı yörünge + model-fit seküler eğim (`berry_data/false_edm_4d.py`);
  alternatif: 4-katlı simetrik örnekleme (sx,sy)=±1. **Düz polyfit yasak**
  (betatron aliası).
- Doğrulama: σ=10→5→2.5 μm taramasında **p = 2.00 ± 0.01** (saf kuadratik
  geometrik faz, lineer kaçak yok; Omarov Fig. 9a'nın birebir reprodüksiyonu).

2.3 **Antisimetrik/simetrik ayrışım ve gözlenebilirlik dili:**
- QF/QD zıt-işaret (antisim, düşük-k, orbit-görünür; 25 boyut) vs aynı-işaret
  (simetrik, k≈24, orbit-kör; 23 boyut).
- Kazanç yasası G_k = C/|Q²−k²| (C≈24.8, Q²≈5); cond(R)=193.
- COD izi (aynı 10 μm RMS): antisim 114.7 μm vs simetrik 1.69 μm
  (`akilli_duzeltme.md §3`).
- **Metrik tuzağı:** corr simetrik felaketi gizler → daima simetrik-bileşen
  hatası (mean-çıkarmadan RMS) raporlanır (`squid_bpm_test.md §9.3` uyarısı).

2.4 **Sistematik hata modelleri (nihai sayılarda belirleyici olanlar):**
- **BPM ofseti:** statik, per-BPM ~100 μm (varyant: 30 μm); k-mod/farkta iptal,
  tek-yörünge inversiyonunda katil (`svd.md §5.1, §6`).
- **BPM gürültüsü:** ~1 μm tek-atış (beyaz); lock-in ile √N.
- **BPM kazanç hatası:** çarpımsal δg_i, σ_g=%1–10 taraması.
- **β-beat:** per-quad gradyan hatası ε (C++ `quad_dG`); LOCO-artığı temsili
  (%0.5–5). **Ortalamayla sönmeyen model-sistematiği** — belirleyici kalem.
- **Quad tilt (roll) ψ:** iki kanal — (a) ölçüme çapraz-düzlem sızıntısı,
  (b) doğrudan geometrik-faz katkısı (C++ estimatörle; kmod_drivers/tiltscan).
- Seed/ensemble politikası: her iddia çok-seed (tipik 3–40); tek-seed'den
  kök-sebep çıkarılmaz (`akilli_duzeltme.md §6.15.2` dersi).

---

## 3. Simülasyon Sonuçları (Simulation Results)

### 3.1 Taban zinciri ve bastırma eğrisi (pozitif çerçeve)

- Ham sahte-EDM (σ=10 μm): ~10⁻⁶ rad/s ≈ 1000× hedef (~10⁻²⁶ e·cm).
- + CW/CCW farkı: **3.4×** → 474× (~5×10⁻²⁷ e·cm) (`cwccw_telafi`).
- + standart-BPM orbit düzeltme (antisim silinir): **7.7×** → **62× hedef
  (6.05×10⁻⁸ rad/s ≈ 6×10⁻²⁸ e·cm)** (`orbit_duzeltme`; `omarov.md §10`).
- Kalan artık **simetrik/orbit-kör**; σ² yasasıyla ölçeklenir → **ana figür:
  ulaşılabilir sahte-EDM (e·cm) vs σ eğrisi.** 10⁻²⁹ için σ_sym ≈ 1.3 μm gerekir.
- Manyetik ≠ mekanik merkez: ~20 μT parazit dipol → 10–100 μm eşdeğer kayma →
  survey bu σ_sym'i garanti edemez; merkezi yalnız ışın görür (`PROJE_ANALIZI §3`).

### 3.2 Simetrik bileşeni yörüngeden ölçme girişimleri (ters sınıf — hepsi aynı duvar)

**3.2.1 Tek-frekans K-mod (ΔR) + SVD/TSVD/regularizasyon:**
- Temiz inversiyon corr=1.000000 (nefes matrisin içinde, engel değil); ama
  cond(ΔR)=3.7×10⁴, antisim/sim kazanç oranı ~1393× (`squid_bpm_test.md §8`).
- **Lock-in:** beyaz gürültü √N ile yenilir (σ=10 μm tek-atış → 1000 s'de
  corr 0.992); **ama simetrik bileşen lock-in tabanında BİLE ölü**: σ_min≈10⁻⁴ →
  10 nm artık gürültü → 96 μm sim-hata; kurtarma **<4 nm** ister (§9.5).
- **β-beat felaketi:** ε=%0.5 → sim-hata 1931 μm (sinyalin ~50 katı); sistematik,
  hiç ortalanmaz → **SQUID/BPM teknolojisi tabanı değiştirmez.**
- SVD metodolojik dersleri (svd.md): regularize tahminde **ham std ≠ tespit
  tabanı** (gain-kalibrasyon şart, §5.1); N=2 stray-alan örneğinde gerçekçi
  ofsette taban ~45 nT ≈ harmonik-fit (SVD üstünlüğü yok).

**3.2.2 Dağıtık-frekans per-quad K-mod (AC-BBA):**
- Demodüle genlik **nefes-domine** (S/B ≈ 0.14): modüle edilen quad'ın kendi
  ofsetini değil, diğer 47 quad'ın kurduğu yörüngeyi okur (kaldıraç ayrıştırması:
  −9.83 μm vs kendi katkısı −0.037 μm) (`squid_bpm_test.md §6`).
- corr: tek BPM +0.07; **48 BPM'de ≈ −0.03** — nefes koherent, ortalamayla/
  SQUID'le sönmez (§7). C++ ile bağımsız doğrulandı (analitik/C++ oranı ~1.00, §5.5).
- **BPM ofseti bu dalda alakasız:** AC demodülasyon DC'yi söndürür (100 μm →
  65 nm sızıntı; pencere-kilitli frekansta tam sıfır) — ofseti 30 μm'e indirmek
  hiçbir şey değiştirmez.
- **Kavramsal ayrım (öğretici):** nefes ≠ no-go — biri koherent kirlilik
  (dağıtık-frekansı öldürür), öbürü kötü-koşullu inversiyonda rastgele gürültü
  büyümesi (tek-frekansı öldürür). İkisi farklı, ikisi de kapatır (§8).

**3.2.3 NN ile kaçıklık geri-çatımı:**
- misalignment→COD **lineer** → NN yalnız R'yi öğrenir; ters yön = R⁻¹.
- Sayı: simetrik geri-çatım hatası NN 5.6 μm ≈ TSVD 6.3 μm (sinyal 9.9 μm) —
  sınır bilgi-teorik, algoritmik değil (`akilli_duzeltme.md §6.8`).

**3.2.4 LOCO ile başlangıç düzeltmesi + drift izleme:**
- **Çalışan kısım (pozitif):** ofset-iptalli drift modu — 50 μm BPM ofseti
  altında 6.5 μm RMS drift hassasiyeti (mutlağa göre 29×); %1 β-beat'te 6.1 μm
  (`makale-taslagi-2.md §3.4, §3.7`).
- **Sınır:** en kötü 8 mod %96 simetrik, 193× gürültü büyütmesi (§3.8) —
  drift gözcüsü de simetriğe kör. Dualite teoremi: ofset-iptal eden her lineer
  estimator ‖ΔR⁻¹‖~‖R⁻¹‖/ε alt sınırına tabidir (§2.4).

**3.2.5 CR-ayrım gözlenebiliri (Omarov'un önerdiği knob):**
- Doğrudan C++ ölçümü: simetrik bastırma **CR-ayrımda 4.5× ≈ tek-yön COD'da
  3.8×** → ayrım da bir yörünge-farkı, **ek pencere açmaz**; no-go CR-ayrıma
  birebir taşınır (`omarov.md §9.3`). **Omarov §9 boşluğunun kapanışı.**

### 3.3 "Simetriği görünür yap / iptal et" girişimleri

**3.3.1 Quad polarite-flip:**
- f gradyanda **ÇİFT** (iki dönmenin çarpımı, ∝g²): flip oranları sim/antisim/
  genel = +2.81/+0.85/+1.17 (işaret korunur) → flip geometrik fazı iptal etmez
  (`akilli_duzeltme.md §6.14a`).
- **Dejenerasyon:** idealize FODO'da f(CCW,g+) = f(CW,g−) (oran 1.00) →
  Eq. C2'nin 4'lü kombinasyonu 2'liye çöker; flip bağımsız knob değil
  (`kmod_drivers/fast_est cwccw`; 20-seed ensemble ile pekiştirildi).

**3.3.2 Yüksek betatron tune:**
- Q=2.3→10.3 (alternasyon korunur, C++ kararlı): simetriğin **yüksek-m** parçası
  açılır (m=10 kazanç 71 ≈ antisim). Ama **düşük-m** k=24−m≥16 → Q≥16 ister;
  Q_max=12 (μ=180°/hücre stopband) → k-mod ile kurtarma ~%8–25 → bastırma **~2×**.
- **Yüksek-Q'da kalınamaz:** aynı kaçıklık için f ~g³ ile büyür (g 0.21→0.69'da
  32×), gerçek EDM sabit → S/N 32× kötü (`akilli_duzeltme.md §6.15, §6.15.1`).

**3.3.3 Uniform-gradient tanı modu:**
- Tüm |g|∈[0.03, 0.42] **kararsız** (Courant–Snyder: uniform statik quad iki
  düzlemi birden odaklayamaz; deflektör zayıf-odaklama yetmez).
- **Kapanış cümlesi:** simetrik körlük = QF/QD alternasyonu = makinenin
  kararlılığı; gizleyen yapı çökertmeden kaldırılamaz (`§6.14b`).

### 3.4 Yörünge-dışı kısa not: off-momentum spin-sensing (kaçış değil)

- Fikir: magic-dışı momentum → ν_s≠0 → geometrik faz ∝1/ν_s amplifiye (doğru,
  ölçüldü). **Ama** 1.-mertebe ISA-eğimi (σ¹) aynı frekansta ve ~3600× büyük;
  tüm ayırma kanalları kapalı (δ-tarama, faz, 4-fold, RMS, 3D n̂-fit, S_z).
- Magic (ν_s=0) tek çalışma noktası: baskın 1.-mertebe etkiyi zararsız sabite
  çevirir (`akilli_duzeltme.md §6.16`). → Yörünge-dışı bu kaçış da kapalı;
  spin-tabanlı null'lama zaten kapsam dışı (sahte/gerçek degenerasyonu, §4).

### 3.5 Sistematik bütçe (Metod §2.4 modelleriyle, nihai sayılar)

| Kalem | Etkisi | Kaynak |
|---|---|---|
| BPM ofseti (100 μm) | inversiyonda katil; k-mod/AC'de iptal | `svd.md §5.1/§6`, `squid §9.1` |
| BPM gürültüsü (1 μm) | lock-in ile yenilir; sınırlayıcı DEĞİL | `squid §9.3` |
| BPM kazanç (%1) | 48-BPM ortalamasıyla ikincil | `ac_bba_systematics` |
| **β-beat (%0.5–1)** | **inversiyonda felaket (1931 μm); belirleyici** | `squid §9.5` |
| **Quad tilt (roll)** | doğrudan geometrik-faz kanalı; tolerans ~0.3 mrad; CW/CCW+flip'te ODD-baskın (söndürülemez, ~8–57× hedef @1 mrad) | `kmod_bba_sonuclar §7.3` (C++ estimatör; superseded-iddia'dan bağımsız) |

---

## 4. Sonuçlar (Conclusions)

4.1 **Nicel ana sonuç:** yörünge-tabanlı zincir (CW/CCW + orbit düzeltme +
drift izleme) sahte-EDM'i σ=10 μm'de **~6×10⁻²⁸ e·cm**'e bağlar; taban σ² ile
ölçeklenir; 10⁻²⁹ için σ_sym ≲ 1.3 μm gerekir ve bu yörüngeyle doğrulanamaz.

4.2 **Sınır teoremi (negatif çekirdek):** simetrik (orbit-kör) bileşen, denenen
tüm yörünge-tabanlı kanallarda (ΔR/SVD/lock-in, per-quad AC, NN, LOCO+drift,
CR-ayrım) gerekli seviyede **gözlenemez**; sınır beyaz gürültü değil, iki
koherent sistematik (optik-nefes; β-beat×koşullanma) + latis yapısı
(alternasyon=stabilite). **Dolayısıyla SQUID dahil hiçbir BPM teknolojisi tabanı
değiştirmez.**

4.3 **Kapsam/pozisyon notları:**
- Spin-tabanlı aktif null'lama kapsam dışı: sahte/gerçek EDM aynı gözlenebilirde
  dejenere (null'lamak gerçeği de siler); ayıran tek araç CW/CCW farkı, o da
  simetrik artığı bırakır. Ayrıca polarimetre istatistiği iteratif spin-null'u
  zaman-yasak kılar (~yıl/ölçüm; `cosy_polarimeter.md §4`).
- Yörüngeden f'i **ileri-öngörme** (öğrenilmiş/analitik fonksiyonel) ayrı bir
  açık problemdir; bu makalenin ölçüm-sınırı iddiası onu kapsamaz (tek cümle).
- Omarov'a göre konum: geometrik-faz kontrolünün fiziği (Fig. 16) doğrulanır;
  açık bırakılan ölçüm-zinciri boşluğu (i-iii, §1.4) nicel kapatılır. Pratik
  öneri: standart-BPM orbit-monitör = SQUID'in ucuz ikamesi (antisim kısımda
  7.7×) + drift gözcüsü; simetrik artık **tasarım toleransıyla** (σ²) sınırlanır.

4.4 **Geçerlilik sınırları (dürüstlük):** tek latis (24 FODO, Q≈2.3), lineer
(sekstüpolsüz) model; 3.4×/7.7×/62× sınırlı seed istatistiği (makale öncesi
ensemble genişletme ucuz sağlamlaştırma); tilt ayrı doğrudan kanal, misalignment
bütçesinin dışında raporlanır.

---

## 5. Figür planı (öneri)

1. **Fig. 1:** Mekanizma + σ² doğrulaması (p=2.00; Omarov Fig. 9a karşılığı).
2. **Fig. 2:** Antisim/sim ayrışım + G_k yasası + cond(R)=193 (SV spektrumu).
3. **Fig. 3 (ANA):** Ulaşılabilir sahte-EDM (e·cm) vs σ — ham/CW-CCW/orbit-düzeltme
   basamakları + simetrik taban; 10⁻²⁹ çizgisi ve σ_sym≈1.3 μm kesişimi.
4. **Fig. 4:** Ters-sınıf duvarı — lock-in corr vs T + sim-bileşen hatası
   (corr tuzağı görselleştirmesi); β-beat taraması.
5. **Fig. 5:** Nefes — per-quad genlik ayrıştırması (kendi ofseti vs komşu
   yörünge; 2→48 BPM corr≈0).
6. **Fig. 6:** CR-ayrım körlüğü (4.5× vs 3.8×) + flip dejenerasyonu.
7. **Fig. 7:** Yüksek-Q: m-bazında kazanç (Q=2.3 vs 10.3) + g³ ölçekleme.

## 6. Sayı → kaynak eşleme (iç referans; makaleye girmez)

| Sayı | Kaynak belge | Script |
|---|---|---|
| p=2.00±0.01 | `omarov.md §10`, `orbit_ileri_olcum.md §9` | `sigma_olcekleme` |
| 3.4× / 7.7× / 62× | `omarov.md §10` | `cwccw_telafi`, `orbit_duzeltme` |
| cond(R)=193; 114.7 vs 1.69 μm | `akilli_duzeltme.md §2-3` | `surrogate`, `analyze` |
| cond(ΔR)=3.7e4; <4 nm; β-beat 1931 μm | `squid_bpm_test.md §8-9.5` | `v27_recheck`, `v27_syst2` |
| Nefes S/B=0.14; 48-BPM corr≈0 | `squid_bpm_test.md §5-7` | `single_bpm_test`, `breathing_cpp` |
| NN 5.6 ≈ TSVD 6.3 μm | `akilli_duzeltme.md §6.8` | `nn_vs_Rinv` |
| Drift 6.5 μm @ 50 μm ofset | `makale-taslagi-2.md §3.4` | `drift_monitor/` |
| CR 4.5× ≈ COD 3.8× | `omarov.md §9.3` | `cr_separation` |
| Flip +2.81/+0.85/+1.17; CCW≡CW+flip | `akilli_duzeltme.md §6.14`, `kmod_bba_sonuclar §7.3.1` | `flip_real`, `cwccw_ens` |
| Yüksek-Q: kazanç 71, ~2×, g³ 32× | `akilli_duzeltme.md §6.15` | `highQ_bba` |
| Uniform-g kararsız | `akilli_duzeltme.md §6.14b` | `diag_mode` |
| Off-momentum σ¹/σ² ~3600× | `akilli_duzeltme.md §6.16` | `momentum_*` |
| Tilt toleransı ~0.3 mrad; ODD-baskın | `kmod_bba_sonuclar.md §7.3` | `tiltscan`, `cwccw_ens` |
| Ofset-iptal dualite ‖ΔR⁻¹‖~‖R⁻¹‖/ε | `makale-taslagi-2.md §2.4` | — |
| Gain-kalibrasyon dersi; 45 nT tabanı | `svd.md §5.1` | `field_n2_nodc` |

## 7. Kaynakça (çekirdek)

1. Z. Omarov et al., PRD **105**, 032001 (2022).
2. S. Hacıömeroğlu, Y. Semertzidis, arXiv:1709.01208.
3. Y. Chung, G. Decker, K. Evans, Proc. PAC 1993, s.2263.
4. S. Wegscheider, A. Vilsmeier et al., PRAB **26**, 032803 (2023).
5. Z. Martí, G. Benedetti, U. Iriso, A. Franchi, PRAB **23**, 012802 (2020).
6. X. Huang, arXiv:2203.14869.
7. M. Mirza et al., PRAB **22**, 072804 (2019).
8. V. Ziemann, M. Ziemann, arXiv:2104.05300.
9. J. Rossbach, Particle Accel. **23**, 121 (1989).
10. COSY LYSO polarimetre, arXiv:2010.13536.
