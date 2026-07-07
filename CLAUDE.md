# CLAUDE.md — manyetik_alan_olcumu

Proton EDM (elektrik dipol momenti) deneyinde halka mıknatıs hizalama hatalarının
sahte EDM sinyaline etkisini inceleyen simülasyon ve analiz kodu. Python (analiz),
C++ (parçacık + spin izleyici), JSON (konfigürasyon) ve LaTeX (makale taslağı) içerir.

> **Not (2026-06):** Repo büyük bir temizlikten geçti. Eski keşif/test scriptleri
> (`test_*.py`, `fig_*.py`, `make_paper_figures.py`, `false_edm_mode_scan.py` vb.)
> ve PNG'ler silindi; mantıkları git geçmişinde ve `.md` günlüklerinde korunuyor.
> Geçici/keşif kodları artık `/tmp` altında tutulup `.md` dosyalarından referans
> veriliyor (kalıcı repoya girmiyor).

---

## Proje Amacı (ve güncel durum)

**Başlangıç hedefi:** Bir proton depolama halkasındaki 48 kuadrupolün hizalama
hatalarını (~10 μm) kapalı yörüngeden ölçmek. Zorluk: BPM ofsetleri (~100 μm)
asıl sinyal (~10 μm) ile aynı büyüklükte → k-modülasyon + Fourier harmonik
yeniden çatımı ile maske kaldırılır.

**Ulaşılan ana sonuç (bkz. `false_edm_harmonic_sinir.md`, `omarov_symmetric_hybrid.md`):**
Sahte EDM, kuadrupol kaçıklığının **dx·dy geometrik (Berry) faz** kanalından gelir
ve misalignment ile **kuadratik (σ²)** ölçeklenir; mekanizma kapalı yörüngenin
çarpımıdır, **f ∝ x_CO·y_CO**.

> **Önemli düzeltme (2026-06, `orbit_ileri_olcum.md`):** "Sahte EDM'yi simetrik
> alt-uzay domine eder" ifadesi **yanıltıcıdır**. Doğru tablo: **ham (düzeltme
> öncesi) sahte EDM'yi ANTİSİMETRİK (orbit-GÖRÜNÜR) alt-uzay domine eder** (~37×;
> antisimetrik kaçıklık büyük kapalı yörünge → büyük x_CO·y_CO). Simetrik kısım
> **düzeltme-SONRASI artık**tır: deneyde kapalı yörünge sıfıra çekilince
> orbit-görünür (antisimetrik) kısım silinir, geriye orbit-kör simetrik artık kalır.

Bu simetrik artık kapalı yörüngeye neredeyse görünmezdir: simetrik ofset →
alternatif (yüksek-k, k≈24) kick → rezonant yörünge tepkisi G_k ∝ 1/|Q²−k²| ile
bastırılır (Q≈2.7 ≪ 24). Sonuç: yörünge-tabanlı **inversiyon** (R⁻¹ ile kaçıklık
geri-çatımı) bir gözlenebilirlik tabanına (~1.8×10⁻⁴) çarpar; 6 farklı metot
(R-LS, CLEAN, Bozoki, R⁻¹, TSVD) aynı tabana takılır. No-go bir *inversiyon*
sınırıdır; sahte EDM kapalı yörüngenin fonksiyoneli olduğundan **ileri yönde**
öngörmek inversiyona girmez — ama bu, **doğru fonksiyonel bilinirse** geçerlidir
ve o fonksiyonel empirik olarak pinlenemedi (basit ⟨x·y⟩ yanlış; Berry yönlü-alan
Σ(x1y2−x2y1) en tutarlı lead ~−0.5; analitik türetme gerekiyor). **Açık problem,
çözülmedi** (`orbit_ileri_olcum.md`).

> **Strateji (2026-06):** Drift monitör (antisimetrik drift, standart BPM) temiz
> ve bitmiş katkı; Omarov'un 1 nrad/s spin-kalibrasyonuyla **yarışmaz, tamamlar**
> — veri-alımı sırasında girişimsiz sürekli "drift gözcüsü" rolü. Orbit→spin
> fonksiyoneli ayrı, yüksek-riskli açık problem; ancak analitik tutamak çıkarsa
> peşine düşülmeli (yoksa trim/Omarov çıkmazına döner).

**Açık stratejik soru:** Bu simetrik alt-uzayı **operasyonel olarak basit** bir
kapalı-yörünge gözlemiyle kurtarmak mümkün mü? (per-quad k-mod ve bilineer yörünge
imzası fikirleri operasyonel açıdan ağır bulundu — `omarov_symmetric_hybrid.md §9`.)
Spin-tabanlı çözüm (Omarov SBA) literatürde mevcut olduğundan, özgün katkı yörünge
tarafındaki kesin sınır teoreminde aranıyor.

---

## Dizin Yapısı (güncel)

```
.
├── integrator.cpp / integrator.py / lib_integrator.so / integrator.dylib
│                               # Ana motor: C++ GL4 semplektik parçacık + spin
│                               #   (Thomas-BMT) izleyici + ctypes köprüsü
├── integrator2.cpp / integrator2.py / lib_integrator2.so
│                               # İkinci motor: alternatif topoloji (QF-d-QD-d-DEFL-d)
├── build_integrator.sh         # Build betiği (Linux g++ / macOS clang++)
│
├── params.json                 # Tüm simülasyon parametreleri (tek kaynak)
│
├── build_response_matrix.py    # 48×48 tepki matrisleri (R_dy/R_dx, paralel)
├── reconstruction.py           # Fourier harmonik + LASSO (ADMM) yeniden çatım
├── fourier_reconstruct.py      # FODO bazı + R-LS + CLEAN; harmonik kalite
├── bozoki_ls.py                # Bozoki 1989 vs R-matris vs CLEAN karşılaştırması
│
│   # ── Drift İzleme (ikinci makale adayı) ──
├── drift_monitor/
│   ├── fodo_lattice.py         # Analitik Twiss + R matrisi (C++ bağımlılığı yok)
│   ├── drift_monitor_sim.py    # Test 4: kalibrasyon-referans drift gösterimi
│   ├── test8_betabeat.py       # Test 8: β-beating sağlamlık taraması
│   ├── permode2.py             # SVD per-mod analizi (no-go bağlantısı)
│   └── test_params.json        # Test parametreleri (BPM ofset/gürültü/drift)
├── makale-taslagi-2.md         # İkinci makale taslağı: drift izleme + dualite teoremi
│
│   # ── Belgeler (.md bağımsız okunabilir; pedagojik) ──
├── README.md                   # Ana proje belgesi (Türkçe, pedagojik, ~1300 satır)
├── YÖNTEM.md                   # Detaylı yöntem açıklaması
├── FOURIER_REKONSTRUKSIYON.md  # Fourier yöntemi teorisi
├── clean_rls_yontemleri.md     # CLEAN / R-LS rekonstrüksiyon yöntemleri
├── false_edm_yontemi.md        # Sahte EDM yöntemi (erken)
├── false_edm_harmonic_sinir.md # ANA GÜNLÜK: sahte EDM analizi (§1–15)
├── trim_yontemi_pedagojik.md   # Trim yöntemi sıfırdan anlatım (iki kademe)
├── cosy_polarimeter.md         # COSY LYSO polarimetre özeti + zaman bütçesi
├── omarov_symmetric_hybrid.md  # Omarov PRD 105,032001 özeti + kanal araştırması
├── omarov.md                   # Omarov PRD 105,032001 DİKKATLİ OKUMA: geometrik-faz
│                               #   kontrolü (CW+CCW+polarite, CR-ayrım); SBA değil;
│                               #   §9 BPM/ayrım-ölçüm boşluğu; §10 nerede duruyoruz
├── orbit_ileri_olcum.md        # İLERİ-ÖLÇÜM: f∝x_CO·y_CO; no-go=inversiyon;
│                               #   standart BPM SQUID yerine; ham=antisim/artık=sim
│                               #   §9 make-or-break sonuçları (7.7×, 62×, p=2.00)
├── squid_bpm_test.md           # K-mod+BPM ölçümü: §7 dağıtık-frekans ÖLÜ (nefes);
│                               #   §8 ΔR no-go; §9 lock-in; §9.5 simetrik no-go (<4nm)
├── akilli_duzeltme.md          # AKILLI DÜZELTME: Kol B (NN COD→f) kavramsal AÇIK (4
│                               #   bulgu: iyi-koşullu/öğrenilebilir/β-beat-şeffaf) AMA
│                               #   pratik HENÜZ ÇALIŞMIYOR (Plan 2c: null'lama orbit-
│                               #   düzeltmeyi geçemiyor; harita kaba). Plan 4 (analitik)
│                               #   gerek. Kol A (spin) çalışır. (ilk "ölü" düzeltildi)
├── akilli_duzeltme_pedagojik.md # ↑ belgenin DERS KİTABI tarzı kardeşi (sıfırdan;
│                               #   hata+düzeltme hikâyesi, ileri-harita≠inversiyon, SSS)
├── MAKALE_POTANSIYELI.md / PROJE_ANALIZI_VE_ONERILER.md / SIMULASYON_PLANI_HIZALAMA.md
├── YAPILACAKLAR.md             # Aktif yapılacaklar
├── injection_kick_raporu.md    # Enjeksiyon kick raporu
│
├── makale_trim_tr.tex          # Türkçe makale taslağı (iki-kademe trim)
│
│   # ── git-tracked PDF referanslar (.gitignore *.pdf'i atlar; -f ile eklenir) ──
├── a-new-polarimeter.pdf       # COSY polarimetre makalesi (arXiv:2010.13536)
├── Comprehensive ... 10−29e·cm.pdf  # Omarov makalesi (PRD 105, 032001)
└── proje_onerisi.pdf
```

---

## Build: C++ İzleyici

**integrator.cpp değiştirildikten sonra her zaman yeniden derleyin.** Eski binary
sessiz hata üretir: quad_dy/quad_tilt dizileri Python'dan gönderilir ama eski
kütüphane uygulamaz → tüm simülasyonlar hizalama hatası yokmuş gibi davranır.

```bash
bash build_integrator.sh
python3 -c "import integrator; print('yüklendi')"   # doğrulama
```

Derleme bayrakları: `-O3 -shared -fPIC -std=c++17`
- Linux: `g++` → `lib_integrator.so`
- macOS: `clang++` → `integrator.dylib`

**İkinci motor (`integrator2.cpp`):** Alternatif kafes topolojisi için bağımsız
kopya; `integrator2.py` yalnız `lib_integrator2.so` yükler. Derleme:
`g++ -O3 -shared -fPIC -std=c++17 integrator2.cpp -o lib_integrator2.so`.
Dikkat: bu topoloji g₀=0.2 T/m ile kararsızdır; g≈0.5 T/m gerekir.

---

## Konfigürasyon: params.json

Tüm simülasyon parametreleri `params.json` içinde; scriptler `json.load()` ile okur.

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `R0` | 95.49 m | Halka yarıçapı |
| `nFODO` | 24 | FODO hücre sayısı (= 48 kuadrupol) |
| `quadLen` | 0.4 m | Kuadrupol uzunluğu |
| `g0`, `g1`, `g2` | 0.2, 0.21, 0.21 T/m | Kuadrupol gradyanları |
| `dt` | 1e-11 s | İntegrasyon adımı (büyütmek hassasiyeti bozar) |
| `t_pr` | 1000 | Poincaré periyot sayısı |
| `error_quad_index` | -1 | Hata verilen kuadrupol (-1: yok) |
| `bpm_noise_sigma` | 0.0 m | BPM elektronik gürültüsü |
| `bpm_offset_sigma` | 0.0 m | BPM ofset RMS'i |
| `dy_harmonics` | [...] | Dikey hizalama hata harmonikleri |
| `k_search_max` | 12 | Aramada max harmonik sayısı |
| `lasso_lambda` | 0.02 | LASSO düzenlileştirme katsayısı |

Sistematik testler `params.json`'u bellek içinde kopyalayarak tarar; dosyayı değiştirmez.

---

## Mimari

```
Analiz / keşif scriptleri (kalıcı: build_response_matrix, fourier_reconstruct,
        │                  reconstruction, bozoki_ls; geçici: /tmp/*.py)
Yeniden Çatım Katmanı (reconstruction.py, fourier_reconstruct.py, build_response_matrix.py)
        │
Python İzleyici Arayüzü (integrator.py + FieldParams)
        │ ctypes
C++ Fizik Motoru (integrator.cpp, GL4, Thomas-BMT)
        ▲
params.json ──→ tüm katmanlar
```

**FieldParams** (`integrator.py`): C++ motoruna aktarılacak alanları, RF
özelliklerini ve hata profillerini tutar. `integrate_particle()` çağrılmadan önce
ilgili alanlar set edilir; `quad_dy`, `quad_dx` dizileri kaçıklıkları taşır.

**Tepki matrisi** (`build_response_matrix.py`): `ProcessPoolExecutor` ile paralel.
Her kuadrupola küçük Δdy perturbasyonu uygulanır, kapalı yörünge ölçülür → 48×48 R.
İki k-mod konfigürasyonu için `R_dy_1.npy`, `R_dy_2.npy`, `R_dx_1.npy`, `R_dx_2.npy`.

---

## Geliştirme İş Akışı

```bash
# 1. İzleyiciyi derle (integrator.cpp değiştiyse)
bash build_integrator.sh

# 2. Tepki matrislerini üret (önce bir kez; *.npy git'te değil)
python3 build_response_matrix.py

# 3. Rekonstrüksiyon / harmonik analiz
python3 fourier_reconstruct.py
python3 bozoki_ls.py
```

**Sahte-EDM / spin analizleri** keşif scriptleriyle yapılır; bunlar kalıcı repoda
tutulmaz, `/tmp` altında yazılıp ilgili `.md` günlüğünden referans verilir (ör.
`false_edm_harmonic_sinir.md` içinde reprodüksiyon yolları ve git-hash'leri).
Yeniden çalıştırmak için o günlüklerdeki snippet/yol notlarına bakın.

### Sahte-EDM ölçümü için doğru reçete (kritik)
- **4-katlı simetrik parçacık örneklemesi:** (sx, sy)=±1 dört kombinasyon ortalaması
  → betatron + ⟨ΔxΔy⟩ artığını tam söndürür (CO arama gerektirmez).
- **Model-fit estimator:** S_y(t)=a+bt+Σ_k[c_k cos+d_k sin]; yalnız sekuler eğim b
  çekilir. Düz polyfit KULLANMAYIN.
- **NOT (2026-06, doğrulandı):** 4D-CO + model-fit (tek ideal parçacık kapalı
  yörünge üzerinde; `berry_data/false_edm_4d.py`) **σ²-testinde p=2.00 verir**
  (σ=10/5/2.5μm; geometrik faz, lineer kaçak yok — `omarov.md §10`,
  `orbit_ileri_olcum.md §9`). Yani CO+model-fit, geometrik-faz sahte-EDM'i DOĞRU
  ölçer; 4-katlı simetrik örnekleme alternatif/eşdeğer yoldur, şart değil. Eski
  "tek-parçacık CO=True kullanma" uyarısı **düz polyfit + CO** kombinasyonu içindir
  (model-fit ile değil). Omarov da aynı yöntemi kullanır (Fig. 9a, Eq. C3–C5).

### Bağımlılıklar
Python 3.6+, `numpy`, `matplotlib`; `ctypes`/`concurrent.futures` (stdlib).
`pip install numpy matplotlib`. PDF okuma için `poppler-utils` (pdftotext).

---

## Ana Bulgular ve Belge Haritası

| Bulgu | Belge |
|-------|-------|
| Sahte EDM ∝ dx·dy geometrik faz, σ² ölçekleme; dy-only doğrusal/küçük | `false_edm_harmonic_sinir.md §13` |
| Yörünge-trim k≤4 tek başına ~2.7× (yetmez); simetrik taban ~3×10⁻⁴ gerçek | `false_edm_harmonic_sinir.md §14` |
| 6 rekonstrüksiyon metodu aynı gözlenebilirlik tabanına çarpar | `false_edm_harmonic_sinir.md §14.5` |
| Spin ölç-trim simetrik artığı ~6000× temizler (→1.6×10⁻⁷) | `false_edm_harmonic_sinir.md §14.6` |
| İki-kademe (orbit + spin) pedagojik anlatım | `trim_yontemi_pedagojik.md` |
| COSY polarimetre performansı + EDM zaman bütçesi (dS_y/dt~1nrad/s tabanı) | `cosy_polarimeter.md` |
| Omarov SBA + yükselt-söndür; quad-flip neden simetrik alt-uzayı açmaz | `omarov_symmetric_hybrid.md §5,§9` |
| Stratejik karar: saf spin-trim'de özgün katkı dar | `omarov_symmetric_hybrid.md §8`, `false_edm_harmonic_sinir.md §15` |
| **Ham sahte EDM'yi ANTİSİMETRİK (orbit-görünür) domine eder (~37×); sim=düzeltme-sonrası artık** | `orbit_ileri_olcum.md §6` |
| **Estimator doğrulandı: sahte EDM σ²-ölçeklenir, p=2.00 (geometrik faz, lineer kaçak yok)** | `orbit_ileri_olcum.md §9`, `omarov.md §10` |
| **Make-or-break: orbit-düzeltme EDM-kanalı kirliliğini 7.7× düşürür (artık 62× EDM, simetrik kör)** | `orbit_ileri_olcum.md §9`, `omarov.md §10` |
| **Gerçek EDM=9.8e-10 (TEK, (CW−CCW)/2=Omarov Eq.C1); CW/CCW telafisi tek başına 3.4×** | `omarov.md §3,§10` |
| **Omarov: geometrik faz=CW+CCW+polarite+CR-ayrım (SBA DEĞİL); SBA→E-alan/vert.velocity** | `omarov.md §2-5` |
| **Omarov BOŞLUK: CR-ayrım ÖLÇÜMÜ (48-BPM/SQUID+K-mod) test edilmedi; simetrik-artığa körlük açık** | `omarov.md §9` |
| **Dejenerasyon: idealize FODO'da CCW≡CW+polarite-flip (Eq.C2 4'lü→2'li)** | `omarov.md §6.1`, git Test 7 |
| f kapalı-yörünge fonksiyoneli ama ⟨x·y⟩ DEĞİL; Berry yönlü-alan en tutarlı lead | `orbit_ileri_olcum.md §2-3` |
| İleri-ölçüm no-go'yu atlar MI? — **AÇIK** (Kol B kapatılmadı; orbit_ileri §7 doğru) | `akilli_duzeltme.md §7` |
| **Akıllı düzeltme Kol B (NN COD→f): ileri-harita İYİ KOŞULLU (∂f/∂COD≈0.15, 1/σ_min DEĞİL)** | `akilli_duzeltme.md §4` |
| **7nm COD ortalamayla ulaşılır (~21s); ofset yönlü-alanla değişmez → gürültü Kol B'yi dışlamaz** | `akilli_duzeltme.md §4` |
| **Simetrik COD→f kanalı ÖĞRENİLEBİLİR: temiz COD'dan CV R² 240 örnekte +0.77 (80'de ~0; harita karmaşık ama var)** | `akilli_duzeltme.md §6` |
| **β-beat model-fidelity ŞEFFAF: %1 β-beat'li makineye sim-harita held-out nominal kadar taşınıyor (R² 0.62 vs 0.61)** | `akilli_duzeltme.md §6.5` |
| **Harita ÇIKARMA için kullanılamaz (~%0.1 mutlak gerek); CW/CCW β-beat'i geri ALMAZ (corr −0.89, EDM-kanalı); yalnız NULL'LAMA** | `akilli_duzeltme.md §6.6` |
| **Büyük-genlik eğitimi: simetrik (asıl) kanal 1mm'e σ²-HOMOJEN (p=2.03); antisim doyar (p=0.9) → orbit-kör deseni büyüt, gerçek makinede öğren, ölçekle** | `akilli_duzeltme.md §6.7` |
| **Omarov CR-ayrım gözlenebilirinin simetriğe körlüğü DOĞRUDAN ölçüldü: bastırma CR 4.5× ≈ tek-yön COD 3.8× → no-go CR'ye taşınır (Omarov §9 boşluğu kapandı)** | `omarov.md §9.3` |
| **NN ile misalignment geri-çatımı R⁻¹'den İYİ DEĞİL (lineer harita → NN=R; simetrik hata NN 5.6μm≈TSVD 6.3μm); fark algoritmada değil problemin yönünde (ileri/ters)** | `akilli_duzeltme.md §6.8` |
| **Plan 2c (make-or-break) NEGATİF: kapalı-döngü null'lama basit orbit-düzeltmeyi GEÇEMİYOR; harita hata tabanı ~300× hedef (slope 0.36) → 240-örnek harita yetersiz; Plan 4 (analitik) gerek** | `akilli_duzeltme.md §6.10` |
| **Güvenli null'lama (ensemble+pessimistic) naif model-istismarını dizginler (geomean 236×→67×) ama orbit-düzeltmeyi (2.1×) yine geçemez → algoritma gerekli-yetersiz, kök çözüm daha doğru harita** | `akilli_duzeltme.md §6.11` |
| **Plan 5 (spin-modülasyon+gradient descent) kavramsal POZİTİF (726×) AMA pratik ZAMAN-YASAK: dS_y/dt'yi EDM-seviyesine ölçmek ~50 yıl (istatistik ∝1/√T); ~40-100 ölçüm → iteratif spin-null İMKÂNSIZ. Asıl bağ: orbit hızlı-ama-kör, spin doğru-ama-yavaş** | `akilli_duzeltme.md §6.12` |
| **Birleşik no-go (inversiyon+lock-in) yalnız misalignment geri-çatımı; ileri-harita o sınıfta DEĞİL** | `akilli_duzeltme.md §4,§7` |
| **Simetrik orbit deseni temsil edilir ($y_{sym}=Σσ_i(v_i·Δq)u_i$) ama İKİ duvar: (1) k-mod ofseti siler, koşullanma (193×/<4nm) kalır; (2) orbit-null≠f-null → düz orbit f'i ham Σdx·dy'ye çevirir, doğru amaç merkezlerden geçmek (BBA=ölçülemeyen simetrik kaçıklık)** | `akilli_duzeltme.md §6.13`, `PROJE_ANALIZI §5.1b` |
| **Quad-flip sahte-EDM'i İPTAL ETMEZ: f gradyanda ÇİFT (g²; iki dönmenin çarpımı) — flip oranı sim/antisim/genel +2.8/+0.85/+1.17 (işaret korunur). Omarov flip'i tek-in-g sistematikleri temizler, geometrik fazı değil** | `akilli_duzeltme.md §6.14` |
| **"Simetriği orbit-görünür yap" kaçışı KAPALI: uniform-gradient tanı modu TÜM \|g\|'de kararsız (Courant–Snyder: uniform quad iki düzlemi odaklayamaz). Simetrik-körlük = AG-alternasyonu = stabilite; gizleyen yapı makineyi çökertmeden kaldırılamaz** | `akilli_duzeltme.md §6.14`, `PROJE_ANALIZI §7` |
| **Tune-yükseltme (Q~10, alternasyon korunur → C++ kararlı) simetriğin YÜKSEK-m parçasını açar (kazanç 71 ≈ antisim; "min-gain" hükmüm yanlıştı). Ama DÜŞÜK-m Q≥16 ister, Q_max=12 (μ=180° stopband) → k-mod ile ~%8-25 kurtarma → sahte-EDM bastırma yalnız ~2× (marjinal). Kaçış kısmen açık ama pratikte yetersiz** | `akilli_duzeltme.md §6.15` |
| **Yüksek-Q'da KALINAMAZ: sahte-EDM ~g³ ile büyür (g=0.21→0.69'da AYNI kaçıklık için 32×, C++), gerçek EDM (E-alan) sabit → S/N 32× kötü. Yüksek-Q yalnız geçici TANI; kalırsan arka plan patlar, dönersen rekonstrüksiyon ~%25 taşır. (Kullanıcının "güçlü kick→f büyür" sezgisi doğru.)** | `akilli_duzeltme.md §6.15.1` |
| **Off-momentum spin-sensing (magic-olmayan p + E-rescale): telafi ÇALIŞIR (tek quad ofseti magic f'i 182→0 @+22μm null'lar, rank duvarı yok) AMA off-momentum proxy YANILTIR — geometrik faz σ² AMPLİFİYE olur (∝1/ν_s, kullanıcı sezgisi doğru) fakat 1.-mertebe ISA-eğimi σ¹ aynı ν_s'te ~3600× büyük gömüyor. Tüm ayırma kapalı: δ(kesişim δ~3e-7), faz(desene bağlı 91°), 4-fold(σ¹ sönmez, oran 1.000), RMS(kesişim 62mm), 3D n̂-fit(artık~0, koni 8e-14 temiz), boylamsal S_z(off-magic knob-duyarsız; çalma magic'te σ⁴). Magic ν_s=0 = 1.-mertebeyi nötrleyen TEK nokta** | `akilli_duzeltme.md §6.16` |
| **Kol A (spin) çalışır (bilgi spinde, doğrudan); Kol B'de bilgi yörüngede ama küçük+karmaşık** | `akilli_duzeltme.md §9`, `§14.6` |
| **K-mod+BPM ölçüm zinciri: dağıtık-frekans ÖLÜ (nefes); v2.7 ΔR no-go; lock-in antisim kurtarır, sim kurtaramaz** | `squid_bpm_test.md §7,§8,§9.5` |
| **Drift izleme (ikinci makale)** | |
| Kalibrasyon-ref drift: 50 μm BPM ofsetine rağmen 6.6 μm RMS; mutlak 197 μm → 29× | `makale-taslagi-2.md §3.4`, `drift_monitor/drift_monitor_sim.py` |
| Dualite teoremi: iki-ölçüm tam-ofset-iptal sınıfı C'de ΔR⁻¹ tek çözüm; ‖ΔR⁻¹‖~‖R⁻¹‖/ε | `makale-taslagi-2.md §2.4` |
| Test 8 (β-beating): %1→6.1 μm, %5→8.6 μm; LOCO kalitesiyle operasyonel | `makale-taslagi-2.md §3.7`, `drift_monitor/test8_betabeat.py` |
| SVD per-mod: en kötü 8 mod %96 simetrik, 193× gürültü; no-go ile doğrudan bağlantı | `makale-taslagi-2.md §3.8`, `drift_monitor/permode2.py` |

---

## Önemli Tuzaklar

1. **Eski binary sessiz hata:** `integrator.cpp` değişip `build_integrator.sh`
   çalıştırılmazsa quad hata dizileri uygulanmayabilir. Belirti: tüm sonuçlar
   hizalama sıfırmış gibi çıkar.
2. **Derlenmiş binary'ler git'te:** `lib_integrator.so`, `integrator.dylib` repoda.
   Cross-platform geliştirmede yanlış platform binary'sini commit etmeyin.
3. **Numpy/PNG/CSV git'te değil:** `.gitignore` `*.npy/*.npz/*.csv/*.png` dışlar.
   Tepki matrisleri her ortamda yeniden üretilmeli.
4. **`*.pdf` ve `*.sh` de .gitignore'da:** referans PDF eklerken `git add -f` şart.
5. **Konfigürasyon tek kaynaktır:** parametreleri hardcode etme; `params.json`.
6. **dt'yi büyütme:** integrasyon adımını büyütmek spin hassasiyetini bozar
   (CO arama için bile). Onun yerine CO ölçeklemesi kullan.
7. **Sahte-EDM estimator'ı:** 4-katlı simetri + model fit (yukarı bkz.); aksi
   halde betatron aliası / ⟨ΔxΔy⟩ artığı yapay düz/yanlış sonuç verir.
8. **cell-0 QF özel eleman (`QUAD_F_MOD`, tip 4):** `quad_dG`'yi OKUMAZ (yalnız
   `quadG0`+AC-mod kullanır). Quad-flip / uniform-gradient için `quad_dG=−2` ile
   naif çevirme cell-0 QF'i atlar → tek yanlış-polariteli quad makineyi PATLATIR
   (sahte kararsızlık). Doğru yol: config'te `g0` VE `g1` işaretini çevir. Bkz.
   `akilli_duzeltme.md §6.14`.
9. **Kararsız harekette tune tahmini:** `cos(2πQ)` recurrence/fit üstel büyüyen
   diziye sahte |cos|<1 verir. Stabilite için **genlik büyüme faktörü** (özdeğer
   büyüklüğü) oku, fitlenen tune değil.

---

## Dil ve Belgeleme Kuralları

- **Kod ve belgeler Türkçedir.** İsimler İngilizce; yorum/docstring Türkçe.
- `README.md` pedagojiktir; `.md` dosyaları bağımsız okunabilir belgelerdir.
- Yeni script eklerken mevcut Türkçe yorum stilini taklit edin.

---

## Git Dalları ve Push Politikası

- `main` — kararlı dal. **Kullanıcı talimatı: tüm çalışmalar her zaman main'e
  push edilir** ("bütün yaptıklarını her zaman main'e push et").
- Aktif oturum dalı: `claude/awesome-babbage-nmi6w9`.
- Sürüm etiketleri: en güncel **v4.3** (doküman tutarlılık geçişi: stratejik
  karar + lineer-model düzeltmesi + COSY/Omarov referans özetleri + no-go/§19 +
  ölü-betik temizliği). Kod v4.x boyunca değişmedi; v4.2 v4.3 ile aynı kodu ama
  eski/hatalı dokümanı işaret ettiğinden kaldırıldı. **Bu ortamda tag push'u 403
  ile engellenir**; tag'ler yerel makineden oluşturulup itilir. Branch (main)
  push'u sorunsuz çalışır.
- Commit ön-ekleri: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`.

---

## Fizik Referansı (Kod Bağlamı)

| Kavram | Kodda karşılığı |
|--------|-----------------|
| FODO hücresi | `nFODO=24` → 48 kuadrupol (`2*nFODO`) |
| Kapalı yörünge bozulması (COD) | `error_quad_dy`, `error_quad_dx` / `quad_dy`, `quad_dx` |
| K-modülasyon | `kmod_configs`, `g2` |
| Thomas-BMT + EDM | `integrator.cpp` spin kısmı, `EDMSwitch` |
| Semplektik integrasyon | GL4 (Gauss-Legendre 4. derece) |
| Fourier harmonik tabanı | `fodo_basis()` (`fourier_reconstruct.py`) |
| CLEAN | iteratif çıkarma (`fourier_reconstruct.py`, `reconstruction.py`) |
| LASSO (ADMM) | `lasso_admm()` (`reconstruction.py`) |
| Yörünge kazanç yasası | G_k = C/\|Q_eff²−k²\| (C≈24.8, Q_eff²≈5.03) |
| Fit eşiği | G_k > σ_b/σ_q → k_max² < Q_eff² + C·σ_q/σ_b |
| Antisim/simetrik ayrışım | QF/QD zıt-işaret vs aynı-işaret; 25+23 boyut |
| Sahte EDM mekanizması | dx·dy geometrik faz, σ² (simetrik alt-uzay domine) |
| Spin kuplaj katsayıları | c_k: rezonant değil, tüm k için ≠0 |
