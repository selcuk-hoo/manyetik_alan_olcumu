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
ve misalignment ile **kuadratik (σ²)** ölçeklenir. Bu kanalı süren misalignment
deseni **simetrik alt-uzaydadır** (hücre içi QF/QD aynı yönlü) ve **kapalı yörüngeye
neredeyse görünmezdir**: simetrik ofset → alternatif (yüksek-k, k≈24) kick →
rezonant yörünge tepkisi G_k ∝ 1/|Q²−k²| ile bastırılır (Q≈2.7 ≪ 24). Sonuç:
yörünge-tabanlı yeniden çatım bir gözlenebilirlik tabanına (~1.8×10⁻⁴) çarpar; 6
farklı metot (R-LS, CLEAN, Bozoki, R⁻¹, TSVD) aynı tabana takılır.

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
  çekilir. Düz polyfit veya tek-parçacık CO=True KULLANMAYIN (ideal-olmayan
  parçacıkların sahte EDM üretimini temsil etmez — bu hata tekrar tekrar yapıldı).

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
