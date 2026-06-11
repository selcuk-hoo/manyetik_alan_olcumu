# CLAUDE.md — manyetik_alan_olcumu

Proton EDM (elektrik dipol momenti) deneyinde halka mıknatıs hizalama hatalarını ölçen simülasyon ve analiz kodu. Python (ana analiz), C++ (parçacık izleyici), JSON (konfigürasyon) ve LaTeX (makale taslakları) içerir.

---

## Proje Amacı

Bir proton depolama halkasındaki 48 kuadrupol mıknatısının hizalama hatalarını (~10 μm hassasiyetle) ölçmek. Ana zorluk: BPM (Beam Position Monitor) ofsetleri (~100 μm) asıl sinyal (~10 μm) ile aynı büyüklükte olduğundan, k-modülasyon + Fourier harmonik yeniden çatımı ile maske etkisi kaldırılır.

---

## Dizin Yapısı

```
.
├── integrator.cpp              # C++ GL4 semplektik parçacık + spin izleyici (ana motor)
├── integrator.py               # ctypes köprüsü + FieldParams sınıfı
├── lib_integrator.so           # Derlenmiş: Linux (git-tracked)
├── integrator.dylib            # Derlenmiş: macOS (git-tracked)
├── build_integrator.sh         # Build betiği (Linux: g++, macOS: clang++)
├── integrator2.cpp             # İKİNCİ motor: alternatif topoloji QF-d-QD-d-ARC-d
├── integrator2.py              # integrator2 köprüsü (lib_integrator2.so yükler)
├── lib_integrator2.so          # Derlenmiş ikinci motor (git-tracked, yalnız Linux)
│
├── params.json                 # Tüm simülasyon parametreleri (tek kaynak)
│
├── build_response_matrix.py    # 48×48 tepki matrislerini hesaplar (paralel)
├── reconstruction.py           # Fourier harmonik + LASSO (ADMM) yeniden çatım
├── fourier_reconstruct.py      # Tek-konfigürasyon harmonik kalite raporu
├── analytic_coupling.py        # x-y kuplaj tahmini, FFT tune ölçümü
├── bozoki_ls.py                # Bozoki 1989 vs R-matris vs CLEAN karşılaştırması
│
├── run_simulation.py           # Ana simülasyon rutini
├── test_kmod_reconstruction.py # K-modülasyon geri çatım testi
├── test_bpm_noise.py           # BPM gürültü sistematikleri
├── test_bpm_offset.py          # BPM ofset etkisi testi
├── test_quad_gradient.py       # Kuadrupol gradyan tarama testi
├── test_quad_tilt.py           # Kuadrupol tilt (x-y kuplaj) testi
├── test_combined_systematics.py# Tüm sistematiklerin birleşik etkisi
│
├── false_edm_mode_scan.py      # Yanlış-EDM hızı vs Fourier modu k taraması
├── false_edm_correction_test.py# Yanlış-EDM düzeltme stratejisi doğrulaması
├── test_b_*.py                 # Spin-sürülü trim serisi: mod haritası, iteratif
│                               #   döngü, BPM etkisi, fırlatma bağımlılığı,
│                               #   rastgele desen + çift kuadratür trim
│
├── test_orbit_trim.py          # Yörünge-sürülü (BPM) trim: A/C/D/B varyantları
├── test_orbit_trim_seeds.py    # Seed evrenselliği (5 seed karşılaştırması)
├── test_orbit_mode_correlation.py # Gram matrisi, sızıntı, kazanç yasası k=7..12
├── test_radial_spin.py         # Radyal vs boylamsal başlangıç polarizasyonu
├── test_symm_vs_antisym.py     # Simetrik/antisim ayrıştırma — spin tabanı kanıtı
├── test_symm_basis_fit.py      # Genişletilmiş baz fit (başarısız ama öğretici)
├── test_new_topology.py        # integrator2 topoloji testi (kararsız çıktı)
├── find_stable_gradient.py     # İnce mercek tek-hücre kararlılık analizi
│
├── make_paper_figures.py       # Yayın kalitesi figürler + tablolar (6 fig, 3 tablo)
├── fig_1_falseedm_scan.py      # Bağımsız figür jeneratörler (1..7)
├── fig_2_svd.py
├── fig_3_amplitude_scales.py
├── fig_4_reconstruction_quality.py
├── fig_5_clean_iterations.py
├── fig_6_combined_systematics.py
│
├── README.md                   # Ana proje belgesi (56 KB, Türkçe, pedagojik)
├── YAPILACAKLAR.md             # Aktif yapılacaklar listesi
├── YÖNTEM.md                   # Detaylı yöntem açıklaması
├── FOURIER_REKONSTRUKSIYON.md  # Fourier yöntemi teorisi
├── false_edm_harmonic_sinir.md # Sahte EDM analiz günlüğü (§1–12.16, ana bulgular)
├── trim_yontemi_pedagojik.md   # Trim yöntemi sıfırdan anlatım (3 test + taban analizi)
├── makale_tr.tex               # Türkçe makale taslağı (k-mod + trim bölümleri)
└── PROJE_ANALIZI_VE_ONERILER.md
```

---

## Build: C++ İzleyici

**integrator.cpp değiştirildikten sonra her zaman yeniden derleyin.** Eski binary sessiz hata üretir: quad_dy/quad_tilt dizileri Python'dan gönderilir ama eski kütüphane bunları uygulamaz → tüm simülasyonlar hizalama hatası yokmuş gibi davranır.

```bash
bash build_integrator.sh
# Doğrulama:
python3 -c "import integrator; print('yüklendi')"
```

Derleme bayrakları: `-O3 -shared -fPIC -std=c++17`
- Linux: `g++` → `lib_integrator.so`
- macOS: `clang++` → `integrator.dylib`

**İkinci motor (`integrator2.cpp`):** Alternatif kafes topolojisi
(QF–drift–QD–drift–DEFLEKTÖR(2Φ)–drift(2d), 6 elemanlı hücre) için bağımsız
kopya. Eski kodu etkilemez; `integrator2.py` yalnız `lib_integrator2.so`
yükler. Derleme: `g++ -O3 -shared -fPIC -std=c++17 integrator2.cpp -o
lib_integrator2.so`. Dikkat: bu topoloji mevcut g₀=0.2 T/m ile kararsızdır;
aynı ton için g≈0.5 T/m gerekir (`find_stable_gradient.py`).

---

## Konfigürasyon: params.json

Tüm simülasyon parametreleri `params.json` içinde tutulur. Scriptler doğrudan `json.load()` ile yükler.

**Temel parametreler:**

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `R0` | 95.49 m | Halka yarıçapı |
| `nFODO` | 24 | FODO hücre sayısı (= 48 kuadrupol) |
| `quadLen` | 0.4 m | Kuadrupol uzunluğu |
| `g0`, `g1`, `g2` | 0.2, 0.21, 0.21 T/m | Kuadrupol gradyanları |
| `dt` | 1e-11 s | İntegrasyon adım boyutu |
| `t_pr` | 1000 | Poincaré periyot sayısı |
| `error_quad_index` | -1 | Hata verilen kuadrupol (-1: yok) |
| `bpm_noise_sigma` | 0.0 m | BPM elektronik gürültüsü |
| `bpm_offset_sigma` | 0.0 m | BPM ofset RMS'i |
| `dy_harmonics` | [...] | Dikey hizalama hata harmonikleri |
| `k_search_max` | 12 | Aramada max harmonik sayısı |
| `lasso_lambda` | 0.02 | LASSO düzenlileştirme katsayısı |

Sistematik testler `params.json`'u bellek içinde kopyalayarak tarama yapar; dosyayı değiştirmez.

---

## Mimari

```
Analiz Scriptleri (test_*.py, false_edm_*.py)
        │
Yeniden Çatım Katmanı (reconstruction.py, build_response_matrix.py)
        │
Python İzleyici Arayüzü (integrator.py + FieldParams)
        │ ctypes
C++ Fizik Motoru (integrator.cpp, GL4, Thomas-BMT)
        ▲
params.json ──→ tüm katmanlar
```

**FieldParams sınıfı** (`integrator.py`): C++ motoruna aktarılacak alanları, RF özelliklerini ve hata profillerini tutar. `integrate_particle()` fonksiyonu çağrılmadan önce ilgili alanlar set edilir.

**Tepki matrisi** (`build_response_matrix.py`): `ProcessPoolExecutor` ile paralel. Her kuadrupola birer birer küçük Δdy perturbasyonu uygulanır, kapalı yörünge ölçülür → 48×48 R matrisi oluşur. İki k-modülasyon konfigürasyonu için `R_dy_1.npy`, `R_dy_2.npy`, `R_dx_1.npy`, `R_dx_2.npy` üretilir.

---

## Geliştirme İş Akışı

### Tipik test döngüsü

```bash
# 1. İzleyiciyi derle (integrator.cpp değiştiyse)
bash build_integrator.sh

# 2. Tepki matrislerini hesapla (önce bir kez)
python3 build_response_matrix.py

# 3. Sistematik test çalıştır
python3 test_bpm_noise.py
python3 test_quad_tilt.py

# 4. Tam geri çatım testi
python3 test_kmod_reconstruction.py

# 5. Yayın figürleri üret
python3 make_paper_figures.py
```

### Yanlış-EDM analizi

```bash
python3 false_edm_mode_scan.py      # dS_y/dt vs Fourier modu k taraması
python3 false_edm_correction_test.py # Düzeltme stratejisi doğrulaması
```

### Bağımlılıklar

Explicit bir `requirements.txt` yoktur. Gerekli paketler:
- Python 3.6+ (f-string, tip ipuçları)
- `numpy`
- `matplotlib`
- `ctypes` (stdlib, ek kurulum gerektirmez)
- `concurrent.futures` (stdlib)

Kurulum: `pip install numpy matplotlib`

---

## Test Yaklaşımı

Resmi test çerçevesi (pytest vb.) yoktur. Her `test_*.py` bağımsız çalışır, konsola tablo ve metrikler basar, PNG grafik üretir.

| Dosya | Test ettiği |
|-------|------------|
| `test_bpm_noise.py` | `bpm_noise_sigma` taraması → RMS hata |
| `test_bpm_offset.py` | Ofset reddi, diferansiyel iptal doğrulaması |
| `test_quad_gradient.py` | g1/g0 perturbasyon büyüklüğü taraması |
| `test_quad_tilt.py` | Kuadrupol tilt → x-y kuplaj etkisi |
| `test_combined_systematics.py` | Tüm sistematikler aynı anda |
| `test_kmod_reconstruction.py` | Bilinen Δq deseni → TSVD/Fourier/LS kıyaslaması |
| `test_reconstruction_quality.py` | SVD gözlenebilirlik + rank limitleri |
| `test_orbit_trim.py` | BPM-sürülü trim, fit genişliği A/C/D/B varyantları |
| `test_orbit_trim_seeds.py` | Trim sonucunun seed evrenselliği (5 seed) |
| `test_symm_vs_antisym.py` | Spin tabanının kaynağı: simetrik QF/QD içeriği |
| `test_symm_basis_fit.py` | Kazanç eşiği: eşik altı mod fit'e eklenemez |
| `test_new_topology.py` | integrator2 ile alternatif hücre dizilimi |

**Yaygın metrikler:** RMS hata [μm], korelasyon katsayısı, koşul sayısı (κ), re-konstruksiyon genliği hatası, sahte EDM hızı dSy/dt [rad/s], yörünge kazancı G_k.

---

## Dil ve Belgeleme Kuralları

- **Kod ve belgeler Türkçedir.** Değişken, fonksiyon, sınıf isimleri İngilizce; yorumlar ve docstring'ler Türkçe.
- `README.md` pedagojik anlatım tarzındadır: kavramlar ilk kullanıldıklarında öğretilir.
- `.md` dosyaları bağımsız okunabilir belgelerdir; sıradan teknik not değil.
- Yeni script eklerken mevcut dosyalardaki Türkçe yorum stilini taklit edin.

---

## Önemli Tuzaklar

1. **Eski binary sessiz hata:** `integrator.cpp` değişti ama `build_integrator.sh` çalıştırılmadıysa quad hata dizileri Python'dan gönderilir fakat C++ kütüphanesi onları uygulamamış olabilir. Belirtisi: tüm testler hizalama sıfırmış gibi sonuç verir.

2. **Derlenmiş binary'ler git'te takip edilir:** `lib_integrator.so` ve `integrator.dylib` repoda bulunur (üretim kolaylığı için). Cross-platform geliştirmede platform binary'sini commit etmemeye dikkat edin.

3. **Numpy dosyaları git'te değildir:** `.gitignore` `*.npy` ve `*.npz` dosyalarını dışlar. Tepki matrisleri her ortamda `build_response_matrix.py` ile yeniden üretilmelidir.

4. **Konfigürasyon tek kaynaktır:** Parametreleri script içinde hardcode etme; `params.json`'a ekle ve `json.load()` ile oku.

5. **Paralel worker sayısı:** `build_response_matrix.py` varsayılan olarak tüm CPU çekirdeklerini kullanır. Bellek kısıtlı ortamlarda `max_workers` parametresini düşürün.

---

## Git Dalları

- `main` — kararlı dal
- `claude/claude-md-docs-spai7t` — aktif geliştirme dalı (bu oturum)

Commit mesajları Türkçe ön-ek kullanır: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`

---

## Fizik Referansı (Kod Bağlamı)

| Kavram | Kodda karşılığı |
|--------|-----------------|
| FODO hücresi | `nFODO=24` → 48 kuadrupol (`2*nFODO`) |
| Kapalı yörünge bozulması (COD) | `error_quad_dy`, `error_quad_dx` dizileri |
| K-modülasyon | `kmod_configs` listesi, `g2` parametresi |
| Thomas-BMT denklemi | `integrator.cpp` spin takip kısmı, `EDMSwitch` |
| Semplektik integrasyon | GL4 (Gauss-Legendre 4. derece) integrator |
| Fourier harmonik tabanı | `fodo_fourier_basis()` / `fodo_basis()` |
| CLEAN algoritması | `bozoki_ls.py` içinde iteratif çıkarma döngüsü |
| LASSO (ADMM) | `reconstruction.py` içinde `lasso_admm()` |
| Yörünge kazanç yasası | G_k = 24.8/\|5.03−k²\| (`test_orbit_mode_correlation.py`) |
| Fit eşiği | G_k > σ_b/σ_q → k_max² < Q_eff² + C·σ_q/σ_b |
| Antisim/simetrik ayrışım | QF/QD zıt-işaret vs aynı-işaret kombinasyonlar; 25+23 boyut (`test_symm_vs_antisym.py`) |
| Spin kuplaj katsayıları | c_k: rezonant değil, tüm k için ≠0 (`test_b_mode_map_cofalse.py`) |
