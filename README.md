# pEDM Halkasında Quad Hizalama Hatalarının Geri Çatımı

Bu depo, **proton elektrik dipol moment (pEDM)** halka tasarımında
kuadrupol mıknatısların hizalama hatalarının kapalı yörünge (COD)
ölçümlerinden geri çatımı üzerine yapılan çalışmaları içerir.

pEDM, "frozen-spin" yöntemiyle proton EDM'inin doğrudan ölçümünü
hedefler. Halkanın 24 FODO hücresinden oluşan 48 kuadrupolünün her
biri 10 μm mertebesinde hassasiyetle hizalanmalıdır; sistematik
yörünge sapmaları aksi takdirde EDM sinyalini taklit edebilir.
Bu çalışmanın amacı **BPM ölçümlerinden quad hizalama hatalarını
geri çatma** problemini, hem klasik tepki-matrisi yaklaşımıyla, hem
de düşük-mertebeli Fourier parametrelendirmesiyle incelemektir.

---

## İçindekiler

1. [Çerçeve](#çerçeve)
2. [Problemin matematiksel iskeleti](#problemin-matematiksel-iskeleti)
3. [Üç modülasyon stratejisi](#üç-modülasyon-stratejisi)
4. [Anahtar bulgu: hedefli Fourier](#anahtar-bulgu-hedefli-fourier)
5. [FODO antisimetri — fiziksel parametrelendirme](#fodo-antisimetri--fiziksel-parametrelendirme)
6. [Adaptif rekonstrüksiyon algoritması](#adaptif-rekonstrüksiyon-algoritması)
7. [Depo yapısı](#depo-yapısı)
8. [params.json — parametre referansı](#paramsjson--parametre-referansı)
9. [Hızlı başlangıç](#hızlı-başlangıç)
10. [Çalışmanın kronolojisi](#çalışmanın-kronolojisi)
11. [Bilinen tuzaklar ve uyarılar](#bilinen-tuzaklar-ve-uyarılar)
12. [Mevcut açık konular](#mevcut-açık-konular)

---

## Çerçeve

Bu yeni bir yöntem icadı değildir; klasik tepki-matrisi (orbit
response matrix, ORM) yaklaşımının pEDM halkasının özel koşullarında
**hangi sınırlarda 10 μm hassasiyetine ulaşabildiğini** sistematik
olarak inceleyen bir mühendislik değerlendirmesidir. Paralel bir
çalışmada aynı problem yapay sinir ağı yaklaşımıyla ele alınmaktadır;
iki yaklaşımı adil karşılaştırmak için bu çalışmanın ürettiği
referans çerçeveye ihtiyaç vardır.

---

## Problemin matematiksel iskeleti

48 BPM'de okunan dikey yörünge:

```
y = R · dy + b + n
```

| Sembol | Boyut | Anlam |
|--------|-------|-------|
| `y`  | 48 | BPM okumaları (m) |
| `R`  | 48×48 | tepki matrisi (Courant–Snyder formülü) |
| `dy` | 48 | quad dikey hizalama hataları (aranan) |
| `b`  | 48 | BPM elektronik ofsetleri (bilinmeyen, statik, ~300 μm) |
| `n`  | 48 | BPM gürültüsü (~1 μm) |

`b` bilinmediği için `R · dy = y − b` doğrudan çözülemez. Yörünge
sapmaları (~10 μm) BPM ofsetleri (~300 μm) yanında küçüktür → ofset
sinyali boğar.

**Çözüm: k-modülasyon.** Quad gradyenlerini iki farklı ayarda ölçüp
fark al; `b` ortak olduğundan iptal olur:

```
Δy  =  y₂ − y₁  =  (R₂ − R₁) · dy  =  ΔR · dy
```

Bu denklem `dy`'yi `b`'siz olarak verir. Ancak `ΔR`'nin koşul sayısı
hangi modülasyon stratejisinin seçildiğine kuvvetle bağlıdır.

---

## Üç modülasyon stratejisi

| Mod | Ne yapar | κ(ΔR) | Sonuç |
|-----|----------|-------|-------|
| **Uniform** | Tüm 48 quad gradyeni aynı anda ×1.02 | ~160 | 6.6 μm RMS, kor ≈ 1 |
| **Tek-quad** | Yalnız 1 quad ×1.10 | ~4.4×10⁸ | başarısız |
| **İki-quad** | İki quad (j₁, j₂) ×1.05, diğer 46 sabit | ~10⁶ | sıkı baz ile 0.02 μm |

Uniform mod mükemmel çalışır ama gerçek bir hızlandırıcıda tüm güç
kaynaklarını eşzamanlı modüle etmek pratikte zordur. Tek-quad mod
fiziksel olarak gerçekçidir ama matris tekildir. İki-quad mod orta
noktadır: pratik olarak uygulanabilir ve **doğru parametrelendirmeyle**
nano-metre hassasiyetine erişir.

---

## Anahtar bulgu: hedefli Fourier

İki-quad modülasyonunun "kötü" koşullanması aslında **doğru çözüm
uzayını yanlış parametrelendirmenin** sonucudur. Düşük-mertebeli
Fourier bazıyla yeniden parametrelendirildiğinde:

| Strateji | κ(ΔR·F) | Ölçüm hatası | Korelasyon |
|----------|---------|--------------|------------|
| Direkt çözüm (R⁻¹) | — | 107 μm | 0.03 |
| TSVD | — | 78 μm | 0.16 |
| Geniş Fourier (k=1..4) | 1.3×10⁴ | 35 μm | 0.88 |
| **Sıkı Fourier ({2,4} = gerçek harmonikler)** | **186** | **0.02 μm** | **1.000** |

Anahtar mesaj: **baz, sinyalde gerçekten var olan harmoniklere sıkıca
hizalanmalı**. Fazla harmonik koşullanmayı bozar; eksik harmonik
veriden sızıntıya yol açar. Detaylı türetim, bias–variance analizi
ve nasıl-ne-neden tartışması için [`YÖNTEM.md`](YÖNTEM.md).

---

## FODO antisimetri — fiziksel parametrelendirme

Aynı FODO hücresindeki QF (odaklayıcı, K > 0) ve QD (saçtırıcı, K < 0)
quad'larının kick formülleri:

```
kick = -K · dy_quad
```

Aynı yönde kayma (`dy_QF = dy_QD = +f`):
- QF kick = `-K·f`  (negatif)
- QD kick = `+K·f`  (pozitif)
- **Birbirini iptal eder** → COD'da neredeyse görünmez

Zıt yönde kayma (`dy_QF = +f`, `dy_QD = -f`):
- QF kick = `-K·f`  (negatif)
- QD kick = `-K·f`  (negatif — aynı!)
- **Birbirini güçlendirir** → halka boyunca smooth sinüs kick deseni

Bu yüzden bu çalışmada smooth dy/dx desenleri **FODO antisimetrik**
üretilir: 24 FODO için tanımlı bir `f(n)` fonksiyonu QF'ye `+f(n)`,
QD'ye `-f(n)` olarak uygulanır. Hem üretim hem rekonstrüksiyon
bazı bu yapıyla tutarlı tanımlanmıştır:

$$
F_k[j] = (-1)^j \cdot \{1,\; \cos(2\pi k \lfloor j/2 \rfloor / 24),\; \sin(...)\}
$$

---

## Adaptif rekonstrüksiyon algoritması

`reconstruction.py` — **greedy matching pursuit** ile dy/dx'in
Fourier harmoniklerini ve fazlarını veriden otomatik tespit eder.
Hangi harmoniklerin var olduğunu önceden bilmiyormuş gibi davranır.

### Akış

```
Girdi:  ΔR (48×48)  ve  Δy (48-boyutlu k-mod sinyali)
Aday harmonikler:  k = 0, 1, ..., k_search_max

Aktif harmonik seti A = {}
Rezidüel = ‖Δy‖
Tekrarla:
  Her k ∉ A için:
    F_test = FODO-antisim Fourier bazı (A ∪ {k})
    M_test = ΔR · F_test
    â = lstsq(M_test, Δy)
    rezidüel_k = ‖Δy − M_test · â‖
  k* = arg min rezidüel_k
  Düşüş = (önceki_rezidüel − rezidüel_k*) / önceki_rezidüel
  Eğer düşüş < threshold:  dur
  A.append(k*)
Son F ile rekonstrüksiyon, her k için (genlik, faz) raporla
```

### Çıktı formatı

- **Greedy geçmişi**: her adımda hangi k eklendi, rezidüel ne kadar düştü
- **Tespit edilen harmonikler tablosu**: `k, A_tahmin, φ_tahmin, A_gerçek, φ_gerçek, hata%`
- **Tam rekonstrüksiyon hatası**: smooth + gürültü tabanı dahil RMS ve korelasyon
- **`reconstruction_result.npz`**: sayısal sonuçlar (dy_gercek, dy_geri, seçilen k'lar)

### Paralelleştirme

Greedy döngüsü tamamen küçük matris işlemleri (~ms). Paralelleştirme
gereksiz. Asıl pahalı kısım simülasyondur (`build_response_matrix.py`),
o zaten `ProcessPoolExecutor` ile paraleldir.

---

## Depo yapısı

### Simülasyon kütüphanesi

| Dosya | İşlev |
|-------|-------|
| `integrator.cpp` | GL4 simplektik integratör (C++); quad/dipol kick'leri |
| `lib_integrator.so` / `integrator.dylib` | Derlenmiş paylaşımlı kütüphane |
| `integrator.py` | C++ kütüphanesinin Python sarmalayıcısı (ctypes) |

### Üst seviye Python kodları

| Dosya | İşlev |
|-------|-------|
| `run_simulation.py` | Tek koşum (test/görselleştirme) |
| `build_response_matrix.py` | R₁, R₂ matrislerini paralel hesaplar |
| `test_kmod_reconstruction.py` | dy/dx üret + sim koş + (Direkt, TSVD, Fourier) çöz |
| `reconstruction.py` | **Adaptif harmonik tespiti** (yeni) |
| `scan_j2.py` | j₁ sabit, j₂ taraması — en iyi quad çiftini bul |
| `show_response.py` | Tepki matrisi görselleştirme |
| `verify_quad_tilt.py` | Quad tilt etkisi doğrulama (kuplaj kontrolü) |

### Belgeler

| Dosya | İçerik |
|-------|--------|
| `README.md` | Bu dosya — depo ve çalışmanın genel haritası |
| `YÖNTEM.md` | Fourier rekonstrüksiyon yönteminin ders kitabı tarzı anlatımı |
| `metot.md` | Makale §3-5 için pedagojik metot taslağı |
| `makale-taslagi-2.md` | Makale taslağı |
| `YAPILACAKLAR.md` | Aktif iş listesi ve mevcut durum |

### Üretilen veri dosyaları

| Dosya | İçerik | Üreten |
|-------|--------|--------|
| `R_dy_1.npy`, `R_dy_2.npy` | Dikey tepki matrisleri (g_nom, g_pert) | build_response_matrix |
| `R_dx_1.npy`, `R_dx_2.npy` | Yatay tepki matrisleri | build_response_matrix |
| `kmod_reconstruction_test.npz` | Δy, Δx, gerçek dy/dx + Fourier sonuçları | test_kmod_reconstruction |
| `reconstruction_result.npz` | Adaptif geri çatım çıktıları | reconstruction |
| `scan_j2_results.npy` | j₂ taraması sonuçları (κ vs j₂) | scan_j2 |

---

## params.json — parametre referansı

### Lattice ve simülasyon

| Anahtar | Anlam |
|---------|-------|
| `R0` | Halka yarıçapı (m) |
| `nFODO` | FODO hücresi sayısı (24 → 48 quad) |
| `quadLen`, `driftLen` | Element uzunlukları (m) |
| `g0`, `g1`, `g2` | Quad gradyenleri — g0: baz, g1: kmod_quad1 için, g2: kmod_quad2 için |
| `sextK1` | Sekstupol güç (varsa) |
| `dt`, `t2`, `t_pr` | Zaman adımı, toplam süre, periyot |
| `quadSwitch`, `sextSwitch`, `EDMSwitch`, `rfSwitch` | Element açma/kapama |

### K-modülasyon konfigürasyonu

| Anahtar | Anlam |
|---------|-------|
| `kmod_quad1_index` (j₁) | Modüle edilen 1. quad (g₁ uygulanır). −1: kullanılmaz |
| `kmod_quad2_index` (j₂) | Modüle edilen 2. quad (g₂ uygulanır). −1: kullanılmaz |

**Mod seçim mantığı:**
- Her ikisi 0…47: **iki-quad** (önerilen — bu çalışmanın odağı)
- Yalnız j₁: tek-quad (matematiksel olarak başarısız)
- İkisi de −1: uniform (tüm 48 quad g₁'e ölçeklenir)

### dy/dx üretimi (smooth + gürültü)

| Anahtar | Anlam |
|---------|-------|
| `dy_harmonics` | `[{k, amp_cos, amp_sin}, ...]` — FODO seviyesi harmonik bileşenler. k=0 sabit (DC). |
| `dx_harmonics` | Aynısı yatay için |
| `dy_random_RMS` | Smooth deseninin üzerine eklenen Gaussian gürültü RMS (m) |
| `dx_random_RMS` | Aynısı yatay için |
| `dy_random_seed`, `dx_random_seed` | Gürültü tohumları |
| `smooth_antisym_fodo` | true: aynı FODO'da QF↔QD zıt işaretli (önerilen, fiziksel). false: simetrik (test amaçlı) |

**Üretim sırası** (test_kmod_reconstruction.py):
1. Eğer `dy_harmonics` veya `dx_harmonics` config'de varsa → smooth+gürültü+FODO modu (öncelikli)
2. Yoksa `--smooth` bayrağı verilmişse → hardcoded sinüzoidal mod
3. Hiçbiri yoksa → varsayılan rastgele uniform mod

### Adaptif rekonstrüksiyon

| Anahtar | Anlam |
|---------|-------|
| `k_search_max` | Greedy aramanın gideceği maksimum FODO harmoniği (default 12 = Nyquist) |
| `greedy_residual_threshold` | Her adımda rezidüelin en az bu kadar oransal düşmesi şart (default 0.02 = %2) |

### Hata kaynakları (modelde yok, fiziksel olarak eklenebilir)

| Anahtar | Anlam |
|---------|-------|
| `dipole_random_tilt_max`, `dipole_random_seed` | Dipol tilt rastgele dağılımı |
| `quad_random_tilt_max`, `quad_random_tilt_seed` | Quad tilt (skew bileşeni → x-y kuplajı) |
| `bpm_noise_sigma`, `bpm_offset_sigma`, `bpm_offset_seed` | BPM gürültü ve ofsetleri |

---

## Hızlı başlangıç

### 1. Kütüphaneyi derle (gerekirse)

```bash
# Linux
g++ -O2 -shared -fPIC -o lib_integrator.so integrator.cpp -std=c++17
# macOS
clang++ -O2 -shared -fPIC -o integrator.dylib integrator.cpp -std=c++17
```

### 2. Tepki matrislerini hesapla (paralel, ~1 saat)

```bash
python3 build_response_matrix.py
```

Üretir: `R_dy_1.npy`, `R_dy_2.npy`, `R_dx_1.npy`, `R_dx_2.npy`.

### 3. K-mod simülasyonu + baseline çözümler

```bash
# params.json'daki dy_harmonics/dx_harmonics ile çalışır (config-driven)
python3 test_kmod_reconstruction.py
```

Çıktı: smooth+gürültü dy/dx üretimi → 2 konfigürasyon koşumu →
Direkt çözüm + TSVD + Fourier (N=1..5) + hedefli harmonik
karşılaştırmaları → `kmod_reconstruction_test.npz` kaydı.

### 4. Adaptif harmonik tespiti

```bash
python3 reconstruction.py
```

Çıktı: greedy arama geçmişi → tespit edilen k, genlik, faz değerleri →
gerçek harmoniklerle karşılaştırma → `reconstruction_result.npz` kaydı.

### 5. (İsteğe bağlı) En iyi j₂'yi tara

```bash
python3 scan_j2.py --step 4      # kaba (12 j₂, ~75 dk)
python3 scan_j2.py               # tam (47 j₂, ~5 saat)
```

---

## Çalışmanın kronolojisi

1. **Uniform k-mod testi** — κ ≈ 160, mükemmel; pratik uygulanabilirliği sorgulanır.
2. **Tek-quad k-mod denemesi** — κ ≈ 4×10⁸, başarısız.
3. **İki-quad k-mod** — κ ≈ 10⁶, %50 rekonstrüksiyon.
4. **Optimum j₂ taraması** — `scan_j2.py`, en iyi quad çiftini ara.
5. **TSVD regülarizasyonu** — 4/48 mod tutulur, korelasyon 0.16.
6. **Fourier parametrelendirmesi** — düşük-N bazı dene, bias-variance gerilimi gözlemlendi.
7. **Hedefli Fourier (kritik bulgu)** — bazda yalnızca veride gerçekten var olan harmonikler → 0.02 μm hata, kor=1.000.
8. **FODO antisimetri** — QF/QD zıt işaret konvansiyonu, smooth sinüs kick deseni.
9. **Adaptif algoritma (`reconstruction.py`)** — greedy matching pursuit, harmonikleri ve fazları veriden otomatik bul.

Detaylar [`YÖNTEM.md`](YÖNTEM.md) ve [`metot.md`](metot.md)'de.

---

## Bilinen tuzaklar ve uyarılar

### `kmod_quad1_index = 0` kullanma

`integrator.cpp:541`'de `current_fodo == 0` için özel "QUAD_F_MOD"
tipi (4) tetiklenir ve `quad_dG` (modülasyon vektörü) yok sayılır.
j₁ ≥ 1 olmalı.

### lib_integrator.so / integrator.dylib güncelliği

`quad_dG` desteği commit `2e44a72` ile eklendi. Eski derlenmiş
kütüphane bu argümanı yok sayar → R₁ ≡ R₂ ve ΔR = 0 olur.
Şüphe varsa **yeniden derle**.

### FODO antisimetri vs simetrik üretim

`smooth_antisym_fodo: false` ile simetrik desen denenirse COD sinyali
~iptal olur (QF/QD kick'leri birbirini söndürür). Adaptif algoritma
zayıf sinyal nedeniyle yanlış harmonik tespit edebilir. Bu mod
"zor durum testi" olarak kalsın.

### Greedy threshold ayarı

`greedy_residual_threshold` çok küçükse spurious harmonikler eklenir
(over-fitting). Çok büyükse zayıf ama gerçek harmonikler kaçırılır.
Tipik aralık: 0.01 (sıkı) – 0.05 (rahat). Varsayılan 0.02.

### k_search_max sınırı

24 FODO için Nyquist = 12. Üzeri anlamsız (aliasing).

### dy / dx kuplajı

Skew quadrupol bileşeni yoksa dy ve dx bağımsız. `quad_random_tilt_max > 0`
ise kuplaj var, rekonstrüksiyon dikkatli yapılmalı.

---

## Mevcut açık konular

- BPM ofset/gürültü artırılırken adaptif yöntemin dayanıklılığı
- `greedy_residual_threshold` için otomatik kriter (BIC, F-test)
- Gerçek yer hareketi (örn. SEISM/Aurora verisi) ile testler
- Çoklu (j₁, j₂) ölçüm birleştirmesi — rank artırma
- Adaptif algoritmanın hata barları (bootstrap)

Aktif iş listesi: [`YAPILACAKLAR.md`](YAPILACAKLAR.md).
