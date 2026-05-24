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

`reconstruction.py` üç farklı stratejiyi uygular ve karşılaştırır.

---

### 1. Hedefli fit (targeted)

**Fikir:** Hangi Fourier harmoniklerinin veride var olduğunu önceden biliyormuşsunuz
gibi davranın — yalnızca o harmonikleri içeren FODO-antisimetrik baz kullanarak
doğrudan en küçük kareler çözümü yapın.

```
F  ← fodo_fourier_basis(n_q, k_list=[0, 2, ...], antisym=True)
M  ← ΔR · F                   # 48 × |k_list × 2 - 1|
â  ← lstsq(M, Δy)
dy_geri ← F · â
```

**Neden işe yarar:** ΔR rank ~2 olsa bile *M = ΔR · F* matrisi
baz boyutu kadar sütun içerdiğinden baz boyutu ≤ 2 koşulunda
belirli/aşırı belirlenmiş sistem elde edilir.

**Ne zaman çalışır, ne zaman çalışmaz:**

| Durum | Sonuç |
|-------|-------|
| Baz tam doğru (gerçek harmonikler) | 0.02 μm hata, kor = 1.000 |
| Bazda fazla harmonik var | κ büyür, gürültü büyütme artar |
| Bazda eksik harmonik var | sızıntı hatası — var olan harmoniklerin katsayısı kirlenir |
| Baz boyutu > rank(M) | minimum-norm çözüm; **katsayılar tek tek yorumlanamaz**, ama *profil* anlamlı olabilir |

**Pratik kullanım:** Harmonikler fizikten biliniyorsa (düşük k, toprak hareketi
baskın) bu mod optimaldır. Bilinmiyorsa greedy veya çok-konfig yaklaşım gerekir.

---

### 2. Greedy matching pursuit (kör)

**Fikir:** Hangi harmoniklerin veride var olduğunu bilmiyormuşsunuz gibi davranın.
Her adımda rezidüeli en çok düşüren k değerini seçin; seçim kriteri sağlanmaz
veya harmonik sayısı sınıra ulaşırsa durun.

```
A ← {}
Rezidüel ← ‖Δy‖
Tekrarla:
  Her k ∉ A için rezidüel_k ← ‖Δy − ΔR·F_{A∪{k}}·â‖
  k* ← argmin rezidüel_k
  Eğer (Rezidüel − Rezidüel_k*) / Rezidüel < threshold: dur
  A ← A ∪ {k*}
```

**Neden rank-2 ΔR ile başarısız olur:**

ΔR'nin SVD ayrışımına göre, $\Delta R = \sigma_1 u_1 v_1^T + \sigma_2 u_2 v_2^T +
\text{(küçük modlar)}$ yazılabilir. Greedy algoritması her adımda yalnızca
bu iki büyük tekil vektör yönünde "görebilir". FODO-antisimetrik $k$'lar
$u_1$, $u_2$ ile hizalı olmayabilir; bunun yerine greedy $v_1$, $v_2$'ye
en yakın proyeksiyonu olan *yanlış* harmonikleri seçer. Buna
**tekil vektör ötüşmesi (aliasing)** denir.

Örnek (QD+QD konfigürasyonu, j₁=3, j₂=9):

| Gerçek harmonikler | Greedy seçimleri |
|--------------------|-----------------|
| k = 0, 2, 4 | k = 1, 8, 7 |

Greedy, rezidüeli düşürür ama fiziksel olmayan harmonikler seçer.

**Rank ≥ 3 olunca:** Farklı quad çiftlerini veya tek-quad ölçümlerini
yığdığınızda (bkz. çok-konfig bölümü) rank artar ve greedy daha güvenilir
hale gelir.

---

### 3. LASSO (kör, seyrek)

**Fikir:** Tüm k = 0, 1, ..., k_max harmoniklerini adaya koyun; L1 ceza
terimi gereksiz katsayıları sıfıra çeksin:

$$
\hat{a} = \arg\min_a \tfrac{1}{2}\|M a - \Delta y\|^2 + \lambda \|a\|_1
$$

Gerçek harmonikler büyük katsayıya, sahte harmonikler küçük katsayıya sahip
olduğundan L1 ceza seyrek çözüm üretecektir.

**Neden rank-2 sistemde çalışmaz:**

M = ΔR · F, rank ~2 ve 25+ sütunlu (k_max=12 için). Bu durumda
çözüm uzayı 23 boyutlu null uzayı içerir; M^T M + ρI sistemi
normalize uzayda her katsayıyı ~1/25'e bölen 25 eş büyüklükte özdeğere
sahiptir. Sonuç: her katsayı genliği ≈ |sinyal|/25 ≈ 10 μm/25 ≈ 0.4 μm,
λ eşiği ise 0.02 (normalize) düzeyinde. Tüm katsayılar eşiğin altına
düşer → **LASSO tümü sıfıra iter**.

Hiçbir λ seçimi bu durumu kurtaramaz: λ küçütülürse ayrımcılık kaybedilir,
büyütülürse sinyalin kendisi silinir.

**Sonuç:** LASSO, "az ölçüm – çok parametre" senaryosunda (sıkıştırmalı
algılama) işe yarar. "Az rank – az parametre" senaryosunda (ΔR rank ~2,
baz boyutu ~3) temel kavramsal uyumsuzluk vardır; LASSO'nun seçiciliği
yitirilir.

---

### 4. Çok-konfigürasyon yığma — rank barikatını aşmak

**Temel problem:**
- Her kmod ölçümü (tek quad) → ΔR rank ~1 → 1 bağımsız denklem
- k=0 ve k=2 için **3 bilinmeyen** (DC, cos₂, sin₂)
- 1 veya 2 kmod ölçümü yeterli değil

**Çözüm — üç ayrı quad ölçümü:**

```
ΔR_c0 @ dy = Δy_c0    (yalnızca j=3 modüle)   → rank ~1
ΔR_c1 @ dy = Δy_c1    (yalnızca j=9 modüle)   → rank ~1
ΔR_c2 @ dy = Δy_c2    (yalnızca j=1 modüle)   → rank ~1

Yığılmış:
[ΔR_c0]         [Δy_c0]
[ΔR_c1]  @ dy = [Δy_c1]    → toplam rank ~3
[ΔR_c2]         [Δy_c2]
```

Yığılmış sistemde M = [ΔR_c0; ΔR_c1; ΔR_c2] @ F boyutu 144×3, rank ~3.
3 bilinmeyen, 3 bağımsız denklem → belirli sistem. Gürültü fazlasıyla
sönümlenir (aşırı belirlenmiş).

**Neden j=1, j=3, j=9 seçildi?**
- Hepsi QD tipi → `integrator.cpp`'nin QUAD_F_MOD bug'ından (j=0 özel durumu) etkilenmez
- j=3 → FODO hücresi 1, j=9 → FODO hücresi 4, j=1 → FODO hücresi 0
- k=2 moduna göre: cos(2π×2×1/24) ≈ 0.87, cos(2π×2×4/24) = -0.5, cos(2π×2×0/24) = 1.0
  → üç farklı projeksiyon → good rank

---

### Algoritma karşılaştırması

| Yöntem | Gereken bilgi | Rank-2 ile | Rank ≥ 3 ile | Katsayı yorumu |
|--------|--------------|------------|--------------|----------------|
| Hedefli fit | Hangi k | ✓ (sınırlı) | ✓ | baz boyutu ≤ rank ise evet |
| Greedy | Hiçbiri | ✗ (yanlış k) | ✓ | her zaman |
| LASSO | Hiçbiri | ✗ (hepsi sıfır) | ∼ | λ ayarına bağlı |
| Çok-konfig yığma | Hangi k | — | ✓ | evet |

---

### Akış özeti

```
Girdi:  ΔR (48×48)  ve  Δy (48-boyutlu k-mod sinyali)

  ┌─ 1. Hedefli fit ──────────────────────────────────────────┐
  │  k_list = params.json'dan                                  │
  │  M = ΔR · F(k_list)                                       │
  │  â = lstsq(M, Δy)  →  dy_geri = F · â                    │
  └───────────────────────────────────────────────────────────┘

  ┌─ 2. Greedy ───────────────────────────────────────────────┐
  │  A = {}; Rezidüel = ‖Δy‖                                  │
  │  Her adım: k* = argmin ‖Δy − ΔR·F_{A∪k}·â‖              │
  │  Dur: düşüş < threshold veya len(A) = max_harmonics        │
  └───────────────────────────────────────────────────────────┘

  ┌─ 3. LASSO (ADMM) ─────────────────────────────────────────┐
  │  M_full = ΔR · F(k=0..k_max); sütun normalize             │
  │  min ½‖M a − Δy‖² + λ‖a‖₁  →  seyrek â                  │
  └───────────────────────────────────────────────────────────┘

  ┌─ 4. Çok-konfig (yığma, mevcutsa) ────────────────────────┐
  │  Her c: ΔR_c = R₂_c − R₁_c                               │
  │  M_stack = [ΔR_c0; ΔR_c1; ΔR_c2] · F(k=[0,2])          │
  │  â = lstsq(M_stack, [Δy_c0; Δy_c1; Δy_c2])             │
  └───────────────────────────────────────────────────────────┘
```

Detaylı matematiksel türetim, bias–variance analizi ve rank-2 ile
neden LASSO/greedy'nin çöktüğünün tam kanıtı için [`YÖNTEM.md`](YÖNTEM.md).

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

**Tek-konfig (eski arayüz):**

| Dosya | İçerik | Üreten |
|-------|--------|--------|
| `R_dy_1.npy`, `R_dy_2.npy` | Dikey tepki matrisleri (g_nom, g_pert) | build_response_matrix |
| `R_dx_1.npy`, `R_dx_2.npy` | Yatay tepki matrisleri | build_response_matrix |
| `dR_dy.npy`, `dR_dx.npy` | Fark matrisleri ΔR = R₂ − R₁ | build_response_matrix |
| `kmod_reconstruction_test.npz` | Δy, Δx, gerçek dy/dx + Fourier sonuçları | test_kmod_reconstruction |
| `reconstruction_result.npz` | Tek-konfig geri çatım çıktıları | reconstruction |

**Çok-konfig (`--config N` ile):**

| Dosya | İçerik | Üreten |
|-------|--------|--------|
| `R_dy_1_cN.npy`, `R_dy_2_cN.npy` | N. konfig tepki matrisleri | build_response_matrix --config N |
| `R_dx_1_cN.npy`, `R_dx_2_cN.npy` | N. konfig yatay matrisleri | build_response_matrix --config N |
| `dR_dy_cN.npy`, `dR_dx_cN.npy` | N. konfig ΔR matrisleri | build_response_matrix --config N |
| `kmod_test_cN.npz` | N. konfig Δy, Δx, gerçek dy/dx | test_kmod_reconstruction --config N |
| `reconstruction_multi_result.npz` | Yığılmış çok-konfig geri çatım | reconstruction |
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

**Tekil konfig (eski):**

| Anahtar | Anlam |
|---------|-------|
| `kmod_quad1_index` (j₁) | Modüle edilen 1. quad (g₁ uygulanır). −1: kullanılmaz |
| `kmod_quad2_index` (j₂) | Modüle edilen 2. quad (g₂ uygulanır). −1: kullanılmaz |

**Çok-konfig (yeni, `--config N` ile):**

| Anahtar | Anlam |
|---------|-------|
| `kmod_configs` | `[{"j1": 3, "j2": -1}, {"j1": 9, "j2": -1}, {"j1": 1, "j2": -1}, ...]` listesi. `--config N` ile N. eleman seçilir. |

**Mod seçim mantığı (her konfig için):**
- `j2 ≥ 0`: **iki-quad** (j₁ ve j₂ aynı anda modüle)
- `j2 = −1`: **tek-quad** (yalnız j₁ modüle)
- `j1 = j2 = −1`: uniform (tüm 48 quad g₁'e ölçeklenir)

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

### 2a. Tek-konfig (eski arayüz, hızlı başlangıç)

```bash
# Tepki matrisleri (params.json'daki kmod_quad1/2_index kullanılır)
python3 build_response_matrix.py

# K-mod simülasyonu + Direkt, TSVD, Fourier, Hedefli karşılaştırma
python3 test_kmod_reconstruction.py

# Hedefli, Greedy ve LASSO geri çatımı
python3 reconstruction.py
```

### 2b. Çok-konfig (k=0 + k=2 yığılmış fit)

```bash
# Her konfig için tepki matrisi hesapla (params.json'daki kmod_configs listesi)
for n in 0 1 2; do python3 build_response_matrix.py --config $n; done

# Her konfig için ölçüm simülasyonu
for n in 0 1 2; do python3 test_kmod_reconstruction.py --config $n; done

# Yığılmış geri çatım — R_dy_1_c0.npy varlığı otomatik çok-konfig modunu tetikler
python3 reconstruction.py
```

**Beklenti:** Yığılmış M matrisi (3 konfig × 48 BPM) × 3 Fourier sütunu,
rank ~3 → k=0 + k=2 (3 bilinmeyen) tam belirlenmiş.

### 3. (İsteğe bağlı) En iyi j₂'yi tara

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
