# 6D Proton EDM Depolama Halkası Simülatörü

**Yazar:** Selcuk H.

Bu proje, Proton Elektrik Dipol Momenti (EDM) deneyleri için tasarlanmış tam 6 boyutlu bir depolama halkası simülasyonudur. Parçacık dinamiği ve spin presesyonu C++ ile yüksek hassasiyetle çözülür; Twiss analizi, yanıt matrisi hesabı ve quad hizalama geri çatımı Python katmanında yapılır.

---

## İçindekiler

1. [Fiziksel Arkaplan](#1-fiziksel-arkaplan)
2. [Halka Geometrisi: FODO Örgüsü](#2-halka-geometrisi-fodo-örgüsü)
3. [Koordinat Sistemi](#3-koordinat-sistemi)
4. [C++ Entegratör: `integrator.cpp`](#4-c-entegratör-integratorcpp)
5. [Python Köprüsü: `integrator.py`](#5-python-köprüsü-integratorpy)
6. [Simülasyon Orkestrasyonu: `run_simulation.py`](#6-simülasyon-orkestrasyonu-run_simulationpy)
7. [Görselleştirme: `plot_results.py`](#7-görselleştirme-plot_resultspy)
8. [Analitik Twiss Kütüphanesi: `fodo_lattice.py`](#8-analitik-twiss-kütüphanesi-fodo_latticepy)
9. [DFT/FFT Tabanlı Quad Geri Çatım: `spectral_inversion.py`](#9-dftfft-tabanlı-quad-geri-çatım-spectral_inversionpy)
10. [Quad Tilt: Skew-Quadrupol ve x-y Kuplajı](#10-quad-tilt-skew-quadrupol-ve-x-y-kuplajı)
11. [Parametreler: `params.json`](#11-parametreler-paramsjson)
12. [Kurulum ve Çalıştırma](#12-kurulum-ve-çalıştırma)

---

## 1. Fiziksel Arkaplan

### Neden bu simülasyon?

Proton EDM deneyi, protonun elektrik dipol momentini ölçerek CP-simetri ihlalini aramayı hedefler. Deney, sihirli momentumda ($p \approx 0.701\ \text{GeV/c}$) dolaşan protonların spinini radyal elektrik alanla "dondurarak" küçük bir EDM sinyali arar.

Bunu yapabilmek için halkadaki her türlü hizalama hatası (quad kaçıklıkları, deflektör açısal sapmaları) hassas biçimde ölçülmeli ve düzeltilmelidir. Bu simülatör iki temel soruyu yanıtlar:

1. **İleri problem:** Verilen bir hata kümesi için kapalı yörünge sapması (COD) ne kadardır?
2. **Ters problem:** BPM ölçümlerinden quad kaçıklıklarını geri çatabilir miyiz?

### Sihirli Momentum

Proton EDM deneyinin can alıcı koşulu:

$$p_{\text{magic}} = \frac{m_p c}{\sqrt{G_p}} \approx 0.7007\ \text{GeV/c}$$

Bu momentumda, elektrik alandan kaynaklanan spin presesyonu tam olarak sıfırlanır (Thomas terimi ile Larmor terimi birbirini götürür). Böylece spin, radyal yönde donmuş kalır ve yalnızca EDM varlığında dikey bileşen kazanır.

---

## 2. Halka Geometrisi: FODO Örgüsü

Halka, 24 özdeş **FODO hücresi**nden oluşur. Her hücre 8 elemandan ibarettir ve sırayla şöyle ilerler:

```
QF → DRIFT → ARC → DRIFT → QD → DRIFT → ARC → DRIFT
elem=0  =1    =2    =3    =4    =5    =6    =7
```

| Eleman | Tipi | Görevi |
|--------|------|--------|
| QF | Odaklayan quadrupol (g₁ > 0) | Yatay düzlemde odaklar, dikey dağıtır |
| QD | Ayrıştıran quadrupol (−g₁) | Dikey düzlemde odaklar, yatay dağıtır |
| ARC | Silindirik kapasitör (elektrik yay, n=1) | Parçacığı büküp halka boyunca taşır |
| DRIFT | Serbest yol | Saha yok, parçacık düz ilerler |

24 hücre × 2 quad/hücre = **48 quadrupol**, dolayısıyla yanıt matrisi 48×48 boyutundadır.

### Betatron Tune

FODO örgüsündeki odaklama gücü, parçacığın halkayı her dolaşımında kaç salınım yaptığını belirler. Temiz simülasyondan (sıfır hizalama hatası, başlangıç yalnız açısal kick) elde edilen değerler:

$$Q_x = 2.6824 \qquad Q_y = 2.3621 \quad (g_1 = 0.21\ \text{T/m, params.json varsayılanı})$$

### Arc Odaklaması: Yatay ve Dikey Farklıdır

n=1 silindirik kapasitörde Maxwell denklemleri $E_z = 0$ gerektirir. Bu nedenle:

- **Dikey düzlem:** `K_y_arc = 0` — arc elemanı saf drift gibi davranır. Dikey tune ($Q_y \approx 2.36$) yalnızca quad odaklamasından gelir.
- **Yatay düzlem:** Merkezkaç kuvveti + relativistik Coriolis kuplajı sıfır olmayan bir yatay odaklama yaratır. Basit analitik formül bu etkiyi tam vermez; `K_x_arc` değeri bisection ile simülasyon referansına ($Q_x = 2.6824$) kalibre edilir.

---

## 3. Koordinat Sistemi

Simülatör **global Kartezyen** koordinat kullanır:

- **X**: Halka düzleminde radyal yön (halka merkezinden dışa doğru)
- **Y**: Halka düzleminde azimutal yön (parçacık bu yönde hareket eder)
- **Z**: Dikey yön

İdeal yörünge, X-Y düzleminde `R₀ = 95.49 m` yarıçaplı bir çemberdir. Parçacık `(X = R₀, Y = 0, Z = 0)` noktasından başlar ve `−Y` yönünde (saat yönünde, `direction = −1`) hareket eder.

### Yerel ↔ Global Dönüşüm (`integrator.py`)

Her yay elemanından sonra `rotate_all()` C++ fonksiyonu koordinat çerçevesini `−Φ_def` kadar döndürür. Bu sayede parçacık her eleman girişinde daima `X ≈ R₀, Y ≈ 0` konumundan başlıyor gibi görünür. Python katmanı ise analiz için bu global koordinatları yerel sapmaya `(x = X − R₀, y = Z)` çevirir:

```python
# integrator.py — convert_global_to_local_matrix
history_local[:, 0] = X_global - R0   # radyal sapma [m]
history_local[:, 1] = Z_global        # dikey konum [m]
history_local[:, 2] = Y_global        # boylamsal konum ≈ yay uzunluğu [m]
```

---

## 4. C++ Entegratör: `integrator.cpp`

### GL4 Simplektik Entegratör

Hareket denklemleri (Newton + Thomas-BMT) **Gauss–Legendre 4. derece örtük Runge–Kutta** yöntemiyle çözülür. GL4'ün tercih edilmesinin iki nedeni vardır:

1. **Simplektiklik**: Faz uzayı hacmini (Liouville teoremi) ve enerjiyi uzun vadede korur. Bu, depolama halkaları için kritiktir; standart RK4 küçük de olsa enerji kayması biriktirir.
2. **4. Derece Hassasiyet**: Adım başına hata $\mathcal{O}(h^5)$ mertebesindedir.

### Elektromanyetik Alanlar: `get_electromagnetic_fields()`

Her eleman tipinde farklı alanlar tanımlıdır:

**Yay (ARC, tip 0):** Silindirik kapasitör. Radyal elektrik alan:

$$E_r(R,Z) = E_0 \left(\frac{R_0}{R}\right)^n \left[1 - \frac{n^2-1}{2}\left(\frac{Z}{R}\right)^2 + \ldots\right]$$

**Quadrupol (QF/QD, tip 2/3):** Kaçıklık bileşenleri dahil saf quadrupol alanı:

$$B_r = G_1\,(Z - d_y) \qquad B_Z = G_1\,(X - R_0 - d_x)$$

Burada $d_y$ dikey, $d_x$ radyal quad kaçıklığıdır:
- **$d_y \neq 0$** → dikey kuvvet → **dikey (y) yörünge** değişir
- **$d_x \neq 0$** → radyal kuvvet → **radyal (x) yörünge** değişir

İki düzlem bu sayede temel düzeyde birbirinden bağımsızdır (quad tilt olmadığı sürece).

### Kapalı Yörünge Verisi: `cod_data.txt`

Her FODO hücresinin eleman sınırında parçacığın konumu arabelleklere alınır. Her devir sonunda bu değerler birikimli toplama eklenir. Simülasyon bitince toplam tur sayısına bölünerek **tur ortalamalı COD** dosyaya yazılır. Betatron salınımları sıfır-ortalıklı olduğundan ortalama işlemi onları yok eder; geriye yalnızca **kapalı yörünge sapması** kalır.

Dosya formatı: her satır `[s_m, x_mm, y_mm]`, toplam `nFODO × 8 + 1 = 193` satır (boundary kapanışı dahil).

### Thomas-BMT Spin Dinamiği

Spin vektörü $(S_x, S_y, S_z)$ Thomas-BMT denklemiyle evrilir:

$$\frac{d\mathbf{S}}{dt} = \boldsymbol{\Omega} \times \mathbf{S}$$

$\boldsymbol{\Omega}$, elektrik ve manyetik alana, hıza ve anomal manyetik moment $G_p = 1.793$'e bağlıdır. Sihirli momentumda yalnızca EDM katkısı kalır ve $S_y$ bileşeni yavaşça büyür.

---

## 5. Python Köprüsü: `integrator.py`

### `FieldParams` Sınıfı

C++'a iletilecek tüm fizik parametrelerini tutar. `to_c_array()` metodu bunları sıralı bir `ctypes.c_double` dizisine çevirir. Dizinin indeksleri C++ tarafındaki `field_params[]` sıralamasıyla birebir eşleşir; bu nedenle yeni parametre eklenirken her iki taraf birlikte güncellenmeli.

### `integrate_particle()` Fonksiyonu

Ana çağrı noktasıdır. Şunları yapar:

1. **Yerel → Global dönüşüm:** `(x_dev, y_vert, z_long, ...)` → `(X_G, Y_G, Z_G, ...)`
2. **C++ çağrısı:** `_lib.run_integration(...)` ile simülasyon çalıştırılır
3. **Global → Yerel dönüşüm:** `convert_global_to_local_matrix()` ile sonuçlar analiz koordinatlarına döndürülür
4. **Poincaré verisi:** C++ 200 000'e kadar nokta saklayabilir; Python bunları numpy dizisine çevirir

Önemli diziler:

| Parametre | Boyut | Açıklama |
|-----------|-------|----------|
| `quad_dy` | (2×nFODO,) | Her quad dikey kaçıklığı [m] |
| `quad_dx` | (2×nFODO,) | Her quad radyal kaçıklığı [m] |
| `quad_tilt` | (2×nFODO,) | Her quad eğim açısı [rad] |
| `dipole_tilt` | (2×nFODO,) | Her deflektör eğim açısı [rad] |
| `quad_dG` | (2×nFODO,) | Her quad gradyan sapması [T/m] |

---

## 6. Simülasyon Orkestrasyonu: `run_simulation.py`

### Sihirli Momentumun Hesabı

```python
p_magic = M2 / sqrt(AMU)          # GeV/c cinsinden, M2=0.938272, AMU=1.792847
beta0   = p_magic / sqrt(p²+M²)   # göreli hız
E0_V_m  = -(p_magic * beta0 / R0) * 1e9  # gerekli radyal elektrik alan [V/m]
```

Elektrik alan miktarı, sihirli momentumdaki protonu `R₀` yarıçaplı dairesel yörüngede tutacak biçimde otomatik hesaplanır.

### Başlangıç Koşulları

`params.json`'dan okunan `dev0` (radyal sapma), `y0` (dikey sapma), `theta0_hor/ver` (açısal sapma) ile başlangıç faz uzayı noktası oluşturulur.

> **Not:** Tune ölçümü için `dev0 = 1e-5 m` tepme değeri atanır. Ancak hizalama hataları (100 μm) büyük dikey kapalı yörünge yarattığında `arctan2` tabanlı tune tahmini güvenilmez sonuç üretir. Doğru tune değerleri `fodo_lattice.py` ile analitik olarak elde edilir (bkz. Bölüm 8).

### Hata Dizileri

Her quad için `quad_dy`, `quad_dx` ve her deflektör için `dipole_tilt` dizileri oluşturulur. `params.json`'dan tek bir elemana veya tüm halkalara rastgele hata verilmesi desteklenir.

---

## 7. Görselleştirme: `plot_results.py`

3×4'lük ana panel oluşturur:

| Sütun | Satır 1 (radyal x) | Satır 2 (dikey y) | Satır 3 (spin) |
|-------|-------|-------|-------|
| 1 | x(t) | y(t) | Sₓ(t) |
| 2 | COD x(s) | COD y(s) | S_y(t) |
| 3 | x–x' faz uzayı | y–y' faz uzayı | S_z(t) |
| 4 | x FFT | y FFT | — |

### `_load_cod()`

`cod_data.txt` dosyasını okur; her FODO hücresi 8 eleman içerdiğinden toplam `nFODO × 8 = 192` satır beklenir.

---

## 8. Analitik Twiss Kütüphanesi: `fodo_lattice.py`

Bu modül simülasyon kütüphanesine (integrator) **hiçbir bağımlılık** taşımaz; tamamen analitik transfer matrisleri kullanır. `spectral_inversion.py` tarafından içe aktarılır.

### Fiziksel Sabitler ve Manyetik Rijitlik

```python
magic_momentum_proton(mom_error=0.0)  # → p_magic [GeV/c]
compute_Brho(p_GeV_c)                 # → Brho = p / (q·c) [T·m]
```

Kuadrupol odaklama gücü: `K_abs = g₁ / Brho` [m⁻²].

### Transfer Matrisleri (2×2, tek düzlem)

```python
drift_matrix(L)                         # serbest drift
thick_quad_matrix(K, L, focusing=True)  # kalın-lens kuadrupol (cos/sin veya cosh/sinh)
arc_matrix(L, K)                        # K>0 odaklayan, K<0 dağıtan, K=0 drift
```

Yatay arc için `K = K_x_arc > 0`; dikey arc için `K = 0` (saf drift).

### K_x_arc Kalibrasyonu

Yatay arc odaklaması analitik olarak bilinemez; bir kez bisection ile kalibre edilir:

```python
K_x_arc = calibrate_K_x_arc(config, Q_x_target=2.6824)
```

Referans değer `QX_REF_CLEAN_SIM = 2.6824`, ideal koşullarda (sıfır hizalama hatası, başlangıç yalnız açısal kick) simülasyondan doğrudan ölçülmüştür.

### Twiss Parametreleri

```python
beta, phi, Q = compute_twiss_at_quads(config, plane, K_x_arc=None, Q_x_target=None)
```

**Döngü mantığı:** Her FODO hücresinin QF ve QD giriş noktalarında (toplam $N = 2 \times \text{nFODO}$ konum):

1. Periyodik çözüm: $M_\text{cell}$'in iz (trace) değerinden $\mu = \arccos(\text{tr}/2)$ ve $\beta_0 = M_{01}/\sin\mu$
2. Her elemanda transfer matrisiyle $\begin{pmatrix}\beta(s)\\\alpha(s)\end{pmatrix}$ takibi
3. Kümülatif faz $\phi_i = \int_0^{s_i} ds/\beta(s)$
4. Tune: $Q = N_\text{hücre} \cdot \mu / (2\pi)$

Düzlem kuralları:

| Düzlem | QF rolü | QD rolü | Arc |
|--------|---------|---------|-----|
| `x` (yatay) | odaklayan | dağıtan | `K_x_arc` |
| `y` (dikey) | dağıtan | odaklayan | `K = 0` (drift) |

### İşaretli KL Dizisi

```python
KL = signed_KL(config, plane)  # shape (N,)
```

Her quad için integre güç, işaretiyle birlikte:

| Düzlem | QF | QD |
|--------|----|----|
| `x` | `+K·L` | `−K·L` |
| `y` | `−K·L` | `+K·L` |

Bu işaret kuralı Courant-Snyder yanıt matrisi formülüyle tutarlıdır.

### Yanıt Matrisi

```python
R = build_response_matrix(beta, phi, Q, KL)  # → N×N ndarray
```

Courant-Snyder formülü:

$$R_{ij} = \frac{\sqrt{\beta_i \cdot \beta_j}}{2\sin(\pi Q)} \cdot \cos\!\bigl(|\phi_i - \phi_j| - \pi Q\bigr) \cdot KL_j$$

**Önemli:** Formülde öncü eksi işareti yoktur. `KL_j` işaretini zaten taşır; ekstra eksi koymak yanlış işaret getirir (geçmişte bu hata corr = −0.999 sonucuna yol açmış ve düzeltilmiştir).

### FFT Geri Dönüşümü (Sirkülant Yaklaşım)

```python
dq_hat = fft_invert(y, beta, phi, Q, KL)
```

Beş adımlı algoritma:

1. $\tilde{y}_i = y_i / \sqrt{\beta_i}$ — beta normalizasyonu
2. $\lambda_k = \text{FFT}(m_k)$,  $m_k = \cos(|\phi_k-\phi_0|-\pi Q)/(2\sin\pi Q)$ — sirkülant özdeğerleri
3. $\tilde{U}_k = \text{FFT}(\tilde{y})_k / \lambda_k$ — Fourier uzayında bölme
4. $u = \text{IFFT}(\tilde{U})$
5. $\Delta q_j = u_j / (\sqrt{\beta_j} \cdot KL_j)$

FODO örgüsü ideal sirkülant değil (QF ve QD farklı $\beta$) → bu algoritma ~5 μm'lik blok-sirkülant yaklaşım hatası bırakır. Kesin çözüm için `direct_invert` kullanılır.

### Kesin Çözücü

```python
dq_hat = direct_invert(R, y)  # np.linalg.solve(R, y)
```

### Kondisyon Sayıları (params.json varsayılanları)

| Matris | κ | Anlam |
|--------|---|-------|
| $R_x$ | 141 | Yatay yanıt iyi koşullu |
| $R_y$ | 161 | Dikey yanıt iyi koşullu |
| $\Delta R_y$ (2% Δg) | ~27 500 | ΔR yöntemi 170× daha kötü koşullu |

### Dahili Test

```bash
python fodo_lattice.py
```

```
--- Düzlem: x ---  Q=2.682400  κ(R)=141  FFT RMS=4.5 μm  Direct RMS=4.8e-13 μm
--- Düzlem: y ---  Q=2.361735  κ(R)=161  FFT RMS=5.9 μm  Direct RMS=7.4e-13 μm
```

---

## 9. DFT/FFT Tabanlı Quad Geri Çatım: `spectral_inversion.py`

Bu betik, BPM ölçümlerinden quad hizalama hatalarını geri çatmak için dört aşamalı bir analiz gerçekleştirir. Simülasyon "gerçek makine" rolünü üstlenir.

### Neden ΔR Yöntemi Yetersiz?

Klasik iki-kmod yaklaşımında iki ölçümün farkı alınır:

$$\Delta y = y_1 - y_2 = (R_1 - R_2) \cdot \Delta q = \Delta R \cdot \Delta q$$

BPM ofseti farkta iptal olduğundan bu yaklaşım cazip görünür. Ancak $\kappa(\Delta R) \approx 27\ 500$ — yani girişteki %1 model hatası çıkışta %275 rekonstrüksiyon hatasına dönüşür. Bu yöntem pratikte başarısız olur.

### İki-kmod Ayrı Tersme Yöntemi

Her ölçüm **ayrı ayrı** tersine çevrilir:

$$v_1 = R_1^{-1} \cdot y_1 = \Delta q + R_1^{-1} \cdot b$$
$$v_2 = R_2^{-1} \cdot y_2 = \Delta q + R_2^{-1} \cdot b$$
$$\hat{\Delta q} = \frac{v_1 + v_2}{2}$$

Her $R_i$ iyi koşulludur ($\kappa \approx 150$); ortalama alma BPM ofset kalıntısını kısmi baskılar. Kondisyon avantajı: $\kappa(\Delta R) / \kappa(R) \approx 170\times$.

---

### Aşama A — İdeal Geri Çatım Üst-Sınır Testi

```python
stage_A_ideal(config, plane='y', N_real=20, sigma_q=100e-6, seed=0)
```

**Amaç:** Model ile makine mükemmel uyumda olsaydı rekonstrüksiyon ne kadar iyi olurdu?

**Prosedür:**
1. Analitik $R$ hesapla
2. $N_\text{real}$ adet rastgele $\Delta q$ vektörü üret ($\sigma = 100\ \mu\text{m}$ Gaussian)
3. $y = R \cdot \Delta q$ (simülasyon yok)
4. FFT ve direct yöntemlerle geri çat, RMS hatayı raporla

**Beklenti ve tipik sonuç:**

| Yöntem | RMS hata | Korelasyon |
|--------|----------|------------|
| Direct (np.linalg.solve) | < 1 pm | 1.000000 |
| FFT (sirkülant yaklaşım) | ~5 μm | ~0.999 |

Direct çözüm makine hassasiyetindedir; FFT hatasının kaynağı FODO'nun tam sirkülant olmamasıdır.

---

### Aşama B — Kondisyon Sayısı Haritası

```python
stage_B_condition_map(config, plane='y', g_pert_frac=0.02, out_dir='.')
```

**Amaç:** Her Fourier modunun ne kadar iyi ya da kötü koşullu olduğunu görsel olarak göster.

**Prosedür:**
1. $R_1$ (nominal $g$), $R_2$ (pertürbe $g$), $\Delta R = R_1 - R_2$ matrislerini hesapla
2. Her matrisin $\beta$-normalize ilk satırını FFT'le → özdeğerler $\lambda_k$
3. $|\lambda_k|^{-1}$ mod bazlı kondisyon faktörünü çiz (log ölçek)

**Çıktı dosyaları:** `stage_B_condition_x.png`, `stage_B_condition_y.png`

**Tipik değerler (y-düzlemi, Δg/g = +2%):**

```
κ(R₁) ≈  161       ← iki-kmod yöntemi bunu kullanır
κ(R₂) ≈  161
κ(ΔR) ≈ 27 560     ← ΔR yöntemi bunu kullanır — 171× daha kötü
```

---

### Aşama C — İki-kmod Rekonstrüksiyonu (Simülasyon = Gerçek Makine)

```python
stage_C_two_kmod(config, plane='y', g_pert_frac=0.02,
                 sigma_dq=100e-6, seed=42, t_end=1.5e-4)
```

**Amaç:** Simülasyon "gerçek makine" rolünü üstlenir; analitik model $R$'yi sağlar. İkisi arasındaki küçük Twiss uyuşmazlığında yöntemin başarısı ölçülür.

**Prosedür:**
1. Rastgele $\Delta q$ vektörü üret (seed sabit, $\sigma = 100\ \mu\text{m}$)
2. Simülasyon $(g_\text{nom},\ \Delta q)$ → $y_1$
3. Simülasyon $(g_\text{pert},\ \Delta q)$ → $y_2$
4. Analitik $R_1$, $R_2$ hesapla
5. $v_1 = R_1^{-1} \cdot y_1$,  $v_2 = R_2^{-1} \cdot y_2$
6. $\hat{\Delta q} = (v_1 + v_2)/2$

**BPM konumları:** `cod_data.txt`'den QF giriş (satır `k×8+2`) ve QD giriş (satır `k×8+6`) okunur; her biri mm cinsinden — m'ye çevrilir.

**Tipik sonuçlar:**

| Düzlem | Yöntem | RMS hata | Korelasyon |
|--------|--------|----------|------------|
| y | İki-kmod ort. | 1.4 μm | 0.9999 |
| y | ΔR direkt | 406 μm | 0.62 |
| x | İki-kmod ort. | 4.8 μm | 0.9980 |
| x | ΔR direkt | 1 626 μm | 0.07 |

İki-kmod yöntemi ΔR yöntemine karşı **280–340× daha iyi RMS** sağlamaktadır.

**Başarı kriteri (YAPILACAKLAR.md):** RMS < 10 μm, corr > 0.95 — sağlandı.

---

### Aşama D — Gürbüzlük Testi

```python
stage_D_robustness(config, plane='y', g_pert_frac=0.02,
                   sigma_dq=100e-6, seed=42, t_end=1.5e-4, N_trials=8)
```

**Amaç:** Dört gerçekçi hata kaynağında rekonstrüksiyon kalitesinin nasıl bozulduğunu sistematik biçimde ölçmek.

**Test edilen hata türleri:**

| # | Hata kaynağı | Tarama aralığı | Uygulama yöntemi |
|---|---|---|---|
| 1 | BPM ölçüm gürültüsü | 0–20 μm RMS | $y_{1,2}$'ye bağımsız Gaussian ekle |
| 2 | BPM sabit ofseti | 0–200 μm RMS | Her iki ölçüme aynı $b$ vektörü ekle |
| 3 | Model beta hatası | 0–5% RMS | $\beta$'yı $(1+\delta)$ ile pertürbe et, $R$'yi yeniden inşa et |
| 4 | Kuadrupol eğimi | 0–2 mrad RMS | Ayrı simülasyon çifti `quad_tilt_arr` ile |

**1 ve 3 için baz simülasyon yeniden kullanılır (hızlı). 4 için her eğim seviyesinde yeni simülasyon koşulur.**

**Çıktı dosyaları:** `stage_D_robustness_y.png`, `stage_D_robustness_x.png`

**Tipik sonuçlar (y-düzlemi):**

| Hata türü | 10 μm eşiği aşıldığı nokta | Yorum |
|---|---|---|
| BPM gürültüsü | σ ≈ 3–4 μm | Kritik: gürültü azaltılmalı |
| BPM sabit ofseti | σ ≈ 10 μm | Hassas: iki-kmod ofseti tam iptal etmez |
| Model β hatası | δβ/β ≈ 2% | Twiss hassasiyeti için üst sınır |
| Kuadrupol eğimi | > 2 mrad dahi iyi | Oldukça gürbüz |

---

### `build_R_for_gradient()` Yardımcı Fonksiyonu

```python
R, beta, phi, Q, KL = build_R_for_gradient(config, g, plane, K_x_arc_x=None)
```

Verilen gradyan $g$ için config'in `g1` alanını geçici güncelleyerek Twiss + $R$ hesaplar. `K_x_arc` yatay kalibrasyonu bir kez yapılıp farklı $g$ değerleri için sabit tutulur (arc geometrisi $g$'den bağımsız).

---

### Tüm Aşamaları Çalıştırmak

```bash
python spectral_inversion.py
```

Sırayla çalışır: A (her iki düzlem) → B (her iki düzlem) → C (y, x) → D (y, x).

Toplam süre: ~30–40 dakika (D aşamasındaki tilt simülasyonları dahil).

---

## 10. Quad Tilt: Skew-Quadrupol ve x-y Kuplajı

### Fizik: Neden Quad Tilt = Skew-Quadrupol?

Bir quadrupol, ışın ekseni (s) etrafında küçük bir $\theta$ açısıyla döndürülürse alan dağılımı şu hale gelir:

$$
\begin{aligned}
B_r &= G_1\,(Z - d_y) + \underbrace{(-2 G_1\,\theta)\,(X - R_0 - d_x)}_{\text{skew bileşeni}} \\
B_z &= G_1\,(X - R_0 - d_x) + \underbrace{(+2 G_1\,\theta)\,(Z - d_y)}_{\text{skew bileşeni}}
\end{aligned}
$$

Faktör 2, normal quad alanının dört-katlı simetrisinden gelir ($\sin 2\theta \approx 2\theta$). Skew terimleri **çapraz** bağlantı yaratır: radyal sapma dikey kuvvet üretir, dikey sapma radyal kuvvet üretir. Sonuç **x-y kuplajıdır**.

### C++ Uygulaması (`integrator.cpp`)

`field_params_local[27]` indeksine her quad için tilt açısı yazılır (QF ve QD ayrı ayrı). Quad alanları hesaplanırken skew bileşenleri eklenir:

```cpp
double q_tilt = field_params[27];
if (q_tilt != 0.0) {
    B_quad_r += -2.0 * current_G1 * q_tilt * dev_quad;  // radyal mesafe × skew
    B_quad_z +=  2.0 * current_G1 * q_tilt * vert_rel;  // dikey mesafe × skew
}
```

### Rekonstrüksiyona Etkisi

`spectral_inversion.py` yanıt matrisi quad tilti modellemez. Tilt, BPM ölçümüne modelsiz kirlilik olarak sızar. Aşama D gürbüzlük testinden elde edilen sayısal tablo:

| σ_tilt | Ek RMS hatası (y-düzlemi) |
|--------|--------------------------|
| 0.0 mrad | 0 μm (baz) |
| 0.5 mrad | +0.03 μm |
| 1.0 mrad | +0.12 μm |
| 2.0 mrad | +0.45 μm |

İki-kmod yöntemi quad eğimine karşı oldukça gürbüzdür: 2 mrad eğimde ek hata 0.5 μm'nin altındadır.

---

## 11. Parametreler: `params.json`

### Geometri ve Fizik

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `R0` | Halka yarıçapı [m] | 95.49 |
| `direction` | Dönüş yönü (−1: saat yönü) | −1 |
| `nFODO` | FODO hücre sayısı | 24 |
| `quadLen` | Quad uzunluğu [m] | 0.4 |
| `driftLen` | Serbest yol uzunluğu [m] | 2.0833 |
| `g1` | QF/QD gradyanı [T/m] | 0.21 |
| `g0` | İkincil quad gradyanı [T/m] | 0.20 |

### Simülasyon Kontrolü

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `t2` | Toplam süre [s] | 0.001 |
| `dt` | Zaman adımı [s] | 1e-11 |
| `return_steps` | Kaydedilen veri noktası sayısı | 10000 |
| `poincare_quad_index` | Poincaré kesiti konumu (−1 = her hücre) | −1 |
| `dev0`, `y0` | Başlangıç tepmesi [m] | 1e-5 |

### Hata Modeli

| Parametre | Açıklama |
|-----------|----------|
| `error_quad_index` | Tek quad hatası indeksi (−1 = devre dışı) |
| `error_quad_dy/dx` | Tek quad kaçıklığı [m] |
| `quad_random_dy_max` | Tüm quadlara rastgele dikey hata genliği [m] |
| `quad_random_dx_max` | Tüm quadlara rastgele radyal hata genliği [m] |
| `quad_random_seed` | Tekrarlanabilirlik için tohum |
| `error_dipole_index` | Tek deflektör hatası indeksi (−1 = devre dışı) |
| `error_dipole_tilt` | Tek deflektör eğim açısı [rad] |
| `dipole_random_tilt_max` | Tüm deflektörlere rastgele eğim açısı [rad] |
| `dipole_random_seed` | Deflektör tilt rastgele tohumu |
| `quad_random_tilt_max` | Tüm quadlara rastgele eğim açısı [rad] |
| `quad_random_tilt_seed` | Quad tilt rastgele tohumu |

### BPM Hata Modeli

| Parametre | Açıklama |
|-----------|----------|
| `bpm_noise_sigma` | Her ölçümde bağımsız Gaussian gürültü std [m] |
| `bpm_offset_sigma` | BPM başına sabit sistematik ofset std [m] |
| `bpm_offset_seed` | Ofset vektörü rastgele tohumu |

---

## 12. Kurulum ve Çalıştırma

### Gereksinimler

```bash
pip install numpy scipy matplotlib
```

### C++ Kütüphanesinin Derlenmesi

```bash
# Linux
g++ -shared -o lib_integrator.so -fPIC -O3 integrator.cpp

# macOS
clang++ -dynamiclib -o lib_integrator.dylib -O3 integrator.cpp
```

### Dosya Yapısı

| Dosya | Rol |
|-------|-----|
| `integrator.cpp` | C++ GL4 entegratör (parçacık dinamiği + spin) |
| `integrator.py` | C kütüphanesine ctypes köprüsü |
| `run_simulation.py` | Simülasyon orkestrasyonu, COD ve Poincaré verisi |
| `plot_results.py` | Simülasyon görselleştirme paneli |
| `fodo_lattice.py` | Analitik Twiss, yanıt matrisi, FFT geri dönüşüm |
| `spectral_inversion.py` | Dört aşamalı quad hizalama geri çatım analizi |
| `params.json` | Örgü ve hata parametreleri |

### Tipik İş Akışı

**Adım 1 — Normal simülasyon** (parçacık dinamiği, COD, Poincaré verisi):

```bash
python run_simulation.py
```

Çıktılar: `cod_data.txt`, `poincare_data.txt`, `history.txt`

**Adım 2 — Görselleştirme:**

```bash
python plot_results.py
# → simulasyon_sonuclari.png
```

**Adım 3 — Analitik Twiss doğrulaması:**

```bash
python fodo_lattice.py
# → Qx, Qy, κ(R) raporlanır; Direct ve FFT geri dönüşüm test edilir
```

**Adım 4 — Quad geri çatım analizi (4 aşama):**

```bash
python spectral_inversion.py
# → Aşama A: ideal test (her iki düzlem)
# → Aşama B: kondisyon haritası — stage_B_condition_x.png, stage_B_condition_y.png
# → Aşama C: iki-kmod rekonstrüksiyon (simülasyonlu)
# → Aşama D: gürbüzlük taraması — stage_D_robustness_y.png, stage_D_robustness_x.png
```

Toplam süre: ~30–40 dakika (tilt simülasyonları dahil).

### Tipik Parametre Değişiklikleri

Rastgele quad hataları eklemek için `params.json`'da:

```json
"quad_random_dy_max": 0.0001,
"quad_random_dx_max": 0.0001,
"quad_random_seed": 42
```

Tek bir quada hata vermek için:

```json
"error_quad_index": 5,
"error_quad_dy": 0.0003
```

Quad eğimi eklemek için:

```json
"quad_random_tilt_max": 0.001,
"quad_random_tilt_seed": 44
```
