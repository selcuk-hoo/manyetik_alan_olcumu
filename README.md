# 6D Proton EDM Depolama Halkası Simülatörü

**Yazar:** Selcuk H.

Bu proje, Proton Elektrik Dipol Momenti (EDM) deneyleri için tasarlanmış tam 6 boyutlu bir depolama halkası simülasyonudur. Parçacık dinamiği ve spin presesyonu C++ ile yüksek hassasiyetle çözülür; parametre yönetimi, analiz ve görselleştirme Python katmanında yapılır.

---

## İçindekiler

1. [Fiziksel Arkaplan](#1-fiziksel-arkaplan)
2. [Halka Geometrisi: FODO Örgüsü](#2-halka-geometrisi-fodo-örgüsü)
3. [Koordinat Sistemi](#3-koordinat-sistemi)
4. [C++ Entegratör: `integrator.cpp`](#4-c-entegratör-integratorcpp)
5. [Python Köprüsü: `integrator.py`](#5-python-köprüsü-integratorpy)
6. [Simülasyon Orkestrasyonu: `run_simulation.py`](#6-simülasyon-orkestrasyonu-run_simulationpy)
7. [Görselleştirme: `plot_results.py`](#7-görselleştirme-plot_resultspy)
8. [Tepki Matrisi: `build_response_matrix.py`](#8-tepki-matrisi-build_response_matrixpy)
9. [Quad Geri Çatım Testi: `test_reconstruction.py`](#9-quad-geri-çatım-testi-test_reconstructionpy)
10. [LOCO Geri Çatım Testi: `test_loco_reconstruction.py`](#10-loco-geri-çatım-testi-test_loco_reconstructionpy)
10b. [K-Modülasyon Geri Çatım Testi: `test_kmod_reconstruction.py`](#10b-k-modülasyon-geri-çatım-testi-test_kmod_reconstructionpy)
11. [Parametreler: `params.json`](#11-parametreler-paramsjson)
12. [Kurulum ve Çalıştırma](#12-kurulum-ve-çalıştırma)

---

## 1. Fiziksel Arkaplan

### Neden bu simülasyon?

Proton EDM deneyi, protonun elektrik dipol momentini ölçerek CP-simetri ihlalini aramayı hedefler. Deney, sihirli momentumda ($p \approx 0.701\ \text{GeV/c}$) dolaşan protonların spinini radyal elektrik alanla "dondurarak" küçük bir EDM sinyali arar.

Bunu yapabilmek için halkadaki her türlü hizalama hatası (quad kaçıklıkları, deflektör açısal sapmaları) hassas biçimde ölçülmeli ve düzeltilmelidir. Bu simülatör iki temel soruyu yanıtlar:

1. **İleri problem:** Verilen bir hata kümesi için kapalı yörünge sapması (COD) ne kadardır?
2. **Ters problem:** Ölçülen COD'dan asıl hataları geri çatabilir miyiz?

### Sihirli Momentum

Proton EDM deneyinin can alıcı koşulu:

$$p_{\text{magic}} = \frac{m_p c}{\sqrt{G_p}} \approx 0.701\ \text{GeV/c}$$

Bu momentumda, elektrik alandan kaynaklanan spin presesyonu tam olarak sıfırlanır (Thomas terimi ile Larmor terimi birbirini götürür). Böylece spin, radyal yönde donmuş kalır ve yalnızca EDM varlığında dikey bileşen kazanır.

---

## 2. Halka Geometrisi: FODO Örgüsü

Halka, 24 özdeş **FODO hücresi**nden oluşur. Her hücre 8 elemandan ibarettir ve sırayla şöyle ilerler:

```
ARC1 → DRIFT → QF → DRIFT → ARC2 → DRIFT → QD → DRIFT
elem=0   =1    =2    =3    =4    =5    =6    =7
```

| Eleman | Tipi | Görevi |
|--------|------|--------|
| ARC1, ARC2 | Silindirik kapasitör (elektrik yay) | Parçacığı büküp halka boyunca taşır |
| QF | Odaklayan quadrupol (G₁ > 0) | Radyal düzlemde odaklar |
| QD | Ayrıştıran quadrupol (−G₁) | Dikey düzlemde odaklar |
| DRIFT | Serbest yol | Saha yok, parçacık düz ilerler |

24 hücre × 2 quad/hücre = **48 quadrupol**, dolayısıyla tepki matrisi 48×48 boyutundadır.

### Betatron Tune

FODO örgüsündeki odaklama gücü, parçacığın halkayı her dolaşımında kaç salınım yaptığını belirler: bu sayıya **betatron tune** denir.

$$Q_x \approx 2.69 \qquad Q_y \approx 2.37 \quad (G_1 = 0.21\ \text{T/m için})$$

Tune tamsayıdan uzak tutulmazsa rezonans instabilitesi oluşur.

---

## 3. Koordinat Sistemi

Simülatör **global Kartezyen** koordinat kullanır:

- **X**: Halka düzleminde radyal yön (halka merkezinden dışa doğru)
- **Y**: Halka düzleminde azimutal yön (parçacık bu yönde hareket eder)
- **Z**: Dikey yön

İdeal yörünge, X-Y düzleminde `R₀ = 95.49 m` yarıçaplı bir çemberdir. Parçacık `(X = R₀, Y = 0, Z = 0)` noktasından başlar ve `−Y` yönünde (saat yönünde, `direction = −1`) hareket eder.

### Yerel ↔ Global Dönüşüm (`integrator.py`)

Her yay elemanından sonra `rotate_all()` C++ fonksiyonu koordinat çerçevesini `−Φ_def` kadar döndürür. Bu sayede parçacık her eleman girişinde daima `X ≈ R₀, Y ≈ 0` konumundan başlıyor gibi görünür (**dönen çerçeve** ya da Frenet–Serret benzeri). Python katmanı ise analiz için bu global koordinatları yerel sapmaya `(x = X − R₀, y = Z)` çevirir:

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

**Yay (ARC, tip 0):** Silindirik kapasitör. Radyal elektrik alan ve isteğe bağlı saçaklanma alanları:

$$E_r(R,Z) = E_0 \left(\frac{R_0}{R}\right)^n \left[1 - \frac{n^2-1}{2}\left(\frac{Z}{R}\right)^2 + \ldots\right]$$

**Quadrupol (QF/QD, tip 2/3):** Kaçıklık bileşenleri dahil saf quadrupol alanı:

$$B_r = G_1\,(Z - d_y) \qquad B_Z = G_1\,(X - R_0 - d_x)$$

Burada $d_y$ dikey, $d_x$ radyal quad kaçıklığıdır. Dikkat edilmesi gereken fizik:
- **$d_y \neq 0$** → dikey kuvvet → **dikey (y) yörünge** değişir
- **$d_x \neq 0$** → radyal kuvvet → **radyal (x) yörünge** değişir

İki düzlem birbirinden bağımsızdır (çift yok).

### Kapalı Yörünge Verisi: `cod_data.txt`

Her FODO hücresiyle eleman sınırında parçacığın konumu `stage_x` (radyal sapma) ve `stage_y` (dikey konum) arabelleklerine alınır. Her devir sonunda bu değerler birikimli toplama eklenir. Simülasyon bitince toplam tur sayısına bölünerek **tur ortalamalı COD** dosyaya yazılır. Betatron salınımları sıfır-ortalıklı olduğundan ortalama işlemi onları yok eder; geriye yalnızca **kapalı yörünge sapması** kalır.

### Poincaré Kesiti

`target_quad < 0` (yani `poincare_quad_index = −1`) seçildiğinde, her FODO hücresinin ARC1 girişinde (`elem = 0`) parçacık durumu kaydedilir: bu **tur başına 24 nokta** demektir. Poincaré verisinden Betatron Tune şöyle çıkarılır:

```
Q = nFODO × ⟨Δφ⟩ / (2π)
```

burada `Δφ` ardışık noktalar arasındaki ortalama faz adımıdır (`arctan2(x', x)` ile hesaplanır). Bunun çalışması için parçacığın `dev0` ya da `theta0` ile küçük bir betatron tepmesi almış olması gerekir.

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

---

## 6. Simülasyon Orkestrasyonu: `run_simulation.py`

### Sihirli Momentumun Hesabı

```python
p_magic = M2 / sqrt(AMU)          # GeV/c cinsinden
beta0   = p_magic / sqrt(p²+M²)   # göreli hız
E0_V_m  = -(p_magic * beta0 / R0) * 1e9  # gerekli radyal elektrik alan [V/m]
```

Elektrik alan miktarı, sihirli momentumdaki protonu `R₀` yarıçaplı dairesel yörüngede tutacak biçimde otomatik hesaplanır.

### Başlangıç Koşulları

`params.json`'dan okunan `dev0` (radyal sapma), `y0` (dikey sapma), `theta0_hor/ver` (açısal sapma) ile başlangıç faz uzayı noktası oluşturulur. Tune ölçümü için `dev0 = 1e-5 m` (10 μm) varsayılan tepme değeri atanmıştır; bu küçük tepme ~385 tur boyunca ortalamada ~0.5 μm COD bırakır ve ölçümleri etkilemez.

### Hata Dizileri

Her quad için `quad_dy`, `quad_dx` ve her deflektör için `dipole_tilt` dizileri oluşturulur. `params.json`'dan tek bir elemana veya tüm halkalara rastgele hata verilmesi desteklenir.

### Tune Tahmini: `_tune_full()`

Poincaré verisi üzerinden `arctan2(x', x)` faz açısı hesaplanır, `np.unwrap` ile 2π atlamaları düzeltilir ve ardışık adımlar arasındaki ortalama açı adımından tune elde edilir:

```python
dphi     = np.diff(np.unwrap(np.arctan2(upc, uc)))
avg_dphi = abs(np.mean(dphi))
Q = (nFODO * avg_dphi) / (2 * pi)   # poincare_quad_index < 0 ise
```

### Savitzky-Golay Spin Trendi

Spin bileşenleri hızlı g-2 salınımı içerir. Bu salınımı yumuşatıp altındaki yavaş EDM trendini ortaya çıkarmak için Savitzky–Golay filtresi uygulanır, ardından doğrusal fit ile eğim hesaplanır.

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

### `_plot_cod()`

COD profilini, RMS değerini ve betatron tune etiketini tek bir yardımcı fonksiyonda çizer.

### `_estimate_tune()`

`plot_results.py` içindeki bağımsız tune tahmincisidir; `poincaré_data.txt` dosyasını okuyarak `arctan2` yöntemiyle $Q_x$ ve $Q_y$ hesaplar.

### `_save_rf_plot()`

`rf.txt` mevcutsa RF kovuk faz diyagramını (`ψ` vs `dp/p`) ayrı bir dosyaya (`rf.png`) kaydeder.

---

## 8. Tepki Matrisi: `build_response_matrix.py`

### Motivasyon

Kapalı yörünge sapması ile hizalama hataları arasındaki doğrusal ilişki matrislerle özetlenebilir:

$$\mathbf{y}_{\text{COD}} = R_{dy} \cdot \mathbf{d}_y + R_{\text{tilt}} \cdot \boldsymbol{\theta} \qquad \mathbf{x}_{\text{COD}} = R_{dx} \cdot \mathbf{d}_x$$

Burada $\mathbf{d}_y$ quad dikey kaçıklıkları, $\mathbf{d}_x$ quad radyal kaçıklıkları ve $\boldsymbol{\theta}$ deflektör eğim açılarıdır.

Dikkat: Dipol eğimi (tilt) B alanına **radyal bir bileşen** ekler ($B_X = B_{eq}\sin\theta$) ve bu yalnızca **dikey COD**'u etkiler. Radyal COD yalnızca quad radyal kaçıklığına ($d_x$) duyarlıdır. İki düzlem bu sayede ayrışır.

### `setup_fields(config, g1_override=None)`

`params.json`'dan bağımsız, sabit başlangıç koşulları oluşturur: her simülasyon `(x=0, y=0)` ideal yörüngeden başlar. `g1_override` parametresi LOCO için ikinci optik konfigürasyonu tanımlar.

### `run_sim(alanlar, state0, config, quad_dy, quad_dx, dipole_tilt=None)`

Tek bir simülasyon çalıştırır. `dipole_tilt` verilmezse sıfır kabul edilir. BPM konumları:

```python
def read_cod_quads(nFODO):
    qf = k * 8 + 2   # QF giriş satırı (elem=2)
    qd = k * 8 + 6   # QD giriş satırı (elem=6)
```

### `build_matrices(config, g1_override=None, ..., n_workers=1)`

Bir optik konfigürasyon için üç matrisi birden hesaplar. Her sütun **sonlu fark** (numerik Jacobian) ile elde edilir:

**Quad matrisleri (R_dy ve R_dx) — ayrı koşumlar:**

Her quad için yalnızca tek bileşen pertürbe edilir:

```python
# R_dy sütunu i: yalnız dy[i] = δ_q
R_dy[:, i] = (y_cod_dy_i - y0) / delta_q

# R_dx sütunu i: yalnız dx[i] = δ_q
R_dx[:, i] = (x_cod_dx_i - x0) / delta_q
```

> **Not (önceki sürümden değişiklik):** Eski uygulama her `i` için `dy[i]` ve `dx[i]`'yi *aynı koşumda* pertürbe ediyordu. Bu, "düzlemler bağımsız" varsayımına dayalı bir optimizasyondu fakat $\partial y/\partial d_x \neq 0$ olduğunda (örn. skew-quad veya solenoid kuplajı varsa) `R_dy` sütunlarına `dx`'ten sızıntı bırakıyordu. Yeni uygulama dy ve dx pertürbasyonlarını ayırır → koşum sayısı 2 katına çıkar, ancak paralelleştirme bunu fazlasıyla telafi eder.

**Dipol tilt matrisi (R_tilt) — ayrı koşum:**
Her `i` için yalnızca `tilt[i] = δ_tilt` uygulanır:

```python
R_tilt[:, i] = (y_cod_tilt_i - y0) / delta_tilt  # [m/rad]
```

Dipol tilt indeksleme: `tilt[2k]` = hücre k'daki ARC1, `tilt[2k+1]` = ARC2.

Toplam koşum sayısı: 1 referans + 48 dy + 48 dx + 48 tilt = **145 koşum** per konfigürasyon.

### Paralelleştirme

Tüm pertürbasyon koşumları birbirinden bağımsızdır → `ProcessPoolExecutor` ile çekirdek sayısı kadar paralel çalıştırılır:

```bash
python build_response_matrix.py --workers 7   # 7 paralel süreç
```

**Çakışma sorunu ve çözümü:** C++ entegratör, `cod_data.txt`'yi geçerli çalışma dizinine yazar. Tüm worker'lar aynı dizinde olsaydı dosyalar birbirini ezerdi. Bu nedenle her worker süreci başlangıçta kendine ait bir geçici dizin oluşturur ve oraya `chdir` eder:

```python
def _worker_init():
    tmp = tempfile.mkdtemp(prefix=f"kmod_w{os.getpid()}_")
    os.chdir(tmp)
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
```

Worker süreci kapanırken (`atexit`) dizin silinir.

**Ölçeklenme:** N worker ile yaklaşık N kat hızlanma beklenir (tek koşum CPU-bağımlı, dosya I/O minimal). 8 çekirdekte 7 worker, 12 çekirdekte 11 worker önerilir (1 çekirdek OS için).

### LOCO: İki Optik Konfigürasyon

Tek ölçümde:
$$\mathbf{y}_{\text{COD}} = R_{dy}\,\mathbf{d}_y + R_{\text{tilt}}\,\boldsymbol{\theta}$$
48 denklem, 96 bilinmeyen → **belirsiz sistem.**

İki farklı optik konfigürasyonla (nominal $g_1$ ve $g_1 \times 1.02$):

$$\begin{pmatrix} \mathbf{y}_1 \\ \mathbf{y}_2 \end{pmatrix} = \underbrace{\begin{pmatrix} R_{dy,1} & R_{\text{tilt},1} \\ R_{dy,2} & R_{\text{tilt},2} \end{pmatrix}}_{M_{96\times96}} \begin{pmatrix} \mathbf{d}_y \\ \boldsymbol{\theta} \end{pmatrix}$$

Optik değişince $R_{dy}$ ve $R_{\text{tilt}}$ farklı oranlarda değişir; $M$'nin koşulunu $\kappa(M)$ ile kontrol ederiz.

### Kaydedilen Dosyalar

| Dosya | İçerik | Boyut |
|-------|--------|-------|
| `R_dy.npy`, `R_dx.npy` | Nominal, geriye dönük uyumluluk | 48×48 |
| `R_dy_1.npy`, `R_dx_1.npy`, `R_tilt_1.npy` | Nominal konfigürasyon | 48×48 |
| `R_dy_2.npy`, `R_dx_2.npy`, `R_tilt_2.npy` | Pertürbe konfigürasyon | 48×48 |
| `M_loco.npy` | Birleşik LOCO matrisi | 96×96 |

### Koşul Sayıları

```
κ(R_dy) ≈ 165,   κ(R_dx) ≈ 152   (nominal, quad-only)
κ(M_loco): simülasyon sonrası raporlanır
```

$\kappa(M) < 10^6$ → doğrudan çözüm; $10^6$–$10^{10}$ → SVD/Tikhonov önerilir.

---

## 9. Quad Geri Çatım Testi: `test_reconstruction.py`

### Yöntem

1. `seed=7` ile tekrarlanabilir rastgele hatalar üretilir (±0.5 mm)
2. Referans ve hatalı simülasyonlar çalıştırılır
3. Net COD farkı alınır
4. 48×48 doğrusal sistem çözülür:

$$\hat{\mathbf{d}}_y = R_{dy}^{-1}\,\mathbf{y}_{\text{COD}} \qquad \hat{\mathbf{d}}_x = R_{dx}^{-1}\,\mathbf{x}_{\text{COD}}$$

### Sonuçlar

| Metrik | dy | dx |
|--------|----|----|
| Korelasyon | 1.000000 | 1.000000 |
| Hata RMS | ~0.05 μm | ~0.08 μm |

COD RMS ~1.7 mm'den 48 quad kaçıklığı ~0.1 μm hassasiyetle geri çatılır.

---

## 10. LOCO Geri Çatım Testi: `test_loco_reconstruction.py`

### Motivasyon

Gerçek bir halkada hem quad kaçıklıkları hem de deflektör eğimleri vardır ve ikisi de dikey COD'u etkiler. Tek bir ölçümde:

$$\mathbf{y}_{\text{COD}} = R_{dy}\,\mathbf{d}_y + R_{\text{tilt}}\,\boldsymbol{\theta}$$

Bu 48 denklem, 96 bilinmeyen içerir → **belirsiz sistem**, doğrudan çözülemez.

**LOCO çözümü:** İki farklı optik konfigürasyonda ölçüm al. Konfigürasyonlar arasında $R_{dy}$ ve $R_{\text{tilt}}$ farklı oranlarda değiştiğinden sistem belirli hale gelir:

$$\underbrace{\begin{pmatrix} R_{dy,1} & R_{\text{tilt},1} \\ R_{dy,2} & R_{\text{tilt},2} \end{pmatrix}}_{M_{96\times96}} \begin{pmatrix} \mathbf{d}_y \\ \boldsymbol{\theta} \end{pmatrix} = \begin{pmatrix} \mathbf{y}_1 \\ \mathbf{y}_2 \end{pmatrix}$$

Konfigürasyon 1: nominal $g_1 = 0.21$ T/m. Konfigürasyon 2: $g_1 \times 1.02$ (tüm quadlar %2 artırılmış). Radyal kaçıklık $d_x$ tilt etkilemediğinden ayrı bir 48×48 sistemle çözülür.

---

### BPM Hata Modeli

Gerçek deneyde iki tür BPM hatası kaçınılmazdır.

**Gürültü** (`bpm_noise_sigma`, $\sigma_n$): Elektronik okuma gürültüsü. Ölçümden ölçüme bağımsız Gaussian değişkendir. Tekrarlı ölçüm ortalamasıyla bastırılabilir, ancak pratikte ~1 μm altına indirilmesi zordur.

**Ofset** (`bpm_offset_sigma`, $\sigma_o$): Mekanik veya elektriksel kalibrasyon hatası. BPM başına sabit ve sistematiktir. Ortalama almakla **giderilmez**.

**Common-mode rejection:** COD ölçümü her zaman bir fark işlemidir:

$$\mathbf{y}_{\text{COD}} = \mathbf{y}_{\text{hatalı}} - \mathbf{y}_{\text{referans}}$$

Her iki ölçüm de aynı BPM'lerden okunduğundan sabit ofset her ikisinde de aynıdır ve farkta iptal olur. Geriye yalnızca gürültü kalır; etkin standart sapma $\sqrt{2}\,\sigma_n$'dir (iki bağımsız gürültü).

Bu nedenle kod, BPM hatalarını ham ölçümlere (fark almadan önce) uygular:

```python
y0_meas    = y0    + offset + noise_ref    # referans ölçümü
y_cod_meas = y_cod + offset + noise_pert   # hatalı ölçüm
dy_cod = y_cod_meas - y0_meas              # offset iptal, √2·noise kalır
```

**Tepki matrisi inşasında:** Ofset yine farkta iptal olur (`y_pert − y_ref`). Yalnızca gürültü her matris sütununu etkiler. Etkin sütun gürültüsü: $\sqrt{2}\,\sigma_n / \delta_q$.

---

### Tikhonov Regularizasyonu

Doğrudan çözüm `np.linalg.solve(M, b)` gürültüyü $\kappa(M) \approx 6.5\times10^5$ katı büyütür; küçük bir ölçüm hatası çözümü mahveder.

**Tikhonov regularizasyonu** gürültü bastırma ile sinyal kaybı arasında denge kurar:

$$\min_{\mathbf{x}} \|\,M\mathbf{x} - \mathbf{b}\,\|^2 + \lambda\|\mathbf{x}\|^2$$

SVD ($M = U\Sigma V^T$) üzerinden kapalı form çözümü:

$$\hat{\mathbf{x}} = V\,\text{diag}\!\left(\frac{\sigma_i}{\sigma_i^2+\lambda}\right)U^T\mathbf{b}$$

**Filtre faktörü** $f_i = \sigma_i/(\sigma_i^2+\lambda)$'nın davranışı:

| $\sigma_i$ | $f_i$ yaklaşımı | Etki |
|---|---|---|
| $\sigma_i \gg \sqrt{\lambda}$ | $\approx 1/\sigma_i$ | mod korunur |
| $\sigma_i \ll \sqrt{\lambda}$ | $\approx \sigma_i/\lambda \to 0$ | mod bastırılır |

Kırpmalı SVD'den farkı: sert sıfırlama yerine sürekli bastırma. $\lambda$ arttıkça küçük modlar yavaşça söner; bu gürültülü sistemlerde daha kararlıdır.

### Lambda Seçimi

**Optimal $\lambda$** (yalnızca simülasyonda): $x_{\text{true}}$ bilinerek $\|\hat{x} - x_{\text{true}}\|$ minimize edilir. Gerçek deneyde uygulanamaz; teorik üst sınırı gösterir.

**Uyumsuzluk ilkesi** (gerçek deneyde uygulanabilir): Mükemmel çözümde bile residual sıfır olamaz — en iyi ihtimalle ölçüm gürültüsü kadar kalır:

$$\|M\hat{x} - \mathbf{b}\| \approx \sigma_{\text{eff}} \cdot \sqrt{n_{\text{obs}}}$$

COD ölçümlerinde ofset farkta iptal olduğundan $\sigma_{\text{eff}} = \sqrt{2}\,\sigma_n$. Bu hedefi sağlayan $\lambda$ log-uzayda ikili arama (binary search) ile bulunur. Yalnızca $\sigma_n$ bilmek yeterlidir.

---

### Sonuçlar

Test koşulları: $\sigma_n = 10\,\mu\text{m}$, $\sigma_o = 50\,\mu\text{m}$; $d_y \in \pm 0.3\,\text{mm}$, $d_x \in \pm 0.3\,\text{mm}$, $\theta \in \pm 0.2\,\text{mrad}$.

| Yöntem | dy hata RMS | dy korelasyon | tilt hata RMS | tilt korelasyon |
|---|---|---|---|---|
| linalg.solve | 92.7 μm | 0.874 | 207 μrad | 0.453 |
| SVD (rcond=1e-6) | 92.7 μm | 0.874 | 207 μrad | 0.453 |
| Tikhonov (optimal $\lambda=1.74$) | 38.2 μm | 0.980 | 39 μrad | 0.925 |
| Tikhonov (uyumsuzluk $\lambda=1.41$) | **37.6 μm** | **0.980** | **40 μrad** | **0.922** |

Radyal kaçıklık $d_x$: her iki yöntemde de 0.67 μm, $r = 0.99999$.

Uyumsuzluk ilkesi ($\lambda = 1.41$), ground truth gerektirmeden optimal $\lambda = 1.74$'e neredeyse eşdeğer performans veriyor. Yöntem gerçek deneyde doğrudan uygulanabilir.

### Nihai Hataların Manyetik Alan Karşılıkları

Halka parametreleri: $p_{\text{magic}} = 0.701\,\text{GeV/c}$, $B\rho = 2.337\,\text{T·m}$, $B_0 = 24.48\,\text{mT}$, $L_{\text{dipol}} = 12.5\,\text{m}$.

**$d_y^{\text{rms}} = 38.2\,\mu\text{m}$** — Kaçık quad başına efektif dipol alanı:

$$B_{\text{eff}} = g_1 \cdot \delta y = 0.21\,\frac{\text{T}}{\text{m}} \times 38.2\,\mu\text{m} = 8.0\,\mu\text{T}$$

$$\Delta BL = g_1 \cdot L_q \cdot \delta y = 3.21\,\mu\text{T·m} \quad (= 1.37\,\text{ppb} \cdot B\rho \text{ per quad})$$

**$\theta^{\text{rms}} = 40.1\,\mu\text{rad}$** — Dönen dipol başına radyal alan bileşeni:

$$B_r = B_0 \cdot \theta = 24.48\,\text{mT} \times 40.1\,\mu\text{rad} = 981\,\text{nT}$$

$$\Delta BL = B_0 \cdot L_{\text{dip}} \cdot \theta = 12.3\,\mu\text{T·m} \quad \left(\frac{B_r}{B_0} = 40.1\,\text{ppm}\right)$$

**$d_x^{\text{rms}} = 0.67\,\mu\text{m}$** — Kaçık quad başına efektif dikey dipol alanı:

$$B_{\text{eff}} = g_1 \cdot \delta x = 141\,\text{nT} \qquad \Delta BL = 56\,\text{pT·m per quad}$$

### Ön Koşul

`build_response_matrix.py` çalıştırılmış ve `M_loco.npy` ile `R_dx_1.npy` mevcut olmalıdır.

---

## 10b. K-Modülasyon Geri Çatım Testi: `test_kmod_reconstruction.py`

### Fikir

Geleneksel COD geri çatımı bir **referans ölçüm** gerektirir: önce hatasız makinede yörüngeyi ölç, sonra hatalı makinede ölç, farkı al. Pratikte hatasız bir makine yoktur; ölçülen şey her zaman BPM ofseti, dipol tilti ve quad kaçıklığının toplamıdır.

K-modülasyon yaklaşımının umudu şudur: aynı hata kümesi sabit kalırken **iki farklı optik konfigürasyonda** (gradyanı $g$ ve $g(1+\varepsilon)$) ölç. İki ölçümü çıkar:

$$\Delta \mathbf{y} = \mathbf{y}(g_{\text{pert}}) - \mathbf{y}(g_{\text{nom}})$$

İdeal beklenti:

| Bileşen | $\mathbf{y}(g_{\text{nom}})$'da | $\mathbf{y}(g_{\text{pert}})$'de | $\Delta \mathbf{y}$'de |
|---------|-------------------------------|--------------------------------|----------------------|
| BPM ofseti $\mathbf{b}$ | $+\mathbf{b}$ | $+\mathbf{b}$ | **0** (common-mode rejection) |
| Quad sinyali $R_{dy}\mathbf{d}_y$ | $R_{dy,1}\mathbf{d}_y$ | $R_{dy,2}\mathbf{d}_y$ | $\Delta R_{dy}\,\mathbf{d}_y$ (büyük, $\propto \varepsilon$) |
| Tilt katkısı $R_{\text{tilt}}\boldsymbol{\theta}$ | $R_{\text{tilt},1}\boldsymbol{\theta}$ | $R_{\text{tilt},2}\boldsymbol{\theta}$ | $\Delta R_{\text{tilt}}\,\boldsymbol{\theta}$ (umut: küçük) |

Eğer $\Delta R_{\text{tilt}}\,\boldsymbol{\theta} \ll \Delta R_{dy}\,\mathbf{d}_y$ ise:

$$\hat{\mathbf{d}}_y = \Delta R_{dy}^{-1}\,\Delta \mathbf{y}$$

ile referans ölçüme **gerek kalmadan** quad kaçıklıkları geri çatılabilir.

### Gerçek Sonuç: Fikir Bu Halkada İşlemiyor

Test çalıştırıldığında tipik çıktı (örneğin $\theta_{\max}=0.2\ \text{mrad}$, $d_{y,\max}=0.3\ \text{mm}$):

```
Δy bileşenleri (gürültüsüz):
  Quad sinyali  ΔR_dy·dy   RMS =    52189 μm
  Tilt kirliliği ΔR_tilt·θ RMS =  1217834 μm
  Oran (kirlilik/sinyal)       = 2333.5%

[Gerçekçi: tilt sabit, BPM hataları var]
  dy   hata RMS = 17090 μm   korelasyon = 0.095
```

Yani quad kaçıklığını geri çatmak istediğimiz hatadan **100 kat daha büyük** bir geri çatım hatası alıyoruz. Tilt kirliliği quad sinyalini gömüyor.

### Neden? Δβ/β ≈ δg/g

Kodun iddiası şuydu: "tilt katkısı yalnız beta fonksiyonu değişiminden gelir, bu da $\Delta\beta/\beta \ll \delta g/g$ olduğu için bastırılır." Bu **yanlış**:

- Tüm quadlar **global** olarak $\varepsilon$ kadar değişince, beta fonksiyonları da $\Delta\beta/\beta \approx \varepsilon$ mertebesinde değişir.
- Tune kayması $\Delta Q$, eğer çalışma noktası bir tamsayı rezonansına yakınsa $\propto \cot(\pi Q)$ ile büyük olabilir.
- Sonuç: $\Delta R_{\text{tilt}}/R_{\text{tilt}} \sim \varepsilon/2 + \pi\cot(\pi Q)\,\Delta Q$

Dolayısıyla:

$$\frac{\|\Delta R_{\text{tilt}}\boldsymbol{\theta}\|}{\|\Delta R_{dy}\mathbf{d}_y\|} \approx \frac{\|R_{\text{tilt}}\boldsymbol{\theta}\|}{2\,\|R_{dy}\mathbf{d}_y\|}$$

Bu halkada elektrostatik bükücü plakaların integre alanı çok büyüktür → **dipol tilt COD'u, quad kaçıklığı COD'undan onlarca kat büyük**. Oran 23× → 2333% kirlilik.

### Çıkarımlar ve Öneriler

1. **Global gradyan değişikliği k-modülasyon için yetersiz.** Tilt kirliliğini bastırmak için ya çok daha küçük gradyan değişikliği (etki yok), ya çok daha büyük (lineerlik bozulur), ya da tamamen farklı bir yaklaşım gerekir.

2. **Önerilen yaklaşım — LOCO birleşik çözüm:** `M_loco` (96×96) matrisini kullanarak $\mathbf{d}_y$ ve $\boldsymbol{\theta}$'yı **birlikte** çöz. Test betiği bunu da yapar (en alttaki "LOCO çözümü" bloğu). $\kappa(M_{\text{loco}})$ küçükse bu yöntem hem dy hem tilt için doğru sonuç verir.

3. **Alternatif — per-quad k-modülasyon:** Tüm quadları aynı anda değiştirmek yerine her birini **tek başına** modüle et. O zaman uzaktaki beta fonksiyonları neredeyse hiç değişmez ($\Delta\beta/\beta \to 0$), tilt kirliliği lokal kalır. Bu klasik k-modülasyon optik ölçüm tekniğidir; bu kodda henüz uygulanmamıştır ve doğal bir sonraki adımdır.

4. **Fizik sınırı:** Eğer bir halkada tilt CO bozulması quad-misalignment CO bozulmasından çok büyükse, hiçbir kmod varyantı dy'yi izole edemez. Tilt'lerin önceden bağımsız ölçülüp düzeltilmesi (veya LOCO ile birlikte çözülmesi) gerekir.

### Test Çıktısı

`test_kmod_reconstruction.py` üç sonucu yan yana raporlar:

```
Özet: dy geri çatım hatası RMS [μm]
  K-mod (yalnız ΔR_dy ile)           : 17090.25   ← tilt'le bozulur
  LOCO (M_loco doğrudan)             :    XX.YY   ← κ(M_loco) küçükse iyi
  LOCO (M_loco + SVD truncate)       :    XX.YY   ← regularize edilmiş
```

Bu sonuç tablosu, LOCO yaklaşımının k-mod'a göre üstünlüğünü doğrudan kanıtlar.

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
| `g0` | Modüle quad gradyanı [T/m] | 0.20 |

### Simülasyon Kontrolü

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `t2` | Toplam süre [s] | 0.001 |
| `dt` | Zaman adımı [s] | 1e-11 |
| `return_steps` | Kaydedilen veri noktası sayısı | 10000 |
| `poincare_quad_index` | Poincaré kesiti konumu (−1 = her hücre) | −1 |
| `dev0`, `y0` | Tune ölçümü için başlangıç tepmesi [m] | 1e-5 |

### Hata Modeli

| Parametre | Açıklama |
|-----------|----------|
| `error_quad_index` | Tek quad hatası indeksi (−1 = devre dışı) |
| `error_quad_dy/dx` | Tek quad kaçıklığı [m] |
| `quad_random_dy/dx_max` | Tüm quadlara rastgele hata genliği [m] |
| `quad_random_seed` | Tekrarlanabilirlik için tohum sayısı |
| `error_dipole_index` | Tek dipol hatası indeksi (−1 = devre dışı) |
| `error_dipole_tilt` | Tek dipol tilt açısı [rad] |
| `dipole_random_tilt_max` | Tüm deflektörlere rastgele açısal hata [rad] |
| `dipole_random_seed` | Dipol tilt rastgele tohumu |
| `quad_random_tilt_max` | Tüm quadlara rastgele tilt açısı [rad] |
| `quad_random_tilt_seed` | Quad tilt rastgele tohumu |

### BPM Hata Modeli

| Parametre | Açıklama |
|-----------|----------|
| `bpm_noise_sigma` | Tur-tura bağımsız Gaussian gürültü std [m]; tepki matrisi inşasında ve geri çatımda etkindir |
| `bpm_offset_sigma` | BPM başına sabit sistematik ofset std [m]; yalnızca geri çatım COD'unda görülür (matriste farkta iptal olur) |
| `bpm_offset_seed` | Ofset vektörü rastgele tohumu |

---

## 12. Kurulum ve Çalıştırma

### Gereksinimler

```bash
pip install numpy scipy matplotlib
```

### Derleme

```bash
# Linux
g++ -shared -o lib_integrator.so -fPIC -O3 integrator.cpp

# macOS
clang++ -dynamiclib -o lib_integrator.dylib -O3 integrator.cpp
```

### Adım Adım Kullanım

**Adım 1 — Normal simülasyon** (fizik sonuçları + tune + COD):
```bash
python run_simulation.py
```

**Adım 2 — Görselleştirme:**
```bash
python plot_results.py
# → simulasyon_sonuclari.png
```

**Adım 3 — Tepki matrisini hesapla** (bir kez yap, sonuçları sakla):
```bash
# Seri (tek çekirdek):
python build_response_matrix.py

# Paralel (önerilen, çekirdek-1 worker):
python build_response_matrix.py --workers 7   # 8 çekirdekli makine için
python build_response_matrix.py --workers 11  # 12 çekirdekli makine için

# Konfigürasyon 1 (nominal g1): R_dy_1, R_dx_1, R_tilt_1
# Konfigürasyon 2 (g1×1.02):   R_dy_2, R_dx_2, R_tilt_2
# Birleşik LOCO matrisi:        M_loco (96×96)
# Geriye dönük uyumluluk:       R_dy, R_dx
# (params.json'da dipole_random_tilt_max>0 ise k-mod matrisleri de inşa edilir)
```

**Adım 4a — Quad-only geri çatım testi:**
```bash
python test_reconstruction.py
# → reconstruction_test.npz ve konsol istatistikleri
```

**Adım 4b — LOCO geri çatım testi** (quad dy + dipol tilt eş zamanlı):
```bash
python test_loco_reconstruction.py
# → loco_reconstruction_test.npz ve konsol istatistikleri
```

**Adım 4c — K-modülasyon ve LOCO karşılaştırması** (referanssız geri çatım):
```bash
python test_kmod_reconstruction.py
# → kmod_reconstruction_test.npz
# → konsola: K-mod ile LOCO yan yana hata RMS karşılaştırması
```

### Tipik Parametre Değişiklikleri

Rastgele quad hataları eklemek için `params.json`'da:
```json
"quad_random_dy_max": 0.0005,
"quad_random_dx_max": 0.0005,
"quad_random_seed": 42
```

Tek bir quada hata vermek için:
```json
"error_quad_index": 5,
"error_quad_dy": 0.0003
```
