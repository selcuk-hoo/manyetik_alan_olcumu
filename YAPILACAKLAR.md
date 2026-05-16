# YAPILACAKLAR — Yeni Rekonstrüksiyon Yöntemi (Ayrıntılı Plan)

Bu belge, manyetik alan ölçüm projesinin bir sonraki aşamasında ne yapılacağını,
neden yapılacağını ve nasıl uygulanacağını hem kavramsal hem matematiksel düzeyde
açıklamaktadır. Okuyucunun doğrusal cebir ve Fourier analizi temellerine sahip
olduğu varsayılmaktadır.

---

## 1. İleri Model: Makine Ne Ölçtürüyor?

Bir parçacık hızlandırıcısında BPM (Beam Position Monitor) dedektörleri, demetin
sapmasını ölçer. Bir kuadrupol mıknatıs tam merkezinden kaymışsa, o mıknatıstan
geçen demet sapar. Bu sapma, halkayı dolaşarak tüm BPM'lere yayılır.

İdeal, ince-lens (thin-lens) yaklaşımında, i. BPM'de ölçülen demet sapması şöyle
yazılır:

```
y_i = Σ_j  R_ij · Δq_j  +  b_i
```

Burada:
- `y_i`  : i. BPM'nin ölçtüğü demet konumu [m]
- `Δq_j` : j. kuadrupolün merkezden yanal kayması [m]
- `R_ij` : j. kuadrupolün kaymasının i. BPM'ye etkisi (yanıt matrisi elemanı) [boyutsuz değil, m/m = boyutsuz]
- `b_i`  : i. BPM'nin sabit kalibrasyon hatası (ofseti) [m]

Matris notasyonuyla:

```
y = R · Δq + b
```

Hedef, y (ölçüm) ve R (hesaplanan model) biliniyorken Δq'yu (kuadrupol
kaymalarını) bulmak. Eğer b=0 ve R tam doğruysa bu basit bir lineer sistem çözümü.
Asıl zorluk b≠0 ve R'nin modelden hesaplanmasının gerçeklikten sapmasıdır.

---

## 2. Yanıt Matrisi R'nin Analitik Formu

### 2.1 Twiss Formülü

Klasik akseptans mekanik sonucu (Courant-Snyder teorisi): j noktasındaki bir yatay
ince tekmeden (kick) θ_j, i noktasında şu sapma yaratır:

```
Δx_i = √(β_i · β_j) · cos(|φ_i − φ_j| − π·Q)
        ────────────────────────────────────────  · θ_j
                    2 · sin(π·Q)
```

Burada:
- `β_i`, `β_j` : i ve j noktalarındaki beta fonksiyonu değerleri [m]
- `φ_i`, `φ_j` : i ve j noktalarındaki Courant-Snyder fazları [rad]
- `Q`          : Betatron tunu (halka boyunca toplam faz ilerlemesi / 2π)

Kaymış bir kuadrupol, ince-lens yaklaşımında şu tekmeyi üretir:

```
θ_j = −KL_j · Δq_j
```

Burada `KL_j = K_j · L_q` integre kuadrupol gücüdür (`K` [1/m²], `L_q` [m]).

Bunu birleştirirsek yanıt matrisi elemanı:

```
R_ij = −√(β_i · β_j) · cos(|φ_i − φ_j| − π·Q)
        ───────────────────────────────────────── · KL_j
                     2 · sin(π·Q)
```

### 2.2 Transfer Matrisi ile Twiss Hesabı

`β` ve `φ`'yi analitik olarak hesaplamak için her optik elemanın 2×2 transfer
matrisi kullanılır.

**Serbest drift (L uzunluğunda boşluk):**
```
M_drift = | 1   L |
          | 0   1 |
```

**İnce-lens kuadrupol (odaklayan, yatay düzlem):**
```
M_QF = | 1    0  |
       | -1/f  1  |
```

**İnce-lens kuadrupol (dağıtan, yatay düzlem):**
```
M_QD = | 1    0  |
       | +1/f  1  |
```

Odak uzaklığı: `f = 1 / (K · L_q)`

Dikey düzlem için odaklayan/dağıtan rolleri yer değiştirir (QF → dağıtır, QD → odaklar).

Bir FODO hücresinin transfer matrisi (QF → drift → QD → drift → QF sırası):

```
M_cell = M_QFhalf · M_drift · M_QD · M_drift · M_QFhalf
```

Bu matrisin izi (trace) faz ilerlemesini verir:

```
cos(μ) = (M_cell[0,0] + M_cell[1,1]) / 2
```

Simetrik FODO hücresi için (α=0 noktalarında):

```
β = M_cell[0,1] / sin(μ)
```

Betatron tunu: `Q = N_hücre · μ / (2π)` (N_hücre hücre sayısı).

Faz ilerlemesi hücre hücre:

```
φ_i = i · μ     (i = 0, 1, ..., N−1)
```

---

## 3. Neden Eski Yöntem (ΔR Terslemesi) Başarısız Oldu?

### 3.1 ΔR Yaklaşımı

İki farklı kuadrupol gücünde (k-mod 1 ve k-mod 2) ölçüm alınır:

```
y₁ = R₁ · Δq + b
y₂ = R₂ · Δq + b
```

Fark alınırsa BPM ofseti `b` yokolur:

```
Δy = y₁ − y₂ = (R₁ − R₂) · Δq = ΔR · Δq
```

Şimdi `ΔR · Δq = Δy` sistemi çözülürse `b`-bağımsız bir sonuç elde edilir.
Teorik olarak çekici görünür.

### 3.2 Kondisyon Sayısı Sorunu

Bir lineer sistemin sayısal "sağlığı" kondisyon sayısıyla ölçülür:

```
κ(A) = σ_max / σ_min
```

Burada `σ_max` ve `σ_min`, `A` matrisinin en büyük ve en küçük tekil değerleridir.
`κ = 1` mükemmel durumu, `κ >> 1` ise tehlikeli bir büyütme faktörünü temsil eder:

```
‖δ(Δq)‖ / ‖Δq‖  ≤  κ(ΔR) · ‖δ(Δy)‖ / ‖Δy‖
```

Yani girişte yüzde ε'luk rölatif hata, çıkışta yüzde `κ·ε` hataya dönüşür.

Analitik hesaplamalar şunu gösterdi:

```
κ(ΔR) ≈ 27.560
```

Bu, girişteki %1'lik model hatasının çıkışta **%275'lik** rekonstrüksiyon hatasına
yol açması anlamına gelir. Simülasyon sonuçlarında da bu doğrulandı: `ΔR` üzerinden
yapılan rekonstrüksiyonda korelasyon 0.521 ve ortalama hata 75 μm çıktı.

### 3.3 Neden ΔR Bu Kadar Kötü Huylu?

`R₁` ve `R₂` birbirine yakın matrislerdir (kmod değişimi küçüktür). İki yakın
matrisin farkını almak, ortak büyük bileşenleri yokeder ve küçük fark bileşenlerini
öne çıkarır. Bu durum, sayısal analizde "catastrophic cancellation" (yıkıcı
sadeleşme) olarak bilinir. `ΔR`'nin tekil değerleri `R₁`'inkinden çok küçük olur
ve kondisyon sayısı fırlar.

---

## 4. Neden FODO Örgüsü Özel Bir Yapı Sunar?

### 4.1 Sirkülant Matris Tanımı

N×N bir matris C'ye sirkülant denir eğer her satır bir öncekinin sağa döngüsel
kaydırmış hâliyse:

```
C = | c₀    c_{N-1}  c_{N-2}  ...  c₁   |
    | c₁    c₀       c_{N-1}  ...  c₂   |
    | c₂    c₁       c₀       ...  c₃   |
    | ...                               |
    | c_{N-1}  c_{N-2}  ...   c₀       |
```

Matris tümüyle ilk satırı `[c₀, c₁, ..., c_{N-1}]` ile belirlenir.

### 4.2 FODO Örgüsü Neden Sirkülant Yanıt Matrisi Üretir?

İdeal FODO örgüsünde:
- Tüm hücrelerde beta fonksiyonu aynıdır: `β_i = β` (sabit)
- Faz ilerlemesi eşit adımlıdır: `φ_i = i · μ`
- Tüm kuadrupoller aynı `|KL|` değerine sahiptir (işaret kuadrupol tipine göre değişir)

Bu durumda yanıt matrisi elemanı:

```
R_ij = −β · KL · cos(|i−j| · μ − π·Q)
        ──────────────────────────────
               2 · sin(π·Q)
```

Bu ifade yalnızca `(i−j) mod N`'e bağlıdır. Dolayısıyla R matrisi **sirkülant**tır
ve ilk satırı:

```
r_k = −β · KL · cos(k · μ − π·Q) / (2 · sin(π·Q))    k = 0, 1, ..., N−1
```

### 4.3 Sirkülant Matrislerin Spektral Ayrışması

Sirkülant matrislerin en önemli özelliği: hepsi **aynı özvektör kümesine** sahiptir
— bu özvektörler DFT matrisinin sütunlarıdır.

```
C = F⁻¹ · diag(λ) · F
```

Burada:
- `F` : DFT matrisi, `F_kj = exp(−2πi·k·j/N) / √N`
- `λ_k` : C'nin k. özdeğeri, ilk satırın ayrık Fourier dönüşümü: `λ = DFT(r)`

### 4.4 DFT vs. FFT: Önemli Bir Ayrım

Belgede bu iki terim birbirinin yerine kullanılmamalıdır:

- **DFT (Ayrık Fourier Dönüşümü)** bir **matematiksel tanımdır**:
  ```
  X_k = Σ_{n=0}^{N-1}  x_n · exp(−2πi·k·n/N)
  ```
  Sirkülant matrisin özvektörleri **DFT matrisinin** sütunlarıdır. Özdeğerler
  ilk satırın **DFT'sidir**. Bu, matematiksel yapıyı tanımlayan ifadedir.

- **FFT (Hızlı Fourier Dönüşümü)** DFT'yi hesaplamak için kullanılan
  **algoritmanın adıdır**. Cooley-Tukey algoritması, naif O(N²) hesabı
  O(N log N)'e indirir. Çıktı matematiksel olarak DFT ile aynıdır.

**Pratik dil:** Teorik açıklamalarda "DFT" (yapı), kodda ve hesaplama
maliyetinden söz ederken "FFT" (algoritma) deriz. `np.fft.fft(x)` çağrısı
FFT algoritmasıyla DFT'yi hesaplar.

### 4.5 Spektral Ayrışmanın Pratik Faydaları

Bu ayrışma üç şeyi mümkün kılar:

1. **Verimli çarpım**: `C · x = IDFT(λ · DFT(x))` — FFT ile O(N log N)
2. **Verimli tersleme**: `C⁻¹ · y = IDFT(DFT(y) / λ)` — FFT ile O(N log N)
3. **Mod bazlı analiz**: Her k modu için ayrı "kazanç" faktörü `|λ_k|` incelenebilir

Karşılaştırma: genel matris çarpımı O(N²), terslemesi O(N³).

---

## 5. Yeni Yöntemin Algoritması

### 5.1 Normalize Edilmiş Sirkülant Yapı

Yanıt matrisinin tüm elemanlarında `√β_i · √β_j` çarpanı bulunur. Bu çarpanlar
matrisi teknik olarak sirkülant yapmaz (farklı hücreler farklı β değeri taşıyabilir
— ideal FODO'da aynıdır, ama gerçekte küçük sapmalar olur). Güvenli ve genel bir
yol, her iki tarafı `√β` ile normalize etmektir.

Normalize edilmiş ölçüm vektörü: `ỹ_i = y_i / √(β_i)`

Normalize edilmiş yanıt: `M_ij = R_ij / (√(β_i) · √(β_j))`

`M` matrisi, ideal FODO'da tam sirkülant, gerçek makinede yaklaşık sirkülant.

### 5.2 Beş Adımlı FFT Geri Dönüşümü

Tek bir kmod durumu için (`y = R · Δq`, `b=0` varsayımıyla):

**Adım 1 — Beta normalizasyonu:**
```
ỹ_i = y_i / √(β_i)
```
Bu adım, matristeki `√β_i` çarpanlarını soldan kaldırır.

**Adım 2 — Özdeğerleri hesapla:**
```
λ_k = DFT(ilk_satır_M)_k    k = 0, 1, ..., N−1
```
Pratikte FFT algoritması kullanılır: `lambda_arr = np.fft.fft(M[0, :])`.
`M`'nin ilk satırı: `M_{0,j} = −√(β_j) · cos(j·μ − πQ) / (2·sin(πQ))`.
Bu satırın DFT'si, sirkülant matrisin tüm spektral içeriğini taşır.

**Adım 3 — İşaretsiz KL faktörünü hesapla:**
```
KL_eff = K · L_q    (kmod gücüne göre, yatay/dikey için işaret dahil)
```

**Adım 4 — Fourier uzayında bölme:**
```
Q̃_k = FFT(ỹ)_k / (KL_eff · λ_k)
```
Bu, her Fourier modunu kendi özdeğeriyle geri çevirir. Küçük `|λ_k|` değerine
sahip modlar gürültüye duyarlıdır — bunlar "zor" modlardır.

**Adım 5 — Geri dönüşüm ve β denormalizasyonu:**
```
Δq_j = IFFT(Q̃)_j / √(β_j)
```

Bu beş adım, N×N matris terslemesini O(N³)'ten O(N log N)'e indirir ve sayısal
kararlılığı dramatik biçimde artırır.

### 5.3 İki-kmod Yaklaşımı ve BPM Ofset Yönetimi

BPM ofsetleri `b` bilinmediğinden iki kmod ölçümü ayrı ayrı işlenir:

**Ölçümler:**
```
y₁ = R₁ · Δq + b     (kmod-1, integre güç KL₁)
y₂ = R₂ · Δq + b     (kmod-2, integre güç KL₂)
```

**Ayrı geri dönüşümler:**
```
v₁ = R₁⁻¹ · y₁ = Δq + R₁⁻¹ · b
v₂ = R₂⁻¹ · y₂ = Δq + R₂⁻¹ · b
```

**Ortalama (ana rekonstrüksiyon):**
```
Δq̂ = (v₁ + v₂) / 2 = Δq + (R₁⁻¹ + R₂⁻¹)/2 · b
```

Ofset kalıntısı `(R₁⁻¹ + R₂⁻¹)/2 · b`, tek kmod durumuna kıyasla kısmi baskılanma
gösterir. Fourier uzayında bu, `b`'nin her k modunun `(1/λ₁ₖ + 1/λ₂ₖ)/2` ile
ağırlıklandırılması anlamına gelir.

**Fark (ofset bilgisi):**
```
v₁ − v₂ = (R₁⁻¹ − R₂⁻¹) · b
```

Eğer `R₁⁻¹ − R₂⁻¹` iyi kondisyonluysa (bu ayrıca kontrol edilmeli), bu denklem
`b`'nin çözümüne imkân tanır. Çözülen `b`, `Δq̂`'dan çıkarılarak ofset giderilmiş
rekonstrüksiyon elde edilebilir.

**Neden her R sirkülant ama ΔR değil?**
`R₁` ve `R₂` aynı Twiss yapısından (aynı β, φ, Q) türer; yalnızca `KL` katsayısı
değişir. `R₁ = KL₁ · M_c` ve `R₂ = KL₂ · M_c` burada `M_c` ortak sirkülant
matristir. Dolayısıyla:

```
ΔR = R₁ − R₂ = (KL₁ − KL₂) · M_c
```

Bu durumda `κ(ΔR) = κ(M_c)` — yani **aynı kondisyon sayısı!** Teorik olarak
ΔR yaklaşımı da aynı `κ(M_c)`'yi paylaşır. Uygulamada gözlemlenen büyük
`κ(ΔR)≈27000`, model hatası nedeniyle `R₁` ve `R₂`'nin farklı `M_c` matrislerine
sahip olmasından kaynaklanmaktadır — yani `ΔR` mükemmel bir skalar çarpım değil,
iki farklı sirkülant matrisin farkıdır. İki-kmod yaklaşımında `R₁` ve `R₂` ayrı
ayrı terslendiğinden model hatası bu şekilde birikmez.

---

## 6. Twiss Parametreleri: Analitik mi, Deneysel mi?

Yöntemin doğruluğu, R matrisinin gerçek makineye ne kadar iyi uyduğuna bağlıdır.
R matrisinin yapı taşları üç Twiss niceliğidir: betatron tunu Q, beta fonksiyonu
β ve faz ilerlemesi μ. Bunların kaynağı kritik bir tasarım kararıdır.

### 6.1 Saf Analitik Yaklaşımın Problemi

`params.json`'dan okunan örgü konfigürasyonu (`g0`, `g1`, `quadLen`, `driftLen`,
`nFODO`) üzerinden transfer matrisleriyle hesaplanan Twiss parametreleri,
**modelin** Twiss değerleridir — gerçek makinenin değil. Aralarındaki fark şu
nedenlerle ortaya çıkar:

- Kuadrupol kalibrasyon hataları (gerçek `g` değeri okunan değerden farklıdır)
- Sextupol etkisi, dipol bozucu manyetik alanları, fringe field düzeltmeleri
- Sıcaklık ve mekanik tolerans kaynaklı dalgalanmalar

Bu fark `(β_gerçek − β_model)/β_model` mertebesinde tipik olarak %1 – %5
düzeyindedir. Doğrudan R'ye girdiğinden rekonstrüksiyon hatasına aynı oranda
katkı yapar.

### 6.2 Tune Q'nun Rolü ve Kritik Önemi

Yanıt matrisinin paydasında `sin(πQ)` bulunur. Q yarı-tam sayıya (0.5, 1.5, …)
yaklaştıkça payda sıfırlanır ve R patlar. Q'daki küçük bir belirsizliğin etkisi
makinenin çalışma noktasına bağlıdır:

```
dR/R  ≈  −π · ΔQ · cot(πQ)
```

Q = 0.25 gibi makul bir çalışma noktasında `cot(π·0.25) = 1`, dolayısıyla
`ΔQ = 0.01` yaklaşık `dR/R ≈ 3.1%` hataya yol açar. Q yarı-tam sayıya yaklaştıkça
bu duyarlılık keskin biçimde artar.

**Sonuç:** Q, deneysel değerine yerleştirilmesi gereken en kritik parametredir.
Neyse ki tune ölçümü hızlandırıcı tesislerinde rutin ve hassas bir prosedürdür
(turn-by-turn BPM verisinin FFT'siyle binde bir doğrulukta okunur).

### 6.3 Önerilen Hibrit Yaklaşım

`fodo_lattice.py`'de Twiss kaynağı **isteğe bağlı** olacak. Varsayılan parametre
imzaları şöyle:

```python
compute_twiss_at_quads(config,
                       g,
                       plane,
                       Q_measured=None,       # None → analitik hesapla
                       beta_measured=None)    # None → analitik hesapla
```

- `Q_measured` verildiyse: bu değer doğrudan kullanılır
- `beta_measured` verildiyse: bu skaler veya N-vektörü R'ye giydirilir
- Hiçbiri verilmediyse: `params.json` + transfer matrisi formülleriyle analitik
  hesaplama yapılır

Bu yapı, hem analitik (Aşama A: ideal test) hem deneysel (Aşama C: gerçekçi
rekonstrüksiyon) modu desteklemeyi mümkün kılar.

### 6.4 Duyarlılık Analizinin Yeri

Q ve β'nın etkisi yalnızca "kullan/kullanma" sorusu değildir; her birinin
belirsizlik seviyesi de raporlanmalıdır. Aşama D'ye iki yeni alt test eklenir:

| Test | Tarama aralığı | Çıktı |
|------|----------------|-------|
| β belirsizliği | `δβ/β` = 0% – 5% RMS | Rekonstrüksiyon RMS hatası eğrisi |
| Q belirsizliği | `ΔQ` = ±0.001, ±0.005, ±0.01, ±0.02 | Rekonstrüksiyon RMS hatası eğrisi |

Bu eğriler makalede "yöntemi kullanmak için Twiss parametrelerinin hangi
hassasiyette bilinmesi gerekiyor" sorusuna nicel yanıt verecek.

---

## 7. Uygulama Planı

### 6.1 `fodo_lattice.py` — Temel Twiss Kütüphanesi

Bu dosya simülasyon programına (integrator) hiçbir bağımlılık taşımayacak.
Tüm hesaplamalar analitik formüllere dayanacak. Dışarıdan `import` edilebilir
bir modül olarak tasarlanacak.

**İşlevler ve sorumlulukları:**

```python
compute_Brho(E_MeV, mass_MeV, charge)
# → Manyetik rijitlik [T·m]
# Formül: Brho = p / (q) = √(E² − m²c⁴) / (q·c)
# Parçacık momentumundan Brho hesaplar; K = gradient / Brho için gerekli

quad_matrix(K, L_q, plane)
# → 2×2 ince-lens kuadrupol transfer matrisi
# plane='x': odaklayan (QF) veya dağıtan (QD) — K işaretine göre
# plane='y': rol tersine döner

drift_matrix(L)
# → 2×2 serbest drift transfer matrisi

propagate_twiss(M_cell)
# → (beta, alpha, mu) Courant-Snyder parametreleri
# M_cell'in izinden mu, M[0,1]/sin(mu) formülünden beta hesaplar

compute_twiss_at_quads(config, K, plane, Q_measured=None, beta_measured=None)
# → (beta_arr, phi_arr, Q) — her kuadrupol konumunda Twiss parametreleri
# config: params.json'dan gelen örgü konfigürasyonu
# phi_arr kümülatif faz, Q toplam betatron tunu
# Q_measured: verilirse analitik Q yerine kullanılır (deneysel mod)
# beta_measured: skaler veya N-vektörü; verilirse analitik β yerine kullanılır

signed_KL(N_fodo, K_abs, L_q, plane)
# → KL işaret dizisi [+KL, -KL, +KL, ...]
# Yatay düzlemde QF odaklar (+), QD dağıtır (-); dikey düzlemde tersi

build_circulant_matrix(phi, Q, beta, KL_arr)
# → N×N sirkülant yanıt matrisi R
# R_ij = -sqrt(beta[i]*beta[j]) * cos(|phi[i]-phi[j]| - pi*Q) * KL[j] / (2*sin(pi*Q))

fft_eigenvalues(R)
# → lambda_arr: R'nin N adet özdeğeri
# İlk satırı FFT'ler; sirkülant matris teorisine göre özdeğerler budur

fft_invert(y, beta, KL_eff, lambda_arr)
# → Δq: kuadrupol kaymaları
# Beş adımlı FFT geri dönüşümünü uygular
```

**Dahili tutarlılık testi (bu dosya doğrudan çalıştırıldığında):**

`R · fft_invert(R·Δq_test) == Δq_test` eşitliğinin makine hassasiyetinde sağlandığı
doğrulanmalı. Bu, FFT terslemesinin doğrudan `np.linalg.solve` ile aynı sonucu
verdiğini gösterir.

---

### 6.2 `spectral_inversion.py` — Ana Analiz Scripti

Dört analiz aşaması sırayla yürütülür. Her aşama kendi sonuçlarını basar ve
grafik üretir; sonraki aşama tamamlanmadan da çalışabilir.

---

#### Aşama A — İdeal Durum: Üst Sınır Testi

**Amaç:** Model ile gerçek makine mükemmel uyumda olsaydı rekonstrüksiyon ne
kadar iyi olurdu? Bu, yöntemin teorik tavanını belirler.

**Prosedür:**
1. `fodo_lattice.py` ile `R₁` matrisini hesapla (kmod-1 gücünde)
2. Rastgele Δq vektörü üret (örn. N(0, 100 μm) dağılımlı)
3. Simüle ölçüm: `y = R₁ · Δq` (BPM ofseti yok, gürültü yok)
4. Beş adımlı FFT geri dönüşümüyle `Δq̂` hesapla
5. `Δq̂ − Δq` farkının RMS'ini ve korelasyonu raporla

**Beklenti:** Bu aşamada hata, yalnızca kayan nokta (floating-point) sayısal
hassasiyetinden kaynaklanmalı; RMS hata < 1 nm düzeyinde olmalı.

---

#### Aşama B — Kondisyon Sayısı Haritası

**Amaç:** Her Fourier modunun (halka boyunca farklı dalga boylarının) ne kadar iyi
ya da kötü koşullu olduğunu görsel olarak anlayın.

**Prosedür:**
1. `R₁`, `R₂` ve `ΔR = R₁ − R₂` matrislerini hesapla
2. Her matris için özdeğerleri FFT ile hesapla: `λ_k = FFT(ilk_satır)`
3. Mod bazlı kondisyon sayısı: `κ_k = |λ_k|⁻¹ / max_j(|λ_j|⁻¹)`
   (büyük `|λ_k|⁻¹` → o mod zor rekonstrükte edilir)
4. Global kondisyon sayıları: `κ(R₁)`, `κ(R₂)`, `κ(ΔR)` (tekil değer oranları)

**Grafik:** k modu numarasına karşı `|λ_k|⁻¹` (log ölçek). R₁/R₂ eğrileri ve
ΔR eğrisi aynı grafikte; görsel karşılaştırma.

**Beklenti:** ΔR eğrisi, R₁ ve R₂ eğrilerinden en az birkaç kat daha yüksek tepe
göstermeli (daha kötü koşullu modlar).

---

#### Aşama C — İki-kmod Rekonstrüksiyonu (Simülasyon Makinesi)

**Amaç:** Simülasyonu "gerçek makine" olarak kullanarak, analitik modelden
farklı ölçümler alın. Model ile gerçek arasındaki küçük Twiss uyuşmazlığında
yöntemin ne kadar dayanıklı olduğunu test edin.

**Prosedür:**
1. Simülasyonu kmod-1 gücüyle çalıştır → `y₁_sim` al (mevcut `run_simulation.py`)
2. Simülasyonu kmod-2 gücüyle çalıştır → `y₂_sim` al
3. Analitik modelden `R₁` ve `R₂` matrislerini hesapla (`fodo_lattice.py`)
4. `v₁ = R₁⁻¹ · y₁_sim` (FFT ile)
5. `v₂ = R₂⁻¹ · y₂_sim` (FFT ile)
6. `Δq̂ = (v₁ + v₂) / 2`

**Karşılaştırma:** Δq simülasyondan bilindiği için `Δq̂ − Δq` RMS ve korelasyon
hesaplanır.

**BPM ofseti alt-testi:**
- Simülasyona sabit `b` vektörü ekle (örn. her BPM'de ±50 μm rastgele)
- `v₁ − v₂` ile `b` tahmini yap
- `b` tahmini ile orijinal `b` arasındaki korelasyonu raporla

---

#### Aşama D — Gürbüzlük Testi

**Amaç:** Yöntemin gerçekçi hata koşullarında nasıl davrandığını sistematik
biçimde ölçmek.

**Test edilecek hata kaynakları:**

| Hata türü | Parametre aralığı | Nasıl eklenir? |
|-----------|-------------------|----------------|
| Kuadrupol eğimi (tilt) | 0 – 2 mrad RMS | Simülasyona `tilt` parametresi ekle |
| BPM ölçüm gürültüsü | 0 – 20 μm RMS | `y`'ye `N(0, σ_noise)` vektörü ekle |
| Model beta hatası | 0 – 5% RMS | Analitik β'yı `β·(1 + δ)` ile boz |
| BPM ofseti | 0 – 200 μm RMS | `y`'ye sabit `b` ekle |

**Prosedür her hata seviyesi için:**
1. Simülasyonu hatayla çalıştır → ölçüm `y`
2. Analitik (ideal) modeli kullan → rekonstrüksiyon `Δq̂`
3. `‖Δq̂ − Δq‖_RMS` hesapla

**Çıktı:** Her hata türü için, hata seviyesi vs. RMS rekonstrüksiyon hatası eğrisi.

**Kritik eşik:** Hangi hata seviyesinde rekonstrüksiyon kalitesi kabul edilemez
hale geliyor? (korelasyon < 0.9 veya RMS hata > 10 μm eşikleri kullanılabilir)

---

## 8. Kontrol Noktaları ve Başarı Kriterleri

Her adımın tamamlandığını doğrulamak için aşağıdaki sayısal testler yapılmalıdır:

### 7.1 `fodo_lattice.py` için

| Test | Beklenen sonuç |
|------|----------------|
| `FFT_invert(R·x) == x` | RMS hata < 1e-12 (makine hassasiyeti) |
| `κ(R_circulant)` | 10 – 1000 aralığında (ΔR'dan çok küçük) |
| `Twiss karşılaştırması` | analitik β vs. simülasyondan elde edilen β farkı < %1 |

### 7.2 `spectral_inversion.py` Aşama A için

| Test | Beklenen sonuç |
|------|----------------|
| İdeal rekonstrüksiyon RMS | < 1 nm (yalnızca kayan nokta hatası) |
| Korelasyon | > 0.9999 |

### 7.3 Aşama C için (ana başarı kriteri)

| Test | Beklenen sonuç |
|------|----------------|
| Simülasyon-model rekonstrüksiyon RMS | < 10 μm |
| Korelasyon | > 0.95 |
| ΔR yöntemiyle karşılaştırma | En az 10× daha iyi RMS |

---

## 9. Dosya Yapısı

| Dosya | Rol |
|---|---|
| `fodo_lattice.py` | **[YENİ]** Twiss, sirkülant matris, FFT geri dönüşüm |
| `spectral_inversion.py` | **[YENİ]** Dört aşamalı ana analiz |
| `run_simulation.py` | Simülasyon arayüzü (değişmez) |
| `plot_results.py` | Simülasyon görselleştirme (değişmez) |
| `integrator.py` | C kütüphane köprüsü (değişmez) |
| `params.json` | Örgü parametreleri (değişmez) |
| `README_v2.8.md` | v2.8 analitik çalışmanın sonuçları |

---

## 10. Kodlama Sırası

1. `fodo_lattice.py` yazılır → dahili tutarlılık testi geçilir
2. `spectral_inversion.py` Aşama A eklenir → ideal test geçilir
3. Aşama B eklenir → kondisyon sayısı grafiği üretilir
4. Aşama C eklenir → simülasyon ile rekonstrüksiyon karşılaştırılır
5. Aşama D eklenir → gürbüzlük eğrileri çizilir
