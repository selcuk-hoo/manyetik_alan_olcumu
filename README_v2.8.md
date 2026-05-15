# v3.1 — Analitik Tepki Matrisi ile Quad Hizalama Geri Çatımı

Bu sürüm, k-modülasyon yöntemiyle quad hizalama hatalarının (dx, dy) geri
çatımında **simülasyon tabanlı** tepki matrisi ile **analitik FODO Twiss tabanlı**
tepki matrisi yaklaşımlarını karşılaştırır. Hedef: analitik yöntemin doğruluk
sınırlarını ve hangi bilinmeyenlerin geri çatımı bozduğunu nicel olarak
belirlemektir.

> Bu README **yalnızca `v3.1` etiketinde bulunur**; `main` dalındaki
> README.md kuram özetini ve genel kullanımı içerir.

---

## 1. Motivasyon

Klasik kapalı yörünge (COD) tabanlı k-modülasyon ölçümü, iki gradyan
ayarında BPM okumalarının farkını alıp `ΔCOD = ΔR · Δquad` lineer
denklemini çözerek hizalama hatalarını tahmin eder.

Soru: **ΔR matrisini her quad'a sırayla simülasyonla pertürbe etmek yerine,
β(s) ve φ(s) Twiss fonksiyonlarından doğrudan analitik olarak hesaplayabilir
miyiz?**

Avantajları:
- Hızlı (97 koşum yerine bir hücre transfer matrisi),
- Şeffaf (hata kaynaklarını analitik olarak izlemek mümkün),
- Beta-fonksiyonu hatasının etkisini doğrudan görmemize izin verir.

Bu sürüm bu soruya **kesin sayısal cevap** verir.

---

## 2. Analitik Tepki Matrisi Formülü

Quad `j`'nin Δq kadar yer değiştirmesi `θ_j = K_j · L_q · Δq_j` kadar bir
dipol kicki yaratır. Bu kickin BPM `i`'deki kapalı yörünge etkisi standart
betatron formülüyle verilir:

```
R[i,j] = (√(βᵢ·βⱼ) / 2sin(πQ)) · KL_j · cos(|φᵢ−φⱼ| − πQ)
```

İki gradyan konfigürasyonu (nominal `g_nom`, pertürbe `g_pert = 1.02·g_nom`)
için ayrı ayrı hesaplanan `R₁` ve `R₂`'nin farkı k-mod sinyalinin tepki
matrisidir:

```
ΔR = R(g_pert) − R(g_nom)
```

`Δy_BPM = ΔR · Δy_quad` (ve `x` için analog) ilişkisi, ölçülen BPM farkından
quad ofsetlerini SVD ile geri çatmamızı sağlar.

### 2.1. İşaret Kuralı

| Düzlem | Quad | KL işareti | Sebep |
|--------|------|-----------|-------|
| Dikey (y) | QF | `−K·L` | QF dikeyde defokalize: +Δy → aşağı kick |
| Dikey (y) | QD | `+K·L` | QD dikeyde fokalize: +Δy → yukarı kick |
| Radyal (x) | QF | `+K·L` | QF radyalde fokalize: +Δx → dışa kick |
| Radyal (x) | QD | `−K·L` | QD radyalde defokalize: +Δx → içe kick |

### 2.2. Hücre Yapısı

Tek bir FODO hücresi (entegratör `integrator.cpp` ile uyumlu):

```
QF(L_q) → drift(L_d) → ARC1(L_def) → drift(L_d) → QD(L_q)
       → drift(L_d) → ARC2(L_def) → drift(L_d) → (sonraki QF)
```

`L_def = π·R₀/nFODO` arc deflektör uzunluğu. **Elektrik deflektör magic
momentumda yalnızca radyal alan üretir; dikey ve radyal Twiss'e katkısı
saf drift olarak modellenir.** (Bu nokta önemli — daha önce yanıltıcı
"deflektör fokalizasyona katkı yapar" yorumu yapılmıştı; bu doğru
değildir.)

Periyodik Twiss çözümü tek hücre matrisinin `μ = arccos((M[0,0]+M[1,1])/2)`
özdeğerinden elde edilir, ardından her quad girişine `propagate_twiss` ile
ilerletilir.

---

## 3. Uygulama: `analytic_kmod.py`

Ana fonksiyonlar:

| Fonksiyon | Görev |
|-----------|-------|
| `compute_Brho` | Magic momentumda `Bρ = p_magic/0.299792458` [T·m] |
| `quad_matrix(K,L)` | Kalın quad 2×2 transfer matrisi (K>0 cos/sin, K<0 cosh/sinh) |
| `propagate_twiss(M,β,α)` | Twiss fonksiyonlarını M elemanından geçirir |
| `phase_step(M,β,α)` | Faz artışı `arctan2(M[0,1], M[0,0]·β − M[0,1]·α)` |
| `compute_twiss_at_quads` | Her quad girişinde β, φ, Q |
| `signed_KL` | İşaretli K·L vektörü (Bölüm 2.1) |
| `build_R_analytic` | Yukarıdaki formülle R matrisi |
| `build_analytic_dR` | `R₂ − R₁` k-mod farkı |
| `reconstruct(dR, Δ)` | Truncated SVD (threshold `s[0]·1e-10`) |

### 3.1. Kritik Birim Düzeltmesi

`build_response_matrix.py`'nin ürettiği `R_dy_1.npy`, `dR_dy.npy`
dosyalarında BPM değerleri `cod_data.txt` üzerinden **mm** birimiyle gelir,
fakat `delta_q` **m** cinsindendir. Bu, sakladığımız `dR_dy.npy` matrisini
`mm/m = 1000 × (m/m)` birimine taşır.

Analitik formül `m/m` (boyutsuz) birimindedir. Geri çatımda simülasyon
matrisini analitik sinyalle uyumlu kullanmak için yükleme sırasında
`1e-3` ile çarparak m/m'ye çevrilir:

```python
dR_sim_y = np.load("dR_dy.npy") * 1e-3   # mm/m → m/m
```

Bu düzeltme yapılmadığında simülasyon ΔR ile geri çatımda hata RMS
≈ 7 mm seviyesinde patlama yaşandı (kappa amplifikasyonu + birim
uyumsuzluğu); düzeltmeyle birlikte aşağıdaki nihai sonuçlara ulaşıldı.

---

## 4. Test Konfigürasyonu

`params.json` (özet):

```json
{
  "nFODO": 24,           "quadLen": 0.4,        "driftLen": 2.0,
  "R0": 41.4,            "g1": 0.21,
  "quad_random_dy_max": 1e-4,   "quad_random_dx_max": 1e-4,
  "quad_random_seed": 42,
  "dipole_random_tilt_max": 0,  "quad_random_tilt_max": 0,
  "bpm_noise_sigma": 0,         "bpm_offset_sigma": 0,
  "kmod_single_quad_index": -1, "kmod_single_quad_eps": 0.10
}
```

Toplam 48 quad (24 QF + 24 QD), ±100 μm rastgele dy/dx, tilt ve BPM
gürültüsü kapalı — saf yöntem testi.

### 4.1. Analitik FODO Twiss Çıktıları

```
Bρ      = 2.3374 T·m
K_abs   = 0.08984 m⁻²
Q_y     = 2.3617    Q_x = 2.3617
β_y min = 41.17 m   β_y max = 76.43 m
β_x min = 41.17 m   β_x max = 76.43 m
kappa(dR_dy) = 2.756×10⁴
kappa(dR_dx) = 2.756×10⁴
```

Yüksek kappa — k-modülasyon ΔR'nin doğal özelliğidir: pertürbasyon yalnızca
%2 (Δg/g), yani matris kendi içinde "küçük fark"tır.

---

## 5. Nihai Sonuçlar

### 5.1. Sinyal Bütçesi

```
Sinyal (ideal ΔR·dy)        RMS = 52.9 μm
Gerçek Δy (simülasyon)      RMS = 52.2 μm
Kirlilik (model − simülasyon) RMS = 0.9 μm  (~%1.7)
```

Yani analitik formül simülasyondan %1.7 sapıyor. Bu sapmanın kaynağı:
sayısal integrasyon artığı, yörünge betatron ortalamasının tam
kapanmaması ve yüksek-mertebe küçük nonlineer terimler.

### 5.2. Geri Çatım Karşılaştırması

| Yöntem | `dy` hata RMS | `dy` corr | `dx` hata RMS | `dx` corr |
|--------|---------------|-----------|---------------|-----------|
| **Analitik ΔR**       | 75.25 μm | 0.5207 | 524.6 μm | 0.0978 |
| **Simülasyon ΔR**     |  1.11 μm | 0.9998 |   2.16 μm | 0.9991 |
| Öz-tutarlılık (anal.) |   ~0 μm  | 1.0000 |    ~0 μm | 1.0000 |

**Simülasyon ΔR ile geri çatım pratik olarak mükemmel** (corr 0.9998).
Analitik formül %52 korelasyonda kalıyor; `dx` için durum daha kötü
(corr 0.10) — bunun nedeni dispersiyon ve radyal düzlemdeki çift-yönlü
fokalizasyon-defokalizasyon yapısının analitik modelin lineer
yaklaşımıyla daha yüksek hata vermesi.

### 5.3. Beta-Fonksiyonu Hatası Duyarlılığı

Gerçek halka tasarımdan farklı bir gradyanda çalışıyorsa, modelin
kullandığı β değerleri sapar. Sonuç (geri çatımda hâlâ tasarım β kullanılıyor):

| `g_err` | dy hata RMS | dy corr | dx hata RMS | dx corr |
|---------|-------------|---------|-------------|---------|
| +0%   |   0.00 μm | 1.0000 |   0.00 μm | 1.0000 |
| +1%   |   8.20 μm | 0.9903 |   4.18 μm | 0.9965 |
| +2%   |  15.35 μm | 0.9688 |   8.34 μm | 0.9866 |
| +5%   |  33.53 μm | 0.8897 |  22.54 μm | 0.9182 |
| +10%  |  69.45 μm | 0.7349 |  65.18 μm | 0.6652 |
| +20%  | 907.4 μm  | 0.1849 | 1055.0 μm | 0.1487 |

%5'e kadar gradyan hatası analitik yöntemi hâlâ kullanılabilir tutar;
%10'da korelasyon 0.73'e düşer; %20'de yöntem tamamen çöker.

---

## 6. Neden Analitik Metod Yetersiz?

Analitik ΔR formülü matematiksel olarak doğru (öz-tutarlılık testi
corr = 1.0000). Asıl sorun **kondisyon sayısının amplifikasyon davranışı**:

```
‖x̂ − x_true‖  ≲  κ(ΔR) · (‖ε‖ / ‖ΔR · x_true‖) · ‖x_true‖
```

Buradaki κ = 2.756×10⁴, ε ise model hatası. %1.7 model hatasıyla
beklenen göreli geri çatım hatası `O(κ·0.017) / N_rank ~ O(1)` yani
geri çatım fiilen bozulur. Bu, gözlenen `≈75 μm` hata ile uyumludur
(gerçek dy RMS ≈ 57 μm).

Yani **ΔR matrisi öyle ince ki, %2'lik bir model uyumsuzluğu bile
geri çatımı silip atıyor**.

Hata kaynakları sırasıyla:

1. **Sayısal entegratör artığı** — RK4 + adaptif adım yörüngenin tam
   kapalı olmamasına yol açıyor. Tipik artık birkaç ppm.
2. **Betatron ortalamasının tamamlanmaması** — `t_end` sınırlı sayıda
   tur içeriyor; analitik formül sonsuz tur ortalamasını varsayar.
3. **Yüksek-mertebe küçük nonlineerlikler** — kombine quad+arc bölgesinde
   küçük chromatic terimler analitik lineer modelde yok.

Bu kaynakların hiçbiri "bug" değil; analitik lineer modelin doğal
sınırları. Onları kaldırmak için ya daha rafine bir analitik model (örn.
chromatic ΔR düzeltmesi) ya da doğrudan simülasyon ΔR gerekir.

### 6.1. Geriye Dönük Bir Gözlem

Simülasyon ΔR ile elde edilen `corr = 0.9998` öz-tutarlılığı kanıtlar:
gerçek bir deneyde de **k-mod farkı** ölçülürse, ölçülen ΔR matrisiyle
geri çatım çalışır. Yani yöntemin kendisi sağlıklı — sorun yalnızca
"tepki matrisi nereden geldi?" sorusunda. Analitik kaynak yetmiyor;
deneysel veya yüksek-doğruluklu simülasyon gerekiyor.

---

## 7. Pratik Sonuç ve Tavsiye

| Kullanım | Önerilen yöntem |
|----------|-----------------|
| Hızlı tasarım keşfi, β ve κ tahmini | **Analitik ΔR** |
| Beta-hata, tilt, gürültü duyarlılığı | **Analitik ΔR** |
| Gerçek BPM verisinden quad ofset geri çatımı | **Simülasyon ΔR** (zorunlu) |
| Hibrit (LOCO-ölçülmüş β + analitik ΔR) | Gelecek araştırma konusu |

Analitik yöntem **ne işe yarar?** — Modeli soyutlar, fiziksel anlamayı
keskinleştirir, gradyan/β duyarlılığını saniyede hesaplar. Ama κ
limitasyonu nedeniyle **ölçüm verisi ile son adıma kadar
götürülemez**. Geri çatımda nihai matris **mutlaka simülasyon
tabanlı** olmalıdır.

---

## 8. Çalıştırma Sırası

```bash
# 1. Simülasyon tabanlı ΔR (97 koşum, ~2-3 dk paralel)
python build_response_matrix.py --workers 7

# 2. Analitik vs simülasyon karşılaştırması, beta-hata taraması
python analytic_kmod.py
```

Çıktı dosyaları:
- `R_dy_1.npy`, `R_dx_1.npy`, `R_dy_2.npy`, `R_dx_2.npy` — gradyan başına R
- `dR_dy.npy`, `dR_dx.npy` — k-mod farkı (mm/m birimde!)
- `analytic_kmod_result.npz` — Twiss + analitik ΔR + geri çatım vektörleri

---

## 9. v2.6 → v3.1 Değişiklik Özeti

- **yeni:** `analytic_kmod.py` — FODO Twiss tabanlı analitik ΔR, geri çatım,
  öz-tutarlılık ve β-hata test paketi.
- **düzenleme:** `analytic_kmod.py` içinde simülasyon ΔR yükleme bölümünde
  `mm/m → m/m` birim dönüşümü.
- **yeni:** Bu README (yalnızca `v3.1` etiketinde).
- **değişmeyen:** `build_response_matrix.py`, `integrator.cpp`,
  `test_kmod_reconstruction.py` — eski davranış korunuyor.

---

## 10. Bundan Sonra

- **Hibrit yöntem:** LOCO'dan ölçülen β fonksiyonlarını analitik formüle
  besleyerek, simülasyon koşumu yapmadan ölçüm-tabanlı ΔR çıkarmak.
- **Chromatic düzeltme:** Analitik formüle `ξ·δp/p` chromatic terimi
  ekleyerek %1.7 sapmanın bir kısmını yakalamak.
- **Kondisyon iyileştirme:** ΔR yerine R(g_pert)/R(g_nom) oranıyla
  çalışmak (μ ölçeklemesinden kaynaklanan singülerliği bastırabilir).
