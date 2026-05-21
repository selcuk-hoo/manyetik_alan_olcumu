# YAPILACAKLAR — pEDM Quad Hizalama İzleme Projesi

> **Yeni sohbet için not:** Bu dosya kendi başına yeterlidir; başka dosya
> okumadan plan kavranabilir. Detaylı bağlam için `metot.md` ve
> `makale-taslagi-2.md`'ye bakılabilir, ama plan için zorunlu değil.

---

## Aktif iş: İki-quad k-modülasyonu (v2.7 yapısından devam)

### Bağlam

pEDM halkasında 48 kuadrupolün hizalama hatalarını BPM okumalarından
geri çatmaya çalışıyoruz. Sistem:

```
y = R · Δq + b + n
```

- `R` (48×48): tepki matrisi (Courant-Snyder), κ(R)≈160 — iyi koşullu
- `Δq`: quad hizalama hataları (aradığımız)
- `b`: BPM ofsetleri (yapısal, statik) — **temel sorun**
- `n`: BPM gürültüsü

`b` bilinmediği için doğrudan `R·Δq = y` çözülemez. Çözüm: gradyan
modülasyonu ile iki ayrı ölçüm alıp farkı çözmek:

```
Δy = y₁ − y₂ = ΔR · Δq    (b iptal olur)
```

burada `ΔR = R(g_perturbed) − R(g_baseline)`.

### Önceki denemeler ve eksik kalan

| Versiyon | Modülasyon | Sonuç |
|---|---|---|
| v2.5/v2.6 | **Tüm 48 quad** aynı anda %2 (uniform k-mod) | Çalışıyor: 6.6 μm RMS, korelasyon 0.993 |
| v2.7 | **Tek quad** %10 (`quad_dG[j₁] = 0.10`) | Çatladı: κ(ΔR)≈4.4×10⁸, 46/96 mod kayıp |
| **(eksik)** | **İki quad** birlikte, faz uzayında ~90° ayrı | **Hiç denenmedi** |

v2.1 notunda "iki quad pertürbasyonu: hücre 0 + faz uzayında ~90° ötede
bir quad" ertelendiği belirtilmişti. v2.7 altyapısı (`quad_dG[48]` per-quad
fraksiyonel gradyan sapması) bunu desteklediği halde deneme yapılmadı.

### Hipotez

Tek-quad ΔR'nin çatlamasının fiziksel nedeni: tek bir quada faz uzayında
~90° uzaktaki BPM'ler için `cos(|φᵢ − φⱼ₁| − πQ) ≈ 0`. O BPM'lerin
satırları neredeyse sıfır → ΔR satırca rank-eksik → `b`'nin o BPM'lerdeki
bileşeni iptal olamaz → rekonstrüksiyon çatlar.

İkinci bir quadı j₁'in faz uzayında ~90° karşısına koyarsak: j₁'e kör
olan BPM'ler j₂'ye duyarlı olur, j₂'ye kör olan BPM'ler j₁'e duyarlı
olur. Hiçbir BPM her ikisine birden kör olmaz → ΔR'de sıfır satır
kalmaz → `b` her BPM'de iptal olur.

Beklenti: κ(ΔR) tek-quad'a göre 3-4 mertebe iner, rekonstrüksiyon
çalışmaya başlar.

---

## Uygulama planı — küçük adımlar

### Adım 1: `params.json` — yeni anahtarlar

Mevcut:
```json
"g0": 0.2,
"g1": 0.21,
"kmod_single_quad_index": -1,
"kmod_single_quad_eps": 0.10
```

Yeni şema (46 quad g0, biri g1, biri g2):
```json
"g0": 0.2,
"g1": 0.21,
"g2": 0.21,
"kmod_quad1_index": 0,
"kmod_quad2_index": 12
```

- `g0`: nominal gradyan (46 quad)
- `g1`: birinci modüle edilen quad'ın gradyanı (index = `kmod_quad1_index`)
- `g2`: ikinci modüle edilen quad'ın gradyanı (index = `kmod_quad2_index`)
- İkisinin de index'i `-1` → uniform k-mod (eski davranış)
- Yalnız `kmod_quad2_index = -1` → tek-quad mod (v2.7 davranışı)

Geriye dönük uyumluluk için `kmod_single_quad_index/eps` da okunabilir
bırakılabilir (alias).

### Adım 2: `integrator.py` / C++ tarafı

**Değişiklik gerekmez.** v2.6'da eklenen `quad_dG[48]` arayüzü zaten
per-quad fraksiyonel sapma kabul ediyor. Sadece iki entry'i nonzero
yapmak yeterli.

### Adım 3: `build_response_matrix.py`

Tek-quad mantığını genişlet. Şu anki kod (yaklaşık):

```python
if single_quad >= 0:
    quad_dG_pert = np.zeros(n_q)
    quad_dG_pert[single_quad] = eps
```

Yeni:

```python
quad_dG_pert = np.zeros(n_q)
if j1 >= 0:
    quad_dG_pert[j1] = (g1 - g0) / g0
if j2 >= 0:
    quad_dG_pert[j2] = (g2 - g0) / g0
```

İki ölçüm:
- Baseline: tüm 48 quad g0 → `R_dy_1`, `R_dx_1`
- Perturbed: 46 quad g0, j₁ → g₁, j₂ → g₂ → `R_dy_2`, `R_dx_2`
- `dR = R_2 − R_1` (iki sütunu nontrivyal şekilde perturbe)

### Adım 4: `test_kmod_reconstruction.py`

Tek değişiklik: yeni params anahtarlarını oku, mod seçimini buna göre
yap. Geri kalan rekonstrüksiyon mantığı (ΔR · Δq = Δy çöz) değişmez.

### Adım 5: İlk koşum ve teşhis

Parametreler:
```
j1 = 0       (ilk QF)
j2 = 12      (faz uzayında ~90° ileri; 24 FODO hücresi için nFODO/2 civarı)
g1 = g2 = 0.21    (%5 modülasyon her birinde)
```

Çıktılar:
- `κ(dR_dy)`, `κ(dR_dx)` — tek-quad'la karşılaştır (4.4×10⁸'den ne kadar düştü?)
- Rekonstrüksiyon RMS hatası (uniform 6.6 μm referans, tek-quad ~m mertebesi)
- Per-BPM `|ΔR_{i,:}|` satır normu — sıfıra yakın satır kaldı mı?

### Adım 6: j₂ taraması (opsiyonel, sonuçtan sonra)

j₁=0 sabit, j₂ ∈ {1, 2, ..., 47} taraması. Her j₂ için κ(ΔR) ve RMS.
Beklenen minimum: j₂ ≈ 12 (90°) ve j₂ ≈ 36 (270°) civarında.

---

## Karar verilmesi gerekenler

1. `g1 = g2` mi tutalım, asimetrik mi (ör. `g1=0.21, g2=0.19`) deneme şansı verelim mi?
2. Geriye uyumluluk: `kmod_single_quad_*` anahtarları silinsin mi, alias olarak kalsın mı?
3. j₁'i `0` (ilk QF) mu sabitliyoruz, kullanıcı seçsin diye parametre mi bırakıyoruz? (Şu anki şema seçim bırakıyor — iyi.)

---

## v2.7'den miras: kod haritası

| Dosya | Rol |
|---|---|
| `integrator.cpp` / `integrator.py` | GL4 simplektik entegratör; `quad_dG[48]` arayüzü mevcut |
| `build_response_matrix.py` | R_dy, R_dx, ΔR inşası; **bu dosya değişecek** |
| `test_kmod_reconstruction.py` | ΔR · Δq = Δy çözümü; **bu dosya değişecek** |
| `run_simulation.py` | Poincare, tune, COD diagnostiği |
| `scan_quad_tilt.py`, `scan_qtilt_contamination.py` | Doğrulama araçları |
| `params.json` | **Bu dosya değişecek** |

> **Not:** Klasörde `compare_regularization.py`, `mode_transfer.py`,
> `test6_fair_comparison.py` gibi yeni refactor sonrası betikler var.
> Bunlar makale analiz altyapısı içindir; iki-quad k-mod çalışması için
> v2.7 yapısı (`build_response_matrix.py` + `test_kmod_reconstruction.py`)
> kullanılacak. v2.7 koduna `git checkout v2.7 -- build_response_matrix.py
> test_kmod_reconstruction.py` ile dönülebilir.

---

## Sonraki sohbette ilk adım

> "v2.7'den `build_response_matrix.py` ve `test_kmod_reconstruction.py`
> dosyalarını geri yükle, sonra `params.json`'a `g2`, `kmod_quad1_index`,
> `kmod_quad2_index` anahtarlarını ekle. Önce Adım 1'i bitirip bana göster."

---

## Ertelendi (iki-quad sonrası)

Aşağıdaki testler bu klasördeki refactor sonrası altyapıyla (analitik R)
yapılacak; iki-quad çalışması bitince geri dönülür:

- **Test 8** — Örgü model hatası (β-beat, tune error) drift modu hassasiyetine
  etkisi. *Makale için kritik.*
- **Test 7** — Tilt seviyesi taraması.
- **Test 6b** — Hata kaynağı ayrıştırması (gürültü/ofset/tilt).
- **SVD spektrum şekli** — κ(ΔR)/κ(R) ≈ 1/ε ölçeklemesinin sayısal doğrulanması.
- **Omarov PRD** — 10 μm hizalama toleransının kaynak künyesi makaleye eklenecek.
