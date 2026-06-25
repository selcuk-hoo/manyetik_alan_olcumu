# Berry Fonksiyoneli: Sahte EDM'yi Kapalı Yörüngeden İleri-Öngörmek

> **Amaç (tek cümle):** Sahte EDM (false EDM, dS_y/dt) ölçülen kapalı yörüngenin
> bir fonksiyoneli mi, ve bu fonksiyonel empirik/ML ile bulunabilir mi?
> **Durum (2026-06): UMUT VERİCİ.** ML ile fonksiyonelin formu bulundu
> (alan-ağırlıklı bilineer çarpım, LOO-R²≈0.88, permütasyon-doğrulamalı); analitik
> türetme ve daha çok config ile pekiştirme açık.
>
> Bu dosya kendi-içinde okunur; başka bir sohbet doğrudan buradan başlayabilir.
> Veri ve scriptler: `berry_data/`.

---

## 1. Bağlam ve neden önemli

Drift-monitör çalışması (`drift_monitor/`) kapalı-yörüngeden hizalama izler ama
**no-go**'ya çarpar: sahte EDM'yi süren simetrik artık orbit-INVERSİYONUNA kördür
(κ≈193). KRİTİK AYRIM (bu oturumda kuruldu): **no-go bir *inversiyon* sınırıdır**
(R⁻¹ ile kaçıklık geri-çatımı). Sahte EDM ise kaçıklığın değil **kapalı yörüngenin**
fonksiyonelidir, ve BPM yörüngeyi doğrudan ölçer. Dolayısıyla f'i ölçülen yörüngeden
**ileri yönde** öngörmek inversiyona girmez → no-go onu yasaklamayabilir — **eğer
doğru fonksiyonel bilinirse.** Bu dosya o fonksiyoneli arar.

(Tam bağlam: `orbit_ileri_olcum.md`. Drift-monitör + literatür: `drift_monitor/`,
`literatur/`.)

## 2. Kurulu fizik

- Sahte EDM ∝ **dx·dy geometrik (Berry) faz**, σ² ölçekleme
  (`false_edm_harmonic_sinir.md §13`; Hacıömeroğlu & Semertzidis 2017,
  arXiv:1709.01208).
- Mekanizma kapalı yörüngenin çarpımıdır ama **basit ⟨x_CO·y_CO⟩ ortalaması YANLIŞ
  proxy'dir** (antisim configlerde orbit büyük, ⟨xy⟩≈0, f büyük). f, global ortalama
  değil **alan-ağırlıklı/yapısal** bir integraldir.
- Ölçüm reçetesi (doğrulanmış): 4D kapalı yörüngede tek ideal parçacık + model-fit
  seküler eğim (`berry_data/false_edm_4d.py`, `false_edm_mode_scan.py`).

## 3. Bu oturumdaki bulgular (özet)

1. **Ham sahte EDM'yi ANTİSİMETRİK (orbit-görünür) domine eder** (~37×). Simetrik
   kısım düzeltme-sonrası artıktır.
2. **No-go = inversiyon sınırı.** Per-quad modülasyon onu atlar (üniform koşulluluk,
   spread 1.3×) ama Omarov/BBA bölgesidir.
3. **⟨xy⟩ proxy sağlam değil** (antisim'de çöker). Berry yönlü-alan Σ(x1y2−x2y1)
   en tutarlı *basit* lead (~−0.5, her iki grupta).
4. **ML kernel-fit fonksiyoneli buldu** (§4) — asıl ilerleme.

## 4. ML ile fonksiyonel keşfi (ASIL SONUÇ)

**Model:** f bilineer kabul edilir, f ≈ Σ_{i,j} W_ij x_i y_j (x_i,y_j = yörüngenin
i. noktadaki değeri). Lokallik kısıtıyla:
- **(1) uniform:** f = Σ x_i y_i  (= ⟨xy⟩, 1 parametre)
- **(2) ağırlıklı diagonal:** f = Σ w_i x_i y_i  (alan-ağırlıklı yerel çarpım)
- **(3) + Berry komşu:** + Σ a_i (x_i y_{i+1} − x_{i+1} y_i)  (yönlü-alan)

**Yöntem:** ridge + LOO-CV + **permütasyon testi** (f karıştırılıp aynı fit → null
dağılımı; gerçek R² null'un çok üstündeyse sahte-uyum değil). 64 config (run1+run2).

**Sonuç (N=12/16/24 noktada tutarlı):**

| Model | LOO-R² | permütasyon null (maks) | Verdikt |
|------|------|------|------|
| (1) uniform ⟨xy⟩ | 0.26–0.36 | — | zayıf |
| **(2) ağırlıklı diagonal** | **0.88–0.89** | ~0.16–0.21 | **GERÇEK** |
| (3) + Berry komşu | 0.91–0.94 | ~0.12–0.16 | GERÇEK (marjinal +) |

**Yorum:** Sahte EDM, yörüngenin **alan-ağırlıklı yerel çarpımı** Σ w_i x_i y_i ile
LOO-R²≈0.88 öngörülüyor — uniform ⟨xy⟩'den (0.3) büyük sıçrama, ve **permütasyon-
temiz** (bu oturumda daha önce düştüğümüz sahte-R² tuzağı DEĞİL). Berry off-diagonal
marjinal katkı verir; baskın yapı ağırlıklı diagonaldır. Öğrenilen w_i ağırlıkları
belirli bir azimutta yoğunlaşıyor (kuplajın yerel kaynağı — deflektör/quad?
fiziksel yorum açık).

→ **Cevap: EVET, orbit→sahte-EDM bilineer, öğrenilebilir bir fonksiyoneldir; ML
(yapı-dayatan, permütasyon-testli) onu buluyor.**

## 5. Açık problemler / sonraki adımlar

1. **Daha çok config** (~birkaç yüz) ile R²'yi ve w_i'yi pekiştir (64 az; tracker
   pahalı, paralel üret). Şu an 64 config / 16 ağırlık — permütasyon geçti ama
   istatistik dar.
2. **w_i'nin fiziksel yorumu:** ağırlık nerede yoğun? Lattice alan profiliyle
   (deflektör/quad konumları) eşleştir → fonksiyonelin fiziksel anlamı.
3. **Analitik türetme:** Berry fazını Thomas-BMT spin denkleminden türetip
   Σ w_i x_i y_i formunu (ve w_i'yi) *kanıtla*. ML keşif aracı; son söz analitik.
4. **48-BPM + gürültü dayanıklılığı:** öğrenilen fonksiyonel gerçekçi BPM
   örneklemesiyle (48 nokta, 1μm gürültü) ne kadar korunur?
5. **Post-düzeltme rejimi:** rutin orbit-düzeltmesi açıkken (antisim telafi edilir)
   simetrik artığın bu fonksiyonelle öngörülebilirliği — asıl deney rejimi.

## 6. Reprodüksiyon (`berry_data/`)

- `run1_data.npz` (24 config), `run2_data.npz` (40: 28 sim + 12 antisim) — her
  config: `f`, tam kapalı yörünge `xo`,`yo` (480 nokta), kaçıklıklar `dx`,`dy`.
- `kernel_fit.py` — bu §4 analizi (tracker gerekmez, npz'den çalışır):
  `python3 berry_data/kernel_fit.py` (yolları /tmp yerine berry_data'ya çevir).
- `false_edm_4d.py` + `false_edm_mode_scan.py` — doğrulanmış sahte-EDM estimator
  (git'ten restore: commit `5cba757`, `41b1c6a~1`; `_BASE` repo köküne yamalı).
- `run1_gen.py` / `run2_gen.py` — yeni config üretimi (tracker; `bash
  build_integrator.sh` sonrası; ~birkaç dk/config, 4 worker paralel).

**Çalıştırma notu:** scriptlerde `sys.path` ve `_BASE` repo köküne işaret etmeli;
tracker için `build_integrator.sh` ile `lib_integrator.so` derlenmeli.
