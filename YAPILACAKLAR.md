# YAPILACAKLAR — pEDM Quad Hizalama İzleme Projesi

Proje durumu ve öncelikli işler. Betikler `params.json` ve `test_params.json`
üzerinden kontrol edilir; testleri kendiniz çalıştırıp parametreleri
inceleyebilirsiniz.

---

## Durum özeti

| Test | Betik | Durum |
|---|---|---|
| 1 — Düzenlileştirme karşılaştırması | `compare_regularization.py` | ✅ Tamamlandı |
| 2 — Uzaysal transfer fonksiyonu | `mode_transfer.py` | ✅ Tamamlandı |
| 3 — Yatay model ters-suç kontrolü | `kxarc_sensitivity.py` | ✅ Tamamlandı |
| 4 — Drift modu gösterimi | `drift_monitor_sim.py` | ✅ Tamamlandı |
| 5 — BPM ofset drift robustluk | `bpm_offset_drift_sim.py` | ✅ Tamamlandı |
| 6 — Üç yöntem adil karşılaştırma | `test6_fair_comparison.py` | ✅ Tamamlandı |
| 6b — Hata kaynağı ayrıştırma | (test6 uzantısı) | 🔲 Bekliyor |
| 7 — Tilt seviyesi taraması | `test7_tilt_scan.py` | 🔲 Bekliyor |
| 8 — Örgü model hatası taraması | `test8_model_error.py` | 🔲 **Kritik** |

---

## Öncelikli iş: Test 8 — Örgü modeli hatasının drift moduna etkisi

### Neden kritik?

Drift modunun 6-7 μm hassasiyeti, $R^{-1}$'in doğruluğuna tamamen bağlıdır.
Gerçek halkada $R_{\text{gerçek}} = R + \delta R$ olur; estimator hatası:

$$\widehat{\delta q} = (R)^{-1}(\mathbf{y}-\mathbf{y}_0)
= \delta q + R^{-1}\delta R \cdot \delta q + \text{gürültü}$$

$R^{-1}\delta R$ terimi model hatasının drift tahminine ne kadar sızdığını
belirler. Bu ölçülmeden "6 μm başarılı" iddiası kırılgandır.

NIM A referee'si bunu ilk major revision isteği olarak soracaktır.

### Test 8 tasarımı

**Tarama parametresi:** `beta_error_rms` — Twiss β fonksiyonuna eklenen
rölatif hata, $\delta\beta/\beta \sim \mathcal{N}(0, \sigma_\beta)$.

**test_params.json'a eklenecek blok:**
```json
"test8": {
    "beta_error_rms_values": [0.0, 0.01, 0.02, 0.05, 0.10, 0.20],
    "tune_error_values":     [0.0, 0.001, 0.003, 0.005, 0.01],
    "N_realizations": 20,
    "BPM_NOISE": 1e-6,
    "BPM_OFFSET": 50e-6,
    "DRIFT_RMS": 10e-6
}
```

**Beklenen çıktı:** Her $\sigma_\beta$ ve $\Delta Q$ için drift modu RMS
hatası. İki grafik: RMS vs $\sigma_\beta$ ve RMS vs $\Delta Q$.

**Başarı kriteri:** pEDM örgüsünde LOCO ile sağlanabilecek model doğruluğu
($\sigma_\beta \lesssim 1\%$, $\Delta Q \lesssim 0.003$) aralığında drift
modu 10 μm hedefini tutturuyor mu?

---

## Test 7 — Tilt seviyesi taraması

### Motivasyon

Test 6'da tilt'ler 0.2 mrad sabit tutuldu. Gerçek makinede tilt seviyesi
değişkendir; hangi tilt RMS'te drift modu 10 μm'ü aşar?

### Test 7 tasarımı

**Tarama:** `quad_tilt_rms` ve `dipole_tilt_rms` birer birer taranır.

**test_params.json'a eklenecek blok:**
```json
"test7": {
    "quad_tilt_rms_values":   [0.0, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
    "dipole_tilt_rms_values": [0.0, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
    "N_realizations": 20,
    "BPM_NOISE": 1e-6,
    "BPM_OFFSET": 50e-6,
    "DRIFT_RMS": 10e-6
}
```

**Beklenen çıktı:** Tilt seviyesi vs drift modu RMS hatası eğrisi.
10 μm'ü aşan tilt seviyesi makale için bir operasyonel sınır verir.

---

## Test 6b — Hata kaynağı ayrıştırması

### Motivasyon

Test 6'da üç tahmin yönteminin toplam performansı ölçüldü; ama "drift
modunun 6 μm hatası nereden geliyor?" sorusu yanıtsız kaldı. Bu kontrol
testleri o soruyu cevaplayacak.

### Yöntem

Simülasyon verisi aynı; sadece post-processing (ofset ve gürültü ekleme)
farklılaştırılır. Dört senaryo:

| Senaryo | BPM gürültü | BPM ofset | Tilt |
|---|---|---|---|
| Tam (mevcut) | 1 μm | 50 μm | 0.2 mrad |
| Gürültüsüz | 0 | 50 μm | 0.2 mrad |
| Ofsetsiz | 1 μm | 0 | 0.2 mrad |
| İkisi de yok | 0 | 0 | 0.2 mrad |

Tilt ayrıştırması için ek re-sim gerekir (pahalı); önce post-processing
senaryolarını tamamlayın.

**Beklenen:** Gürültüsüz ve ofsetsiz durumda A/C estimatörleri de
iyileşmeli (ofset zaten ΔR ile iptal oluyor; sıfır gürültüde iyi çalışır).
Drift modu B'nin 6 μm'ünün kaçı gürültüden, kaçı tiltten geliyor görülür.

---

## Makale için kalan işler

### Eksik şekiller (mevcut testlerden üretilebilir)

- [ ] **SVD spektrum karşılaştırması:** $R$ ve $\Delta R$'nin singular
  değerleri yan yana — §2.4'teki $\kappa(\Delta R)/\kappa(R)\approx 1/\varepsilon$
  ölçeklemesini sayısal olarak gösterir. `fodo_lattice.py`'den R matrisi
  alınıp `np.linalg.svd` uygulanır, ε taraması yapılır.
- [ ] **Test 6 görselleştirmesi:** Mevcut `test6_fair_comparison.png`
  var; yeterli mi değerlendirin.

### Omarov PRD referansı

10 μm hizalama toleransını türeten Omarov vd. PRD makalesinin tam
künyesi ve ilgili eşitlik/bölüm numarası makaleye eklenecek.

### Sonra: NN karşılaştırması

Test 8 ve 7 tamamlandıktan sonra, NN yaklaşımı aynı senaryo altında
karşılaştırılacak. Ortak kıyaslama altyapısı: Test 6 senaryosu + Test 8
(model hata) + Test 7 (tilt) kombinasyonu.

---

## Dosya haritası (güncel)

| Dosya | Rol |
|---|---|
| `params.json` | Hızlandırıcı parametreleri |
| `test_params.json` | Test-spesifik parametreler |
| `fodo_lattice.py` | Twiss, analitik R matrisi, FFT/direct tersleme |
| `reconstruct.py` | İki-gradient mutlak rekonstrüksiyon |
| `compare_regularization.py` | Test 1 |
| `mode_transfer.py` | Test 2 |
| `kxarc_sensitivity.py` | Test 3 |
| `drift_monitor_sim.py` | Test 4 |
| `bpm_offset_drift_sim.py` | Test 5 |
| `test6_fair_comparison.py` | Test 6 |
| `metot.md` | Pedagojik açıklama |
| `makale-taslagi-2.md` | Formal makale taslağı |
