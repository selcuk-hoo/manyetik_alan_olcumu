# YAPILACAKLAR — pEDM Quad Hizalama İzleme Projesi

> Son güncelleme: 2026-06-10. Önceki versiyon (v2.7 iki-quad k-mod planı)
> artık geçerli değil; proje o aşamayı geçti.

---

## ✅ Tamamlananlar

### Altyapı
- [x] **C++ GL4 entegratör** (`integrator.cpp`) — simplektik, Thomas-BMT spin takibi
- [x] **Python köprüsü** (`integrator.py`) — ctypes arayüzü, `FieldParams` sınıfı
- [x] **Tepki matrisi** (`build_response_matrix.py`) — paralel, 48×48 R_dy / R_dx
- [x] **CLEAN algoritması** (`reconstruction.py`) — iteratif harmonik geri çatım
- [x] **`params.json`** — tek kaynak konfigürasyon, `g2` anahtarı eklendi
- [x] **`CLAUDE.md`** — kod tabanı yapısı, iş akışı, tuzaklar

### Sistematik testler
- [x] `test_bpm_noise.py` — BPM elektronik gürültüsü taraması
- [x] `test_bpm_offset.py` — BPM statik ofset etkisi, diferansiyel iptal
- [x] `test_quad_gradient.py` — gradyan pertürbasyon büyüklüğü taraması
- [x] `test_quad_tilt.py` — kuadrupol tilt → x-y kuplaj etkisi
- [x] `test_combined_systematics.py` — tüm sistematikler birlikte (N=200 MC)
- [x] `test_reconstruction_quality.py` — SVD gözlenebilirlik + rank limitleri
- [x] `test_kmod_reconstruction.py` — bilinen Δq → TSVD/Fourier/LS kıyaslaması

### False EDM analizi
- [x] `false_edm_mode_scan.py` — dSy/dt vs Fourier modu k=1..24 taraması
- [x] `harmonic_orbit_correction.py` — kademeli harmonik düzeltme, %30 etkinlik
- [x] `false_edm_correction_test.py` — düzeltme stratejisi doğrulaması
- [x] `test_false_edm_modes.py` — tek-mod + kombinasyon senaryoları
- [x] `test_cross_correlation.py` — 3×3 c_ij matrisi, rezonans işaret yapısı, lineer ölçekleme
- [x] `false_edm_harmonic_sinir.md` — §1-9: CO=True/False farkı, betatron kirlenmesi,
  eşit-katkı, side band çürütmesi, çapraz korelasyon, optimizasyon stratejisi

### Yeni testler
- [x] `test_kick_correction.py` — kick sayısı vs false EDM (Test A)
  - Ana bulgu: N=2..24 korrektör orbiti %50-95 azaltır ama false EDM yalnızca ~%15 düşer
  - N=48 (tüm quadlar korrektör) → sıfır false EDM
  - **Orbit minimizasyonu ≠ dSy/dt minimizasyonu** — farklı optimizasyon hedefleri

---

## 🔲 Aktif: Sıradaki Test

### Test B — Harmonik Kombinasyon ile False EDM İptali
**Dosya:** `test_harmonic_cancellation.py` (henüz yok)
**Öncelik:** Şimdiki — kısa (~1-2 saat) ve teorik tahminle doğrudan karşılaştırılabilir.

k=2 (+) ve k=3 (−) işaret farkından yararlanarak bilerek eklenen k=3
orbit bileşeniyle k=2 kaynaklı false EDM'yi iptal etmek:

- **Adım 1:** A₂=10μm sabit, A₃ ∈ [−20, +20] μm taraması → dSy/dt=0 geçişini bul.
  Teorik beklenti: $A_3^* \approx +(c_2/c_3) \times A_2 = +(13.84/6.16) \times 10\,\mu\text{m} \approx 22.5\,\mu\text{m}$
- **Adım 2:** İptal noktası etrafında hassasiyet: δA₃ = 1 μm → kaç rad/s hata?
- **Adım 3:** Rassal hizalama (k=4..10 kirleticiler mevcut) — iptal ne kadar sağlam?

```python
# Anahtar parametreler
A2       = 1e-5
c_ratio  = 13.84 / 6.16    # cross-corr matrisinden (test_cross_correlation.py)
t2       = 8e-4
co_turns = 36
```

---

## 🔲 Ertelendi

| Madde | Neden ertelendi | Öncelik |
|---|---|---|
| Makale revizyonu: rezonans işaret yapısı ve CO=True/False | Yazılmayı bekliyor | Yakın |
| `false_edm_harmonic_sinir.md` §10-11 ekleme | Yazılmayı bekliyor | Yakın |
| Kurum bilgisi + teşekkür bölümü (`makale_tr.tex`) | Yazara özgü | Yayın öncesi |
| Omarov PRD referansı — 10 μm tolerans kaynağı | Kütüphane erişimi gerekiyor | Yayın öncesi |
| Gerçek makine verisiyle doğrulama | Prototip halka gerekiyor | Uzun vadeli |
| Tepki matrisi kalibrasyonu (LOCO/beta-BBA) | Ölçüm altyapısı gerekiyor | Uzun vadeli |
| Test 8: örgü model hatası (β-beat, tune error) | Mevcut altyapıyla yapılabilir | Orta |
| Çoklu yörünge ortalaması ile BPM ofset bastırımı | Mevcut altyapıyla yapılabilir | Orta |
| Yatay düzlem ve kuadrupol tilti genişlemesi | `R_dx` matrisi mevcut | Düşük |
