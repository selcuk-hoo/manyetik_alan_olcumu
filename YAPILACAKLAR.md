# YAPILACAKLAR — pEDM Quad Hizalama İzleme Projesi

> Son güncelleme: 2026-06-11. `makale_trim_tr.tex` taslağı konsolide edildi
> (polarimetre bütçesi dahil). g₀ tarama testi çalışıyor; makale için
> hakem yorum yanıtlarına odaklanılıyor.

---

## ✅ Tamamlananlar

### Altyapı
- [x] **C++ GL4 entegratör** (`integrator.cpp`) — simplektik, Thomas-BMT spin takibi
- [x] **Python köprüsü** (`integrator.py`) — ctypes arayüzü, `FieldParams` sınıfı
- [x] **İkinci motor** (`integrator2.cpp` + `lib_integrator2.so`) — alternatif
  topoloji QF-d-QD-d-ARC-d; eski koddan bağımsız
- [x] **Tepki matrisi** (`build_response_matrix.py`) — paralel, 48×48 R_dy / R_dx
- [x] **CLEAN algoritması** (`reconstruction.py`) — iteratif harmonik geri çatım
- [x] **`params.json`** — tek kaynak konfigürasyon
- [x] **`CLAUDE.md`** — kod tabanı yapısı, iş akışı, tuzaklar (2026-06-11 güncel)

### K-modülasyon geri çatım hattı (v1–v3 dönemleri)
- [x] Sistematik testler: BPM gürültüsü/ofseti, gradyan, tilt, birleşik (N=200 MC)
- [x] `test_kmod_reconstruction.py` — TSVD/Fourier/LS kıyaslaması
- [x] SVD gözlenebilirlik + rank limitleri

### Sahte EDM ve spin-sürülü trim serisi (v4.x → bu dal)
- [x] `false_edm_mode_scan.py` — c_k haritası k=1..24, üç arka planda evrensellik
- [x] `test_b_*` serisi — iteratif ölç-trimle döngüsü ~1000×; CO=False kesin
  sonuçları (doğrusallık %0.0, tek atış 2×10⁷×); BPM etkisi (tabana oturur);
  fırlatma bağımlılığı (demet ortalaması geçerli); rastgele desen + çift
  kuadratür trim (faz problemi 2 ölçümde çözülür)
- [x] `test_orbit_trim.py` — BPM-sürülü EDM-kör kaba kademe; varyant C
  (k=1..4) seed=321'de 101×
- [x] `test_orbit_trim_seeds.py` — 5 seed: artık ~2.5×10⁻⁴ rad/s RMS (taban)
- [x] `test_orbit_mode_correlation.py` — Gram matrisi (korelasyon ≤%1.1),
  sızıntı (≤0.002), kazanç yasası G_k=24.8/|5.03−k²| k=7..12'de öngörü olarak
  doğrulandı; fit kesimi eşik meselesi, korelasyon değil
- [x] `test_radial_spin.py` — radyal polarizasyon EDM'yi 3.5×10⁶× bastırır;
  kaçıklık sinyalinin %5'i Ω_z kanalında kalır (EDM-kör sistematik kanalı adayı)
- [x] `test_symm_vs_antisym.py` — **taban kanıtı:** 23 boyutluk simetrik içerik
  COD kazancı ~3, spin kuplajı ~12× bastırılmış ama sıfır değil →
  1.0×10⁻⁴ rad/s taban
- [x] `test_symm_basis_fit.py` — simetrik modlar fit'e eklenemez (kazanç eşiği);
  deneme 9× kötüleştirdi
- [x] `test_new_topology.py` + `find_stable_gradient.py` — deflektörü quad
  çiftinin dışına alan hücre mevcut gradyanla kararsız; aynı ton g≈0.5 T/m ister
- [x] Belgeler: `false_edm_harmonic_sinir.md` §1–12.16,
  `trim_yontemi_pedagojik.md` §1–9, `makale_tr.tex` trim bölümü
- [x] **§13 — "kuadratik nerede" araştırması** (`false_edm_harmonic_sinir.md`
  §13.1–13.8, `test_dxdy_geometric_phase.py`): Omarov'un kuadratik sahte-EDM'i
  **dx·dy geometrik-faz çapraz kanalıdır** (iki düzlem birlikte, σ²); proje
  yalnız dy-only (doğrusal birinci-derece) ölçüyordu. Çok-seed RMS σ^2.01,
  10μm'de 3.3×10⁻⁶ → Omarov ~10⁻⁵ ile **3× içinde**. dx·dy için demet=ideal
  (CO=True temsilci); dy-only'de emittans kanalı 220×. Madde 2 model-fit
  estimator'ı (`measure_dSy_dt_model`) betatron sızıntısını temizler. Çürütülen
  hipotezler ve yöntem kaydı §13.8'de — **bu yollardan tekrar geçmeyin.**
- [x] `makale_trim_tr.tex` — bağımsız trim makalesi taslağı: tüm semboller
  tanımlı, 9 şekil çağrıldığı yerde, polarimetre istatistik modeli ve
  süre bütçesi (Müller 2020 referansla), CW−CCW EDM güvenlik kanıtı

---

## 🔲 Aktif: Sıradaki İşler

### 1. g₀ tarama testi (öncelik: yüksek — çalışıyor)
`test_g0_scan.py` — eşik teorisinin g₀=0.15/0.20/0.25 T/m üç kafes
noktasında genelliğini test eder. Şu an g₀=0.15 tohum bataryası aşamasında.
Çıktı: `test_g0_scan.json`, `test_g0_scan.png`.
Sonuç gelince §8 "Tartışma" içine ve CLAUDE.md'ye eklenecek.

### 2. Makale kalan \todo maddeleri (öncelik: yüksek)
`makale_trim_tr.tex` içinde kalan eksikler:
- `\todo{kurum bilgisi}` — yazara özgü
- `\bibitem{todo-pedm}` — pEDM tasarım raporu
- `\bibitem{todo-omarov}` — Omarov ve ark. PRD
- `\bibitem{todo-bba}` — BBA/k-modülasyon referansları
- `\bibitem{todo-loco}` — LOCO tepki matrisi referansı
- `\todo{teşekkür ve fon bilgisi}` — yazara özgü

### 3. Spin kademesinin uçtan uca gösterimi (öncelik: orta)
Yörünge trimi sonrası kalan ~10⁻⁴ artığı boylamsal spin geri-beslemesiyle
10⁻⁵ altına indiren tam zincir simülasyonu:
`yörünge trimi → spin ölç → c_k tabanında trim → tekrar` döngüsünün
simetrik içerik dahil çalıştığının gösterilmesi. Mevcut altyapı yeterli
(`test_b_iterative_trim.py` + `test_orbit_trim.py` birleşimi).

---

## 🔲 Ertelendi

| Madde | Neden ertelendi | Öncelik |
|---|---|---|
| Yeni topoloji kafes yeniden tasarımı (g≈0.5 T/m + kapalı yörünge türetimi) | Tam lattice redesign; kazanım belirsiz | Düşük |
| Radyal polarizasyonla trim kanalı | Simetrik içeriğe o da kör (Ω_z ∝ β_y); kazanım sınırlı | Düşük |
| Kurum bilgisi + teşekkür bölümü (`makale_tr.tex`) | Yazara özgü | Yayın öncesi |
| Omarov PRD referans değerlerinin birebir teyidi | Kütüphane erişimi | Yayın öncesi |
| Gerçek makine verisiyle doğrulama | Prototip halka gerekiyor | Uzun vadeli |
| RF + sekstüpol açıkken trim doğrusallığı | Mevcut altyapıyla yapılabilir | Orta |
| x-yönü kaçıklıkları ve quad tilt çapraz terimleri | `R_dx` matrisi mevcut | Orta |
| Zamanla sürüklenen hizalamada izleme (tracking) kipi | Tasarım gerekiyor | Orta |
