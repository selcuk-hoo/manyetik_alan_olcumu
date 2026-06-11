# YAPILACAKLAR — pEDM Quad Hizalama İzleme Projesi

> Son güncelleme: 2026-06-11. Trim analiz serisi (yörünge + spin + taban
> analizi) tamamlandı; sıradaki ağırlık makale yazımı ve spin kademesinin
> uçtan uca gösterimi.

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

---

## 🔲 Aktif: Sıradaki İşler

### 1. Spin kademesinin uçtan uca gösterimi (öncelik: yüksek)
Yörünge trimi sonrası kalan ~10⁻⁴ artığı boylamsal spin geri-beslemesiyle
10⁻⁵ altına indiren tam zincir simülasyonu:
`yörünge trimi → spin ölç → c_k tabanında trim → tekrar` döngüsünün
simetrik içerik dahil çalıştığının gösterilmesi. Mevcut altyapı yeterli
(`test_b_iterative_trim.py` + `test_orbit_trim.py` birleşimi).

### 2. Makale konsolidasyonu (öncelik: yüksek)
`makale_tr.tex` trim bölümü büyüdü; ayrı bir trim makalesi olarak bölme
kararı (bkz. `MAKALE_POTANSIYELI.md`).

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
