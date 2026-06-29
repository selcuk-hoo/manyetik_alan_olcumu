# YAPILACAKLAR — pEDM Quad Hizalama İzleme Projesi

> Son güncelleme: 2026-06-19. İkinci makale adayı (drift izleme) canlandırıldı:
> `drift_monitor/` klasörü eklendi, Test 8 (β-beating) ve per-mod SVD analizi
> tamamlandı, `makale-taslagi-2.md` güncellendi.

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
- [x] **K-modülasyon geri çatımı** — TSVD/Fourier/LS kıyaslaması
- [x] SVD gözlenebilirlik + rank limitleri

> Not: aşağıdaki tamamlanan maddelerin keşif scriptleri 2026-06 temizliğinde
> kaldırıldı; mantıkları git geçmişinde ve ilgili `.md` günlüklerinde.

### Sahte EDM ve spin-sürülü trim serisi (v4.x → bu dal)
- [x] **Sahte-EDM mod taraması** — c_k haritası k=1..24, üç arka planda evrensellik
- [x] **Spin ölç-trim serisi** — iteratif ölç-trimle döngüsü ~1000×; CO=False kesin
  sonuçları (doğrusallık %0.0, tek atış 2×10⁷×); BPM etkisi (tabana oturur);
  fırlatma bağımlılığı (demet ortalaması geçerli); rastgele desen + çift
  kuadratür trim (faz problemi 2 ölçümde çözülür)
- [x] **Yörünge-trim** — BPM-sürülü EDM-kör kaba kademe; varyant C
  (k=1..4) seed=321'de 101×
- [x] **Yörünge-trim seed taraması** — 5 seed: artık ~2.5×10⁻⁴ rad/s RMS (taban)
- [x] **Mod korelasyonu / kazanç yasası** — Gram matrisi (korelasyon ≤%1.1),
  sızıntı (≤0.002), kazanç yasası G_k=24.8/|5.03−k²| k=7..12'de öngörü olarak
  doğrulandı; fit kesimi eşik meselesi, korelasyon değil
- [x] **Radyal polarizasyon kanalı** — EDM'yi 3.5×10⁶× bastırır;
  kaçıklık sinyalinin %5'i Ω_z kanalında kalır (EDM-kör sistematik kanalı adayı)
- [x] **Simetrik/antisim ayrışım** — **taban kanıtı:** 23 boyutluk simetrik içerik
  COD kazancı ~3, spin kuplajı ~12× bastırılmış ama sıfır değil →
  1.0×10⁻⁴ rad/s taban
- [x] **Genişletilmiş baz fit** — simetrik modlar fit'e eklenemez (kazanç eşiği);
  deneme 9× kötüleştirdi
- [x] **Alternatif topoloji + kararlı gradyan analizi** — deflektörü quad
  çiftinin dışına alan hücre mevcut gradyanla kararsız; aynı ton g≈0.5 T/m ister
- [x] Belgeler: `false_edm_harmonic_sinir.md`,
  `trim_yontemi_pedagojik.md`, `makale_trim_tr.tex` trim bölümü
- [x] **§13 — "kuadratik nerede" araştırması** (`false_edm_harmonic_sinir.md`
  §13.1–13.8): Omarov'un kuadratik sahte-EDM'i
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

## ✅ Tamamlanan (2026-06-19 eklemeleri)

### Drift izleme — ikinci makale adayı
- [x] **`drift_monitor/` klasörü** — `fodo_lattice.py`, `drift_monitor_sim.py`,
  `test8_betabeat.py`, `permode2.py`, `test_params.json` repoya alındı
- [x] **Test 4 (kalibrasyon-ref demo)** — 50 μm ofset altında 6.6 μm drift hassasiyeti;
  mutlak 197 μm → 29× iyileşme (`drift_monitor_sim.py`)
- [x] **Test 8 (β-beating sağlamlığı)** — %1 β-beating: 5.98→6.08 μm; %5→8.6 μm;
  hedef 10 μm altında güvenli (`test8_betabeat.py`)
- [x] **SVD per-mod analizi** — κ(R)=193; en kötü 8 mod %96 simetrik, 193× gürültü
  büyütmesi; no-go bağlantısı kuruldu (`permode2.py`)
- [x] **`makale-taslagi-2.md`** — Test 8 ve per-mod sonuçları ile güncellendi;
  dualite teoremi §2.4; 8 test özeti §3.9; simetrik sınır §5'te

---

## 🔲 Aktif: Sıradaki İşler

### 1. g₀ tarama testi (öncelik: yüksek)
Eşik teorisinin g₀=0.15/0.20/0.25 T/m üç kafes noktasında genelliğini test eder
(keşif scripti 2026-06 temizliğinde kaldırıldı; git geçmişinde). Sonuç gelince
§8 "Tartışma" içine ve CLAUDE.md'ye eklenecek.

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
(spin ölç-trim + yörünge-trim birleşimi; keşif scriptleri git geçmişinde / `/tmp`).

### 4. ⭐ "Akıllı düzeltme" — sahte-EDM'yi sıfırla, misalignment'ı DEĞİL (öncelik: stratejik, 2026-06-29 notu)

> **Fikrin özü (kullanıcı, 2026-06-29):** No-go bir *inversiyon* sınırıdır
> (COD → misalignment, simetrik alt-uzayda cond ~10⁴; bkz. v2.7 yeniden-incelemesi
> ve §19.2). Ama **nihai amacımız misalignment'ı ölçmek değil, sahte EDM'yi
> sıfırlamak.** O hâlde misalignment'ı tam sıfırlamaya çalışmak yerine, doğrudan
> sahte-EDM'yi null'layan quad düzeltmelerini ararız.

**Plan:**
1. **NN ile COD → sahte-spin (Berry) ileri-haritası** kur (demet/spin
   simülasyonundan eğit). Bu, `orbit_ileri_olcum.md §2-3,§5,§7`'deki "f kapalı-yörünge
   fonksiyoneli ama ⟨x·y⟩ değil; analitik pinlenemedi" açık problemini *analitik
   yerine öğrenilmiş* harita ile aşma denemesidir.
2. NN, **sıfır sahte-EDM için COD'un nasıl modifiye edilmesi gerektiğini** öğrenir.
3. **Dayanıklılık testi:** tilt, β-beat, BPM ofset/gürültü değiştikçe harita
   kayıyor mu? (Harita gizli parametrelere bağlıysa kırılır.)
4. Dayanıklıysa: misalignment'ı sıfırlamak yerine, **orbit-görünür knob'larla
   (quad/corrector) sahte-EDM'yi hedefleyen "akıllı düzeltme"** uygula.

**Neden no-go'yu atlayabilir (hipotez):** (a) bu bir **ileri-harita** (COD→EDM),
inversiyon değil; (b) düzeltme **orbit-görünür** kollarla **EDM-hedefli** yapılır,
görünmez simetrik misalignment'ı geri-çatmaya çalışmaz. No-go simetrik
misalignment'ı *ölçmeyi* yasaklar; ama sahte-EDM'yi orbit-görünür müdahaleyle
*null'lamak* belki ayrı bir problemdir. **Açık — test edilecek.**

**Bağlantı:** `orbit_ileri_olcum.md §5,§7` (ileri-ölçüm no-go'yu atlar mı — açık
problem), `omarov.md §10`, `squid_bpm_test.md §8` (NN bağlayıcılık şüphesi: NN
gözlenebilirlik tabanını yenemez — burada fark, NN'in misalignment yerine
*EDM-hedefli düzeltme* öğrenmesi). **Şimdi değil; daha sonra çalışılacak.**

---

## 🔲 Ertelendi

| Madde | Neden ertelendi | Öncelik |
|---|---|---|
| Yeni topoloji kafes yeniden tasarımı (g≈0.5 T/m + kapalı yörünge türetimi) | Tam lattice redesign; kazanım belirsiz | Düşük |
| Radyal polarizasyonla trim kanalı | Simetrik içeriğe o da kör (Ω_z ∝ β_y); kazanım sınırlı | Düşük |
| Kurum bilgisi + teşekkür bölümü (`makale_trim_tr.tex`) | Yazara özgü | Yayın öncesi |
| Omarov PRD referans değerlerinin birebir teyidi | Kütüphane erişimi | Yayın öncesi |
| Gerçek makine verisiyle doğrulama | Prototip halka gerekiyor | Uzun vadeli |
| RF + sekstüpol açıkken trim doğrusallığı | Mevcut altyapıyla yapılabilir | Orta |
| x-yönü kaçıklıkları ve quad tilt çapraz terimleri | `R_dx` matrisi mevcut | Orta |
| Zamanla sürüklenen hizalamada izleme (tracking) kipi | Tasarım gerekiyor | Orta |
