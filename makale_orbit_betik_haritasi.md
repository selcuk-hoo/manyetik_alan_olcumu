# makale_orbit_betik_haritasi.md — `makale_orbit_bastirma` Betik Haritası

Bu belge, **`makale_orbit_bastirma.tex`** makalesinin dayandığı tüm betikleri
tek yerde toplar: her betik **ne yapar**, **nasıl çalışır**, ve **makalenin
hangi bölümü/figürü/sayısını** üretir. Amaç: makaleyi sıfırdan yeniden
üretebilmek ve her sayının izini sürebilmek.

> İlgili belgeler: `bba_betikleri.md` (BBA betiklerinin daha ayrıntılı
> kullanım kılavuzu), `makale_orbit_bastirma.md` (makale günlüğü + §6 sayı→kaynak
> eşleme). Bu belge o ikisini **paper→betik yönünde** birleştirir ve günceller.

---

## 0. Büyük resim: katmanlı mimari

Her sonuç, en altta tek bir C++ motoru sürer. Üstteki katmanlar onu farklı
amaçlarla çağırır:

```
Katman 3  FİGÜR ÜRETİCİLERİ        make_orbit_figures.py  (Fig 1,2,4,5,6,7)
             │                       (Fig-3 üreticisi /tmp'de — bkz. §5)
Katman 2  KAMPANYALAR / ANALİZ     classic_bba_{full,iter,pipeline}.py,
             │                      kmod_drivers/{paper_runs,fast_est,channel_split}.py,
             │                      build_response_matrix.py, analytic_kmod.py
Katman 1  SAHTE-EDM ESTİMATÖRÜ     berry_data/false_edm_4d.py
             │                      (+ false_edm_mode_scan.py yardımcıları)
Katman 0  FİZİK MOTORU (C++)       integrator.cpp → lib_integrator.so
                                    integrator.py (ctypes köprüsü, FieldParams)
             ▲
params.json ─┴─ tüm katmanlara tek konfigürasyon kaynağı
```

**Altın kural:** Herhangi bir *spin* (sahte-EDM) sayısı en sonunda
`integrator.cpp`'yi çağırır (C++ derlenmemişse sonuçlar sessizce yanlış çıkar —
bkz. CLAUDE.md tuzak #1). *Yörünge-only* analitik sayılar (koşullanma, kazanç
yasası, β-beat) C++ gerektirmez; `analytic_kmod.py` / `make_orbit_figures.py`
içindeki analitik matrislerle hesaplanır.

---

## 1. Bir bakışta: betik → ne → makale bölümü

| Betik | Ne üretir | Makale yeri |
|-------|-----------|-------------|
| `integrator.cpp` / `.py` | GL4 semplektik yörünge + Thomas-BMT spin motoru | §II (altyapı) |
| `berry_data/false_edm_4d.py` | 4D-CO + model-fit sahte-EDM estimatörü (dS_y/dt) | §II (estimatör) |
| `kmod_drivers/fast_est.py` | Hızlı tek-atış sahte-EDM (BBA döngüleri için) | §V |
| `analytic_kmod.py` | Analitik Twiss, R, ΔR, kazanç yasası, koşullanma | §III–IV |
| `build_response_matrix.py` | Sayısal (C++) R_dy/R_dx, ΔR + koşullanma | §IV (ΔR), §V (ölçülen yönlendirme) |
| `make_orbit_figures.py` | Fig 1,2,4,5,6,7 (+ yardımcılar) | §III–VII |
| `kmod_drivers/paper_runs.py` | σ² kampanyası, crsep, fourpart, fourfold/gscale | §II, §III, §IV, §VI |
| `classic_bba_full.py` | Uçtan-uca tek-geçiş BBA + BPM gürültü taraması | §V, §VII |
| `classic_bba_iter.py` | İteratif BBA (β-beat/nefes çözümü) → Fig-7 verisi | §V |
| `classic_bba_pipeline.py` | Çok-seed BBA+OC ensemble → tablo/Fig-1(b) | §V, §VII |
| `kmod_drivers/channel_split.py` | BBA artığının simetrik/antisim kanal ayrışımı | §V (f_anti/f_sym) |
| `classic_bba_{sim,cpp_check}.py`, `diag_bbeat.py` | Prototip / null-doğrulama / tanı | destek (bkz. `bba_betikleri.md`) |

---

## 2. Figür figür harita (7 figür)

Makaledeki `\includegraphics` sırası ve üreticileri:

| Fig | Dosya | Üretici | Veri kaynağı | Bölüm |
|-----|-------|---------|--------------|-------|
| **1** | `fig_orbit_suppression.png` | `make_orbit_figures.fig_suppression()` | Gömülü `RAW_F`/`MEAS_F` dizileri (kaynak: `paper_runs.py sigma`, C++) | §III |
| **2** | `fig_orbit_modes.png` | `fig_modes()` | Satır-içi analitik `R_perquad` SVD + kazanç yasası | §III |
| **3** | `fig_orbit_channels.png` | ⚠️ `/tmp` keşif betiği (repoda yok — §5) | Bilineer kanal ayrışımı f_ss/f_sa/f_as/f_aa (C++) | §III |
| **4** | `fig_orbit_lockin.png` | `fig_lockin()` | Satır-içi analitik ΔR inversiyonu, gürültü tabanında | §IV |
| **5** | `fig_orbit_breathing.png` | `fig_breathing()` | Satır-içi analitik nefes (%2 mod, `R_perquad`) | §IV |
| **6** | `fig_orbit_crsep.png` | `fig_crsep()` (elle çağrılır) | `paper_runs_results.json["crsep"]` (C++) | §IV |
| **7** | `fig_orbit_bba_convergence.png` | `fig_bba_convergence()` | Gömülü `BBA_*` dizileri (kaynak: `classic_bba_iter.py --rmatrix cpp`) | §V |

`make_orbit_figures.py` `__main__` bloğu **Fig 1,2,5,7 ve 4'ü** üretir; **Fig-6
(`fig_crsep`)** ayrıca elle çağrılır (JSON'dan okur), **`fig_sigma()`** ise
yardımcı σ²-doğrulamadır ve makale gövdesinde kullanılmaz.

**Kritik ayrım — hangi figür C++ ister:**
- **Analitik (C++ gerekmez, saniyeler):** Fig-2 (modes), Fig-4 (lockin), Fig-5
  (breathing). Bunlar `git pull` sonrası her yerde anında yeniden üretilir.
- **C++ verisi gömülü/JSON'dan:** Fig-1 (suppression), Fig-3 (channels),
  Fig-6 (crsep), Fig-7 (bba_convergence). Sayılar pahalı C++ koşumlarından gelir;
  figür betiği yalnız çizim yapar. Yeniden üretmek için ilgili kampanyayı
  (Katman 2) koşup diziyi/JSON'u güncellemek gerekir.

---

## 3. Katman katman betik ayrıntıları

### 3.1 Katman 0 — Fizik motoru
**`integrator.cpp` → `lib_integrator.so`; `integrator.py`**
GL4 (Gauss-Legendre 4. derece) semplektik parçacık takibi + Thomas-BMT spin
(EDM anahtarıyla). `integrator.py` ctypes köprüsüdür; `FieldParams` alanları
(gradyanlar, RF, hata profilleri) tutar; `integrate_particle(...)` çağrısı
`quad_dx`, `quad_dy`, `quad_tilt`, `quad_dG` dizileriyle kaçıklık/gradyan-hatası
uygular. **Değişince `bash build_integrator.sh` şart.** → §II.

### 3.2 Katman 1 — Sahte-EDM estimatörü
**`berry_data/false_edm_4d.py`** (makalenin ölçüm reçetesi)
- `find_co_4d(...)`: 4D kapalı yörüngeyi (x,x',y,y') betatron pozisyon
  varyansını 4-boyutlu Newton ile minimize ederek bulur (lattis lineer → 1–2
  adım yeter).
- `measure_false_edm(...)`: CO üzerinde tek ideal parçacık fırlatır, her tur
  S_y örnekler, **model-fit** ile seküler eğim b'yi çeker
  (S_y(t)=a+bt+Σ_k[c_k cos+d_k sin]; düz polyfit DEĞİL — tuzak #7).
- `--scan-s` / `--scan-tilt`: Ŝ (simetri) ve tilt kampanyaları.
- **Doğrulama:** σ²-testinde p=2.00 (geometrik faz), işaret yapısı saf bilineer.
→ §II "estimatör".

**`berry_data/false_edm_mode_scan.py`** — `setup_fields`, `_make_state`,
`measure_dSy_dt_model`, sabit `C`. Yukarıdakinin ve diğerlerinin ortak
yardımcıları.

**`kmod_drivers/fast_est.py`** — `fast_measure(dx, dy, tilt, n_turns, t2,
direction, gflip, gscale, dG)`: BBA döngüleri içinde çok kez çağrıldığından
daha az turla hızlı sahte-EDM verir. `gflip` (polarite), `gscale` (flip-kalibrasyon
hatası ε), `dG` (β-beat) parametreleri. → §V, §VI.

### 3.3 Katman 2 — Tepki matrisleri, analitik kafes, kampanyalar

**`analytic_kmod.py`** (yörünge-only, C++ gerekmez)
- `compute_twiss_at_quads(cfg, g, plane)`: FODO periyodik Twiss (β, φ, Q).
- `build_R_analytic(β, φ, Q, KL)`: analitik kapalı-yörünge tepki matrisi
  R[i,j] = √(β_iβ_j)/(2sinπQ)·KL_j·cos(|φ_i−φ_j|−πQ).
- `build_analytic_dR(cfg, g_nom, g_pert, plane)`: iki optikten ΔR.
- `reconstruct(dR, delta)`: TSVD geri-çatım.
- Kazanç yasası G_k=C/|Q²−k²| ve koşullanma sayıları buradan. → §III (kazanç),
  §IV (inversiyon, cond≈3.7×10⁴).

**`build_response_matrix.py`** (sayısal, C++, paralel)
- İki konfigürasyon (nominal + pert) için 48×48 R_dy/R_dx üretir; farkı ΔR.
- `--rmatrix cpp` BBA'sının kullandığı **ölçülen** matrisler (`R_d{x,y}_1.npy`).
- cond(R), cond(ΔR) yazdırır. **Uyarı:** `g0=g1=g2` ise modülasyon 0 → ΔR tam
  sıfır → cond=inf (bir bug değil; bkz. §5). Gerçek ΔR için `g1≠g0` veya k-mod
  indekslerini `-1` (uniform mod). → §IV (ΔR), §V (ölçülen yönlendirme).

**`kmod_drivers/paper_runs.py`** — modlar:
- `sigma`: sahte-EDM vs σ (2.5/5/10 μm × seed) → **σ²-üssü p≈2.00** ve Fig-1'in
  ham/ölçülen dizileri. → §III.
- `crsep`: CW/CCW demet ayrımı → **Fig-6** verisi (`...results.json["crsep"]`). → §IV.
- `fourpart` / `fourpart_co`: 4-parçacık estimatör doğrulaması (CO etrafında). → §II dipnotu.
- `fourfold` / `gscale`: CW/CCW+polarite 4'lü iptali ve ε (flip-kalibrasyon). → §VI.

### 3.4 Katman 2 — BBA kampanyaları (§V, §VII)
Ayrıntılı kullanım ve tuzaklar için **`bba_betikleri.md`**; özet:

**`classic_bba_full.py`** — uçtan-uca tek-geçiş klasik BBA (ölçülen yönlendirme):
47 quad × 2 düzlem gradyan-mod + komşu-düzeltici bump + 2-nokta tarama → null →
merkez kestirimi → düzeltme → kalan sahte-EDM (C++ spin). Ek olarak
`--bpm-noise` + `--navg` ile **√N averaj gürültü taraması** (#7). → §V, §VII.

**`classic_bba_iter.py`** — iteratif BBA (β-beat/nefes çözümü): her geçişte
yörüngeyi küçült, tekrar BBA. `--rmatrix analytic|cpp`, `--passes`, `--resume`
(checkpoint'li). **Fig-7'nin kaynağı** (`--rmatrix cpp`). ~3 saat/geçiş → lokal iş.
→ §V.

**`classic_bba_pipeline.py`** — çok-seed tam boru hattı: her seed için iteratif
BBA + son SVD yörünge düzeltmesi → nihai sahte-EDM dağılımı. **Otomatik resume**
(biten seed atlanır, `--resume` bayrağı YOK). → tablo (chain) + **Fig-1(b)**
ensemble tabanı. ~8 saat/seed → lokal iş.

**`kmod_drivers/channel_split.py`** — bir BBA koşumunun son-geçiş artığını
(`/tmp/kmod_recover/{rkey}_state.json`) yükler, P_sym/P_anti'ye projekte eder,
her kanalın sahte-EDM'ini C++ spin ile ölçer → **f_anti/f_sym** (§V'in 16×/3.5×
sayıları). `--rkey bba_iter_cpp`.

**Destek betikleri** (sonuç değil, doğrulama/tanı): `classic_bba_sim.py`
(analitik prototip), `classic_bba_cpp_check.py` (null doğrulaması),
`diag_bbeat.py` (β-beat/nefes tanısı). Bkz. `bba_betikleri.md §2`.

### 3.5 Katman 3 — Figür üreticisi
**`make_orbit_figures.py`** — yukarıdaki §2 tablosu. Ayrıca ortak yardımcılar:
`R_perquad(g_arr)` (analitik per-quad R), `sym_anti_projectors()` (hücre-içi
P_sym/P_anti), `twiss_perquad`, `signed_K_vertical`, `_sig_fit` (σ²-üssü fit).
`G_NOM`, `NQ`, `nFODO` sabitleri buradan (diğer betikler import eder).

---

## 4. Sıfırdan yeniden üretme reçetesi

```bash
# 0) motoru derle + ölçülen matrisleri üret
bash build_integrator.sh
python3 build_response_matrix.py                 # R_d{x,y}_1.npy (cpp-BBA için)

# 1) ANALİTİK figürler (C++ gerekmez, saniyeler) — Fig 2,4,5
python3 make_orbit_figures.py                    # __main__: Fig 1,2,4,5,7

# 2) C++ KAMPANYALARI (pahalı; sayıları besler) — lokal makine
python3 kmod_drivers/paper_runs.py sigma  -w 7 --seeds 3     # Fig-1 ham/ölçülen, p≈2
python3 kmod_drivers/paper_runs.py crsep  -w 7               # Fig-6 verisi (JSON)
python3 classic_bba_iter.py -w 7 --passes 5 --rmatrix cpp --resume   # Fig-7
python3 kmod_drivers/channel_split.py --rkey bba_iter_cpp    # §V f_anti/f_sym
python3 classic_bba_pipeline.py -w 7 --seeds 0 1 2 3 4       # Fig-1(b)/tablo
python3 classic_bba_full.py -w 7 --bpm-noise 1e-6 --navg 100,1000,10000   # #7

# 3) figürleri C++ sayılarıyla güncelle
#    Fig-1 (RAW_F/MEAS_F), Fig-7 (BBA_*): make_orbit_figures.py içindeki
#    gömülü dizileri kampanya çıktısıyla elle güncelle; Fig-6 fig_crsep()
#    JSON'dan otomatik okur.
```

---

## 5. Notlar, boşluklar, tuzaklar

1. **Fig-3 (channels) üreticisi repoda yok.** `fig_orbit_channels.png` diskte
   var ama onu üreten betik `/tmp`'deki keşif kodundaydı (proje geleneği:
   geçici kod `/tmp`'de). Yeniden üretmek gerekirse bilineer kanal ayrışımı
   (f_ss/f_sa/f_as/f_aa) `false_edm_4d` + `sym_anti_projectors` ile yeniden
   yazılmalı — `kmod_drivers/channel_split.py` iskeleti buna en yakın başlangıç.
2. **Gömülü diziler ≠ canlı hesap.** Fig-1 (`RAW_F`/`MEAS_F`) ve Fig-7
   (`BBA_*`) sayıları `make_orbit_figures.py` içinde **elle gömülü**; kaynak C++
   kampanyasını yeniden koşarsan bu dizileri güncellemen gerekir (otomatik değil).
   Fig-6 ise JSON'dan okur (otomatik).
3. **`cond(ΔR)=inf` normaldir** (bug değil): `g0=g1=g2` → modülasyon 0 →
   ΔR=0 matris. Anlamlı ΔR için `g1≠g0` ya da uniform mod (k-mod indeksleri −1).
4. **Taşınabilirlik:** betikler repo kökünü `__file__`'den türetir (hardcoded
   yol yok). Eski `berry_data/run{1,2}_gen.py` hâlâ hardcoded + `/tmp/spin_meas`
   bağımlı ölü keşif betikleridir; makalede kullanılmaz.
5. **C++ senkron tutulmalı:** `integrator.cpp` değişip yeniden derlenmezse tüm
   spin sonuçları sessizce "hizalama sıfır" gibi çıkar (tuzak #1). `params.json`
   tek konfigürasyon kaynağıdır; parametre hardcode etme.
6. **β-beat etiketi:** kampanyalardaki `--bbeat X` = **fraksiyonel gradyan
   hatası** RMS'i; gerçek Δβ/β ≈ 5×X (tam-tur Twiss'ten). Makale bunu dürüstçe
   "%1 gradyan hatası ≈ %5 β-beat" olarak yazar.
