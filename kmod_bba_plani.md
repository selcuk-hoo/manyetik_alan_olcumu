# kmod_bba_plani.md — Yeni makale planı: all-quad K-modülasyon BBA ile geometrik-faz sahte-EDM ölçümü

> **Amaç:** Omarov'un (PRD 105,032001) açık bıraktığı **CR-ayrım/hizalama ÖLÇÜMÜ
> boşluğunu** kapatan bir metot makalesi. Çekirdek iddia: **tüm quad'lar
> K-modüle edilip standart (kapasitif) BPM'lerle geri-çatım yapılırsa**, geometrik-
> faz sahte-EDM'i süren quad hizalama hataları **gerçek-zamanlı, yeterli hassasiyetle**
> ölçülebilir — SQUID-BPM gerektirmeden. Bu dosya yeni bir oturumda sıfırdan
> devam etmek için **tüm gerekli detayları** içerir; bağımsız okunur.
>
> İlgili belgeler: `omarov.md` (§9 boşluk, §10 nerede duruyoruz), `orbit_ileri_olcum.md`
> §9, `false_edm_harmonic_sinir.md` (no-go), `literatur/` (prior-art), `CLAUDE.md`
> (proje kuralları). Bu plan onları **özetler ve birleştirir**.

---

## 0. Tek paragrafta hikâye

Omarov geometrik-faz sahte-EDM'ini CW+CCW+polarite + CR-ayrım küçültmeyle hedefin
(1 nrad/s) altına indiriyor **ama o ayrımı/hizalamayı ölçecek enstrümanı
(48-BPM/SQUID + K-mod reconstruction) test etmiyor** ve simetrik (orbit-kör)
bileşene körlük ihtimali açık. Biz: (1) sahte-EDM'in Berry-faz kökenini fitlerle
gösteriyoruz; (2) antisim/sim mod ayrışımını veriyoruz; (3) bir-birkaç quad
modülasyonunun ΔR düşük-rank'inden ötürü yetmediğini; (4) drift+LOCO'nun simetrik
modlara kör olduğunu gösteriyoruz; (5) **tüm quad'ları modüle edip standart BPM ile
geri-çatımın** işe yarayıp yaramadığını test ediyoruz; (6) **kapasitif-BPM içgörüsü**
ile SQUID gereksinimini kaldırıyoruz. Pozitif çıkarsa: ucuz, gerçek-zamanlı,
nondestructive hizalama-ölçüm yöntemi. Negatif çıkarsa: keskin gözlenebilirlik-
sınırı teoremi.

---

## 1. Yayın anlatısı (bölüm sırası)

1. **Giriş + Omarov boşluğu.** pEDM, frozen-spin, geometrik-faz sistematiği.
   Omarov'un önerisi: CR-ayrımı SQUID-BPM/48-BPM + K-mod ile ölç, dipole-korektörle
   küçült. **Test edilmedi** (omarov.md §9). Hizalama gereksinimi ~10 μm (mekanik
   su terazisi veya SQUID-BPM).
2. **Mekanizma: Berry fazı ↔ quad misalignment.** Bizim fitlerimiz: sahte EDM
   ∝ σ² (geometrik faz), **p=2.00** doğrulaması (`sigma_olcekleme`); Berry
   fonksiyoneli f ≈ Σ wᵢ xᵢyᵢ (LOO-R²≈0.88, permütasyon-doğrulamalı; `berry.md`).
   = Omarov Fig. 9a'nın nicel karşılığı.
3. **Antisim/simetrik mod ayrışımı.** QF/QD zıt-işaret (antisim, düşük-k,
   orbit-görünür) vs aynı-işaret (sim, yüksek-k, orbit-kör). Ham sahte-EDM'i
   antisim domine eder (~37×); düzeltme-sonrası artık simetrik (orbit-kör).
4. **NEGATİF-1: bir/birkaç quad modülasyonu yetmez.** ΔR düşük-rank → simetrik
   modlar gözlenemez. (v2.7 tek-quad k-mod + bizim ΔR analizi.)
5. **NEGATİF-2: drift + baştan-LOCO yetmez.** Drift monitör simetrik modlara büyük
   oranda kör (mevcut drift makalesi `drift_monitor/drift_makalesi.md` buraya **bir
   adım** olarak girer); LOCO da BPM-ofset altında simetrik tabana çarpar (no-go).
6. **TEST: all-quad modülasyon + CR-ayrım ölçümü Omarov'un dediği gibi yeterli mi?**
7. **POZİTİF ÇEKİRDEK: all-quad K-modülasyon + standart BPM geri-çatım.** Gerçek-
   zamanlı, yeterli hassasiyette quad-offset ölçümü → sahte-EDM kontrolü.
8. **Kapasitif-BPM içgörüsü.** SQUID CR-ayrımı (manyetik) ölçer, kapasitif ölçemez;
   AMA tek-demetin quad-modülasyon **salınım genliği ∝ quad-merkez ofseti (yatay
   düzleme mesafe)** → kapasitif BPM yeter. SQUID gereksinimi kalkar.
9. **Sonuç + literatür konumlandırma.**

**Negatiflerin işlevi:** her biri daha ucuz/basit alternatifi eler; geriye all-quad
BBA'yı **gerekli** çözüm olarak bırakır. Dolgu değil, gerekçedir.

---

## 2. Başlangıç noktası: eski tag'lerden geri alınacak kod

Tüm-quad **uniform (aynı-frekans) K-modülasyon → hizalama geri-çatım** zaten
yapılmıştı; büyük temizlikte (commit `41b1c6a`) silindi, **tag'lerden geri alınır**:

| Dosya | Kaynak | Ne yapar |
|-------|--------|----------|
| `analytic_kmod.py` | `git show v2.8:analytic_kmod.py` | **EN UYGUN BAŞLANGIÇ.** Analitik FODO Twiss → R[i,j]=√(βᵢβⱼ)/(2sin πQ)·KL_j·cos(\|φᵢ−φⱼ\|−πQ); iki gradyan (g_nom, g_pert=1.02·g) → ΔR; ölçülen Δy'den dy SVD geri-çatım. C++ gerektirmez. |
| `build_response_matrix.py` | `git show v2.7:...` veya `v2.0:...` | Simülasyon-tabanlı R; **uniform k-mod modu (a): tüm quadlar g1→g1·1.02**; tek-quad modu (b). ProcessPoolExecutor. |
| `test_kmod_reconstruction.py` | `git show v2.7:...` | ΔR·dy=Δy çözerek hizalama geri-çatım; BPM ofset common-mode iptali, quad_tilt/dipol_tilt kirliliği, BPM gürültüsü ile test. |
| `test_loco_reconstruction.py` | `git show v2.0:...` | İki-konfig LOCO; v2.0'da dy corr=0.999973, RMS~1.4μm (BPM ofset/gürültü YOK — ideal). |

Geri alma örneği:
```bash
git show v2.8:analytic_kmod.py > analytic_kmod.py
git show v2.7:test_kmod_reconstruction.py > test_kmod_reconstruction.py
git show v2.7:build_response_matrix.py > build_response_matrix_kmod.py   # mevcut olanı ezmemek için
```

**DİKKAT — eksik olan ve eklenecek (kullanıcının notu):** Bu eski çalışmada
**4-katlı simetrik parçacık KULLANILMADI**, yani **sahte-EDM'deki kuadratik
(geometrik-faz) terim izole edilmedi**. Geri-çatım sadece **hizalama → orbit →
hizalama** zinciriydi; sahte-EDM'e bağlanmamıştı. Yapılacak değişiklik: geri-çatılan
hizalamadan **doğrulanmış estimator ile sahte-EDM'i ileri-hesaplayıp** (aşağı bkz.)
kalan sahte-EDM'in hedefin altına inip inmediğini ölçmek.

> **Kritik kavramsal not:** v2.0/v2.8 geri-çatımı **kapalı-yörünge-FARKI (ΔR)**
> tabanlı → BPM-ofset altında **simetrik modlara kör** (no-go; v2.8 κ(ΔR)=2.76e4).
> Bu, NEGATİF adım 4-5'i destekler. POZİTİF adım 7'nin işe yaraması için ölçüm
> **kapalı-yörünge-farkı değil, tek-demet K-mod salınım GENLİĞİ** (per-quad BBA)
> olmalı — bu uniform-conditioning'li, no-go'yu atlar (özet bulgu: 1.3× spread).
> **Yeni oturumun çözmesi gereken ana fizik sorusu budur:** uniform-frekans mı
> yoksa frekans-çoğullamalı per-quad mı; hangisi simetrik modu görür?

---

## 3. Doğrulanmış sahte-EDM estimator (DEĞİŞTİRMEDEN kullan)

`berry_data/false_edm_4d.py` → `measure_false_edm(dx, dy, tilt)`:
- 4D kapalı yörünge (find_co_4d: betatron varyansını Newton ile minimize) +
  tek ideal parçacık CO üzerinde + **model-fit seküler eğim** (`measure_dSy_dt_model`).
- **σ-testi p=2.00 ile DOĞRULANDI** (geometrik faz, lineer kaçak yok). 4-katlı
  simetrik örnekleme **eşdeğer alternatiftir, şart değil** (omarov.md §10).
- Gerçek EDM: `fields.EDMSwitch = 1.0` (η=1.88e-15 sabit) → 9.81e-10 rad/s.
- Yön/polarite: `fe.CFG["direction"]=±1` (CW/CCW), `fe.CFG["g0"]/["g1"]=±0.21`
  (polarite-flip). **Dejenerasyon:** idealize FODO'da CCW≡CW+flip (Eq. C2 4'lü→2'li).
- Düz polyfit KULLANMA.

Eğer 4-katlı simetrik örneklemeye geçilecekse: (sx,sy)=±1 dört başlangıç kombinasyonu
ortalaması → betatron + ⟨ΔxΔy⟩ söner, CO arama gerektirmez. İki yol da kuadratik
terimi izole eder; p=2.00 ile çapraz-doğrula.

---

## 4. Bu oturumda elde edilen nicel bağlam (make-or-break)

| Ölçüm | Değer | Script (/tmp) |
|-------|-------|---------------|
| Estimator: sahte EDM ∝ σ² | **p=2.00±0.01** | `sigma_olcekleme.py` |
| Gerçek EDM (TEK/diferansiyel) | 9.81e-10 rad/s | `edm_only.py` |
| 10 μm sahte EDM (seküler) | ~10⁻⁶ (worst 6.5e-6) | `cwccw_ensemble.py` |
| CW/CCW telafisi tek başına | 3.4× (artık 474× EDM) | `cwccw_telafi.py` |
| **Orbit-düzeltme (antisim çıkar)** | **7.7×** (artık 62× EDM) | `orbit_duzeltme.py` |
| Kalan simetrik orbit-kör artık | 62× EDM (6.05e-8) | `orbit_duzeltme.py` |
| Dejenerasyon CCW≡CW+flip | özdeş | `cwccw_validate` |

Bu scriptler `/tmp`'de (proje konvansiyonu). Yeni oturum yeniden üretebilir;
desen: `gen_random(seed)` her quad bağımsız N(0,10μm); `sym_proj` = antisim çıkar.

---

## 5. Literatür konumlandırma (elimizdeki referanslar)

| Argüman | Referans (`literatur/`) | Kullanım |
|---------|--------------------------|----------|
| BBA/K-mod prior-art (ayrışma!) | `ref_simultaneous_bba.md` (Mirza arXiv:2104.05300), `ref_fast_bba_ac.md`, `ref_pac1993_2263.md` | **En yakın prior-art.** Bizim özgünlük teknik DEĞİL; EDM/geometrik-faz hedefleme + simetrik-mod analizi + kapasitif-BPM fizibilitesi |
| Simetrik-mod/dejenerasyon yapısı | `ref_mirza_symmetric_circulant.md`, `ref_wegscheider_degeneracy.md` | Antisim/sim ayrışımı, ΔR dejenerasyonu |
| Drift/yer-hareketi körlüğü | `ref_rossbach_groundwaves.md` | NEGATİF-2 (drift simetrik-kör) |
| Tüm-halka ORM / süreklilik | `ref_continuous_orm.md` | Geri-çatım çerçevesi |
| All-electric quad misplacement | `ref_allelectric_quad_misplacement.md` | Kullanıcının kendi makalesi; bağlam |

**En büyük özgünlük riski:** Simultane AC-BBA (Mirza) tekniği zaten var. Referee
"bu biliniyor" diyecek. Savunma: **(a)** sahte-EDM/geometrik-faz sistematiğine
bağlama (Berry fonksiyoneli köprüsü: "hangi mod sahte-EDM sürüyor" × "hangi mod
BBA-görünür"); **(b)** simetrik-mod gözlenebilirlik analizi; **(c)** pEDM gerçek-
zamanlı + kapasitif-BPM fizibilitesi; **(d)** orbit/SQUID-ayrım yolunun simetrik
modlarda çöküp BBA'nın çökmediğini kanıtlamak.

---

## 6. Yeni oturumun yapacağı testler (öncelik sırası)

1. **Kod geri-alma + estimator bağlama.** v2.8 `analytic_kmod.py` + v2.7
   `build_response_matrix.py` (uniform k-mod) main'e al; `false_edm_4d.py` estimator'a
   bağla. Build gerekiyorsa `bash build_integrator.sh`.
2. **LINCHPIN — all-quad K-mod + standart BPM geri-çatım (adım 7).**
   48-quad K-modülasyon → tek-demet salınım genliği (per-quad BBA) → quad-offset
   geri-çatım → kalan sahte-EDM (doğrulanmış estimator ile ileri-hesap) hedefin
   (1 nrad/s) altına iniyor mu? **Gerçekçi kapasitif-BPM ofset (~100μm) + gürültü
   (~1μm) + gerçek-zamanlı** koşulları altında.
   - Uniform-frekans (ΔR) vs frekans-çoğullamalı per-quad: hangisi simetrik modu görür?
   - Simetrik-mod gözlenebilirliğini ölç (no-go'yu atlıyor mu?).
3. **Negatif kontroller (adım 4-5).** Tek/birkaç quad mod → ΔR düşük-rank → yetmez.
   Drift+LOCO → simetrik-kör → yetmez. (Mevcut drift makalesi sonuçları + v2.7/v2.8.)
4. **Kapasitif-BPM içgörüsü (adım 8).** Salınım genliği ∝ quad-offset ilişkisini
   nicel göster; SQUID gerekmediğini kanıtla.
5. **Karar kapısı:** linchpin başarılı → POZİTİF metot makalesi; başarısız →
   keskin NEGATİF (gözlenebilirlik-sınırı teoremi). Her iki halde yayınlanır;
   hangisi olduğunu bu test belirler.

**HER ML/geri-çatım iddiasını doğrula:** permütasyon testi, σ²-testi (p≈2),
gerçekçi BPM ofset+gürültü. Aşırı-iddiadan kaçın (bu projede tekrar tekrar oldu).

---

## 7. Proje kuralları (yeni oturum için)

- **Dil:** kod İngilizce isim + Türkçe yorum; .md Türkçe, bağımsız okunur.
- **Push:** tüm çalışma **main'e** (`git push -u origin main`). Tag push'u bu
  ortamda 403 — tag kullanıcı lokalden atar.
- **Tracker build:** `integrator.cpp` değişirse `bash build_integrator.sh` ŞART
  (yoksa sessiz hata: hizalama uygulanmaz). Derlenmiş `.so` git'te.
- **Geçici kod `/tmp`'de**, .md'den referansla (kalıcı repoya girmez). `.npy/.npz/
  .png/.csv` .gitignore'da; PDF/`.sh` için `git add -f`.
- **params.json tek kaynak**; hardcode etme.
- **Estimator tuzağı:** 4D-CO+model-fit veya 4-katlı simetri; düz polyfit YASAK.
  σ²-testi (p≈2) ile her zaman doğrula.

---

## 8. Açık fizik soruları (çözülmesi gereken)

1. **Uniform vs per-quad K-mod:** Aynı-frekans uniform modülasyon ΔR-tabanlı
   (simetrik-kör görünüyor); frekans-çoğullamalı per-quad BBA uniform-conditioning'li
   (no-go'yu atlar). Hangisi pratik+yeterli? **Linchpin bunu çözecek.**
2. **Gerçek-zamanlılık:** 48-quad modülasyonu eşzamanlı mı (frekans-çoğullama) yoksa
   sıralı mı; veri-alımını bozmadan sürekli çalışır mı?
3. **Kapasitif-BPM hassasiyeti:** salınım-genliği→offset ölçümü, ~1μm hizalama
   hassasiyetini gerçek gürültüde verir mi?
4. **Simetrik-mod kapanışı:** BBA simetrik modu gerçekten görüyorsa, bu projenin
   no-go'sunu (orbit-inversiyon sınırı) **atlatır** — çünkü BBA inversiyon değil,
   doğrudan per-quad ölçüm. Bunu kesin göster.
