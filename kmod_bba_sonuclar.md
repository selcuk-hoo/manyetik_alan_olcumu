# kmod_bba_sonuclar.md — All-quad AC-BBA linchpin: sonuçlar ve doğrulama

> **Durum (2026-06):** Bu belge, `kmod_bba_plani.md`'deki LINCHPIN testinin
> (all-quad K-modülasyon → per-quad demet-tabanlı hizalama → kalan geometrik-faz
> sahte-EDM) ilk nicel sonuçlarını kaydeder. Bağımsız okunur. Kod:
> `ac_bba_observability.py`, `ac_bba_linchpin.py` (kalıcı, dalda);
> `analytic_kmod.py` (v2.8'den geri alındı). Estimator:
> `berry_data/false_edm_4d.py` (4D-CO + model-fit, σ²-doğrulamalı).
>
> **Tek cümlelik sonuç:** Per-quad AC-BBA, yörünge-farkı (ΔR) inversiyonunun
> simetrik-mod körlüğünü (no-go) **atlar** — simetrik ve antisimetrik modları
> EŞİT (üniform koşullanmış, 1.35× spread) ve BPM-ofsetine bağışık ölçer; kalan
> geometrik-faz sahte-EDM hedefin (1 nrad/s) altına **optik-model (β-beating)
> ε ≲ %0.5–1 kontrol edilirse** iner. Sınır artık istatistiksel değil, sistematik.

---

## 1. Çerçeve: hangi ölçüm hizalamayı geri-çatabilir?

Omarov (PRD 105,032001) geometrik-faz sahte-EDM'i CR-ayrım küçültmeyle hedefin
altına indirir ama o ayrımı/hizalamayı **ölçecek** enstrümanı (48-BPM/SQUID +
K-mod reconstruction) simüle etmez ve simetrik (orbit-kör) moda körlük ihtimali
açıktır (`omarov.md §9`). İki aday ölçüm:

- **(A) UNIFORM-frekans ΔR** (eski v2.7/v2.8 yolu, `analytic_kmod.py`): iki
  gradyanda kapalı-yörünge FARKI Δy = ΔR·dy → dy = ΔR⁻¹Δy. ΔR simetrik (yüksek-k)
  modlarda küçük tekil değerli (G_k ∝ 1/|Q²−k²|) → **no-go**.
- **(B) PER-QUAD AC-BBA** (önerilen): her quad ayrı frekansta modüle; BPM'de o
  frekansın genliği A_ij = T_ij·o_j ölçülür (o_j = demet-quad merkez ofseti).
  ô_j = Σ_i T_ij A_ij / Σ_i T_ij² → **per-quad projeksiyon, matris-ters YOK.**

---

## 2. Gözlenebilirlik (ac_bba_observability.py — analitik, C++ gerektirmez)

FODO: 48 quad, Q_y = 2.303, β_y = 41.5–75.8 m, modülasyon derinliği ΔK/K = %2.

| Ölçüm | Değer | Yorum |
|-------|-------|-------|
| Per-quad σ_o spread (max/min) | **1.35×** | üniform koşullanma (no-go YOK) |
| κ(ΔR_dy) | 3.74×10⁴ | uniform-frekans ΔR ill-conditioned |
| En küçük 12 ΔR-modunun simetrik içeriği | **%92** | simetrik = küçük-SV = gözlenemez |
| En büyük 12 ΔR-modunun simetrik içeriği | %8 | antisim = büyük-SV = gözlenir |

**Mod-ayrımlı geri-çatım** (gerçekçi BPM: ofset 100 μm + gürültü 1 μm, 40 seed):

| Yöntem | simetrik corr | antisim corr |
|--------|---------------|--------------|
| **AC-BBA (B)** | **0.997 ± 0.001** | **0.997 ± 0.001** |
| ΔR ters (A) | 0.128 ± 0.030 | 0.588 ± 0.088 |

→ AC-BBA simetrik ve antisim'i **EŞİT** geri-çatar; ΔR simetrikte **çöker**
(antisim'i kısmen kurtarır — projedeki "orbit-düzeltme antisim'i 7.7× temizler"
ile tutarlı; simetriğe kör). **BPM-ofset bağışıklığı:** AC demodülasyon DC ofseti
otomatik söndürür (cos(ωt) ile ortogonal) → 100 μm ofset altında geri-çatım
korelasyonu = 1.000000; ΔR'nin ihtiyaç duyduğu common-mode iptal gerekmez.

**Sonuç:** No-go bir *ΔR-inversiyon* sınırıdır; per-quad AC-BBA inversiyon
yapmadığından onu **atlar**. (`orbit_ileri_olcum.md §7`'deki "no-go atlanır mı?"
açık sorusunun, ileri-fonksiyonel değil, **doğrudan per-quad ölçüm** üzerinden
yanıtı.)

---

## 3. İstatistiksel hassasiyet (ac_bba_linchpin.py --stat)

Tur-tur BPM okumaları (f_rev = 224.3 kHz; T_rev = 4.458 μs) modülasyon frekansında
demodüle edilir: σ_stat,j = σ_BPM·√(2/N_tur)/√(Σ_i T_ij²).

- depth = %2, σ_BPM = 1 μm, t_int = 1 s → N_tur = 2.24×10⁵
- **σ_stat per-quad = 20–27 nm** (ort 23.6 nm), her iki düzlem.

→ İstatistiksel taban hedefin **çok altında** (1 s entegrasyonla ~24 nm). Gerçek
sınır istatistik değil, **optik-model bilgisi (β-beating)**.

---

## 4. LINCHPIN — kalan sahte-EDM (β-beating sınırlı)

Düzeltme artığı: gerçek optik T_true (β-beating ε) ≠ model T_model → ô per-quad
gain hatası → düzeltme sonrası kalan demet-quad ofset e = o − ô. Bu artık deseni
DOĞRULANMIŞ estimator'a misalignment olarak verilir (konservatif: perfect-steering
CO sönümünü saymaz → üst sınır).

**Kalan ofset RMS (lineer model, 10 μm giriş misalignment, gerçek ofset 53.8 μm):**

| β-beating ε | kalan ofset RMS |
|-------------|-----------------|
| %0 (yalnız istatistik) | 0.024 μm |
| %0.1 | 0.038 μm |
| %0.5 | 0.149 μm |
| %1.0 | 0.296 μm |
| %5.0 | 1.507 μm |

**Estimator kalibrasyonu** (beyaz misalignment, azaltılmış-ayar n_turns=14/t2=3e-4;
p≈2 doğrulaması aynı koşumda):

| σ (misalignment) | \|sahte-EDM\| | Not |
|------------------|---------------|-----|
| 10 μm | 9.12×10⁻⁷ ± 6.4×10⁻⁷ rad/s | n=3 seed |
| 5 μm | 2.28×10⁻⁷ ± 1.6×10⁻⁷ rad/s | n=3 seed |
| **üs p (10→5 μm)** | **2.002** | geometrik faz (beklenen 2.00) ✓ |
| ölçek A = f/σ² | 9.12×10³ rad/s/m² | f@10μm = 9.1×10⁻⁷ (projedeki ~1e-6 ile tutarlı) |

p = 2.00 azaltılmış-ayarda korunduğundan, göreli karşılaştırmalar (β-beating
süpürmesi) geçerlidir. Ölçek yasası: **f_res = A·σ_res²**.

**Kalan sahte-EDM** (DOĞRUDAN estimator, kalan ofset deseni misalignment olarak;
azaltılmış-ayar, p=2.00 doğrulanmış). Hedef = 1 nrad/s = 1×10⁻⁹; gerçek EDM =
9.81×10⁻¹⁰ rad/s. Ölçek yasası A_eff = 1.18×10⁴ rad/s/m² (doğrudan ε=%1
noktasından; beyaz-A=9.1×10³'ten %29 büyük → antisim-baskın artığa uygun):

| β-beating ε | kalan ofset RMS | \|sahte-EDM\| | hedef<1e-9? | kaynak |
|-------------|-----------------|---------------|-------------|--------|
| %0 (stat tabanı) | 0.024 μm | **4.2×10⁻¹¹** | ✓ (24×) | doğrudan |
| %0.1 | 0.038 μm | ~1.7×10⁻¹¹ | ✓ | ölçek yasası |
| %0.5 | 0.149 μm | ~2.6×10⁻¹⁰ | ✓ (4× marj) | ölçek yasası |
| %1.0 | 0.296 μm | **1.03×10⁻⁹** | ✗ (≈ gerçek EDM) | doğrudan |
| %5.0 | 1.507 μm | **1.74×10⁻⁸** | ✗ | doğrudan |

**Geçiş (crossover):** 1 nrad/s hedefi σ_res ≈ 0.29 μm'de, yani **β-beating ε ≈ %1**'de
aşılır. Rahat marj (×4) için **ε ≲ %0.5** gerekir.

### VERDİKT (make-or-break)

**POZİTİF — koşullu.** All-quad per-quad AC-BBA + standart (kapasitif) BPM:
- Simetrik (orbit-kör) modu görür (no-go'yu atlar) — gözlenebilirlik kanıtlandı (§2).
- BPM ofsetine (100 μm) bağışık; istatistiksel taban hedefin çok altında (~24 nm/1s).
- Kalan geometrik-faz sahte-EDM **β-beating optik bilgisi ε ≲ %0.5–1 kontrol
  edilirse** hedefin (1 nrad/s) altına iner. Bu, standart LOCO ile ulaşılabilir
  bir optik-kalibrasyon hedefidir (drift makalesi: LOCO %1 operasyonel,
  `makale-taslagi-2.md §3.7`).
- **Sınırlayıcı etken artık istatistik/no-go DEĞİL, optik-model (β-beating)
  bilgisidir** — bu, tasarım/kalibrasyonla iyileştirilebilir somut bir hedeftir
  (no-go'nun indirgenemez simetrik tabanından farklı).

**Aşırı-iddia koruması:** Tüm geri-çatım iddiaları permütasyon-eşdeğeri (40 tohum
mod-ayrımlı), σ²-testi (p=2.002) ve gerçekçi BPM ofset+gürültü altında doğrulandı.
β-beating modeli β,φ'ye ε-mertebeli rastgele bozulma (LOCO-artığı temsili);
düzeltme artığı konservatif (perfect-steering CO sönümü sayılmadı → üst sınır).

---

## 5. Reprodüksiyon

```bash
# Gözlenebilirlik (hızlı, C++ yok)
python3 ac_bba_observability.py

# İstatistiksel hassasiyet (hızlı)
python3 ac_bba_linchpin.py --stat

# Estimator kalibrasyonu + β-beating süpürmesi (C++; pahalı)
python3 ac_bba_linchpin.py --calib --workers 3 --seeds 3
python3 ac_bba_linchpin.py --sweep --workers 3 --seeds 3
```

```bash
# Figürler (fig_kmod_obs.png + fig_kmod_linchpin.png; matplotlib gerekir)
python3 make_kmod_figures.py
```

**Not:** Pahalı estimator koşumları (CO-bulma + spin takibi) için azaltılmış
ayar (n_turns=14 CO, t2=3e-4) sürücüsü `/tmp/kmod_recover/fast_est.py`;
p≈2 aynı koşumda doğrulanır.

## 6. Sistematik bütçe: BPM ofset/kazanç ve quad-tilt (`ac_bba_systematics.py`)

β-beating bütçesi §4'te. Üç ek sistematik nicellendi (soru: "BPM-ofset ve quad-tilt
analizi yapıldı mı?" — evet, aşağıda):

### 7.1 BPM OFSET — bağışıklık RİGORÖZ doğrulandı (varsayım değil)
Önceki gözlenebilirlik testinde ofset bağışıklığını *model kurarken* (genliğe
eklemeyerek) varsaymıştık. Şimdi zaman-domeni demodülasyonunu gerçekten hesapladık:
BPM her tur okur, modülasyon frekansında demodüle edilir; statik ofset (DC) Dirichlet
çekirdeğiyle sızabilir.

| Frekans seçimi | ô ofset-yanlılığı (100 μm ofset altında) |
|----------------|------------------------------------------|
| serbest (1–10 kHz keyfi) | RMS **65 nm**, maks 261 nm |
| pencere-kilitli (f=tam·f_rev/N) | **≈ makine-epsilon (TAM sıfır)** |

→ Serbest frekansta bile (~65 nm) bütçe eşiğinin (~300 nm) **ÇOK altında**;
frekansları entegrasyon penceresine kilitlersek ofset bağışıklığı **tam**. BPM
ofseti AC-BBA için sistematik DEĞİL (ΔR'nin aksine, common-mode iptal gerekmez).

### 7.2 BPM KAZANÇ hatası — asıl BPM sistematiği, 48 BPM'de ortalanır
Gerçek BPM sistematiği ofset değil çarpımsal kazanç hatası δg_i'dir.
ô_j = o_j(1 + Σ_i w_ij δg_i), w_ij = T_ij²/Σ_i T_ij² (toplam 1).

| σ_gain | kalan ofset RMS | kalan sahte-EDM | hedef<1e-9? |
|--------|-----------------|------------------|-------------|
| %1 | 0.087 μm | 8.8×10⁻¹¹ | ✓ |
| %5 | 0.43 μm | 2.2×10⁻⁹ | ✗ |
| %10 | 0.87 μm | 8.8×10⁻⁹ | ✗ |

→ Kazanç 48 BPM üzerinden ortalandığından (per-quad değil) β-beating'den **baskın
değil**: %1 kazanç → 0.087 μm < %1 β-beating → 0.30 μm. %1 kazanç kalibrasyonu rutin.

### 7.3 QUAD TİLT — İKİ kanal, ayrı ayrı
**(a) BBA-ölçüm yanlılığı (çapraz-düzlem sızıntısı, lineer model):** eğik quad'ın
modülasyonu skew kick verir → ô_y += 2ψ·o_x. Düzeltme artığı ∝ 2ψ·o → kalan
sahte-EDM ∝ (2ψ)²·f₀ (tilt'te İKİNCİ derece): ψ=1 mrad → 1.4×10⁻¹⁰ (küçük).

**(b) Tilt'in DOĞRUDAN geometrik-faz kanalı (C++ estimator):** tilt, sahte-EDM'i
BBA'dan bağımsız da besler. Sabit misalignment + tilt taraması (azaltılmış ayar,
2 seed; `/tmp/kmod_recover/fast_est.py tilt`):

| mis σ | tilt ψ | \|sahte-EDM\| | hedef<1e-9? |
|-------|--------|---------------|-------------|
| 0 | 1 mrad | 1.0×10⁻⁸ ± 4.5×10⁻⁹ | ✗ (büyük saçılma; kısmen CO-bulma artığı) |
| 0.30 μm | 0 (baseline) | 7.6×10⁻¹⁰ ± 6.8×10⁻¹⁰ | ✓ |
| 0.30 μm | **0.1 mrad** | 3.8×10⁻¹⁰ ± 3.1×10⁻¹⁰ | ✓ (baseline'ı değiştirmez) |
| 0.30 μm | **1 mrad** | 3.0×10⁻⁹ ± 1.9×10⁻⁹ | ✗ |
| 10 μm | 1 mrad | 8.3×10⁻⁷ | (dx·dy kanalı domine) |

→ **Tilt'in doğrudan kanalı (b), BBA-yanlılığından (a) BASKINDIR.** Tilt toleransı
**≈ 0.1–0.3 mrad**: 0.1 mrad'da bütçe değişmez, 1 mrad'da hedefi 3× aşar.

**Önemli kapsam notu (dürüst sınır):** dx,dy AC-BBA tilt'i **ölçmez**. Tilt ayrı
kontrol gerektirir: (i) mekanik roll hizalama ≲0.1 mrad (ulaşılabilir ama tipik
survey'den sıkı), veya (ii) skew-bileşen modülasyonuyla bir **skew-BBA uzantısı**.
Bu, yöntemin açık bir sınırıdır; β-beating bütçesinden ayrı bir kalemdir.

### 7.3.1 Pratiklik (0.2 mrad makul mu?) ve CW/CCW+flip tilt'i giderir mi?

**0.2 mrad roll pratik mi?** Evet, *standart-ila-sıkı* ama olağan:
- Modern lazer-tracker survey'i roll'u ~0.1–0.2 mrad RMS hizalar; 0.2 mrad
  ulaşılabilir sınırda.
- Kuplaja duyarlı halkalar (ışık kaynakları) zaten ~0.1–0.5 mrad roll şart koşar.
- pEDM tasarımı skew-kuplaj sistematiği için (omarov.md App. E) zaten sıkı roll
  ister → bizim ~0.1–0.3 mrad toleransımız tasarımın *zaten* talep ettiğiyle
  tutarlı, EK yük değil.
- Mekanik karşılık: ψ=0.2 mrad, yarı-açıklık ~4 cm quad'da ~8 μm kutup kayması →
  survey hassasiyeti seviyesinde.
- Uyarı: 48-quad RMS'i; drift'e karşı korunmalı; tek-quad outlier önemli.

**CW/CCW + quad-flip tilt etkisini giderir mi? — HAYIR (estimator testi).**
Aynı 10 μm misalignment deseni; sahte-EDM (EDMSwitch=0); yön (CW/CCW) × polarite
(g→−g) 4'lü (`/tmp/kmod_recover/fast_est.py cwccw`):

| tilt | f(CW,g+) | f(CCW,g+) | f(CW,g−) | f(CCW,g−) |
|------|----------|-----------|----------|-----------|
| 0 mrad | +5.24e−7 | +1.556e−6 | +1.556e−6 | +5.24e−7 |
| 1 mrad | +5.25e−7 | +1.571e−6 | +1.572e−6 | +5.25e−7 |

İki kritik bulgu:
1. **DEJENERASYON KESİN:** f(CCW,g+) = f(CW,g−) (oran 1.00). Yani **quad-flip,
   CW/CCW ile ÖZDEŞ** (idealize FODO ters-çevirme simetrisi; Omarov Eq. C2 4'lü →
   2'li). Flip **bağımsız ikinci bir iptal knob'u SAĞLAMAZ** → "CW/CCW + flip" =
   etkin olarak tek ters-çevirme.
2. **CW/CCW geometrik-faz sahte-EDM'i NULL'lamaz:** (CW−CCW)/2 artığı = 5.16e−7 ≈
   ham (kazanç ~1× bu seed'de; ensemble ~3.4×, proje). Tilt EKLEMEK her kombinasyonu
   yalnız ~%1 değiştirir (tilt'in mutlak katkısı ~birkaç×10⁻⁹, 10 μm fonu altında
   küçük; ama 0.3 μm kalan üstünde baskın — §7.3). Tilt katkısı (CW−CCW)/2
   kanalında **kalır** (~6.7e−9 marjinal) → ters-çevirme tilt'i ayrıca temizlemez.

→ **Sonuç:** CW/CCW+flip, tilt sahte-EDM'ini *özel olarak* gidermez; tilt aynı
bending-kuplaj/geometrik-faz mekanizmasından geçer ve ters-çevirme onu yalnız
misalignment kanalıyla aynı mütevazı oranda (~3.4× ensemble) bastırır. Tilt yine
**≲0.1–0.3 mrad kontrol** (veya skew-BBA) gerektirir. (Tek-seed; marjinal farklar
CO-bulma gürültüsü mertebesinde — ensemble ile pekiştirilebilir; dejenerasyon
bulgusu kesin.)

### 7.4 Sistematik bütçe özeti

| Kalem | Eşik (hedef <1 nrad/s için) | Ulaşılabilirlik |
|-------|------------------------------|------------------|
| β-beating (optik model) | ε ≲ %0.5–1 | LOCO (rutin) |
| BPM ofset | — (bağışık; kilitli f'de tam) | sorun değil |
| BPM kazanç | σ_g ≲ %1–2 | kalibrasyon (rutin) |
| Quad tilt (roll) | ψ ≲ 0.1–0.3 mrad | mekanik veya skew-BBA |
| İstatistik (BPM gürültü) | — (~24 nm/1s) | sorun değil |

Bağlayıcı kalemler: **β-beating ve quad-tilt**. İkisi de tasarım/kalibrasyon-
ulaşılabilir; no-go'nun indirgenemez tabanından farklı.

## 7. Makale

İki sürüm: `makale_kmod_bba.tex` (sıkı/teknik) ve **`makale_kmod_bba_pedagojik.tex`**
(pedagojik — sezgi kutuları + figürler + tüm eski TODO'lar dolduruldu: Berry
fonksiyoneli, drift/LOCO negatifi, kapasitif-BPM gürültü/süre modeli, Mirza/Huang
ayrışması). Figürler: `fig_kmod_obs.png` (gözlenebilirlik: ΔR-SV spektrumu +
mod-ayrımlı geri-çatım), `fig_kmod_linchpin.png` (kalan sahte-EDM vs β-beating),
`berry_data/berry_weights.png` (Berry ağırlık profili). Üretici: `make_kmod_figures.py`.
