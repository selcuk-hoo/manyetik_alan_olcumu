# why_bba_works.md — Önceki yöntemler çalışmazken klasik BBA neden çalıştı?

> **Durum (2026-07):** Bu belge, `separation_bba_testleri.md`'deki T5 test
> zincirinin (klasik null-arayan BBA) **neden** çalıştığını, aynı problemde
> daha önce denenen tüm yöntemlerin **neden** çalışmadığını ve simülasyonun
> nasıl kurulduğunu tek yerde açıklar. Bağımsız okunur. Gerçekçi-sistematik
> koşumu (β-beat + BPM ofset + gürültü) yazım sırasında sürüyordu; nihai
> sayılar `separation_bba_testleri.md §5.2`'ye işlenir. **Kural:** analitik
> hesaplar yalnız yol göstericidir; sonuç olarak yalnız C++ simülasyonu sunulur.

---

## 1. Problemin özü (bir paragraf)

Sahte EDM'i, düzeltme sonrası, kaçıklığın **simetrik** (hücre içinde QF ve
QD'nin birlikte kaydığı) bileşeni sürer. Bu bileşenin kapalı-yörünge izi
~iki mertebe bastırılmıştır (G_k = C/|Q²−k²|, k≈24 ≫ Q≈2.3; cond(R)=193).
Aylarca süren testlerde yörünge-tabanlı her yöntem bu bileşende duvara çarptı
ve "orbit'ten ölçülemez" hükmüne varıldı. İşbirliği yorumunun tetiklediği
yeniden inceleme, bu hükmün **ölçüm sınıfına bağlı** olduğunu gösterdi:
duvarlar iki belirli sınıfın duvarıymış; **yerel null-arayan üçüncü sınıf**
(klasik BBA) hiç test edilmemişti — ve test edilince çalıştı.

---

## 2. Üç ölçüm sınıfı ve duvarları

### Sınıf 1 — Global geri-çatım (inversiyon): ΔR⁻¹, SVD/TSVD, NN, LOCO+drift
48 BPM okumasından 48 kaçıklığı **birlikte** çözer: q̂ = R⁻¹y. Çözüm her
modda 1/σ_i ile ölçeklenir; simetrik modların σ_i'si küçük (ΔR'de
σ_min≈10⁻⁴) → ölçümün HER kusuru simetrik yönde ~10⁴× büyür.
- Beyaz gürültü: lock-in ile yenilir (√N) — ama simetrik için <4 nm ister.
- **β-beat (asıl katil):** model hatası ortalanmaz; %0.5'te simetrik hata
  283 μm (sinyalin ~7 katı), corr hâlâ 0.69 (yanıltıcı). NN/LSTM aynı duvar
  (lineer haritada mimari fark yaratmaz; bilgi-teorik sınır).

### Sınıf 2 — Genlik okuma (dağıtık-frekans per-quad K-mod)
Her quad ayrı frekansta modüle edilir, BPM'deki f_i genliği **sabit yörüngede**
okunur ve kalibrasyona bölünür. Genlik ∝ (y(i) − m_i): ama y(i), 48 quad'ın
birlikte kurduğu **global yörüngedir** (~0.37 mm) ve aranan m_i'den (~10 μm)
kat kat büyüktür → "optik-nefes": ölçülen şey komşuların yörüngesi (kaldıraç
265×). Koherent olduğundan BPM sayısı/SQUID söndürmez; 48 BPM'de corr≈0.

### Sınıf 3 — Yerel null arama (klasik BBA) ← ÇALIŞAN
Quad i modüle edilirken demet o quad'da **yerel bump'la taranır**; tepkinin
**sıfırlandığı** demet konumu bulunur. Sıfır noktası = quad'ın manyetik
merkezi. 48 quad **tek tek, bağımsız** ölçülür.

**Aynı fiziksel sinyal — üç farklı kullanım:** üç sınıf da ε·KL·(y(i)−m_i)
kick'ini görür. Sınıf 2 onu sabit y(i)'de okur (y(i) bilinmez ve devasa →
çöp); Sınıf 1 hepsini birden ters çevirir (1/σ_min → çöp); Sınıf 3 y(i)'yi
**tarayıp sıfır-geçişini** arar → y(i)=m_i noktası, ne y(i)'nin mutlak
değerine, ne diğer 47 quad'a, ne modele bağlı.

---

## 3. Anahtar fizik: modülasyon tepkisinin rank-1 YERELLİĞİ

Quad i'nin gradyanı g→g(1+ε) yapılınca alan farkı yalnız quad i'de ve
şu kadardır:

    ΔB = ε·g·(y − m_i)   →   ekstra kick  θ = ε·K·L·(y(i) − m_i).

Yani modülasyonun kapalı yörüngeye TÜM birinci-mertebe etkisi, quad i'de
oturan TEK bir kick'tir; Green fonksiyonunun gradyan pertürbasyonu **rank-1**
olduğundan (δG = G[:,i]·δk·G[i,:]) bu, bump dahil bütün kaynaklara göre
kesindir (pertürbatif el sallama değil):

    A(tüm BPM'ler) = G[:,i] · ε·K·L · (y(i) − m_i).

Sonuçlar:
1. **Her BPM'in tepkisi AYNI noktada sıfırlanır:** y(i) = m_i. Null,
   48-boyutlu bir sistemin değil, tek bir skalerin sıfırıdır.
2. **Simetrik/antisimetrik ayrımı ALAKASIZDIR:** o ayrım kolektif haritanın
   (R'nin) özelliğiydi; yerel null R'yi hiç ters çevirmez. Her quad'ın
   merkezi bağımsız ölçüldüğünde, "simetrik bileşen" sadece 48 bağımsız
   ölçümün bir lineer kombinasyonudur — hatası da per-quad hatanın izdüşümü.
3. **Null model-bağımsızdır:** sıfır-geçişinin YERİ, eğimden (G, β, faz)
   bağımsız. β-beat eğimi değiştirir, null'u kaydırmaz. (İnversiyonu %0.5
   β-beat'te öldüren şeyin buraya işlememesinin sebebi budur.)
4. **"Nefes" kirlilik değil, sinyalin kendisidir:** Sınıf 2'nin düşmanı olan
   büyük y(i) terimi, taramada x-ekseni olur; null onu kullanarak bulunur.

Eski duvarların tek tek neden uygulanmadığı:

| Duvar | Sınıf 1-2'de | Sınıf 3'te (BBA) |
|---|---|---|
| Kötü koşullanma (1/σ_min) | inversiyonda hata ×10⁴ | inversiyon YOK — per-quad skaler sıfır |
| β-beat (model hatası) | ΔR_model ≠ ΔR_true → felaket | null model-bağımsız (eğim değişir, yer değişmez) |
| Optik-nefes | genlik komşu-yörünge-domine | tarama ekseni; null'a etkisiz |
| BPM ofseti | mutlak okumayı kirletir | tüm büyüklükler FARK (mod aç−kapa; tarama farkı) → düşer; düzeltmede golden-orbit ile geri alınır |
| BPM gürültüsü | simetrikte <4 nm ister | yerel ölçüm; düz 1/√N_avg ortalanır, büyütme yok |

---

## 4. Simülasyonu nasıl kurdum (C++ detayları)

Zincir: `classic_bba_cpp_check.py` (null doğrulaması) → `classic_bba_full.py`
(47 quad × 2 düzlem uçtan uca). Tümü gerçek izleyiciyle (`integrator.cpp`,
GL4 semplektik + Thomas-BMT); analitik katman yalnız knob seçimi/ölçekleme
kılavuzu.

1. **Modülasyon:** `quad_dG[i] = +0.02` (per-quad kesirsel gradyan; C++'ta
   mevcuttu, integrator değişmedi). İstisna: cell-0 QF (`QUAD_F_MOD`)
   `quad_dG` okumaz (tuzak #8) → quad 0 sim'de ölçülemiyor; kaçıklığı 0
   alındı ve bu açıkça raporlanıyor (gerçek makinede o quad başka yoldan
   modüle edilir).
2. **Bump = komşu quad'a `quad_dy/dx` ek terimi.** Fizik özdeşliği: quad'a
   bindirilmiş dipol düzeltici bobini B alanını g(y−m)+ΔB = g(y−(m−ΔB/g))
   yapar → **etkin merkez kayması** — demet ve spin için quad'ı taşımakla
   birebir aynı. (Ayrıca `dipole_tilt` KULLANILMAZ: o, elektrik deflektör
   tilti değil "eşdeğer manyetik" radyal-B enjeksiyonudur → spin için EDM
   taklidi; tuzak #10.) Komşu adayları (±1, ±2) arasından, İLGİLİ DÜZLEMİN
   analitik R'sinde |R[i,j]| en büyük olan seçilir — dikeyde ±1, yatayda ±2
   çıkıyor (ilk koşumda x-bump'ın dikey R ile boyutlanması dx hatasını 4×
   şişirmişti; düzeltildi).
3. **Tarama:** 2 nokta, ±150 μm (tepki lineer; gürültüsüz C++'ta 2 nokta
   doğruyu tanımlar; gürültü çalışması çok-noktayı analitik katmanda yapar).
   Tarama ekseni **BPM i'nin kendi okuması** — bump'ın modelden kurulmuş
   olması bu yüzden bias yaratmaz (x-ekseni ölçülür, varsayılmaz).
4. **Yörünge okuma — kritik ders:** yörünge, ideal eksenden fırlatılıp zaman
   ortalamasıyla OKUNMAZ; parçacık önce **4D kapalı yörüngeye** oturtulur
   (`find_co_4d`), sonra okunur. İdeal-eksen fırlatma ~0.2 μm betatron
   artığı bırakır ve bu, null kestiriminde 1/eğim ≈ 25× büyüyerek ~5 μm
   sahte bias verir — ilk C++ koşumu tam bunu gösterdi (bias RMS 5.8 μm),
   CO-oturtmalı okuma 0.125 μm'e düşürdü (46×). Gerçek makinedeki karşılığı:
   okuma çok-turluk ortalamayla doğal olarak betatron-arındırılmıştır.
5. **Null kestirimi:** her BPM için A_b = s_b·x + a_b doğrusu; ortak sıfır,
   w=s_b² ağırlıklı en-küçük-kareler ile (eğimi büyük BPM'ler daha çok söz
   sahibi). 48 BPM'in hepsi kullanılır.
6. **Düzeltme:** kestirilen merkezler `quad_dx/dy`'den düşülür (madde 2'deki
   özdeşlik gereği bu, quad-üstü düzeltici bobinle birebir aynı işlem).
7. **Nihai metrik:** kalan sahte-EDM **spin izleyicisiyle doğrudan** ölçülür
   (`kmod_drivers/fast_est.fast_measure`: 4D-CO + model-fit; p=2.00
   doğrulamalı estimatör). f = A·σ² formül kestirimi SONUÇ OLARAK KULLANILMAZ.

**Sistematikler** (süren koşum; `--bbeat 0.01 --bpm-offset 100e-6
--bpm-noise 1e-6`): β-beat per-quad statik `quad_dG` olarak **dinamiğe
gömülü** (modülasyonun üstüne biner); BPM ofseti okuma katmanında (statik →
tüm farklarda düşer; düzeltme golden-orbit'e sürülünce tamamen iptal — sentetik
testte 0 nm); gürültü okuma katmanında (nokta başına √2·σ_n/√N_avg; 60
gerçekleme + her N_avg'de C++ spin çapası).

---

## 5. Şu ana kadarki C++ sonuçları

| Test | Sonuç |
|---|---|
| Null doğrulaması (3 quad, CO-oturtmalı, gürültüsüz) | bias −0.006/+0.167/−0.137 μm (RMS **0.125 μm**) |
| Uçtan uca, ilk geçiş (47 quad × 2 düzlem; x-bump hatalı ölçekli) | merkez hataları dy 0.19 μm (sym 0.16), dx 0.83 μm (sym 0.49); **ham f 356× → düzeltme sonrası 1.62× hedef (220× bastırma)** |
| x-bump düzeltmesi + β-beat %1 + ofset 100 μm + gürültü | koşuyor → `separation_bba_testleri.md §5.2` |

Not: 1.62× hedef, **iterasyonsuz tek geçişin ve hatalı x-ölçeklemenin**
sonucu. f ∝ dx·dy olduğundan dx hatasının dy seviyesine inmesi (düzeltilen
koşum) kalan f'i ~(0.16/0.49) oranında, hedefin altına taşımalı — bu bir
beklentidir, C++ sonucu gelince yerini ölçüme bırakır.

---

## 6. Analitik kılavuz (SONUÇ DEĞİL — nerede ne beklediğimizi söyler)

`classic_bba_sim.py` (per-quad Twiss, nefes dahil, 48 quad, BPM ofset 100 μm):
- simetrik = antisimetrik hassasiyet (0.32 ≈ 0.31 μm @ σ_pt=0.14 μm) —
  körlük yok;
- β-beat %1 **şeffaf** (0.32 → 0.31 μm);
- hata saf 1/√N_avg (2.8 → 0.45 → 0.11 μm; ~0.1 μm'e kadar taban yok);
- null hassasiyeti ≈ birkaç × σ_okuma/eğim; eğim ~ ε×(orbit kazancı) olduğundan
  ε=%2'de "okumanın ~25 katı" kuralı iş görür (0.14 μm okuma → ~0.4 μm merkez).
- **Kurgu tuzağı:** bump R⁻¹e_i ("yalnız i'de oynat") ile kurulursa simetrik
  yönlerdeki ~mm'lik dev düzeltici desenleri model tutarsızlıklarını
  amplifiye edip ~40 μm sahte bias verir. Gerçek BBA gibi yerel/mütevazı
  bump kullanınca kayboluyor. (Aynı aile: ideal-eksen 4-parçacık ortalaması
  CO-kaynaklı ortak betatronu söndüremez, −3×10⁻⁴ artefaktına düşer.)

---

## 7. Bunu neden daha önce görmedik? (dürüst muhasebe)

1. **No-go'lar sınıf-özgüldü ama sınıf-bağımsız cümlelerle kaydedildi.**
   "Simetrik mod orbit'ten ölçülemez" hükmünün kanıtı Sınıf 1 (inversiyon) ve
   Sınıf 2 (genlik) içindi; Sınıf 3 hiç koşulmadı. Negatif sonuç, hangi ölçüm
   sınıfını kapsadığı açıkça yazılmadan genellenince kör nokta doğdu.
2. **Dal erken kapandı:** per-quad K-mod fikri "operasyonel olarak ağır"
   işaretlenmişti (README §19.2) ve genlik-okuma varyantı nefesle ölünce
   (`squid_bpm_test §7`) TÜM per-quad ailesi kapanmış sayıldı. Oysa nefes,
   ailenin yalnız genlik-okuyan üyesini öldürüyordu.
3. **Çıkışı sağlayan şey dış itiraz + "lafla değil simülasyonla" ilkesi:**
   işbirliği yorumundaki plan (K-mod + 1 μm göreli BPM + HLS), kullanıcının
   "belki metot gerçekten çalışıyordur, sayıları görelim" ısrarıyla T5'e
   dönüştü. Analitik prototip yönü gösterdi; hüküm C++'la verildi.
4. **Ders (yönteme):** her negatif sonuç, "hangi ölçüm sınıfı için" etiketiyle
   kaydedilmeli; bir sınıfın ölümü aileyi öldürmez.

---

## 8. Sınırlar ve açık işler

- **Donanım (simülasyon dışı, muhtemel gerçek taban):** gradyan modülasyonu
  sırasında manyetik merkezin oynaması (histerezis, ısınma) — gerçek
  makinelerde ~μm görülebilir; BBA'nın ölçtüğü şey "modülasyon-altındaki"
  merkezdir. Ayrıca quad 0 (sim-kısıtı), tilt/skew (ayrı kanal, ~0.3 mrad),
  sekstüpolsüz lineer model.
- **Zaman bütçesi (kaba):** nokta başına N_avg=10⁴ tur ≈ 45 ms demet süresi;
  48 quad × 2 düzlem × birkaç nokta → demet-zamanı önemsiz; gerçek süreyi
  güç kaynağı ayar/oturma süreleri belirler. t=0 komisyonlamada rahat.
- **Makale:** ana iddia revize edilecek (kullanıcı onayıyla): "orbit'ten
  ölçülemez" → "global/tek-atış sınıflar ölçemez (duvarlar nicel); yerel
  null-arayan sınıf ölçer ve hedefe taşır". Omarov'un §9 boşluğu böylece
  iki yönde kapanır: ayrım-sürme tek başına yetmez (simetriğe kör), ama aynı
  donanımla (K-mod + BPM) yapılan null-BBA yeter.

## 9. Reprodüksiyon

```bash
python3 classic_bba_sim.py --seeds 5                 # analitik kılavuz
python3 classic_bba_cpp_check.py -w 4                # C++ null doğrulaması (~25 dk)
python3 classic_bba_full.py -w 4                     # uçtan uca + sistematikler (~3 s)
# sonuçlar: kmod_drivers/paper_runs_results.json [bba_cpp_check, bba_full, bba_full_syst]
```
