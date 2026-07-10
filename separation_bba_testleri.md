# separation_bba_testleri.md — İşbirliği yorumu (CR-ayrım + HLS planı) ve test programı T1–T5

> **Durum (2026-07): AKTİF test programı.** Bir işbirliği üyesinin yorumu
> (aşağıda) üzerine yapılan tartışmanın kaydı ve karara bağlanan simülasyon
> testleri. Bağımsız okunur. İlgili: `omarov.md §4,§9`, `akilli_duzeltme.md
> §6.13`, `makale_orbit_bastirma.md/.tex`, `literatur/ground_motion.txt`
> (Shiltsev, ATL/HLS derlemesi).

---

## 1. Yorumun özeti (işbirliği, 2026-07)

- Omarov Fig. 9(a): 10 μm → ~10⁻²⁵ e·cm (ham; "really bad").
- Fig. 9(c): 10 μm quad kaçıklığı → ~200 μm CR-demet ayrımı. Plan: **göreli
  ayrım çözünürlüğü 1 μm** (spec 10 μm); ayrım "her yerde <10 μm"e sürülecek.
  Göreli ölçüm mutlaktan çok kolay; BPM'ler iki demetin **27.5 ns arayla**
  geldiği noktalara konacak (aynı elektronik iki demeti ayrı zamanda okur →
  BPM ofseti ayrımda ortak-mod → düşer).
- Konum kararlılığı: her quad ve BPM **HLS** ile izlenecek (SLS'te işletildi,
  çözünürlük 0.5 μm).

## 2. Tartışmanın kayda geçen sonuçları

### 2.1 Örtüşmeler (çelişki yok)
- Ham ölçek ve Fig. 9(c) ölçeği bizim simülasyonla birebir (10 μm →
  **174 μm** ayrım; `paper_runs.py crsep`).
- "Relative ≫ absolute" = bizim drift/dualite bulgumuz; 27.5 ns numarası
  ayrımı BPM-ofsetinden arındırır (modelimizle tutarlı).
- HLS **mekanik** ölçüm olduğundan yörüngenin simetrik körlüğünden muaftır:
  simetrik desenin **zamansal kaymasını** 0.5 μm'de gerçekten sınırlar.

### 2.2 Bilineer kaldıraç (kullanıcı içgörüsü — T3'ün konusu)
f ∝ σ_x·σ_y olduğundan TEK düzlemi mıhlamak f'i o düzlemin artığıyla
**lineer** bastırır. Hedef σ_x·σ_y ≲ (1.3 μm)² ≈ 1.7 μm²:
σ_y = 0.5 μm (HLS-sınıfı) + σ_x = 10 μm → 5 μm² ≈ 3× hedef;
σ_y ≈ 0.2 μm → hedef. **Asimetrik tolerans** meşru bir kaçış adayı;
dikey düzlem tam da HLS + yerçekimi-referanslı hizalamanın güçlü olduğu yer.
Çekince: kaldıraç **manyetik** merkezin dikey bileşeni içindir; HLS t=0
hizalamasını değil sadece drift'i sınırlar.

### 2.3 SLS'te HLS neden yeterli, bizde neden tek başına değil
Işık kaynağının gereksinimi foton-demeti **kararlılığı** (göreli, sub-μm,
dakika-ay); HLS + hızlı orbit feedback + periyodik BBA üçlüsü bunu çözer;
mutlak manyetik-merkez birkaç μm şaşsa LOCO'yla düzeltilir optik hatası
doğurur, fizik sinyali taklit etmez. pEDM'de ise belirli bir **simetrik
MUTLAK** merkez-ofset kombinasyonu doğrudan sinyal taklidi üretir ve demet
ona kördür — bu gereksinim sınıfının ışık kaynağında karşılığı yok.

### 2.4 ⚠️ dipole_tilt TUZAĞI (kod okuması, integrator.cpp:153-170)
`dipole_tilt`, elektrik deflektörün düzlem-tilti DEĞİL; "eşdeğer manyetik
deflektör" tilti olarak yazılmış: ΔB_r = B_eq·sin(φ), B_eq = p/(e·R0)
**radyal-B enjeksiyonu**. Yörünge için gerçek E-tiltiyle ~eşdeğer; **spin
için değil**: magic momentumda dikey-E spini (momentuma göre) döndürmez,
radyal-B ise MDM üzerinden doğrudan dikey presesyon verir (EDM'in klasik
birinci-sınıf taklidi). → `dipole_tilt` COD çalışmaları için uygun,
**spin-duyarlı işlerde/düzeltici olarak KULLANILMAZ.** (CLAUDE.md tuzak #10.)

### 2.5 Düzeltici ≡ quad_dy (kritik eşdeğerlik)
Quad'a bindirilmiş dipol düzeltici (işbirliğinin gerçek planı, omarov §4):
B_x = g(y−dy) + ΔB = g·(y − (dy − ΔB/g)) → **etkin manyetik merkezi ΔB/g
kaydırmakla özdeş** — hem yörünge hem spin için. Simülasyonda düzeltici =
`quad_dy`'ye ek terim; `dipole_tilt` gerekmez. "Düzeltici vs taşıyıcı"
ayrımı lineer alanda fiziksel değil; fark yalnız **ayarın neyle seçildiği**.

### 2.6 "Sürme = kılık değiştirmiş inversiyon" (T2'nin çerçevesi)
48 knob ≡ 48 etkin merkez kaydırması; ayrım = ΔR_yön·dy_etkin. Gürültüsüz
lineer modelde ayrımı 48 BPM'de TAM sıfırlamak dy_etkin=0'ı (mükemmel
hizalamayı) zorlar — **ideal dünyada plan çalışır.** Duvar: sıfırlama
döngüsü gözlenebilirin pseudo-tersini uygular; simetrik modların
ayrım-tekil-değerleri küçük → **1 μm'lik çözünürlük tabanı simetrik yönde
büyütülür** ya da (regularize sürüşte) simetrik bileşen hiç dokunulmadan
kalır. T2 bu kalıntıyı ölçer.

### 2.7 LOCO ≠ BBA (t=0 sorusu — T5'in konusu)
- **LOCO** corrector→BPM tepki matrisini fit'ler → **gradyan** hataları/BPM
  kazançları; sekstüpolsüz lineer latiste quad ofsetleri R'ye hiç girmez →
  LOCO ofsetleri (ortak ofset dahil) ilkece göremez.
- **Klasik BBA** başka sınıf: quad gücünü modüle et + demeti YEREL bump'la
  tara + tepkinin **null**'unu bul → merkez, global inversiyon OLMADAN,
  quad-quad bağımsız ölçülür. Projede öldürülenler genlik-okuma (nefes) ve
  global inversiyonlardı; **null-arayan iteratif klasik BBA hiç test
  edilmedi.** Nefes ona *bias* olarak girer (null, mevcut global yörüngeyle
  orantılı kayar) ama iterasyonla yörünge küçüldükçe bias da küçülür.
  Per-quad ~1–2 μm bağımsız hata → σ_sym ~1–2 μm → f ~0.6–2.4× hedef:
  **plan + t=0 klasik BBA gerçekten çalışıyor olabilir. Karar T5'te.**

## 3. Test programı (öncelik sırasıyla)

| # | Test | Soru | Yöntem | Durum |
|---|------|------|--------|-------|
| **T5** | **Klasik BBA (null-arayan)** | t=0'da simetrik mod dahil merkezler ~1 μm'e mıhlanır mı? | Analitik per-quad Twiss (nefes dahil); 48 quad × (modülasyon + yerel bump taraması + null); BPM ofset 100 μm + gürültü; metrik: sym/anti merkez-hata RMS. C++ null-doğrulaması ayrı betikte. | **✅ POZİTİF (aşağıda §5)** |
| T1 | Ortak ofset (Duvar-2'nin saf gösterimi) | Ayrım≈0 iken f≠0 | dx=dy=c uniform; f=A·c² öngörüsüyle karşılaştır; quad'ları geri taşı → f→0 | bekliyor |
| T2 | Ayrım-sürme vs hizalama | 1 μm çözünürlükle sürülen makinede kalan simetrik dy_etkin ve f | Düzeltici=quad_dy ek terimi; CW/CCW ayrımını 48 BPM'de sıfırla; f'i C++ ile ölç; kontrol: antisim bileşeni desenden çıkar | bekliyor |
| T2b | Düzeltici gücü | 22 μm'lik yüksek-k ayrım izini sürmek ne kadar kick ister (1/G_k)? | T2'nin yan çıktısı | bekliyor |
| T3 | Dikey-düzlem (HLS) kaldıracı | f(σx=10, σy) ∝ σy lineer mi; HLS-sınıfı kontrol kaç × alır? | σy ∈ {10,2,0.5,0.2} μm taraması, C++ estimatör | bekliyor |

**Karar ağacı:** T5 pozitif (σ_sym ≲1–2 μm'e yakınsıyor) → makale "hangi
ölçüm sınıfı çalışır (yerel null), hangisi çalışmaz (global inversiyon +
sürme)" makalesine döner; işbirliğine pozitif haber. T5 negatif (bias
yakınsamıyor / taban ≫1 μm) → mevcut sınır-teoremi güçlenir ve "klasik BBA
da dahil" denebilir. Her iki hâlde yayınlanabilir; T1/T2 destek figürleri.

## 5. T5 ANALİTİK ÖN-İNCELEME (yol gösterici — SONUÇ DEĞİL)

> **⚠️ KULLANICI KURALI (2026-07): analitik çözümler yalnız YOL GÖSTERİCİDİR;
> sonuç olarak yalnız C++ simülasyonu sunulur.** Bu bölümün tablosu, C++
> testinin nerede aranacağını gösteren ön-incelemedir; nihai hüküm §5.1'deki
> (ve devamındaki) C++ ölçümlerine aittir.

**Teorik netleşme (rank-1 yerellik).** Quad i'nin modülasyonunun kapalı
yörüngeye birinci-mertebe etkisi TEK ve YEREL bir kick'tir:
θ = ε·K·L·(y(i) − m_i) — Green fonksiyonunun gradyan pertürbasyonu rank-1
olduğundan (δG = G[:,i]·δk·G[i,:]) bu, bump dahil TÜM kaynaklara göre
kesindir. Yani null tam "demet = merkez"de oturur; komşuların yörüngesi
null'u kaydırmaz. **"Nefes"**, sabit yörüngede genlik okuyanın
(y(i) − m_i) içindeki dev global-yörünge terimini kalibrasyon sanmasıydı;
null-arayan tarama bunu tanım gereği aşar. (Projede öldürülen genlik-okuma
ve global-inversiyon sınıfları bu sınıfı KAPSAMIYORDU.)

**Kurgu tuzağı (kayda değer):** bump R⁻¹·e_i ("tek noktada oynat") ile
kurulursa simetrik yönlerdeki ~mm'lik dev düzeltici desenleri, formül-R'nin
küçük iç tutarsızlıklarını amplifiye edip null'u ~40 μm kaydırıyor (ilk
koşum). Gerçekçi TEK-komşu-düzeltici bump ile (tarama ekseni zaten BPM
i'den okunur) sorun yok. Gerçek BBA'da da yerel bump kullanılır.

**Sonuçlar** (σ_mis=10 μm, BPM ofset 100 μm, tek-atış gürültü 1 μm,
ε=%2, 9 noktalı ±150 μm tarama, 3 seed; `classic_bba_sim.py`):

| Ayar | merkez-hata RMS | **SİMETRİK** | antisim | kalan sahte-EDM |
|---|---|---|---|---|
| N_avg=1 (σ_pt=1.4 μm) | 4.4 μm | 3.3 μm | 2.8 μm | ~260× hedef |
| N_avg=100 (σ_pt=0.14 μm) | 0.45 μm | **0.32 μm** | 0.31 μm | **2.4× hedef** |
| N_avg=10⁴ (σ_pt=14 nm) | 0.16 μm | **0.11 μm** | 0.11 μm | **0.40× hedef ✓** |
| N_avg=100 + **β-beat %1** | 0.45 μm | **0.31 μm** | 0.33 μm | 2.4× hedef |

**Üç kritik bulgu:**
1. **Simetrik = antisimetrik hassasiyet** (0.32 ≈ 0.31 μm): körlük YOK —
   yerel null, inversiyonun 1/σ_min büyütmesini hiç içermiyor.
2. **β-beat ŞEFFAF** (0.32→0.31 μm; ΔR-inversiyonunda %0.5 → 283 μm idi):
   sıfır-geçişi optik-model bilgisi istemiyor.
3. **Saf gürültü-sınırlı** (∝1/√N_avg; ~0.1 μm'e kadar sistematik taban
   görülmedi) → entegrasyonla hedefin altına iniliyor.

**Analitik ön-hüküm (C++ ile sınanacak hipotez):** yerel null-arayan klasik
BBA lineer modelde no-go'nun dışında görünüyor — simetrik mod dahil merkezler
gürültü-sınırlı, β-beat-şeffaf. Makale iddiasının revizyonu ANCAK C++
doğrulamasından sonra düşünülür.

### 5.1 C++ DOĞRULAMASI (gerçek dinamik — ASIL SONUÇ BURASI)

**Koşum 1 (ideal-eksen fırlatmalı yörünge okuma; `classic_bba_cpp_check.py`
ilk sürüm):** 3 quad × 5 nokta × (mod aç/kapa), gürültüsüz. Sonuç: fit'ler
mükemmel lineer (artık ~0.0003 μm) ama null'lar gerçek merkezden kaymış —
**bias −3.9/−5.2/+7.7 μm (RMS 5.8 μm)**. Yani analitik tablo C++'a bu hâliyle
TAŞINMADI.

**Teşhis (hipotez):** `run_sim` yörüngeyi ideal eksenden fırlatıp zaman-
ortalamayla okur; betatron artığı okumaya ~0.2 μm kirlilik bırakır ve bu,
null kestiriminde 1/eğim ≈ 25× büyüyerek ~5 μm sahte bias verir — büyüklük
tutuyor.

**Koşum 2 (CO-oturtmalı okuma) — HİPOTEZ DOĞRULANDI, BIAS ÇÖKTÜ:**
aynı üç quad, gürültüsüz, yörüngeler 4D-CO'dan fırlatılarak okundu:

| quad | koşum-1 bias (ideal-eksen) | **koşum-2 bias (CO-oturtmalı)** |
|---|---|---|
| 5 | −3.852 μm | **−0.006 μm** |
| 20 | −5.223 μm | **+0.167 μm** |
| 33 | +7.666 μm | **−0.137 μm** |
| RMS | 5.8 μm | **0.125 μm** (46× düşüş) |

→ 5.8 μm tümüyle okuma artefaktıydı. **Gerçek dinamikte klasik-BBA null'u,
manyetik merkezi ~0.1 μm seviyesinde buluyor** (kalan: okuma hassasiyeti +
CO-arama artığı + ε² düzeyi; hangisi olduğu tam-zincir sonucunda önemsizleşir).
Operasyonel ders (gerçek makine karşılığı): null-tarama sırasında yörünge
okuması betatrondan arındırılmış olmalı (çok-turluk ortalama bunu doğal yapar).

**Koşum 3 — UÇTAN-UCA TAM ZİNCİR (`classic_bba_full.py`; 376 CO-oturtmalı
koşum + 2 spin ölçümü, ~3.1 saat): ✅ POZİTİF.**
47 quad × 2 düzlem BBA (2-noktalı tarama; cell-0 QF quad_dG okumadığından
quad 0 hariç, kaçıklığı 0 alındı) → kestirilen merkezler düşüldü (düzeltici ≡
merkez kayması) → kalan sahte-EDM **spin izleyicisiyle DOĞRUDAN** ölçüldü:

| Büyüklük | Değer (C++, seed 0, gürültüsüz) |
|---|---|
| Merkez-kestirim hatası dy | RMS 0.185 μm (**sym 0.156**, anti 0.099) |
| Merkez-kestirim hatası dx | RMS 0.834 μm (**sym 0.485**, anti 0.679) |
| Ham sahte-EDM | 3.56×10⁻⁷ rad/s (356× hedef) |
| **BBA-düzeltme sonrası** | **1.62×10⁻⁹ rad/s = 1.62× hedef** |
| **Bastırma** | **220×** |

**Hüküm (C++, uçtan uca):** null-arayan klasik BBA, gerçek dinamikte simetrik
mod dahil merkezleri sub-μm ölçüyor ve sahte-EDM'i hedefin kapı eşiğine
(1.6×) indiriyor — orbit-tabanlı yöntemler için kayıtlı no-go, bu ölçüm
sınıfını KAPSAMIYOR. Üstelik bu ilk geçiş: iterasyonsuz, 2-noktalı tarama,
x-düzlemi bump ölçeği kaba (dx hatası 4× daha büyük — iyileştirilebilir).
İkinci bir BBA geçişi/iyileştirilmiş x-taraması muhtemelen hedefin altına
iner (test edilecek).

**Dürüst sınırlar:** (i) gürültüsüz — istatistik kalemi üstüne gelir
(analitik kılavuz: saf gürültü-sınırlı, ortalamayla yenilir; C++ ile ayrıca
doğrulanmalı); (ii) tek seed; (iii) quad 0 sim-kısıtı; (iv) donanım
sistematikleri (modülasyonda merkez oynaması, histerezis) simülasyon dışı —
gerçek makinede muhtemel asıl taban; (v) tilt dahil değil.

**MAKALEYE ETKİSİ (kritik):** `makale_orbit_bastirma.tex`'in ana negatif
iddiası ("simetrik bileşen orbitten hiçbir standart teknikle gerekli seviyede
ölçülemez") bu sonuçla REVİZE EDİLMELİ: doğru sınıflandırma "global/tek-atış
yöntemler (inversiyon, genlik-okuma, ayrım-sürme) göremez; YEREL NULL-ARAYAN
sınıf (klasik BBA) görür". Revizyon kullanıcı onayıyla yapılacak.

**Gerçek-makine kalemleri (C++ pozitif çıksa bile makalede tartışılacak):**
gradyan modülasyonunda manyetik merkezin oynaması (histerezis/ısınma —
gerçek makinelerde ~μm olabilir; muhtemel asıl taban), x-düzlemi eş yöntemi,
tilt/skew, 48 quad × 2 düzlem zaman bütçesi (t=0'da sorun değil),
merkez-drift'i (HLS + bilineer kaldıraç, §2.2).

## 6. Reprodüksiyon

```bash
python3 classic_bba_sim.py --seeds 5                # T5 analitik (hızlı)
python3 classic_bba_sim.py --seeds 3 --bbeat 0.01   # β-beat şeffaflık
python3 classic_bba_sim.py --seeds 2 --navg 10000   # gürültü ölçekleme
python3 classic_bba_cpp_check.py -w 4               # C++ null-doğrulaması (~6 dk)
# T1-T3: kmod_drivers/paper_runs.py'a eklenecek modlarla (C++)
```
