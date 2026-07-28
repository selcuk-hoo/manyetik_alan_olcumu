# omarov.md — Omarov ve ark. (PRD 105, 032001, 2022) dikkatli okuma

> **Amaç:** "Comprehensive symmetric-hybrid ring design for a proton EDM
> experiment at below 10⁻²⁹ e·cm" makalesini, **özellikle quad hizalama hatası
> kaynaklı geometrik-faz/sahte-EDM'in pratikte nasıl kontrol edildiği** sorusu
> ekseninde özetler. Bizim simülasyon bulgularımızla (CCW≡CW+flip dejenerasyonu,
> simetrik orbit-kör artık, orbit-monitör = SQUID-BPM ikamesi) doğrudan
> karşılaştırır. Sayfa/denklem/figür referansları makaleye göredir (032001-N).
>
> Mevcut `omarov_symmetric_hybrid.md` özet/strateji belgesini **tamamlar**; bu
> dosya metne-dayalı (snippet değil) tam okumadır.
>
> Künye: Z. Omarov, H. Davoudiasl, **S. Hacıömeroğlu**, V. Lebedev, W. M. Morse,
> Y. K. Semertzidis, A. J. Silenko, E. J. Stephenson, R. Suleiman. PRD 105,
> 032001 (2022). (Selçuk yazarlardan.)

---

## 0. Bir cümlede

Sahte (EDM-benzeri) dikey spin presesyonunu domine eden sistematikler **simetrik-
hibrit latis** + **eşzamanlı CW/CCW (counter-rotating) depolama** + **quad polarite
değiştirme** + **CR demet ayrımının dipole-korektörlerle küçültülmesi** ile
hedefin (1 nrad/s ≈ 10⁻²⁹ e·cm) altına çekiliyor. **Spin-based alignment (SBA)**
asıl olarak E-alanı/vertical-velocity eksenini hizalar; **geometrik fazı doğrudan
düzeltmez** (onun için vertical-polarizasyon bunch'ları önerilir ama bu makalede
KULLANILMADI — "unused in this study").

---

## 1. Temel ilke ve neden hizalama kritik

`dS_y/dt ∝ E·η` (032001-4): dikey spin presesyon hızı protonun EDM'ini ölçer.
Ama MDM (manyetik moment) kuplajı EDM kuplajından **mertebelerce** büyük; bu
yüzden EM alanların **katı hizalama gereksinimi** var. Sahte EDM = hizalama/alan
kusurlarının ürettiği EDM-benzeri `dS_y/dt`.

Latis: 24 FODO, 800 m, elektrik bending + alternatif (güçlü) **manyetik** focusing
(quad gradyanı ~0.2 T/m). Manyetik focusing radyal-B'yi doğal olarak perdeler.

---

## 2. Geometrik faz (Sec. III D + App. E) — bizim simülasyonun birebir karşılığı

**Mekanizma:** Berry/geometrik faz = ardışık dönüşlerin **non-komütatifliğinden**
doğan ekstra spin presesyonu; "spin precession ∝ ardışık dönüş genliklerinin
**çarpımı**" (032001-10).

**σ² doğrulaması (Fig. 9a):** "tüm manyetik quadrupolleri rastgele rms σ ile (hem
x hem y) kaydırıp dikey presesyon hızının büyümesini gözleyerek **kare bağımlılık**
doğrulanır." → **Bu tam olarak bizim yaptığımız ve `sigma_olcekleme` testinde
p = 2.00 ile teyit ettiğimiz şey.** Aynı seed'ler her yerde kullanılınca `y=kx²`
fiti mükemmel (032001-9, Fig. 9 başlığı) — bizim per-seed p=2.00 ile birebir.

**Kritik nüans (032001-10):** "**Random misalignments of quads ALONE do not cause
vertical spin build-up**" — düz latis (sadece quad, bending yok) çalışmasıyla
gösterilmiş. Fig. 9a'daki etki, misalignment'ın **diğer sistematiklerle
karışımından** (vertical velocity + 2. derece etkiler) doğuyor. **Çelişki yok:**
bizim tam halkamız elektrik bending'i içerdiğinden bu karışımı zaten barındırır;
ölçtüğümüz σ² sahte EDM = Omarov'un Fig. 9a'sının aynısı.

**App. E — sahte dikey presesyon terimleri (önem sırasıyla):**
1. `dS_y/dt ∝ S_x·β_y·E_x` — "vertical velocity" / "twist" (Sec. III A). **Ana**
   sistematik; simetrik-hibrit latis bunu birkaç mertebe bastırır.
2. `dS_y/dt ∝ S_s·β_s·E_y` (Sec. III B).
3. `dS_y/dt ∝ S_s·β_y·E_s` — boylamsal polarizasyona (EDM aramasına) **doğrudan**
   kuplar; "**SBA ile çevirmek (circumvent) kolay değil**" (032001-17).
   Omarov'un dayanağı: `∮E_s ds = 0` (statik alan net hızlanma yapmaz) → etki
   yalnız non-statik boylamsal E için; yüksek-hassasiyet izleme `E_s < 5 V/m` için
   sahte EDM < 1 nrad/s gösteriyor.
4. `dS_y/dt ∝ S_x·β_x·E_y`.

---

## 3. İptal reçetesi (App. C) — gerçek EDM nasıl ayıklanıyor

**Eq. (C1):** `(dS_y/dt)_EDM = ½(dS_y/dt)_CW − ½(dS_y/dt)_CCW`.
Gerçek EDM, CW ve CCW dikey presesyonlarının **FARKI** (pozitif-helisite CR
demetler için ters işaret). → **Bizim `edm_only` testimizle birebir:** EDMSwitch=1,
hizalamasız, CW=+9.74e-10, CCW=−9.88e-10, (CW−CCW)/2=+9.81e-10 → gerçek EDM TEK
(diferansiyel). ✓

**Eq. (C2) — polarite-switch ile 4'lü kombinasyon:**
```
(dS_y/dt)_EDM = ¼[CW − CCW]_Polarite1 + ¼[CW − CCW]_Polarite2
```
Polarite-switch = tüm manyetik quad akımlarının yönünü çevirmek → beta
fonksiyonlarını **faz kaydırır**; "beta'nın lokal değerine bağlı sistematiklerin
önemli kısmını bastırır" (032001-14, knob ii). İsteğe bağlı ek: quad gradyanı k'yı
değiştirip `dS_y/dt ∝ 1/k → 0` ekstrapolasyonu (Ref. [38], k-modülasyon fikri).

**Estimator inceliği (App. C, Eq. C3–C5) — basit eğim DEĞİL:**
Ham `S_y(t)` saf doğrusal değil; radyal spine sızıntı `dS_y/dt = η₀S_s + δ₀S_x`
(Eq. C3) ve `S_x` zamanla doğrusal aktığından `S_y ~ t²` (kuadratik) bir drift
kazanır. Doğru η₀ için önce `S_y ← S_y − δ₀Γt²S_s/2` düzeltmesi yapılır (δ₀, Γ
spin verisinden fit; Fig. 19), sonra η₀ çekilir. **Toplam kombinasyon (Fig. 16):
η₀ < 1 nrad/s.**

> **Bizim estimator'la ilişki:** Biz 4D-CO + model-fit (cos/sin + seküler) ile
> seküler eğimi çekiyoruz; bu, betatron + salınımı söndürüp σ² geometrik fazı
> verir (p=2.00). Omarov'un δ₀Γt² düzeltmesi, **gerçek EDM ile birlikte** ölçerken
> radyal-sızıntıyı ayıklamak için; biz EDMSwitch=0 ile saf sahte EDM ölçtüğümüzden
> bu terim bizde küçük (null testi: hizalamasız tüm kanallar ~1e-11).

---

## 4. Asıl kontrol knob'u: CR demet ayrımı (Sec. III C + IV A)

Geometrik faz ve E-alan sistematiklerinin çoğu **CR demet ayrımına** (CW ve CCW
kapalı yörüngelerinin farkı) bağlı. Mantık zinciri:

- "Küçük quad misalignment bile **büyük CR demet ayrımı** yaratır" (Fig. 9c);
  ayrım, quad'lardaki **dipole korektör B-alanlarıyla 100 μm'nin çok altına**
  ince ayarlanır (032001-10).
- "Geometrik faz kaynaklı EDM-benzeri presesyon, **CR ayrımı birkaç yüz μm
  altındayken önemsiz**" (= quad misalignment ~birkaç μm) (032001-10).
- Gereksinim (Sec. IV A): CR demetler **5 μm içinde örtüşmeli** (toplam 10 μm),
  **50 μm genel dikey kapalı-yörünge planaritesi**.
- 10 μm planarite ya **mekanik** (su terazisi, Ref. [50,51]) ya da **"SQUID-tabanlı
  BPM benzeri teknoloji (10 nm/√Hz çözünürlük, Ref. [64])"** ile (dipnot 8).

**Ayrım nasıl ÖLÇÜLÜR (032001-13):**
- μm-seviyesi dikey ayrım → birkaç cm uzakta **pT-seviyesi radyal B** üretir (CR
  demetlerden); manyetik pickup'larla ölçülür.
- SNR için **K-modülasyon** (quad alanları 1–10 kHz'de %1).
- Flux-gate magnetometre (birkaç pT/√Hz) **veya** SQUID-BPM (10 fT/√Hz, Ref. [64]).

> **Bizim orbit-monitör buraya oturuyor:** CR ayrımı = kapalı-yörünge farkı.
> Omarov bunu SQUID-BPM / manyetik-pickup + K-mod ile ölçüyor. **Bizim önerimiz:
> standart BPM ile aynı ayrımı ucuz ölçmek** → korektörleri sürmek için SQUID-BPM
> ikamesi. Değer önermesinin tam yeri burası.

---

## 5. SBA (spin-based alignment) — ne yapar, ne YAPMAZ

"Spin ölçümleriyle E-alanlarını kontrol etme" = SBA (032001-10). Halkayı yüksek
mertebeden tesviye eder; çeşitli bunch polarizasyon kombinasyonları kullanır:

- **Radyal polarizasyon** → vertical-velocity'ye duyarlı → **dikey orbit
  corrugation feedback'i**. (SBA'nın asıl iş gören kolu; DM/DE ve vertical
  velocity ekseni.)
- **Vertical polarizasyon** → "**geometrik fazı ve henüz bilinmeyen sistematikleri
  TEST etmek**" için kullanılabilir — ama knob (vi)'de açıkça: "**unused in this
  study**" (bu çalışmada kullanılmadı).

→ **Senin saptaman doğru:** SBA bu makalede geometrik-faz/sahte-EDM'i **düzeltmiyor**;
asıl iş E-alan/vertical-velocity hizalaması. Geometrik faz iptali **CW+CCW+polarite
kombinasyonu + CR-ayrım küçültme** ile yapılıyor. SBA'nın geometrik-faz kolu
(vertical-pol) gelecek-iş olarak işaretli.

---

## 6. "Net olmayan" yerler (senin sezgin) + bizim bulgularla bağlantı

Makale "sistematikler hedefin altında" sonucunu **toplam kombinasyon (Fig. 16)** ve
**birkaç μm misalignment** senaryosunda gösteriyor; ancak **geometrik-fazın
alt-uzay yapısı** açık bırakılmış. Net olmayan üç nokta:

1. **Polarite-switch gerçekten bağımsız 4. kol mu?** Eq. (C2) CW, CCW ve iki
   polariteyi 4 ayrı ölçüm gibi birleştiriyor. **Bizim bulgumuz (ve git'teki Test
   7, "(CW,n)≡(CCW,f) ayna simetrisi"):** idealize periyodik FODO'da **CCW ≡
   CW+polarite-flip özdeş** → polarite-switch CW/CCW'nin ötesinde **ek iptal
   vermiyor**; 4'lü kombinasyon 2'liye çöküyor. Simetrik-hibrit gerçek halkada bu
   dejenerasyonun ne kadar kırıldığı makalede **nicelenmiş değil**. Eğer kırılmıyorsa
   Eq. (C2)'nin ekstra bastırması beklenenden az olur.

   > **⛔ DÜZELTME (2026-07-15, doğrudan C++ testi — bkz. §11):** Yukarıdaki
   > çıkarımın YÖNÜ YANLIŞTI. Dejenerasyon (CCW≡CW+flip) Eq. C2'yi zayıflatmaz,
   > tam tersine **egzakt yapar**: [CW−CCW]₊g = −[CW−CCW]₋g olduğundan 4'lü
   > kombinasyon sahte EDM'i **birebir sıfırlar** (ölçüm: 871× → −0.02×).
   > Gerçek EDM dejenerasyona uymadığı (E-alan kökenli: terslemede tek,
   > flip'te çift) için hayatta kalır. "Ek iptal vermiyor" ifadesi 4'lünün
   > 2'liden farksız olduğu izlenimi veriyordu — oysa 2'li 87× artık bırakır,
   > 4'lü sıfırlar.

2. **Simetrik (orbit-kör) artık.** Omarov kontrolü **CR-ayrım** üzerinden kuruyor;
   ama simetrik misalignment (QF/QD aynı-işaret, yüksek-k) **küçük CR-ayrımı**
   üretir (orbit-kör) → dipole-korektör/SQUID-BPM **göremez**. Bizim ölçtüğümüz:
   CW/CCW telafisi sonrası artığın **orbit-görünür (antisim) kısmı orbit-düzeltmeyle
   7.7× düşüyor**, ama **orbit-kör simetrik artık gerçek EDM'in 62 katında takılıyor**
   ve onu ne biz, ne SQUID-BPM, ne CW/CCW kapatıyor. Omarov "birkaç μm misalignment
   için toplam < 1 nrad/s" diyor ama **simetrik alt-uzayı ayrı izole etmiyor** —
   "net olmayan" tam burası.

3. **`S_s·β_y·E_s` terimi (App. E #3).** Boylamsal polarizasyona (EDM kanalı)
   doğrudan kuplar, "SBA ile çevirmek kolay değil", ortalaması sıfır değil;
   `∮E_s ds=0` enerji-korunum argümanına dayanılıyor. Bu, statik-alan varsayımına
   bağlı ince bir argüman (non-statik B-akı değişimi için ayrıca < 1 nrad/s
   gösterilmiş, ama varsayıma duyarlı).

---

## 7. Bizim çalışmaya çıkarımlar (dürüst konumlandırma)

- **Ölçtüğümüz sahte EDM = Omarov'un geometrik fazı** (σ², p=2.00; Fig. 9a karşılığı).
  Estimator'ımız doğru (kuadratik yakalıyor, lineer kaçak yok).
- **Gerçek EDM ölçeği** (η=1.88e-15 → 9.8e-10 rad/s) Omarov'un 1 nrad/s hedefiyle
  tutarlı; CW/CCW işareti (TEK/diferansiyel) Eq. (C1) ile birebir.
- **Bizim orbit-monitör = SQUID-BPM ikamesi**, SBA'nın tamamlayıcısı DEĞİL.
  Omarov geometrik fazı **CR-ayrım küçültme** (dipole korektör + SQUID-BPM/pickup)
  ile kontrol ediyor; biz aynı ayrımı **standart BPM ile ucuz** ölçmeyi öneriyoruz.
  Doğrudan testimiz: orbit-düzeltme EDM-kanalı kirliliğini **7.7×** azaltıyor.
- **Paylaşılan sınır:** simetrik orbit-kör artık (62×) hem bizim hem SQUID-BPM hem
  SBA hem CW/CCW için kör. Bu, bizim "no-go"muzun (simetrik alt-uzay orbit-tarafından
  indirgenemez) Omarov tarafında **açıkça nicelenmemiş** karşılığı.
- **Özgün katkı adayı:** Omarov'un geometrik-faz kontrolünün alt-uzay yapısını
  (orbit-görünür antisim vs orbit-kör simetrik) **nicelleştirmek** ve standart-BPM
  monitörünün antisim kısımdaki değerini (7.7×) + simetrik artığın paylaşılan
  sınırını göstermek. Yani "ucuz SQUID ikamesi" + "simetrik artık no-go'sunun
  nicelenmesi" — ikisi birlikte.

---

## 8. Yeniden üretim (referans figür/denklem eşlemesi)

| Omarov | Karşılık / bizim test |
|--------|-----------------------|
| Fig. 9(a) σ² geometrik faz | `sigma_olcekleme` (p=2.00) |
| Fig. 9(b) CW+CCW+polarite iptali | `cwccw_telafi` (telafi 3.4×) + dejenerasyon bulgusu |
| Fig. 9(c) CR-ayrım ↔ σ | orbit-monitör ölçüm hedefi |
| Eq. (C1) EDM=½(CW−CCW) | `edm_only` (gerçek EDM TEK, 9.8e-10) |
| Eq. (C2) 4'lü polarite | git Test 7 + bizim dejenerasyon (CCW≡CW+flip) |
| Eq. (C3–C5) δ₀Γt² estimator | bizim 4D-CO + model-fit (EDM=0'da terim küçük) |
| Sec. IV A: 5/10/50 μm, SQUID-BPM | orbit-monitör = ucuz ikame |
| App. E #1 vertical velocity | ana sistematik; simetrik latis bastırır |
| App. E #3 S_s·β_y·E_s | "SBA ile kolay değil"; ∮E_s ds=0 argümanı |

---

## 9. KRİTİK BOŞLUK: CR-ayrım ÖLÇÜMÜ önerildi ama test EDİLMEDİ

Geometrik-faz kontrolünün tüm zinciri **CR demet ayrımını ölçüp dipole-korektörlerle
küçültmeye** dayanıyor (§4). Ama bu ölçümün kendisi makalede **gösterilmiyor**:

**Makalede VAR olan (öneri düzeyinde, 032001-13):**
- Konum, halka çevresinde **48 bin'e** bölünüp örnekleniyor (Fig. 15; 24 FODO =
  48 quad = 48 azimut konumu → fiilen **48-BPM** örneklemesi).
- Ayrım "magnetic pickup'larla ölçülebilir"; μm-ayrım → birkaç cm'de pT-radyal B.
- SNR için **K-modülasyon** (quad alanları %1, 1–10 kHz) gerekiyor.
- Enstrüman: flux-gate magnetometre (birkaç pT/√Hz) **veya** SQUID-BPM
  (10 fT/√Hz, Ref. [64]).

**Makalede TEST EDİLMEYEN (boşluk):**
1. **48-BPM/pickup'larla CR-ayrım profilinin doğru GERİ-ÇATILABİLDİĞİ** hiç simüle
   edilmemiş. Reconstruction prosedürü (BPM ofsetleri, gürültü, K-mod çözümü →
   ayrım haritası) yok. Tek paragraflık bir enstrüman önerisi.
2. **Ölçümün doğruluğu** (ayrım gerçekten hedef seviyede ölçülebilir mi) test
   edilmemiş. Makalenin testleri **fiziği** doğruluyor (σ²; CW+CCW+polarite iptali
   <1 nrad/s; σ=100μm/>1mm ayrım → <1nrad/s) — yani "EĞER ayrımı küçültürsen
   geometrik faz düşer". Ama "ayrımı **ölçüp** küçültme" enstrüman zincirinin
   kendisi nicel sınanmamış.
3. **Simetrik (orbit-kör) bileşene körlük ihtimali açıkça ele alınmamış.** Ayrım
   ölçümü kapalı-yörünge-farkı tabanlı; simetrik misalignment küçük ayrım üretir
   (yüksek-k, G_k ∝ 1/|Q²−k²| ile bastırılır) → 48-BPM/SQUID-BPM bu bileşeni
   göremeyebilir. Makale bunu test etmiyor.

> **§9.3 DOĞRUDAN ÖLÇÜLDÜ (2026-06, gerçek C++; `/tmp/akilli_duzeltme/cr_separation.py`):**
> CW ve CCW kapalı yörüngelerini ayrı çıkarıp **CR-ayrım = COD_CW − COD_CCW**'yi
> hesapladık (direction=±1, pz işareti ile gerçek karşı-dönüş; CCW orbiti CW'nin
> yaklaşık ayna-tersi, oran ≈ −0.4). Simetrik (QF/QD aynı-işaret) vs antisimetrik
> 10 μm desenler için **bastırma (antisim/sim):**
>
> | gözlenebilir | bastırma |
> |---|---|
> | tek-yön COD | 3.8× |
> | **CR-ayrım (Omarov)** | **4.5×** (biraz DAHA kör) |
>
> **Sonuç:** CR-ayrım, simetrik alt-uzaya **tek-yön yörünge KADAR (hatta biraz
> daha) kördür** — *ek pencere açmaz.* Yani Omarov'un ayrım-ölçüm zinciri, sıradan
> yörünge gibi simetrik (sahte-EDM'i süren) bileşeni göremez; ayrımı küçültmek
> simetrik geometrik fazı bırakır. **No-go CR-ayrıma birebir taşınır.** (Not: w=1
> jenerik simetrik desende COD bastırması yalnız ~4×; en-küçük-σ artıkta çok daha
> büyük olur, ama KRİTİK olan ORAN: CR/COD ≈ 1.) ![](/tmp/akilli_duzeltme/fig_kolb_crsep.png)

→ **Özetle Omarov, geometrik-faz "düzeltme knob'unun" (CR-ayrım) FİZİĞİNİ
kanıtlıyor; ama o knob'u sürecek ÖLÇÜMÜN (48-BPM/SQUID-BPM + K-mod reconstruction)
yapılabilirliğini ve simetrik-artığa körlüğünü açık bırakıyor.** Tam da bizim
çalışmanın oturduğu yer burası — ve **§9.3 körlüğü artık doğrudan gösterildi.**

---

## 10. NEREDE DURUYORUZ (bu oturum, make-or-break + Omarov karşılaştırması)

**Doğrulanmış estimator (kritik temel):** σ=10→5→2.5 μm ölçeklemesinde sahte EDM
**p = 2.00 ± 0.01** (her seed) → saf kuadratik geometrik faz, lineer kaçak YOK.
CO+model-fit yöntemimiz Omarov Fig. 9a'sını birebir üretiyor. (`/tmp/sigma_olcekleme.py`)

**Bu oturumun nicel zinciri (EDMSwitch ile, doğrulanmış):**

| Ölçüm | Değer | Script (/tmp) |
|-------|-------|---------------|
| Gerçek EDM (η=1.88e-15), TEK/diferansiyel | 9.81×10⁻¹⁰ rad/s | `edm_only.py` |
| 10 μm sahte EDM (seküler, σ²) | ~10⁻⁶ (worst ~6.5e-6) | `cwccw_ensemble.py` |
| CW/CCW telafisi (tek başına) | **3.4×** (artık 474× EDM) | `cwccw_telafi.py` |
| **Orbit-düzeltme kazancı** (antisim çıkar) | **7.7×** (artık 62× EDM) | `orbit_duzeltme.py` |
| Kalan simetrik orbit-kör artık | 62× EDM (6.05e-8) | `orbit_duzeltme.py` |
| Dejenerasyon: CCW ≡ CW+polarite-flip | özdeş (4-lü → 2-li) | `cwccw_validate` |

**Bizim konumumuz:** Orbit-monitör (standart BPM) = **SQUID-BPM'in ucuz ikamesi**,
SBA'nın tamamlayıcısı DEĞİL. Geometrik-faz sahte-EDM'in **orbit-görünür (antisim)
kısmını 7.7× temizliyor** (doğrudan test). **Yeni katkı:** Omarov'un §9'da açık
bıraktığı iki şeyi nicelliyoruz — (a) ayrım/orbit ölçümünün alt-uzay yapısı
(antisim görünür / simetrik kör), (b) simetrik orbit-kör artığın **paylaşılan
sınırı** (62×, hem bizim hem SQUID-BPM hem CW/CCW için kör).

**Omarov'un bu konuda durduğu yer:** Geometrik fazı CW+CCW+polarite + CR-ayrım
küçültmeyle hedefin altına indirdiğini **fiziksel olarak** gösteriyor (Fig. 16,
<1 nrad/s); ama (i) polarite-switch'in idealize latiste CW/CCW'ye dejenere olması
(§6.1), (ii) simetrik alt-uzayın ayrı izole edilmemesi (§6.2), (iii) CR-ayrım
ÖLÇÜMÜNÜN test edilmemesi (§9) — üç noktada **prosedür nicel olarak açık**.

> **Reprodüksiyon:** Bu oturumun scriptleri `/tmp` altında (proje konvansiyonu:
> keşif kodu repoda tutulmaz). Çekirdek estimator `berry_data/false_edm_4d.py`
> (`measure_false_edm`, 4D-CO + model-fit). Yön/polarite: `CFG["direction"]=±1`,
> `CFG["g0"]/["g1"]=±0.21`. Gerçek EDM: `fields.EDMSwitch=1.0`. σ-testi p=2.00 ile
> estimator doğrulanmıştır.

---

## 11. Eq. C2 DOĞRUDAN TEST (2026-07-15): 4'lü kombinasyon idealize latiste EGZAKT; tilt bozmuyor; flip-kalibrasyonu asıl gereksinim

**Motivasyon:** Kullanıcı, Omarov Fig. 9 ("CW/CCW+flip sahte EDM'i sinyal
seviyesine indirir") ile bizim bulgular ("flip fayda etmiyor, CW/CCW <1 mertebe")
arasındaki tutarsızlığı sordu. Doğrudan test: sabit 10 μm desen (seed 1), dört
konfigürasyon, C++ spin izleyici (`fast_est.fast_measure`, direction/gflip/gscale).

**Sonuçlar (× hedef):**

| konfigürasyon | idealize | +1 mrad tilt | +ε=10⁻³ flip hatası |
|---|---|---|---|
| f_CW | +870.90 | +889.03 | — |
| f_CCW | +696.36 | +672.12 | — |
| f_CW(−g) | +696.28 | +672.19 | +686.22 |
| f_CCW(−g) | +870.91 | +888.79 | +864.30 |
| **ayna testi** f_CCW−f_CW(−g) | +0.08 | −0.07 | — |
| **Eq. C1 2'li** (CW−CCW)/2 | 87.3 | 108.5 | — |
| **Eq. C2 4'lü** | **−0.02** | **+0.08** | **−0.89** |

**Çıkarımlar:**

1. **Tutarsızlık yok.** Omarov'un TAM reçetesi (Eq. C2) sahte EDM'i idealize
   latiste **egzakt** siler (871× → 0.02×, estimator tabanı). Ayna dejenerasyonu
   (CCW(g) ≡ CW(−g), %0.01 hassasiyetle doğrulandı) iptali garanti eder:
   [CW−CCW]₊g = −[CW−CCW]₋g. Gerçek EDM (E-kökenli; terslemede tek, flip'te
   çift) dejenerasyona uymaz → C2'den sağ çıkar (Eq. C1 doğrulaması: ±9.8e-10).
2. **Eski bulgularımız yanlış değil, eksik kombinasyondu:** "flip fayda etmiyor"
   = tek-demet flip (f(−g)/f(+g)≈0.8, iptal yok — doğru); "CW/CCW <1 mertebe" =
   2'li fark (87×/108× artık — doğru). 4'lü diferansiyel hiç test edilmemişti.
3. **Tilt (1 mrad) ayna simetrisini BOZMUYOR** → β-beat dahil tüm *göreli*
   manyetik-latis kusurları da bozmaz (aynı yön-tersleme argümanı; akım
   çevrildiğinde geometri-kaynaklı alan hataları da çevrilir).
4. **C2'nin gerçek gereksinimleri manyetik-latis-DIŞI:** ölçülen duyarlılık
   **artık ≈ 885× × ε** (ε = flip bağıl akım hatası). Ham makinede (871×)
   hedef-altı kalmak için **ε ≲ 10⁻⁴** akım tekrarlanabilirliği gerekir. Ayrıca:
   dört konfigürasyon arasında hizalama kayması, flip'le dönmeyen ortam/artık
   alanlar, E-deflektör kusurları — hepsi ayna-dışı, hepsi C2 artığı bırakır.
5. **BBA+OC'nin yeni konumu (savunma derinliği):** C2 artığı kaynaktaki sahte
   EDM ile orantılı → BBA+OC kaynağı ~10³× düşürünce C2'nin tekrarlanabilirlik
   gereksinimleri aynı oranda gevşer (ε=10⁻³ bile ~0.001× hedef bırakır).
   Makalenin CW/CCW bölümü bu çerçeveyle güncellenecek (kullanıcı onayı bekliyor).

> **Reprodüksiyon:** `/tmp/omarov_ideal_sym.py` (idealize 4'lü),
> `/tmp/omarov_sym_tilt.py` + `/tmp/omarov_tilt2.py` (tilt),
> `/tmp/flip_calib.py` (ε=10⁻³). `fast_est.fast_measure`'a `gscale` parametresi
> eklendi (flip-kalibrasyon hatası: `gflip=True, gscale=1+ε`).

---

## 12. MAKE-OR-BREAK ÇÖZÜLDÜ (2026-07-15): 4'lü egzakt ama drift-kırılgan; OC+BBA polarite-switch'in yerini tutar

**Bağlam:** Kullanıcı haklı olarak sordu — "CW/CCW + quad flip zaten uygulanınca
sahte EDM tam iptal oluyorsa, simetrik-mod tezimiz (BBA şart) çöküyor mu?"
Doğrudan C++ testleriyle çözüldü. **Eski "quad flip işe yaramaz" bulgum
YANLIŞTI** — tek-demet flip'i (f(−g)/f(+g)≈0.8, iptal yok) test etmiştim, ama
Omarov'un reçetesi **4'lü diferansiyel** (C2); onu hiç hesaplamamıştım ve
dejenerasyonu ters yorumlamıştım (§6.1 ⛔).

### 12.1 Saf-mod C2 ölçümleri (seed 1, 10μm)

| desen | tek-demet | 2'li (CW−CCW)/2 | 4'lü mükemmel | 4'lü ε=10⁻³ |
|---|---|---|---|---|
| saf-SİMETRİK | 50–77× | **14×** | 0.009× | 0.48× |
| saf-ANTİSİM | 2100–3080× | **491×** | −0.017× | −4.04× |

- **Mükemmel 4'lü ikisini de egzakt siler** (dejenerasyon CCW(g)≡CW(−g),
  v×B değişmezliği). → "statik indirgenemez simetrik taban" çerçevesi YANLIŞ.
- **Drift-siz 2'li** (eşzamanlı counter-rotating) simetrik-mod tabanı bırakır:
  **14×**. Antisim 491× → OC siler → geriye 14× simetrik → **BBA olmadan inilmez.**
- Kusurlu 4'lü ε=10⁻³: antisim 4× (OC siler), sim 0.48× (hedef altı).

### 12.2 4'lü KONFİG-ARASI DRİFT'e kırılgan (asıl bulgu)

4'lü, +g ve −g konfigürasyonlarının **özdeş** olmasını ister (ardışık ölçüm,
eşzamanlı olamaz). Konfig-arası simetrik drift dejenerasyonu bozar:

| δ_sym (konfig-arası) | C2 (10μm hizalanmamış taban) |
|---|---|
| 0 | −0.02× |
| 1 μm | +37.55× |
| 2 μm | +21.79× (işaretli-bilineer saçılma) |

**Duyarlılık hizalama-seviyesine bağlı** (C2-drift ∝ σ_taban×δ + δ² öz-terim).
Sabit 1μm drift deseni, iki taban:

| taban | sim rms | C2 (δ=1μm) |
|---|---|---|
| OC-only | 5 μm | **−5.05×** (hedef üstü) |
| BBA+OC | 1.9 μm | **+0.07×** (hedef altı) |

→ **BBA-seviyesi hizalama 4'lüyü drift'e ~70× dayanıklı kılar.** Simetrik drift
orbit-kör → sürekli OC yakalayamaz → **yalnız BBA sınırlar.** Air-core quad'da
flip elektriksel temiz (histerezissiz, küçük ε) AMA bobinler-arası Lorentz
kuvveti flip'te değişir → mekanik kayma = konfig-arası drift kaynağı.

### 12.3 BBA + OC drift-siz 2'li ile hedefe iner (polarite-switch'siz)

| durum | tek-demet | 2'li (CW−CCW)/2 |
|---|---|---|
| BBA artığı | 7.96× | 10.24× |
| BBA + OC | 3.25× | **1.57×** (tek seed, ~hedef) |

### 12.4 MAKALE ÇERÇEVESİ (kullanıcı onaylı, 2026-07-15)

> **Omarov: BBA + 4'lü (polarite-switch).**
> **Bu çalışma: yörünge düzeltmesi BBA'nın yerini tutmaz (sim/antisim mod), ama
> BBA'ya EKLENDİĞİNDE polarite-switch'in yerini tutar.**

- OC, antisimetrik (orbit-görünür) kanalı **sürekli, girişimsiz** siler —
  polarite-switch diferansiyelinin işi.
- BBA, simetrik kanalı kaynakta halleder — ne OC ne tek-polarite oraya erişir.
- BBA+OC+2'li ≈ hedef, **flip'siz** → (a) ışın zamanını ~2 kat azaltır,
  (b) konfig-arası mekanik-kayma sistematiğini tamamen kaldırır.
- 4'lüyü kullanmayı seçersen bile: drift-kırılganlığı yüzünden **yine BBA gerekir**
  (§12.2). Yani BBA her iki yolda da zorunlu; OC polarite-switch'i ikame eder.

> **Reprodüksiyon:** `/tmp/sym_c2_test.py` (saf-mod C2), `/tmp/drift_breaks_c2.py`
> (konfig-arası drift), `/tmp/drift_vs_align.py` (hizalama-bağımlı duyarlılık),
> `/tmp/bba_cwccw.py` (BBA+OC 2'li). `fast_est.fast_measure`'a `gflip`, `gscale`
> (flip-kalibrasyon), `dG` (β-beat) parametreleri eklendi. β-beat/tilt 4'lüyü
> BOZMAZ (§11): C2 sırasıyla −0.086× / +0.08×.

---

## 13. CW-tek MUTLAK SEVİYE ÇAPRAZ-KONTROLÜ (2026-07-28): T-BMT birebir + Fig 9(c) yörünge uyumu → kalan ~10× kuyruk-fiti hipotezi

**Motivasyon:** Kullanıcı, bizim CW-tek sahte-EDM'imizin (σ=10 μm, 15 seed RMS
1.6×10⁻⁶) Omarov **Fig 9(a)** fitinden (1.5×10⁻⁵ @ σ=10 μm) ~10× düşük görünmesini
sordu. Sistematik eleme: örnekleme (15 seed teyit), estimator/pencere (t2-taraması
düz, b~1.1e-6 90→900 tur), kanal (T-BMT tam), gradyan (ikisi de ~0.2 T/m).

### 13.1 T-BMT denklemi referansla BİREBİR
Kullanıcının referans `ds_dt` fonksiyonu ile `integrator.cpp:324-331` terim-terim
eşleşiyor (MDM: B(G+1/γ), (β·B)β, (β×E); EDM: (β×B), E/c, (β·E)β; ölçek 0.5·η·Q/M).
Tek fark işaret **konvansiyonunda ama fizik özdeş**: referans `s×cross`, biz
`Ω×s` (Ω=−cross·Q/M) → `Ω×s = +(Q/M)(s×X) = s×(X·Q/M)` = referansla aynı.
Anomali sabiti `G_P = 1.792847356` (satır 46) = referans `AMU`. **Spin motoru doğru.**

### 13.2 CR-ayrım MUTLAK genliği Fig 9(c) ile UYUMLU
Kapalı yörünge misalignment'ta doğrusal → tepki matrisleri (R_dy/R_dx) genliği tam
kodlar; ayrıca doğrudan C++ (CW − CCW) ile teyit. (`kmod_drivers/orbit_amp_check.py`)

| ölçüm | bizim | Omarov (Fig 9c / Sec IV) | durum |
|---|---|---|---|
| tek-quad 100 μm → CR-ayrım | ~300 μm (dy), ~234 μm (dx) | ~250 μm | ✅ |
| σ=100 μm rms → COD tepe | 1.32 mm | >1 mm | ✅ |
| **σ=10 μm → CR-ayrım tepe** (5 seed, C++) | **~226 μm** (rms 126 μm) | **~200 μm** | ✅ |

→ Yörünge genliğimiz Omarov halkasıyla **~1.5× içinde eşleşiyor.** f ∝ x_CO·y_CO
(kuadratik) olsaydı ~10× için düzlem başına ~3.2× yörünge açığı gerekirdi — **yok.**

### 13.3 Estimator DEĞİL (kullanıcı düzeltmesi)
Erken bir açıklama — "biz 1.-mertebe dikey-hız katkısını model-fit ile ayıklıyoruz,
Fig 9(a) onu içeriyor" — **YANLIŞ**: Omarov-tarzı ham/bulut fiti de ~1.1×10⁻⁶
veriyor. Ölçtüğümüz nicelik zaten Omarov'unkiyle aynı tanım. Fark estimator'da değil.

### 13.4 Kalan ~10× → KUYRUK etkisi (kısmen doğrulandı, 2026-07-28; n=40, taze .dylib)
Fizik: f ∝ yörünge², yörünge rezonansa yakın ağır-kuyruklu → f ağır-kuyruklu.
`sigma_dist_fit.py` (40 seed × 3σ, bağımsız desenler) sonuçları:

| σ | medyan | ort | RMS | p90 | max | max/medyan |
|---|---|---|---|---|---|---|
| 2.5μm | 7.4e-8 | 1.1e-7 | 1.7e-7 | 2.6e-7 | 6.9e-7 | 9.4× |
| 5μm | 1.9e-7 | 2.5e-7 | 3.4e-7 | 5.5e-7 | 9.9e-7 | 5.1× |
| 10μm | 1.56e-6 | 2.11e-6 | 3.08e-6 | 4.7e-6 | **9.07e-6** | 5.8× |

y=kx² fit: k_medyan→f(10)=1.52e-6 (Omarov 9.9×); k_all(bulut)→2.04e-6 (7.3×);
**k_max→8.77e-6 (1.7×).**

**Okuma (dürüst):**
- (a) Dağılım GERÇEKTEN ağır-kuyruklu — σ=10μm max **9.07e-6**, 40 seed'de Omarov'un
  1.5e-5'inin **0.6 katına** ulaştı. Kullanıcının Fig 9(a) ekstrem-nokta okuması DOĞRU:
  tekil konfigürasyonlar ~10⁻⁵'e çıkıyor.
- (b) AMA sıradan bulut-LS (2.04e-6) medyanın yalnız 1.35 katı — tek başına 1.5e-5'e
  ÇEKMİYOR. Fiti Omarov'a yaklaştıran **üst-zarf/max** (8.77e-6, 1.7× içinde).
- (c) Ağır-kuyruklu dağılımda örnek-ortalaması AŞAĞI YANLI (nadir dev olaylar 40
  seed'de eksik) → daha çok seed'le k_all tırmanır. 2.04e-6 bir ALT sınır.

**Sonuç:** Omarov'un 1.5e-5'i bizim örneklediğimiz AYNI dağılımın üst kuyruğunda
(max'ımızın 1.65 katı); ~10× bir FİZİK farkı değil, medyan-vs-üst-zarf + ağır-kuyruk
örnekleme etkisi. Yörünge (9c) ve T-BMT zaten uyumlu. **Açık uç:** 200–400 seed ile
k_all'ın tırmanışını ve bir seed'in 1.5e-5'i aşıp aşmadığını görmek (isteğe bağlı teyit).

### 13.5 Makale çıkarımı
**σ² ölçekleme (p=2.00) + Fig 9(c) yörünge uyumu (~226 vs ~200 μm) iddia edilir;
Fig 9(a) fit DEĞERİYLE mutlak sayısal eşitlik İDDİA EDİLMEZ.** Sim düşük değil —
Fig 9(a) fiti ağır-kuyruklu dağılımın üst zarfını izliyor. Sub-hedef marjinler
tehlikede değil (§II.D validasyon sayıları geçerli).

> **Reprodüksiyon (kalıcı, /tmp değil):** `kmod_drivers/orbit_amp_check.py`
> (cached R veya --rebuild; tek-quad + σ-rms genlik), `kmod_drivers/sigma_dist_fit.py`
> (dağılım + kuyruk-fiti testi, lokal — ağır spin koşumları). T-BMT: `integrator.cpp:283-335`.
