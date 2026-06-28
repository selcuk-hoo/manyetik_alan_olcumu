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
