# Olası Hakem Soruları ve Hazır Cevaplar

**Makale:** "Accessing the symmetric component of quadrupole misalignment through
counter-rotating closed orbits in EDM storage rings" (Abedrabbo & Hacıömeroğlu, PRAB hedefli)

**Amaç:** Gönderim öncesi, muhtemel hakem itirazlarını ve bunlara verilecek
cevapları tek yerde toplamak. Her madde: **kaynak** (kim gündeme getirdi), **soru**,
**hazır cevap/durum**, ve gerekiyorsa **ek koşu gerekli mi**.

**Kaynaklar:**
- **Grok** — tam bir taslak hakem raporu üretti (minor revision önerisi).
- **ChatGPT** — ifade/çerçeve ve AI-politika yorumları.
- **İç-inceleme** (`review-1.md`, 2026-07-20, "majör revizyon") — M1–M9; çoğu çözüldü,
  reframe'in temeli. En bilgili kaynak.
- **Bu oturum (Claude) + yeni simülasyon** — özellikle yatay/dikey düzlem ayrımı
  (`/tmp/.../scratchpad/q4_plane_check.py`).

> **Genel strateji (mutabık):** Makaleyi bu aşamada değiştirmiyoruz. En dürüst
> haliyle gönderiyoruz; hakem isterse revizyonda karşılıyoruz. İstisna: merkez
> iddiaya dokunan noktaları (yatay düzlem, ofset/drift) **göndermeden önce bilmek**
> istiyoruz — kumar değil, sağlamlık meselesi.

---

## A. Merkez iddiaya dokunan sorular (en kritik)

### A1 — Yatay mı, dikey mi? İki-demet her iki düzlemin simetriğini de kurtarıyor mu?
**Kaynak:** Grok Q4; iç-inceleme M3/M6c.
**Soru:** Artık rakamları dikey (dy) gibi görünüyor. İki-demet OC **yatay** simetrik
bileşeni de aynı seviyede kurtarıyor mu, yoksa fark var mı?

**Cevap (yeni simülasyonla ölçüldü — temiz optik, 14 nm, 15 seed):**
| Simetrik artık (μm) | ham | iki-demet |
|---|---|---|
| **Yatay (dx)** | 7.06 | **0.017** (368× bastırma) |
| **Dikey (dy)** | 6.38 | **2.84** (2.3× bastırma) |

- İki-demet **yatayı neredeyse sıfıra** indiriyor (368×), **dikeyi** ise ancak 2.3×.
  Koşullanma ile tutarlı: iki-demet yığınında cond(dx)=73 < cond(dy)=118.
- **Sahte EDM neden yine hedef-altı?** Çünkü $f \propto dx\cdot dy$ bir **çarpım**.
  Yatay ≈ 0 olunca çarpım da ≈ 0 — dikeyin zayıf düzeltilmesi önemsizleşir.
  Kampanya sonucu (15/15 hedef-altı, medyan 0.043×, en kötü 0.62×) bu mekanizmayla
  açıklanıyor: **bir düzlemi (yatay) sıfırlayıp bilineer çarpımı öldürüyoruz.**
- Bu, "analitik R yatayda kötüydü → simülasyon-R ile kurunca yatay yıldız düzlem oldu"
  hikâyesini doğruluyor (makale §, `line 206`).

**β-beat (%5) altında (yeni koşu, nominal model vs β-beat makine, 15 seed):**
| Simetrik artık (μm) | ham | temiz 2-demet | **%5 β-beat 2-demet** |
|---|---|---|---|
| Yatay (dx) | 7.06 | 0.017 (368×) | **1.12 (5.3×)** |
| Dikey (dy) | 6.38 | 2.84 (2.3×) | **3.49 (1.9×)** |

β-beat, temiz-optikteki "yatay yıldız"ı **çökertiyor** (368×→5.3×). Yani temiz
optikteki "yatayı sıfırla → bilineer çarpımı öldür" mekanizması β-beat altında
**geçerli değil**: her iki düzlemin simetrik kaçıklığı da ancak birkaç × kurtarılıyor.

**KRİTİK UZLAŞMA (yanlış yorumlamamak için):** Buna rağmen **spin-ölçülü sahte EDM**
kampanyada aynı işletim noktasında (%5 β-beat) **15/15 hedef-altı** (en kötü 0.62×,
"hedefe yakın"). Çelişki değil: per-düzlem RMS bir **tanı göstergesi**; sahte EDM ise
ağırlıklı bilineer fonksiyonel $f=\sum_{ij}W_{ij}v^s_{x,i}v^s_{y,j}$ + beam-reversal-tek
iptali ($C=(f_{CW}-f_{CCW})/2$) — RMS-çarpımı sahte EDM'e **eşit değil** (bu yüzden
kaba çarpım tahmini hedef-üstü verirken gerçek spin ölçümü hedef-altı).

**Dürüstlük notu / önerilen çerçeve:** Makalenin **savunulabilir** iddiası
"**sahte EDM'i hedef-altına bastırıyoruz**" (spin-ölçülü, doğru) — "simetrik kaçıklık
**desenini** geri çatıyoruz" DEĞİL (kısmi; temiz optikte yatay mükemmel, β-beat'te
her iki düzlem de birkaç ×; marj β-beat'te inceliyor). Hakem Q4'ü sorarsa:
- Temiz optik: yatay 368× (çarpım nedeniyle bir düzlem yeter).
- %5 β-beat: mekanizma tek-düzlem sıfırlamaya dayanmıyor; yine de sahte EDM 15/15
  hedef-altı (paper zaten "brings it close" diyor) — **çünkü ölçülen şey $C$, per-düzlem
  RMS değil.**
- "recover the symmetric component" ifadesini "**suppress the false EDM**" lehine
  yumuşatmak dürüstlüğü artırır.

**Q2 bağlantısı:** β-beat'te marj daraldığından (0.62×), bağımsız BPM/quad drift'i +
β-beat birlikte marjı hedef-üstüne itebilir → A2/Q2 koşusu bu yüzden değerli.

### A2 — Statik BPM ofseti (~100 μm) iki-demet testinde var mı? Drift iddiası simüle edildi mi?
**Kaynak:** İç-inceleme M1 (en kritik) + M4; Grok Q2.
**Soru:** §IV'ün motivasyonu 100 μm statik ofset; §V iki-demet testinde yalnız 14 nm
gürültü var, statik ofset yok. Ofset iptal ediliyor mu? Drift monitör iddiası
iki-demet feedback ile simüle edildi mi?

**Cevap/durum (kısmen çözülü — reframe'in temeli):**
- İç-inceleme M1 doğrudan test edildi: **100 μm ofset iki-demet OC'yi 0.17×→198×
  çökertti (0/5).** Fizik: ofset ile simetrik kaçıklık karşı-dönen **toplamda dejenere**.
- Bu yüzden makale iki parçaya ayrıldı: **(i)** dejenerasyon teoremi — mutlak hizalama
  bir kez BBA ister; **(ii)** diferansiyel drift monitörü — drift **fark-yörüngede**,
  ofset iptal olur, sürekli iki-demet OC simetrik drifti tutar → periyodik BBA'yı kaldırır.
- Yani "BBA'sız mutlak hizalama" İDDİA EDİLMİYOR; iddia "bir kez BBA + sürekli ucuz
  iki-demet drift monitörü."

**Açık kalan (Grok Q2 = M4):** Drift monitörünün kendisi (bağımsız BPM/quad drift'i
ile) tam simüle edilmedi. Rossbach (BPM-quad birlikte hareket eder) bir varsayım.
Bağımsız spektrumlar altında artık sahte-EDM büyüme hızı gösterilmeli.
→ **Bu üzerine kumar oynanabilir**; hakem sorarsa σ_d taramalı iki-demet drift koşusu
yapılır. Metinde ilgili cümle zaten "beklenir/nicelenecek" düzeyinde yumuşak.

### A3 — Düzeltici donanım modeli ne? (aktüatör fizibilitesi)
**Kaynak:** İç-inceleme M2 — **hâlâ açık**; Grok dolaylı.
**Soru:** İki-demet düzeltme kaçıklık uzayında uygulanıyor (kuad merkezlerini kaydırmak).
Hangi aktüatörle? Statik manyetik dipol düzelticiler karşı-dönen demete $q\,v\times B$
ile **zıt** etki eder → yalnız yön-tek (antisimetrik) kombinasyonları tarar. Yön-ortak
(simetrik-görünür) düzeltme için ne gerekiyor?

**Cevap (kavramsal, koşu gerektirmez):**
- Yön-ortak düzeltme ancak **yön-ortak kuvvetle** yapılır: ya **kuadrupol-merkez
  kaydırıcıları** (bobin akım asimetrisi = manyetik merkez kayması; makale §IV'te BBA
  bağlamında bu eşdeğerliği zaten kuruyor) ya da **elektrik düzelticiler**.
- Sıradan manyetik dipol düzelticiler bunu yapamaz — bu güzel bir fizik noktası ve
  makalede **açıkça yazılmalı** (aktüatör seti + serbestlik dereceleri).
- **Durum:** Metne henüz eklenmedi; sonraki turda yazılacak. Gerçek bir hakem bunu
  büyük olasılıkla sorar → **proaktif bir cümle eklemek akıllıca olur.**

---

## B. Nicelleştirme soruları (kolay–orta)

### B1 — Artığı hedef biriminde ver
**Kaynak:** Grok Q1.
**Soru:** "0.1–0.7 μm artık" kaçıklık cinsinden; kalan **sahte EDM'i hedef biriminde**
(×target) ve tam kusur kombinasyonunu (gradyan hatası + tilt + β-beat) açıkça verin.
Fig. 4'e küçük bir tablo/panel iyi olur.

**Cevap/durum:** Rakam **var**: %1 β-beat, 15 seed → medyan **0.043×**, **15/15 hedef-altı**,
en kötü **0.62×**. Bu zaten tabloda; hakem isterse Fig. 4'e β-beat vs C paneli (tek/iki
demet) eklenir (iç-inceleme minör #2 de bunu öneriyor). **Cila düzeyinde, veri hazır.**

### B2 — Regularizasyon reçetesi ve gürültü stabilitesi
**Kaynak:** Grok Q3.
**Soru:** Regularize edilmiş pseudo-inverse nasıl kuruluyor (kesme / Tikhonov / TSVD)?
%1 BPM gürültüsüyle stabil mi?

**Cevap:** **TSVD** — `np.linalg.pinv(rcond=0.01)`, yani en büyük tekil değerin %1'inin
altındaki modlar atılıyor. Yığma ([R_CCW; R_CW]) simetrik ailenin tekil değerlerini
bu eşiğin üstüne taşıdığı için iki-demet çalışıyor (tek-demette altında kalıyorlar).
Gürültü: testler zaten 14 nm beyaz BPM gürültüsüyle; sonuç stabil.

**ÖNEMLİ (yeni teşhis — Q4 dikey bilmecesini de çözer):** rcond=0.01 **düzleme
duyarlı**. SVD teşhisi (`q4_plane_check` + rcond taraması):
- **Yatay (dx):** 48/48 mod tutulur, simetrik altuzay **%100** kapsanır → 368×.
- **Dikey (dy):** rcond=0.01 **8 modu keser**; simetrik altuzayın **%24'ü kesilen
  modlarda** (en küçük simetrik-baskın mod σ/σ_max=0.0085 < 0.01) → yalnız %76 → 2.3×.
- **rcond'u 0.005'e düşürünce dikey de 390×'e çıkar (0.017 μm), 14 nm gürültüde
  cezasız** (sıfır gürültüde makine hassasiyeti). Yani dikeyin zayıflığı **analitik R
  değil**, sadece fazla agresif kesme eşiğiydi; ikisi de simülasyon R.

**Cevap (Q3):** rcond seçimi raporlanmalı ve **0.005 kullanmak dikey simetriği de tam
kurtarır** (yatayla eşit). Gürültü-stabilitesi 14 nm'de korunuyor. ⚠️ **Uyarı:** rcond'u
düşürmek β-beat *model uyuşmazlığını* büyütebilir (kesme, uyuşmazlığa karşı da koruma);
β-beat altında rcond taraması ayrıca yapılmalı (koşuluyor).

---

## C. Çerçeve / ifade soruları

### C1 — "recover the symmetric component" fazla mı iddialı?
**Kaynak:** Claude (A1'den).
**Cevap:** Evet, sıkı okumada. Gerçek: yatay simetriği kurtarıyoruz; sahte EDM bilineer
olduğu için bu yeterli. Revizyonda A1'deki tek cümle eklenir. **Şimdilik gönderilebilir**,
bilerek.

### C2 — Abstract'ta "no single-beam orbit measurement reaches it" aşırı-genel
**Kaynak:** İç-inceleme M7 — **çözüldü**.
**Cevap:** Zaten daraltıldı: "no single-beam orbit *inversion or amplitude readout*"
(null-arayan BBA erişir ama yavaş). Hakem tekrar sorarsa referans bu.

### C3 — Aynı işletim noktası "realistic" ve "unrealistically large 5%" olarak çelişik
**Kaynak:** İç-inceleme M6a — **çözüldü**.
**Cevap:** Tek dile çekildi: "conservative stress test, above LOCO". Tutarlı.

### C4 — "Halving the time" / süre iddiası, drift iddiası fazla güçlü
**Kaynak:** ChatGPT; Grok minör.
**Cevap:** Süre iddiası zaten "potential to reduce by ~2×"e yumuşatıldı. Drift
cümlesi "maintains" → "can reduce" yapılabilir (A2 koşusu gelene kadar). **Küçük dokunuş.**

---

## D. Ufak / dizgi (kolay)

| # | Kaynak | Nokta | Durum |
|---|--------|-------|-------|
| D1 | Grok | "the offset cancels" → "cancels **in the drift**" | Kabul; doğru ve biz de bu ayrımı koruyoruz |
| D2 | Grok | Fig. 1(b): ilk simetrik tekil-değer indisini etiketle | Kolay figür güncellemesi |
| D3 | Grok | Fig. 3 yeşil yıldız (hibrit) overhead'i bir cümle | Kolay |
| D4 | Grok | Eq. (3): β/faz ideal latisten mi, hatalı latisten mi? | Tek cümle netleştirme |
| D5 | Grok | "Ref. [2] hem omarov hem Omarov et al." | **Non-issue** — `.tex` anahtarını basılı atıfla karıştırmış |
| D6 | Grok | e-posta/ORCID tutarlılığı | **Halüsinasyon** — makalede e-posta yok; Grok istinye adresi uydurmuş |
| D7 | İç-inc. | fig:suppression yıldızına iki-demet noktası ekle | Figür betiği güncellemesi (bekliyor) |
| D8 | İç-inc. | Uzun cümleler (§V mekanizma) bölünmeli | Kozmetik |
| D9 | İç-inc. | Abstract ~260 kelime; PRAB için kısalt | Opsiyonel |

---

## E. Güçlü yanlarımız (hakeme karşı zırh)

Bunlar rapor edildiğinde makaleyi savunur:
- **Estimator doğrulaması:** σ² ölçekleme **p = 2.00 ± 0.01**; T-BMT özdeşliği referansla
  birebir; Omarov Fig. 9(a) (σ²) ve 9(c) (~200 μm demet ayrımı) ile uyum; null testi 10⁻¹¹;
  bilinen EDM geri-ölçümü 9.81×10⁻¹⁰ (Omarov Eq. C1).
- **Kanal ayrışımının %0.1 kapanışı**, işaret-yapısı testleri (bilineer $dx\cdot dy$).
- **Mekanizma sağlamlığı:** iki-demet yığma koşullanmayı iki düzlemde de iyileştiriyor
  (dx 135→73, dy 228→118); simetrik mod payı %90-98'den %63-79'a düşüyor.
- **İç-inceleme geçmişi:** M1–M9'un çoğu zaten test edilip çözüldü (özellikle M1 ofset
  dejenerasyonu → dürüst reframe).

---

## F. Açık kalan koşular (öncelik sırası)

1. **A3 / M2 — Aktüatör donanım modeli** (kavramsal, koşu YOK) — proaktif cümle ekle.
   *En olası hakem sorusu; en ucuz savunma.*
2. **A2 / M4 / Grok Q2 — Bağımsız BPM/quad drift'i ile iki-demet drift monitörü koşusu**
   (σ_d taraması) — kumar oynanabilir, sorulursa yap.
3. **A1 β-beat uzantısı — YAPILDI:** yatay kurtarma β-beat'te 368×→5.3× çöküyor;
   ama spin-ölçülü sahte EDM yine 15/15 hedef-altı (per-düzlem RMS ≠ sahte EDM).
   İddiayı "suppress the false EDM"e yumuşatmak önerilir.
4. **B1 — Fig. 4'e β-beat vs C paneli** (opsiyonel görsel güçlendirme).
5. **BPM gain hatası testi** — opsiyonel doğrulama (§6/§7'de "nicelenecek").

---

*Not: Keşif kodları `/tmp/.../scratchpad/` altında (q4_plane_check.py, q4_bbeat_check.py);
kampanya `kmod_drivers/twobeam_oc.py`; iç-inceleme `review-1.md`.*
