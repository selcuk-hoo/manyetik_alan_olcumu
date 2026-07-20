# PRAB Hakem Raporu — Review 1

**Makale:** "Suppression of the quadrupole-misalignment false EDM signal by
orbit-based methods in a proton EDM storage ring"
**Tarih:** 2026-07-20
**Karar önerisi:** Majör revizyon

---

## Genel değerlendirme

Makale, proton EDM halkasında kuadrupol kaçıklığı kaynaklı sahte EDM'in yapısını
(bilineer, simetrik/antisimetrik ayrışım) net biçimde ortaya koyuyor ve özgün,
önemli bir ana sonuç içeriyor: simetrik kaçıklık bileşeni tek-demet yörünge
ölçümüne görünmezken, iki karşı-dönen demetin yörüngelerinin **toplamında**
tamamen görünür olması ve iki demeti birlikte düzeltmenin BBA'sız hedef-altına
inmesi. Doğrulama altyapısı (semplektik izleyici + bağımsız analitik katman,
kanal ayrışımının %0.1 kapanışı, p=2.00 ölçekleme, işaret-yapısı testleri) PRAB
standardının üstünde. Anlatı yayı tutarlı.

**Karar: Majör revizyon.** Ana iddia iki kritik eksik testin (M1–M2) sonucuna
bağlı; bunlar giderilmeden merkezî sonuç savunulamaz.

---

## Majör noktalar

### M1 — Statik BPM ofsetleri iki-demet testinde yok (en kritik nokta)
§IV'ün hata modeli açıkça "static BPM offsets ~100 μm" içeriyor ve fark-yörünge
yönteminin varlık sebebi olarak sunuluyor: ofsetler ancak iki-ayar farkıyla
iptal edilir, bunun bedeli ‖ΔR⁻¹‖ amplifikasyonudur. Oysa §V'teki iki-demet
düzeltme fark almayan, doğrudan yörünge okuma kullanıyor — yani ofsetler iptal
edilmiyor — ama testlerde yalnız 14 nm beyaz gürültü var, 100 μm statik ofset
yok. Bu sessiz bir çelişki: §IV'ün tüm motivasyonu (ofsetler differencing'i
zorlar) §V'te atlanıyor. Fiziksel risk açık: iki demeti BPM elektriksel
sıfırlarına düzlemek, demeti kuadrupol merkezlerine değil ofset-yüklü referansa
oturtur; kuad-merkez-göreli yörünge ~100 μm kalır ve f ∝ σ² ile bu ~100× büyük
bir sahte EDM tabanı demektir. CW ve CCW aynı pickupları gördüğünden ofsetlerin
tek-fark C'de kısmen iptali mümkündür, ama bu makalenin kendi bulgusu
(yön-asimetrisi Δ; aynı kaçıklıkta bile f_CW ≠ f_CCW) nedeniyle varsayılamaz,
gösterilmelidir. **Gerekli:** 100 μm statik BPM ofsetli tam iki-demet kampanyası.
Sonuç kötüyse, iki-demet OC'nin BBA-türü bir ofset kalibrasyonuna muhtaç olduğu
dürüstçe yazılmalı — bu, ana tezin "BBA'sız" kısmını doğrudan belirler.

### M2 — Düzeltici donanım modeli belirtilmemiş
Kod düzeltmeyi kaçıklık uzayında uyguluyor (res → res − pinv·y): fiilen
"kuadrupol merkezlerini kaydırmak". Bu donanımda mümkündür (bobin akım
asimetrisi = manyetik merkez kayması; makale §IV'te BBA bağlamında bu
eşdeğerliği zaten kuruyor) ama sıradan manyetik dipol düzelticilerle mümkün
olmayabilir: statik manyetik kick q v×B gereği karşı-dönen demete zıt etki eder,
yani dipol düzelticiler yalnız yön-tek (antisimetrik-görünür) kombinasyonları
tarar. İki-demet düzeltmenin ihtiyaç duyduğu yön-ortak serbestlik ya kuad-merkez
kaydırıcılarını ya da elektrik düzelticileri gerektirir. Makale hangi aktüatör
setini varsaydığını, kaç serbestlik derecesi olduğunu ve bunun
gerçeklenebilirliğini açıkça yazmalı. (Bu aynı zamanda güzel bir fizik noktası:
yön-ortak düzeltme ancak yön-ortak kuvvetle yapılır — elektrik alan veya gerçek
hizalama.)

### M3 — Δ'nın fiziksel kökeni açıklanmamış
§V, R_CW = −R_CCW + Δ ayrışımını sayısal veriyor (‖Δ‖/‖anti‖ ≈ 0.21) ama Δ'nın
neden var olduğunu söylemiyor. İdeal ayna-simetrik, saf-manyetik bir FODO'da
ters-yön tepkisi tam −R olurdu; Δ'yı ne üretiyor? (Elektrik deflektörlerin
yön-ortak kuvveti? Yatay bükme dizisinin ters sırayla kat edilmesi? Faz-ilerleme
asimetrisi?) Dahası metin yalnız dx düzlemi sayılarını veriyor; dy düzleminde de
aynı oran (~0.21) mevcut — deflektörlerin dikeyde drift olduğu bir latiste dikey
Δ'nın kökeni ayrıca açıklanmalı. Bu açıklama sonucun latise-özgü mü genel mi
olduğunu belirler; hakem/okur için vazgeçilmez. Ek olarak "visibility 1.01 /
cond 30" gibi normlar birimsiz matris normları — tanım ve normalizasyon (m/m,
hangi taban) belirtilmeli.

### M4 — §VI'daki drift iddiası simüle edilmemiş
"Sürekli iki-demet düzeltme simetrik drifti de tutar → periyodik BBA gerekmez"
cümlesi makalenin operasyonel sonucunu taşıyor, ama metindeki drift
simülasyonları tek-demet feedback dönemine ait. Ya iki-demet feedback ile drift
taraması eklenmeli (σ_d taraması, iki-demet düzeltmeli) ya da iddia "beklenir"
düzeyine yumuşatılmalı.

### M5 — İstatistik tabanı dar
Ana sonuç 5 seed × tek β-beat gerçekleşmesi (sabit desen) × tek roll deseni
üzerine kurulu. Sahte EDM'in seed'den seed'e işaretli ve geniş saçıldığını
makale kendisi vurguluyor. En azından 10–20 kaçıklık seed'i ve birkaç
β-beat/roll gerçekleşmesi; medyanla birlikte en-kötü-durum raporu.

### M6 — İç tutarsızlıklar (sayılar)
- **(a)** l.599: "Under the **realistic** 1%-gradient-error optics (≈5%
  β-beat)" — §V aynı noktayı "**unrealistically large** 5%" diye niteliyor; aynı
  işletim noktası iki yerde zıt sıfatlarla anılıyor. "Conservative/stress-test"
  gibi tek bir dil seçilmeli.
- **(b)** §III cond(R)=193 (analitik, dy), §V cond=135 (izleme, dx) ve dy için
  228 — hangi matris/düzlem/kaynak olduğu her seferinde belirtilmeli.
- **(c)** İki-demet mekanizma sayıları (33.8/7.3/17 mod/7×/0.7×) yalnız dx; dy
  paraleli verilmeli.

### M7 — Abstract'taki aşırı-genel ifade
"no single-beam orbit measurement reaches it" — makalenin kendi §IV'ü BBA'nın
(tek-demet, orbit-tabanlı, null-arayan) simetriğe eriştiğini gösteriyor. İfade
sınıf-düzeyinde daraltılmalı ("no single-beam orbit *inversion or amplitude
readout*").

### M8 — Ölçüm fizibilitesi: 14 nm ve eşzamanlı CW/CCW okuma
14 nm etkin gürültünün averaj bütçesi (kaç edinim, hangi bant) verilmeli; ayrıca
aynı pickuplarda iki karşı-yönlü demetin yörüngelerinin ayrı okunması (bunch
timing/gating) bir cümleyle de olsa ele alınmalı — referans tasarım CR-separasyon
ölçtüğü için muhtemelen çözülmüş, ama iki-demet OC bunun sürekli ve mutlak (fark
değil) versiyonunu istiyor.

### M9 — Adil karşılaştırma notu
"0.17× vs four-fold'un 0.009×" karşılaştırmasında iki sayı farklı koşullarda
(iki-demet: β-beat'li makine; four-fold: idealize doğrulama). Ya four-fold da
β-beat+drift altında aynı protokolle koşulmalı ya da dipnotla eşitsizlik
belirtilmeli.

---

## Minör noktalar

1. **tab:chain** satırı "orbit corr. + CW/CCW ~47×" → "single-beam orbit corr. +
   CW/CCW" olarak nitelenmeli; yoksa tab:twobeam ile çelişir görünür.
2. **fig:suppression** yıldızı hâlâ BBA+OC uç noktası; yeni teze göre iki-demet
   OC noktası da eklenmeli (veya caption'da rolü netlenmeli). β-beat taraması şu
   an yalnız tablo — bir panel figür (β-beat vs C, tek/iki demet) sonucu çok
   güçlendirir.
3. `rossbach` bibitem atıfsız — atıf ekleyin ya da çıkarın. `wegscheider` yazar
   listesi bozuk ("S. Wegscheider and J. Vilsmeier *et al.*").
4. Giriş §I.C ile §sec:fourfold four-fold anlatımı kısmen mükerrer; giriş
   kısaltılabilir.
5. Kullanılmayan etiketler (sec:chainclose, sec:symanti…) — kozmetik temizlik.
6. Roll dipnotu (ikinci-mertebe kuplajın düzeltmeye verilmediği) dürüst ve
   yerinde — kalsın; benzer bir kapsam cümlesi sextupolsüz lineer model için
   sonuçta zaten var.
7. Abstract uzun (~260 kelime); PRAB için kısaltma önerilir — özellikle mekanizma
   cümlesi (sum/difference) tek cümleye inebilir.
8. Veri/kod erişilebilirliği ifadesi yok; PRAB için "data available upon request"
   ya da depo referansı ekleyin.
9. Birkaç aşırı uzun cümle (özellikle §V mekanizma paragrafı, 6+ satır)
   bölünmeli.

---

## Özet

Ana sonuç (toplam-yörünge görünürlüğü + iki-demet düzeltme) yeni, önemli ve iyi
doğrulanmış bir *mekanizma* içeriyor; ancak makale bu iddiayı statik BPM
ofsetleri (M1) ve aktüatör gerçeklenebilirliği (M2) testlerinden geçirmeden
"BBA'sız hedef-altı" diye sunamaz. M1–M2 olumlu çıkarsa bu, alan için değerli bir
makale olur; olumsuz çıkarsa tez "iki-demet OC + minimal ofset kalibrasyonu"
biçiminde yeniden çerçevelenmelidir — bu da yayımlanabilir, ama farklı bir
iddiadır.

---

## Durum notu (yazar tarafı)

- **M1 testi koşuluyor** (100 μm statik BPM ofseti + %1 β-beat + 14 nm, 5 seed,
  iki-demet vs tek-demet). Lineer ön-analiz: 100 μm ofset iki-demet düzeltmede
  ~87 μm sahte simetrik artık bırakıyor; tek-artık C'de ortak-mod iptali
  izleyiciyle sınanıyor. Sonuç tezi belirleyecek.
- Metin düzeltmeleri (M6a, M7, tab:chain, rossbach) M1'den bağımsız; sonraya
  bırakıldı.
