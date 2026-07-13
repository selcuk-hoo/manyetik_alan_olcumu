# Makale Revizyon Notları (makale_orbit_bastirma)

> **Amaç:** Kullanıcı yorumlarını burada topla, her birini tartış/karara bağla,
> SONRA hepsini tek seferde uygula (ekonomik). Makale ŞU AN değiştirilmiyor.
> **Durum:** çok-seed pipeline + ek testler arka planda dönüyor; kesilmeyecek.

Tarih: 2026-07-13

---

## A. Kullanıcı yorumları ve değerlendirme

### A1. Abstract çok uzun → kısalt
**Yorum:** Abstract çok uzun, o kadar detay gerekmiyor.
**Değerlendirme:** Katılıyorum. Şu an ~1 tam paragraf, çok sayıda spesifik sayı
(474×, 62×, 7.7×, lock-in, SQUID, ...) içeriyor. Bunlar gövdeye ait.
**Önerilen değişiklik:** ~130–160 kelimeye indir. Yalnız hikâye yayı kalsın:
(1) sahte-EDM problemi + σ² + ~1000× hedef; (2) yörünge düzeltme tek başına
simetrik moda takılıyor; (3) sim/antisim mekanizması (bir cümle); (4) BBA +
yörünge düzeltme birlikte tabanı kırıyor; (5) dürüst kısıt (tek-seed, donanım
tabanı). Spesifik ara-sayıları at.

### A2. Makalenin anlatı yapısını değiştir (BÜYÜK — öncelik)
**Yorum:** Şöyle bir akış daha iyi olmaz mı: "Şöyle bir sahte-EDM problemimiz
var → çözmek için yörünge düzeltmeye başvurmak istiyoruz → hızlandırıcılarda
yörünge düzeltme şöyle yapılır → AMA fark ettik ki yörünge düzeltme tek başına
sahte-EDM'i bastırmayı garantilemiyor → çünkü daha önce farkında olmadığımız
simetrik/antisimetrik mod gerçeği var, mekanizma şöyle → ama şöyle yapınca
bastırabiliyoruz, çünkü şu mekanizma."
**Değerlendirme:** Bence kesinlikle daha iyi. Mevcut yapı "başarısız yöntemler
kataloğu"nu (matris tersleme, K-mod, NN, drift, harmonik, CR-ayrım) merkeze
koyuyor; asıl mesaj (mekanizma + çözüm) sona gömülü. Kullanıcının önerdiği yay
daha güçlü ve okunur: **problem → yörünge düzeltme (standart) → keşif: tek başına
yetmiyor → sim/antisim mekanizması (bilineer kanallar) → çözüm: BBA + yörünge
düzeltme → neden çalışıyor.**
**Önerilen değişiklik (yeniden kurgu iskeleti):**
1. Giriş: pEDM + frozen-spin + sahte-EDM (kısa).
2. Yörünge düzeltme: neden doğal araç; hızlandırıcılarda nasıl yapılır (standart,
   kısa özet — burada ilgili literatür AKICI prose olarak, bkz. A5).
3. **Keşif:** yörünge düzeltme tek başına sahte-EDM'i garantilemiyor → simetrik
   mod yörünge-kör kalıyor.
4. **Mekanizma:** sim/antisim ayrışımı + bilineer dört kanal (şu anki §II.D
   matematiği burada merkezî). Buradan "tek RMS parametresi yetmez" çıkarımı (A4).
5. **Çözüm:** null-BBA simetrik kanalı ölçer (tersleme değil, sıfır-geçişi) +
   yörünge düzeltme antisimetriği siler → birlikte hedef.
6. Başarısız yöntemler → **kısa bir "neden naif tersleme/genlik-okuma yetmez"
   bölümüne SIKIŞTIR** (şu anki uzun katalog yerine). Merkez artık mekanizma+çözüm.
7. Operasyon + zamansal bütçe (yeni, bkz. B) + dürüst kısıtlar.
**Not:** Bu, en büyük değişiklik. Kullanıcı onayıyla iskeleti kesinleştirip
uygulayalım. Başarısız-yöntem kataloğunu tamamen atmıyoruz, kısaltıyoruz
(hakemler "diğer yöntemler neden olmuyor" diye soracaktır).

### A3. "Either rotation alone is nearly harmless..." → "precession" + fizik
**Yorum:** "rotation" yerine "precession" demek daha doğru; çünkü spin yatay
düzlemde dönerse (in-plane presesyon) gerçek EDM sinyali de birikmez.
**Değerlendirme:** Haklı. Doğru fizik: dikey ofset dy → radyal B_x → **radyal
eksen etrafında presesyon** (spini düzlem-dışına eğebilir); yatay ofset dx →
dikey B_y → **dikey eksen etrafında presesyon = düzlem-İÇİ** (S_y biriktirmez,
tam da frozen-spin düzlemi). "Rotation" gevşek; "precession" doğru terim.
**Önerilen değişiklik:** "rotation" → "precession". Ve cümleyi netleştir: dikey
eksen etrafındaki presesyon düzlem-içidir (S_y'ye katkısız); radyal eksen
etrafındaki tek başına halka boyunca ortalanır. Tehlike, ikisinin
komüte-etmemesinden doğan net dikey presesyon. (Kesin ifadeyi uygularken özenle.)

### A4. "Tek bir RMS parametresiyle ifade edilemez" çıkarımı eksik mi?
**Yorum:** Sim/antisim modların farkına vardıktan sonra, olayın rms-misalignment
gibi TEK parametreyle ifade edilemeyeceğini anlamadık mı?
**Değerlendirme:** Evet, ve bu AÇIKÇA vurgulanmalı — önemli kavramsal mesaj.
Sahte-EDM bilineer bir fonksiyonel (dört kanal); σ² yalnız GENLİK ölçeklemesi,
ama verilen bir RMS'te DEĞER desene (sim/antisim içerik + çapraz terimler)
bağlı. Yani "sahte-EDM = f(σ_rms)" YANLIŞ; doğru ifade "f = bilineer form(v_x,v_y),
σ²-homojen ama desen-bağımlı." Fig. kanal saçılımı (seed'e göre 3.5–900×) bunu
zaten gösteriyor.
**Önerilen değişiklik:** Mekanizma bölümüne (A2-4) net bir cümle/çıkarım ekle:
"Sahte-EDM tek bir hizalama-RMS sayısıyla ÖZETLENEMEZ; iki düzlemin
kaçıklık-desenlerinin bilineer bir fonksiyonelidir. σ² yalnız ölçeklemedir;
sabit σ'da değer sim/antisim içeriğe göre onlarca kat değişir." Bu, aynı zamanda
Omarov-tarzı "σ=10μm bütçesi"nin neden eksik olduğunu da açıklar.

### A5. "C. What is already known" tuhaf listeleme → prose
**Yorum:** Oradaki metin tuhaf bir listeleme olmuş; makale formatına uygun mu?
**Değerlendirme:** Haklı. Şu an \paragraph{Orbit correction}/{BBA}/{False-EDM
budgets}/{Neighboring work} şeklinde açıklamalı-kaynakça gibi; PRAB'de ilgili-iş
genelde giriş içinde AKICI prose olarak örülür.
**Önerilen değişiklik:** Bu alt-bölümü akıcı 1–2 paragrafa dönüştür ve A2'deki
yeni yapıda "yörünge düzeltme standart bir araçtır (kaynaklar...)" anlatısının
içine yedir. Ayrı bir "bilinenler listesi" olarak durmasın. "Neighboring work"
tek-cümle kaynak yığını da düzyazıya çekilsin ya da azaltılsın.

### A6. (kullanıcı devam edecek — mesaj "*" ile açık bırakıldı)
Bekleyen ek yorumlar buraya eklenecek.

---

## B. Birleşik operasyonun zamansal bütçesi (gerçekçi, varsayımlar İŞARETLİ)

> Amaç: BBA + yörünge düzeltme + CW/CCW + sürekli drift-düzeltme zincirinin
> gerçek deneyde zaman maliyeti. Simülasyon DUYARLILIK verir; MUTLAK süreler için
> makine-özel drift HIZI gerekir → ⚠️ ile işaretli varsayımlar.

**Simülasyondan gelen sağlam girdiler (C++):**
- Sürekli yörünge düzeltme, drift'in antisimetrik (baskın) parçasını siler:
  rastgele drift σ_d ≤ ~5 μm iken sahte-EDM hedefin ALTINDA kalır; 10 μm'de
  birkaç× (drift_cwccw_test: 2μm→0.05×, 5μm→0.3–0.6×, 10μm→3–6×, feedback'li).
- Feedback YOKSA drift hızla geri yükler (10μm → yüzlerce–binlerce×).
- BBA yalnız yavaş biriken SİMETRİK drift için gerekir (antisim'i orbit halleder).
- CW/CCW farkı ~2–3× ek (tek-seed; ensemble ile kesinleşecek).

**Zaman ölçekleri (kaba, ⚠️ varsayımlı):**
| adım | süre | kadans | not |
|---|---|---|---|
| İlk BBA (47 quad × 2 düzlem, iteratif) | ⚠️ ~saatler–1 gün | bir kez (kurulum) | her quad: grad-mod + yörünge yerleşme + ortalama |
| Yörünge düzeltme | saniye–dakika | **sürekli** | rutin; antisim drift'i temizler |
| BBA tekrarı | ⚠️ ~saatler | ⚠️ ~günde bir? | simetrik drift ~2μm'ye ulaşınca |
| CW/CCW | demet ters çevirme ek yükü | ölçüm boyunca | ~2× veri süresi ya da eşzamanlı |
| **EDM veri alımı (polarimetri)** | **⚠️ aylar–yıl** | — | nrad/s istatistiği; ASIL baskın süre |

**Kritik köprü (⚠️ makine verisi gerek):** simetrik drift hızı. Örnek varsayım:
toplam rastgele drift ~1 μm/gün (termal+zemin, stabilizasyon sonrası) →
simetrik parça ~0.7 μm/gün → ~3 günde ~2 μm → **~3 günde bir BBA tekrarı**.
BBA ~saatler sürerse, veri-alım görev-çevrimi kaybı ~%birkaç.

**Sonuç (dürüst):** Baskın süre EDM istatistiği (aylar–yıl). Hizalama BAKIMI
(sürekli orbit + periyodik BBA) küçük bir görev-çevrimi ek yükü — DRIFT HIZI
~μm/gün mertebesindeyse. Yani zaman-yasak DEĞİL; ama bu, drift hızının gerçekten
~μm/gün olmasına bağlı (yüksekse BBA kadansı sıkışır, kayıp artar). Simülasyon
duyarlılığı verdi; kadansı sabitlemek için makine drift ölçümü (ya da literatür
tahmini) şart. Ayrıca donanım tabanı (grad-mod sırasında manyetik merkez
oynaması) BBA'nın kendi tekrar-doğruluğunu sınırlayabilir — modellenmedi.

---

## C. Uygulama planı (onay sonrası, toplu)
1. A2 iskeletini kesinleştir (en büyük iş — yeniden kurgu).
2. Abstract'ı kısalt (A1).
3. Mekanizma bölümüne "tek-RMS yetmez" çıkarımı (A4) + precession düzeltmesi (A3).
4. "What is already known" → akıcı prose (A5).
5. Zamansal bütçe bölümü ekle (B).
6. Kalan kullanıcı yorumları (A6) + çok-seed/CW-CCW sonuçları geldiğinde sayıları
   güncelle.
