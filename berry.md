# Berry Fonksiyoneli: Sahte EDM'yi Kapalı Yörüngeden Öngörmek (lisans-seviyesi anlatım)

> **Tek cümlelik soru:** Parçacığın halka içinde çizdiği **yörüngeyi** (BPM'lerle
> ölçülen) bilirsek, o yörüngenin ürettiği **sahte EDM sinyalini** (spinin yanlış
> dönmesini) hesaplayabilir miyiz?
> **Cevap (2026-06): EVET, yaklaşık olarak.** Basit makine-öğrenmesiyle, sahte
> EDM'yi yörüngeden **%88 doğrulukla** öngören bir formül bulduk. Aşağıda *nasıl*
> yaptığımız adım adım, hiç bilmeyen birine anlatılıyor.

---

## 1. Problem nedir? (sezgi)

Bir EDM deneyinde proton spininin **çok yavaş** dikey dönmesini ararız — gerçek
sinyal bu. Ama mıknatıslar mükemmel hizalı değilse, spin **sahte** bir dikey dönme
de yapar; buna "sahte EDM" (false EDM) deriz. Bu sahte dönme, parçacığın halka
içindeki **kapalı yörüngesine** bağlıdır: yörünge yatay (x) ve dikey (y) yönde
kıvrıldıkça, spin her turda biraz dikeye kayar (bir tür "geometrik faz" / Berry
fazı — bir döngüde geriye kalan artık).

**Anahtar fikir:** Sahte EDM, kaçıklığın değil, **yörüngenin** bir fonksiyonudur.
BPM'ler yörüngeyi zaten ölçer. O hâlde yörüngeden sahte EDM'yi *ileri yönde*
hesaplayabilmeliyiz — **eğer doğru formülü bilirsek.** Bu dosya o formülü arar.

(Niye önemli: kaçıklığı yörüngeden geri-çözmek "kötü-koşullu" bir ters problemdir
[no-go]. Ama sahte EDM'yi *doğrudan* yörüngeden hesaplamak ters problem değil; bu
yüzden no-go'ya takılmayabilir.)

## 2. "Fonksiyonel" ve "çekirdek" ne demek? (basitçe)

Yörüngeyi halka boyunca N noktada örnekleyelim: her noktada bir yatay konum xᵢ ve
bir dikey konum yᵢ. Sahte EDM tek bir sayı (f). Soru: f, xᵢ ve yᵢ'lerden nasıl
çıkar?

- **En basit tahmin:** f, x ve y'nin çarpımının ortalaması, f ≈ ⟨xᵢyᵢ⟩.
  (Fizik "f ∝ x·y geometrik faz" dediği için doğal başlangıç.) **Ama bu YANLIŞ
  çıktı** (§4): bazı yörüngelerde ⟨xy⟩≈0 iken f büyük.
- **Daha esnek tahmin:** her nokta **farklı ağırlıkta** olsun:
  $$f \approx \sum_{i} w_i\, x_i\, y_i$$
  Burada wᵢ, "halkanın i. noktasındaki x·y çarpımı sahte EDM'ye ne kadar katkı
  veriyor" demek. wᵢ'ler bilinmiyor; onları **veriden öğreneceğiz.** İşte
  "fonksiyonel" bu (yörüngeyi sayıya çeviren kural); wᵢ dizisine "çekirdek" denir.

## 3. Algoritma adım adım (gerçekte ne yaptık)

**Adım 1 — Veri üret.** Simülatörle (C++ parçacık+spin izleyici) 64 farklı
kaçıklık senaryosu koşturduk. Her senaryo için kaydettik:
- f = sahte EDM (tam izleyiciden, "gerçek cevap")
- yörünge: halka boyunca 480 noktada x(s) ve y(s).
(`berry_data/run1_data.npz`, `run2_data.npz`.)

**Adım 2 — Özellikleri kur.** Yörüngeyi N noktaya indir (ör. N=24). Her senaryo
için N tane "özellik" hesapla: pᵢ = xᵢ·yᵢ. Yani her senaryo bir özellik-vektörü,
ona karşılık bir f sayısı.

**Adım 3 — Ağırlıkları öğren (regresyon).** wᵢ'leri öyle seç ki Σ wᵢ pᵢ tüm
senaryolarda gerçek f'e en yakın olsun. Bu **en küçük kareler** problemi. Ama N
ağırlık, 64 veriye yakın → ezberleme (overfit) riski. Bunu önlemek için
**ridge** (Tikhonov) cezası ekledik: ağırlıkların gereksiz büyümesini bastıran
küçük bir terim. (Tek "ayar düğmesi" λ; birkaç değer denedik.)

**Adım 4 — Dürüstçe doğrula (KRİTİK).** "Ezbere mi öğrendi, gerçekten mi buldu?"
iki testle:
- **LOO çapraz-doğrulama:** Her senaryoyu sırayla dışarıda bırak, kalan 63'ten
  ağırlıkları öğren, dışarıdakini **tahmin et**. Tahminler ne kadar iyi? (R²:
  1.0 mükemmel, 0 işe yaramaz.) Bu, *görmediği* veride performansı ölçer.
- **Permütasyon testi (sahteyi yakalar):** f etiketlerini **rastgele karıştır**
  (yörünge–f eşleşmesini boz), aynı fiti yap. Eğer yöntem sağlamsa, karışık veride
  R² ≈ 0 çıkmalı. Eğer karışıkta da yüksek R² çıkıyorsa → yöntem gürültüye uyuyor,
  sonuç sahte. (Bu oturumda daha önce bu tuzağa düşmüştük; o yüzden bu test şart.)

## 4. Sonuç

| Model | LOO-R² (görmediği veride) | Permütasyon (karışık) | Verdikt |
|------|------|------|------|
| (1) uniform ⟨xy⟩ | 0.26–0.36 | — | zayıf |
| **(2) ağırlıklı Σ wᵢxᵢyᵢ** | **0.88** | maks ~0.2 | **GERÇEK** |
| (3) + komşu "Berry-alan" terim | 0.91–0.94 | maks ~0.15 | gerçek (az ek) |

Okuması: **uniform ortalama** sahte EDM'nin yalnız ~%30'unu açıklar; ama **noktalara
göre ağırlıklı** çarpım **%88'ini** açıklar — ve permütasyon temiz olduğundan bu
ezber değil, **gerçek bir formül.** Yani sahte EDM, yörüngenin *ağırlıklı bilineer*
bir fonksiyonelidir, ve makine bunu buldu. (3 farklı N'de tutarlı; `kernel_fit.py`.)

## 5. wᵢ'nin fiziksel yorumu (öğrenilen ağırlık nerede yoğun?)

Öğrenilen wᵢ'yi halka konumuna karşı çizdik (`berry_data/berry_weights.png`):

- **Kararlı:** farklı N ve λ'da aynı profil → fit hilesi değil, **gerçek** bir yapı.
- **Uniform DEĞİL:** ağırlık halka boyunca değişiyor (bu yüzden basit ⟨xy⟩ yetersiz).
- **Salınımlı + en güçlü s/C≈0'da:** kuplaj halkanın başlangıç bölgesinde (turun
  ilk ~%2'si) en büyük (negatif); ayrıca s/C≈0.17, 0.6, 0.8 civarında ikincil yapı.
- **Orbit genliğini basitçe izlemiyor:** yani ağırlık "yörünge nerede büyükse orası"
  değil; kendine ait, alan-yapılı bir profil.

**Fiziksel okuma (kısmi, dürüst):** Salınımlı profil, kuplajın belirli halka
bölgelerinde (muhtemelen radyal-alanın etki ettiği yerler — deflektör/quad
bölgeleri ya da betatron-faz yapısı) yoğunlaştığını gösterir; uniform değil. **Ama
tam element-eşlemesi (hangi ağırlık tepesi hangi fiziksel elemana karşılık?) henüz
yapılmadı** — bunun için lattice'in alan-haritasını (deflektörlerin s-konumları,
betatron fazı) profile bindirmek gerekir. Sıradaki en bilgilendirici adım budur.

## 6. Açık problemler (öncelik sırası)

1. **wᵢ'nin element-eşlemesi:** ağırlık profilini deflektör/quad konumları ve
   betatron faziyle bindir → "kuplaj nerede" sorusunu fiziksel cevapla. (Ucuz,
   en bilgilendirici; analitik türetmeye köprü.)
2. **Daha çok config:** 64 az; ~birkaç yüz ile R² ve wᵢ pekişir (tracker pahalı,
   paralel üret).
3. **48-BPM + gürültü:** öğrenilen fonksiyonel gerçekçi BPM örneklemesiyle korunur mu?
4. **Analitik türetme (uzak hedef):** Berry fazını Thomas-BMT spin denkleminden
   türetip Σwᵢxᵢyᵢ formunu ve wᵢ'yi *kanıtla*. Şimdilik ML keşif aracı; bu son adım.
5. **Makaleye entegrasyon:** "hangi yörünge modları sahte EDM'yi sürüyor" (wᵢ'den)
   ile "hangi modlar gözlenebilir" (drift makalesi) birleştirilip **izlenmesi
   öncelikli modlar haritası** çıkarılabilir → makaleyi özgün/faydalı yapar (bkz.
   sohbet tartışması).

## 7. Reprodüksiyon (`berry_data/`)

- `run1_data.npz` (24 config) + `run2_data.npz` (40 config): her config `f`, tam
  yörünge `xo`,`yo` (480 nokta), kaçıklıklar `dx`,`dy`.
- `kernel_fit.py` — §3–4 analizi (tracker GEREKMEZ, npz'den çalışır):
  `python3 berry_data/kernel_fit.py`.
- `berry_weights.png` — §5 ağırlık profili figürü.
- `false_edm_4d.py`, `false_edm_mode_scan.py` — doğrulanmış sahte-EDM estimator
  (git'ten restore: `5cba757`, `41b1c6a~1`).
- `run1_gen.py`, `run2_gen.py` — yeni config üretimi (tracker; `bash
  build_integrator.sh` sonrası).
