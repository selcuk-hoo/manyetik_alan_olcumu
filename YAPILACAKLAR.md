# YAPILACAKLAR — Yeni Rekonstrüksiyon Yöntemi

Bu belge, manyetik alan ölçüm projesinin bir sonraki aşamasında ne yapılacağını, neden yapılacağını ve nasıl test edileceğini açıklamaktadır. Teknik bilgisi olmayan bir okuyucunun da izleyebileceği bir dille yazılmıştır.

---

## 1. Neden Eski Yöntem Terk Edildi?

### Eski yöntemin özeti
Önceki yaklaşımda şu mantık izleniyordu:

- İki farklı kuadrupol gücünde (kmod-1 ve kmod-2) makineden ölçüm alınır.
- İki ölçüm arasındaki **fark** hesaplanır (bunun BPM ofsetlerini iptal edeceği düşünülüyordu).
- Bu fark, "fark yanıt matrisi" denilen büyük bir matris aracılığıyla geriye çevrilerek kuadrupol hatalarına ulaşılmaya çalışılır.

### Neden işe yaramıyor?
Sorun, bu "fark yanıt matrisi"nin matematiksel olarak son derece kötü huylu olmasında yatıyor.

İki büyük sayının farkını almak, küçük hataları orantısız biçimde büyütür. Bunu şöyle düşünebilirsiniz: elinizde iki cetvel var, biri 100.00 cm, diğeri 99.99 cm. Farkı alırsanız 0.01 cm elde edersiniz. Ama her cetvel sadece 0.05 cm'lik ölçüm hassasiyetine sahipse bu fark tamamen anlamsızlaşır — hata, sinyalin 5 katına çıkmış olur.

Bizim durumumuzda durum bundan çok daha kötüydü: matrisin "bozukluk oranı" (kondisyon sayısı) yaklaşık 27.000 idi. Bu, girişteki yüzde 1'lik model hatasının çıkışta yüzde 270'lik rekonstrüksiyon hatasına dönüşmesi demek. Analitik hesaplamalarla bu sorun doğrulandı.

---

## 2. Neden Yeni Yöntem Daha İyi?

### Örgünün simetrisini kullanmak
Makine, büyük bir yüzük şeklinde döngüsel bir yapıya sahip. Her biri aynı mıknatıs düzeninden oluşan, birbirini tekrar eden hücrelerden meydana geliyor. Bu periyodik yapı, yanıt matrisine özel bir iç simetri (sirkülant yapı) kazandırıyor.

Sirkülant matrisler, normal matrislere kıyasla son derece verimli biçimde tersine çevrilebilir. Bunu olanaklı kılan araç, matematikten aşina olduğumuz Fourier dönüşümü — ya da dijital versiyonu olan Hızlı Fourier Dönüşümü (FFT).

### Temel fikir
Eski yöntemde fark matrisini ters çevirmeye çalışıyorduk (ki bu son derece güçtü). Yeni yöntemde ise:

- kmod-1 ölçümünü kendi yanıt matrisiyle ayrıca ters çeviriyoruz,
- kmod-2 ölçümünü kendi yanıt matrisiyle ayrıca ters çeviriyoruz,
- İki sonucun ortalamasını alıyoruz.

Her iki yanıt matrisi de (kmod-1 için olan ve kmod-2 için olan) sirkülant yapıya sahip. Dolayısıyla her birinin kondisyon sayısı çok daha makul — fark matrisininki gibi 27.000 değil, muhtemelen yüzlerle ifade edilen bir değer. Bu fark, rekonstrüksiyon hassasiyetine doğrudan yansıyor.

### BPM ofsetleri ne olacak?
BPM dedektörleri sıfırdan sapma gösterir — yani her dedektörün sabit bir "yanlış okuma" değeri vardır. Bu değerlere BPM ofseti diyoruz.

Eski yöntemde bu ofsetlerin fark alındığında iptal olacağı varsayılıyordu. Yeni yöntemde ise:

- Her iki ayrı kmod için geri dönüşüm yapıldığında, sonuç "gerçek deplasman + ofset etkisi" şeklinde çıkıyor.
- İki sonucun **ortalaması** alındığında ofset etkisi zayıflıyor.
- İki sonucun **farkı** alındığında ise saf ofset bilgisi elde edilebiliyor — bu değerli bir yan ürün.

Böylece hem kalibrasyondan hem de ofsetten kaynaklanan kirlilikler ayrıştırılabiliyor.

---

## 3. Ne Yapılacak — Adım Adım

### Adım 1: Temel Twiss modülünü yaz (`fodo_lattice.py`)

Bu dosya, makinenin ideal modelini içerecek. Simülasyon programına bağımlı olmayacak; sadece analitik hesaplamalar yapacak.

İçereceği işlevler:
- Manyetik rijitlik hesabı (parçacık enerjisinden)
- Tek bir kuadrupolün ve boşluğun transfer matrislerini oluşturma
- Twiss parametrelerini (beta fonksiyonu ve faz ilerlemesi) hücre hücre hesaplama
- Kuadrupolün "işaretli integre alan gücü" (KL) değerini hesaplama (yatay ve dikey düzlem için işaret değişir)
- Sirkülant matris oluşturma
- FFT ile sirkülant matrisin özdeğerlerini hesaplama
- FFT tabanlı geri dönüşüm: ölçülen sapmalara bakarak kuadrupol yerleştirme hatalarını hesaplama

Bu dosya, bağımsız bir kütüphane gibi davranmalı. Başka bir dosya tarafından `import` edilebilmeli; kendi başına çalıştırıldığında hiçbir şey yapmamalı.

---

### Adım 2: Ana analiz scriptini yaz (`spectral_inversion.py`)

Bu dosya, asıl işi yapacak. Dört aşamadan oluşacak:

**Aşama A — İdeal Durum Analizi**
Makine tamamen mükemmel varsayılır. Yani gerçek Twiss parametreleri modeldekiyle aynıdır. Bu durumda yöntem ne kadar iyi çalışıyor? Teorik üst sınırı belirlememizi sağlar.

**Aşama B — Kondisyon Sayısı Karşılaştırması**
Eski yöntemin fark matrisinin bozukluk oranı, yeni yöntemin iki ayrı matrisinin bozukluk oranlarıyla karşılaştırılır. Her Fourier modunda (halka boyunca farklı dalga boylarında) bu oran ayrı ayrı hesaplanır. Hangi modların daha güvenilir olduğu görülür.

**Aşama C — İki-kmod Yöntemi**
kmod-1 ve kmod-2 için ölçümler ayrı ayrı simülasyondan alınır. Her biri kendi sirkülant matrisiyle geri çevrilir. Sonuçların ortalaması alınarak rekonstrüksiyon yapılır. BPM ofset etkisi ayrıca incelenir.

**Aşama D — Gürbüzlük Testi**
Simülasyon "gerçek makine" olarak kullanılır ve şu hatalar eklenir:
- Kuadrupol eğim hataları (tilts): kuadrupolün biraz döndürülmüş olduğu varsayılır
- BPM gürültüsü: dedektörlere rastgele küçük hatalar eklenir
- Model hatası: analitik modelde kullanılan Twiss parametreleri, simülasyondaki gerçek değerlerden biraz farklı tutulur

Bu koşullarda rekonstrüksiyon hassasiyeti ölçülür. Amaç, "Bu yöntem sadece mükemmel koşullarda mı çalışıyor, yoksa gerçekçi hataları da kaldırıyor mu?" sorusunu yanıtlamak.

---

## 4. Beklenen Çıktılar

Her analiz aşamasında şunlar üretilecek:

- Rekonstrüksiyon doğruluğu (korelasyon katsayısı ve ortalama hata, mikrometre cinsinden)
- Sirkülant matrisin her Fourier modundaki kondisyon sayısı grafiği (hangi modların güvenilir olduğunu gösterir)
- Gerçek vs. rekonstrükte edilmiş deplasman grafiği
- Gürbüzlük testi için: hata seviyesi vs. rekonstrüksiyon kalitesi eğrisi

---

## 5. Dosya Yapısı

Projede bundan böyle şu dosyalar bulunacak:

| Dosya | Amacı |
|---|---|
| `fodo_lattice.py` | Twiss hesabı, sirkülant matris, FFT geri dönüşüm — temel kütüphane |
| `spectral_inversion.py` | Ana analiz: ideal test, karşılaştırma, iki-kmod, gürbüzlük |
| `run_simulation.py` | Simülasyonu çalıştırma arayüzü (mevcut, değişmeyecek) |
| `plot_results.py` | Simülasyon sonuçlarını görselleştirme (mevcut, değişmeyecek) |
| `integrator.py` | C kütüphanesiyle köprü (mevcut, değişmeyecek) |
| `params.json` | Makine parametreleri (mevcut, değişmeyecek) |
| `README_v2.8.md` | Önceki analitik çalışmanın sonuçları ve dersleri |
| `README.md` | Proje genel açıklaması |

Silinen dosyalar: `analytic_kmod.py`, `build_response_matrix.py`, `test_kmod_reconstruction.py`, `test_reconstruction.py`, `scan_quad_tilt.py`, `scan_qtilt_contamination.py`

---

## 6. Kodlama Sırası

1. Önce `fodo_lattice.py` yazılır ve bağımsız olarak test edilir.
2. Ardından `spectral_inversion.py` yazılır; her aşama sırayla eklenir.
3. Aşama A çalışınca B'ye geçilir, B çalışınca C'ye, C çalışınca D'ye.
4. Her aşama kendi çıktısını üretebilir durumda olmalı — sonraki aşama henüz yazılmadan da çalışabilmeli.

---

## 7. Kritik Kontrol Noktaları

Kodlama sırasında şu sorular yanıtlanmalı:

- Sirkülant matrisin kondisyon sayısı gerçekten fark matrisinden çok mu küçük? (Sayısal olarak doğrulanmalı)
- FFT geri dönüşümü, doğrudan matris terslemesiyle aynı sonucu veriyor mu? (Tutarlılık kontrolü)
- kmod-1 ve kmod-2 ortalandığında BPM ofseti gerçekten azalıyor mu? (Simülasyonla gösterilmeli)
- Yüzde kaç model hatası tolere edilebiliyor? (Gürbüzlük testi bu soruyu yanıtlar)
