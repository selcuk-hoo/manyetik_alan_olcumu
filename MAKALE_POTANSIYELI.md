# Makale Potansiyeli Değerlendirmesi

> Tarih: 2026-06-11. Bu rapor, repo'nun tüm sürüm geçmişi (v1.1 → v4.1 +
> aktif dal, 163 commit) ve mevcut belgeler/taslaklar incelenerek
> hazırlanmış bağımsız bir değerlendirmedir. Amaç: hangi iş paketlerinin
> hakemli yayına dönüşebileceğini, hangi dergiye uyacağını ve yayın öncesi
> eksikleri dürüstçe ortaya koymak.

---

## Projenin üç dönemi (sürüm geçmişinden)

| Dönem | Tag aralığı | Tema | Çıktı |
|---|---|---|---|
| I | v1.1–v2.8 (Mayıs ilk yarı) | K-modülasyon + tepki matrisi geri çatımı; SVD/Tikhonov; BPM hata modeli | `makale-taslagi-2.md` (drift izleme + k-mod alt sınırı) |
| II | v3.0–v4.1 (Mayıs sonu–Haziran başı) | Tek-yörünge CLEAN ile baskın harmonik kestirimi; sahte EDM mod taraması | `paper_draft.tex` (İng.), `makale_tr.tex` (Tür.) |
| III | aktif dal (Haziran 8–11) | Ölç-trimle mimarisi: spin-sürülü + yörünge-sürülü trim, eşik teorisi, taban analizi | `false_edm_harmonic_sinir.md` §12, `trim_yontemi_pedagojik.md`, `makale_tr.tex` trim bölümü |

Üç dönem üç farklı makale adayına karşılık geliyor. Aşağıda potansiyel
sırasına göre değerlendiriyorum.

---

## Aday 1 — Trim mimarisi makalesi (EN GÜÇLÜ ADAY)

**Çalışma başlığı önerisi:** *"Sahte-EDM'nin kaynağında bastırılması:
dondurulmuş-spin proton halkasında EDM-kör yörünge trimi ve spin-sürülü
ölç-trimle döngüsü"*

**Hedef dergi:** Physical Review Accelerators and Beams (PRAB).
İkincil: NIM-A.

### Neden güçlü?

Dönem III'ün sonuçları, tek tek ilginç bulgular değil, **kapalı bir
teori + doğrulama + sınır haritası** bütünü oluşturuyor — bir PRAB
makalesinin tam iskeleti:

1. **Pozitif sonuç (yöntem):** c_k kalibrasyonu desenden bağımsız
   (korelasyon 1.0000), doğrusal (%0.0 sapma), fırlatma koşulundan
   bağımsız (%0.5). Rastgele fazlı desende çift kuadratür ile faz iki
   ölçümde çözülür; iteratif döngü ≥10⁷× bastırır. Bu, "hizalama hatasını
   ölçmeden sahte EDM'yi bastır" yaklaşımının ilk sistematik gösterimi
   (bildiğim kadarıyla pEDM literatüründe yok; Omarov vd. toleransları
   verir ama aktif trim döngüsü önermez).

2. **Analitik çatı:** Yörünge kazanç yasası G_k = C/|Q_eff²−k²| ve
   kapalı form fit eşiği k_max² < Q_eff² + C·σ_q/σ_b. Yasa k≤6'ya
   oturtulup k=7..12'de **öngörü olarak** ≤%1 doğrulandı — hakemler bunu
   sever: fit edilen değil öngörülen doğrulama. Eşik formülü makineden
   bağımsız bir prosedür sunar (G_k yerinde ölçülür, eşik yeniden
   hesaplanır) — genellenebilirlik iddiasını taşıyan kısım bu.

3. **Sınır haritası (negatif sonuçlar — değerli):**
   - *Yörünge minimizasyonu ≠ sahte-EDM minimizasyonu* (korrektör testi:
     COD %50–95 düşer, sahte EDM ~%15). Tek başına alıntılanabilir bir
     bulgu; deney tasarımına doğrudan etki eder.
   - *Simetrik taban kanıtı:* 48 DOF'un 23'ü (QF/QD aynı-yön) yörüngede
     ~2.6×, spinde ~12× bastırılmış ama sıfır değil → BPM-trim tabanı
     ~10⁻⁴ rad/s. Dik-izdüşüm ayrıştırmasıyla kanıtlanmış.
   - *Eşik altı mod fit edilemez:* simetrik baz genişletmesi 9× kötüleştirir
     (ofset enjeksiyonu). "Neden daha fazla mod fit etmiyorsunuz?" hakem
     sorusunun cevabı baştan makalede.

4. **EDM-kör kanallar:** Yörünge trimi EDM sinyaline yapısal olarak kör
   (BPM EDM'yi görmez); radyal polarizasyon EDM'yi 3.5×10⁶× bastırırken
   kaçıklık sinyalinin %5'ini taşır → sistematik-yalnız kanal. Bu ikisi,
   "trim EDM sinyalini de söndürür mü?" itirazını kapatıyor.

5. **Deneysel reçete:** Üç kademe (yörünge → CW/CCW+flip → spin),
   her kademenin giriş/çıkış seviyeleri sayısal olarak bağlanmış.

### Yayın öncesi eksikler (dürüst liste)

| Eksik | Ağırlık | Çaba |
|---|---|---|
| Spin kademesinin uçtan uca gösterimi (yörünge trimi artığı → spin trimi → 10⁻⁵ altı) — şu an iki kademe ayrı ayrı gösterildi, zincirleme tek koşum yok | Yüksek | ~1 gün simülasyon |
| RF + sekstüpol açıkken doğrusallık teyidi | Orta | Mevcut altyapı |
| Tek kafes (24 hücreli FODO) — en azından bir farklı ton/hücre sayısıyla eşik formülünün taşınabilirliği | Orta | ~1 gün |
| t₂=1 ms izleme — spin koherans süresine (~1000 s) ekstrapolasyon argümanı yazılmalı | Orta | Yazım |
| x-düzlemi kaçıklıkları ve tilt çapraz terimleri | Düşük (kapsam dışı bırakılabilir, "future work") | — |
| Omarov zinciri değerlerinin (10⁻⁵→10⁻⁹) birebir referans teyidi | Düşük | Kütüphane |

**Değerlendirme:** Eksikler tamamlanırsa bu, alanına gerçek katkı yapan,
kendi ayakları üzerinde duran bir PRAB makalesi. Mevcut `makale_tr.tex`
trim bölümü zaten makalenin yarısı durumunda; ayrı makale olarak
çıkarılmalı (şu an k-mod makalesinin içinde ek bölüm gibi duruyor ve onu
şişiriyor).

---

## Aday 2 — K-modülasyon alt sınırı + drift izleme (SAĞLAM İKİNCİ)

**Kaynak:** Dönem I, `makale-taslagi-2.md` (428 satır, özet + yapı hazır).

**Hedef dergi:** NIM-A. İkincil: PRAB (kısa makale).

### Neden değerli?

İki tamamlayıcı sonuç içeriyor:

1. **Yapısal alt sınır (negatif sonuç):** İki-ölçümlü tam-ofset-iptal
   eden estimator sınıfında ‖ΔR⁻¹‖ ~ ‖R⁻¹‖/ε. pEDM koşullarında
   (ε≈0.02, BPM gürültüsü ~1 μm) k-modülasyon ruhundaki yöntemlerin
   10 μm hedefine ulaşamayacağını (10³ μm mertebesi) gösteriyor.
   k-modülasyon LEP/LHC'de standart araç olduğundan, "pEDM'de neden
   çalışmaz" sonucu topluluk için pratik değer taşır.

2. **Çalışan alternatif:** Kalibrasyon-referanslı drift izleme
   ŷ = R⁻¹(y−y₀) ile 6–7 μm RMS. Negatif sonucu yapıcı bir öneriyle
   dengeliyor.

### Yayın öncesi eksikler

- **Kritik açık uç taslağın kendisinde itiraf edilmiş:** örgü modeli
  hatalarının (β-beat, tune kayması) drift kestirim performansına etkisi
  ölçülmemiş. Bu test yapılmadan makale savunulamaz — hakem ilk bunu
  sorar. Mevcut altyapıyla yapılabilir (YAPILACAKLAR'da "Test 8" olarak
  zaten kayıtlı).
- Kalibrasyon anı referansının zamanla bayatlaması (BPM ofset drifti
  ölçüm driftinden ayrıştırılabilir mi?) — en azından tartışma bölümünde
  ele alınmalı.

**Değerlendirme:** Orta-yüksek potansiyel. Aday 1'den bağımsız bir soru
("hizalamayı *ölç*") üzerine kurulu olduğundan ayrı makale olması doğru.
Ama Aday 1'in mesajı ("ölçmek zorunda değilsin, trimle") bu makalenin
motivasyonunu kısmen aşındırıyor — giriş bölümünde iki yaklaşımın
tamamlayıcılığı (mutlak hizalama bilgisi hâlâ gerekli: ilk kurulum,
mekanik bakım, trim bütçesi) net yazılmalı.

---

## Aday 3 — Tek-yörünge CLEAN kestirimi (TEK BAŞINA ÖNERMEM)

**Kaynak:** Dönem II, `paper_draft.tex` (854 satır, İngilizce, ileri taslak).

### Dürüst değerlendirme

Bu çalışma kendi başına zayıfladı, çünkü Dönem III onu kısmen geçersiz
kılan iki şey gösterdi:

1. Gram matrisi ölçümü: mod parmak izleri yörünge uzayında zaten ~dik
   (korelasyon ≤%1.1). CLEAN gibi iteratif çıkarma algoritmasının çözdüğü
   problem — örtüşen bileşenlerin ayrıştırılması — burada baştan yok
   denecek kadar küçük; basit LSQ aynı işi tek adımda yapıyor.
2. "Baskın k=2 harmoniğini kestir" hedefi, "k=1..4'ü birden trimle"
   stratejisi karşısında ara ürün konumuna düştü; üstelik tekil mod
   bastırmanın sahte EDM'yi artırabildiği de gösterildi (§12.14).

**Önerim:** `paper_draft.tex`'i bağımsız makale olarak sürdürme. İçindeki
sağlam parçaları (tek-yörünge kestirim formülasyonu, CLEAN/LS/TSVD
karşılaştırması, BPM ofset beyazlık analizi) Aday 1'in "yöntem" bölümüne
veya Aday 2'nin kestirim bölümüne eritmek daha verimli. İngilizce metin
varlığı, Aday 1'in İngilizce versiyonu için hazır malzeme demek — bu da
bir değer.

---

## Yan ürün — Pedagojik malzeme

`README.md` (56 KB), `trim_yontemi_pedagojik.md` ve
`false_edm_harmonic_sinir.md` birlikte, depolama halkası EDM
sistematikleri üzerine ders notu / uzun-format eğitim makalesi
(ör. arXiv lecture notes, ulusal hızlandırıcı okulu materyali)
kalitesinde. Hakemli yayın değil ama görünürlük ve atıf açısından
maliyetsiz bir ek çıktı.

---

## Önerilen yol haritası

1. **Önce Aday 1'i bitir** (spin kademesi uçtan uca + RF/sekstüpol
   teyidi + ikinci kafes noktası) → `makale_tr.tex`'ten trim bölümünü
   ayır, bağımsız makale yap, `paper_draft.tex` İngilizce iskeletini
   buna dönüştür. Hedef: PRAB.
2. **Paralelde Aday 2'nin kritik eksiğini kapat** (β-beat/tune-error
   testi — mevcut altyapıyla). Sonuç olumluysa NIM-A'ya kısa makale.
3. CLEAN taslağını arşivle; parçalarını 1 ve 2'ye dağıt.

İki makalelik sağlam malzeme var; üçe bölmek hepsini inceltir.

---

## Genel gözlem (özgün değerlendirme)

Bu repo'nun en güçlü yanı, sonuçların çoğunun *çürütme denemesinden sağ
çıkmış* olması: side-band hipotezi kuruldu ve çürütüldü, korelasyon
hipotezi Gram matrisiyle test edilip elendi, kazanç yasası öngörüyle
sınandı, genişletilmiş baz denendi ve başarısızlığı mekanizmasıyla
açıklandı, topoloji fikri kararlılık analiziyle elendi. Bu "başarısız
denemeler + mekanizma açıklaması" zinciri makale taslaklarına şu an
yeterince yansımıyor — hakem güveni tam da buradan kazanılır. Aday 1
yazılırken bu reddedilen alternatifler ayrı bir alt bölümde
("yöntem sınırları ve elenen alternatifler") açıkça anlatılmalı.

En zayıf yan: tüm sonuçlar tek kafes, tek enerji, çoğunlukla RF'siz
idealize koşulda. Bir PRAB hakemi "bu bir simülasyon demonstrasyonu mu,
yoksa genellenebilir bir prosedür mü?" diye soracak. Eşik formülünün
makine-bağımsız prosedür olarak sunulması bu sorunun cevabı — ama en az
bir ikinci çalışma noktasıyla desteklenirse iddia kanıta dönüşür.
