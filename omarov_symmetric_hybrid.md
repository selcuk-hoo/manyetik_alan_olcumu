# Omarov Simetrik-Hibrit Halka Tasarımı — Referans Özeti (Kuadrupol Hizalama Odaklı)

> **Kaynak:** Z. Omarov, H. Davoudiasl, ... Y. K. Semertzidis ve ark.,
> *"Comprehensive symmetric-hybrid ring design for a proton EDM experiment at
> below 10⁻²⁹ e·cm"*, **Phys. Rev. D 105, 032001 (2022)**.
>
> Bu belge makalenin **kuadrupol/element hizalama hatalarına karşı önerdiği
> strateji** açısından özetidir. Diğer konular (SCT, sekstupol optimizasyonu,
> genel görelilik, polarimetre) yalnızca hizalamayla ilişkili olduğu ölçüde
> geçer. Son bölüm (§7) bizim simülasyonumuzla (yörünge-görünmezlik, spin
> istatistik sınırı) doğrudan karşılaştırır.

---

## 1. Halka ve Temel Felsefe

| Parametre | Değer |
|-----------|-------|
| Tasarım | Simetrik-Hibrit: elektrik büküm + manyetik odaklama |
| Periyot sayısı | **24 FODO**, toplam **800 m** |
| Tünler | Q_x = 2.699, Q_y = 2.245 (**düşük tün ≈ 2**) |
| Quad gradyanı | ~0.2 T/m |
| Hedef | proton EDM, d_p < **10⁻²⁹ e·cm** ↔ dS_y/dt < **1 nrad/s** |

**Temel felsefe — üç katmanlı savunma.** Makale, "mıknatısları μm hassasiyetle
mekanik olarak hizalayalım" demez. Bunun yerine sistematiği hizalama hatasına
**duyarsız** kılar:

1. **Pasif (yapısal):** kafes simetrisi → sistematiği σ²-bastır.
2. **Aktif iptal:** CW/CCW karşı-dönen demetler + quad polarite çevirme.
3. **Aktif ölçüm-düzeltme:** spin-tabanlı hizalama (SBA), **yükselt-sonra-söndür**
   numarasıyla.

Mekanik tolerans bu sayede ~100 μm'ye gevşer; asıl hassasiyet demet/spin
geri-beslemesiyle elde edilir.

---

## 2. Kuadrupol Hizalama Hatasının Yarattığı Sistematikler (Tablo IV)

| Sistematik | T-BMT terimi | Hizalamayla ilişkisi |
|-----------|--------------|----------------------|
| **Dikey hız** ("rollercoaster") | S_x·β_y·E_x | Dikey misaligned quad → dikey yörünge dalgalanması → ⟨β_y⟩≠0 (büküm bölgelerinde). DM/DE için ana kaynak; **simetrik kafeste birkaç mertebe azalır**. |
| **Dipol E (Ey)** | S_s·β_s·E_y | Deflektör plakası tilt'inden. CR demetleriyle EDM'den ayrışır; **trim Ey plakalarıyla** sıfırlanır. |
| **Kuadrupol E (Equad)** | S_s·β_s·E_y (Equad=Ke·Δy) | CR demetleri By ile Δy kadar ayrıldığında parazit quad-E zıt yönde etkir → EDM-benzeri. **SBA ile ölçülüp düzeltilir.** |
| **Skew kuadrupol E** | Eskew·By | Quad x-y dönmesinden. SBA ile (By enjekte ederek) ölçülür. |
| **Geometrik (Berry) faz** | ∝ ardışık dönüş çarpımı ∝ **dx·dy** | **σ²** ölçeklenir. CR ayrımı < birkaç yüz μm (≈ birkaç μm quad kaçıklığı) ise ihmal edilebilir. |

> **Bizim için kritik:** Geometrik faz teriminin **dx·dy ∝ σ²** olması, bizim
> `false_edm_harmonic_sinir.md §13`'te bulduğumuz mekanizmanın aynısıdır.
> Omarov bunu Fig. 9(a)'da σ taramasıyla doğruluyor: aynı seed'lerle y = k·σ²
> uyumu **kusursuz** (her nokta herhangi bir σ'ya ekstrapole edilebilir).

---

## 3. Pasif Savunma: Kafes Simetrisi

- **Hybrid (4-fold)** tasarımında 4 uzun düz kesim simetriyi bozar → yalnızca
  azimutta "halka her iki yönde aynı görünen" özel quad'lar misalignment'a
  toleranslı (Fig. 6a'daki çukurlar).
- **Simetrik-Hibrit**'te tüm quad'lar boylamsal olarak eşdeğer → **hepsi**
  dikey misalignment'a toleranslı (Fig. 6b). Tek-tek 100 μm misalignment'ın
  dikey presesyon katkısı Hybrid-4fold'a göre mertebelerce düşük.
- Rastgele σ = 100 μm dağıtım (→ >1 mm CR ayrımı) bile radyal-polarize demette
  arka-plan dikey presesyonu **< 1 nrad/s**.

---

## 4. Aktif İptal: CR Demetleri + Quad Polarite Çevirme

Gerçek EDM, CW ve CCW demetlerinde **zıt işaretli** dS_y/dt verir; hizalama
kaynaklı sistematikler ise (dipol Ey gibi) **aynı işaretli**. Fark alınır:

$$\left.\frac{dS_y}{dt}\right|_{\text{EDM}} = \frac{1}{2}\left.\frac{dS_y}{dt}\right|_{\text{CW}} - \frac{1}{2}\left.\frac{dS_y}{dt}\right|_{\text{CCW}}$$

**Quad polarite çevirme** ikinci bir iptal kolu ekler. Fig. 9(b): CW + CCW +
quad-flip kombinasyonu, rastgele misalignment'ta bile arka-planı hedefin altında
tutar. Önemli: rastgele quad misalignment **tek başına** dikey spin birikimi
üretmez (düz-kafes testi); etki, dikey hız + ikinci-derece sistematiklerle
karışmadan doğar.

---

## 5. Aktif Ölçüm-Düzeltme: Spin-Tabanlı Hizalama (SBA)

**Ana fikir:** Spin dinamiği EM alanlara demet dinamiğinden **çok daha
duyarlıdır**; bu yüzden spin, kafes kusurlarının hassas probu olarak kullanılır.

### 5.1 Yükselt-sonra-söndür numarası (kilit fikir)

Parazit kuadrupol-E alanı E_quad başlangıçta **bilinmez**. Onu doğrudan minik
haliyle ölçmek yerine:

1. Manyetik quad'ların **dipol düzelticileriyle** seçilen bir N harmoniğinde
   **bilerek büyük** B_x uygula.
2. Bu, bilinmeyen E_quad'ın etkisini **yükseltir** → dS_y/dt ∝ E_quad × B_x
   artık **büyük ve hızlı ölçülebilir**.
3. dS_y/dt'den E_quad'ı (her N için) çıkar → düzelt (söndür).
4. **Düşük tün (≈2) sayesinde yalnızca birkaç harmonik N probe edilmesi yeter.**

Aynı reçete skew için: By enjekte et → dS_y/dt ∝ Eskew × By ölç → tüm N için
Eskew çıkar. Görüntü-yük, demet-demet vb. tüm even-multipol E alanları aynı
yöntemle ele alınır (etki E alanının kaynağına bağlı değil).

### 5.2 Hangi polarizasyon neyi ölçer

- **Radyal polarize demet** → dikey hız (orbit corrugation) geri-beslemesi.
- **Dikey polarize demet** → geometrik faz ve bilinmeyen sistematik testleri.
- **Boylamsal polarize demet** → asıl EDM araması.

### 5.3 Demet ayrımı = hızlı proxy

E_quad = K_e · Δy olduğundan, **CR demet ayrımı Δy** doğrudan ölçülebilir bir
vekildir. Dipol düzelticilerle Δy ayarlanır; **SQUID-tabanlı BPM**'lerle
(çözünürlük ~10 nm/√Hz) ölçülür. Demet ayrımı 100 μm'nin çok altına ince
ayarlanır. Bu, spin-istatistiğine değil, **demet-dinamiğine** (bol istatistik,
her tur) dayanan bir ölçümdür.

---

## 6. Hizalama Gereksinimleri ve Erişilebilirlik

| Gereksinim | Değer | Nasıl |
|-----------|-------|-------|
| CR demet dikey örtüşmesi | **5 μm** (toplam 10 μm) | dipol düzelticiler + SQUID BPM |
| Genel dikey kapalı-yörünge düzlemselliği | **50 μm** | mekanik + düzeltme |
| Mekanik düzlemsellik (erişilen) | **~10 μm** | su terazileri ile [50,51] |
| Tolere edilen B alanı | **alt-nT** | flux-gate manyetometre + Helmholtz bobinleri |
| Trim Ey ile aynı-işaret presesyon | **< 10⁻⁶ rad/s** | simetrik dağıtılmış trim plakaları |
| Polarimetre cross-ratio sistematik | **10⁻⁵** (COSY'de kanıtlı) | konum/açı sürücü terim modeli |

**Sonuç (Fig. 16):** Tüm gerçekçi kusurlar (yatay+dikey quad misalignment +
deflektör tilt) dahil edilip CW/CCW + quad-flip kombinasyonu uygulandığında,
artık EDM-benzeri arka plan **dS_y/dt < 1 nrad/s** → 10⁻²⁹ e·cm hedefi
karşılanır. "Hizalama gereksinimleri mevcut teknolojiyle erişilebilir."

---

## 7. Bizim Simülasyonumuzla Karşılaştırma (kritik)

| Konu | Bizim yaklaşım (mevcut) | Omarov'un yaklaşımı |
|------|------------------------|---------------------|
| Sahte EDM mekanizması | dx·dy geometrik faz, **σ²** | Aynı: geometrik faz ∝ dx·dy, σ² (Fig. 9a) |
| Misalignment ölçümü | Kapalı yörünge + k-mod + Fourier | **CR demet ayrımı Δy** (farklı gözlemlenebilir) |
| Yörünge görünmezliği | Simetrik alt-uzay (yüksek-k) görünmez | Onlar tek-demet kapalı yörüngesini değil, **CR ayrımını** kullanır |
| Spin istatistik sınırı | Artığı 10⁻⁹'da ölçmek ~1 yıl | **Yükselt-sonra-söndür**: büyük B_x ile sinyali yükselt → hızlı ölç |
| Harmonik sayısı | k=1..12 tarıyoruz | Düşük tün ≈2 → **yalnız birkaç N** gerekli |
| Nihai iptal | CW/CCW + quad-flip (planlanan) | CW/CCW + quad-flip (uygulanan, Fig. 9b/16) |

### Omarov'un bizim "yöntem kullanılamaz" ikilemine üç cevabı

1. **Artığı asla doğrudan ölçme — yükselt.** Bilinmeyen E_quad'ı küçük haliyle
   ölçmeye çalışmak yerine, bilinen büyük B_x düğmesiyle yükseltip dS_y/dt'yi
   büyük yapar, hızlı ölçer, katsayıyı çıkarır, söndürür. Spin hiçbir zaman
   10⁻⁹'luk artığı doğrudan ölçmez → istatistik duvarı aşılır.

2. **Demet ayrımı = bol-istatistikli proxy.** E_quad ∝ Δy olduğundan, asıl
   ölçüm SQUID-BPM ile demet ayrımıdır (her tur, bol istatistik), spin değil.
   Spin yalnızca katsayı kalibrasyonu ve nihai doğrulama için.

3. **Düşük tün → az harmonik.** Q≈2 olduğu için yalnızca birkaç azimutal
   harmonik probe edilmesi yeterli; 48-boyutlu tam tarama gerekmez.

> **Çıkarım:** Bizim "spin-trim'i 10⁻⁹'a zorlamak ~1 yıl alır" endişemiz doğru,
> ama Omarov bunu **yapmıyor**. Onun reçetesi (a) demet ayrımını hızlı ölç, (b)
> bilinen düğmelerle spin sinyalini yükseltip katsayı çıkar, (c) gerisini
> CR+flip ile söndür. Bizim yöntemimizin "kullanılabilir" sürümü tam olarak bu
> mimaridir.

---

## 8. Stratejik Sonuç: Spin Yolunda Katkı Sınırı

Bu üç makaleyi (Omarov + COSY polarimetre + kendi simülasyonumuz) yan yana
koyunca varılan dürüst değerlendirme:

**Saf "spin ile ölç-düzelt" çerçevesinde özgün katkı payı dardır.** Omarov işin
çekirdeğini yayınlamış durumda:
- Spin-tabanlı hizalama (SBA),
- yükselt-sonra-söndür numarası (bilinen B_x/B_y düğmesiyle bilinmeyen E_quad'ı
  amplifiye edip hızlı ölç → istatistik duvarını aş),
- CR demet ayrımı = bol-istatistikli proxy,
- CW/CCW + quad-flip iptali,
- gerçekçi misalignment'la dS_y/dt < 1 nrad/s gösterimi.

Bizim bir spin-trim yöntemimiz bunların çoğunu yeniden türetir → manşetlik
yenilik değil.

**Omarov'un YAPMADIĞı (olası boşluklar):**
1. **Per-quad misalignment haritası çıkarmıyor.** SBA, alan harmoniklerini
   (her N için net E_quad) söndürür; "şu quad şu kadar kaçık" demez. Ama biz de
   gösterdik ki bu harita fizik olarak belirlenemez (kapalı yörünge simetrik
   alt-uzayı göremez; spin yalnız net harmonik etkiyi verir) → **negatif sonuç.**
2. **SBA'nın zaman/istatistik bütçesini vermiyor.** Mümkün olduğunu gösterir,
   "kaç depolama / kaç gün / hangi tolerans seviyesinde istatistik duvarı"
   sorusunu yanıtlamaz. Bizim t ∝ 1/f² ölçeklemesi + COSY FoM + yükselt-söndür
   kazancı bunu sayısal bütçeye çevirebilir — pratik ama ikincil seviye katkı.
3. **Farklı kafes** (elektrik-büküm hibrit vs bizim manyetik odaklamalı) →
   bağımsız doğrulama değeri sınırlı.

**Karar:** Katkıyı spin tarafında aramak yerine **yörünge/ölçüm tarafının gerçek
sınırını** (ne kurtarılabilir, ne kurtarılamaz — kesin gözlenebilirlik teoremi)
ya da Omarov'un dokunmadığı bir gözlemlenebilir kanalı araştırmak daha verimli.
Açık aday kanallar §9'da tartışılıyor.

---

## 9. Açık Kapalı-Yörünge Kanalı Araştırması (devam eden)

**Soru:** Spini bir kenara bırakıp kapalı yörüngeyi zorlarsak, simetrik
alt-uzayı (sahte EDM'i süren, orbit-görünmez kısım) kurtaracak gözden kaçmış bir
kanal var mı?

**Görünmezliğin kökü (polariteden bağımsız).** Simetrik OFSET (hücre içi
QF=QD=a_c) → kuadrupol gradyan işareti QF/QD'de değiştiği için **alternatif
(yüksek-k, k≈24) KICK** deseni üretir. Kapalı yörünge rezonant bir alçak-geçiren
filtredir (kazanç tünde, Q≈2.7, tepe yapar; G_k ∝ 1/|Q²−k²|). k≈24 ≫ Q olduğundan
bastırılır. Antisimetrik ofset ise düzgün (düşük-k) kick → görünür.

**Quad-flip bu kanalı AÇMAZ.** Gradyanları g→−g çevirmek tüm kick'lerin işaretini
toptan çevirir; alternatif desen alternatif kalır, k≈24 yine tünden uzak. Flip
ikinci ve optik-olarak-farklı bir tepki matrisi R′ verir (Q_x↔Q_y kayar) →
**marjinal (tüne yakın) modların** koşullanmasını iyileştirir, ama derin-bastırılmış
yüksek-k simetrik kanalı kurtarmaz. "Flip-öncesi QF / flip-sonrası QD" ayrıştırması
yalnız zaten görünür modlarda işe yarar.

**İntuisyona göre gerçekten açık olabilecek adaylar (test edilmeli):**
1. **Per-quad k-modülasyonu (demet-tabanlı quad-merkez bulma).** Her quad'ın
   gradyanını modüle edip yörünge tepkisini izlemek, o quad'daki demet-göreli
   ofseti **yerel** ölçer — rezonant alçak-geçirenden geçmez, bireysel ofsetleri
   (simetrik dahil) görebilir. Sınır: BPM ofseti (~100 μm) — projenin başlangıç
   problemi. **k-mod × quad-flip** kombinasyonu BPM-ofsetini quad-merkezinden
   ayırabilir (iki polarite, aynı geometriye iki bağımsız kısıt) → BBA tabanını
   düşürebilir. Omarov'da yok.
2. **İkinci-derece / bilineer yörünge gözlemlenebilirı (∝ dx·dy).** Kapalı yörünge
   ofsette doğrusaldır ama sahte EDM bilineerdir. Belirlenemez ofset haritasını
   kurmak yerine, sahte-EDM'i asıl süren ⟨dx·dy⟩ harmoniklerini bir ikinci-derece
   yörünge imzasından doğrudan ölçmek mümkün olabilir. Spekülatif — genliğin
   gürültü üstünde olup olmadığı simülasyonla sınanmalı.

Öncelik: (1) en somut ve en az spekülatif → önce `k-mod × flip` BBA tabanı
sayısal çıkarılmalı.

---

*Bu özet kuadrupol hizalamaya odaklanır. Tam tasarım (elektrot geometrisi, SCT,
sekstupol optimizasyonu, GR düzeltmeleri, polarimetre detayı) için orijinal
makaleye — Phys. Rev. D 105, 032001 (2022) — bakınız.*
