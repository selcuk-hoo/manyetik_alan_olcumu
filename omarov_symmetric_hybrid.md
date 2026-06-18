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

*Bu özet kuadrupol hizalamaya odaklanır. Tam tasarım (elektrot geometrisi, SCT,
sekstupol optimizasyonu, GR düzeltmeleri, polarimetre detayı) için orijinal
makaleye — Phys. Rev. D 105, 032001 (2022) — bakınız.*
