# COSY LYSO Polarimetresi — Referans Özeti ve Spin Yöntemi Zaman Bütçesi

> **Kaynak:** F. Müller, ... E. Stephenson ve ark. (JEDI işbirliği),
> *"A New Beam Polarimeter at COSY to Search for Electric Dipole Moments of
> Charged Particles"*, JINST'e gönderim için hazırlanmış, arXiv:2010.13536v1
> [physics.ins-det], 22 Ekim 2020.
>
> Bu belge makalenin **polarimetre performansı** ve **EDM araması için gereken
> istatistik/süre** açısından özetidir. Son bölüm (§4) bu sayıları bizim
> simülasyondaki sahte-EDM eşiğimize (dS_y/dt ≲ 1×10⁻⁹ rad/s) ve spin ölç-trim
> zincirimize bağlar.

---

## 1. Cihaz ve Çalışma İlkesi

Depolama halkasında dolaşan polarize **proton/deuteron** demetinin
polarizasyonunu, ileri açılı **karbon hedeften esnek saçılma** (spin-yörünge
etkileşimi) ile sürekli ölçen kalorimetrik bir polarimetre.

| Bileşen | Özellik |
|---------|---------|
| Kalorimetre | **52 LYSO** kristal modülü, 3×3×8 cm³ |
| Geometri | 4 simetrik blok: **up / down / left / right** |
| Okuma | SiPM (silikon foto-çoğaltıcı) |
| PID | İnce plastik sintilatör (ΔE), SiPM okumalı |
| Hedef | Hareketli **karbon blok**, 20 mm kalınlık (devreye alma) |
| Boyut | ~620 mm cihaz; bir COSY düz kesiminde **1.3 m**'ye sığar |
| Çıkış penceresi | 800 µm paslanmaz çelik |

**Çalışma ilkesi.** Donmuş-spin (frozen-spin) koşulunda boylamsal polarizasyon
hız vektörüne paralel tutulur (RF'ye gerçek-zamanlı geri besleme ile). EDM,
parçacık çerçevesindeki radyal E alanı üzerinden bir tork uygulayar
polarizasyonu yavaşça **düşeye** döndürür → hedeften saçılmada **sol/sağ (L/R)
asimetrisi** zamanla artar. EDM sinyali = bu L/R asimetrisinin sekuler değişimi.
Düzlem-içi yön ise **yukarı/aşağı (U/D)** asimetrisi sıfırda tutularak (feedback)
denetlenir.

Saçılmada spine bağlı tesir kesiti:

$$\sigma_{\text{pol}}(\theta,\phi) = \sigma_{\text{unpol}}(\theta)\,
[\,1 + p_y\,A_y(\theta)\,\cos\phi\,]$$

(deuteron spin-1 için $pA$ teriminin önünde ek 1.5 çarpanı). Buradan ölçülen
asimetri:

$$\epsilon = \frac{L-R}{L+R} = \tfrac{3}{2}\,p_y A_y \quad(\text{deuteron}).$$

---

## 2. Performans Nicelikleri

### 2.1 Analiz gücü (analyzing power, A_y)

Analiz gücü, saçılmanın polarizasyona duyarlılığının ölçüsüdür; büyük olması hem
istatistik (∝ A_y²) hem de sistematiklere karşı kaldıraç sağlar.

| Nicelik | Değer | Not |
|---------|-------|-----|
| Proton donmuş-spin noktası | **p = 700.7 MeV/c** (T_p = 232.8 MeV) | yalnız E alanı ile büküm; CW/CCW iptali mümkün |
| p+C analiz gücü maksimumu | T_p ≈ 210 MeV civarı | donmuş-spin noktası tesadüfen maks. yakın |
| Kalibrasyon (transfer hattı, 40°) | **A_y = 0.61 ± 0.04** | esnek saçılma referansı |
| Polarimetre dizisi (toplam) | **A_y ≈ 0.15 ± 0.01** | ilk kolonun küçük-açı yüksek hız ağırlığı düşürür |
| Kullanışlı açı aralığı | **4° – 15°** (lab) | Coulomb-nükleer girişim kenarından yukarı |

> A_y'nin "dizi-toplam 0.15" değeri yanıltıcı olmasın: bu, küçük açılı (yüksek
> tesir kesiti, düşük A_y) olayların ağırlığından gelir. Açıya göre çözüldüğünde
> (Fig. 11) A_y dış kolonlarda 0.5–0.6'ya çıkar. İstatistik için doğru nicelik
> aşağıdaki FoM'dir.

### 2.2 Liyakat ölçütü (Figure of Merit)

$$\text{FOM} = \sigma(\theta)\,A_y^2(\theta), \qquad
\text{FOM}_{\text{mod}} = \sigma(\theta)\,A_y^2(\theta)\,\sin\theta$$

(sinθ çarpanı, büyük polar açıda artan katı açıyı hesaba katar). Polarizasyon
ölçümündeki istatistik hatanın **karesinin tersi** ile ölçeklenir → polarimetre
açı kapsaması bu büyüklüğün integralini maksimize edecek şekilde seçilir. 52
LYSO modülünün ön yüzü, en verimli (sol/sağ) bölgeyle güçlü örtüşür (Fig. 2).

### 2.3 Verim (efficiency)

$$\eta = \frac{\text{polarizasyon ölçümünde kullanılan olay sayısı}}
{\text{polarimetre ile etkileşip kaybedilen demet parçacığı sayısı}}$$

| Nicelik | Değer |
|---------|-------|
| Ölçülen verim (devreye alma) | **η = 1.11 %** |
| Beklenen / tasarım | ~1 % (herhangi bir polarimetre için sınıra yakın) |
| Devreye alma demeti | deuteron, p = 970 MeV/c (T_d = 238 MeV), dolum başına ~10⁹ |

Bu %1.1 verim, referans [3]'ün proton EDM araması için 10⁻²⁹ e·cm
gereksinimini (L/R sayım hızı temelli) **karşılar**.

---

## 3. Sistematik Hata Bastırma (cross-ratio)

İki zıt polarizasyon durumu × iki taraf (L/R) → 4 sayım hızı; **cross-ratio**
şeması birinci-derece hataların çoğunu iptal eder:

$$pA = \frac{r-1}{r+1}, \qquad r^2 = \frac{L_+ R_-}{L_- R_+} \tag{1.2}$$

Tek bir sistematik sürücü terim (konum + açı hataları) ileri-saçılma seçimiyle
tahmin edilebilir:

$$\phi = \frac{s-1}{s+1}, \qquad s^2 = \frac{L_+ L_-}{R_+ R_-} \tag{1.3}$$

COSY'de daha önceki testler [8] bu düzeltmelerin hataları **sinyalin 10⁻⁵
seviyesinin altına** indirdiğini ve o seviyeye dek sorun çıkmadığını gösterdi —
EDM araması için zorunlu eşik. Polarize-olmayan dolumlarla "yarı cross-ratio"
(Eq. 1.4) ek kaldıraç sağlar.

> **Bizim için bağlam:** Bu 10⁻⁵ rakamı, polarimetre+cross-ratio'nun
> *ölçüm-tarafı* sistematik tabanıdır. Bizim simülasyonda uğraştığımız sahte EDM
> ise *halka-tarafı* (kuadrupol hizalama) sistematiğidir; ikisi bağımsız ve her
> ikisinin de sinyal seviyesinin altına bastırılması gerekir.

---

## 4. EDM Araması için İstatistik ve Süre Bütçesi

### 4.1 Makaledeki temel rakamlar (§1.1)

| Nicelik | Değer |
|---------|-------|
| Proton EDM hedef duyarlılığı | **d_p = 10⁻²⁹ e·cm** |
| Gözlenecek düşey dönme | **µrad** seviyesi |
| Demet depolama süresi | **≥ 1000 s** (dolum başına) |
| Toplam çalışma süresi | ~**1 yıl** |
| Gereken "iyi" polarimetre olayı | **> 10¹²** |
| Dolum başına parçacık | **10¹⁰** (CW/CCW arasında bölünür) |
| Seçim öncesi tetik hızı | ~**10⁶ /s** |
| Cross-ratio sistematik tabanı | sinyalin < **10⁻⁵**'i |

### 4.2 Tek-ölçüm açı çözünürlüğü

Asimetriden polarizasyon-açısı çıkarımının istatistik hatası (cross-ratio,
N analiz edilmiş olay):

$$\sigma_\theta(N) \;\approx\; \frac{1}{A_y\,P\,\sqrt{N}}$$

Makale değerleriyle ($A_y \approx 0.6$ etkin, demet polarizasyonu $P \approx 0.8$,
$N = 10^{12}$):

$$\sigma_\theta \approx \frac{1}{0.6\cdot 0.8\cdot 10^{6}} \approx 2\times10^{-6}\ \text{rad} = 2\ \mu\text{rad}$$

→ makalenin "µrad seviyesinde dönme" ifadesiyle **tutarlı**.

### 4.3 Sekuler hız (dS_y/dt) çözünürlüğü → bizim eşiğimizle bağlantı

EDM sinyali bir doğrusal düşey birikimdir: $S_y(t) = (dS_y/dt)\,t$. T süreli bir
depolamada, olaylar zamanda düzgün dağılı ise eğim kestiriminin hatası:

$$\sigma_{\dot S_y} \;=\; \frac{\sqrt{12}}{A_y\,P\,\sqrt{N}\;T}$$

**Yıllık kampanya tahmini** (sırf mertebe — kesin değil):
- Canlı ölçüm süresi ~ yarım yıl ≈ 1.5×10⁷ s, T = 1000 s/dolum → ~1.5×10⁴ dolum.
- N_tot = 10¹² olay → dolum başına N ≈ 7×10⁷.
- Dolum başına: $\sigma_{\dot S_y}^{\text{dolum}} = \sqrt{12}/(0.6\cdot0.8\cdot\sqrt{7\times10^7}\cdot 1000) \approx 9\times10^{-7}$ rad/s.
- 1.5×10⁴ dolumu birleştir: $\sigma_{\dot S_y}^{\text{yıl}} \approx 9\times10^{-7}/\sqrt{1.5\times10^4} \approx \mathbf{7\times10^{-9}\ \text{rad/s}}$.

**Sonuç — neden hedefimiz 10⁻⁹ rad/s?**
10⁻²⁹ e·cm'lik gerçek proton EDM'i, donmuş-spin halkasında **dS_y/dt ~ 10⁻⁹
rad/s** mertebesinde bir düşey spin birikimi üretir. Yukarıdaki hesap, bir yıllık
polarimetre istatistiğinin tam da bu mertebeyi (~7×10⁻⁹ rad/s) çözebildiğini
gösterir. Dolayısıyla:

> **Herhangi bir sahte-EDM sistematiği (kuadrupol hizalama dahil) bu istatistik
> tabanın — yani ~10⁻⁹ rad/s'nin — altına bastırılmalıdır.** Bizim
> simülasyondaki dS_y/dt ≲ 1×10⁻⁹ rad/s hedefimiz doğrudan bu polarimetre
> duyarlılığından gelir; daha sıkı olması istatistiksel olarak gereksiz, daha
> gevşek olması ise sahte sinyali gerçek EDM'e karıştırır.

### 4.4 Spin ölç-trim zincirimizle ilişki

`trim_yontemi_pedagojik.md` ve `false_edm_harmonic_sinir.md §14`'te gösterilen
zincir:

1. **Yörünge-trim (Kademe 1):** yalnız antisimetrik (k≤4) alt-uzayı görür;
   simetrik alt-uzayı gözlenebilirlik sınırı nedeniyle ~1.8×10⁻⁴ tabanında
   bırakır (6 rekonstrüksiyon metodu aynı tabana çarpar).
2. **Spin ölç-trim (Kademe 2):** polarimetre ile ölçülen dS_y/dt'yi geri besleme
   olarak kullanır; simetrik modu trimler (~6000× ek bastırma → ~10⁻⁷ rad/s).
3. **CW/CCW + quad-flip iptali:** kalanı 10⁻⁹ altına indirir.

Bu makale, Kademe 2 ve 3'ün **ölçüm aracını** sağlar: dS_y/dt'yi μrad/1000 s
duyarlılıkla okuyan polarimetre. §4.3'teki hesap, ölç-trim döngüsünün her
adımında dS_y/dt'yi ne kadar hızlı/kesin ölçebileceğimizin üst sınırını verir —
yani trim kalibrasyonu (±A_cal vur, Δf ölç) için gereken depolama sayısını.

---

## 5. Hızlı Referans Tablosu

| Sembol / nicelik | Değer | Kullanım |
|------------------|-------|----------|
| A_y (esnek, kalibrasyon) | 0.61 ± 0.04 | demet polarizasyon kalibrasyonu |
| A_y (dizi-etkin, FoM ağırlıklı) | ~0.4–0.6 | istatistik hesapları |
| η (verim) | 1.11 % | olay hızı bütçesi |
| Demet polarizasyonu P | ~0.5–0.8 | $\sigma_\theta \propto 1/P$ |
| p (proton donmuş-spin) | 700.7 MeV/c | halka tasarımı |
| Hedef d_p | 10⁻²⁹ e·cm | → dS_y/dt ~ 10⁻⁹ rad/s |
| Depolama süresi | ≥ 1000 s | dolum başına |
| Toplam olay | > 10¹² | bir yıllık kampanya |
| µrad çözünürlük | ~2 µrad | tek-kampanya açı |
| dS_y/dt tabanı | ~7×10⁻⁹ rad/s | **sahte-EDM bastırma hedefimizin kaynağı** |
| Cross-ratio sistematik | < 10⁻⁵ sinyal | ölçüm-tarafı taban |

---

*Bu özet yalnızca polarimetre performansı ve EDM araması zaman/istatistik
bütçesine odaklanır. Dedektör inşası, SiPM okuma, kalibrasyon spektrumları ve
DAQ ayrıntıları için orijinal makaleye (arXiv:2010.13536) bakınız.*
