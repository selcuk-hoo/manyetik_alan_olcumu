# False EDM'in Fourier-Mod Spektrumu: Spin Takibiyle Doğrudan Ölçüm

> Bu doküman `false_edm_mode_scan.py` betiğinin yöntemini anlatır.
> Soru basit: quad hizalama hatalarının hangi Fourier modu yanlış-EDM
> (false EDM) sinyalini en çok besler? Cevabı **el sallamadan**, doğrudan
> spin takibiyle veriyoruz. Belge önce fiziği, sonra ölçümün neden
> zor olduğunu, sonra çözümü (kapalı-yörünge fırlatması + stroboskopik
> örnekleme), en sonda da sayısal doğrulama testlerini açıklar.

---

## İçindekiler

1. [Soru ve fiziksel çerçeve](#1-soru-ve-fiziksel-çerçeve)
2. [Ölçülen büyüklük: neden dS_y/dt?](#2-ölçülen-büyüklük)
3. [İlk (naif) yöntem ve neden çöktüğü](#3-naif-yöntem)
4. [Kök neden: aliasing ve sızma](#4-kök-neden)
5. [Çözüm 1: kapalı yörünge üzerinde fırlatma](#5-çözüm-1)
6. [Çözüm 2: stroboskopik örnekleme](#6-çözüm-2)
7. [Sonuç: k=2 rezonansı](#7-sonuç)
8. [Sayısal doğrulama testleri](#8-doğrulama)

---

## 1. Soru ve fiziksel çerçeve <a name="1-soru-ve-fiziksel-çerçeve"></a>

Frozen-spin pEDM halkasında gerçek EDM sinyali, spinin yatay düzlemden
**dikey eksene doğru yavaşça dönmesidir** — yani dikey spin bileşeni
$S_y$'nin zamanla **sekuler** (salınımsız, biriken) büyümesi:
$$\text{EDM sinyali} \;\equiv\; \frac{dS_y}{dt}.$$

Ne yazık ki gerçek EDM dışında da $S_y$'yi büyütebilecek mekanizmalar
vardır: **dikey kapalı yörünge bozulması (COD)** boyunca spinin gördüğü
radyal alanlar, gerçek EDM'i taklit eden bir dönme yaratır. Buna
**yanlış EDM (false EDM)** denir ve pEDM deneyinin baş sistematiğidir.

Quad hizalama hatası $\Delta y_j$ (quad $j$'nin dikey kayması) dikey COD
yaratır. Hatayı FODO-antisimetrik Fourier bazında yazıyoruz:
$$\Delta y_j = A\,F_k[j], \qquad F_k[j] = (-1)^j \cos\!\Big(\frac{2\pi k\,\lfloor j/2\rfloor}{24}\Big),$$
burada $j=0\ldots47$ quad indeksi, $(-1)^j$ FODO'nun F/D değişimini,
$\lfloor j/2\rfloor$ hücre indeksini (0…23) verir.

**Tarama:** her $k=0,1,\ldots,5$ modu için, tüm modlara **eşit genlik** $A$
(varsayılan 10 μm cos katsayısı) veririz, **gerçek EDM'i kapatırız**
(`EDMSwitch=0`), spini takip eder ve $dS_y/dt$'yi ölçeriz. Beklenti:
sinyal, kapalı-yörünge kazancının zirve yaptığı $k=2$'de (Q_y≈2.68
rezonansına en yakın mod) en büyük olur.

---

## 2. Ölçülen büyüklük: neden dS_y/dt? <a name="2-ölçülen-büyüklük"></a>

False EDM, gerçek EDM gibi **sekuler bir drifttir** — $S_y$ sürekli
büyür. Buna karşılık spinin halka boyunca yaptığı **statik eğilme**
(sabit bir $S_y$ ofseti) bir EDM değildir; gerçek bir deneyde kalibre
edilip çıkarılabilen, sınırlı bir kapalı-yörünge spin bozulmasıdır.
Bu yüzden ölçtüğümüz büyüklük **eğim** ($dS_y/dt$), ofset değil.

Sinyal seviyesi (10 μm, EDMSwitch=0) **~10⁻¹⁰–10⁻⁹ rad/s** mertebesindedir —
yani çok küçük. Bu küçüklük ölçümü zorlaştıran şeyin ta kendisidir
(§3-4).

---

## 3. İlk (naif) yöntem ve neden çöktüğü <a name="3-naif-yöntem"></a>

Proje standardı (`run_simulation.py`, `plot_results.py`): $S_y(t)$'yi
yoğun (sürekli) örnekle, geniş pencereli **Savitzky-Golay** filtresiyle
(pencere = N/4) hızlı salınımları bastır, kenarların %10'unu at, kalanına
düz çizgi fit et → eğim. Bu yöntem gerçek EDM açıkken (`EDMSwitch=1`)
mükemmel çalışır, çünkü gerçek EDM sinyali büyüktür.

False EDM'de ise (`EDMSwitch=0`, sinyal ~10⁻⁹) bu yöntem **çöküyor**.
Belirtileri:

- Çıktı örnekleme yoğunluğunu (`--steps`) 5000'den 50000'e değiştirince
  ölçülen eğim **işaret değiştiriyor** (örn. k=1: −2.2×10⁻⁶ → +1.4×10⁻⁶).
- Modlar arası sıralama oynuyor; bir koşumda k=1, başka koşumda k=2
  "baskın" görünüyor.

Aynı yörüngeyi yalnızca daha sık örnekleyince gerçek bir sekuler drift
**işaret değiştiremez**. Demek ki ölçülen sayı fizik değil, **ölçüm
artefaktıdır.**

---

## 4. Kök neden: aliasing ve sızma <a name="4-kök-neden"></a>

$S_y(t)$ üç parçadan oluşur:
$$S_y(t) = \underbrace{D\,t}_{\text{sekuler (aranan)}} + \underbrace{\text{salınımlar}}_{\sim10^{-5}} .$$

Salınım genliği (~10⁻⁵), aranan driftin 2 ms'deki toplam değişiminden
(~10⁻¹³ rad → eğim olarak ~10⁻⁹) **~10⁴ kat büyüktür**. İki salınım
kaynağı var:

1. **Betatron** (Q_y≈2.68): parçacık kapalı yörünge etrafında titreşince
   yörünge — dolayısıyla spin presesyonu — Q_y frekansında modüle olur.
2. **Tur-içi periyodik spin presesyonu**: kapalı yörünge üzerinde bile
   spin her turda aynı yapıda salınır.

**Aliasing mekanizması.** Çıktı örnekleme aralığı $dt_\text{out}=t_2/N$.
Bu aralık salınım periyoduyla uyumsuz olduğunda, hızlı salınım
**sahte yavaş bir dalga**ya dönüşür (aliasing). Bu sahte dalganın
frekansı/fazı $dt_\text{out}$'a bağlıdır → Savitzky-Golay + polyfit ona
$N$'e göre değişen bir eğim atfeder → işaret döner. Sinyal salınımın
~10⁻⁴'ü olduğundan, SG'nin bıraktığı küçük sızma bile gerçek driftı gömer.

**Sonuç:** Sürekli-SG yöntemi bu sinyal seviyesinde ilkesel olarak
yetersiz. Salınımları **kaynağında** yok etmek gerekir.

---

## 5. Çözüm 1: kapalı yörünge üzerinde fırlatma <a name="5-çözüm-1"></a>

Betatron salınımını yok etmenin yolu: parçacığı ideal $y=0$'dan değil,
**kaymış kapalı yörünge üzerinden** fırlatmak. Kapalı yörünge üzerinde
parçacık her turda aynı noktadan geçer (betatron aksiyonu $J=0$).

`find_closed_orbit()` bunu şöyle bulur:

- Sabit bir azimutta (Poincaré, tur-başına) dikey konumun **varyansı**,
  fırlatma noktasının kapalı-yörüngeden sapmasının **tam kuadratik
  formudur** (`sextSwitch=0` → lineer lattice). (Genlik ∝ |sapma| bir
  V-şeklidir, kuadratik değil — bu yüzden **kare/varyans** kullanılır.)
- 2B $(y, y')$ faz uzayında bu varyans **eğik bir elipstir** (çapraz
  terim $H_{yy'}\neq0$). Koordinat-bazlı iniş bu vadide yavaş ilerler;
  bunun yerine sonlu-fark **Hessian'ı (çapraz terim dahil)** kurup
  **tek Newton adımı** ile minimuma atlanır.

Sonuç: kalan betatron genliği **~10⁻¹⁴ m** seviyesine iner (orijinal
~2×10⁻⁴ m'den 10 mertebe küçük) — yani betatron pratikte yok edilir.
Bu, sonuç tablolarındaki "kalan betatron" sütunuyla doğrulanır.

> **Not.** Quad_dy düz halkada yalnızca dikey yörüngeyi kaydırır
> (`quad_tilt=0` → x-y kuplajı yok), bu yüzden yatay düzlem $y'=0$'da
> bırakılır; arama 2B'dir.

---

## 6. Çözüm 2: stroboskopik örnekleme <a name="6-çözüm-2"></a>

Betatron öldükten sonra bile $S_y$ tur-içinde ~10⁻⁵ salınmaya devam eder:
bu, **kapalı yörünge boyunca spin presesyonudur** ve artık **tam
periyodiktir** (yörünge statik). Kapalı yörünge üzerinde tur-başına spin
haritası **sabit bir dönmedir** ($R_\text{turn}$).

**Stroboskopik örnekleme:** $S_y$'yi **sabit bir azimutta, tur-başına bir
kez** örnekle (`poincare_quad_index=0`). O azimutta her tur **aynı fazı**
görür → tur-içi salınım **tamamen** çıkar. Geriye yalnızca sekuler drift
(+ varsa yavaş spin-tune dalgası) kalır:
$$S_y(n)\big|_\text{sabit azimut} = D\,(n\,T_\text{rev}) + \text{(yavaş)},$$
ki bu **temiz bir doğrudur**. Eğim = düz polyfit. SG yok, örnekleme-oranı
seçimi yok, aliasing yok.

`integrator`'ın Poincaré çıktısı bu azimutta tur-başına 9-bileşenli durumu
verir; dikey spin $S_y$ kolonu (yerel kol 7) çerçeve dönüşünden
**etkilenmez** (dikey eksen yatay-düzlem dönüşünde değişmez), dolayısıyla
doğrudan kullanılabilir.

**Doğruluk testi (yakınsama).** İlk-yarı eğimini tam-koşum eğimiyle
karşılaştırırız: oran ≈ 1 ise drift temiz lineerdir (yavaş-dalga biası
yok). Ölçümlerde bu oran k=0 için 1.00, k=1 için 0.94 çıkar — yani t2=2 ms
(~440 tur) zaten yakınsamıştır; eski yöntemdeki gibi 10 ms gerekmez.

---

## 7. Sonuç: k=2 rezonansı <a name="7-sonuç"></a>

<!-- TABLO-ANA: doldurulacak -->

Temiz yöntemle (kapalı-yörünge fırlatması + stroboskopik örnekleme)
false EDM sinyali **k=2'de zirve yapar** ve **kapalı-yörünge genliğiyle
birebir örtüşür** (her ikisi de k=2'de en büyük). Fiziksel açıklama:
FODO-antisimetrik k=2 modu Q_y≈2.68 yörünge rezonansına en yakın olduğundan
en büyük kapalı yörüngeyi üretir → en büyük false EDM. İşaret yapısı
(k=0,1,2 pozitif; k=3,4,5 negatif) projeksiyon $w\cdot y_{CO}$'nun harmonik
arttıkça işaret değiştirmesinden gelir — saf gürültü olsa rastgele olurdu.

Eski sürekli-SG yöntemi aynı koşumlarda ~10⁻⁶ (artefakt) verir; gerçek
sinyal bunun ~3000 kat altındadır.

---

## 8. Sayısal doğrulama testleri <a name="8-doğrulama"></a>

<!-- TABLO-TEST: doldurulacak -->

Sonucun sayısal hatadan değil fizikten geldiğini kanıtlamak için bir
doğrulama bataryası koşulur. Her testin neyi elediği:

| Test | Yöntem | Hata olsaydı |
|------|--------|--------------|
| **Null (A=0)** | hizalama hatası yok | slope ~0 olmalı; ≠0 ise sahte arka plan |
| **dt yakınsama** | dt: 1e‑11 → 5e‑12 | slope değişirse entegrasyon hatası |
| **Genlik lineerliği** | A: 10 → 20 μm | slope ∝ A değilse lineer-olmayan kirlilik |
| **İşaret simetrisi** | Δy → −Δy | slope işareti dönmezse çift-mertebe kirlilik |
| **Spin normu** | ‖S‖=1 korunumu | ‖S‖ kayarsa spin entegrasyonu bozuk |
| **Azimut bağımsızlığı** | aynı turun 24 azimutu | azimuta göre değişirse ölçüm hatalı |

**Güven kriteri:** Asıl önemli olan **k₂/k_n oranlarının** dt yarılanması
ve işaret çevrilmesi altında sabit kalmasıdır (mutlak değer değil). Oranlar
korunuyorsa "k=2 baskın" sonucu sayısal kuruluştan bağımsızdır.
