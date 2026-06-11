# Harmonik Trim Yöntemi — Pedagojik Anlatım

Bu belge, sahte EDM sinyalini "ölç–trimle" döngüsüyle bastırma
yöntemini ve bu oturumda yapılan iki kritik testi — **fırlatma koşulu
bağımlılığı** ile **rastgele desen + fazlı trim** — hiçbir ön bilgi
varsaymadan, adım adım anlatır. Teknik rapor için
`false_edm_harmonic_sinir.md` §12'ye, makale diline dökülmüş haline
`makale_tr.tex`'e bakınız.

---

## 1. Problem: Sahte EDM nedir, neden çıkar?

Proton EDM deneyi, protonun spininin düşey yönde çok yavaş dönmesini
($dS_y/dt$) arar. Gerçek bir EDM bu dönmeyi yaratır — ama ne yazık ki
**mıknatıs hizalama hataları da aynı şeyi yapar**.

Zincir şöyle işler:

1. Halkadaki 48 kuadrupol mıknatıstan bazıları ideal konumundan
   birkaç mikron **düşey kaymış** olsun (Δy).
2. Kaymış kuadrupol, içinden geçen demete küçük bir düşey itme verir
   → demetin kapalı yörüngesi düşeyde **bozulur** (artık tam düzlemde
   değildir).
3. Bozulmuş yörünge üzerinde proton, kuadrupollerin **radyal manyetik
   alanını** görür (ideal yörüngede görmezdi).
4. Radyal manyetik alan, Thomas-BMT denklemi gereği spini düşey
   eksende döndürür → $dS_y/dt \neq 0$.

Sonuç: hiç EDM olmasa bile spin düşeyde döner. Buna **sahte EDM**
(false EDM) diyoruz. Deneyin en sinsi sistematiği budur, çünkü gerçek
sinyalle **aynı imzayı** taşır.

Sayısal ölçek: 10 μm'lik tipik hizalama hatası ~10⁻⁴ rad/s sahte
sinyal üretir; hedeflenen EDM duyarlılığı bunun milyonlarca kat
altındadır. Yani sahte sinyali en az ~10⁶–10⁷ kat bastırmak gerekir.

---

## 2. Anahtar fikir: Hata desenini Fourier modlarına ayır

48 kuadrupolun her birinin kayması ayrı bir sayı: 48 bilinmeyen.
Bunların hepsiyle tek tek uğraşmak yerine, kayma *desenini* halka
çevresince **Fourier modlarına** ayırırız:

```
Δy_j = Σ_k A_k · cos(2πk·n_j/N − φ_k)        (N = 24 FODO hücresi)
```

Her mod k, halka çevresinde k kez dalgalanan bir kayma desenidir:
- k=1: halkanın bir yanı yukarı, öbür yanı aşağı (tek dalga)
- k=2: iki tepe, iki çukur
- k=3: üç tepe, üç çukur... vb.

**Neden bu ayrıştırma işe yarar?** Çünkü sahte EDM her moda eşit tepki
vermez. Demetin düşey betatron ayar frekansı Q_y ≈ 2.68'dir; **k=2
modu bu rezonansa en yakın olduğundan yörüngeyi en çok bozar** ve sahte
EDM'nin aslan payını üretir. Yüksek modlar (k≥5) neredeyse zararsızdır.

Doğrusal bir kafeste her modun katkısı toplanır:

```
dS_y/dt = Σ_k  c_k · A_k        (faz şimdilik göz ardı — §6'da gelecek)
```

Buradaki **c_k katsayısı, k modunun "kaldıraç kolu"dur**: 1 metre mod
genliği başına kaç rad/s sahte sinyal ürettiğini söyler. Ölçtüğümüz
değerler (cos fazı, CO=False):

| k | c_k [rad/s/m] | yorum |
|---|---|---|
| 1 | +23.5 | pozitif (k < Q_y) |
| **2** | **+88.8** | **baskın — rezonansa en yakın** |
| 3 | −22.5 | negatif (k > Q_y; rezonansın öbür yakası) |
| 4 | −7.7 | hızla küçülür (~1/k²) |
| 5 | −3.9 | |
| 6 | −2.3 | |

İşaret kuralına dikkat: k=2 ile k=3 **zıt işaretli**. Bu, birinin
ürettiği sahte sinyali diğeriyle **iptal edebileceğimiz** anlamına
gelir — trim yönteminin temeli budur.

---

## 3. Ölç–trimle döngüsü: yöntem üç cümlede

1. **Ölç:** Spin polarimetresiyle $f = dS_y/dt$'yi ölç (BPM'e gerek yok!).
2. **Trimle:** Kuadrupolleri bilinçli olarak küçük bir Fourier modu
   deseninde kaydır: genlik $a = -f / c_\mathrm{trim}$. Bu, ölçülen
   sinyali tam iptal edecek "karşı-kirlilik"tir.
3. **Tekrarla:** Kalan sinyali ölç, gerekirse küçük bir düzeltme daha uygula.

Önemli incelik: hangi kuadrupolun ne kadar kaydığını **bilmemize gerek
yok**. Tek bilmemiz gereken kalibre edilmiş c_k katsayısı ve ölçülen
tek bir sayı (f). Desen ne olursa olsun, f skalerini sıfırlamak sahte
EDM'yi sıfırlamaktır.

Bir benzetme: terazinin bir kefesinde bilinmeyen ağırlıklar var
(hizalama hataları). Hangi ağırlıkların orada olduğunu bilmiyoruz ama
ibrenin ne kadar saptığını görüyoruz. Öbür kefeye ibreyi sıfırlayacak
kadar bilinen ağırlık (trim) koyarız. İçerik bilgisi gerekmez; sadece
ibre okuması yeter.

Tüm bu testlerde **CO=False** koşulu kullanılır: parçacık ideal
eksenden fırlatılır, kapalı yörünge hiç aranmaz (kick ile yörüngeye
oturtmanın pratik olmadığına daha önce karar verilmişti —
`injection_kick_raporu.md`). Betatron salınımı ölçümün doğal
parçasıdır ve — kritik bulgu — bu koşulda sistem **tam doğrusaldır**:
f/A oranı genlikten bağımsız, sapma %0.0.

---

## 4. Test I: Trim eksende kalibre ediliyor — ya demetin geri kalanı?

**Dosya:** `test_b_trim_launch_dep.py`

### 4.1 Soru neden önemli?

c_k katsayılarını tek bir parçacıkla, onu tam eksenden (y=0, açı=0)
fırlatarak kalibre ediyoruz. Ama gerçek demet binlerce parçacıktan
oluşur ve her biri **farklı noktadan, farklı açıyla** yola çıkar.
Akla şu kuşku gelir:

> "Eksendeki parçacık için sinyali sıfırlayan trim, kenardaki
> parçacıklar için sıfırlamıyorsa, demet *ortalamasında* artık
> sinyal kalır ve deney bunu gerçek EDM sanır."

Bu kuşkuyu sayısal olarak sınamak gerekir.

### 4.2 Nasıl test ettik?

Üç bölümlü tarama (hepsi k=2 modu, A=10μm, t₂=1ms):

- **Konum taraması:** Parçacığı eksenden 0, 0.1, 0.2, 0.5, 1, 2 mm
  kaydırıp fırlat; her seferinde c_k'yi yeniden ölç.
- **Açı taraması:** Konum sıfır, ama fırlatma açısı α = 0 … 1 mrad.
- **Demet benzetimi:** 30 parçacığı gerçekçi bir demet dağılımından
  çek (σ_y=0.5mm, σ_α=0.2mrad), her birinin gördüğü sinyali ölç,
  eksen kalibrasyonuyla karşılaştır.

### 4.3 Ne bulduk?

**Konum etkisi minicik.** 2 mm kaymada bile c_k yalnız %1.3 değişiyor;
gerçekçi 0.5 mm'de etki %0.3. Önemsiz.

**Açı etkisi görünürde büyük: %3 / 0.1 mrad.** 1 mrad'da %19'a çıkıyor.
İlk bakışta korkutucu — ama bu **fiziksel bir c_k değişimi değil,
ölçüm yapaylığı**. Nedeni şu: büyük açıyla fırlatılan parçacık büyük
betatron salınımı yapar; bu salınım spin izine de süzülür ve sonlu
ölçüm penceresinde (1 ms) doğrusal eğim fitini hafifçe biyaslar.
Salınımın *gerçek* ortalaması sıfırdır; sadece fit penceresinin
kenar etkisi görünür. Ölçüm süresi uzadıkça bu yapaylık 1/t₂ ile erir.

**Demet ortalaması eksene çok yakın: fark yalnız %0.47.** 30 parçacığın
ortalama sinyali, eksen parçacığının sinyalinden binde beş kadar düşük.
Yani eksen kalibrasyonu demeti %0.5 doğrulukla temsil ediyor.

### 4.4 Asıl güvence: trim bir *kafes* işlemidir

En önemli kavramsal nokta şu: trim, parçacığa değil **kafese**
uygulanır. Trim kuadrupolleri fiziksel olarak kaydırıp k=2 kirliliğini
söker (a_k → 0). Kirlilik bir kez söküldü mü, **hangi koşulla
fırlatılırsa fırlatılsın hiçbir parçacık** seküler spin sürüklenmesi
yaşamaz — çünkü sürüklenmeyi yaratan şey ortadan kalkmıştır.

Demetteki parçacıkların "farklı c_k görmesi" sadece ölçüm
yapaylığıdır; trimin etkisi ise fizikseldir ve evrenseldir. Testteki
muhafazakâr alt sınır bile (eksen sinyalini doğrusal çıkarma modeli)
213× bastırma verir; gerçek bastırma çok daha derindir.

**Test I'in dersi:** Eksen kalibrasyonu güvenlidir. Demet sıcaklığı
trimi bozmaz.

---

## 5. Faz problemi: rastgele desende yeni bir boyut

Şimdiye dek testler hep "cos fazlı" yapay desenler kullandı:
mod tam `cos(2πkn/N)` şeklindeydi. Ama gerçek hizalama hatası
rastgeledir ve her modun bir de **fazı** vardır:

```
A_k · cos(2πk·n/N − φ_k)
```

φ_k, dalganın halka çevresindeki "kayma açısıdır": aynı genlikteki k=2
modu, tepeleri başka kuadrupollere denk gelecek şekilde döndürülmüş
olabilir. Trigonometri bunu iki bileşene ayırır:

```
A·cos(θ − φ) = (A·cosφ)·cos θ + (A·sinφ)·sin θ
                └── cos kuadratürü ──┘ └── sin kuadratürü ──┘
```

Yani her modun aslında **iki bağımsız düğmesi** var: cos bileşeni ve
sin bileşeni. Şimdiye dek yalnız cos düğmesini kalibre etmiştik
(c_k^cos). Sin düğmesinin kaldıraç kolu (c_k^sin) farklı olabilir!

### 5.1 Etkin faz ve ölü faz

Doğrusallık gereği, φ fazında uygulanan trimin etkinliği:

```
c_k(φ) = c_k^cos·cosφ + c_k^sin·sinφ = |c_k| · cos(φ − ψ_k)
```

Bu bir sinüzoittir. İki özel açı vardır:

- **ψ_k (etkin faz):** trimin birim genlik başına EN GÜÇLÜ olduğu faz.
  `ψ_k = atan2(c_k^sin, c_k^cos)`.
- **ψ_k ± 90° (ölü faz):** trimin HİÇ etki etmediği faz. Bu fazda ne
  kadar kaydırırsanız kaydırın sinyal kıpırdamaz!

Benzetme: sıkışmış bir kapıyı itiyorsunuz. Menteşe ekseni ψ_k+90°
yönü gibidir — o yönde ittirmek kapıyı hiç döndürmez. Kapı yüzeyine
dik itmek (ψ_k yönü) en az kuvvetle en çok döndürür. Ara açılarda
da kapı döner ama daha çok kuvvet gerekir: kuvvet ihtiyacı
1/cos(φ−ψ_k) ile büyür.

Pratik sonuç: **mod başına iki kalibrasyon ölçümü** (φ=0 ve φ=90°)
yapılırsa hem |c_k| hem ψ_k bilinir ve ölü faza düşme riski kalmaz.

---

## 6. Test II: Rastgele desen + fazlı çok-modlu trim

**Dosya:** `test_b_random_trim.py`

### 6.1 Kurulum

Gerçek deneye en yakın senaryo: 48 kuadrupolun HER BİRİNE bağımsız
rastgele kayma verildi (Gauss, RMS=10μm, seed=123). Bu desen Fourier'e
ayrıştırılınca tüm modlar dolu çıkar (A_k ≈ 1–4 μm) ve fazlar
rastgeledir — örneğin baskın k=2 içeriği A₂=2.82μm, φ₂=170.5°.

Test dört bölümden oluşur.

### 6.2 Bölüm 1 — Çift kuadratür kalibrasyon

k=1..6 için hem cos hem sin fazında ölçüm (12 simülasyon):

| k | c_k^cos | c_k^sin | \|c_k\| | ψ_k |
|---|---|---|---|---|
| 1 | +23.5 | −2.4 | 23.6 | −5.9° |
| 2 | +88.8 | −18.4 | **90.7** | **−11.7°** |
| 3 | −22.5 | +7.1 | 23.6 | 162.6° |
| 4 | −7.7 | +3.3 | 8.4 | 157.1° |
| 5 | −3.9 | +2.1 | 4.5 | 151.9° |
| 6 | −2.3 | +1.5 | 2.8 | 147.2° |

**Sürpriz keşif — faz rampası:** Etkin fazlar rastgele değil; düzenli
bir merdiven izliyor: ψ_k ≈ −5.85° × k (180° sıçramaları, k>Q_y işaret
değişiminin faz dilindeki ifadesidir: −22.5 = +22.5 @ 180°). Mod
başına sabit faz eğimi, klasik sinyal işlemedeki "zaman gecikmesi =
frekansla doğrusal faz" kuralının halka karşılığıdır: gözlenebilir,
halka üzerinde belli bir azimut noktasına (fırlatma/gözlem civarı)
referanslıdır. Bu güzel bir iç tutarlılık kontrolüdür — altı bağımsız
ölçüm tek bir geometrik parametreyle açıklanıyor.

### 6.3 Bölüm 2 — Faz modeli gerçekten sinüzoit mi?

Model φ=0 ve 90'daki ölçümlerle kuruldu; sonra **bağımsız** iki ara
fazda (45° ve 135°) test edildi:

| φ | ölçülen | model | sapma |
|---|---|---|---|
| 45° | 49.795 | 49.795 | %0.000 |
| 135° | −75.793 | −75.793 | %0.000 |

Sapma sıfır — sistem faz uzayında da kusursuz doğrusal. İki ölçüm,
modun tüm faz davranışını eksiksiz belirliyor.

### 6.4 Bölüm 3 — Desenin sinyalini önceden tahmin edebiliyor muyuz?

Rastgele desenin ölçülen sinyali: **f₀ = −2.503×10⁻⁴ rad/s**.

Desenin spektrumunu (A_k, φ_k) kalibre katsayılarla çarpıp toplayınca
(yalnız k=1..6): tahmin −2.694×10⁻⁴ → **%7.6 doğru**. Kalan fark
kalibre edilmeyen k≥7 modlarından. Katkıların dökümü öğreticidir:

- k=2: −2.56×10⁻⁴ → **tek başına f₀'ın ~%102'si!**
- diğer beş mod toplamı: küçük ve kısmen birbirini iptal ediyor

Yani rastgele bir desende bile hikâye k=2'nin hikâyesidir — kuram
bölümündeki rezonans argümanının doğrudan sayısal teyidi.

### 6.5 Bölüm 4 — Üç trim stratejisi yarışıyor

Aynı desen üzerinde üç farklı reçeteyle ölç–trimle döngüsü (3'er adım):

**Strateji A — "tek mod, doğru faz":** k=2 trimi, etkin fazı ψ₂=−11.7°'de.
**Strateji B — "iki mod, doğru fazlar":** k=2 ve k=3, her biri kendi
ψ'sinde, genlik eşit bölüşmüş (toplam kaldıraç |c₂|+|c₃| olduğundan
mod başına daha az kayma gerekir).
**Strateji C — "faz cahili":** k=2 trimi ama yalnız cos fazında —
sanki çift kuadratür kalibrasyonu hiç yapılmamış gibi.

Sonuçlar:

| | Adım 1 bastırma | Nihai taban | Toplam | Trim bütçesi |
|---|---|---|---|---|
| A | 3.2×10⁷× | ~10⁻¹² rad/s | 2.1×10⁸× | 2.76 μm |
| **B** | **7.8×10⁷×** | **~10⁻¹⁵ rad/s** | **1.0×10¹¹×** | 4.38 μm |
| C | 1.6×10⁷× | ~4×10⁻¹⁵ rad/s | 6.5×10¹⁰× | 2.82 μm |

Okunması gereken üç mesaj:

1. **Tek adım yetiyor.** Üç strateji de ilk trimde sinyali on milyon
   kat bastırdı. Doğrusal sistemde ölçülen skalerin iptali kesindir;
   ikinci adım sadece sayısal kırıntıları süpürür.

2. **Faz cahili olmak ölümcül değil.** Strateji C de mükemmel çalıştı.
   Bedeli ne? Bütçesi A'dan %2 fazla (2.82 vs 2.76 μm) — çünkü
   cos fazı, etkin faza yalnız 11.7° uzak: cos(11.7°)=0.979. Faz
   ancak ölü faza (ψ₂±90°) yaklaşırsa felaket olur; o zaman trim
   etkisiz kalır ve bütçe ıraksar. Çift kuadratür kalibrasyon bu
   riski tamamen ortadan kaldırır.

3. **Doğru fazlı trim, kirliliğin kendisini söküyor.** Strateji A'nın
   uyguladığı trim: 2.76μm @ −11.7°. Desenin gerçek k=2 içeriği:
   2.82μm @ 170.5°. Bu ikisi neredeyse tam **anti-paralel**
   (170.5° ≈ −11.7° + 182°). Yani skaleri sıfırlamak için hesaplanan
   trim, farkında olmadan desendeki fiziksel k=2 kirliliğinin hemen
   hemen tam negatifini üretmiş — sinyali maskelememiş, kaynağı
   temizlemiş.

---

## 7. Test III: Yörünge-sürülü trim — spin ölçümü olmadan, BPM'den

*Dosya: `test_orbit_trim.py`, sonuçlar: `test_orbit_trim.json`*

### 7.1 Yeni problem: spin kullanmak neden riskli?

Şimdiye kadar trim için spini rehber olarak kullandık: f₀ ölç, hedef modu
trimle, f sonrasını ölç. Bu işe yarar — ama bir güvenlik açığı taşır.

Spin, hem hizalama kirliliğinden hem de gerçek EDM'den etkilenir:

    dS_y/dt = (sahte EDM'den gelen katkı) + (gerçek EDM'den gelen katkı)

Trim f₀'ı sıfırlamayı hedeflediğinde, **gerçek EDM sinyalini de yutabilir**.
Bunu önlemek için CW/CCW ayrıştırması gerekir — ama bu, deneyin en hassas
adımıdır.

Alternatif: kapalı yörüngeyi rehber al. Gerçek bir EDM,
kapalı yörüngeyi hiç bozmuyor. Dolayısıyla BPM okumaları **tamamen
EDM'den bağımsız**; yanlışlıkla sinyali yutmak mümkün değil.

### 7.2 Senaryo: gerçekçi ve zor

- 48 quad rastgele dikey kaçıklık, RMS = **100 μm** (seed=321)
- 48 BPM statik ofset, RMS = **100 μm** (seed=777) — sinyal ve gürültü
  eşit büyüklükte
- k-modülasyon yok; tek optik konfigürasyon
- Başlangıç spin hızı: **f₀ = −1.623×10⁻³ rad/s**

Desenin Fourier spektrumu (her mod genliği, μm):

    k=1: 37.6μm  k=2: 16.2μm  k=3: 40.9μm  k=4: 45.0μm
    k=5: 17.0μm  k=6:  6.1μm  k=7: 26.8μm  k=8: 31.9μm

### 7.3 Yöntem: kalibrasyon → kestirim → trim

**Adım 1 — Kalibrasyon:** Her k=1..6 modu için cos ve sin desen
(12 düğme toplamda), 50 μm genliğinde uygulanır. Her seferinde 48
BPM'den tur-ortalamalı kapalı yörünge okunur. Bu 12 ölçüm, tepki matrisi
O [48×12]'yi oluşturur. Kalibrasyon **diferansiyel** olduğundan statik
BPM ofseti sıfırlanır — sadece uygulanan modun yarattığı fark görülür.

**Adım 2 — Kestirim:** Gerçek desenin BPM okuması:

    y_ölç = y_gerçek + b   (b: statik ofset, bilinmiyor)

Least-squares çözümü O·â ≈ y_ölç verir → mod kestirimleri â. Ama
b bilinmediği için â içinde bir sistematik yanlılık bulunur. Bu yanlılık,
orbitsal kazancı küçük olan modlarda büyür (aşağıya bakınız).

**Adım 3 — Trim:** Quad kaçıklık dizisine Δp = −Σₖ âₖ·mod_k eklenir.
Doğrulama: spin takibi ile f ölçülür.

### 7.4 Kazanç hiyerarşisi: yörünge hangi modları görür?

Mod-k kaçıklığı, halka boyunca transfer matrisinin rezonans faktörüyle
büyütülür. Bu faktör, betatron tununa (Q_y ≈ 2.68) yakınlıkla belirlenir.
k=2, Q_y'ye en yakın mode → 24.1× büyütme. k değeri Q_y'den uzaklaştıkça
kazanç düşer:

| k | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Yörünge kazancı | 6.2 | **24.1** | 6.3 | 2.26 | 1.24 | 0.79 |

100 μm BPM ofsetiyle kestirim yanlılığı ≈ σ_b·√(2/N)/kazanç ≈ 20.4μm/kazanç
(N=48 BPM; √(2/N) çarpanı, 48 rastgele ofsetin tek bir mod şekline
ortalama izdüşümünden gelir):

| k | Yanlılık (formül) | Ölçülen (O⁺b) | Yorumu |
|---|---|---|---|
| k=2 | ~0.9 μm | 0.1–0.5 μm | Güvenli; gerçek genlik ≫ yanlılık |
| k=4 | ~9 μm | 0.2–1.0 μm | Kullanılabilir; gerçek A₄=45 μm ≫ yanlılık |
| k=5 | ~16 μm | 6.7–7.6 μm | Sınırda; gerçek A₅=17 μm ile aynı mertebede |
| k=6 | ~26 μm | ~28 μm | Zararlı; gerçek A₆=6 μm'nin 4–6 katı |

Kıyas ölçütü desenin kendi mod içeriğidir: 100 μm RMS rastgele desende
kuadratür başına beklenen genlik ~σ_q·√(2/48) ≈ 20 μm. Yanlılık bu
değere yaklaştığında fit etmek zarar vermeye başlar. Kazanç hiyerarşisi,
hangi modları trimleyebileceğimizi doğrudan söyler.

### 7.5 Dört deneme: adım adım spin değerleri

Kaç modu fit edeceğimizi değiştirip f sonrasını ölçüyoruz:

**Başlangıç:** f₀ = −1.623×10⁻³ rad/s.

---

**Varyant A (k=1..3 fit edildi):**

Trim k=1, 2, 3 içeriklerini azaltır:

| k | Önce | Sonra |
|---|---|---|
| 1 | 37.6 μm | 3.8 μm |
| 2 | 16.2 μm | 7.0 μm |
| 3 | 40.9 μm | 9.2 μm |
| 4 | 45.0 μm | **45.0 μm (dokunulmadı!)** |

k=4, fit kapsamına girmediği için trim onu tamamen pas geçer.
k=4 = 45 μm olarak kalır, c₄ × A₄ ~ birkaç×10⁻⁵ katkı yapar.

**f sonrası = −6.72×10⁻⁵ rad/s.  Bastırma: 24.2×.**

---

**Varyant C (k=1..4 fit edildi) — en iyi:**

A'ya ek olarak k=4 de hedeflenir. k=4 ofset yanlılığı ~1 μm (kazanç 2.26
hâlâ ofseti bastırmaya yetiyor), gerçek A₄ = 45 μm — temiz kestirim.

| k | Önce | Sonra |
|---|---|---|
| 1 | 37.6 μm | 3.8 μm |
| 2 | 16.2 μm | 6.9 μm |
| 3 | 40.9 μm | 9.2 μm |
| 4 | 45.0 μm | **6.0 μm** ← k=4 nihayet azaldı |
| 5 | 17.0 μm | 17.0 μm (dokunulmadı) |
| 6 | 6.1 μm | 6.1 μm (dokunulmadı) |
| 7 | 26.8 μm | 26.8 μm (dokunulmadı) |
| 8 | 31.9 μm | 31.9 μm (dokunulmadı) |

**f sonrası = +1.61×10⁻⁵ rad/s.  Bastırma: 100.8×.**

İşaret bile tersine döndü (−1.623×10⁻³ → +1.61×10⁻⁵): k=4 katkısı
bastırıldı, kalan artık k=5 ve üzerinden geliyor.

---

**Varyant D (k=1..5 fit edildi):**

k=5 ofset yanlılığı ~7–12 μm (ölçülen), gerçek A₅ = 17 μm — kestirimin
yarısı ofset hayaleti. Trim k=5'i kısmen **yanlış yönde** oynatır ve
c₅ = 4.5 rad/s/m bu hatayı doğrudan spin sinyaline taşır.

**f sonrası = +1.37×10⁻⁴ rad/s.  Bastırma: 11.9× — C'den 8.5× KÖTÜ.**

---

**Varyant B (k=1..6 fit edildi):**

k=6 kazancı = 0.79 < 1. Ölçülen yanlılık: cos −24.7 μm, sin −26.9 μm
(bileşke 36.5 μm) — gerçek A₆ = 6.1 μm'nin 6 katı. Trim k=6 içeriğini
**6.1 μm'den 36.5 μm'e ÇIKARIR** (BPM ofseti yanlışlıkla hata gibi
görülüp sisteme enjekte edilir).

**f sonrası = +1.07×10⁻⁴ rad/s.  Bastırma: 15.2×.**

---

**Özet tablosu:**

| Varyant | Fit | dS_y/dt | Bastırma |
|---|---|---|---|
| — | Trim yok | −1.623×10⁻³ | — |
| A | k=1..3 | −6.72×10⁻⁵ | 24.2× |
| **C** | **k=1..4** | **+1.61×10⁻⁵** | **100.8× ★** |
| D | k=1..5 | +1.37×10⁻⁴ | 11.9× |
| B | k=1..6 | +1.07×10⁻⁴ | 15.2× |

> **⚠ Evrensellik uyarısı:** Bu tablo TEK seed çiftine (321/777) aittir.
> 4 yeni seed ile yapılan tarama (`test_orbit_trim_seeds.py`) şunu
> gösterdi: 100.8× faz şansıydı; tipik bastırma ~5×, ve A/C/D/B
> artıkları istatistiksel olarak ayırt edilemez (hepsi ~2.5×10⁻⁴ rad/s
> RMS). Spin artığını fit edilen modlar değil, fit bazının **dışında**
> kalan içerik belirler: 48 serbestlik derecesinin yalnızca 25'i
> antisym k=0..12 bazındadır; kalan 23 boyut (simetrik QF/QD
> kombinasyonları, ~60–90 μm) yörüngeye zayıf, spine ise belirgin
> bağlanır. Evrensel olan sonuçlar: (1) k=1..4 yörünge içeriği her
> seed'de birkaç μm'e iner, (2) kazançlar analitik yasaya uyar
> (G_k = 24.8/|5.03−k²|, sapma %0.6), (3) fit eşiği G_k > σ_b/σ_q
> formülüyle önceden hesaplanabilir (σ_b=σ_q'da k_max=5.5 → k=5'in
> seed'e göre kararsızlığı eşikte olmasındandır).

### 7.6 Neden yörünge k≥7'yi göremez ama spin görür?

Kısa cevap: iki farklı fiziksel mekanizma — çelişki değil, beklenen sonuç.

**Yörünge kazancı rezonans kökenlidir.** Transfer matrisi, betatron tununa
(Q_y ≈ 2.68) yakın modları büyütür. k=7 için tahmin: ~0.45×. 100 μm
BPM gürültüsüyle, kazancı < 1 olan bir modun sinyali tamamen gürültü içinde
boğulur. BPM verisinden bu modu çıkarmak mümkün değildir.

**cₖ geometrik bir integraldır, rezonans gerektirmez.** dS_y/dt = Σ cₖ·Aₖ
formülündeki cₖ, halka boyunca radyal B alanının geometrik yol
integralidir. Bu integralin içinde Q_y yoktur. cₖ değerleri yavaş düşer:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7* | 8* |
|---|---|---|---|---|---|---|---|---|
| |cₖ| (rad/s/m) | 23.6 | **90.7** | 23.6 | 8.4 | 4.5 | 2.8 | ~1.96 | ~1.52 |
| Yörünge kazancı | 6.2 | 24.1 | 6.3 | 2.26 | 1.24 | 0.79 | ~0.45 | ~0.35 |

*Ekstrapolasyon*

**Sayısal kanıt:** C trimi sonrası k=7 = 26.8 μm, k=8 = 31.9 μm değişmedi.
Bunların spin katkısı (üst sınır, faz bağımlı):

    k=7:  1.96 × 2.68×10⁻⁵ ≤ 5.3×10⁻⁵ rad/s
    k=8:  1.52 × 3.19×10⁻⁵ ≤ 4.8×10⁻⁵ rad/s

C-trim artığı = +1.61×10⁻⁵ rad/s (bu seed'de). Çok-seed taraması artığın
genel kaynağını daha da netleştirdi: k≥7 antisym içeriğin yanında,
antisym bazın hiç KAPSAMADIĞI 23 boyutluk simetrik içerik de (~60–90 μm)
spine bağlanır; 5-seed artık tabanı ~2.5×10⁻⁴ rad/s'tir. **Yörünge
triminin tavanı, onun göremediği VE bazın temsil edemediği içerikten
kaynaklanır.**

Bu bir kısıtlama değil, **tanımlanmış bir sınır**: nerede durulacağını
biliyoruz ve neden biliyoruz. Sınırı itmenin iki yolu da bellidir:
σ_b'yi düşürmek (k_max² = Q² + 24.8·σ_q/σ_b büyür) ve baza simetrik
düğmeler eklemek.

### 7.7 İkinci iterasyon neden kazandırmaz?

Aynı BPM ölçümü → aynı yanlı kestirim → aynı (yanlı) trim → 2. iterasyon
güncellemesi: **tam 0.0000 μm**. Statik BPM ofseti aynı sistematik yanlılığı
yaratır; bu yanlılık, orbit döngüsünün birinci adımda oturduğu tabandır.
Tekrar etmek hiçbir şeyi değiştirmez.

---

## 8. Büyük resim: üç-kademe mimari

Tüm testleri bir araya getirince, sahte EDM bastırımının doğal üç kademesi
ortaya çıkıyor:

| Kademe | Araç | Referans | Bastırma | f sonrası |
|---|---|---|---|---|
| 0 | Ham | — | — | −1.623×10⁻³ rad/s |
| **1** | **Yörünge trimi** | BPM, EDM-kör | **~100×** | **+1.61×10⁻⁵ rad/s** |
| 2 | CW/CCW iptali | Yön simetrisi | ~10³–10⁸× | ~10⁻⁸–10⁻¹³ rad/s |
| 3 | Spin trimi (son) | Yalnız-sistematik | polarimetre sınırı | EDM tabanı |

**Yörünge kademesi** (Test III): k=1..4 içeriği birkaç μm'e indirilir.
EDM sinyaline hiç dokunmaz. Saniyeler içinde tamamlanır.

**CW/CCW iptali**: CW ve CCW ışınlarda Δy = +a ve −a; sahte EDM işaret
değiştirir, gerçek EDM değiştirmez → basit çıkarma ile 10³–10⁸× ek bastırım.

**Spin kademesi**: Kalan küçük artık (k≥5 ve trim hatalarından), yalnızca
polarimetre istatistiğiyle sınırlı spin ölçümü ile temizlenir.

### 8.1 Beş soruya beş yanıt

| Soru | Test | Yanıt |
|---|---|---|
| c_k doğrusal mı? | trim_realistic | Evet, %0.0 sapma |
| c_k desene bağlı mı? | mode_map_cofalse | Hayır, korelasyon 1.0000 |
| c_k fırlatma koşuluna bağlı mı? | launch_dep | Fiziksel olarak hayır; %0.5 doğruluk |
| BPM gürültüsü/ofseti döngüyü bozar mı? | trim_bpm | Tabana oturur, polarimetre sınırı |
| Rastgele fazlı gerçek desende çalışır mı? | random_trim | Evet, ≥10⁷×; faz 2 ölçümde çözülür |
| EDM sinyaline dokunmadan trim mümkün mü? | **orbit_trim** | **Evet; BPM rehberliğiyle 100×, EDM-kör** |

### 8.2 Deneysel reçete

1. **Başlangıç kalibrasyonu:** k=1..4 için cos+sin (8 ölçüm) →
   yörünge tepki matrisi O. Spin kalibrasyonu gerekmez bu adımda.
2. **BPM okuması:** Tur-ortalamalı y_BPM → k=1..4 mod kestirimleri â.
3. **Yörünge trimi:** Δp = −Σ âₖ·mod_k. Bütçe: birkaç ×10 μm — mekanik
   kaydırıcıların rahat aralığı.
4. **Spin doğrulama:** f ölçülür; beklenen değer ~10⁻⁵ rad/s mertebeсinde.
5. **Gerekirse CW/CCW:** Kalan ~10⁻⁵ rad/s'yi yön iptaliyle temizle.

Açık başlıklar: RF ve sekstüpol açık kafeste doğrusallık; x-yönü
kaçıklıkları ve quad tilt çapraz terimleri; zamanla sürüklenen hizalamada
izleme (tracking) kipi.
