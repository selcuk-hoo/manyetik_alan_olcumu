# Akıllı Düzeltme: Sahte EDM'i Misalignment'ı Bilmeden Sıfırlamak — Ders Kitabı Tarzı

Bu belge, proton EDM (pEDM) deneyinde bir **fikri** sıfırdan anlatır: kuadrupol
hizalama hatalarını (misalignment) tek tek ölçüp düzeltmeye çalışmak yerine, asıl
zararlı büyüklüğü — **sahte EDM** sinyalini — *doğrudan* sıfırlamak. Bu fikre
"**akıllı düzeltme**" diyoruz. Belge hiçbir ön bilgi varsaymaz.

> **Bu belgenin bir hikâyesi var.** İlk araştırma, fikrin "yörünge tarafı" kolunu
> *ölü* ilan etti. Sonra kullanıcı haklı bir itirazda bulundu — *"simetrik alt-uzay
> kapalı yörüngeyi az dahi olsa değiştirir; harita vardır, fakat basit değildir"* —
> ve bu, akıl yürütmedeki gerçek bir hatayı açığa çıkardı. Aşağıda **hem hatayı hem
> düzeltilmiş anlayışı** anlatıyoruz; çünkü hatanın kendisi öğreticidir. Terse
> teknik kayıt `akilli_duzeltme.md`'dedir.

İçindekiler:
1. Hatırlatma: sahte EDM nedir, neden tehlikeli?
2. Fikir: misalignment'ı değil, sahte-EDM'i hedeflemek
3. İki kol: spin-gözlemli (A) vs yörüngeden-öğrenilmiş (B)
4. Anahtar geometri: yörünge neyi *az* görür?
5. İlk akıl yürütme ve içindeki hata
6. Düzeltme: ileri-harita, inversiyon DEĞİLDİR (koşullanma)
7. Gereken hassasiyet ulaşılabilir mi? (7 nm ve ortalama)
8. O zaman gerçek engel ne? (karmaşık fonksiyonel)
9. Kol A vs Kol B: yine de neden farklılar?
10. Nerede duruyoruz (dürüst durum)
11. Sık sorulan sorular

---

## 1. Hatırlatma: sahte EDM nedir, neden tehlikeli?

Proton EDM deneyi, protonun spininin yatay düzlemden **düşeye doğru çok yavaş
dönmesini** arar. Ölçtüğümüz büyüklük dikey spin bileşeninin birikme hızıdır:
$$f \equiv \frac{dS_y}{dt}.$$
Anlamlı bir ölçüm için $f \sim 1\ \text{nrad/s}$ ($10^{-9}$ rad/s) seviyesi gerekir.
Sorun: **mıknatıs hizalama hataları da $f \neq 0$ üretir** ve gerçek EDM'den ayırt
edilemez. Buna **sahte EDM** denir.

Mekanizma (ayrıntı `trim_yontemi_pedagojik.md §1`): bir kuadrupol yatayda $dx$,
düşeyde $dy$ kayarsa, spinin iki farklı eksen etrafındaki dönmeleri **sıra
değiştirilemez** → net bir **geometrik (Berry) fazı** kalır:
$$f \;\propto\; dx \cdot dy \;\propto\; \sigma^2.$$
($\sigma^2$ yasası bu oturumda da doğrulandı: üs $p = 2.002$.)

---

## 2. Fikir: misalignment'ı değil, sahte-EDM'i hedeflemek

Doğal yaklaşım: 48 kuadrupolün kaçıklığını yörüngeden (BPM'lerle) ölç, düzelt. Ama
bunun bir **no-go**'su var: kaçıklığı yörüngeden geri-çatmak bir **inversiyon**dur
ve sahte-EDM'i süren kısımda **patolojik kötü koşulludur** (gürültüyü ~$10^4$ kat
büyütür; `squid_bpm_test.md §8`).

**Kullanıcının fikri:** Asıl amacımız kaçıklığı ölçmek değil, **sahte-EDM'i
sıfırlamak.** Neden kaçıklığı tam çözmeye uğraşalım? Belki sahte-EDM'i **doğrudan**
sıfırlayan bir düzeltme buluruz. (Radyoda cızırtıyı, kaynağını teşhis etmeden,
sadece dinleyip düğme çevirerek susturmak gibi.)

---

## 3. İki kol: spin-gözlemli (A) vs yörüngeden-öğrenilmiş (B)

**Kol A — spin-gözlemli.** Sahte-EDM'i ($f$) **spinle ölç**, düğmeyi $f$ sıfırlanana
dek çevir. Bu "spin ölç-trim" olarak biliniyor ve **çalışıyor**
(`false_edm_harmonic_sinir.md §14.6`, ~6000×). Dezavantajı: spin ölçmek yavaş ve
pahalı.

**Kol B — yörüngeden-öğrenilmiş ileri-harita.** *Sadece yörüngeye (BPM'lere)
bakarak* sahte-EDM'i öngörebilir miyiz? Bir **sinir ağına** simülasyondan
**yörünge → sahte-EDM** haritasını (ileri-harita) öğret; orbit-görünür düğmelerle
EDM'i hedefleyerek düzelt. Bu oturumun açık sorusu Kol B'ydi.

---

## 4. Anahtar geometri: yörünge neyi *az* görür?

Bir hücrede QF ve QD kaçıklıkları **zıt işaretli** ise *antisimetrik*, **aynı
işaretli** ise *simetrik* desen denir.

- **Antisimetrik kaçıklık → büyük kapalı yörünge** (BPM'ler net görür).
- **Simetrik kaçıklık → çok küçük kapalı yörünge** (BPM'ler *zar zor* görür).

Neden? Simetrik desen yörüngeyi yüksek-$k$ harmonikle dürter; tepki $G_k =
C/|Q^2-k^2|$ yüksek-$k$'da bastırılır. Sayıyla (SVD, tekil değerler): simetrik
desenler yörüngede en küçük tekil değer yönlerinde yaşar; oran
$\sigma_\text{max}/\sigma_\text{min} = 193$. Yani **simetrik kaçıklık, yörüngede
aynı büyüklükteki antisimetrik kaçıklığa göre 193 kat daha az iz bırakır.**

**Kritik incelik (hikâyenin dönüm noktası): 193 kat *az*, ama *sıfır değil*.**
Simetrik kaçıklık yörüngede minik ama **gerçek** bir iz bırakır. Bunu unutmak,
birazdan göreceğimiz hataya yol açtı.

---

## 5. İlk akıl yürütme ve içindeki hata

**Deney (gerçek C++ spin izleyicisiyle).** İki makine: A, ve A + simetrik (orbit-kör)
10 μm pertürbasyon. Ölçtük:

| pertürbasyon | yörünge izi (ΔCOD) | sahte-EDM değişimi \|Δf\| |
|--------------|-------------------:|--------------------------:|
| antisimetrik (görünür) | 115 μm | $9.3\times10^{-6}$ (~9270× hedef) |
| **simetrik (kör)** | **1.7 μm** | $2.5\times10^{-7}$ (**~247× hedef**) |

**İlk (yanlış) sonuç:** "Yörüngeyi yalnız 1.7 μm değiştiren bir kaçıklık, sahte-EDM'i
247× hedef oynatıyor. Demek ki yörünge sahte-EDM'i belirlemiyor; Kol B ölü."

**Kullanıcının itirazı (haklı):** *Ama 1.7 μm sıfır değil. Simetrik alt-uzay
yörüngeyi az dahi olsa değiştiriyor; bu dejenerasyonu bozar. Harita vardır, sadece
basit değildir.* — İşte hatayı bulan cümle. "Yörünge f'i belirlemiyor" demek
yanlıştı; doğrusu "yörünge f'i **küçük bir izle** belirliyor."

---

## 6. Düzeltme: ileri-harita, inversiyon DEĞİLDİR

Hata neredeydi? Sahte-EDM'i (skaler $f$) yörüngeden öngörmeyi, **48 kaçıklığı
geri-çatmakla** (inversiyon) karıştırdım. İkisi çok farklı koşullanır.

**İnversiyon (COD → 48 kaçıklık):** simetrik yönde hata $1/\sigma_\text{min} \approx
7\times$ (tam k-mod matrisinde $\sim10^4\times$) **büyür.** Felaket.

**İleri-harita (COD → skaler $f$):** duyarlılığı doğrudan ölçtük:
$$\frac{\partial f}{\partial \text{COD}}\Big|_\text{sim} = 0.146,\qquad
\frac{\partial f}{\partial \text{COD}}\Big|_\text{anti} = 0.081\ \frac{\text{rad/s}}{\text{m}}.$$
**İkisi de mütevazı ve ~eşit.** $1/\sigma_\text{min}$ büyütmesi **yok!**

Neden büyütme yok? Çünkü simetrik kaçıklığın *hem* yörünge izi ($\sigma_\text{min}$
ile) *hem de* sahte-EDM üretimi küçüktür; ikisi birbirini götürür, oranları (yani
yörünge-izi başına $f$) normal kalır. İnversiyon yörünge-izini bölmek (büyütmek)
zorundadır; ileri-harita bölmez, doğrudan $f$'i okur.

> **Sonuç:** "Kol B = inversiyon no-go, aynı duvar" iddiası **YANLIŞTI, geri
> çekildi.** İleri-harita iyi koşulludur; inversiyonun duvarına çarpmaz.

---

## 7. Gereken hassasiyet ulaşılabilir mi? (7 nm ve ortalama)

İleri-harita iyi koşullu ama yine de ince: $f$'i hedefe ($10^{-9}$) öngörmek için
yörüngeyi $f$-yönünde
$$\delta\text{COD} = \frac{10^{-9}}{0.146} \approx 7\ \text{nm}$$
doğrulukta bilmek gerekir. Kulağa imkânsız geliyor (BPM'ler μm seviyesinde). Ama:

**BPM gürültüsü ortalamayla yenilir.** $\sigma_\text{eff} = \sigma_\text{BPM}/\sqrt N$.
1 μm tek-atış gürültüden 7 nm'ye: $N = (1\,\mu\text{m}/7\,\text{nm})^2 \approx
2\times10^4$ ölçüm $\approx$ **21 saniye** (1 kHz'de). Tamamen makul.

**BPM ofseti (~100 μm) bile duvar değil.** Şaşırtıcı: Berry yönlü-alan özniteliği
$\sum_i(x_iy_{i+1}-x_{i+1}y_i)$, kapalı bir halkada **sabit ofsete değişmezdir**
(ofset terimleri toplamda iptal olur). Yani akıllı bir öznitelik 100 μm ofseti
otomatik aşar.

> Demek ki **ölçüm (gürültü/ofset) Kol B'yi dışlamıyor.** 7 nm, inversiyonun
> "< 4 nm"siyle sayısal olarak benzer ama **anlamı farklı:** inversiyonda o sayı
> $1/\sigma_\text{min}$ büyütmesinden *sonra* gerekir ve β-beat ile felakete döner;
> ileri-haritada doğrudan ölçülebilir bir gürültü eşiğidir, ortalamayla yenilir.

---

## 8. O zaman gerçek engel ne? (karmaşık fonksiyonel)

Eğer gözlenebilirlik engel değilse, Kol B neden hâlâ çözülmüş değil? Çünkü **doğru
ileri-haritayı bulmak zor.** $f$, yörüngenin **doğrusal-olmayan, yapısal** bir
fonksiyonelidir (Berry yönlü-alan tipi); basit ⟨x·y⟩ yanlış proxy
(`orbit_ileri_olcum.md §2-3`). Empirik olarak pinlemek 40–80 konfigürasyonla
yakınsamıyor.

Bunu somut gösterdik: temiz-yörünge verisinde gerçek bir model (GradBoost) eğitip,
**asıl iş olan orbit-kör simetrik kanalın** öğrenilebilirliğini örnek sayısıyla
izledik:

| eğitim örneği N | simetrik-kanal CV R² |
|---|---|
| 60 | $-0.52$ (az veri, öğrenilemiyor) |
| 100 | $+0.64$ |
| 180 | $+0.74$ |
| 240 | $+0.77$ |

**Orbit-kör simetrik kanal — yani sahte-EDM'i süren kısım — temiz yörüngeden
ÖĞRENİLİYOR** (R² $-0.5 \to 0.77$). 80 örnekte zar zordu (R²~0), ama veri arttıkça
net pozitife yükseliyor ve ~0.77'de doyuyor.

Bu, kullanıcının teşhisinin **tam doğrulamasıdır:** *harita vardır, fakat basit
değildir* — karmaşık (az veriyle pinlenmiyor) ama **yeterli veriyle öğreniliyor.**
`orbit_ileri_olcum.md §3`'ün "40 config ile yakınsamıyor" gözlemini de açıklar:
veri azdı, fonksiyon yok değil.

### 8.1 "Ama harita sim'den öğreniliyor — gerçek makineye taşınır mı?" (β-beat)

En kritik itiraz: ileri-haritayı **simülasyondan** öğreniyoruz, ama gerçek makinenin
optiği (β fonksiyonları) sim'den farklı olacak (β-beat: kuadrupol gradyanlarındaki
~%1 hata). Sim'de öğrenilen harita gerçeğe taşınmazsa Kol B yine kırılır. *(İnversiyon
tam burada ölür: β-beat'i $1/\sigma_\text{min}$ ile büyütür, `squid_bpm_test §9.5`.)*

Bunu doğrudan sınadık: %1 per-quad β-beat'li bir "gerçek makine" kurduk (C++,
`quad_dG`), haritayı **β-beat'siz** 200 config'de eğittik, **β-beat'li** 40 config'de
test ettik (test config'leri eğitimden dışlandı — sızıntı yok).

| test | R² |
|------|---|
| held-out NOMİNAL (referans) | +0.61 |
| **β-beat makineye transfer** | **+0.62** |
| β-beat, simetrik kanal (w≥0.75) | **+0.83** |

**Sonuç: β-beat ŞEFFAF.** %1 β-beat sahte-EDM'i ~%18 oynatıyor, ama nominal-eğitimli
harita β-beat makineyi **held-out nominal kadar iyi** öngörüyor (0.62 vs 0.61 — **ek
bozulma yok**). Yani **sahte-EDM, yörüngenin ~sabit bir fonksiyonelidir**; optik
değişse de "aynı yörünge ≈ aynı sahte-EDM" kalıyor (harita β-beat kaymasını COD
üzerinden kısmen bile izliyor). İnversiyonun aksine, ileri-harita β-beat'le **kırılmıyor.**

→ Geriye **tek** açık şey kalıyor: mutlak doğruluğu EDM-hedefine indirmek (daha çok
veri / öznitelik / analitik Berry fonksiyoneli + null'lama iterasyonu) ve gerçekçi
gürültü bütçesi. Bunlar **mühendislik/veri** problemleri — fizik imkânsızlığı değil.

---

## 9. Kol A vs Kol B: yine de neden farklılar?

Sahte-EDM, kuadrupoldeki **yerel demet ofsetini** ($x_{\text{CO},i} - dx_i$) görür.
Simetrik kaçıklık için kapalı yörünge $x_\text{CO}\approx 0$ ama yerel ofset
$\approx -dx_i$ — **büyük.** Yani:

- **Kol A** (spin) bu yerel ofseti **doğrudan** gözler → bilgi tam ve temiz →
  geri-besleme çalışır. (Maliyeti: spin ölçümü.)
- **Kol B** (yörünge) yerel ofseti **yalnız dolaylı**, $\sigma_\text{min}$ ile
  bastırılmış 1.7 μm'lik bir iz olarak görür → bilgi *var* ama **küçük ve karmaşık
  kodlanmış.** Onu çözmek mümkün (§6-7) ama doğru, karmaşık fonksiyoneli ve yeterli
  ortalamayı gerektirir.

Yani fark "var/yok" değil, **"doğrudan/temiz" vs "dolaylı/küçük/karmaşık".** Kol A
kolay yoldan görür; Kol B zor yoldan — ama **imkânsız değil.**

---

## 10. Nerede duruyoruz (dürüst durum)

- **Kol B kapatılmadı; üstelik DÖRT bulgu destekliyor.** "Ölü/aynı-duvar" sonucu,
  ileri-harita ile inversiyonu karıştıran bir hataydı (kullanıcı düzeltti).
  (1) **iyi koşullu** ($\partial f/\partial\text{COD}\approx0.15$, $1/\sigma_\text{min}$
  değil); (2) gereken 7 nm **ortalamayla ulaşılır**, ofset değişmez öznitelikle
  aşılır; (3) orbit-kör simetrik kanal temiz yörüngeden **öğreniliyor** (CV R²
  $\to0.77$); (4) **β-beat şeffaf** — sim-eğitimli harita %1 β-beat'li makineye
  held-out nominal kadar iyi taşınıyor (R² 0.62 vs 0.61). → İnversiyon no-go'su
  Kol B'yi **bağlamaz.**
- **Açık (tek kalan):** mutlak doğruluğu EDM-hedefine indirmek (daha çok veri /
  öznitelik / analitik Berry fonksiyoneli + null'lama iterasyonu) ve gerçekçi-ölçüm
  gürültü bütçesi. Bu **mühendislik/veri** işidir; `orbit_ileri_olcum.md §7`'nin
  açık problemini **pozitif yönde ilerletir** (fizik no-go'su yok).
- **Kol A** çalışır, ama spin gerektirir (orbit-tarafı değil; Omarov/spin-trim).
- **Birleşik no-go** (orbit-inversiyon + lock-in) yalnız *misalignment geri-çatımı*
  içindir; **ileri-harita f-öngörüsü o sınıfa girmez.**

---

## 11. Sık sorulan sorular

**S: Yani "yörünge sahte-EDM'i belirlemez" yanlış mıydı?**
C: Evet. Doğrusu: yörünge sahte-EDM'i **küçük bir izle** belirler. İz küçük olduğu
için *okumak* zor, ama bilgi oradadır (dejenerasyon tam değildir). İlk sürüm bunu
"belirlemez"e indirgeyerek hata yaptı.

**S: İleri-harita gerçekten inversiyondan iyi koşullu mu?**
C: Skaler $f$ için evet. İnversiyon 48 kaçıklığı çözer ve simetrik yönde
$1/\sigma_\text{min}$ ile patlar. İleri-harita doğrudan $f$ verir; ölçülen
duyarlılık 0.146 (mütevazı), $1/\sigma_\text{min}$ büyütmesi yok. Fark, "bir küçük
sayıya bölmek" zorunda olup olmamaktır.

**S: O zaman Kol B çalışır mı?**
C: Pozitif yöne gidiyor. **Gözlenebilirlik yüzünden dışlanmadı**, ve temiz veriyle
asıl zor kanal (orbit-kör simetrik) **öğrenilebilir çıktı** (CV R²→0.77). Kalan
engeller *fizik no-go'su* değil, *mühendislik*: gerçekçi ölçüm gürültüsü/ortalama
bütçesi ve model-fidelity (β-beat). Yani "çalışır mı" sorusu artık "ölçüm ve
kalibrasyon yeterince iyi olur mu" sorusuna indi — kapalı bir kapı değil.

**S: 7 nm yörünge ölçümü gerçekten yapılabilir mi?**
C: Gürültü açısından evet (ortalama: ~21 s). Sabit ofset, yönlü-alan gibi
ofset-değişmez özniteliklerle aşılır. Kalan risk *sürüklenme* ve *model hatası*;
bunlar tasarımda ele alınmalı.

**S: Bu hikâyeden ders ne?**
C: "Küçük" ile "sıfır"ı karıştırmamak. Simetrik kanalın yörünge izi *küçük* (1.7 μm,
193× bastırılmış) ama *sıfır değil*; ve onu okuyan ileri-harita *iyi koşullu*. Bir
imkânsızlık ilan etmeden önce, doğru problemi (ileri-harita ≠ inversiyon)
çözdüğümüzden emin olmalıyız.

---

> **Kayıt notu (2026-06-29, düzeltilmiş):** Bu belge `akilli_duzeltme.md`'nin
> (terse teknik kayıt) pedagojik kardeşidir. İlk sürüm "Kol B ölü" diyordu;
> kullanıcının haklı itirazıyla düzeltildi (ileri-harita ≠ inversiyon; iyi koşullu;
> engel fonksiyonel karmaşıklığı, gözlenebilirlik değil). Sayılar gerçek C++
> izleyiciyle ($p=2.002$) üretildi; keşif kodu `/tmp/akilli_duzeltme/`.
> `integrator.cpp` değiştirilmedi.
