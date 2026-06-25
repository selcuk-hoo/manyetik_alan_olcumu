# REFERANS: COD drift paper (sürekli ORM / kapalı yörünge drift)

> Kaynak PDF'ten pdftotext ile çıkarıldı (özgünlük karşılaştırması için referans).

---

            Dondurulmuş-spin proton EDM halkasında
           kapalı-yörünge tabanlı hizalama drift izlemesi:
               performans ve gözlenebilirlik sınırları∗
                                                   (yazar bilgisi)
                                                   June 25, 2026


Abstract                                                          için sonraki nesil simetrik-hibrit tasarım, manyetik
                                                                  kuadrupollerle alternating-gradient (FODO) odak-
Proton EDM (pEDM) deneyinin alternating-gradient                  lamaya geçti [1]. Bu geçişle sistematik öncelikler de
(AG) versiyonunda, manyetik kuadrupol hizalama                    değişti: baskın sistematik artık kuadrupol hizalama
hataları baskın sistematik kaynağıdır ve spin koherans            hatalarıdır.
zaman ölçeğinde ∼ 10 µm seviyesinde kontrol altında                  Hizalanmamış bir kuadrupol, üzerinden geçen
tutulmalıdır [1]. Bu çalışmada, bu hizalamayı sürekli             demete net bir kuvvet uygular; bu kuvvetin halka
izlemek için kalibrasyon anına göreli bir kapalı-yörünge          boyunca uygun bileşeni EDM sinyalini taklit eden sahte
tabanlı online hizalama drift izleme yöntemi öneriyor             bir dikey spin presesyonu üretir. Omarov vd. [1],
ve sistematik simülasyon testleriyle hem yeteneklerini            manyetik-quad’lı pEDM tasarımında bu sahte alanın
hem de gözlenebilirlik sınırını niceliyoruz. Yaklaşım,            sistematik bütçesini ayrıntılı türetmiş ve hedef has-
problemi bilinçli olarak yeniden tanımlar: amaç mutlak            sasiyeti (dp < 10−29 e·cm, eşdeğer dSy /dt < 1 nrad/s)
hizalama rekonstrüksiyonu (∆q = R−1 y) değil, göreli              korumak için hizalama hatalarının spin koherans za-
hizalama kararlılığı izlemesidir. Yöntem, kalibrasyon             man ölçeğinde 10 µm RMS seviyesinde bilin-
anındaki BPM okumasını referans alarak δq(t)      b      =        mesi/kontrol edilmesi gerektiğini göstermiştir. Bu
R (y(t)−y0 ) ile hizalama driftini kestirir; sabit BPM
  −1
                                                                  sayı, bu çalışmanın hedef hassasiyetini doğrudan belir-
elektronik ofseti zaman farkında iptal olur. Gerçekçi             ler.
hata bütçesinde (50 µm RMS ofset, 1 µm gürültü) yön-                 Halkada 2 × 24 = 48 manyetik kuadrupol vardır; her
tem 6–7 µm RMS hassasiyetle çalışır (mutlak rekon-                biri yatay (dx) ve dikey (dy) eksende bağımsız kaya-
strüksiyona göre ∼ 29× iyileşme) ve LOCO kalitesinde              bilir. Ölçüm aracı 48 BPM çiftidir. Temel zorluk
(%1) β-beating ile gerçekçi quad tilt altında dahi                şudur: BPM elektronik ofsetleri (∼ 100 µm) ölçülmek
hedefin altında kalır. Analitik tepki matrisini tam               istenen hizalama sinyaliyle (∼ 10 µm) aynı büyük-
parçacık (semplektik + spin) izleyicisiyle doğruluy-              lüktedir; mutlak bir kapalı-yörünge ölçümü bu ofsete
oruz. Yöntemin gözlenebilirlik sınırını singüler-                 boğulur.
değer (SVD) per-mod analiziyle ortaya koyuyoruz:
monitör, antisimetrik (hücre içi QF/QD zıt yönde)
hizalama driftini güçlü, simetrik (QF/QD aynı yönde)              1.2    Alanın     benimsediği             hizalama
driftini ise zayıf çözer — en kötü koşullanmış mod-                      stratejisi
lar %96 simetriktir ve en kötü modda en iyiye göre
                                                               Omarov vd.’nin simetrik-hibrit tasarımı, hizalamayı
∼ 193 kat gürültü dezavantajına uğrar. Böylece yön-
                                                               mekanik olarak µm seviyesinde dayatmak yerine sis-
temin kesin tanımlı bir geçerlilik alanı (antisimetrik
                                                               tematiği hizalamaya duyarsız kılan üç katmanlı bir
drift) ve kesin tanımlı bir kör noktası (simetrik drift)
                                                               savunmaya dayanır: (i) pasif kafes simetrisi (σ 2
vardır; ikincisinin belirli bir sistematik bütçe (ör. sahte
                                                               bastırma), (ii) CW/CCW karşı-dönen demetler +
EDM) için önemi makineye bağlıdır ve ayrı bir çalış-
                                                               kuadrupol polarite çevirmeyle aktif iptal, (iii) spin-
maya bırakılır.
                                                               tabanlı hizalama (SBA), “yükselt-sonra-söndür” nu-
                                                               marasıyla. Bu çerçevede asıl hassas hizalama ölçümü
1 Giriş                                                        tek-demet kapalı yörüngesinden değil, karşı-dönen
                                                               demet ayrımının (∆y) SQUID-tabanlı BPM’lerle okun-
1.1 Bağlam ve sistematik önceliği                              masından ve spinin kalibrasyon probu olarak kul-
                                                               lanılmasından gelir; mekanik tolerans ∼ 100 µm’ye
Proton EDM deneyinin ilk önerileri tamamen elek- gevşetilir.
trostatik, zayıf-odaklamalı halka tasarımına dayanıy-             Dolayısıyla bu makalenin amacı klasik kapalı-
ordu [2]; baskın sistematik ortalama dikey manyetik yörünge/k-modülasyon yöntemini bu yerleşik strate-
alandı. Demet dinamiğini ve kararlılığı iyileştirmek jinin yerine koymak değildir. Aksine, mevcut BPM
   ∗ Taslak v0.1 (2026-06-19).     makale-taslagi-2.md ve
                                                               altyapısıyla sürekli, ucuz ve fizik-veri-toplamayı boz-
drift_monitor/ test sonuçlarından konsolide edilmiştir; başlık madan çalışabilen bir tamamlayıcı online drift iz-
geçicidir.                                                     leyicinin neyi başarabileceğini ve nerede yapısal olarak


                                                              1
duracağını nicelemektir. Kısaca: kapalı yörüngeyi               βi , ϕi her quad girişindeki Twiss parametreleri, Q tune,
spin-tabanlı hizalama teşhisinin yerine koymuyoruz;             (KL)j quad’ın işaretli integral gücüdür (QF/QD tip
onun sürekli bir hizalama-drift izleyici olarak hizmet          işareti ve düzlem işareti içinde saklanır). Gürültü
edebileceği gözlenebilir alt-uzayı ve bu alt-uzayın             için bu çalışmada ση = 1 µm alınır; bu, tek BPM
sınırını niceliyoruz.                                           okuma gürültüsü değil, çok-tur ortalaması sonrası
                                                                etkin yörünge belirleme belirsizliğini temsil eden bir
1.3    Çalışmanın katkısı                                       model parametresidir.
                                                                   Tek bir incelik yatay düzlemdedir.         Halkadaki
 1. Yöntem ve doğrulama. Kalibrasyon-referanslı                 elektrostatik ark deflektörleri yatay düzlemde demeti
    drift modu δq(t)
                 b      = R−1 (y(t) − y0 ), sabit BPM           hafifçe odaklar; bu odaklamanın gücü Kx,arc , R’yi
    ofsetini yapısal olarak iptal eder ve iyi-koşullu R         doğru kurmak için gerekir ama temiz bir analitik kapalı
    (κ ≈ 193) kullanır; gerçekçi hata bütçesinde 6–             formu yoktur. Onu tune-eşleme ile belirleriz: analitik
    7 µm RMS sağlar (§3). Kullanılan analitik R, tam            modelin ürettiği yatay tune Qx , tam izleyicinin verdiği
    parçacık izleyicisiyle doğrulanır (§2.2).                   Qx değerine eşitlenene kadar tek serbest parametre
                                                                Kx,arc ayarlanır (bir kök-bulma). Dikey düzlemde bu
 2. Sağlamlık. Yöntem; BPM ofsetine, β-beating’e
                                                                serbestlik yoktur: Maxwell gereği elektrostatik deflek-
    (gerçek gradyan hatasıyla) ve gerçekçi quad tilt’in
                                                                törün orta düzlemde dikey alanı sıfırdır (Ez = 0), ark-
    yarattığı x–y kuplajına karşı sağlamdır (§3).
                                                                lar dikeyde odaklamaz ve Ky,arc = 0 — kesin, kali-
 3. Yapısal arka plan (destekleyici). İki-ölçümlü,              brasyonsuz. Bir model parametresini sonra karşı sınay-
    tam-ofset-iptal eden lineer estimatör sınıfı için bir       acağımız simülasyona uydurmak bir ters-suç (inverse
    tekillik önermesi verilir (∥∆R−1 ∥ ∼ ∥R−1 ∥/ε); bu,         crime) riski taşır; bunu §3.3’te denetliyoruz: Kx,arc ’ı
    “iki gradient farkıyla ofseti iptal et” gibi bariz          ±%10 bozmak drift kurtarımını < 0.5 µm değiştirir,
    alternatiflerin neden yapısal olarak kötü-koşullu           yani sonuç bu kalibrasyona bağlı değildir.
    olduğunu, drift modunun (iki zaman) ise neden                  Bu inşa simülasyon izleyicisinden bağımsızdır ve
    bu sınırın dışında kaldığını açıklar (§2.4).                fodo_lattice.py içinde uygulanmıştır. Varsayılan
                                                                örgüde κ(R) ≈ 193, σmax ≈ 28.4, σmin ≈ 0.147.
 4. Gözlenebilirlik karakterizasyonu. Per-mod
    SVD analiziyle, monitörün hangi hizalama desen-
    lerini iyi/kötü çözdüğü nicelenir: antisimetrik drift       Analitik R’nin tam parçacık takibiyle doğru-
    güçlü, simetrik drift zayıf gözlenir (§4). Bu, yön-         lanması. Analitik R tüm sonuçların temeli olduğun-
    temin geçerlilik alanını ve kör noktasını tanımlar;         dan, onu bağımsız bir tam parçacık izleyicisiyle (C++
    kör noktanın belirli bir sistematik bütçe için önemi        GL4 semplektik, integrator.cpp) karşılaştırıyoruz.
    (ör. sahte EDM) makineye bağlı ayrı bir sorudur.            İzleyici R’si, her kuadrupole küçük bir δy (veya δx)
                                                                perturbasyonu uygulayıp kapalı yörünge tepkisini 48
                                                                BPM’de ölçerek kurulur. Sonuç (Şekil 1): analitik ve
2     Model ve Yöntem                                           izleyici R’leri eleman-eleman korelasyon 0.9992 (dikey)
                                                                / 0.9977 (yatay), göreli fark ∥Rsim − Ran ∥/∥Ran ∥ =
2.1    Lineer model                                             5.3%/7.2%, kondisyon sayıları tutarlı (κ: 228 vs 193
Lineer rejimde BPM okumaları ile hizalama hataları              dikey, 135 vs 140 yatay). Böylece analitik R üzerine
arasındaki ilişki:                                              kurulan tüm gözlenebilirlik analizi tam parçacık di-
                                                                namiğiyle doğrulanmış olur.
                  y = R ∆q + b + η,                  (1)

burada ∆q ∈ R48 misalignment vektörü, R ∈ R48×48                2.3    Aday estimatörler
tepki matrisi, b BPM elektronik ofseti, η ölçüm
                                                                Aday yöntemler iki sınıfa ayrılır: ofseti tahmin etm-
gürültüsüdür. Modelleme varsayımı: yatay–dikey
                                                                eye/iptal etmeye çalışan mutlak rekonstrüksiyon
kuplaj yoktur (skew bileşeni = kuadrupol dönmesi sıfır
                                                                yaklaşımları (i–iii) ve ofseti bir nuisance (rahatsızlık)
kabul edilir), dolayısıyla problem birbirinden bağım-
                                                                parametresi gibi eleyip yalnız değişimi kestiren göreli
sız iki 48 × 48 sisteme (dy → dikey yörünge, dx →
                                                                kararlılık izlemesi (iv).
yatay yörünge) ayrılır. Bu düzlem-ayrıklığın, drift
izlemenin sahte-EDM ile ilişkisi açısından önemli bir
sonucu vardır (§4.4): lineer yörünge tepkisi tek bir dü-        (i) Mutlak tek-gradient. ∆q    c = R−1 y. κ(R) küçük
zlemin yer değiştirmesini taşır, iki düzlemin çarpımını         olduğundan gürültü iyi kontrol altındadır, ancak BPM
değil.                                                          ofseti R−1 b olarak tahmine doğrudan sızar. Mutlak
                                                                hizalama bilgisini ofsete kurban eder.
2.2    Tepki matrisinin analitik inşası
Periyodik FODO örgüsünde, Courant-Snyder formal-                (ii) İki-gradient ∆R−1 (ofset-iptal). İki farklı
izmiyle [3] kapalı-yörünge tepki matrisi:                       gradient ayarında ölçüm alalım: g1 ve g2 = g1 (1 + ε),
                                                                tepki matrisleri R1 , R2 . Aynı ∆q ve aynı ofset b için
                                                                ya = Ra ∆q + b + ηa (a = 1, 2); fark alınca ofset düşer
             p
               βi βj
    Rij =             cos |ϕi − ϕj | − πQ (KL)j , (2)
                                         
           2 sin(πQ)                                            ve y1 −y2 = ∆R ∆q+(η1 −η2 ), ∆R ≡ R1 −R2 . Böylece


                                                            2
                                                             R∆q0 + b + η0 kaydedilir; sonra

                                                                                  = R−1 y(t) − y0 .               (4)
                                                                                                 
                                                                            δq(t)
                                                                            b

                                                             y(t)−y0 = R(∆q(t)−∆q0 )+(η(t)−η0 ) (sabit b düşer),
                                                             dolayısıyla δq(t)
                                                                          b    = (∆q(t) − ∆q0 ) + R−1 (η(t) − η0 ).
                                                             İki bağımsız gürültü örneği toplamda 2ση2 varyans
                                                             verir; hata kovaryansı R−1 (2ση2 I)R−⊤ , RMS mertebesi
                                                             √
                                                               2 ση ∥R−1 ∥. Kritik fark (ii) ile: burada terslenen ma-
                                                             tris iyi-koşullu R (κ ≈ 193), kötü-koşullu ∆R değildir
                                                             — gürültü ∼ 1/ε değil O(1)’dir. Yöntem mutlak hiza-
                                                             lamayı değil kalibrasyondan beri değişimi verir; mut-
                                                             lak referans dış kaynaktan (LOCO/BBA) gelir (§5).

                                                             2.4    Neden (ii) çıkmaz ama (iv) çalışır?
                                                             (ii) ile (iv) farklı sorular çözer; aradaki fark yöntemin
                                                             tüm hikâyesidir. Ofseti yok etmenin “doğal” yolu eşza-
                                                             manlı iki ölçümün farkını almaktır (ii); ama eşzamanlı
                                                             iki ölçümde ofseti tam iptal eden her lineer, ön-yargısız
                                                             estimatör kaçıklığı sonuçta ∆R üzerinden geri çatmak
                                                             zorundadır — ve ∆R yapısal olarak ε-küçüktür, yani
                                                             gürültü 1/ε patlar. Bu algoritma seçimiyle aşılamaz:
                                                             ön-yargısızlık A1 R1 + A2 R2 = I ve tam ofset iptali
                                                             A1 + A2 = 0 koşulları birlikte tek çözüme zorlar,
                                                             A1 = ∆R−1 . Düzenlileştirme (iii) bu sınıfın dışına bias
                                                             ekleyerek çıkar ama bedeli ∼ 50 µm’dir. Simülasyon
                                                             doğrular: ham ∆R−1 Test 1’de 1865 m, Test 6’da
Figure 1: Analitik Courant-Snyder R (teori) ile tam          ∼3000 m; Şekil 3 gürültünün ∝ 1/ε patladığını gös-
parçacık izleyicisinden (C++ GL4, integrator.cpp)            terir. Drift modu (iv) ise ofseti eşzamanlı iptal etmeye
kurulan R’nin eleman-eleman karşılaştırması; (a)             çalışmaz; iki farklı zamanda ölçer, sabit ofset zaman
dikey, (b) yatay düzlem. Noktalar y = x üzerinde:            farkında kendiliğinden düşer ve terslenen matris iyi-
korelasyon 0.9992/0.9977, κ tutarlı. Drift makalesinin       koşullu R’dir (κ ≈ 193) — bu yüzden gürültü O(1)’dir.
analitik temelini tam parçacık dinamiğiyle doğrular.         Bedeli yalnız değişimi vermesidir; ama aradığımız da
                                                             odur.
∆q
c = ∆R−1 (y1 − y2 ) = ∆q + ∆R−1 (η1 − η2 ). Sorun,
∆R’nin küçük olmasıdır: R ∝ (KL) ∝ g olduğundan              3     Sayısal Deneyler
                             ∂R
         ∆R = R1 − R2 ≈ −ε g    ≡ −εR′ ,               (3) Tüm testler ortak bir altyapıda koşuldu (Ek A):
                             ∂g
                                                           semplektik izleyici, gerçekçi BPM gürültü ve of-
yani ∆R, R ölçeğinin yalnız ε (≈ 0.02) katıdır. Tersini set modeli, opsiyonel quad/dipol tilt’leri.           Hız-
                                      √
almak
 √      gürültüyü  ∥ ∆q
                     c − ∆q∥ ≈          2 σ η ∥∆R −1
                                                     ∥  =  landırıcı  parametreleri params.json,     test parame-
( 2 ση /ε) ∥R′−1 ∥ gibi 1/ε ile patlatır. (Dikkat: 1/ε treleri test_params.json içindedir.
ile büyüyen şey kondisyon sayısı κ(∆R) değil — o
≈ κ(R′ ), ε’dan bağımsız; bizzat gürültü büyütmesi
                                                           3.1 Test 1 — Düzenlileştirme iki-
∥∆R−1 ∥’dir.) Pratikte ∥∆R−1 ∥ ∼ 104 × (Şekil 2,
Şekil 3).                                                          gradient yöntemini kurtarır mı?
                                                           Soru: §2.3-(iii)’teki düzenlileştirme, (ii)’nin gürültü
(iii) ∆R−1 + düzenlileştirme. (ii)’nin gürültü pat- patlamasını 10 m hedefine indirebilir mi? En lehte du-
lamasını dizginlemek için: Tikhonov ham ters yerine rumu kuruyoruz: ofset sıfır ((ii)’nin tek derdi gürültü
(∆R⊤ ∆R + λI)−1 ∆R⊤ kullanır (λ küçük tekil değer- olsun) ve aynı veride dört estimatörü karşılaştırıyoruz
lerin patlamasını keser); TSVD yalnız en büyük k (Tablo 1). “y-kor”, geri çatılan desenin gerçek desenle
tekil değeri tutup gerisini sıfırlar. İkisi de gürültüyü korelasyonudur (1 = biçim korunmuş, 0 = bozulmuş).
azaltır ama bunu kestirimi bozarak (bias) yapar:              İki ders: (a) düzenlileştirme gürültüyü ∼35×
zayıf-gözlenen modlar atıldığından geri çatılan desen bastırır (1865 →∼ 52 µm) ama hedefin hâlâ 5×
gerçeğinden sapar. En iyi ayarda bile ∼ 50 µm’de kalır üstünde, ve bunu biçimi bozarak yapar (korelasyon
(§3.1) — 10 m hedefin çok üstünde.                         0.998 → 0.35: RMS düşer çünkü estimatör zayıf mod-
                                                           ları “0 tahmin et”meye kayar). (b) Direct R−1 (3.5 m)
(iv) Drift modu (kalibrasyon-referans) — öner- burada en iyi görünse de operasyonel rakip değildir
ilen. Tek gradient, iki zaman. t0 ’da referans y0 = — ofset=0 varsaydığından gerçek halkada (b ̸= 0)


                                                         3
Table 1: Test 1: ofsetsiz ideal şartta dört estimatör.
“y-kor” biçim korunumudur.
Estimatör               y-RMS     y-kor    Açıklama
                          [µm]
Direct R−1 (ofsetsiz)      3.5    0.998    yalnız referans (ofset=0 tabanı)
Ham ∆R−1 (ii)             1865    0.085    düzenlileştirmesiz: gürültü patlar
Tikhonov (iii)              53    0.348    λ dengeli; biçim kayıp
TSVD (k=3, iii)             52    0.383    en iyi 3 mod (k ideal seçilmiş)



200+ µm’e fırlar (Test 4); tabloda sadece gürültü ta-
banı referansıdır. Yani hiçbir iki-gradient varyantı
yetmiyor; çözüm farklı bir problem kurmaktan (drift
modu) geçer.


3.2    Test 2 — Uzaysal transfer fonksiy-
       onu ve SVD spektrumu
Bu test, (ii)’nin neden çuvalladığını R ile ∆R’nin
singüler-değer spektrumlarını yan yana koyarak gös-
terir (Şekil 2). ∆R’nin tekil değerleri R’ninkilerin
kabaca ε katı çıkar (empirik: ε = 0.02’de büyük mod-
larda oran ∼ 0.02). Bir incelik: en küçük birkaç
mod bu ε çizgisinin de altına “çöker” — yani
∆R = R(g1 ) − R(g2 ) farkında bazı modlar için R(g1 )
ile R(g2 ) neredeyse birbirini götürür; o modların tekil
değeri εσ(R)’nin altına iner (uniform ε ölçeklemesin-              Figure 2: Tepki matrisi R ve iki-gradient farkı ∆R =
den hızlı küçülür). Sonuç: σmin (∆R) ekstra küçük,                 R(g1 )−R(g1 (1+ε)) (ε = 0.02) için singüler-değer spek-
κ(∆R) daha da büyük (∼ 104 , R’den ∼2 mertebe                      trumları; (a) dikey, (b) yatay düzlem. Kesikli gri çizgi
kötü). (Düzenlileştirilmiş estimatörler yüksek-k mod-              ε σ(R) beklenen bulk ölçeklemesidir. ∆R’nin büyük
ları söndürür: 48 modun yalnız 3–5’i geri çatılabilir;             singüler değerleri bu çizgiyi izler; en küçük modlar
biçim bozulmasının kaynağı budur.)                                 daha da çöker. R iyi koşulludur (κ ≈ 193/140), ∆R
   ε taraması (Şekil 3) asıl ölçeklemeyi netleştirir: ofset-       ise ∼2 mertebe kötüdür.
iptal eden estimatörün gürültü büyütmesi ∥∆R−1 ∥ =
1/σmin (∆R) tüm ε ∈ [0.005, 0.10] aralığında temiz
biçimde ∝ 1/ε patlarken, kondisyon sayısı κ(∆R)
kabaca sabit (∼ 104 ) kalır. Yani ε → 0 sınırında yön-              Table 2: Test 4: mutlak vs drift modu (RMS hata).
temi kullanılamaz kılan κ değil, ∥∆R−1 ∥’dir — bu,
                                                                         Yöntem                Düzlem y   Düzlem x
§2.4’teki alt-sınır önermesinin (∥∆R−1 ∥ ∼ ∥R−1 ∥/ε)
                                                                                   −1
doğrudan sayısal doğrulamasıdır.                                         Mutlak R y(t)         ∼197 µm     ∼185 µm
                                                                         Drift R−1 (y − y0 )   6.6 µm      6.5 µm
                                                                         İyileşme                29×         28×
3.3    Test 3 — Yatay model ters-suç kon-
       trolü
Kx,arc kalibrasyonu simülasyondan alınır (klasik ters-
suç). Kx,arc (1 + δ), δ ∈ [−10%, +10%] taramasında 3.5 Test 5 — BPM ofseti zamanla ka-
yatay RMS 3.48–4.01 µm (0.5 µm değişim); dikey sabit           yarsa?
(Maxwell). Gerçek halkada LOCO’nun sağladığı < 1%
doğrulukta yöntem operasyonel olarak ters-suçtan
bağımsız.                                               Drift modu b(t) ≈ b0 varsayar; ofset kayma hızı ḃ
                                                        arttıkça kestirime R−1 ḃ ∆t olarak sızar. Simülasyonda
3.4 Test          4     —      Drift     modunun drift modu ile per-epoch iki-gradient + 30-epoch orta-
        kalibrasyon-referansla gösterimi                lama karşılaştırıldığında, drift modunun üstünlüğünü
                                                        kaybettiği geçiş ∼ 2 µm/epoch ofset kayma hızın-
Senaryo. t = 0: 100 µm RMS hizalama + 50 µm RMS dadır. Bu eşiğin gerçek bir pEDM BPM sisteminde
BPM ofseti kaydedilir. t = 1..10: hizalama 10 µm RMS sağlanıp sağlanmadığı deneysel olarak doğrulan-
yavaşça kayar, ofset sabit. 50 µm RMS ofset yöntemi ması gereken bir gerekliliktir (§6); burada bir do-
etkilemez; geriye yalnızca gürültü-kaynaklı taban kalır nanım iddiası yapılmamakta, yalnızca yöntemin geçerli
(Şekil 4, Tablo 2).                                     kaldığı kayma-hızı bütçesi nicelenmektedir.


                                                               4
Figure 3: ε taraması: (a) ofset-iptal eden estimatörün       Figure 4: Test 4 kalibrasyon-referans drift izleme, (a)
gürültü büyütmesi ∥∆R−1 ∥ = 1/σmin (∆R), nok-                dikey, (b) yatay düzlem. Gerçek drift, kestirilen drift
talı 1/ε referansıyla — temiz 1/ε ölçeklemesi; (b)           ve naif mutlak rekonstrüksiyon hatası epoch’a karşı
kondisyon sayısı κ(∆R), ε’dan kabaca bağımsız (∼             (log ölçek). 50 µm BPM ofseti mutlak rekonstrüksiy-
104 ). ε → 0’da yöntemi bozan κ değil ∥∆R−1 ∥’dir.           onu ∼200 µm’de boğarken drift modu 6–7 µm RMS’te
                                                             izler.
3.6    Test 6 — Üç               yöntemin        adil
       karşılaştırması                                   kinin yalnız %0.33’ü; drift takip hatası değişmez
                                                         (6.27 µm). 1 mrad’a kadar (kuplaj %1.1) hata yine
Aynı veri, aynı bütçe (50 µm ofset, 1 µm gürültü, 10 µm sabit. Çapraz kuplaj, drift modunun ölçtüğü değişime
drift, 0.2 mrad tilt’ler); Tablo 4. Her iki ∆R inşası ∼(kuplaj)×(drift) ≈ 0.1 µm katkı verir — 6 µm ta-
da κ ∼ 104 mertebesinde kötü koşulludur (düzleme banının çok altında. Düzlem-ayrık model gerçekçi
bağlı ∼ 7 × 103 –4 × 104 ; bkz. Şekil 2). Sayısal R, quad tilt altında geçerlidir.
∆R yaklaşımını kurtarmaz — κ(∆R) matrisin nasıl
inşa edildiğine değil ε ve örgü fiziğine bağlıdır. Drift
modu yapısal olarak farklı bir problem (mutlak değil,
                                                         3.7 Test 8 — Örgü modeli hatası al-
değişim) çözdüğü için bu sınırın dışındadır.                    tında β-beating sağlamlığı
   Tablodaki x ve y sütunları bağımsızdır: bu çalış- Drift modu R−1 ’in doğruluğuna bağlıdır. Gerçek
madaki lineer model yatay ve dikey düzlemleri ayrık halkada LOCO + BBA sonrası ∼%1–5 β-beating ve faz
ele alır (skew kuplajı = kuadrupol dönmesi sıfır), hatası kalır. Nominal model Rnom ile gerçek makine
dolayısıyla bir x−y çapraz (kuplaj) terimi yoktur; her Rtrue (β, ϕ’de εβ bozunumu) arasında kasıtlı uyum-
düzlem kendi 48 × 48 sistemidir. (Sahte EDM’i süren suzluk; veri Rtrue , kestirim R−1 ile (15 tohum me-
                                                                                      nom
dx·dy kuplajı bu lineer yörünge modelinde değil, ikinci- dyanı, Tablo 5). LOCO sonrası tipik ∼%1 β-beating’de
derece bir spin etkisidir — bkz. §4.4.)

Peki gerçek quad tilt bu varsayımı kırar mı? Table 3: Quad tilt kaynaklı x − y kuplajı ve düzlem-
Kuadrupol dönmesi (tilt) gerçek bir skew bileşeni ayrık drift kurtarımına etkisi (izleyiciden kuplajlı R;
yaratır ve dx → dikey, dy → yatay yörünge çapraz 15-tohum medyanı). drift_quadtilt_sim.py
terimlerini açar. Bunu doğrudan test ettik: rastgele Quad tilt ∥R ∥/∥R ∥ y-takip [µm] x-takip [µm]
                                                                  yx     xx
quad tilt’lerle kuplajlı tam tepki matrisini (dört blok
Ryy , Ryx , Rxy , Rxx ) izleyiciden kurup, düzlem-ayrık 0          %0.00          6.27          5.95
                                                         0.2 mrad %0.33           6.27          5.95
monitörle (Ryy , Rxx ) drift kurtarımı yaptık (Tablo 3).
              −1     −1
                                                         1 mrad    %1.08          6.26          5.95
Gerçekçi 0.2 mrad tilt’te çapraz kuplaj diyagonal tep-


                                                         5
                   Table 4: Test 6: aynı senaryoda üç estimatörün yan yana karşılaştırması.
                       Estimatör           y-RMS [µm]     y-kor x-RMS [µm] x-kor
                         A: analitik ∆R             3282        −0.02        3756     0.09
                         B: drift modu              6.25         0.85        7.18     0.85
                         C: sayısal ∆R               980        −0.02        1357    −0.11



Table 5: Test 8: β-beating altında drift takip hatası.
          εβ   y [µm]    x [µm]   Yorum
          0%      5.98     5.88   gürültü tabanı
        0.5%      5.94     5.91   önemsiz
          1%     6.08     6.09    hedef altında
          2%      6.48     6.58   < 7 µm
          5%      8.57     9.13   < 10 µm
         10%     13.00    14.51   sınır aşılır



taban yalnızca 0.1 µm artar (5.98→6.08 µm); hedefe
3.9 µm marj kalır. Yöntem, standart-kalite LOCO’su
olan bir hızlandırıcıda operasyonel olarak kullanılabilir
(Şekil 5).
                                                         Figure 5: Test 8: örgü-modeli hatası (β-beating) al-
                                                         tında drift takip hatası, εβ ’ya karşı (15-tohum me-
İzleyici doğrulaması — ve focal length hatası dyanı). Kesik kırmızı çizgi 10 µm hedef, noktalı gri
nereye düşüyor. Yukarıdaki tarama β-beating’i çizgi LOCO-gerçekçi %1 seviyesi. %1’de hata 6.1 µm;
analitik olarak (β, ϕ doğrudan bozularak), yani be- %5’e kadar hedef altında.
lirtiyi modelliyordu. Bunu β-beating’in fiziksel kay-
nağıyla tekrarladık: kuadrupol odak-uzaklığı (fo-
cal length) hataları. Bir kuadrupolün odak uzak- 3.9 Test özeti
lığı 1/f = (G/Bρ) L gradyanla belirlenir; fraksiyonel Tablo 6 sistematik testlerin ana sayısını derler.
bir gradyan hatası δG/G doğrudan bir focal-length
hatasıdır. Yani focal-length hatasını β-beating’in içine
gömmedik — tersine, onu kaynak olarak verip sonu- 4             Gözlenebilirlik Sınırı
cunu izleyiciye ürettirdik. Bu yaklaşım focal hatanın
tam etkisini yakalar (yalnız β şekil bozulması değil, §3’te drift modunun bir gürültü tabanına (∼ 6–
tune kayması da dahil), çünkü hepsi Rnom ile Rtrue 7 m, Test 4) oturduğunu              √ gördük: 1 m BPM
uyumsuzluğuna yansır. Her kuadrupole bağımsız rast- gürültüsü R−1 ’den geçerken 2 ση ∥R−1 ∥ mertebesinde
gele δG/G verilip Rtrue izleyiciden kuruldu; monitör bir hataya büyür (§2.3-iv). Bu bölüm o tabanın
nominal Rnom−1
                 ile çalıştı. Gradyan (focal) hata RMS’i rastgele bir sayı olmadığını, doğrudan R’nin mod
0, 2%, 5% için drift takip hatası 6.17, 6.17, 6.25 µm yapısından geldiğini gösterir: 6–7 m tabana en büyük
(κ: 228 → 239 → 233). Gerçek focal-hata kay- katkı R’nin en küçük tekil değerli (en kötü koşullu)
naklı β-beating analitik taramadan bile daha sağlam modlarından gelir — ve bu modların simetrik desen-
çıktı; Test 8 tam parçacık dinamiğiyle doğrulandı. ler olduğunu göreceğiz. Yani §3’teki taban ile buradaki
(drift_betabeat_sim.py)                                  kör nokta aynı olgunun iki yüzüdür: monitör simetrik
                                                         driftleri zayıf çözer, bu da hem gürültü tabanını yük-
                                                         seltir hem de geçerlilik alanını antisimetrik driftle sınır-
3.8 Test 9 — BPM kazanç hataları                         lar. (Bu karakterizasyon tümüyle monitörün kendi
                                                         özelliğidir; fiziksel bir sistematik bütçesinden bağım-
Gerçek BPM’ler bir kazanç hatasıyla okur: yölç,i = sızdır.)
(1 + gi ) ygerçek,i + bi + gürültü, gi per-BPM kali-
brasyon hatası (tipik %1–2).            Drift modu za- 4.1 Hangi kaçıklık desenleri kapalı
man farkı aldığından sabit ofset g’den bağımsız
                                                                 yörüngede görünür?
yine iptal olur; geriye çarpımsal bir model uyum-
suzluğu kalır: δq   b = R−1 diag(1+g)R δq = δq + Önce kurulumu somutlaştıralım. Halka 24 FODO
R diag(g)R δq. Per-BPM g ∼ N (0, σg ) taraması hücresinden oluşur; her hücrede bir odaklayıcı
  −1

(Şekil 6): σg = 0, %1, %2, %5, %10 için takip kuadrupol (QF) ve bir dağıtıcı kuadrupol (QD) vardır
hatası 5.7, 6.0, 6.4, 8.6, 13.4 µm.     Tipik %2’de ta- — toplam 48 kuadrupol. Bir “hizalama hatası deseni”,
ban yalnız 0.6 µm artar; %5’e kadar hedef altında. 48 kuadrupolün dikey kaymalarını veren 48 sayılık bir
(drift_gain_sim.py)                                      vektördür (∆q ∈ R48 ). Bir kaymış kuadrupolün kapalı


                                                            6
                                       Table 6: Sayısal deneylerin özeti.
             Test   Soru                                  Ana sayı
              1     Düzenlileştirme ∆R’yi kurtarır mı?       1865 → 52 µm, ama direct 3.5 µm
              2     Düzenlileştirme nasıl çuvallıyor?        48 modun 43–45’i siliniyor
              3     Yatay modelde ters-suç?                  ±10% → <0.5 µm
              4     Drift modu ofseti tolere eder mi?        197 µm → 6.6 µm (29×)
              5     Ofset kayarsa?                           < 2 µm/epoch’a kadar üstün
              6     Aday yöntemler yan yana?                 ∆R: 1000–3700 µm, drift: 6–7 µm
              8     β-beating (gerçek focal hatası)?         %1→6.1 µm; %5→8.6 µm
              —     quad tilt / x–y kuplajı?                 0.2 mrad: kuplaj %0.33, takip değişmez
              9     BPM kazanç hatası?                       %2→6.4 µm; %5→8.6 µm


                                                            öyle ki qQF,c = sc + dc ve qQD,c = sc − dc . Her gerçek
                                                            kaçıklık — genelde ne saf simetrik ne saf antisimetrik
                                                            — bu iki bileşene tek türlü ayrılır. Örnek: bir hü-
                                                            crede QF = 3, QD = 1 ise s = 2, d = 1; yani kayma
                                                            (2, 2)sim + (1, −1)antisim olarak yazılır. (48-boyutlu de-
                                                            sen uzayı böylece 24 simetrik + 24 antisimetrik genlik-
                                                            ten oluşan iki dik aileye ayrılır; “her hata = iki aileden
                                                            birer bileşen” — iki sabit desenin toplamı değil.)
                                                              Bir desenin ne kadar simetrik olduğunu tek bir
                                                            sayıyla ölçeriz:

                                                                              ∥s∥2 − ∥d∥2
                                                                         χ≡               ∈ [−1, +1],
                                                                              ∥s∥2 + ∥d∥2

                                                           χ = +1 tümüyle simetrik, χ = −1 tümüyle anti-
                                                           simetrik, χ ≈ 0 karışık. (Spin S, singüler değer σ veya
Figure 6: Test 9: BPM kazanç hatası RMS σg altında tepki R ile karışmasın diye χ harfi.)
drift takip hatası (30-tohum medyanı, analitik R). Ke-        Belirleyici ayrıntı gradyan işaretidir: QF odak-
sik kırmızı 10 µm hedef, noktalı gri tipik %2 seviyesi. lar (gradyan +), QD dağıtır (−); kick = gradyan ×
Sabit ofset gain’den bağımsız iptal olduğundan etki kayma olduğundan kick dizisi kj = g(−1) qj . Bu-
                                                                                                         j

yalnız çarpımsal model uyumsuzluğudur; %5’e kadar          radan:
hedef altında.
                                                             • Antisimetrik bileşen (+a/ − a): kickler aynı
                                                                işarette → düzgün, düşük-k → Gk büyük →
yörüngeye etkisi bir kicktir (küçük açısal sapma) ve            görünür.
büyüklüğü (gradyan × kayma) ile orantılıdır.
   Bir kick dizisinin kapalı yörüngeye toplam etk-           • Simetrik bileşen (+a/ + a): kickler zıt işarette
isi, kick deseninin azimutal harmoniğine — halka                → hücre-içi alternatif, yüksek-k → Gk küçük →
çevresi boyunca kaç kez tekrarladığına, k — bağlıdır.           neredeyse görünmez.
Kapalı yörünge harmonik k’ya bir kazançla yanıt verir:
                                                           (−1)j çarpanı (Nyquist harmoniği, k = 24) spektrumu
                                                       (5) 24 kaydırır: antisimetrik bileşen k ∈ [0, 12], simetrik
                  C
     Gk =                  , C ≈ 24.8, Q2eff ≈ 5.03,
            |Q2eff − k 2 |                                 bileşen k ∈ [12, 24] bandına düşer (üniform simetrik
                                                           desen k = 24 özel durumu). Tümü k ≫ Q olduğun-
burada Q betatron tune’udur (≈ 2.7); sabitler bu
                                                           dan simetrik içerik bastırılır. Sezgi: simetrik kayma
örgüye özgüdür. Kazanç tune’a yakın (k ≈ Q) har-
                                                           kendi kendini söndüren bir kick örüntüsü üretir; kapalı
moniklerde en büyük, k ≫ Q harmoniklerde küçüktür.
                                                           yörünge bunu görmez, antisimetriği güçlü görür.
Yani kapalı yörünge bir mod-seçici (rezonant)
filtredir: yalnız belirli uzaysal harmonikleri güçlü gös-
terir. (Bu klasik bir alçak-geçiren filtre değildir — tepe 4.3 Hangi desenler ne kadar gürültüyle
k = 0’da değil k ≈ Q’dadır.)                                       kestirilir? (per-mod SVD)
                                                            §4.2’deki simetrik/antisimetrik ayrım kategorikti (iki
4.2    İki tür hizalama deseni: simetrik ve uç durum). Tepki matrisinin tekil değer ayrışımı
       antisimetrik                         (SVD) bunu nicel ve sürekli hale getirir. R = U ΣV ⊤
Hangi desenin hangi harmoniğe düştüğünü görmek için yazıldığında her sağ-tekil vektör Vi bir hizalama de-
her hücredeki (QF, QD) çiftini iki bileşene ayırırız:     seni (“mod”), karşılık gelen tekil değer σi ise o des-
                                                          enin yörüngede ne kadar güçlü göründüğüdür; monitör
sc = 12 (qQF,c +qQD,c ) (simetrik),  dc = 12 (qQF,c −qQD,co) (antisimetrik),
                                                              modu kestirirken gürültüyü 1/σi ile büyütür. Her


                                                        7
Table 7: Seçili singüler modlar: gürültü duyarlılığı ve
simetrik içerik.
              Mod        σ    1/σ    Sim. içerik χi
         0 (en iyi)    28.4   0.04             %4
                  2    10.1   0.10             %6
                10      2.0   0.51            %13
                20     0.49   2.03            %42
                40     0.16   6.39            %91
      47 (en kötü)    0.147   6.82            %98



mod için gürültü büyütmesi 1/σi ve modun §4.2 an-
lamında simetrik içeriği χi hesaplanır (σmax = 28.4,
σmin = 0.147, κ(R) = 193; Tablo 7). En iyi 8
mod ortalama %4 simetrik (antisimetrik baskın, iyi
koşullu); en kötü 8 mod ortalama %96 simetrik. En
kötü modun (σmin ) koşulluluk-kaynaklı duyarlılığı en          Figure 7: R’nin (dikey düzlem) singüler modları: sol
iyiye göre ∼ 193 kat (= κ(R)) daha kötüdür; bu fark            eksen (mavi) gürültü duyarlılığı 1/σi , sağ eksen (kır-
tümüyle simetrik alt-uzayda yoğunlaşır. (Bu, “tüm              mızı) modun simetrik alt-uzaydaki güç oranı. En kötü
simetrik modlar 193 kat kötü” demek değildir; tablo-           koşullanmış modlar (sağ) %96–98 simetriktir; gürültü
daki 1/σ sütunu mod-mod artışı gösterir.) Bu ilişki            duyarlılığı arttıkça simetrik içerik %100’e tırmanır.
Şekil 7’te açıkça görülür: gürültü duyarlılığı 1/σ art-
tıkça simetrik içerik χi +1’e tırmanır.
                                                               panmaz; bu alt-uzaya erişim ilkesel olarak farklı bir
                                                               gözlemlenebilir (ör. karşı-dönen CW/CCW demet
Bu      sıralamanın       fiziksel    kökeni. Modların         ayrımı veya doğrudan spin presesyonu [1]) gerektirir.
simetrik içeriğinin σ küçüldükçe artması bir tesadüf
                                                                  Bu kör noktanın belirli bir sistematik bütçe
değil, latisin periyodikliğinin doğrudan sonucudur.
                                                               için ne kadar önemli olduğu ayrı bir sorudur.
Halka 24 özdeş FODO hücresinden oluştuğundan
                                                               pEDM bağlamında ilgili sistematik, kaçıklığın üret-
R (yaklaşık) periyodiktir; bunun cebirsel sonucu,
                                                               tiği sahte EDM’dir; baskın kanalı yatay×dikey kaçık-
singüler modlarının uzaysal harmonikler olmasıdır
                                                               lığın çarpımına (dx · dy) bağlı, misalignment’ta ikinci-
(sayısal olarak modların ≳ %96’sı tek bir harmonikten
                                                               dereceden bir geometrik-faz etkisidir [1]. Bu kanalın
ibaret). Böylece her modu bir kick-harmoniği k ile
                                                               ağırlığının monitörün zayıf-gözlediği simetrik alt-uzaya
etiketleyebiliriz. Kapalı yörüngenin harmonik-k bir
                                                               ne kadar düştüğü desen-bağımlı, inceliklidir ve tek
kick’e tepkisi Hill denkleminin rezonans paydasıyla
                                                               bir kapalı-yörünge ölçümüyle çözülemez. Bu nedenle
ölçeklenir, σ ∼ 1/|Q2eff − k 2 | (§4.2’deki kazanç
                                                               sahte-EDM ↔ gözlenebilirlik bağını niceleme işini
yasası Gk ile aynı iskelet): betatron tune’una yakın
                                                               — tam parçacık + spin izleyicisiyle, orbit-düzeltme
(k ≈ Q ≈ 2.3) kick’ler yükseltilir (büyük σ), uzak
                                                               öncesi/sonrası ayrımıyla — bu çalışmanın kapsamı
(k ≫ Q) olanlar bastırılır (küçük σ). Simetri buraya
                                                               dışında, ayrı bir incelemeye bırakıyoruz. Bu
şöyle bağlanır: hücre içinde aynı yönlü (simetrik)
                                                               makalenin iddiası yöntem düzeyinde kalır: monitör,
bir kaçıklık, odaklayıcı/odaksızlaştırıcı kuadrupol
                                                               antisimetrik hizalama driftini ucuza ve sürekli izleyen,
diziliminin (−1)j işaret değişimi nedeniyle alternatif
                                                               geçerlilik alanı ve kör noktası net tanımlı bir araçtır.
bir kick (yüksek-k, k → 24) üretir — tune’dan uzak,
dolayısıyla küçük σ ve yüksek gürültü. Antisimetrik
kaçıklık ise düşük-k (tune’a yakın) bir kick verdiğinden 5      Önerilen İşletme Modu
büyük σ ile iyi ölçülür. Şekil 7’teki monoton tırmanışın
mekanizması budur. (Bu ölçekleme periyodiklikten Tek bir yöntem her ihtiyacı karşılamaz; iki katmanlı
gelen yapısal
         √     bir özelliktir; mod-mod düzeyinde σ/Gk bir mimari öneriyoruz (Şekil 8).
oranı,      β ve KL ağırlıkları yüzünden ∼1.2–3.4
arasında değişir, yani kesin bir eşitlik değil, doğru bir
iskelettir.)                                              Yavaş mutlak katman (saatlik–günlük). LOCO
                                                          [4] + BBA + survey ile mutlak hizalama ∆q0 , BPM
                                                          ofseti b0 ve örgü modeli (β, ϕ, Q, dolayısıyla R). Bu,
4.4 Kör noktanın anlamı ve sistematik hızlandırıcı operasyonunda zaten var olan standart
        bütçeyle ilişkisi (ileri bakış)                   prosedürdür.
§4.1–4.3’ün sonucu yöntem düzeyinde kesin bir
ifadedir: kapalı-yörünge drift monitörü antisimetrik           Hızlı drift katmanı (sürekli, saniye–dakika).
hizalama driftini güçlü, simetrik driftini ise ∼ 193 kat       Bu çalışmanın katkısı: δq(t)
                                                                                       b     = R−1 (y(t) − y0 ). Fizik
daha gürültülü çözer. Bu, monitörün geçerlilik alanını         run’ı boyunca sürekli çalışır, veri toplamayı bozmaz,
(antisimetrik) ve kör noktasını (simetrik) tanımlar ve         kalibrasyondan beri (antisimetrik) hizalama değişimini
daha iyi BPM donanımı ya da daha fazla veriyle ka-             izler.


                                                           8
                                                              mayan yapısal nedenlerle erişimi farklı bir gözlem-
                                                              lenebilir gerektirir. Bu kör noktanın belirli bir sis-
                                                              tematik bütçe (ör. kaçıklığın ürettiği sahte EDM) için
                                                              önemi makineye bağlı, ayrı bir sorudur ve bu çalış-
                                                              manın kapsamı dışındadır (§4.4).
                                                                 Sağlamlık, başlıca model-hata kanalları için açıkça
                                                              test edildi: BPM kazanç hataları (Test 9, %2→6.4 µm),
                                                              kuadrupol tilt’inin yarattığı x–y skew kuplajı (Tablo 3,
                                                              0.2 mrad’da etkisiz) ve β-beating / focal-uzunluk hata-
                                                              ları (Test 8). Üçünde de yöntem hedefin altında kalır.
                                                                 Çalışmanın kalan kısıtları:
                                                               • Model ötesi diğer etkiler. Sonuçlar Eş. (1)
                                                                 lineer modeline dayanır; test edilmeyen ikin-
                                                                 cil kanallar arasında sekstupol feed-down, fringe
                                                                 alanlar, manyetik histerezis ve akım dalgalanması
                                                                 var. Bunlar gelecek çalışmadır; ama test edilen-
                                                                 ler (gain, skew-kuplaj, β-beating) bu sınıfın baskın
                                                                 üyeleridir.
                                                               • Tek kafes. Tüm sonuçlar 24-hücreli FODO’da.
                                                                 Genellenebilirlik kazanç yasası Gk üzerinden tah-
                                                                 min edilebilir ama doğrulanmamıştır.
                                                               • BPM ofset kararlılığı operasyonel bir
                                                                 gerekliliktir. Yöntem BPM ofsetine duyarsız
                                                                 değildir; ofsetin hızına duyarsızdır. Drift modu
                                                                 b(t) ≈ b0 varsayar ve geçerliliği bir kayma-hızı
                                                                 bütçesine bağlıdır: ḃ ≲ 2 µm/epoch (§3.5). Bu
                                                                 eşiğin pEDM BPM donanımında saatler–günler
                                                                 ölçeğinde sağlanıp sağlanmadığı deneysel olarak
Figure 8: İki-katmanlı hizalama izleme mimarisi: yavaş           karakterize edilmelidir; yöntemin pratik kullanıla-
mutlak katman (LOCO/BBA → ∆q0 , b0 , R) ve hızlı                 bilirliğinin belirleyicisidir.
drift katmanı (R−1 (y(t) − y0 )). Simetrik (sahte-EDM-
kritik) alt-uzay her iki katmanda da gürültü sınırın-
dadır.                                                        A     Sayısal simülasyon altyapısı
                                                              Tüm sonuçlar Python tabanlı bir altyapıda üretildi.
   İki katman tamamlayıcıdır: yavaş katman mutlak             Parçacık takibi dördüncü-mertebe Gauss-Legendre
referansı ve R’yi sağlar, hızlı katman o referansa göre       (GL4) semplektik integratörüyle; analitik tepki matrisi
değişimi takip eder. §4’ün sınırı gereği, her iki kat-        Courant-Snyder formalizmiyle (fodo_lattice.py)
man da simetrik alt-uzayı kapalı yörüngeden kurtara-          inşa edildi. Ana gösterim drift_monitor_sim.py
maz; bu bilgi dış bir gözlemlenebilirden (demet ayrımı        (Test 4), β-beating sağlamlığı test8_betabeat.py
/ spin) gelmelidir.                                           (Test 8), per-mod SVD analizi permode2.py
                                                              içindedir.   Şekiller make_figures.py (Şekil 1–4,
                                                              6) ve make_fig5_architecture.py (Şekil 5) ile
6    Tartışma ve Sonuç                                        üretilir.

Bu çalışmada pEDM AG halkasında kuadrupol hiza-
lama driftini sürekli izlemek için kalibrasyon-referans References
yöntemi önerildi ve gerçekçi BPM hata bütçesi al-
                                                           [1] Z. Omarov, H. Davoudiasl, S. Hacıömeroğlu,
tında (50 µm ofset, 1 µm gürültü, 0.2 mrad tilt) 6–7 µm
                                                               V. Lebedev, W. M. Morse, Y. K. Semertzidis,
RMS hassasiyetle, ∼%1 β-beating’e dayanıklı şekilde
                                                               A. J. Silenko, E. J. Stephenson, and R. Suleiman,
çalıştığı sistematik testlerle gösterildi. Yöntem iki kat-
                                                               “Comprehensive symmetric-hybrid ring de-
manlı bir mimaride (yavaş LOCO/BBA + hızlı drift)
                                                               sign for a proton EDM experiment at be-
operasyonel olarak konumlandırıldı.
                                                               low 10−29 e·cm,” Phys. Rev. D 105, 032001
   Yöntemin gözlenebilirlik sınırı da nicelendi (§4):
                                                               (2022),         doi:10.1103/PhysRevD.105.032001;
per-mod SVD analizi, monitörün kör noktasının %96
                                                               arXiv:2007.10332.
simetrik içerikli en kötü koşullanmış modlar olduğunu
gösterir (en kötü modda en iyiye göre ∼ 193 kat du- [2] V. Anastassopoulos et al., “A storage ring ex-
yarlılık dezavantajı). Yani monitör antisimetrik hiza-         periment to detect a proton electric dipole mo-
lama driftini hassasça izler; simetrik drift yöntemin          ment,” Rev. Sci. Instrum. 87, 115116 (2016),
geçerlilik alanı dışındadır ve donanım/veriyle kapan-          doi:10.1063/1.4967465; arXiv:1502.04317.

                                                          9
[3] S. Y. Lee, Accelerator Physics, 3rd ed. (World Sci-
    entific, 2011), doi:10.1142/8335.
[4] J. Safranek, “Experimental determination of stor-
    age ring optics using orbit response measure-
    ments,” Nucl. Instrum. Meth. A 388, 27 (1997),
    doi:10.1016/S0168-9002(97)00309-4.




                                                      10
