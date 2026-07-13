# Makale Revizyon Notları (makale_orbit_bastirma)

> **Amaç:** Kullanıcı yorumlarını burada topla, her birini tartış/karara bağla,
> SONRA hepsini tek seferde uygula (ekonomik). Makale ŞU AN değiştirilmiyor.
> **Durum:** çok-seed pipeline + ek testler arka planda dönüyor; kesilmeyecek.

Tarih: 2026-07-13

---

## A. Kullanıcı yorumları ve değerlendirme

### A1. Abstract çok uzun → kısalt
**Yorum:** Abstract çok uzun, o kadar detay gerekmiyor.
**Değerlendirme:** Katılıyorum. Şu an ~1 tam paragraf, çok sayıda spesifik sayı
(474×, 62×, 7.7×, lock-in, SQUID, ...) içeriyor. Bunlar gövdeye ait.
**Önerilen değişiklik:** ~130–160 kelimeye indir. Yalnız hikâye yayı kalsın:
(1) sahte-EDM problemi + σ² + ~1000× hedef; (2) yörünge düzeltme tek başına
simetrik moda takılıyor; (3) sim/antisim mekanizması (bir cümle); (4) BBA +
yörünge düzeltme birlikte tabanı kırıyor; (5) dürüst kısıt (tek-seed, donanım
tabanı). Spesifik ara-sayıları at.

### A2. Makalenin anlatı yapısını değiştir (BÜYÜK — öncelik)
**Yorum:** Şöyle bir akış daha iyi olmaz mı: "Şöyle bir sahte-EDM problemimiz
var → çözmek için yörünge düzeltmeye başvurmak istiyoruz → hızlandırıcılarda
yörünge düzeltme şöyle yapılır → AMA fark ettik ki yörünge düzeltme tek başına
sahte-EDM'i bastırmayı garantilemiyor → çünkü daha önce farkında olmadığımız
simetrik/antisimetrik mod gerçeği var, mekanizma şöyle → ama şöyle yapınca
bastırabiliyoruz, çünkü şu mekanizma."
**Değerlendirme:** Bence kesinlikle daha iyi. Mevcut yapı "başarısız yöntemler
kataloğu"nu (matris tersleme, K-mod, NN, drift, harmonik, CR-ayrım) merkeze
koyuyor; asıl mesaj (mekanizma + çözüm) sona gömülü. Kullanıcının önerdiği yay
daha güçlü ve okunur: **problem → yörünge düzeltme (standart) → keşif: tek başına
yetmiyor → sim/antisim mekanizması (bilineer kanallar) → çözüm: BBA + yörünge
düzeltme → neden çalışıyor.**
**Önerilen değişiklik (yeniden kurgu iskeleti):**
1. Giriş: pEDM + frozen-spin + sahte-EDM (kısa).
2. Yörünge düzeltme: neden doğal araç; hızlandırıcılarda nasıl yapılır (standart,
   kısa özet — burada ilgili literatür AKICI prose olarak, bkz. A5).
3. **Keşif:** yörünge düzeltme tek başına sahte-EDM'i garantilemiyor → simetrik
   mod yörünge-kör kalıyor.
4. **Mekanizma:** sim/antisim ayrışımı + bilineer dört kanal (şu anki §II.D
   matematiği burada merkezî). Buradan "tek RMS parametresi yetmez" çıkarımı (A4).
5. **Çözüm:** null-BBA simetrik kanalı ölçer (tersleme değil, sıfır-geçişi) +
   yörünge düzeltme antisimetriği siler → birlikte hedef.
6. Başarısız yöntemler → **kısa bir "neden naif tersleme/genlik-okuma yetmez"
   bölümüne SIKIŞTIR** (şu anki uzun katalog yerine). Merkez artık mekanizma+çözüm.
7. Operasyon + zamansal bütçe (yeni, bkz. B) + dürüst kısıtlar.
**Not:** Bu, en büyük değişiklik. Kullanıcı onayıyla iskeleti kesinleştirip
uygulayalım. Başarısız-yöntem kataloğunu tamamen atmıyoruz, kısaltıyoruz
(hakemler "diğer yöntemler neden olmuyor" diye soracaktır).

### A3. "Either rotation alone is nearly harmless..." → "precession" + fizik
**Yorum:** "rotation" yerine "precession" demek daha doğru; çünkü spin yatay
düzlemde dönerse (in-plane presesyon) gerçek EDM sinyali de birikmez.
**Değerlendirme:** Haklı. Doğru fizik: dikey ofset dy → radyal B_x → **radyal
eksen etrafında presesyon** (spini düzlem-dışına eğebilir); yatay ofset dx →
dikey B_y → **dikey eksen etrafında presesyon = düzlem-İÇİ** (S_y biriktirmez,
tam da frozen-spin düzlemi). "Rotation" gevşek; "precession" doğru terim.
**Önerilen değişiklik:** "rotation" → "precession". Ve cümleyi netleştir: dikey
eksen etrafındaki presesyon düzlem-içidir (S_y'ye katkısız); radyal eksen
etrafındaki tek başına halka boyunca ortalanır. Tehlike, ikisinin
komüte-etmemesinden doğan net dikey presesyon. (Kesin ifadeyi uygularken özenle.)

### A4. "Tek bir RMS parametresiyle ifade edilemez" çıkarımı eksik mi?
**Yorum:** Sim/antisim modların farkına vardıktan sonra, olayın rms-misalignment
gibi TEK parametreyle ifade edilemeyeceğini anlamadık mı?
**Değerlendirme:** Evet, ve bu AÇIKÇA vurgulanmalı — önemli kavramsal mesaj.
Sahte-EDM bilineer bir fonksiyonel (dört kanal); σ² yalnız GENLİK ölçeklemesi,
ama verilen bir RMS'te DEĞER desene (sim/antisim içerik + çapraz terimler)
bağlı. Yani "sahte-EDM = f(σ_rms)" YANLIŞ; doğru ifade "f = bilineer form(v_x,v_y),
σ²-homojen ama desen-bağımlı." Fig. kanal saçılımı (seed'e göre 3.5–900×) bunu
zaten gösteriyor.
**Önerilen değişiklik:** Mekanizma bölümüne (A2-4) net bir cümle/çıkarım ekle:
"Sahte-EDM tek bir hizalama-RMS sayısıyla ÖZETLENEMEZ; iki düzlemin
kaçıklık-desenlerinin bilineer bir fonksiyonelidir. σ² yalnız ölçeklemedir;
sabit σ'da değer sim/antisim içeriğe göre onlarca kat değişir." Bu, aynı zamanda
Omarov-tarzı "σ=10μm bütçesi"nin neden eksik olduğunu da açıklar.

### A5. "C. What is already known" tuhaf listeleme → prose
**Yorum:** Oradaki metin tuhaf bir listeleme olmuş; makale formatına uygun mu?
**Değerlendirme:** Haklı. Şu an \paragraph{Orbit correction}/{BBA}/{False-EDM
budgets}/{Neighboring work} şeklinde açıklamalı-kaynakça gibi; PRAB'de ilgili-iş
genelde giriş içinde AKICI prose olarak örülür.
**Önerilen değişiklik:** Bu alt-bölümü akıcı 1–2 paragrafa dönüştür ve A2'deki
yeni yapıda "yörünge düzeltme standart bir araçtır (kaynaklar...)" anlatısının
içine yedir. Ayrı bir "bilinenler listesi" olarak durmasın. "Neighboring work"
tek-cümle kaynak yığını da düzyazıya çekilsin ya da azaltılsın.

### A6. Method: "Analytic Twiss/CO solver" alt başlığı + "A methodological caution" → metne yedir
**Yorum:** "b. Analytic Twiss/closed-orbit solver" alt başlığını metodun uygulamasını
anlatırken metne yedir. "A methodological caution." başlığı tamamen gereksiz;
zaruri bir şey varsa onu da metne yedir.
**Değerlendirme:** Katılıyorum. İki-katmanlı araç anlatımı alt-başlıklara bölünmüş;
akıcı olmuyor. Analitik çözücü "hızlı taramalar için C++'ı doğrulayan ikinci,
bağımsız katman" olarak tek paragrafta anılabilir. "Methodological caution"
(idealize R en küçük tekil modu gizleyebilir) önemli ama tek cümleyle tersleme
tartışmasının içine girer. **Öneri:** alt-başlıkları kaldır, akıcı prose.

### A7. "4D closed orbit" — Newton iterasyonunu kısaca açıkla
**Yorum:** Newton iteration'ı kısaca açıkla.
**Değerlendirme:** Makul. **Öneri:** bir cümle: "kapalı yörünge, bir turluk harita
M'nin sabit noktasıdır (v* = M(v*)); (x,x',y,y') uzayında v_{n+1}=v_n−(M−I)⁻¹(M(v_n)−v_n)
Newton adımıyla birkaç iterasyonda bulunur." Jargonu az, sezgi net.

### A8. "A trap worth reporting" (ideal-eksen 4-parçacık tuzağı) — gerekli mi?
**Yorum:** Gerekli mi? Değilse at.
**Değerlendirme:** Katılıyorum, ATILABİLİR. Bu, bizim geliştirme sürecimizdeki bir
tuzaktı (ideal eksenden fırlatılan 4 parçacık CO betatron'unu söndürmez); okuyucu
için gerekli değil, estimator'ın doğruluğu zaten σ²/işaret testleriyle kanıtlı.
**Öneri:** çıkar (belki tek dipnot cümle kalır ya da hiç).

### A9. Genel ton: "yukarıdan bakan" değil, resmi-ama-pedagojik; yazar-okuyucu denk
**Yorum:** "Measuring the false EDM correctly" (ve genel metin) daha resmi ama
pedagojik olsun. Anlatım okuyucuya yukarıdan bakar gibi; yazarla okuyucuyu denk
tut, sadece okuyucunun bazı kavramlara aşina olmadığını unutma.
**Değerlendirme:** Çok haklı ve ÖNEMLİ. Şu an bazı yerler "the goal looks simple...
the difficulty is..." / "a trap worth reporting" gibi hafif tepeden. **Öneri:**
tüm metinde: (1) "we"/"one" dengeli, öğretici ama eşit; (2) "looks simple/naive/
trap/unforgiving" gibi dramatik-tepeden ifadeleri sadeleştir; (3) kavramları
tanımla ama "işte bu kadar basit" havası verme. Bu bir GENEL geçiş; her bölümde
uygulanacak.

### A10. Başta hybrid-symmetric örgü yapısını tanıt (KRİTİK — okuyucu için temel)
**Yorum:** Başlarda pEDM'de öngörülen hybrid-symmetric örgünün yapısını anlat:
FODO hücresi nasıl, deflektör/quad/drift nasıl yerleşiyor, elektrik mi manyetik mi.
Okuyucu komşu quad'lar arasında ELEKTRİKSEL deflektör + drift olduğunu anlasın.
**Değerlendirme:** Kesinlikle gerekli — makalenin geri kalanı (deflektör=drift model
hatası, yatay odaklama, sim/antisim) buna dayanıyor. Şu an örgü ancak Method'ta
dağınık geçiyor. **Öneri:** Giriş sonrası (ya da Method başında) kısa bir "The ring"
alt-bölümü: hücre dizilişi QF–drift–DEFL–drift–QD–drift–DEFL–drift–(QF...),
elektrik deflektörler (bükme + magic momentum), manyetik quad'lar (odaklama),
R0=95.49m, 24 hücre/48 quad, Q_y≈2.3. Bir küçük şema figürü bile düşünülebilir.
Bu, A2 yeniden-kurgunun doğal parçası.

### A11. Orbit gain formülü ile paragraf uyumlu mu? (kontrol edildi → uyumlu, ama netleştir)
**Yorum:** G_k = C/|Q²−k²| formülü ile onu anlatan paragraf gerçekten uyumlu mu?
**Değerlendirme (kontrol ettim):** Niteliksel olarak UYUMLU. Q²≈5 (Q≈2.24):
düşük-k (antisim, k=2→G≈24.8, k=3→G≈6.2) büyük yörünge; k≈24 (sim) → G≈0.043,
ezik. Yani "antisim düşük-k rezonansa yakın büyük, sim k≈24 uzak küçük" doğru.
**Ancak:** "115μm vs 1.7μm" oranı (~68) ile cond(R)=193 aynı şey değil (biri tek
mod genliği, diğeri koşullanma). Uygularken bu iki sayıyı karıştırmayacak şekilde
netleştir; C≈24.8 ve Q²≈5 değerlerini sayısal olarak bir kez daha doğrula.

### A12. "right-singular vector of R=UΣV^⊤ ... modes nearly purely antisym..." → dünyevileştir
**Yorum:** Bu cümle deneysel fizikçiler için korkutucu; daha dünyevi anlat :-)
**Değerlendirme:** Haklı. **Öneri:** SVD jargonunu düşür: "Tepki matrisini
inceleyince, demetin yörüngesini GÜÇLÜ bozan kaçıklık desenleri neredeyse tümüyle
antisimetrik; yörüngeyi neredeyse hiç bozmayan (görünmez) desenler ise simetrik
çıkıyor. İkisi arasındaki görünürlük farkı ~193 kat." Tekil-vektör/UΣVᵀ dilini
ya kaldır ya dipnota al.

### A13. Şekil 1 ↔ Şekil 3 çok benziyor + "sahte EDM vs RMS hata" iyi gösterge mi?
**Yorum:** Şekil 1 (σ² doğrulama) ve Şekil 3 (bastırma eğrisi) çok benziyor. Ayrıca
sim/antisim örüntü olduğu için "RMS hataya göre sahte EDM" ne kadar iyi gösterge?
**Değerlendirme:** Çok yerinde — hem A4 hem A27 ile bağlı. İki figür de "sahte-EDM
vs σ, log-log, σ² eğim" gösteriyor → gereksiz tekrar. Ve asıl mesele: **RMS tek
başına gösterge DEĞİL** (bilineer, desene bağlı). **Öneri:** (1) σ² doğrulamasını
tek figürde birleştir (validation + suppression aynı eksende ya da biri kalksın);
(2) figür/metinde açıkça "σ yalnız ölçekleme; verilen σ'da değer sim/antisim
içeriğe göre onlarca kat değişir" de (kanal figürü bunu gösteriyor). Yani RMS-eksenli
grafiği koru AMA "bu bir banttır, tek eğri değil; genişliği desen-bağımlılığıdır"
mesajını ver.

### A14. Korelasyon final işte kullanılıyor mu? "correlation trap" başlığı net değil
**Yorum:** Değerlendirmede korelasyon kullanmıyoruz sanırım; kullanıyorsak sadece
tersleme-reconstruction'da. Final işte hiç kullanmıyoruz değil mi? "How success is
measured: the correlation trap" açık/net değil.
**Değerlendirme (kontrol ettim):** Doğru. Korelasyon metinde 64 yerde ama HEPSİ
tersleme/genlik-okuma yöntemlerinin eleştirisinde (lock-in corr 0.69, breathing
0.07, NN...). **Final BBA+yörünge-düzeltme boru hattında korelasyon YOK** — sahte-EDM
(fast_measure) ve bileşen-RMS kullanılıyor. **Öneri:** A2 ile başarısız-yöntem
kataloğu kısalınca "correlation trap" da küçülür → ayrı bir metrik bölümü olmaktan
çıkıp, tersleme eleştirisinin içinde tek uyarı cümlesine iner: "bu yöntemlerde
korelasyon aldatıcı yüksek çıkabilir (0.99), oysa simetrik bileşen tümüyle yanlış."
Final metriğimiz ise doğrudan sahte-EDM.

### A15. BPM gain error'u çalışmaya kattık mı? (kontrol → HAYIR, sadece tanımda)
**Yorum:** BPM gain error sistematiğini dahil ettik mi?
**Değerlendirme (kontrol ettim):** Sistematikler listesinde tanımlı (1–10%) ama
HİÇBİR sonuçta fiilen kullanılmıyor (ona özel bir koşum yok). **Öneri:** ya (a)
listeden çıkar / tek cümleye indir ("çok-BPM ortalamasıyla yumuşar, baskın değil"),
ya da (b) dürüstçe "modellenmedi, baskın sınır değil" de. Kullanılmayan bir
sistematiği "kattık" gibi sunmak dürüst değil → çıkarmaya meyilliyim.

### A16. Tablo 1 sayıları güncel değil (eski no-go çerçevesi)
**Yorum:** Tablo 1'deki sayılar güncel değil sanırım.
**Değerlendirme (kontrol ettim):** Haklı. Tablo 1 "raw ~1000× → CW/CCW 474× →
orbit 62× → herhangi tersleme yöntemi 62× (taban)" — bu ESKİ pesimist çerçeve.
İki sorun: (1) raw ~1000× ile BBA işindeki 356× (belirli seed) tutarsız; (2)
tablo BBA+yörünge-düzeltmenin hedefe indiğini İÇERMİYOR. **Öneri:** yeni çerçevede
(A2) tabloyu güncelle: raw → CW/CCW → orbit-corr (62×, simetrik taban) → **+BBA
(simetriği ölçer)** → **+son orbit-corr (antisim temizler) → hedef**. raw sayısını
tek referansla (356× ya da ensemble) tutarlı kıl.

### A17. Şekil 2 referansı geç/yok + grafikleri basitleştir
**Yorum:** Şekil 2'ye referans yok ya da çok sonra; kontrol et. O grafikleri
(anlatımı) basitleştirmek?
**Değerlendirme (kontrol ettim):** fig:modes (mod yapısı) tanım satır 427,
referans 324/327 — aslında referans var ve önce geliyor (float). Ama figür
SIRALAMASI karışık (kanal figürünü sigma'dan önce ekledim → PDF numaraları
kaydı). **Öneri:** tüm figür sırası+referanslarını A2 sonrası bir kerede gözden
geçir. Mod-yapısı figürünü (SVD sol panel + G_k sağ panel) basitleştir: sol
paneli A12'deki dünyevi dille "görünür=antisim / görünmez=sim" olarak sunan sade
bir çizime indir; iki panelden biri yeterli olabilir.

### A18. K-mod optics breathing + feed-down kick açıklaması net değil
**Yorum:** K-modülasyondaki optics breathing'i uygun yerde güzelce açıkla; feed-down
kick ve etkisi net değil; o iki paragraf anlaşılır değil.
**Değerlendirme:** Katılıyorum. **Öneri:** "breathing" için somut resim: bir quad'ın
gradyanını değiştirince (a) o quad'ın KENDİ kaçıklığından gelen dipol kick'i modüle
olur (aradığımız sinyal, ~0.9μm) AMA (b) tüm halkanın optiği (β, faz, Q) de modüle
olur → mevcut BÜYÜK yörünge (0.37mm, 48 quad'ın toplamı) bu değişen optikten yeniden
taşınır → BPM'de birkaç-μm "nefes". Sinyal/nefes ~1/7. Feed-down = quad-merkezinden
kaçık geçen demetin gördüğü dipol bileşen; bunu tek cümlede tanımla. İki paragrafı
tek net paragrafa indir, "266×" gibi çarpıcı tek sayıyı koru.

### A19. Şekil 4/6'daki sim/antisim ayrımı NASIL yapılıyor — uygun yerde açıkla
**Yorum:** Şekil 4 ve 6'daki sim/antisim kaynaklı sahte-EDM'in nasıl ayırt edildiğini
açıkla; belki başta "sim/antisim etkilerinin ayrıştırılması" alt-başlığında.
**Değerlendirme:** Evet — ve elimizde tam bu var (kanal ayrışımı: P_s/P_a ile
projekte edip her parçayı AYRI C++ izleyiciye verip f ölçüyoruz; fig_orbit_channels).
**Öneri:** Mekanizma bölümüne (A2-4) "How we separate the two channels" diye kısa
bir yer: "kaçıklığı sim/antisim parçalarına ayırıp her birini ayrı ayrı spin
izleyiciye veririz; ölçülen f'ler bilineer toplamı %0.1'de doğrular." Böylece
sonraki tüm sim/antisim sayıları (Fig crsep dahil) bu yönteme dayanır.

### A20. Bazı cümleler benim sorularımı cevaplıyor, normal okuyucuda o soru yok
**Yorum:** Bazı cümleler doğrudan benim takıldığım yerleri cevaplamak için yazılmış;
normal okuyucuda o sorular yok, sırıtıyor.
**Değerlendirme:** Çok haklı, ve iyi bir gözlem. Örn. "the natural question is then:
is the beam not itself a measuring device?", bazı "trap"/"caution" kutuları, "one
might hope..." savunmaları. **Öneri:** metni tarayıp bu "iç-diyalog" cümlelerini
ya kaldır ya nesnel önermelere çevir. Genel ilke: her cümle okuyucunun bilmesi
gerekeni söylesin, bizim keşif yolculuğumuzu değil.

### A21. NN backprop ≈ ters matris → NN argümanını zayıflat (YORUM, makaleye ekleme değil)
**Yorum:** NN'deki backpropagation aslında inverse matrix gibi değil mi? Bu özellik
kullanılarak NN daha verimli olamaz mı? (Yorum için soruyorum.) Açık sorular varsa
NN argümanını zayıflatmak iyi olabilir.
**Değerlendirme (senin sezgin doğru):** Lineer haritada (y=Rq) backprop, kayıp
fonksiyonunu en-küçük-kareye indirir → sonuç PSEUDO-İNVERS (R⁺). Yani lineer+Gauss
gürültü rejiminde NN, R⁺'yi yeniden öğrenmekten fazlasını yapamaz; sınır ESTIMATOR
değil, BİLGİ (SNR). Nonlineerlik/prior olsa NN kazanabilir ama bizim problem lineer.
**Sonuç:** senin dediğin gibi NN argümanını ZAYIFLATALIM — "NN asla yardım edemez"
DEME; "bu lineer, gürültü-sınırlı kuruluşta NN pseudo-inverse'e iner; sınır
estimator-bağımsız bir bilgi tabanıdır" DE. A2 ile başarısız-yöntem kataloğu
kısalınca NN zaten tek cümleye iner; bu da aşırı-iddiayı önler. (Verimli-NN açık
sorusu gerçek; onu kapatan iddia kurmayalım.)

### A22. "Harmonic approaches" — basitleştir, çok bilgi sıkışmış
**Yorum:** Anlatım çok yoğun; basitleştir.
**Değerlendirme:** Katılıyorum. **Öneri:** İki alt-fikri (bilinen-harmonik Fourier;
bilinen-imza fit) tek kısa paragrafa indir; sayı yığınını (13000→186, 53nT, 45nT,
2.6nT) azalt, tek temsili sayı bırak. A2 kataloğu-kısaltma ile uyumlu.

### A23. CR beams neden tam temizlemiyor — sezgisel anlat (quad sırasının tersi ≠ simetri)
**Yorum:** CR (karşı-dönen) demetlerin sahte-EDM'i neden tam temizlemediğini sezgisel
anlat. Bir FODO hücresindeki quad'ları TERS sıradan görmek, düz sıradan görmenin
simetrisi değil gibi. Bunu nasıl daha net görürüz?
**Değerlendirme (güzel fizik sorusu):** İki tamamlayıcı sezgi:
(1) **Komütatör pariteli:** geometrik faz ≈ Σ_{i<j}[θ_i,θ_j] (SIRALI kick
komütatörleri; dönmeler komüte etmez). Demet yönünü çevirince kick SIRASI ters
döner → Σ_{i<j} → Σ_{i>j} = −Σ_{i<j}; yani ideal-simetrik halkada geometrik faz
sıra-tersine ~TEK. Ama gerçek makinede kick'ler demetin OTURDUĞU yere (kapalı
yörüngeye) bağlı ve CW/CCW yörüngeleri tıpatıp aynı değil → iptal KISMİ (~3.4×),
tam değil. "Ters sıra = düz sıranın simetrisi" varsayımı bu yüzden yanlış.
(2) **Yörünge-körlüğü:** CR-AYRIMI bir yörünge gözlenebiliridir; tüm yörünge
gözlemleri gibi SİMETRİK kaçıklığa kör. CR-ayrımını sıfırlasan bile simetrik parça
sahte-EDM üretmeye devam eder. Fig crsep tam bunu ölçüyor (antisim/sim oranı
tek-demette ~5.7×, CR-ayrımında ~8× → ek pencere YOK).
**Öneri:** Bu iki cümleyi CR bölümüne koy. **Sim ile netleştirilebilir** (opsiyonel,
sonra): f'in (a) demet-ters, (b) quad-sıra-ters, (c) polarite-flip altındaki
paritesini ölçüp hangi simetriye uyduğunu göster (omarov Test 7 kısmen var).

### A24. "polarity switching...geometric phase" → "...quad alignment based geometric phase"
**Yorum:** Bu ifadeyi "...for the quad alignment based geometric phase" yap; çünkü
başka yerlerde polarite-flip işe yarıyor gibi.
**Değerlendirme:** Haklı, kapsam düzeltmesi. Polarite-flip g'de TEK olan
sistematikleri temizler; bizim quad-kaçıklık geometrik fazı g²'de (çift) olduğu
için onu temizlemez. **Öneri:** ifadeyi aynen önerildiği gibi daralt. (Aynı yerde
"geometrik faz g'de çift" argümanı zaten var; sadece cümleyi kapsamla.)

### A25. Yüksek-Q bölümünde k ve m'yi açıkla
**Yorum:** "raising Q toward k... the high-m part" — k ve m nedir açıkla.
**Değerlendirme:** Gerekli. **k** = kick dağılımının halka etrafındaki AZİMUTAL
harmoniği (G_k'daki k). **m** = simetrik kaçıklık ailesinin KENDİ içindeki harmonik
indeksi (simetrik desenlerin yavaş/hızlı değişenleri; "high-m" = hızlı değişen
simetrik alt-desenler, düşük-k'ya karşılık gelenler ilk açılır). **Öneri:** her
ikisini ilk geçişte bir yan-cümleyle tanımla.

### A26. Off-momentum paragrafını aç
**Yorum:** Biraz daha aç.
**Değerlendirme:** Makul (şu an çok sıkışık, tek paragrafta 6 kapatma-testi). **Öneri:**
mekanizmayı (magic-olmayan p → ν_s≠0 → geometrik faz ∝1/ν_s yükselir AMA 1.-mertebe
ISA-eğimi aynı frekansta ~3600× daha büyük) 2-3 cümleye yay; 6 test yerine en güçlü
2'sini anlat, gerisini "ve diğer ayırma kanalları da kapandı (δp/p, faz, 4-fold,
RMS, 3D fit, boylamsal)" diye özetle. Magic momentumun neden tek çalışan nokta
olduğunu vurgula.

### A27. ANA DERS: yörünge düzeltme rastgeleliği sim/antisim ÖRÜNTÜLERE çeviriyor
**Yorum:** Önemli ders: yörünge düzeltme yöntemleri sonucunda rastgeleliğin yerini
sim/antisim bazlarda ÖRÜNTÜLER alıyor; bunların sahte-EDM üzerindeki etkisi rastgele
hatalardan farklı bir örüntüye sahip.
**Değerlendirme:** Bu, makalenin ASIL KAVRAMSAL KATKISI olabilir — öne çıkar. A4
(tek-RMS yetmez) + A13 (RMS gösterge değil) + drift testi (rastgele drift'in sim/
antisim parçaları farklı davranıyor) hepsi buna bağlanıyor. **Öneri:** Sonuç ve
mekanizma bölümüne net tema cümlesi: "Düzeltme, sahte-EDM'i azaltmaktan çok onun
İSTATİSTİĞİNİ değiştirir: rastgele kaçıklık → yapılı sim/antisim artık; sim/antisim
kanalları farklı ölçeklenir (biri yörünge-görünür ve düzeltilebilir, biri kör), bu
yüzden 'ne kadar küçük' değil 'hangi kanalda' sorusu belirleyici." Bu, bütün
bulguları birleştiren bir ders.

---

## B. Birleşik operasyonun zamansal bütçesi (gerçekçi, varsayımlar İŞARETLİ)

> Amaç: BBA + yörünge düzeltme + CW/CCW + sürekli drift-düzeltme zincirinin
> gerçek deneyde zaman maliyeti. Simülasyon DUYARLILIK verir; MUTLAK süreler için
> makine-özel drift HIZI gerekir → ⚠️ ile işaretli varsayımlar.

**Simülasyondan gelen sağlam girdiler (C++):**
- Sürekli yörünge düzeltme, drift'in antisimetrik (baskın) parçasını siler:
  rastgele drift σ_d ≤ ~5 μm iken sahte-EDM hedefin ALTINDA kalır; 10 μm'de
  birkaç× (drift_cwccw_test: 2μm→0.05×, 5μm→0.3–0.6×, 10μm→3–6×, feedback'li).
- Feedback YOKSA drift hızla geri yükler (10μm → yüzlerce–binlerce×).
- BBA yalnız yavaş biriken SİMETRİK drift için gerekir (antisim'i orbit halleder).
- CW/CCW farkı ~2–3× ek (tek-seed; ensemble ile kesinleşecek).

**Zaman ölçekleri (kaba, ⚠️ varsayımlı):**
| adım | süre | kadans | not |
|---|---|---|---|
| İlk BBA (47 quad × 2 düzlem, iteratif) | ⚠️ ~saatler–1 gün | bir kez (kurulum) | her quad: grad-mod + yörünge yerleşme + ortalama |
| Yörünge düzeltme | saniye–dakika | **sürekli** | rutin; antisim drift'i temizler |
| BBA tekrarı | ⚠️ ~saatler | ⚠️ ~günde bir? | simetrik drift ~2μm'ye ulaşınca |
| CW/CCW | demet ters çevirme ek yükü | ölçüm boyunca | ~2× veri süresi ya da eşzamanlı |
| **EDM veri alımı (polarimetri)** | **⚠️ aylar–yıl** | — | nrad/s istatistiği; ASIL baskın süre |

**Kritik köprü (⚠️ makine verisi gerek):** simetrik drift hızı. Örnek varsayım:
toplam rastgele drift ~1 μm/gün (termal+zemin, stabilizasyon sonrası) →
simetrik parça ~0.7 μm/gün → ~3 günde ~2 μm → **~3 günde bir BBA tekrarı**.
BBA ~saatler sürerse, veri-alım görev-çevrimi kaybı ~%birkaç.

**Sonuç (dürüst):** Baskın süre EDM istatistiği (aylar–yıl). Hizalama BAKIMI
(sürekli orbit + periyodik BBA) küçük bir görev-çevrimi ek yükü — DRIFT HIZI
~μm/gün mertebesindeyse. Yani zaman-yasak DEĞİL; ama bu, drift hızının gerçekten
~μm/gün olmasına bağlı (yüksekse BBA kadansı sıkışır, kayıp artar). Simülasyon
duyarlılığı verdi; kadansı sabitlemek için makine drift ölçümü (ya da literatür
tahmini) şart. Ayrıca donanım tabanı (grad-mod sırasında manyetik merkez
oynaması) BBA'nın kendi tekrar-doğruluğunu sınırlayabilir — modellenmedi.

---

## C. Uygulama planı (onay sonrası, toplu)

**Sıra (bağımlılığa göre):**
1. **A2 iskeletini kesinleştir** (en büyük iş — anlatı yeniden-kurgu: problem →
   yörünge düzeltme → keşif → mekanizma → çözüm). Diğer her şey buna oturur.
2. **A10** örgü tanıtımı ("The ring" alt-bölümü) — mekanizmanın ön-koşulu.
3. **Mekanizma bölümü** (merkez): sim/antisim (A12 dünyevi dil) + bilineer dört
   kanal + **A19** kanal-ayırma yöntemi + **A4/A27** "tek-RMS yetmez / örüntü"
   çıkarımı + **A11** formül-paragraf tutarlılığı.
4. **Çözüm bölümü:** null-BBA + yörünge düzeltme (mevcut §III.G içeriği, sadeleşmiş).
5. **Başarısız-yöntemler → SIKIŞTIR** (A2): tersleme/K-mod/**A18** breathing-net/
   **A21** NN-zayıflat/**A22** harmonik-sadeleştir/**A14** korelasyon-tek-uyarı/
   **A23** CR-sezgisel/**A24** polarite-kapsam.
6. **Abstract kısalt** (A1) — gövde bittikten sonra.
7. **Zamansal bütçe** bölümü (B).
8. **Genel geçişler:** A9 ton (resmi-pedagojik, tepeden bakma yok), A20 iç-diyalog
   cümlelerini temizle, A8 gereksiz tuzak-kutusunu at, A6 method alt-başlıklarını
   yedir, A7 Newton bir cümle, A25 k/m tanımla, A26 off-momentum aç.
9. **Figürler** (A13/A17): σ²-doğrulama ve bastırma figürlerini birleştir/sadeleştir;
   mod figürünü dünyevileştir; figür sırası+referanslarını tek elden gözden geçir;
   Tablo 1'i (A16) yeni çerçeveyle güncelle; A15 gain-error'u çıkar/indir.
10. Çok-seed pipeline + (opsiyonel) ensemble CW/CCW sonuçları gelince sayıları
    kesinleştir.

**Açık/opsiyonel simülasyonlar (yorumlardan doğan, sonra):**
- A23: f'in demet-ters / quad-sıra-ters / polarite-flip paritesi (CR sezgisini
  sağlamlaştırmak için).
- Ensemble CW/CCW (doğru Omarov Eq.C1–C2 kombinasyonu; ~2–3× faktörünü kesinleştir).
- Zamansal bütçe için makine drift-hızı (literatür/varsayım → kadans).

## D. Bekleyen
- Kullanıcının ek yorumları (varsa) buraya.
- Onay: A2 iskeleti + genel ton kararı → sonra toplu uygulama.
