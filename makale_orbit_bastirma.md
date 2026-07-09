# Makale Taslağı — Proton EDM Deneyinde Sahte EDM Sinyalinin Yörünge Düzeltmesiyle Bastırılması ve Sınırları

> **Durum (2026-07): İSKELET TASLAK — tartışma/iterasyon için.** Bölüm yapısı ve
> her bölüme girecek sonuçlar (sayılar + kaynak belge) burada; düz metin (prose)
> henüz yazılmadı. Tüm sayılar repodaki doğrulanmış günlüklerden alınmıştır; her
> maddenin yanında kaynak belge/script referansı vardır. Dil şimdilik Türkçe;
> İngilizce çekirdek özet `PROJE_ANALIZI_VE_ONERILER.md §6.1`'de hazır.
>
> **Kapsam kararları (kullanıcı, 2026-07):**
> - Pozitif çerçeve: "kaç e·cm'e kadar bastırılabilir" belgelenir; no-go, eğrinin
>   doyma gerekçesi olarak içeride kalır.
> - **Spin-tabanlı null'lama ve NN ileri-harita (Kol B) KAPSAM DIŞI.** Spin için
>   gerekçe fiziksel: sahte/gerçek EDM aynı gözlenebilirde (dS_y/dt) dejenere —
>   sahte'yi null'lamak gerçeği de null'lar (`omarov.md §10`, `svd.md §9.1`).
>   İleri-harita için gerekçe editoryal: ayrı, olgunlaşmamış açık problem
>   (`akilli_duzeltme.md §6.10`); Conclusions'ta tek cümleyle "kapsam dışı" denir.
> - Off-momentum spin-sensing Results sonunda KISA alt bölüm (yörünge-dışı kaçış).
> - Metod'a giren sistematikler: **β-beat, quad tilt, BPM ofset/kazanç/gürültü**
>   (nihai sayılarda belirleyici olanlar).
> - **Uniform-gradient / zayıf-odaklama tartışması makaleye GİRMEZ** (kullanıcı).
>
> **ÜSLUP (kullanıcı talebi):** Makale, konuya aşina olmayan okur için **mümkün
> olduğunca basit ve açıklayıcı** yazılacak — her kavram ilk kullanımda tanımlanır
> (BPM, kapalı yörünge, tune, k-modülasyon, geometrik faz...), her sonuçtan önce
> "neyi neden denedik" sezgisi verilir, formüller sözle de anlatılır.
> `PROJE_ANALIZI_VE_ONERILER.md` ve `akilli_duzeltme_pedagojik.md` üslup şablonudur.

---

## 0. Özet (Abstract) — yazılacak

Çekirdek iddia (İngilizcesi `PROJE_ANALIZI §6.1`'de):
- Yörünge-tabanlı hizalama kontrolü, quad-kaynaklı sahte-EDM'i σ=10 μm toleransta
  **~6×10⁻²⁸ e·cm**'e (62× hedef) kadar bastırır; taban **σ² ile ölçeklenir**.
- Bu taban **BPM teknolojisinden bağımsızdır**: sınır beyaz gürültü değil,
  ortalamayla sönmeyen koherent sistematikler (optik-nefes; β-beat × koşullanma).
- 10⁻²⁹ e·cm'e inmek, yörüngeyle **doğrulanamayan** ≲1.3 μm simetrik
  manyetik-merkez hizalaması gerektirir (manyetik ≠ mekanik merkez).

---

## 1. Giriş (Introduction) — DÜZ METİN v0.1 (pedagojik)

### 1.1 Deney: protonun elektrik dipol momentini aramak

Bir parçacığın **elektrik dipol momenti (EDM)**, pozitif ve negatif yük
dağılımlarının parçacığın spin ekseni boyunca birbirinden çok küçük bir miktar
ayrışması demektir. Sıfırdan farklı bir EDM, zaman-tersinme (T) — dolayısıyla
CP — simetrisinin kırılmasını gerektirir. Evrende gözlediğimiz madde–antimadde
dengesizliği, Standart Model'in içerdiğinden çok daha güçlü CP-kırılması
kaynaklarına işaret ettiği için, proton EDM'ini Standart Model öngörüsünün
(~10⁻³¹ e·cm'nin altı) üstünde bir seviyede bulmak yeni fiziğin doğrudan kanıtı
olur. Önerilen deney [Omarov ve ark., PRD 105, 032001] duyarlılık hedefini
**d = 10⁻²⁹ e·cm** olarak koyar.

Ölçüm ilkesi **frozen-spin (donmuş spin)** yöntemidir. Protonlar, elektrik
alanla bükülen ve manyetik kuadrupollerle odaklanan bir depolama halkasında,
özel "magic" momentumda (p ≈ 0.7 GeV/c) döndürülür. Bu momentumda spin,
momentum vektörüne kilitli kalır — halka düzleminde ne öne geçer ne geri kalır.
Protonun bir EDM'i varsa, halkanın radyal elektrik alanı bu dipole tork uygular
ve spin **çok yavaşça halka düzleminden dışarı, dikeye döner**. Ölçülen
büyüklük bu dönme hızıdır:

$$f \equiv \frac{dS_y}{dt}, \qquad d = 10^{-29}\,e\!\cdot\!\text{cm}
\;\leftrightarrow\; f \approx 1\ \text{nrad/s}.$$

Saniyede bir milyarda-bir radyan: deneyin bütün zorluğu, spin yönünü bu
hassasiyette etkileyebilecek **başka her etkinin** ya yok edilmesi ya da
güvenle sınırlanması gereğinden doğar. EDM'i taklit eden böyle etkilere
**sahte EDM (false EDM)** denir; bu makalenin konusu, bunların en inatçısıdır.

### 1.2 Sahte EDM: hizalama hatalarının geometrik fazı

Halkadaki 48 kuadrupol mıknatısın her biri, ideal konumundan gerçekte birkaç
mikron kaymış olacaktır. Kuadrupolün merkezinde manyetik alan sıfırdır; demet
merkezden geçmezse, kaymayla orantılı bir alan görür. **Kaymanın yönü neyi
etkilediğini belirler:** dikey kayma dy demete radyal alan B_x gösterir ve
spin **x-ekseni** etrafında döner; yatay kayma dx dikey alan B_y gösterir ve
spin **y-ekseni** etrafında döner.

Tek başına bu dönmeler zararsızdır: halka boyunca ortalamaları büyük ölçüde
sıfırlanır. Tehlike, ikisinin **birlikte** var olmasından doğar. Farklı
eksenler etrafındaki dönmeler **sıra değiştirmez** (önce x sonra y etrafında
döndürmek, tersini yapmakla aynı sonucu vermez); aradaki fark, her turda
üçüncü eksende — tam da EDM'in görüneceği dikey dönme kanalında — küçük bir
net artık bırakır. Bu, fizikte iyi bilinen **geometrik (Berry) fazının** bir
örneğidir. Sonuç:

$$f_{\text{sahte}} \;\propto\; dx \cdot dy \;\;\Rightarrow\;\;
f_{\text{sahte}} \propto \sigma^2,$$

yani sahte EDM, kaçıklıkların RMS büyüklüğü σ'nın **karesiyle** ölçeklenir
(simülasyon doğrulaması: üs p = 2.00 ± 0.01, §2.2; Omarov Fig. 9a ile birebir).
Sayısal ölçek acımasızdır: σ = 10 μm'lik tipik bir hizalama, hedefin
**~1000 katı** sahte EDM üretir. Yani aranan sinyal, hizalama kaynaklı arka
planın bin kat altındadır ve bu arka plan, gerçek EDM ile **aynı
gözlenebilirde** (dS_y/dt) yaşar.

Bu noktada doğal soru şudur: *demetin kendisi bir ölçüm aleti değil midir?*
Halkadaki 48 konum monitörü (BPM), demetin kapalı yörüngesini sürekli okur;
kaçık kuadrupoller yörüngeyi bozar; öyleyse yörüngeden kaçıklıkları ölçüp
düzeltmek — hızlandırıcı dünyasının standart **beam-based alignment (BBA)**
pratiği — sahte EDM'i de temizlemeli. Bu makale, bu doğal beklentinin **nereye
kadar doğru olduğunu ve nerede, neden durduğunu** nicel olarak belirler.

### 1.3 Bilinenler: literatürde çözülmüş problemler ve belirlenmiş etkiler

Sorunun parçalarının çoğu, ayrı ayrı, iyi çözülmüş problemlerdir; bu çalışmanın
yaptığı, onları sahte-EDM sorusu etrafında birleştirmektir. (Karşılaştırmalar
tam metin üzerinden yapılmıştır; `literatur/`.)

- **Yörünge düzeltme ve gözlenebilirlik matematiği.** Tepki matrisinin
  tekil-değer ayrışımıyla (SVD) yörünge düzeltme, hangi hata desenlerinin
  yörüngede iyi/kötü görüneceği ("decoupled" modlar), küçük tekil değerlerin
  atılması (regularizasyon) — bu aygıtın tamamı Chung, Decker ve Evans'a
  (PAC 1993) kadar gider. Bizim kullandığımız harmonik kazanç yasası
  G_k = C/|Q²−k²| bu çerçevenin harmonik limitidir. Dejenere ters-problemlerde
  estimator yanlılığının sistematiği Wegscheider ve Vilsmeier tarafından
  (PRAB 26, 032803) formelleştirilmiştir — onların parametresi kuadrupol
  gücü, bizimki transvers kaçıklıktır.
- **Beam-based alignment.** Demeti kuadrupolün manyetik merkezinden geçirme
  teknikleri modern hızlandırıcılarda standarttır: gradyanı değiştirip
  yörünge tepkisinin sıfırlandığı noktayı bulmak, bunu AC uyarımla
  hızlandırmak (Martí ve ark., PRAB 23, 012802) ve çok mıknatısta eşzamanlı
  yapmak (Huang, arXiv:2203.14869). Bu makaledeki hiçbir ölçüm tekniği bu
  anlamda yeni değildir; **yeni olan, bu tekniklerin sahte-EDM'i süren belirli
  bir kaçıklık bileşeni üzerindeki performansının belirlenmesidir.**
- **Sahte-EDM bütçesi.** Kaçıklık→sahte-EDM ileri hesabı all-elektrik halka
  için Hacıömeroğlu ve Semertzidis'te (arXiv:1709.01208); manyetik-kuadrupollü
  simetrik-hibrit tasarımın tam sistematik bütçesi Omarov ve ark.'da vardır.
  Omarov, geometrik-faz sahte-EDM'inin **CW/CCW eşzamanlı karşı-dönen
  demetlerin farkı** + **kuadrupol polarite değiştirme** + **karşı-dönen
  demet ayrımının (CR-ayrım) dipol düzelticilerle küçültülmesi** ile hedefin
  altına indirilebileceğini fiziksel olarak gösterir.
- **Komşu çalışmalar.** Simetrik örgülerde circulant tepki matrisiyle yörünge
  düzeltme (Mirza, PRAB 22, 072804); tepki matrisinin işletim sırasında
  güncellenmesi (Ziemann & Ziemann, arXiv:2104.05300); koherent yer
  hareketinin yörüngeye etkisi (Rossbach, 1989); polarimetre performansı ve
  istatistik bütçesi (arXiv:2010.13536).

### 1.4 Açık kalan soru (bu makalenin boşluğu)

Omarov'un kontrol zinciri, CR-ayrımını **ölçüp** küçültmeye dayanır; ama
makalede (i) bu ölçümün kendisi (48 BPM/SQUID + K-modülasyon geri-çatımı)
simüle edilmemiş, (ii) kaçıklığın **yörüngeye görünmeyen** bileşenine karşı
körlük ihtimali ele alınmamış, (iii) polarite değiştirmenin CW/CCW'den
bağımsız bir düğme olup olmadığı nicelenmemiştir. Bu üç boşluk kritiktir,
çünkü — bu makalede gösterileceği gibi — kaçıklığın sahte-EDM'i süren kısmı
tam da yörüngeye en az görünen kısmıdır.

### 1.5 Bu makalenin katkısı ve planı

Gerçekçi BPM sistematikleri (ofset, kazanç, gürültü) ve optik-model hataları
(β-beating, kuadrupol tilt'i) altında, yörünge-tabanlı ölçüm+düzeltme
zincirinin ulaşabileceği **sahte-EDM tabanını** hizalama toleransının
fonksiyonu olarak belirliyoruz: σ = 10 μm'de taban ~6×10⁻²⁸ e·cm'dir ve σ² ile
ölçeklenir. Tabanın **BPM teknolojisinden bağımsız** olduğunu — sınırın beyaz
gürültü değil, ortalamayla sönmeyen iki koherent sistematik (per-quad
modülasyonda optik "nefes"; matris inversiyonunda β-beating × koşullanma)
olduğunu — gösteriyoruz ve Omarov'un önerdiği CR-ayrım gözlenebilirinin aynı
körlüğü paylaştığını doğrudan ölçüyoruz.

Makalenin planı: §2 simülasyon altyapısını, sahte-EDM estimatörünü,
simetrik/antisimetrik ayrışımı ve sistematik hata modellerini tanımlar. §3
önce ulaşılabilir bastırma zincirini (pozitif sonuç), sonra simetrik bileşeni
yörüngeden ölçmeye yönelik denenen yöntemleri ve her birinin nerede durduğunu
sunar. §4 sonuçları, deney tasarımına pratik çıkarımlarla birlikte toplar.

---

## 2. Metod

2.1 **Simülasyon altyapısı:**
- C++ izleyici (`integrator.cpp`): GL4 semplektik parçacık + Thomas-BMT spin;
  elektrik defleksiyon + manyetik FODO; `EDMSwitch` ile gerçek EDM açılır
  (η=1.88×10⁻¹⁵ → 9.81×10⁻¹⁰ rad/s, Eq. C1 ile birebir).
- Latis: 24 FODO, 48 quad, R0=95.49 m, g=0.2 T/m, Q_y≈2.3 (`params.json` tek kaynak).
- Analitik Twiss/R çözücüsü (`analytic_kmod.py`): C++ ile **%1 içinde**
  çapraz-doğrulanmış (`squid_bpm_test.md §5.5`) — hızlı taramalar analitik,
  tüm kritik sonuçlar C++ ile teyit.
- **Metodolojik uyarı (svd.md dersi):** tepki matrisi analitik-Twiss ile değil
  **gerçek demet dinamiğiyle** kurulmalı; aksi hâlde null-mod/koşullanma gizlenir
  (cond 193 vs gerçek 1.2×10⁸, `svd.md §5`).

2.2 **Sahte-EDM estimatörü (kritik reçete):**
- Ana yöntem: **4D kapalı yörünge + model-fit seküler eğim**
  (`berry_data/false_edm_4d.py`) — tek ideal parçacık kapalı yörüngeye oturtulur,
  S_y(t) = a + bt + Σ_k[c_k cos + d_k sin] fitinden yalnız seküler eğim b çekilir.
  **Düz polyfit yasak** (betatron salınımını sinyal sanır).
- Çapraz-doğrulama: **dört-parçacık simetrik başlangıç** — aynı betatron genliğinde
  başlangıç konumları (+Δx,+Δy), (+Δx,−Δy), (−Δx,+Δy), (−Δx,−Δy) olan dört
  parçacığın ortalaması; betatron katkısı ve ⟨ΔxΔy⟩ artığı simetriyle söner.
  *(Makalede "(sx,sy)=±1" kısaltması KULLANILMAZ — spin bileşeni sanılıyor;
  açıkça "dört simetrik başlangıç konumu kombinasyonu" diye yazılır.)*
- Doğrulama: σ=10→5→2.5 μm taramasında **p = 2.00 ± 0.01** (saf kuadratik
  geometrik faz, lineer kaçak yok; Omarov Fig. 9a'nın birebir reprodüksiyonu).

2.3 **Antisimetrik/simetrik ayrışım — TANIM ve KANIT (kullanıcı: "ikna edici
göster"):**
- **Tanım (projektörlerle, açık formül):** 48-bileşenli kaçıklık vektörü v için,
  her FODO hücresinde (QF=2k, QD=2k+1 indisleri):
  - simetrik bileşen: (P_sym v)[2k] = (P_sym v)[2k+1] = (v[2k]+v[2k+1])/2
    (QF ve QD **birlikte** kayar),
  - antisimetrik bileşen: (P_anti v)[2k] = −(P_anti v)[2k+1] = (v[2k]−v[2k+1])/2
    (QF ve QD **zıt** kayar).
  P_sym + P_anti = I, P_sym·P_anti = 0 (ortogonal ayrışım; kod:
  `make_orbit_figures.sym_anti_projectors`).
- **Neden bu ayrım fizikseldir:** QF ve QD gradyanları zıt işaretli olduğundan,
  zıt-yönlü kayma (antisim) **aynı-yönlü** kick verir (üst üste biner → düşük-k,
  büyük yörünge); aynı-yönlü kayma (sim) **zıt** kick verir (hücre içinde iptal →
  k≈24 hızlı-değişen desen, bastırılmış yörünge).
- **Kanıt (Fig. 2):** R'nin SVD'sinde her sağ-tekil vektörün simetrik içeriği
  ‖P_sym v_i‖² hesaplanır: büyük tekil değerli modlar ~saf antisimetrik (mavi),
  küçük tekil değerli modlar ~saf simetrik (kırmızı) — ayrışım varsayım değil,
  R'nin kendi yapısından ÇIKIYOR. cond(R)=193.
- Kazanç yasası G_k = C/|Q²−k²| (C≈24.8, Q²≈5) aynı gerçeğin harmonik dili.
- COD izi (aynı 10 μm RMS): antisim 114.7 μm vs simetrik 1.69 μm
  (`akilli_duzeltme.md §3`).

2.4 **Metrikler — korelasyonu NASIL hesaplıyoruz ve tuzağı (kullanıcı talebi):**
- **corr tanımı:** geri-çatılan v̂ ile gerçek v arasında 48 bileşen üzerinden
  **Pearson korelasyon katsayısı**: corr = Σ(v̂−⟨v̂⟩)(v−⟨v⟩) / (48·σ_v̂·σ_v).
  Makalede her kullanımda bu tanım verilecek.
- **Tuzak:** Pearson corr **ölçek- ve ofset-değişmezdir**; 48 modun çoğunu
  oluşturan kolay antisimetrik modlarca domine olur ve **üniform/simetrik hatayı
  görmez** (mean-çıkarma tam da simetrik ailenin en kötü modunu siler).
  corr=0.99 iken simetrik bileşen tamamen yanlış olabilir (Fig. 4'te gösterilir).
- **Doğru metrik:** hata vektörü e = v̂ − v'nin simetrik-bileşen RMS'i
  ‖P_sym e‖_RMS (mean-ÇIKARMADAN) ve antisimetrik eşi — sinyalin kendi
  bileşen RMS'iyle (41/41 μm @ ±100 μm uniform) karşılaştırılır.
  (`squid_bpm_test.md §9.3` uyarısının formelleştirilmesi.)

2.5 **Sistematik hata modelleri (nihai sayılarda belirleyici olanlar):**
- **BPM ofseti:** statik, per-BPM ~100 μm (varyant: 30 μm); k-mod/farkta iptal,
  tek-yörünge inversiyonunda katil (`svd.md §5.1, §6`).
- **BPM gürültüsü:** ~1 μm tek-atış (beyaz); lock-in ile √N.
- **BPM kazanç hatası:** çarpımsal δg_i, σ_g=%1–10 taraması.
- **β-beat:** per-quad gradyan hatası ε (C++ `quad_dG`); LOCO-artığı temsili
  (%0.5–5). **Ortalamayla sönmeyen model-sistematiği** — belirleyici kalem.
- **Quad tilt (roll) ψ:** iki kanal — (a) ölçüme çapraz-düzlem sızıntısı,
  (b) doğrudan geometrik-faz katkısı (C++ estimatörle; kmod_drivers/tiltscan).
- Seed/ensemble politikası: her iddia çok-seed (tipik 3–40); tek-seed'den
  kök-sebep çıkarılmaz (`akilli_duzeltme.md §6.15.2` dersi).

---

## 3. Simülasyon Sonuçları (Simulation Results)

### 3.1 Taban zinciri ve bastırma eğrisi (pozitif çerçeve)

- Ham sahte-EDM (σ=10 μm): ~10⁻⁶ rad/s ≈ 1000× hedef (~10⁻²⁶ e·cm).
- + CW/CCW farkı: **3.4×** → 474× (~5×10⁻²⁷ e·cm) (`cwccw_telafi`).
- + standart-BPM orbit düzeltme (antisim silinir): **7.7×** → **62× hedef
  (6.05×10⁻⁸ rad/s ≈ 6×10⁻²⁸ e·cm)** (`orbit_duzeltme`; `omarov.md §10`).
- Kalan artık **simetrik/orbit-kör**; σ² yasasıyla ölçeklenir → **ana figür:
  ulaşılabilir sahte-EDM (e·cm) vs σ eğrisi.** 10⁻²⁹ için σ_sym ≈ 1.3 μm gerekir.
- Manyetik ≠ mekanik merkez: ~20 μT parazit dipol → 10–100 μm eşdeğer kayma →
  survey bu σ_sym'i garanti edemez; merkezi yalnız ışın görür (`PROJE_ANALIZI §3`).

### 3.2 Simetrik bileşeni yörüngeden ölçme girişimleri (ters sınıf — hepsi aynı duvar)

**3.2.1 Tek-frekans K-mod (ΔR) + SVD/TSVD/regularizasyon:**
- Temiz inversiyon corr=1.000000 (nefes matrisin içinde, engel değil); ama
  cond(ΔR)=3.7×10⁴, antisim/sim kazanç oranı ~1393× (`squid_bpm_test.md §8`).
- **Lock-in:** beyaz gürültü √N ile yenilir (σ=10 μm tek-atış → 1000 s'de
  corr 0.992); **ama simetrik bileşen lock-in tabanında BİLE ölü**: σ_min≈10⁻⁴ →
  10 nm artık gürültü → 96 μm sim-hata; kurtarma **<4 nm** ister (§9.5).
- **β-beat felaketi:** ε=%0.5 → sim-hata 1931 μm (sinyalin ~50 katı); sistematik,
  hiç ortalanmaz → **SQUID/BPM teknolojisi tabanı değiştirmez.**
- SVD metodolojik dersleri (svd.md): regularize tahminde **ham std ≠ tespit
  tabanı** (gain-kalibrasyon şart, §5.1); N=2 stray-alan örneğinde gerçekçi
  ofsette taban ~45 nT ≈ harmonik-fit (SVD üstünlüğü yok).

**3.2.2 Dağıtık-frekans per-quad K-mod (AC-BBA) — NEFES, DETAYLI ANLATIM
(kullanıcı talebi; kaynak `squid_bpm_test.md §2-7`, Fig. 3):**
- **Yöntemin vaadi:** her quad ayrı frekansta (%2 derinlik) modüle edilir; BPM
  sinyalinde f_i frekansının genliği A_i, o quad'ın demet-merkez ofsetiyle
  orantılı sanılır → per-quad okuma, matris-inversiyonu yok → no-go'yu atlama
  umudu. (Işık kaynaklarındaki AC-BBA'nın [Martí PRAB 23, 012802] bu makineye
  uyarlanması.)
- **Simülasyon inceliği (adyabatik eşdeğer):** 1–10 kHz modülasyon, betatron
  frekansından (~515 kHz) çok yavaş → demet her an o anki gradyanın kapalı
  yörüngesinde oturur → genlik = kapalı yörüngenin gradyana göre **statik
  türevi**; iki statik kapalı yörünge farkıyla hesaplanır (zaman-takibi
  gerekmez, adyabatik limitte exact).
- **Neden çöküyor — iki etki:** quad i'yi modüle etmek (1) kendi feed-down
  kick'ini değiştirir (aranan sinyal, ~0.9 μm) VE (2) **bir-tur matrisini** —
  dolayısıyla β(s), φ(s), Q'yu — tüm halkada değiştirir. Mevcut ~0.37 mm'lik
  kapalı yörünge (48 quad'ın ortak eseri) bu değişmiş optikten yeniden taşınır →
  BPM'de birkaç μm'lik "**optik-nefes**" terimi. **Nefes/sinyal ≈ 7** (S/B≈0.14).
- **Kaldıraç ayrıştırması (kesin kanıt):** quad 37 modülasyonuna tepki, tüm
  kaçıklık deseni varken −9.83 μm; yalnız dy[37] varken −0.037 μm (**265×
  küçük**). Yani genlik, modüle edilen quad'ın ofsetini değil **komşuların
  kurduğu yörüngeyi** ölçer; kalibrasyona bölmek anlamsız.
- **Sayısal sonuç:** corr tek BPM'de +0.07, **48 BPM'de ≈ −0.03**. Nefes
  **koherenttir** (≈ mevcut yörünge × tek skaler): BPM sayısı/SQUID çözünürlüğü
  onu **ortalamayla söndüremez**. Nefessiz idealizasyonda corr=1.000 —
  suçlunun nefes olduğunun kontrolü. C++ ile bağımsız doğrulandı
  (analitik/C++ duyarlılık oranı ~1.00; iki kod yalnız params.json paylaşır).
- **Dört sağlama** (kod hatası değil): tune patolojik değil (sinπQ=0.815);
  etki global (tüm halka 12 μm kayar); derinlikle lineer (oran sabit ~0.14 —
  modülasyonu küçültmek kurtarmaz); minik-türev ekstrapolasyonu tutarlı.
- **BPM ofseti bu dalda alakasız:** AC demodülasyon DC'yi söndürür (100 μm →
  65 nm sızıntı; pencere-kilitli frekansta tam sıfır) — ofseti 30 μm'e indirmek
  hiçbir şey değiştirmez.
- **Kavramsal ayrım (öğretici):** nefes ≠ no-go — biri koherent kirlilik
  (dağıtık-frekansı öldürür), öbürü kötü-koşullu inversiyonda rastgele gürültü
  büyümesi (tek-frekansı öldürür). İkisi farklı, ikisi de kapatır (§8).

**3.2.3 NN ile kaçıklık geri-çatımı:**
- misalignment→COD **lineer** → NN yalnız R'yi öğrenir; ters yön = R⁻¹.
- Sayı: simetrik geri-çatım hatası NN (MLP 128,128; 4000 örnek) 5.6 μm ≈ TSVD
  6.3 μm (sinyal 9.9 μm) — sınır bilgi-teorik, algoritmik değil
  (`akilli_duzeltme.md §6.8`).
- **"Daha iyi mimari kurtarır mı?" (kullanıcı sorusu) — HAYIR, ve makalede bu
  gerekçeyle söylenecek:** öğrenilecek harita **tam lineer** olduğundan
  (COD = R·mis) MLP zaten yeterli kapasitede; CNN/transformer/GNN gibi mimariler
  ancak sömürülecek doğrusal-olmayan yapı varsa kazandırır — burada yok.
  Simetrik iz gürültü+sistematik tabanının ALTINDA → hiçbir estimator veride
  olmayan bilgiyi çıkaramaz (Cramér-Rao; beyaz gürültüde lock-in/en-küçük-kareler
  zaten optimal). Kazanç ancak ek **ön-bilgiden** gelebilirdi (seyreklik →
  LASSO/CLEAN zaten denendi, aynı taban, `false_edm_harmonic_sinir.md §14.5`);
  zaman-serisi mimarileri (LSTM) de aynı duvara çarpar — asıl engel gürültü
  değil, koherent sistematik + koşullanma (`akilli_duzeltme.md §6.9`).

**3.2.4 LOCO ile başlangıç düzeltmesi + drift izleme:**
- **Çalışan kısım (pozitif):** ofset-iptalli drift modu — 50 μm BPM ofseti
  altında 6.5 μm RMS drift hassasiyeti (mutlağa göre 29×); %1 β-beat'te 6.1 μm
  (`makale-taslagi-2.md §3.4, §3.7`).
- **Sınır:** en kötü 8 mod %96 simetrik, 193× gürültü büyütmesi (§3.8) —
  drift gözcüsü de simetriğe kör.
- **Ofset-iptal alt sınırı (MÜTEVAZI sunum — kullanıcı: "çok iddialı olma"):**
  Sezgi önce: BPM ofsetini iptal etmenin bedavası yok. Ofseti iki-ölçüm farkıyla
  (k-mod: g₁/g₂) sildiğinde, elindeki bilgi artık R değil **ΔR = R(g₂)−R(g₁)**
  olur; iki konfigürasyon birbirine yakınsa (ε = göreli fark) ΔR ~ ε·(∂R/∂g)
  küçülür ve tersi ~1/ε büyür: ‖ΔR⁻¹‖ ~ ‖R⁻¹‖/ε. Yani **ofset bağışıklığının
  bedeli, gürültü büyütmesinin 1/ε katına çıkmasıdır.** Makalede bu, "teorem"
  iddiasından çok **belirli bir estimator sınıfı için yapısal alt sınır** olarak
  sunulur: (i) lineer, (ii) ön-yargısız, (iii) statik ofseti TAM iptal eden
  iki-ölçüm estimatorları sınıfında geçerli; sınıf dışı yollar (ör. ofseti
  Bayesçi ön-bilgiyle kısmen modellemek) kapsam dışıdır ve açıkça söylenir.
  (`makale-taslagi-2.md §2.4` türetmesi; burada iddia "başka hiçbir şey
  yapılamaz" DEĞİL, "bu doğal sınıfın tabanı budur".)

**3.2.5 Harmonik yaklaşımlar (kullanıcı: "unutmuşum, ekleyelim"):**
- **Hedefli Fourier geri-çatımı:** kaçıklığın harmonik içeriği önceden
  biliniyorsa taban Fourier modlarıyla kurulur → koşullanma 13000→186'ya iner
  ve geri-çatım çalışır; ama **keyfî (bilinmeyen) desen** için greedy/LASSO
  harmonik seçimi rank-yoksulluğundan çöker (`FOURIER_REKONSTRUKSIYON.md`,
  `README §`; R-LS/CLEAN/Bozoki aynı gözlenebilirlik tabanı,
  `false_edm_harmonic_sinir.md §14.5` — 6 yöntem tek duvar).
- **Harmonik fit (tek-mod okuma) — svd.md vaka çalışması:** bilinen bir alan
  imzasını (örn. N=2 radyal stray alan → dikey k=2 yörünge) 48-BPM verisinin
  cos2θ/sin2θ izdüşümüyle okumak **iyi-koşulludur (cond=1) ama YANLIDIR**:
  k=2 katsayısına alan + kaçıklığın-k=2'si + ofsetin-k=2'si birlikte düşer,
  tek sayı dördünü ayıramaz → 10 μm kaçıklıkta ~53 nT yanlılık (gerçek 1 nT
  görünmez). SVD-TSVD ile "ayırma" da gerçekçi BPM ofsetinde harmonik fit'e
  üstünlük sağlamaz (gain-kalibre taban ~45 nT ≈ harmonik ~53 nT; `svd.md §5.1`).
  **Ders:** ileri-okuma (izdüşüm) ofseti büyütmez ama dejenerasyonla yanlıdır;
  ters-çevirme dejenerasyonu çözer ama ofseti büyütür — iki yol da simetrik/
  dejenere bilgiye ulaşamaz (Wegscheider MSE=Var+Bias² dilinde: biri
  Bias-domine, öbürü Var-domine).

**3.2.6 CR-ayrım gözlenebiliri (Omarov'un önerdiği knob):**
- Doğrudan C++ ölçümü: simetrik bastırma **CR-ayrımda 4.5× ≈ tek-yön COD'da
  3.8×** → ayrım da bir yörünge-farkı, **ek pencere açmaz**; no-go CR-ayrıma
  birebir taşınır (`omarov.md §9.3`). **Omarov §9 boşluğunun kapanışı.**

### 3.3 "Simetriği görünür yap / iptal et" girişimleri

**3.3.1 Quad polarite-flip:**
- f gradyanda **ÇİFT** (iki dönmenin çarpımı, ∝g²): flip oranları sim/antisim/
  genel = +2.81/+0.85/+1.17 (işaret korunur) → flip geometrik fazı iptal etmez
  (`akilli_duzeltme.md §6.14a`).
- **Dejenerasyon:** idealize FODO'da f(CCW,g+) = f(CW,g−) (oran 1.00) →
  Eq. C2'nin 4'lü kombinasyonu 2'liye çöker; flip bağımsız knob değil
  (`kmod_drivers/fast_est cwccw`; 20-seed ensemble ile pekiştirildi).

**3.3.2 Yüksek betatron tune:**
- Fikir: Q'yu simetrik desenin harmoniğine (k≈24) yaklaştır → rezonans paydası
  |Q²−k²| küçülür → simetrik mod orbit-görünür olur.
- Kısmen çalışır: Q=2.3→10.3 (alternasyon korunur, C++ kararlı): simetriğin
  **yüksek-m** parçası açılır (m=10 kazanç 71 ≈ antisim). Ama **düşük-m**
  k=24−m≥16 → Q≥16 ister; Q_max=12 (μ=180°/hücre stopband) → k-mod ile kurtarma
  ~%8–25 → bastırma **~2×** (marjinal).
- **Asıl bedel — quad güçleri (VURGULANACAK, kullanıcı):** Q'yu yükseltmek
  ancak **gradyanı büyüterek** olur: Q 2.3→10.3 için g = 0.21→0.69 T/m
  (~3.3×). Daha güçlü quad'lar, AYNI kaçıklık için demete daha sert kick verir
  → geometrik faz iki dönmenin çarpımı olduğundan sahte-EDM ~g² büyür, üstüne
  büyüyen betatron orbiti eklenir → net ölçüm **~g³: 32× daha büyük sahte-EDM.**
  Gerçek EDM sinyali ise elektrik deflektörden gelir ve **quad gradyanından
  bağımsızdır** → sinyal/arka-plan yüksek-Q'da 32× KÖTÜLEŞİR. Yani yüksek-Q'da
  kalınamaz; geçici tanı olarak kullanılıp dönülse bile rekonstrüksiyon
  simetriğin ancak ~%25'ini taşır (`akilli_duzeltme.md §6.15, §6.15.1`).
- *(Uniform-gradient / zayıf-odaklama varyantı makaleye alınmıyor — kullanıcı
  kararı.)*

### 3.4 Yörünge-dışı kısa not: off-momentum spin-sensing (kaçış değil)

- Fikir: magic-dışı momentum → ν_s≠0 → geometrik faz ∝1/ν_s amplifiye (doğru,
  ölçüldü). **Ama** 1.-mertebe ISA-eğimi (σ¹) aynı frekansta ve ~3600× büyük;
  tüm ayırma kanalları kapalı (δ-tarama, faz, 4-fold, RMS, 3D n̂-fit, S_z).
- Magic (ν_s=0) tek çalışma noktası: baskın 1.-mertebe etkiyi zararsız sabite
  çevirir (`akilli_duzeltme.md §6.16`). → Yörünge-dışı bu kaçış da kapalı;
  spin-tabanlı null'lama zaten kapsam dışı (sahte/gerçek degenerasyonu, §4).

### 3.5 Sistematik bütçe (Metod §2.4 modelleriyle, nihai sayılar)

| Kalem | Etkisi | Kaynak |
|---|---|---|
| BPM ofseti (100 μm) | inversiyonda katil; k-mod/AC'de iptal | `svd.md §5.1/§6`, `squid §9.1` |
| BPM gürültüsü (1 μm) | lock-in ile yenilir; sınırlayıcı DEĞİL | `squid §9.3` |
| BPM kazanç (%1) | 48-BPM ortalamasıyla ikincil | `ac_bba_systematics` |
| **β-beat (%0.5–1)** | **inversiyonda felaket (1931 μm); belirleyici** | `squid §9.5` |
| **Quad tilt (roll)** | doğrudan geometrik-faz kanalı; tolerans ~0.3 mrad; CW/CCW+flip'te ODD-baskın (söndürülemez, ~8–57× hedef @1 mrad) | `kmod_bba_sonuclar §7.3` (C++ estimatör; superseded-iddia'dan bağımsız) |

---

## 4. Sonuçlar (Conclusions)

4.1 **Nicel ana sonuç:** yörünge-tabanlı zincir (CW/CCW + orbit düzeltme +
drift izleme) sahte-EDM'i σ=10 μm'de **~6×10⁻²⁸ e·cm**'e bağlar; taban σ² ile
ölçeklenir; 10⁻²⁹ için σ_sym ≲ 1.3 μm gerekir ve bu yörüngeyle doğrulanamaz.

4.2 **Sınır teoremi (negatif çekirdek):** simetrik (orbit-kör) bileşen, denenen
tüm yörünge-tabanlı kanallarda (ΔR/SVD/lock-in, per-quad AC, NN, LOCO+drift,
CR-ayrım) gerekli seviyede **gözlenemez**; sınır beyaz gürültü değil, iki
koherent sistematik (optik-nefes; β-beat×koşullanma) + latis yapısı (düşük-m
simetrik modların rezonansı Q≥16 ister, stopband Q≤12'ye izin verir).
**Dolayısıyla SQUID dahil hiçbir BPM teknolojisi tabanı değiştirmez.**

4.3 **Kapsam/pozisyon notları:**
- Spin-tabanlı aktif null'lama kapsam dışı: sahte/gerçek EDM aynı gözlenebilirde
  dejenere (null'lamak gerçeği de siler); ayıran tek araç CW/CCW farkı, o da
  simetrik artığı bırakır. Ayrıca polarimetre istatistiği iteratif spin-null'u
  zaman-yasak kılar (~yıl/ölçüm; `cosy_polarimeter.md §4`).
- Yörüngeden f'i **ileri-öngörme** (öğrenilmiş/analitik fonksiyonel) ayrı bir
  açık problemdir; bu makalenin ölçüm-sınırı iddiası onu kapsamaz (tek cümle).
- Omarov'a göre konum: geometrik-faz kontrolünün fiziği (Fig. 16) doğrulanır;
  açık bırakılan ölçüm-zinciri boşluğu (i-iii, §1.4) nicel kapatılır. Pratik
  öneri: standart-BPM orbit-monitör = SQUID'in ucuz ikamesi (antisim kısımda
  7.7×) + drift gözcüsü; simetrik artık **tasarım toleransıyla** (σ²) sınırlanır.

4.4 **Geçerlilik sınırları (dürüstlük):** tek latis (24 FODO, Q≈2.3), lineer
(sekstüpolsüz) model; 3.4×/7.7×/62× sınırlı seed istatistiği (makale öncesi
ensemble genişletme ucuz sağlamlaştırma); tilt ayrı doğrudan kanal, misalignment
bütçesinin dışında raporlanır.

---

## 5. Figürler — durum (üretici: `make_orbit_figures.py`, İngilizce etiketli)

**ÜRETİLDİ (analitik; repoda, 2026-07):**
1. **`fig_orbit_suppression.png` (ANA):** Ulaşılabilir sahte-EDM vs σ —
   ham/CW-CCW/orbit-düzeltme basamakları (σ=10 μm C++ noktaları) + σ² eğrileri,
   10⁻²⁹ hedef çizgisi, σ_sym≈1.3 μm kesişimi, sağ eksen e·cm.
2. **`fig_orbit_modes.png`:** R'nin SV spektrumu — her mod ‖P_sym v_i‖² ile
   renkli (büyük-σ=antisim mavi, küçük-σ=sim kırmızı; cond=193, analitik
   üretimde belge değeriyle birebir) + G_k=C/|Q²−k²| kazanç yasası paneli.
3. **`fig_orbit_breathing.png`:** Nefes — nefessiz corr=1.000 vs nefesli
   1-BPM corr=+0.07 / 48-BPM corr=−0.03 saçılımı + kaldıraç barları (266×;
   belgedeki 265× ile birebir).
4. **`fig_orbit_lockin.png`:** Tek-frekans ΔR inversiyonu, lock-in tabanında
   (10 nm): sim/antisim bileşen hatası vs β-beat (%0→283 μm@%0.5→3967 μm@%5;
   sinyal 41 μm çizgisi) + corr'un yanıltıcı yüksek kalışı
   (cond(ΔR)=3.74×10⁴, σ_min≈10⁻⁴ — belge değerleriyle birebir).

**C++ GEREKTİRENLER (üretilecek; uzun koşum):**
5. σ² doğrulaması p=2.00 (Omarov Fig. 9a karşılığı) —
   `kmod_drivers/fast_est.py calib` (~saatler, 3 seed).
6. CR-ayrım körlüğü (4.5× vs 3.8×) + flip dejenerasyonu — `cr_separation.py`
   ve `kmod_drivers/fast_est.py cwccw` desenleri /tmp'den yeniden kurulacak.
7. Yüksek-Q g³ ölçeklemesi (0.21/0.40/0.69 T/m → 1×/4.1×/32×) — üç ayar ×
   C++ estimatör koşumu.

## 6. Sayı → kaynak eşleme (iç referans; makaleye girmez)

| Sayı | Kaynak belge | Script |
|---|---|---|
| p=2.00±0.01 | `omarov.md §10`, `orbit_ileri_olcum.md §9` | `sigma_olcekleme` |
| 3.4× / 7.7× / 62× | `omarov.md §10` | `cwccw_telafi`, `orbit_duzeltme` |
| cond(R)=193; 114.7 vs 1.69 μm | `akilli_duzeltme.md §2-3` | `surrogate`, `analyze` |
| cond(ΔR)=3.7e4; <4 nm; β-beat 1931 μm | `squid_bpm_test.md §8-9.5` | `v27_recheck`, `v27_syst2` |
| Nefes S/B=0.14; 48-BPM corr≈0 | `squid_bpm_test.md §5-7` | `single_bpm_test`, `breathing_cpp` |
| NN 5.6 ≈ TSVD 6.3 μm | `akilli_duzeltme.md §6.8` | `nn_vs_Rinv` |
| Drift 6.5 μm @ 50 μm ofset | `makale-taslagi-2.md §3.4` | `drift_monitor/` |
| CR 4.5× ≈ COD 3.8× | `omarov.md §9.3` | `cr_separation` |
| Flip +2.81/+0.85/+1.17; CCW≡CW+flip | `akilli_duzeltme.md §6.14`, `kmod_bba_sonuclar §7.3.1` | `flip_real`, `cwccw_ens` |
| Yüksek-Q: kazanç 71, ~2×, g³ 32× | `akilli_duzeltme.md §6.15` | `highQ_bba` |
| Uniform-g kararsız | `akilli_duzeltme.md §6.14b` | `diag_mode` |
| Off-momentum σ¹/σ² ~3600× | `akilli_duzeltme.md §6.16` | `momentum_*` |
| Tilt toleransı ~0.3 mrad; ODD-baskın | `kmod_bba_sonuclar.md §7.3` | `tiltscan`, `cwccw_ens` |
| Ofset-iptal dualite ‖ΔR⁻¹‖~‖R⁻¹‖/ε | `makale-taslagi-2.md §2.4` | — |
| Gain-kalibrasyon dersi; 45 nT tabanı | `svd.md §5.1` | `field_n2_nodc` |

## 7. Kaynakça (çekirdek)

1. Z. Omarov et al., PRD **105**, 032001 (2022).
2. S. Hacıömeroğlu, Y. Semertzidis, arXiv:1709.01208.
3. Y. Chung, G. Decker, K. Evans, Proc. PAC 1993, s.2263.
4. S. Wegscheider, A. Vilsmeier et al., PRAB **26**, 032803 (2023).
5. Z. Martí, G. Benedetti, U. Iriso, A. Franchi, PRAB **23**, 012802 (2020).
6. X. Huang, arXiv:2203.14869.
7. M. Mirza et al., PRAB **22**, 072804 (2019).
8. V. Ziemann, M. Ziemann, arXiv:2104.05300.
9. J. Rossbach, Particle Accel. **23**, 121 (1989).
10. COSY LYSO polarimetre, arXiv:2010.13536.
