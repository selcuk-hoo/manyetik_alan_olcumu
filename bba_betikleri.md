# bba_betikleri.md — BBA Simülasyon Betiklerinin Basit Kılavuzu

> **Bu belge ne için?** Sahte-EDM'i "demet-tabanlı hizalama" (BBA) ile ölçüp
> düzeltme fikrini test eden beş Python betiğini, hiçbir ön bilgi varsaymadan
> anlatır: her biri neyi yapar, ne zaman kullanılır, nasıl çalıştırılır, ne
> üretir, nelere dikkat edilir. Betiklerin *bilimsel* sonuçları için
> `separation_bba_testleri.md` ve `why_bba_works.md`'ye bakın; burası
> **kullanım kılavuzu.**

---

## 0. Büyük resim: hepsi ortak bir motoru sürüyor

Önce en önemli şey: **hiçbir betik fiziği kendi yazmaz.** Hepsi aynı üç parçayı
kullanır ve bu parçalar bu oturumda **değişmedi**:

```
        (senaryo betiği: aşağıdaki 5 taneden biri)
                        │
                        ▼
   integrator.py  ──►  lib_integrator.so   (integrator.cpp'den derlenmiş
   (ctypes köprüsü)     GERÇEK parçacık + spin motoru — DOKUNULMADI)
                        │
   berry_data/false_edm_4d.py  /  kmod_drivers/fast_est.py
        └─ doğrulanmış sahte-EDM ölçeri (4D kapalı yörünge + model-fit; p=2.00)
                        │
   analytic_kmod.py  ─ yalnız bump ölçekleme/düzeltici seçimi için hızlı KILAVUZ
                        (nihai sonuç için ASLA kullanılmaz)
```

Yani "C++ koşumu yapıyorum" demek: kaçıklık desenini, gradyan modülasyonunu,
bump'ları Python'da hazırlayıp asıl dinamiği **değişmemiş C++ motoruna**
hesaplattırmak. Betikler sadece "hangi senaryoyu koştur, sonucu nasıl analiz
et" katmanıdır.

**Tek altın kural:** sahte-EDM'in nihai değeri **her zaman** spin izleyicisiyle
(`fast_measure`) ölçülür — "f = A·σ²" gibi bir formülle **değil.** Analitik
hesaplar sadece nereye bakacağımızı söyler.

---

## 1. Beş betik, bir bakışta

| Betik | Bir cümlede | Gerçek C++? | Süre |
|-------|-------------|:-----------:|------|
| `classic_bba_sim.py` | Hızlı **analitik prototip** — yön gösterir, hüküm vermez | Hayır | saniyeler |
| `classic_bba_cpp_check.py` | **Null doğrulaması**: birkaç quad'ın merkezi doğru bulunuyor mu | Evet | ~25 dk |
| `classic_bba_full.py` | **Uçtan uca**: 47 quad × 2 düzlem BBA + düzelt + sahte-EDM ölç | Evet | ~3 saat |
| `classic_bba_iter.py` | **İterasyon**: yörüngeyi küçült, tekrar BBA (β-beat çözümü) | Evet | ~3 saat/geçiş |
| `diag_bbeat.py` | **Tanı**: bir sorunun kaynağını ince taramayla ayrıştır | Evet | ~35 dk |

Neden beş ayrı betik? Çünkü araştırma soru-soru ilerledi ("çalışıyor mu?" →
"uçtan uca ne verir?" → "β-beat neden bozdu?" → "iterasyon çözer mi?"). Her
soru bir betik oldu. Ortak parçalar `import` ile paylaşılır (kopyalanmaz):
örn. `classic_bba_iter.py`, `classic_bba_full.py`'den tek-yörünge koşumunu
(`_orbit_xy`) ve null-kestirimini (`estimate`) doğrudan çağırır.

---

## 2. Betik betik

### 2.1 `classic_bba_sim.py` — analitik prototip (kılavuz, sonuç değil)

**Ne yapar.** Gerçek C++'ı hiç çağırmadan, hızlı bir analitik model (per-quad
Twiss, optik-nefes dâhil) üstünde BBA'yı taklit eder. Amacı bir *sonuç üretmek*
değil, "gürültü/β-beat şu seviyedeyken ne beklemeliyiz, C++'ta nereye bakmalı"
sorusuna dakikalar içinde cevap vermek.

**Ne zaman.** Yeni bir fikri C++'a 3 saat harcamadan önce hızlıca yoklamak için.

**Çalıştırma.**
```bash
python3 classic_bba_sim.py --seeds 5                 # temel
python3 classic_bba_sim.py --seeds 3 --bbeat 0.01    # β-beat etkisi
python3 classic_bba_sim.py --navg 10000              # gürültü ortalaması
```
Argümanlar: `--noise` (BPM tek-atış gürültüsü), `--navg` (nokta başına atış
ortalaması), `--bbeat` (optik model hatası), `--seeds` (kaç rastgele desen).

**Üretir.** Terminale: merkez-hata RMS'i (simetrik/antisimetrik ayrı) + kaba bir
sahte-EDM tahmini.

> ⚠️ **Uyarı:** Bu betiğin sayıları **hüküm değildir.** Nitekim "β-beat şeffaf"
> öngörüsü burada çıkmış ama C++ onu **çürütmüştü** (bkz. `why_bba_works.md §8`).
> Analitik model nefesi eksik yakalayabilir; her zaman C++ ile teyit et.

---

### 2.2 `classic_bba_cpp_check.py` — null doğrulaması

**Ne yapar.** Birkaç seçili quad için, gradyan modülasyonu tepkisinin
sıfırlandığı demet konumunu (null) gerçek C++ dinamiğiyle bulur ve bunun
quad'ın **gerçek merkezine** oturup oturmadığına bakar. Yani "yöntem tek bir
quad'da doğru çalışıyor mu?" sorusunu, uçtan uca 3 saat harcamadan yanıtlar.

**Ne zaman.** Bir ölçüm-kurulumunun sağlığını test etmek için (ör. okuma
yöntemi doğru mu). Bu betik "ideal eksenden okuma → 5.8 μm sahte bias" dersini
verdi; düzeltince (4D kapalı yörüngeye oturtarak okuma) bias 0.13 μm'e indi.

**Çalıştırma.**
```bash
python3 classic_bba_cpp_check.py -w 4                       # varsayılan quad'lar
python3 classic_bba_cpp_check.py -w 4 --quads 5 20 33 --npts 5
```
Argümanlar: `-w` (paralel işçi), `--quads` (hangi quad'lar), `--npts` (tarama
noktası sayısı), `--seed` (kaçıklık deseni).

**Üretir.** Terminale: her quad için `est − gerçek merkez` (bias). JSON'a:
`kmod_drivers/paper_runs_results.json` → `[bba_cpp_check]`.

---

### 2.3 `classic_bba_full.py` — uçtan uca (asıl ölçüm)

**Ne yapar.** Tam zinciri gerçek C++'la koşar:
1. 47 quad × 2 düzlem (dikey+yatay) için BBA yapıp her merkezi kestirir
   (hücre-0 QF hariç — o motorun `quad_dG`'sini okumaz, sim-kısıtı).
2. Kestirilen merkezleri düşerek "düzeltir".
3. Kalan sahte-EDM'i **spin izleyicisiyle doğrudan ölçer** (formülle değil).

Ayrıca üç gerçekçi sistematiği modeller: **β-beat** (optik model hatası, C++
dinamiğine gömülü), **BPM ofseti** (golden-orbit'te iptal — gösterilir), **BPM
gürültüsü** (farklı ortalama sayılarında, her seviyede C++ ile çapalanır).

**Ne zaman.** "Yöntem, tüm makinede, gerçekçi kusurlarla ne veriyor?" Asıl
sonuç betiği.

**Çalıştırma.**
```bash
# temiz optik (kusursuz):
python3 classic_bba_full.py -w 4 --bbeat 0 --bpm-offset 0 --bpm-noise 0
# gerçekçi sistematikler:
python3 classic_bba_full.py -w 4 --bbeat 0.01 --bpm-offset 100e-6 \
                                 --bpm-noise 1e-6 --navg 100,10000
```
Argümanlar: `-w`, `--scan` (bump tarama genliği), `--seed`, `--bbeat`,
`--bpm-offset`, `--bpm-noise`, `--navg` (virgülle ayrık N_avg listesi).

**Üretir.** Terminale: merkez hataları (sym/anti, iki düzlem), ham vs
düzeltilmiş sahte-EDM, ofset-iptal kontrolü, gürültü taraması. JSON'a:
`[bba_full_syst]`.

**Bilinen sonuç.** Temiz optikte 356× → 1.6× hedef (çalışıyor). β-beat %1'de
çöküyor (nefes) — çözümü `classic_bba_iter.py`.

---

### 2.4 `classic_bba_iter.py` — iterasyon (β-beat/nefes çözümü)

**Ne yapar.** `full` betiği β-beat altında çöktü çünkü BBA'yı ham (büyük)
yörüngeli makinede yapmak, optik-nefesi ölçüme sızdırıyor. Standart BBA
pratiği: **önce yörüngeyi küçült, sonra ölç.** Bu betik onu iterasyonla yapar:

```
geçiş 1: ham makinede BBA → merkezleri kestir → "taşı" (yörünge küçülür)
geçiş 2: daha küçük yörüngeli makinede BBA → daha az nefes → daha iyi kestir
...
```
Her geçiş sonunda kalan sahte-EDM'i **spin izleyicisiyle** ölçer.

İki sağlamlaştırma içerir: **kalite-kesmesi** (BPM'lerin null'da anlaşmadığı
kötü quad'ların — ör. latis dikişindeki quad 45 — düzeltmesi kısılır) ve
**under-relaxation** (her geçişte kısmi düzeltme → ıraksamayı dizginler).

**Ne zaman.** β-beat'li makinede BBA'yı kurtarmak. Şu an aktif geliştirme.

**Çalıştırma.**
```bash
python3 classic_bba_iter.py -w 4 --passes 3 --bbeat 0.01 --relax 0.5
python3 classic_bba_iter.py -w 4 --passes 3 --resume     # restart sonrası devam
```
Argümanlar: `-w`, `--passes` (geçiş sayısı), `--bbeat`, `--relax`
(under-relaxation kazancı 0–1), `--seed`, `--resume` (kayıtlı residual'dan
devam et).

**Üretir.** Terminale: her geçiş için kalan sym-hata (dikey/yatay), yörünge
büyüklüğü, min-kalite, ve sahte-EDM. JSON'a: `[bba_iter]`. Ayrıca restart-güvenli
durum: `/tmp/kmod_recover/bba_iter_state.json` (konteyner çökerse `--resume`
kaldığı yerden alır).

> **Not (restart-güvenlik):** Uzun (~9 saat) koşumlar konteyner yeniden
> başlarsa ölür; bu betik her geçişte residual'ı diske yazdığı için `--resume`
> ile devam edilebilir. Diğer uzun betikler için sonuç JSON'u artımlı yazılır.

---

### 2.5 `diag_bbeat.py` — tanı (sorunun kaynağını ayrıştır)

**Ne yapar.** Bir şey ters gittiğinde "neden?" sorusunu yanıtlar. Birkaç quad
için **ince tarama** (2 yerine 9 nokta) yapıp tepkinin şeklini inceler:
gerçekten lineer mi, null nerede, temiz optik ile β-beat farkı ne. İki özel
modu var:
- `--orbit-scale 0` : **izole test** — ölçülen quad hariç tüm kaçıklıkları
  sıfırlar (nefesin kaynağını ayırmak için). Bu test, β-beat null-kaymasının
  "diğer quad'ların yörüngesinden" geldiğini kanıtladı (bias 2 μm → 0.02 μm).
- `--plane x` : yatay düzlemi test eder (dikeyle karşılaştırmak için). Yatay
  ıraksamanın "birkaç kötü quad" olduğunu bu ortaya çıkardı.

**Ne zaman.** Bir çöküş/ıraksama gördüğünde, "fizik mi yoksa uygulama sorunu
mu, hangi quad'lar, hangi düzlem" diye kök-neden aramak için.

**Çalıştırma.**
```bash
python3 diag_bbeat.py -w 4 --quads 5 20                    # dikey, tam yörünge
python3 diag_bbeat.py -w 4 --orbit-scale 0.0 --quads 5 20  # izole (nefes kaynağı)
python3 diag_bbeat.py -w 4 --plane x --quads 3 11 45       # yatay düzlem
```
Argümanlar: `-w`, `--quads`, `--npts` (tarama noktası), `--half` (tarama yarı-
genliği), `--orbit-scale` (diğer quad'ların kaçıklık ölçeği), `--plane` (x/y).

**Üretir.** Terminale: her quad için null (2-nokta ve 9-nokta), gerçek merkeze
göre hata, lineerlik ölçütü — temiz optik (bb0) vs β-beat (bb1) yan yana.

---

## 3. Hangi soru için hangi betik?

| Sorun / soru | Betik |
|--------------|-------|
| Bu fikir kabaca tutar mı? (hızlı) | `classic_bba_sim.py` |
| Tek quad'da doğru ölçüyor muyuz? | `classic_bba_cpp_check.py` |
| Tüm makinede gerçekçi sonuç ne? | `classic_bba_full.py` |
| β-beat'i iterasyonla yeniyor muyuz? | `classic_bba_iter.py` |
| Neden bozuldu / hangi quad / hangi düzlem? | `diag_bbeat.py` |

---

## 4. Ortak tuzaklar (hepsinde geçerli)

1. **`integrator.cpp` değiştirilmez.** Gereken düğmeler motorda zaten var
   (`quad_dx/dy` kaçıklık, `quad_dG` gradyan modülasyonu, `quad_tilt`). Betikler
   bunları Python'dan sürer.
2. **Hücre-0 QF (`quad_dG` okumaz).** O quad sim'de modüle edilemez; kaçıklığı
   0 alınır ve açıkça raporlanır. Gerçek makinede başka yolla modüle edilir.
3. **Yörünge okuma 4D kapalı yörüngeden yapılmalı.** İdeal eksenden fırlatma
   ~0.2 μm betatron artığı bırakır, null'da ~25× büyür → ~5 μm sahte bias.
   (`classic_bba_cpp_check.py`'nin öğrettiği ders.)
4. **Düzeltici ≡ merkez kayması.** Bump, komşu quad'a `quad_dx/dy` ek terimiyle
   modellenir (dipol düzeltici bobinle fizik olarak özdeş). `dipole_tilt`
   KULLANILMAZ — o spin için sahte-EDM taklidi üretir.
5. **Nihai sahte-EDM her zaman spin ile ölçülür**, merkez-RMS ile değil; çünkü
   sahte-EDM ∝ (yatay artık)·(dikey artık) ve düşük RMS bile kötü korelasyonla
   büyük sahte-EDM verebilir.
6. **Sonuçlar tek latis, lineer model (sekstüpolsüz), çoğunlukla tek seed.**
   Genelleme için bu sınırlar akılda tutulmalı.

---

## 5. Sonuçlar nerede

- **Sayılar:** `kmod_drivers/paper_runs_results.json` (anahtarlar: `bba_cpp_check`,
  `bba_full_syst`, `bba_iter`).
- **Bilimsel yorum + günlük:** `separation_bba_testleri.md` (T5 test zinciri).
- **"Neden çalışıyor" öğretici anlatım:** `why_bba_works.md`.
- **Proje kuralları + tuzaklar:** `CLAUDE.md`.
