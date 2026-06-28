# Kapalı Yörünge ↔ Sahte EDM: İleri-Ölçüm Sorusu ve Berry-Fonksiyoneli (AÇIK PROBLEM)

> **Durum (2026-06): AÇIK PROBLEM, çözülmedi.** Bu günlük, "kapalı yörüngeden
> sahte EDM'yi (spin sistematiğini) ileri yönde öngörebilir miyiz?" sorusunun
> dürüst kaydıdır. Cevap **belirsiz**: bilgi yörüngede *kısmen* var ama doğru
> fonksiyonel empirik olarak pinlenemedi. İleride devam edilecekse **analitik
> türetme** gerekir — empirik fishing bu veriyle yakınsamıyor.
>
> **Strateji notu:** Bu soru drift-monitör çalışmasının ÖTESİNDE ve terk
> ettiğimiz trim metoduna komşu. Drift monitör temiz/bitmiş katkı; bu ise
> yüksek-riskli açık problem. Ancak analitik tutamak çıkarsa peşine düşülmeli.

---

## 1. Soru

SQUID-BPM önerisi modülasyon + düşük-mertebe harmonik ölçümüne dayanır. Bizim
sorumuz: standart BPM'lerle, kapalı yörüngeyi **ileri yönde** okuyarak (kaçıklığı
geri-çözmeden) sahte EDM'yi öngörebilir miyiz? Motivasyon: no-go bir *inversiyon*
(R⁻¹) sınırıdır; sahte EDM kapalı yörüngenin fonksiyoneli olduğundan, ileri öngörü
inversiyona girmez — *eğer* doğru fonksiyonel biliniyorsa.

## 2. Kurulu: sahte EDM bir kapalı-yörünge fonksiyonelidir, AMA basit ⟨x·y⟩ DEĞİL

- Sahte EDM ∝ dx·dy geometrik (Berry) faz, σ² ölçekleme (`false_edm_harmonic_sinir.md §13`).
- Mekanizma kapalı yörüngenin çarpımıdır ama **basit ⟨x_CO·y_CO⟩ ortalaması yanlış
  proxy'dir.** Doğrulama (Run 2, 40 config): antisimetrik configlerde orbit BÜYÜK
  (~72 μm) ama ⟨x·y⟩ ≈ 10⁻⁹ (sıfıra yakın, çünkü x_CO ve y_CO global korelesiz),
  yine de f büyük (~10⁻⁶). Yani **f, global ortalama değil, alan-ağırlıklı/yapısal
  bir integral.** (Erken "f ∝ ⟨x·y⟩ doğrulandı" iddiası küçük-n şansıydı, geri çekildi.)

## 3. Empirik fonksiyonel araması — sonuçlar (Run 2, 40 config)

Hücre içi ardışık iki quad'ın yörüngesi (x1,x2,y1,y2) ile aday fonksiyoneller:

| Aday | antisim corr | simetrik corr | TÜMÜ |
|------|------|------|------|
| (x1−x2)(y1−y2) | +0.36 | −0.10 | +0.36 |
| **Σ(x1·y2−x2·y1) [yönlü alan / Berry]** | **−0.57** | **−0.48** | **−0.57** |
| Σ(x1y1+x2y2) | −0.25 | −0.58 | −0.26 |
| 4-bilineer taban (LOO-R²) | −0.10 | 0.20 | 0.12 |

**En umut verici lead: yönlü alan Σ(x1·y2 − x2·y1)** — Berry fazının doğal
büyüklüğü, ve **her iki grupta da tutarlı** çalışan tek aday (~−0.5). Ama −0.5
orta düzey, temiz kazanç değil.

Ayrıca yörünge çarpım profilinin yüksek Fourier modlarıyla kurulan zengin
fonksiyonel, **simetrikte LOO-R²=0.87** verdi (permütasyon-doğrulamalı: null maks
0.08, yani aşırı-uyum değil GERÇEK sinyal) — ama bu fiziksel bir forma karşılık
gelmiyor ve antisimetrikte çöküyor (R²=0.21). Tutarsızlık, **empirik keşfin 40
config ile yetersiz-belirlenmiş** olduğunu gösteriyor.

## 4. Neden iki grup birden açıklanamıyor (hipotez)

- Simetrik orbit küçük + yüksek-k → çarpım yapısı birkaç Fourier moduyla yakalanıyor.
- Antisimetrik orbit büyük + geniş-bantlı → f, deflektörlerdeki **yerel alan
  ağırlığına** bağlı; global Fourier modları onu kaçırıyor.
- Doğru form muhtemelen **∮ W(s)·(geometrik faz yoğunluğu) ds**, W(s) lattice alan
  profili. Bu, ancak **analitik türetmeyle** (spin EOM'undan Berry fazı) bulunur.

## 5. Açık problem ve tek temiz yol

**Soru:** Berry fazını proton spin denkleminden (Thomas-BMT) türetip, kapalı
yörünge x_CO(s), y_CO(s) cinsinden kapalı-form fonksiyoneli yaz. Beklenen form:
ardışık quad'ların (x,y)'siyle kurulan **yönlü alan** tipi bir toplam (Σ x1y2−x2y1
lead'i bunu destekliyor). Türetilirse:
- f'i ölçülen yörüngeden gerçek-zamanlı öngörmek (izleyip çıkarmak) mümkün olabilir.
- BPM örnekleme/gürültü sınırları nicel olarak sınanabilir.

**Türetilmeden empirik kovalama önerilmez** — bu veriyle yakınsamıyor ve bizi
trim/Omarov çıkmazına geri sokuyor.

## 6. Ham vs düzeltme-sonrası (sağlam kalan ayrım)

- **Ham sahte EDM'yi ANTİSİMETRİK (orbit-görünür) domine eder** (~37×): büyük
  kapalı yörünge → büyük geometrik faz. (`fedm_vs_shat.npy`, analitik R teyidi.)
- Deneyde yörünge düzeltilince antisimetrik kısım silinir; geriye **simetrik
  (orbit-kör) artık** kalır → no-go (`false_edm_harmonic_sinir.md §14`). Deneyin
  asıl sınırı bu artıktır; onu kapatmak spin (Omarov) ister.

## 7. İleri-ölçüm no-go'yu atlar mı? — DÜRÜST CEVAP: bilinmiyor

No-go *inversiyon* sınırıdır (R⁻¹ simetrikte κ≈193). İleri-ölçüm inversiyon
yapmaz, ama ancak doğru fonksiyonel biliniyorsa işe yarar. Simetrik f yörünge
ince-yapısında *kısmen* var (Fourier R²=0.87) ama fonksiyonel pinlenmediğinden
"ileri-ölçüm no-go'yu atlıyor" **kanıtlanmadı**. (Önceki turda fazla iddialıydım;
düzeltildi.)

## 8. Reprodüksiyon

Keşif kodu `/tmp/spin_meas/` (kalıcı repoda değil):
- `false_edm_4d.py`, `false_edm_mode_scan.py` — git'ten restore (`5cba757`,
  `41b1c6a~1`); doğrulanmış estimator (4D CO + model-fit eğim).
- `run1_gen.py`/`run2_gen.py` — veri üretimi (f + tam kapalı yörünge).
- `run2_data.npz` — 40 config (28 sim + 12 antisim): f, ince yörünge, kaçıklıklar.
- `run2_analyze.py` + permütasyon/per-hücre analizleri.

---

## 9. NEREDE DURUYORUZ (2026-06, make-or-break + Omarov karşılaştırması)

> Tam analiz ve Omarov makalesinin (PRD 105,032001) bu konudaki konumu için
> **`omarov.md` §9–10**'a bakın. Burası kısa durum kaydı.

**Estimator doğrulandı (kritik temel):** σ=10→5→2.5 μm ölçeklemesinde sahte EDM
**p = 2.00 ± 0.01** (her seed) → saf kuadratik geometrik faz, lineer kaçak YOK.
CO+model-fit yöntemi Omarov Fig. 9a'sını birebir üretir. Bu, CLAUDE.md'deki
"tek-parçacık CO=True kullanma" uyarısının aksine, **CO+model-fit'in geometrik
fazı doğru ölçtüğünü** ampirik kanıtlar (4-katlı simetrik örnekleme şart değil).

**Bu oturumun nicel zinciri (EDMSwitch ile):**
- Gerçek EDM (η=1.88e-15) = **9.81×10⁻¹⁰ rad/s**, TEK/diferansiyel ((CW−CCW)/2;
  Omarov Eq. C1 ile birebir).
- 10 μm sahte EDM (seküler) ~10⁻⁶ → gerçek sinyali ~1000× gömer.
- **CW/CCW telafisi tek başına: 3.4×** (artık = 474× EDM).
- **Orbit-düzeltme (antisim çıkar) ek 7.7×** (artık = 62× EDM). → **doğrudan
  make-or-break: monitör DEĞERLİ; korkulan "kazanç~1" GERÇEKLEŞMEDİ.**
- Kalan **simetrik orbit-kör artık 62× EDM** → paylaşılan sınır (biz + SQUID-BPM
  + SBA + CW/CCW hepsi kör).
- **Dejenerasyon:** idealize FODO'da **CCW ≡ CW+polarite-flip** (Eq. C2 4'lü → 2'li).

**Konumlandırma (düzeltildi):** Orbit-monitör = **SQUID-BPM ucuz ikamesi** (SBA
tamamlayıcısı DEĞİL — SBA E-alan/vertical-velocity hizalar, geometrik fazı
düzeltmez). Omarov geometrik fazı CW+CCW+polarite + CR-ayrım küçültmeyle hedefin
altına indirdiğini **fiziksel** gösteriyor, ama (i) polarite dejenerasyonu, (ii)
simetrik alt-uzayın izole edilmemesi, (iii) **CR-ayrım ÖLÇÜMÜNÜN (48-BPM/SQUID-BPM
+ K-mod reconstruction) test edilmemesi** — üç noktada prosedür açık (omarov.md §9).
Özgün katkı adayı: bu üç boşluğu nicelleştirmek.
