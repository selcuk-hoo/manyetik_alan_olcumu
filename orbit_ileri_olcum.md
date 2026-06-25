# Kapalı-Yörünge İleri-Ölçümüyle Sahte EDM: Standart BPM, SQUID Yerine Geçer mi?

> **Kısa cevap:** Sahte EDM kapalı yörüngenin bir fonksiyonelidir (f ∝ ⟨x_CO·y_CO⟩);
> standart BPM'ler (48 adet, 1 μm gürültü) bu yörüngeyi öngörüyü bozmadan ölçer.
> No-go teoremi bir **inversiyon** (R⁻¹ ile kaçıklık geri-çatımı) sınırıdır; sahte
> EDM'yi ölçülen yörüngeden **ileri yönde** öngörmek inversiyona girmez ve no-go'ya
> takılmaz — simetrik kör-noktaya bile kısmen erişir. SQUID-BPM'nin modülasyon +
> düşük-mertebe harmonik yaklaşımının aksine, burada ekstra donanım gerekmez.

Bu günlük, drift-monitör çalışmasının (`drift_monitor/`) spin tarafına bağlanması
sorusunu inceler: "Kapalı yörüngeyi gören bir BPM sistemi spini (sahte EDM'yi) ne
kadar görür?" — kullanıcının SQUID-BPM ile karşılaştırma ve "standart BPM yeter"
tezini test eder.

---

## 1. Soru ve bağlam

SQUID-tabanlı BPM önerisi modülasyon + düşük-mertebe (n≈2) Fourier harmoniği
ölçümüne dayanır. Bu projede iki şey gösterildi: (a) düşük-mertebe harmonikleri
tek başına ölçmek simetrik (yüksek-k) kaçıklığı kurtarmaz — çok yüksek mertebe
gerekir; (b) k-modülasyon operasyonel olarak hem ağır hem yetersiz
(`omarov_symmetric_hybrid.md §9`). Açık soru: **standart BPM'ler SQUID yerine
geçip sahte-EDM sistematiğini izleyebilir mi?**

Asıl risk (kullanıcının işaret ettiği): spin ile demet farklı dinamiklere sahip
olabilir → kapalı yörüngeyi gören BPM, spini görmeyebilir. Bu günlük o riski
sayısallaştırır.

---

## 2. Kurulu fizik (önceki çalışmadan)

- Sahte EDM (dikey spin presesyonu dS_y/dt) **dx·dy geometrik (Berry) faz**
  kanalından gelir, misalignment ile **kuadratik (σ²)** ölçeklenir
  (`false_edm_harmonic_sinir.md §13`).
- Mekanizma kapalı yörüngenin çarpımıdır: **f ∝ x_CO·y_CO** (§13.4–13.6). Bu
  günlükte sayısal olarak da doğrulandı (bkz. §4): ⟨x·y⟩ fonksiyoneli f ile
  korele; ∮x dy, ∮y dx gibi diğer geometrik adaylar ~0.
- Ölçüm reçetesi: 4D kapalı yörüngede tek ideal parçacık + model-fit seküler eğim
  (`false_edm_4d.py`, `false_edm_mode_scan.py` — git'ten /tmp/spin_meas'e
  restore edildi).

---

## 3. Kritik ayrım: HAM sahte EDM vs DÜZELTME-SONRASI artık

Belgelerin (eski CLAUDE.md dahil) "sahte EDM'yi simetrik alt-uzay domine eder"
ifadesi **yanıltıcıydı**. Veri ve analitik R ile doğru tablo:

- **Ham (düzeltme öncesi) sahte EDM'yi ANTİSİMETRİK alt-uzay domine eder.**
  Çünkü f ∝ x_CO·y_CO ve antisimetrik kaçıklık BÜYÜK kapalı yörünge üretir
  (düşük-k, rezonant, G_k büyük). Antisimetrik = orbit-GÖRÜNÜR.
- **Deneyde kapalı yörünge BPM+düzelticilerle sıfıra çekilir.** Bu, orbit-görünür
  (antisimetrik) kısmı siler. Geriye kalan **simetrik (orbit-kör) artık**tır →
  no-go (`§14`). Yani belgelerdeki "simetrik domine" ifadesi aslında
  **düzeltme-sonrası artık** için doğru, ham sinyal için değil.

Sayısal teyit (analitik R, dy deseninin ürettiği kapalı yörünge RMS'i):

| Desen | Orbit RMS | BPM görür mü? | Ham \|dS_y/dt\| (Ŝ-taraması) |
|------|------|------|------|
| Antisimetrik (Ŝ=−1) | 7.9×10⁻⁵ m | **GÖRÜR** | **3.6×10⁻⁶** rad/s |
| Karışık | 5.6×10⁻⁵ m | kısmen | 2.0×10⁻⁶ |
| Simetrik (Ŝ=+1) | 2.0×10⁻⁵ m | zayıf | 9.5×10⁻⁸ |

---

## 4. Bulgular

### Test A — Sahte EDM hangi simetri bileşeninden? (mevcut `fedm_vs_shat.npy`)
Antisimetrik (orbit-görünür) ham sahte EDM'yi ~37× domine eder. → Orbit'in
gördüğü kaçıklıklar, sahte EDM'yi en çok sürenlerdir (ikisi de aynı kapalı
yörüngeden beslenir).

### Test B — Ham sahte EDM'nin ne kadarı orbit-görünür bileşenden? (tracker, N=8)
Kaçıklığın yalnız antisimetrik (orbit-görünür) bileşeni alınıp f ölçülünce:
- **Pearson(f_full, f_görünür) = 0.98**, RMS oranı 0.92, işaret de takip ediyor.
→ Orbit, dominant sürücüyü hem görür hem öngörür.

### Run 1 — İleri-ölçüm: f, ÖLÇÜLEN yörüngeden öngörülür mü? (tracker, 24 config)
Her config için f_true + tam kapalı yörünge ölçülüp, doğru fonksiyonel ⟨x·y⟩'nin
f ile korelasyonu (48-BPM örnekleme + 1 μm gürültü dahil):

| Grup | corr(ince) | corr(48-BPM+gürültü) | R² | orbit |
|------|------|------|------|------|
| Antisim (görünür) | −0.78 | −0.81 | 0.65 | 81 μm |
| Karışık | −0.74 | −0.76 | — | 61 μm |
| **Sim (KÖR/no-go)** | −0.54 | −0.57 | **0.32** | 14 μm |
| Tümü | −0.73 | −0.75 | 0.57 | — |

---

## 5. Kavramsal sonuç: ileri-ölçüm ≠ inversiyon

No-go (`§14`, ~1.8×10⁻⁴ taban; 6 metot aynı tabana çarpar) bir **inversiyon**
ifadesidir: BPM yörüngesinden *kaçıklığı geri çatmak* (R⁻¹) simetrik alt-uzayda
kötü-koşulludur (κ≈193). Ama sahte EDM, kaçıklığın değil **kapalı yörüngenin**
fonksiyonelidir, ve BPM kapalı yörüngeyi *doğrudan* ölçer. Dolayısıyla:

- Sahte EDM'yi ölçülen yörüngeden **ileri yönde** öngörmek R⁻¹ inversiyonuna
  girmez → no-go onu yasaklamaz.
- 48-BPM + 1 μm gürültü öngörüyü bozmaz (corr −0.73 → −0.75).
- Simetrik kör-noktada bile orbit ölçülebilir (14 μm ≫ 1 μm) ve f'i kısmen
  öngörür (R²=0.32). No-go *inversiyon* için geçerli, *ileri-ölçüm* için değil.

**Katkı (Omarov'a göre özgün):** Omarov doğrudan spin/CR-ayrımı yoluna girer;
burada **orbit'in tam olarak neyi öngörebildiğine** kesin (kısmen) sınır konur:
standart BPM'ler dominant (antisimetrik) sürücüyü güçlü, simetrik artığı kısmen
öngörür — SQUID/modülasyon olmadan.

---

## 6. Çekinceler ve açık sorular

- **R²'ler alt sınırdır:** ⟨x·y⟩ doğru *form* ama ağırlıksız. Gerçek f, alanların
  etki ettiği yerlerde (kuadrupol/deflektör) ağırlıklıdır. Ağırlıklı fonksiyonel
  korelasyonu yükseltir. R²=0.32 (sim) ve 0.65 (antisim) **kaba alt sınırlar**.
- **İstatistik:** n=8/grup küçük; simetrik R²=0.32 güçlü işaret ama tek başına
  kesin değil (n=8 için geniş hata payı). Kesinleştirmek için ~60–100 config +
  ağırlıklı/öğrenilmiş fonksiyonel gerekir.
- **Düzeltme-sonrası rejim:** Bu testler ham (düzeltilmemiş) yörüngede. Deneyin
  asıl sınırı düzeltme-sonrası simetrik artıktır; ileri-ölçümün o rejimde de
  çalışıp çalışmadığı (orbit düzeltildikten sonra kalan küçük simetrik yörüngeyi
  BPM'lerle öngörmek) ayrı bir kesin testtir.
- **Drift izleme (Test D):** f ∝ ⟨x·y⟩ ve orbit ölçülebilir olduğundan, f'in
  *driftini* izlemek de aynı sadakatle (R²≈0.57) mümkün olmalı; ayrı pahalı koşum
  yerine bu ilişkiden çıkarsanabilir, ama doğrudan zaman-serisi gösterimi
  yapılabilir.

---

## 7. Reprodüksiyon

Keşif kodu kalıcı repoda tutulmaz; `/tmp/spin_meas/` altında:
- `false_edm_4d.py`, `false_edm_mode_scan.py` — git'ten restore (commit `5cba757`,
  `41b1c6a~1`); doğrulanmış sahte-EDM estimator'ı (4D CO + model-fit eğim).
- `test_A/B/C.py`, `run1_gen.py`, `run1_analyze.py` — bu oturumun testleri.
- Ham veri: `run1_data.npz` (24 config: f + tam kapalı yörünge + kaçıklıklar).

Çalıştırma: `_BASE` repo köküne yamalı; `PYTHONPATH` repo + /tmp/spin_meas.
Tracker: `bash build_integrator.sh` (her config ~birkaç dk; 4 worker paralel).
