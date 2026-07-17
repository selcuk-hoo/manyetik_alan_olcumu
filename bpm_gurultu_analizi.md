# bpm_gurultu_analizi.md — Sahte-EDM'in BPM Gürültüsüne Duyarlılığı

**Amaç (hakem hazırlığı):** "BBA'daki BPM gürültüsü sahte-EDM tahminini ne kadar
bozar?" sorusuna hazır, yeniden-üretilebilir cevap. Rastgele gürültünün averajla
1/√N azalması ders kitabıdır (figür GEREKMEZ); **önemli olan gürültünün sinyale
taşınma duyarlılığı** ve ondan çıkan **operasyonel gereksinim**.

> Makale gövdesine figür/satır olarak KONULMADI (yazar tercihi). §IV'te zaten
> nitel olarak var: "beyaz BPM gürültüsü ~1μm/okuma, averajla azaltılır; ~1 kHz
> lock-in ile dakikalar içinde ~14 nm". Bu belge o cümleyi nicel destekler.

---

## Sonuç: $f(\sigma_{\rm read})$ — 5 nokta

Koşum: `classic_bba_full.py`, seed 0, %5 β-beat (%1 gradyan), 100 μm BPM ofset
(iptal), 1 μm/okuma beyaz gürültü, **tek-geçiş** BBA. $\sigma_{\rm read} =
1\,\mu\text{m}/\sqrt{N_{\rm avg}}$ (averaj-sonrası okuma gürültüsü).

| $N_{\rm avg}$ | $\sigma_{\rm read}$ | sym artık $dx$/$dy$ | sahte-EDM $f$ | taban katı |
|---:|---:|---:|---:|---:|
| 1 | 1000 nm | 17.98 / 8.01 μm | **3888×** | 9.3× |
| 10 | 316 nm | 10.46 / 2.27 μm | **803×** | 1.9× |
| 100 | 100 nm | 6.04 / 1.20 μm | **569×** | 1.36× |
| 1 000 | 32 nm | 4.47 / 1.06 μm | **400×** | 0.95× |
| 10 000 | 10 nm | 4.11 / 1.05 μm | **390×** | 0.93× |
| gürültüsüz | 0 | 4.11 / — | **419×** (taban) | 1× |

---

## Mekanizma

Gürültü, BBA merkez-bulmanın (sıfır-geçişi) hatası üzerinden **simetrik artığa
karesel eklenir**:

$$\sigma_{\rm sym}^2 = \sigma_{\rm sys}^2 + \sigma_{\rm noise}^2,\qquad
\sigma_{\rm sys}\approx 4.1\,\mu\text{m},\qquad \sigma_{\rm noise}\propto\sigma_{\rm read}$$

ve sahte-EDM $f\propto\sigma^2$ (geometrik faz) olduğundan gürültü $f$'i tabanın
üstüne iter. Çıkarılan gürültü-artığı: $\sigma_{\rm read}=$ 1000/316/100/32 nm →
$\sigma_{\rm noise}\approx$ 17.5/9.6/4.4/1.7 μm.

---

## Operasyonel gereksinim

$$\boxed{\ \sigma_{\rm read}\lesssim 30\ \text{nm}\quad (N_{\rm avg}\gtrsim 10^3)\ }$$

Bu, gürültü-kaynaklı sahte-EDM'i sistematik tabanın altında tutar. **Rahatça
karşılanır:** §IV lock-in (~14 nm) eşiğin altında. Averajsız ($\sigma_{\rm
read}=1\,\mu$m) $f$ tabanın ~9 katına fırlar → averaj zorunlu.

---

## Dürüstlük uyarısı

Bu tarama **tek-geçiş** BBA tabanı (419×) üstünde ölçüldü — nihai şema (iterasyon
+ orbit-düzeltme) tabanı çok daha düşük (hedef-altı, bkz. Fig-1 ensemble). Gürültü
**ölçeklemesi** ($\sigma_{\rm noise}\propto\sigma_{\rm read}$, eşik $N_{\rm
avg}\gtrsim10^3$) şemadan bağımsız geçerlidir; ama gürültüyü *nihai* tabanın
altında tutmanın kesin sayısı için tarama iteratif pipeline üstünde tekrarlanmalı
(çok daha pahalı, henüz yapılmadı).

---

## Yeniden üretme (hakem sorarsa)

```bash
python3 classic_bba_full.py -w 7 --navg 1,10,100,1000,10000 --seed 0
```

- **Ham veri:** `kmod_drivers/paper_runs_results.json` → `["bba_full_syst"]`
  (anahtarlar: `f_bbeat_noiseless`, `noise.{N_avg}.{sig,sym_dx,sym_dy,f_cpp}`).
- **Betik:** `classic_bba_full.py` (gürültü sweep'i satır 207–247;
  `estimate(by, meta, noise_sigma=sig, ...)` merkez-bulmaya gürültü enjekte eder,
  averaj K=60; C++ spin `fast_measure` ile f çapası).
- **Süre:** ~ ilk BBA (376 CO-oturtmalı C++ koşumu) + N_avg başına 1 spin ölçümü.
