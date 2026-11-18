# Mac koşuları özeti (MAC_PROMPT.md)

Makine: iMac M1, 8 çekirdek (`-w 8`), macOS 13. Python: conda `ml` ortamı (numpy 2.4.6, scipy 1.17.1).
Dal: `claude/brave-allen-92wn61`, başlangıç commit'i `70c7178`. Motorlar Mac'te clang++ ile yeniden derlendi
(`integrator.dylib`, `revizyon/integrator.dylib`; ikisi de commit edilmedi).

## 0) β-beat matrisleri — `build_rtrue.py --sig-g 0.002 -w 8`
- 18.11.2026 11:58–12:00 (CCW 59 s, CW 59 s). Gradyan deseni rng(0), σ_g=0.002, quad 0 = 0.
- Nominal R'den göreli fark (Frobenius): x ccw 1.57 %, x cw 1.71 %, y ccw 1.92 %, y cw 1.53 %. NaN yok.
- Çıktı: `revizyon/sonuclar/Rtrue_d{x,y}_{ccw,cw}_g20.npy` (npy, git'te değil).

## 1) `makale_yeniden_olcum.py -w 8`
- 18.11.2026 12:00–12:26 (**26 dk**; tahmin ~2 saatti). 74/74 görev tamam, günlükte hata yok.
- Çıktı: `revizyon/sonuclar/makale_yeniden_olcum.jsonl`. `f_target` = f / 1e-9.
- Gürültü: BPM 14 nm (seed 50000+s), rcond = 0.01.

| tür | σ (μm) | kanal | yön | n | ortalama f* | RMS f* |
|-----|-------|-------|-----|---|-----------|--------|
| ch | 10 | full | −1 | 3 | −109.1 | 462.6 |
| ch | 10 | aa | −1 | 3 | −88.96 | 381.4 |
| ch | 10 | sa | −1 | 3 | 38.72 | 80.39 |
| ch | 10 | as | −1 | 3 | −58.89 | 63.55 |
| ch | 10 | ss | −1 | 3 | 0.044 | 4.797 |
| oc1 | 5 | − | −1 | 3 | 0.0029 | 0.629 |
| oc1 | 5 | − | +1 | 3 | −2.349 | 10.84 |
| oc1 | 20 | − | −1 | 3 | 0.090 | 10.22 |
| oc1 | 20 | − | +1 | 3 | −37.67 | 173.3 |
| rawcw | 2.5 | − | +1 | 3 | 11.38 | 30.58 |
| rawcw | 5 | − | +1 | 3 | 45.55 | 122.4 |
| rawcw | 10 | − | +1 | 3 | 182.2 | 489.4 |
| tb20 | 10 | − | −1 | 15 | 0.098 | 0.205 |
| tb20 | 10 | − | +1 | 15 | 0.071 | 0.196 |

σ² kontrolü (ch, seed 0, σ = 2.5→5→10 μm): sa, as, aa kanallarında oran 4.00–4.03; ss kanalında 3.79 / 4.09
(en küçük sinyal). Geometrik faz beklentisiyle uyumlu.

Not: oc1 sonucu yöne göre belirgin biçimde asimetrik (yön +1'in RMS'i yön −1'inkinin ~17 katı). Bu tek-demet
düzeltme yalnız CCW (R_1) matrisiyle yapıldığı için beklenen bir durum olabilir; yorumu makale tarafına bırakıyorum.

## 2) BBA — `classic_bba_iter_donmus.py -w 8 --passes 8 --bbeat 0.01 --rmatrix cpp --resume`
- 18.11.2026 12:26 – 19.11.2026 00:53 (**44 820 s, 12.5 saat**; tahmin ~19 saat). Hatasız, kesinti yok.
- Tek gerçekleşme: seed 0, kaçıklık σ = 10 μm, β-beat %1, relax 0.5, scan 150 μm.
- Çıktı: `revizyon/sonuclar/bba_iter_cpp_donmus_state.json`, `kmod_drivers/paper_runs_results.json["bba_iter_cpp_donmus"]`.

| geçiş | \|f\|/hedef | sim dx (μm) | sim dy (μm) | yörünge (μm) | min-kalite x/y |
|---|---|---|---|---|---|
| ham | 78 | – | – | ~48 | – |
| 1 | 4.17 | 4.27 | 3.269 | 24.9 | 0.10 / 0.99 |
| 2 | 13.40 | 2.90 | 1.697 | 14.1 | 0.18 / 1.00 |
| 3 | 9.60 | 2.29 | 0.908 | 8.6 | 0.28 / 1.00 |
| 4 | 4.37 | 1.92 | 0.500 | 5.4 | 0.52 / 1.00 |
| 5 | 0.01 | 1.51 | 0.283 | 3.5 | 0.44 / 1.00 |
| 6 | 1.21 | 1.25 | 0.164 | 2.3 | 0.50 / 1.00 |
| 7 | 1.09 | 1.09 | 0.096 | 1.5 | 0.57 / 1.00 |
| 8 | **0.72** | 0.90 | 0.057 | 1.0 | 0.70 / 1.00 |

Toplam bastırma 108×. Yatay düzlem dikeyden belirgin biçimde yavaş yakınsıyor.

## 3) Ek inceleme: geçiş 2'de sahte EDM neden arttı?
Her geçişin kalan deseni ss/sa/as/aa kanallarına (simetrik/antisimetrik dx × dy) ayrılıp donmuş-spin sahte EDM
işaretiyle yeniden ölçüldü (aynı ayar: CCW, 28 tur, 4 yineleme). Veri: `bba_gecis_kanallar.jsonl` (45 ölçüm).
Kanallar tam toplanabilir (geçiş 0: Σ = 77.92, tam = 77.91).

| geçiş | ss | sa | as | aa | tam | Σ\|f_ij\|/\|f\| |
|---|---|---|---|---|---|---|
| 0 | −4.89 | +35.44 | −80.52 | +127.89 | +77.91 | 3.2 |
| 1 | +4.04 | +6.65 | −31.76 | +16.91 | −4.17 | 14.2 |
| 2 | +2.63 | +2.29 | −13.89 | −4.44 | −13.40 | 1.7 |
| 3 | +1.39 | +1.58 | −6.27 | −6.30 | −9.60 | 1.6 |
| 4 | +0.71 | +1.10 | −2.39 | −3.80 | −4.37 | 1.8 |
| 5 | +0.30 | +0.48 | −0.22 | −0.55 | +0.01 | 105 |
| 6 | +0.10 | +0.11 | +0.41 | +0.60 | +1.21 | 1.0 |
| 7 | +0.01 | −0.07 | +0.40 | +0.75 | +1.09 | 1.1 |
| 8 | −0.02 | −0.11 | +0.26 | +0.60 | +0.72 | 1.4 |

- **Geçiş 2'deki artış bozulma değil**: geçiş 1'de aa (+16.9) baskın as kanalını (−31.8) rastlantısal olarak
  götürüyordu (sönüm 14). Geçiş 2'de her kanal küçüldü ama aa işaret değiştirdi (−4.4); götürme kalktı, toplam büyüdü.
  as geçiş 4'e kadar ~2.3×/geçiş düzgün küçülüyor (80.5 → 2.4). Geçiş 5'teki 0.01× de aynı türden rastlantı (sönüm 105).
- **Geçiş 6–8**: as ve aa işaret değiştirip ~0.3–0.8'de kalıyor; son değer 0.72× gerçek kalıntı (sönüm 1.4).
  Kalıntıyı antisimetrik dx içeren kanallar (as, aa) taşıyor; simetrik dx (0.90 μm) içeren ss/sa küçük.
- **Taban testi** (geçiş 8, `bba_gecis8_taban.json`): aa +0.599; dy işareti ters −0.600; dy=0 veya dx=0 → +0.002;
  tam desen +0.724, dy ters −0.728, dy=0 +0.002. Kalıntı tümüyle dx×dy çarpımı, taban yok. aa'nın RMS çarpımıyla
  küçülmemesi desen şeklinden (aa/RMS-çarpımı 4.2 → 17.8 μm⁻²). Hipotez (sınanmadı): dy'nin güçlü eşleşen bileşenleri
  en yavaş düzeliyor.
- **Öneri**: tek seed'de tek geçişin \|f\|'i çok oynak; makale için birden çok seed RMS'i ya da Σ\|f_ij\| (en kötü durum) kullanılmalı.
- Rapor (PDF): `revizyon/sonuclar/mac_rapor.pdf` (kaynak `mac_rapor.tex`, şekil `bba_kanallar.pdf`).

## Notlar
- `makale_yeniden_olcum.py` 2 saat yerine 26 dk'da bitti (74/74).
- Zincirdeki otomatik commit ilk iki seferde çalışmadı (`git add` listesinde henüz olmayan dosyalar); elle atıldı, veri kaybı yok.
- 18.11 13:45'te yerel e-posta taraması (Qwen 9B) BBA ile ~1.5 saat çakıştı; kullanıcı isteğiyle durduruldu. BBA etkilenmedi.
- Mac'te polyglossia (makecmds eksik) derlenmedi; rapor babel + XeLaTeX ile derlendi.
