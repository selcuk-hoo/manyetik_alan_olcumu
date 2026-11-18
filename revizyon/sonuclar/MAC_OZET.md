# Mac koşuları özeti (MAC_PROMPT.md) — ARA SÜRÜM

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
- Başlangıç 18.11.2026 12:26. **Devam ediyor** (tahmin ~19 saat).
