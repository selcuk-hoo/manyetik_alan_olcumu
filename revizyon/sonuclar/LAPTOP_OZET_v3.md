# Laptop üçüncü tur — kusurlu polarite özeti

Tarih: 21.11.2026 05:47. Makine: selcuk-inspiron, 12 çekirdek, -w 10. Betik: revizyon/laptop_v3.sh.
Ayrıntı: LAPTOP_PROMPT_v3.md. E = ½|odd₊ + odd₋|, hedef (1e-9) biriminde.

```
E = ½|odd₊(S0) + odd₋(durum)|  (hedef biriminde; odd₊: birinci koşu, normal polarite)

σ_d = 0: n=3  |odd₊| (polarite çevirmeden C) medyan 0.0221
  kusursuz çevirme                 n= 3 medyan 0.0148  %84 0.0181  maks 0.0196  hedef-üstü 0
  η = 1e-4                         n= 3 medyan 0.011  %84 0.0218  maks 0.0269  hedef-üstü 0
  η = 1e-3                         n= 3 medyan 0.0223  %84 0.0275  maks 0.03  hedef-üstü 0
  Δq 0.3 μm, düzeltme yok          n= 3 medyan 0.0321  %84 0.0416  maks 0.046  hedef-üstü 0
  Δq, tek demet izleyici           n= 3 medyan 0.0746  %84 0.099  maks 0.111  hedef-üstü 0
  Δq, iki demet izleyici           n= 3 medyan 0.00773  %84 0.0115  maks 0.0133  hedef-üstü 0
  Δε,Δφ 1e-7, trim yok             n= 3 medyan 0.421  %84 0.896  maks 1.12  hedef-üstü 1
  Δε,Δφ, iki demet fit + trim      n= 3 medyan 0.014  %84 0.016  maks 0.0169  hedef-üstü 0

σ_d = 1e-06: n=6  |odd₊| (polarite çevirmeden C) medyan 2.79
  kusursuz çevirme                 n= 6 medyan 0.00935  %84 0.0138  maks 0.0175  hedef-üstü 0
  η = 1e-4                         n= 6 medyan 0.0287  %84 0.0326  maks 0.0329  hedef-üstü 0
  η = 1e-3                         n= 6 medyan 0.0307  %84 0.0347  maks 0.0407  hedef-üstü 0
  Δq 0.3 μm, düzeltme yok          n= 6 medyan 3.7  %84 5.32  maks 7.66  hedef-üstü 5
  Δq, tek demet izleyici           n= 6 medyan 1.13  %84 1.82  maks 2.2  hedef-üstü 3
  Δq, iki demet izleyici           n= 6 medyan 0.0254  %84 0.0432  maks 0.0841  hedef-üstü 0
  Δε,Δφ 1e-7, trim yok             n= 6 medyan 0.701  %84 1.29  maks 1.88  hedef-üstü 3
  Δε,Δφ, iki demet fit + trim      n= 6 medyan 0.197  %84 0.246  maks 0.281  hedef-üstü 0

σ_d = 1e-05: n=12  |odd₊| (polarite çevirmeden C) medyan 446
  kusursuz çevirme                 n=12 medyan 0.0224  %84 0.0325  maks 0.0582  hedef-üstü 0
  η = 1e-4                         n=12 medyan 0.0734  %84 0.13  maks 0.184  hedef-üstü 0
  η = 1e-3                         n=12 medyan 0.693  %84 1.11  maks 1.25  hedef-üstü 4
  Δq 0.3 μm, düzeltme yok          n=12 medyan 43.7  %84 77.6  maks 200  hedef-üstü 12
  Δq, tek demet izleyici           n=12 medyan 11.5  %84 18.1  maks 25.8  hedef-üstü 11
  Δq, iki demet izleyici           n=12 medyan 0.23  %84 0.503  maks 0.706  hedef-üstü 0
  Δε,Δφ 1e-7, trim yok             n=12 medyan 2.91  %84 6.48  maks 8.86  hedef-üstü 9
  Δε,Δφ, iki demet fit + trim      n=12 medyan 1.89  %84 2.54  maks 3.47  hedef-üstü 8

σ_d = 3e-05: n=6  |odd₊| (polarite çevirmeden C) medyan 3.67e+03
  kusursuz çevirme                 n= 6 medyan 0.0435  %84 0.0636  maks 0.0672  hedef-üstü 0
  η = 1e-4                         n= 6 medyan 0.573  %84 0.822  maks 0.838  hedef-üstü 0
  η = 1e-3                         n= 6 medyan 5.37  %84 7.72  maks 8.17  hedef-üstü 6
  Δq 0.3 μm, düzeltme yok          n= 6 medyan 111  %84 159  maks 230  hedef-üstü 6
  Δq, tek demet izleyici           n= 6 medyan 37  %84 53.4  maks 68.6  hedef-üstü 6
  Δq, iki demet izleyici           n= 6 medyan 0.629  %84 1.82  maks 2.3  hedef-üstü 2
  Δε,Δφ 1e-7, trim yok             n= 6 medyan 11.4  %84 20.9  maks 24.5  hedef-üstü 6
  Δε,Δφ, iki demet fit + trim      n= 6 medyan 6.37  %84 7.69  maks 8.17  hedef-üstü 6

σ_d = 0.0001: n=6  |odd₊| (polarite çevirmeden C) medyan 4.07e+04
  kusursuz çevirme                 n= 6 medyan 0.0116  %84 0.0558  maks 0.0861  hedef-üstü 0
  η = 1e-4                         n= 6 medyan 6.22  %84 8.64  maks 8.67  hedef-üstü 6
  η = 1e-3                         n= 6 medyan 62  %84 86.4  maks 88  hedef-üstü 6
  Δq 0.3 μm, düzeltme yok          n= 6 medyan 372  %84 527  maks 768  hedef-üstü 6
  Δq, tek demet izleyici           n= 6 medyan 124  %84 178  maks 229  hedef-üstü 6
  Δq, iki demet izleyici           n= 6 medyan 2.25  %84 5.97  maks 7.63  hedef-üstü 4
  Δε,Δφ 1e-7, trim yok             n= 6 medyan 38  %84 68.5  maks 79.3  hedef-üstü 6
  Δε,Δφ, iki demet fit + trim      n= 6 medyan 21.3  %84 25.6  maks 27  hedef-üstü 6
```

Hata/uyarı satırları (laptop_v3.log):
