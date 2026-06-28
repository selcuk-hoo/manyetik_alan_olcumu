# kmod_drivers/ — all-quad AC-BBA estimator sürücüleri (sistematik bütçe)

Bu klasör, `kmod_bba_sonuclar.md`'deki **sistematik bütçe** (β-beating, quad-tilt,
CW/CCW) sayılarını üreten C++ spin-estimator sürücülerini içerir. Repoda kalıcıdır
(paper claim'lerini ürettikleri için; `/tmp` ephemeral'dir → kaybolur).

Hepsi `berry_data/false_edm_4d.py` çekirdek estimator'ını (4D-CO + model-fit) ve
`ac_bba_linchpin.py`'ı kullanır. Çıktılar `/tmp/kmod_recover/` altına yazılır
(JSONL/npy; `.gitignore`'da).

## Önemli: azaltılmış ayar
`fast_est.fast_measure` estimator'ı **azaltılmış ayarla** çağırır:
`n_turns=14` (CO-bulma), `t2=3e-4` (orijinal 28 / 5e-4). Bu, ensemble koşumlarını
~2× hızlandırır. **Geçerlilik:** `calib` modu σ²-üssünü p=2.002 ile aynı koşumda
doğrular → azaltılmış ayar göreli karşılaştırmalar için geçerli. Mutlak ölçek
`A_eff ≈ 1.18×10⁴ rad/s/m²`.

## Sürücüler

| Dosya | Ne üretir | Çalıştırma |
|-------|-----------|------------|
| `fast_est.py` | calib (p=2.002, A_eff), sweep (β-beating linchpin), tilt (tilt taraması) | `python3 kmod_drivers/fast_est.py {calib,sweep,tilt} -w 4` |
| `cwccw_ens.py` | 20-seed CW/CCW+flip tilt-cancellation (EVEN/ODD ayrımı, dejenerasyon) | `python3 kmod_drivers/cwccw_ens.py --nseed 20 -w 4` |
| `tiltscan.py` | 6-seed eşlenik ψ-taraması (0.2 mrad'da sahte-EDM 1 nrad/s'de mi) | `python3 kmod_drivers/tiltscan.py` |

Her uzun koşum artımlı JSONL yazar (kesilse bile kısmi sonuç korunur); sonradan
`--analyze-only` ile özet alınır:
```bash
python3 kmod_drivers/cwccw_ens.py --analyze-only
python3 kmod_drivers/tiltscan.py  --analyze-only
```

## Maliyet
Tek estimator koşumu ~250 s (azaltılmış ayar, 4 çekirdek). Ensemble'lar:
cwccw_ens 80 koşum ~50 dk; tiltscan 24 koşum ~25 dk. `bash build_integrator.sh`
önce çalıştırılmış olmalı (C++ kütüphane).

## Sonuç haritası (hangi sürücü → `kmod_bba_sonuclar.md` hangi bölüm)
- `fast_est calib/sweep` → §4 (linchpin, p=2.002, β-beating)
- `fast_est tilt` → §7.3(b) (tilt doğrudan kanalı)
- `cwccw_ens` → §7.3.1 (CW/CCW+flip tilt'i gidermez; ODD/EVEN≈1.3–1.9)
- `tiltscan` → §7.3(c) (0.2 mrad → ~1 nrad/s; tilt katkısı ≈0)
