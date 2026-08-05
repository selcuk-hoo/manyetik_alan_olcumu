# Figür Betikleri (kontrol kopyası)

Bu klasör, `makale_orbit_bastirma_en.tex` içindeki figürleri üreten betiklerin
**kontrol amaçlı kopyasıdır**. Asıl çalışan sürümler repo kökündedir; buradakiler
inceleme kolaylığı için kopyalandı (bu klasörden çalıştırmak için veri yollarını
ayarlamak gerekebilir — aşağıya bakın).

## Betikler
- **`make_orbit_figures.py`** (634 satır) — 8 figür fonksiyonu; tüm makale figürlerini üretir.
- **`analytic_kmod.py`** — tek yerel bağımlılık (analitik Twiss/R matrisi; saf numpy).

## Figür → fonksiyon → veri haritası

| Makale figürü | `.tex` etiketi | Fonksiyon | Veri kaynağı |
|---|---|---|---|
| Fig. 1 (mod görünürlüğü) | `fig:fig_orbit_modes` | `fig_modes()` | `params.json` (analitik R + SVD, betik içi üretir) |
| Fig. 2 (4 kanal) | `Fig:channels` | `fig_channels()` | `kmod_drivers/paper_runs_results.json` `["channels"]` |
| Fig. 3 (BBA yakınsama) | `fig:bba_convergence` | `fig_bba_convergence()` | sabitler `BBA_F`/`BBA_F_RAW` (kaynak: json `["bba_iter_cpp"]`) |
| Fig. 4 (bastırma özeti) | `fig:orbit_suppression` | `fig_suppression()` | sabitler `RAW_F`/`MEAS_F` (kaynak: json `["sigma"]`) |
| Fig. 5 (CW/CCW toplam) | `fig:orbit_cw_ccw` | `fig_cwccw_sum()` | `R_dy_1.npy`, `R_dy_cw.npy` |
| Ek: nefes | `fig:breathing` | `fig_breathing(eps=0.02)` | betik içi üretir (`analytic_kmod`) |
| Ek: lock-in | `fig:orbit_lock` | `fig_lockin(nseed=40)` | betik içi üretir |

> **Not:** Betikte ayrıca `fig_sigma()` (→ `fig_orbit_sigma.png`) var ama bu figür
> **güncel makalede kullanılmıyor**.

## Veri bağımlılıkları
- **`params.json`** (repo kökü) — tek konfigürasyon kaynağı.
- **`kmod_drivers/paper_runs_results.json`** — kampanya sonuçları (`sigma`, `channels`,
  `bba_iter_cpp`). Fig. 3 ve Fig. 4'ün sayıları betikte **elle sabitlenmiş**; yorumları
  bu json'un ilgili anahtarlarına işaret ediyor (izlenebilirlik için).
- **`R_dy_1.npy`, `R_dy_cw.npy`** — tepki matrisleri; `.gitignore`'da, `build_response_matrix.py`
  ile üretilir (Fig. 5 için).

## Dikkat (bu kopyayı çalıştırırken)
- `make_orbit_figures.py` içinde `BASE` repo kökünü işaret eder ve veri dosyalarını
  `os.path.join(BASE, "kmod_drivers", ...)` ile açar. Yani **kopyadan çalıştırsan bile**
  veriyi repo kökünden okur (yol bozulmaz), ama `savefig` çıktısı çalıştırıldığı dizine
  düşer. `analytic_kmod` importu için bu klasör `sys.path`'te olmalı (aynı dizinde
  olduğundan kopyada da çalışır).
- Kanonik/çalışan sürüm her zaman **repo kökündeki** `make_orbit_figures.py`'dir;
  değişiklikleri orada yap, burası yalnız inceleme kopyasıdır.
