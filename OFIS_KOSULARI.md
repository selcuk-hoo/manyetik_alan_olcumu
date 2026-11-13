# Ofis bilgisayarında koşulacak simülasyonlar (makale revizyonu v2)

Dal: `claude/brave-allen-92wn61`. Önce: `git pull`, `bash build_integrator.sh`,
`g++ -O3 -shared -fPIC -std=c++17 revizyon/integrator_rev.cpp -o revizyon/lib_integrator_rev.so`.
Tepki matrisleri (`R_d?_1.npy`, `R_d?_cw.npy`) yoksa: `python3 build_response_matrix.py` (CW için `direction=+1`).
β-beat matrisleri yoksa: `cd revizyon && python3 build_rtrue.py --sig-g 0.002`.

| # | Komut | Ne üretir | Makaledeki yeri | Süre (4 çekirdek, tahmin) |
|---|-------|-----------|-----------------|------------------------|
| 1 | `cd revizyon && python3 makale_yeniden_olcum.py -w N` | Şekil 2 (kanallar, `ch`), Şekil 4 tek demet σ=5,20 μm (`oc1`), iki demet %0.2 gradyan hatası 15 seed (`tb20`), ham CW (`rawcw`) | Sec. III.B, Fig. 4 | ~2 saat (kaldığı yerden devam eder) |
| 2 | `python3 classic_bba_iter_donmus.py -w N --passes 8 --bbeat 0.01 --rmatrix cpp` | BBA yakınsaması, donmuş spin ölçümüyle; geçiş başına kalan desenler `revizyon/sonuclar/bba_iter_cpp_donmus_state.json` ve `kmod_drivers/paper_runs_results.json["bba_iter_cpp_donmus"]` | Sec. IV.A (TODO) | ~19 saat (geçiş başına ~2.3 saat; `--resume` ile devam) |

Sonuçlar gelince: (1) bitince bana haber ver; Şekil 2'yi (`fig_v2_channels.png`) ve Şekil 4'ü üretip metni
güncelleyeceğim (`\IfFileExists` sayesinde kanal şekli dosya oluşunca makaleye kendiliğinden girer).
(2) bitince BBA paragrafındaki sahte EDM ifadesini ve TODO'yu güncelleyeceğim.
