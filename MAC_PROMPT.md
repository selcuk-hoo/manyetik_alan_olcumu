# Mac'te Claude Code'a verilecek prompt (uzun simülasyonlar)

## A) Bir kez, terminalde (Mac)
```bash
# İlk kez:
git clone -b claude/brave-allen-92wn61 https://github.com/selcuk-hoo/manyetik_alan_olcumu.git
cd manyetik_alan_olcumu
# Sonraki seferlerde güncellemek için:
#   cd manyetik_alan_olcumu && git pull origin claude/brave-allen-92wn61
pip3 install numpy matplotlib scipy        # eksikse
caffeinate -dims &                          # Mac uyumasın (ya da: Sistem Ayarları > Enerji > uyutma kapalı, fiş takılı)
claude                                      # Claude Code'u repo klasöründe aç
```
Uzaktan çalışacaksan: `claude --dangerously-skip-permissions` onay sormadan çalışır
(yalnız bu klasörde, güvendiğin bir ortamda kullan); ya da `.claude/settings.json` ile git/python/bash izinlerini ver.

## B) Claude Code'a yapıştırılacak prompt
```
Bu repo, proton EDM halkası makalesinin (arXiv:2608.04647) revizyonu için simülasyon kodunu içerir.
Dal: claude/brave-allen-92wn61. Ofis dışındayım; sorma, kendi başına yürüt. Konuşma/rapor Türkçe olsun.

KURALLAR
- Mevcut kodu bozma; yalnız yeni dosya ekle. Betik değiştirmen gerekirse kopyasını al.
- Hata/negatif sonucu olduğu gibi raporla; sonuçları (seed, parametre, belirsizlik) kaydet.
- Birkaç dakikada bir değil, işler bitince ya da takılınca bana kısa özet yaz.

ADIMLAR
1. `git pull origin claude/brave-allen-92wn61`; `OFIS_KOSULARI.md` dosyasını oku.
2. Derle (macOS):
   - `bash build_integrator.sh`   (kökte integrator.dylib üretir)
   - `cd revizyon && clang++ -O3 -shared -fPIC -std=c++17 integrator_rev.cpp -o integrator.dylib && cd ..`
     (integrator_rev.py macOS'ta revizyon/integrator.dylib arar; kökteki dosya ile KARIŞTIRMA)
   - Doğrula: `python3 -c "import integrator"` ve `cd revizyon && python3 -c "import integrator_rev"`.
3. Tepki matrisleri (R_d?_1.npy, R_d?_cw.npy) ve β-beat matrisleri yoksa OFIS_KOSULARI.md'deki komutlarla üret.
4. Çekirdek sayısını bul (`sysctl -n hw.ncpu`), N olarak kullan. Sırayla çalıştır, arka planda, log dosyasına:
   a) `cd revizyon && nohup python3 makale_yeniden_olcum.py -w N > yo.log 2>&1 &`  (~2 saat; kaldığı yerden devam eder)
   b) (a) bitince: `nohup python3 classic_bba_iter_donmus.py -w N --passes 8 --bbeat 0.01 --rmatrix cpp --resume > bba.log 2>&1 &`
      (~19 saat; geçiş başına ~2.3 saat; kesilirse --resume ile devam)
   İşleri paralel çalıştırma (çekirdekler yarışır).
5. Süreç kesilirse (uyku, kapanma) yeniden başlat; sonuç dosyalarından devam edilir.
6. Her ~2 saatte bir ya da bir aşama bitince: sonuç dosyalarını (revizyon/sonuclar/*.jsonl, *_state.json,
   kmod_drivers/paper_runs_results.json) ve küçük özet .md dosyasını commit et ve push et:
   `git add -f` gerekirse; commit mesajı "test: <ne>" ön ekiyle; `git push origin claude/brave-allen-92wn61`
   (push başarısızsa 4 kez, artan bekleyişle dene; çakışmada `git pull --rebase` sonra tekrar).
7. Hepsi bitince `revizyon/sonuclar/MAC_OZET.md` yaz: her iş için süre, seed/parametreler, ana sayılar,
   anormallikler; commit + push et.
```

## C) Sonra bu oturumda (bulut)
"Mac sonuçları geldi" de; `git pull` ile alıp Şekil 2 (kanallar), Şekil 4 ve BBA paragrafını güncelleyip TODO'ları kaldırırım.
