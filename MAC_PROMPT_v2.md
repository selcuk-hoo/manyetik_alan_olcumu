# Mac (ofis) ikinci tur koşular — Claude Code'a verilecek prompt

İlk tur: `MAC_PROMPT.md` (kanallar, tb20, BBA; sonuçlar `revizyon/sonuclar/MAC_OZET.md`).
Bu tur iki iş içerir; ikisi de `revizyon/ofis_v2_kosulari.py` ile çalışır (tahmini toplam ~2–3 saat, 8 çekirdek).

## A) Terminal (Mac), bir kez
```bash
cd manyetik_alan_olcumu
git pull origin claude/brave-allen-92wn61
caffeinate -dims &
claude          # gerekirse: claude --dangerously-skip-permissions
```

## B) Claude Code'a yapıştırılacak prompt
```
Bu repo proton EDM halkası makalesinin (arXiv:2608.04647) revizyonu için simülasyon kodudur.
Dal: claude/brave-allen-92wn61. Ofis dışındayım; sorma, kendi başına yürüt. Konuşma ve raporlar Türkçe.
Bu, ikinci tur ofis koşusudur; ilk tur MAC_PROMPT.md ile yapıldı (özet: revizyon/sonuclar/MAC_OZET.md).

KURALLAR
- Mevcut kodu bozma; betik değiştirmen gerekirse kopyasını al. Hata ya da negatif sonucu olduğu gibi yaz.
- Seed, parametre, süre ve anormallikleri kaydet. İşler bitince ya da takılınca kısa özet yaz.
- İki işi PARALEL çalıştırma (çekirdekler yarışır).

ADIMLAR
1. `git pull origin claude/brave-allen-92wn61`; MAC_PROMPT_v2.md ve revizyon/ofis_v2_kosulari.py
   başındaki açıklamayı oku.
2. Motorları yeniden derle (ilk turdan beri integrator_rev.cpp değişmedi ama emin olmak için):
   `bash build_integrator.sh`
   `cd revizyon && clang++ -O3 -shared -fPIC -std=c++17 integrator_rev.cpp -o integrator.dylib && cd ..`
   Doğrula: `cd revizyon && python3 -c "import integrator_rev, ofis_v2_kosulari; print('ok')"`.
   β-beat matrisleri (revizyon/sonuclar/Rtrue_*_g20.npy) ve saptırıcı matrisleri (D_eps_*, D_roll_*,
   *_g20.npy) yoksa: `cd revizyon && python3 build_rtrue.py --sig-g 0.002 -w 8` ve OFIS_KOSULARI.md'deki
   adımlar. Tepki matrisleri (R_d?_1.npy, R_d?_cw.npy) yoksa: `python3 build_response_matrix.py`.
3. İş B (polarite, ~30 dk) ÖNCE:
   `cd revizyon && nohup python3 ofis_v2_kosulari.py polar -w 8 > polar.log 2>&1 &`
   Bitince `python3 ofis_v2_kosulari.py ozet` çıktısını kaydet, commit + push et.
4. Sonra İş A (haftalık senaryo, ~2 saat):
   `nohup python3 ofis_v2_kosulari.py hafta -w 8 > hafta.log 2>&1 &`
   Kesilirse aynı komutla yeniden başlat (kaldığı yerden devam eder).
5. Her ~2 saatte bir ve her iş bitince: revizyon/sonuclar/donmus_adim6_kisa.jsonl,
   revizyon/sonuclar/saptirici_testleri.jsonl, revizyon/sonuclar/dinamik_saptirici_lin.json dosyalarını
   `git add -f` ile ekle, "test: <ne>" ön ekiyle commit et, `git push origin claude/brave-allen-92wn61`
   (başarısızsa 4 kez artan bekleyişle dene; çakışmada `git pull --rebase` sonra tekrar).
6. Hepsi bitince `revizyon/sonuclar/MAC_OZET_v2.md` yaz: her iş için süre, seed aralığı, `ozet` çıktısı
   (hafta: W1-E2 ve W1-E4 için n, medyan, %84, maks, hedef-üstü sayısı; polar: |odd+|, |odd-|, Omarov
   birleşimi, korelasyon), anormallikler. Commit + push et.
```

## C) Ne öğreneceğiz
- **İş A:** İki demetin tek demete üstünlüğü (şimdi 15 değer: medyan 0.65 vs 0.05, maks 2.1 vs 0.17)
  25 desende de tutuyor mu? Makalenin yeni çerçevesi buna dayanıyor.
- **İş B:** Kuadrupol polaritesini çevirmek saptırıcı ε×φ sahte EDM'sini söndürüyor mu? Söndürüyorsa
  Omarov'un şeması bu sistematikten muaf, bizimki değil → makalede açıkça yazılmalı. Söndürmüyorsa
  saptırıcı gereksinimi (~1e-7 / trim aralığı) iki şema için de geçerli bir halka sistematiğidir.
- Birleşik senaryo (saptırıcı + yer hareketi + BPM kayması) bilinçli olarak sona bırakıldı.

Sonuçlar push edilince bu oturuma "Mac v2 sonuçları geldi" yaz.
