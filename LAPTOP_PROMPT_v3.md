# Laptop üçüncü tur — kusurlu polarite testi, daha çok desen ve σ_d taraması

Önceki turlar: `MAC_PROMPT.md` (özet `revizyon/sonuclar/MAC_OZET.md`), `MAC_PROMPT_v2.md` (özet `MAC_OZET_v2.md`).
Bu tur: `revizyon/kusurlu_polarite.py`. Bulutta 3 desenle koşuldu (sonuç aşağıda); amaç sağlamlaştırmak.

**Neden:** Omarov'un polarite çevirmeli birleşimi E = ½(odd₊ + odd₋) modelimizde sahte EDM'yi tabana indiriyor.
Ama statik saptırıcı hatası σ_d = 1e-5 iken, iki polarite koşusu arasındaki 0.3 μm kuadrupol sürüklenmesi
E'yi hedefin 34 katına çıkarıyor; tek demetli izleyici 17'ye, iki demetli izleyici 0.48'e indiriyor (3 desen).
Makalenin yeni çerçevesi ("iki demetli izleyici, polarite çevirmenin gerektirdiği koşular arası kararlılığı
sağlar") buna dayanacak. Sonuç σ_d'ye çok duyarlı ve gerçek σ_d bilinmiyor → tarama gerekli.

Bulut sonucu (3 desen, σ_d = 1e-5, medyan): kusursuz çevirme 0.017; η=1e-3 0.70; Δq yok 34 / tek demet 17 /
iki demet 0.48; saptırıcı 1e-7 sürüklenmesi trim yok 5.1 / trim 2.3.

## En kolay yol: betik (önerilen)
Asistan gerekmez; her şey otomatik (güncelleme, derleme, iki iş sırayla, 2 saatte bir push, özet):
```bash
cd manyetik_alan_olcumu
git pull origin claude/brave-allen-92wn61      # betiği almak için (yerel .npy çakışırsa aşağıdaki A'ya bak)
nohup bash revizyon/laptop_v3.sh 10 > laptop_v3.log 2>&1 &
tail -f laptop_v3.log                          # ilerleme (Ctrl-C yalnız izlemeyi durdurur)
```
Kesilirse aynı `nohup` komutunu yeniden çalıştır. Bitince özet `revizyon/sonuclar/LAPTOP_OZET_v3.md`'de ve dala
itilmiş olur. Aşağıdaki A/B, Claude Code ile yaptırmak istersen içindir.

## A) Terminal (laptop, Linux), bir kez
```bash
cd manyetik_alan_olcumu
# Tepki matrisleri artık dalda (git add -f ile). Yerelde aynı adlı izlenmeyen dosyalar varsa pull reddeder;
# önce yedekle:
mkdir -p ~/npy_yedek && for f in R_dx_1.npy R_dy_1.npy R_dx_cw.npy R_dy_cw.npy revizyon/sonuclar/D_*.npy \
  revizyon/sonuclar/Rtrue_*_g20.npy; do [ -f "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1 \
  && mv "$f" ~/npy_yedek/; done
git pull origin claude/brave-allen-92wn61
claude          # gerekirse: claude --dangerously-skip-permissions
```

## B) Claude Code'a yapıştırılacak prompt
```
Bu repo proton EDM halkası makalesinin (arXiv:2608.04647) revizyonu için simülasyon kodudur.
Dal: claude/brave-allen-92wn61. Ben yokum; sorma, kendi başına yürüt. Konuşma ve raporlar Türkçe.
Bu, üçüncü tur koşudur: LAPTOP_PROMPT_v3.md ve revizyon/kusurlu_polarite.py'nin başındaki açıklamayı oku.

KURALLAR
- Mevcut kodu bozma; betik değiştirmen gerekirse kopyasını al. Hata/negatif sonucu olduğu gibi yaz.
- Seed, parametre, süre ve anormallikleri kaydet. İki işi PARALEL çalıştırma.
- Tepki matrislerini (R_d?_1.npy, R_d?_cw.npy, revizyon/sonuclar/D_*.npy, Rtrue_*_g20.npy) YENİDEN ÜRETME;
  dalda gelenleri kullan (bulut sonuçlarıyla birebir tutarlılık için).

ADIMLAR
1. `git pull origin claude/brave-allen-92wn61`.
2. Motoru derle (Linux): `bash build_integrator.sh` ve
   `cd revizyon && g++ -O3 -shared -fPIC -std=c++17 integrator_rev.cpp -o lib_integrator_rev.so && cd ..`.
   Doğrula: `cd revizyon && python3 -c "import integrator_rev, kusurlu_polarite; print('ok')"`.
   Hızlı tutarlılık kontrolü: `python3 kusurlu_polarite.py --ozet` bulut sonucunu (σ_d=1e-5, 3 desen) basmalı.
3. İş 1 (öncelikli; ~2 saat, -w 10): σ_d = 1e-5'te desen sayısını 3'ten 12'ye çıkar:
   `cd revizyon && nohup python3 kusurlu_polarite.py -w 10 --sigd 1e-5 --seeds 3 4 5 6 7 8 9 10 11 > kp1.log 2>&1 &`
4. İş 2 (İş 1 bitince; ~3.5 saat): σ_d taraması, 6 desen:
   `nohup python3 kusurlu_polarite.py -w 10 --sigd 1e-6 3e-5 1e-4 --seeds 0 1 2 3 4 5 > kp2.log 2>&1 &`
   Not: σ_d = 1e-4'te düzeltici yörüngeleri ~1.3 mm olur; kapalı yörünge araması ya da donmuş spin
   kilitlenmesi sorun çıkarabilir (log'da büyük w1, NaN, hata). Çıkarsa o σ_d'yi atla, özetinde yaz.
5. Kesilirse aynı komutla yeniden başlat (kaldığı yerden devam eder).
6. Her ~2 saatte bir ve her iş bitince: `git add -f revizyon/sonuclar/kusurlu_polarite.jsonl`, "test: <ne>"
   ön ekiyle commit, `git push origin claude/brave-allen-92wn61` (başarısızsa 4 kez artan bekleyişle;
   çakışmada `git pull --rebase` sonra tekrar).
7. Bitince `python3 kusurlu_polarite.py --ozet` çıktısını `revizyon/sonuclar/LAPTOP_OZET_v3.md`'ye yaz:
   her σ_d için tablo (durum, n, medyan, %84, maks, hedef-üstü), süre, anormallikler; özellikle
   "Δq iki demet izleyici" ile "Δq tek demet izleyici" ve "Δq düzeltme yok" farkının σ_d ile nasıl
   ölçeklendiğini ve "Δε,Δφ trim" durumunun hedefi aşıp aşmadığını belirt. Commit + push.
```

## C) Ne öğreneceğiz
- İki demetli izleyicinin Omarov birleşimine katkısı 12 desende de tutuyor mu (σ_d = 1e-5)?
- Etki σ_d ile nasıl ölçekleniyor (beklenti: Δq terimi ~σ_d ile doğrusal, tahmin)? Hangi σ_d'de izleyici
  gereksiz, hangisinde yetersiz kalıyor?
- Koşular arası saptırıcı sürüklenmesi gereksinimi σ_d'ye nasıl bağlı?

Sonuçlar push edilince bu oturuma "laptop v3 sonuçları geldi" yaz.
