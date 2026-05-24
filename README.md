# pEDM Halkasında Quad Hizalama Hatalarının Geri Çatımı

Bu depo, **proton elektrik dipol moment (pEDM)** halka tasarımında
kuadrupol mıknatısların hizalama hatalarının kapalı yörünge (COD)
ölçümlerinden geri çatımı üzerine yapılan çalışmaları içerir.

pEDM, "frozen-spin" yöntemiyle proton EDM'inin doğrudan ölçümünü
hedefler. Halkanın 24 FODO hücresinden oluşan 48 kuadrupolünün her
biri 10 μm mertebesinde hassasiyetle hizalanmalıdır; sistematik
yörünge sapmaları aksi takdirde EDM sinyalini taklit edebilir. Bu
çalışmanın amacı **BPM ölçümlerinden tek tek quad hizalama hatalarını
geri çatma** problemini hem klasik tepki-matrisi yaklaşımıyla, hem
de düşük-mertebeli Fourier parametrelendirmesiyle incelemektir.

---

## Çerçeve

Bu yeni bir yöntem icadı değildir; klasik tepki-matrisi (orbit
response matrix, ORM) yaklaşımının pEDM halkasının özel koşullarında
**hangi sınırlarda 10 μm hassasiyetine ulaşabildiğini** sistematik
olarak incelenen bir mühendislik değerlendirmesidir. Paralel bir
çalışmada aynı problem yapay sinir ağı yaklaşımıyla ele alınmaktadır;
iki yaklaşımı adil karşılaştırmak için bu çalışmanın ürettiği
referans çerçeveye ihtiyaç vardır.

---

## Problemin matematiksel iskeleti

48 BPM'de okunan dikey yörünge:

```
y = R · dy + b + n
```

| Sembol | Boyut | Anlam |
|--------|-------|-------|
| `y`  | 48 | BPM okumaları (m) |
| `R`  | 48×48 | tepki matrisi (Courant–Snyder formülü) |
| `dy` | 48 | quad dikey hizalama hataları (aranan) |
| `b`  | 48 | BPM elektronik ofsetleri (bilinmeyen, statik) |
| `n`  | 48 | BPM gürültüsü (~1 μm) |

`b` bilinmediği için `R · dy = y − b` doğrudan çözülemez. Yörünge
sapmaları (10 μm) BPM ofsetleri (~300 μm) yanında küçüktür → ofset
sinyali boğar.

**Çözüm: k-modülasyon.** Quad gradyenlerini iki farklı ayarda ölçüp
fark al; `b` ortak olduğundan iptal olur:

```
Δy  =  y₂ − y₁  =  (R₂ − R₁) · dy  =  ΔR · dy
```

Bu denklem `dy`'yi `b`'siz olarak verir. Ancak `ΔR`'nin koşul sayısı
hangi modülasyon stratejisinin seçildiğine kuvvetle bağlıdır.

---

## Üç modülasyon stratejisi

| Mod | Ne yapar | κ(ΔR) | Sonuç |
|-----|----------|-------|-------|
| **Uniform** | Tüm 48 quad gradyeni aynı anda ×1.02 | ~160 | 6.6 μm RMS, kor ≈ 1 |
| **Tek-quad** | Yalnız 1 quad ×1.10 | ~4.4×10⁸ | başarısız |
| **İki-quad** | İki quad (j₁, j₂) ×1.05, diğer 46 sabit | ~10⁶ | tek başına %50 |

Uniform mod mükemmel çalışır ama gerçek bir hızlandırıcıda nadiren
uygulanabilir (tüm güç kaynaklarını eşzamanlı modüle etmek pratik
değildir). Tek-quad mod fiziksel olarak gerçekçidir ama matris tekil.
İki-quad mod orta noktadır: pratik olarak uygulanabilir ama matris
fena koşullu.

---

## Bu deponun bulgusu

İki-quad modülasyonunun "kötü" koşullanması aslında **doğru çözüm
uzayını yanlış parametrelendirmenin** sonucudur. Düşük-mertebeli
Fourier bazıyla yeniden parametrelendirildiğinde, doğru baz seçimi
**ölçüm hatasını 0.02 μm**'ye indirir (gürültü tabanı). Detaylar
[`YÖNTEM.md`](YÖNTEM.md) içinde, ders kitabı tarzı açıklanmıştır.

Özet:

| Strateji | κ(ΔR·F) | Ölçüm hatası | Korelasyon |
|----------|---------|--------------|------------|
| Direkt çözüm (R⁻¹) | — | 107 μm | 0.03 |
| TSVD | — | 78 μm | 0.16 |
| Geniş Fourier (k=1..4) | 1.3×10⁴ | 35 μm | 0.88 |
| **Sıkı Fourier ({2,4} = gerçek harmonikler)** | **186** | **0.02 μm** | **1.000** |

Mesaj: **bazı, sinyalde gerçekten var olan harmoniklere sıkıca
hizala**. Fazla harmonik koşullanmayı bozar; eksik harmonik veriden
sızıntıya yol açar.

---

## Depo yapısı

### Kod dosyaları

| Dosya | İşlev |
|-------|-------|
| `integrator.cpp` | GL4 simplektik integratör, C++ — quad/dipol kick'leri ile parçacık takip |
| `lib_integrator.so` / `integrator.dylib` | Derlenmiş paylaşımlı kütüphane |
| `integrator.py` | C++ kütüphanesinin Python sarmalayıcısı (ctypes) |
| `run_simulation.py` | Tek simülasyon (test/görselleştirme amaçlı) |
| `build_response_matrix.py` | R₁ ve R₂ tepki matrislerini paralel hesaplar |
| `test_kmod_reconstruction.py` | K-mod ile geri çatım (TSVD, Fourier, hedefli) |
| `scan_j2.py` | j₁ sabit, j₂ taraması — en iyi quad çiftini bul |
| `show_response.py` | Tepki matrisi görselleştirme |
| `verify_quad_tilt.py` | Quad tilt etkisi doğrulama (kuplaj kontrolü) |

### Konfigürasyon

| Dosya | İçerik |
|-------|--------|
| `params.json` | Tüm fiziksel parametreler (R, g, modülasyon, hata seedleri) |

### Belgeler

| Dosya | İçerik |
|-------|--------|
| `README.md` | Bu dosya — depo ve çalışmanın genel haritası |
| `YÖNTEM.md` | Fourier rekonstrüksiyon yönteminin ders kitabı tarzı anlatımı |
| `metot.md` | Makale §3-5 için pedagojik metot taslağı |
| `makale-taslagi-2.md` | Makale taslağı |
| `YAPILACAKLAR.md` | Aktif iş listesi ve mevcut durum |

---

## Hızlı başlangıç

### 1. Kütüphaneyi derle (gerekirse)

```bash
# Linux
g++ -O2 -shared -fPIC -o lib_integrator.so integrator.cpp -std=c++17
# macOS
clang++ -O2 -shared -fPIC -o integrator.dylib integrator.cpp -std=c++17
```

### 2. Tepki matrislerini hesapla (paralel)

```bash
python3 build_response_matrix.py
```

Üretir: `R_dy_1.npy`, `R_dy_2.npy`, `R_dx_1.npy`, `R_dx_2.npy`.
Süre: ~1 saat (10 worker), modülasyon konfigürasyonuna göre değişir.

### 3. Geri çatımı çalıştır

```bash
# Rastgele dy/dx hataları (gerçekçi senaryo)
python3 test_kmod_reconstruction.py

# Sinüzoidal (smooth) dy/dx — algoritma testi
python3 test_kmod_reconstruction.py --smooth
```

Çıktı: Direkt çözüm + TSVD + Fourier (N=1..5) + hedefli harmonik
sonuçları.

### 4. (İsteğe bağlı) En iyi j₂'yi tara

```bash
python3 scan_j2.py --step 4      # kaba (12 j₂, ~75 dk)
python3 scan_j2.py               # tam (47 j₂, ~5 saat)
```

---

## Modülasyon konfigürasyonu (params.json)

```json
{
  "g0": 0.20,                  // baz gradyen (46 quad)
  "g1": 0.21,                  // j₁ quad'ı için
  "g2": 0.21,                  // j₂ quad'ı için
  "kmod_quad1_index": 2,       // j₁ — pertürbe edilen ilk quad
  "kmod_quad2_index": 8        // j₂ — pertürbe edilen ikinci quad
}
```

- `kmod_quad1_index = -1` ve `kmod_quad2_index = -1`: uniform mod
  (tüm 48 quad'ı `g1`'e ölçekler)
- Yalnız `kmod_quad1_index` set: tek-quad mod
- Her ikisi set: iki-quad mod

⚠️ `kmod_quad1_index = 0` kullanılmamalı — integrator.cpp:541'de
`current_fodo == 0` için özel "QUAD_F_MOD" tipi (4) tetiklenir ve
`quad_dG` yok sayılır. j₁ ≥ 1 olmalı.

---

## Çalışmanın kronolojisi

1. **Klasik tepki-matrisi (uniform k-mod) testi** — κ ≈ 160, mükemmel
   çalışıyor. Pratik uygulanabilirliği sorgulanır.
2. **Tek-quad k-mod denemesi** — κ ≈ 4×10⁸, tamamen başarısız.
3. **İki-quad k-mod** — κ ≈ 10⁶, %50 rekonstrüksiyon.
4. **Optimum j₂ taraması** — `scan_j2.py`, en iyi quad çiftini ara.
5. **TSVD regülarizasyonu** — 4/48 mod, korelasyon 0.16.
6. **Fourier parametrelendirmesi** — düşük-N bazı dene, bias-variance
   gerilimi gözlemlendi.
7. **Hedefli Fourier (kritik bulgu)** — bazda yalnızca veride
   gerçekten var olan harmonikler → 0.02 μm ölçüm hatası, kor=1.000.

Detaylar [`YÖNTEM.md`](YÖNTEM.md) ve [`metot.md`](metot.md)'de.

---

## Mevcut açık konular

- Gerçekçi senaryo: `dy = smooth_part + small_random` testleri
- Adaptif baz seçimi: rezidüel analiziyle hangi harmoniklerin var
  olduğunu otomatik bulma
- Çoklu (j₁, j₂) ölçüm birleştirmesi (rank artırma)
- BPM ofset ve gürültüsünün hedefli yöntem üzerindeki etkisi

Aktif iş listesi: [`YAPILACAKLAR.md`](YAPILACAKLAR.md).
