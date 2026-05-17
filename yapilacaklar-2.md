# Yapılacaklar — Faz 2: Simülasyon-Tabanlı Doğrulama

Bu belge, makalenin yayına hazır hale gelmesi için yapılacak somut simülasyon ve test çalışmalarını listeler. Strateji: teoriye fazla girmeden, akıllı simülasyon testleriyle her iddiayı somutlamak.

İlgili belgeler:
- `makale_taslagi.md` — bu testlerin sonuçlarını alacak yer
- `fodo_lattice.py` — analitik R formalizmi
- `spectral_inversion.py` — mevcut Stage A–D testleri
- `reconstruct.py`, `show_response.py`, `verify_quad_tilt.py` — yardımcı scriptler

---

## Genel prensip

ChatGPT eleştirisinin özü: "çok güçlü iddialar var ama yeterli sayısal doğrulama yok." Bu fazda her iddiayı bir simülasyona bağlıyoruz. Makaledeki teoreme bir paragraflık kısa türetme yeterli; ağırlık merkezi **Bölüm 4: Numerical Experiments** olacak.

---

## Test 1 — `compare_regularization.py`

**Amaç:** ChatGPT'nin "haksız karşılaştırma" eleştirisini çözmek. Şu an "ham ΔR=1865 μm" diye sunduğumuz rakam yanıltıcı çünkü hiç kimse ham ΔR'yi inversion yapmaz. Optimal regularize edilmiş ΔR ne yapar? Cevabı somutlamak.

**Girdi:**
- Stage C ile aynı simülasyon verisi (100 μm misalignment, iki gradient)
- Aynı $\mathbf{y}_1, \mathbf{y}_2$ vektörleri

**Estimator'lar:**
1. Direct $R_1^{-1}\mathbf{y}_1$
2. Direct $R_2^{-1}\mathbf{y}_2$
3. Direct $(v_1+v_2)/2$
4. Raw $\Delta R^{-1}(\mathbf{y}_1-\mathbf{y}_2)$
5. Tikhonov: $\lambda \in \{10^{-6}, 10^{-5}, ..., 10^{-1}\}$, L-curve ile optimal seçimi
6. TSVD: truncation level $k \in \{4, 8, 16, 24, 32, 40, 48\}$

**Çıktı:**
- Tek bir tablo: her estimator için RMS hata, max hata, korelasyon (y ve x düzlemi)
- L-curve grafiği: Tikhonov $\|\widehat{\Delta q}\|_2$ vs $\|\Delta R\widehat{\Delta q}-(\mathbf{y}_1-\mathbf{y}_2)\|_2$
- TSVD scree plot: RMS hata vs truncation level

**Beklenen sonuç:** Optimal Tikhonov ve TSVD raw ΔR'den çok daha iyi (belki ~50-200 μm), ama direkt inversion'dan (~3-5 μm) hâlâ önemli ölçüde geride. Eğer aralarındaki fark dramatik değilse → makale çerçevesini bir kez daha gevşetmek gerekecek.

**Süre tahmini:** ~45 dk kodlama + birkaç dk koşma

---

## Test 2 — `mode_transfer.py` (signature figure)

**Amaç:** Makalenin en güçlü görseli. RMS sayıları sade ve tek boyutludur; gerçek fiziksel hikaye **uzaysal bant genişliği**dir.

**Yöntem:**
- N=48 lattice'inde sinüsoidal misalignment patternleri üret: $\Delta q^{(k)}_j = A\cos(2\pi k j / N)$, $k = 0, 1, ..., N/2$
- Her $k$ için her estimator'ı uygula
- "Reconstructed amplitude / true amplitude" oranını her $k$ için hesapla
- Bu, estimator'ın uzaysal transfer fonksiyonudur

**Estimator'lar:** Test 1'deki ile aynı altı seçenek

**Çıktı:**
- Bir grafik: x ekseni Fourier mod indeksi $k$, y ekseni transfer ratio (0 ile 1 arası), her estimator için bir eğri
- Direct inversion: tüm $k$ için ratio ≈ 1 (düz çizgi)
- Raw ΔR: yüksek gürültü (kontrolsüz)
- Tikhonov/TSVD: düşük $k$ için ratio ≈ 1, yüksek $k$ için sıfıra düşer — uzaysal alçak-geçiren filtre davranışı
- Her estimator için "etkili bant genişliği" tanımı: ratio > 0.5 olduğu en yüksek $k$

**Beklenen sonuç:** Regularize ΔR uzaysal düşük-geçiren filtre gibi davranır; local/yüksek-frekans misalignment'ları kaybeder. Direct inversion tüm bandı korur ama ofset bağışıklığı yoktur. Makalenin ana fiziksel mesajı bu grafikte tek bakışta görülecek.

**Süre tahmini:** ~45 dk kodlama

---

## Test 3 — `kxarc_sensitivity.py`

**Amaç:** Inverse crime endişesini somutlamak. Şu an $K_{x,\text{arc}}$ hem forward simülasyondan kalibre ediliyor hem de inverse için kullanılıyor. "Gerçekte $K_{x,\text{arc}}$'ı %5 yanlış bilseydik ne olurdu?"

**Yöntem:**
- Stage C simülasyonu olduğu gibi koş (gerçek $K_{x,\text{arc}}$ ile)
- Reconstruction'ı bir dizi *perturbed* $K_{x,\text{arc}}$ değeriyle yap: nominal × $(1 + \delta)$, $\delta \in [-10\%, +10\%]$
- Her $\delta$ için RMS rekonstruksiyon hatası

**Çıktı:**
- Grafik: $\delta$ vs RMS reconstruction error (x düzlemi)
- y düzlemi: aynı testi $K_{y,\text{arc}} = 0$ varsayımıyla yap → Maxwell garantili, hata değişmez (kontrol)

**Beklenen sonuç:** ~%5 hatasına dayanıklı olmalı; daha büyük hatalarda lineer-üstü bozulma. Bu, gerçek bir halkada $K_{x,\text{arc}}$'ı LOCO ile kalibre etmenin yeterli olduğunu gösterir.

**Süre tahmini:** ~30 dk kodlama (mevcut reconstruct.py'yi parametrize et)

---

## Test 4 — `drift_monitor_sim.py`

**Amaç:** Drift monitoring çerçevesinin asıl gösterimi. Makalenin "online observer" iddiasının somut kanıtı.

**Senaryo:**
1. $t=0$: kalibrasyon epoch'u. Misalignment vektörü $\Delta q_0$ (100 μm RMS rastgele). BPM ofseti $\mathbf{b}_0$ (50 μm RMS rastgele). $\mathbf{y}_{1,0}, \mathbf{y}_{2,0}$ kaydedilir.
2. $t=1...10$: her epoch'ta misalignment'a küçük bir drift eklenir, $\delta q(t) = \delta q_{\text{ramp}} \cdot t / 10$ (toplam 10 μm ramp). BPM ofseti **sabit** tutulur (varsayım: ofset kayması ihmal edilebilir).
3. Her epoch'ta $\mathbf{y}_i(t)$ ölçülür, estimator $\widehat{\delta q}(t) = R^{-1}(\mathbf{y}(t) - \mathbf{y}_0)$ uygulanır.
4. $\widehat{\delta q}(t)$ ile gerçek $\delta q(t)$ karşılaştırılır.

**Çıktı:**
- Grafik: t vs $\|\delta q(t)\|_{\text{RMS}}$ — gerçek ve estimator ayrı eğriler
- Tablo: her epoch için RMS tracking hatası, korelasyon
- Tek-epoch belirsizlik bandı (BPM noise dahil)

**Beklenen sonuç:** Estimator drift'i tek-epoch gecikmesi olmadan, ~1-2 μm hassasiyetle takip eder. BPM ofsetinin mutlak değeri (50 μm RMS) sonucu *etkilemez* çünkü farkı alıyoruz. Bu, "BPM offset problem dissolves" iddiasının deneysel kanıtı.

**Süre tahmini:** ~60 dk (simülasyonu birkaç kez koşmak gerekiyor)

---

## Test 5 — `bpm_offset_drift_sim.py`

**Amaç:** Drift monitoring varsayımının test edilmesi. "BPM ofseti kayması magnet hareketinden çok yavaş" varsayımı kırıldığında ne olur?

**Senaryo:**
- Test 4'ün aynısı, ama bu sefer BPM ofseti de zamanla kayıyor: $\mathbf{b}(t) = \mathbf{b}_0 + \mathbf{b}_{\text{drift}} \cdot t / 10$
- BPM drift hızı $\|\mathbf{b}_{\text{drift}}\|_{\text{RMS}}$'yi 0'dan 10 μm'ye değiştir
- Her hız için reconstruction hatası

**Çıktı:**
- Grafik: BPM ofset drift hızı (μm/epoch) vs reconstruction RMS hatası
- Threshold: hangi BPM drift hızında reconstruction hatası 5 μm sınırını aşar?

**Beklenen sonuç:** Eşik ~2-3 μm/epoch civarında. Bu, "kaç saatte bir yeniden kalibre etmek gerekir" sorusuna sayısal cevap verir. Mesela 1 epoch = 1 dakika ise, BPM ofseti 3 μm/dakika'dan yavaş kaymalıdır → modern BPM elektroniği için kolay sağlanabilir (literatür: termal coef. ~0.1 μm/°C).

**Süre tahmini:** ~30 dk (Test 4'ün varyantı)

---

## Yan iş: terimler ve tonlama

Kod testlerinden bağımsız olarak makale metninde yapılacak küçük düzeltmeler:

- "Offset–Fidelity Conjugate Theorem" → **"offset–noise duality"** (sadeleştirme)
- "Theorem" formatından **kısa türetme (paragraf)** formatına geç — teoriye fazla girmiyoruz
- Bölüm 2'yi (analitik R) hafifçe sıkıştır — odak Bölüm 4 (numerical experiments) olsun
- "ΔR is unusable" tonu zaten yumuşatıldı, kontrol et
- "Inverse crime" tartışmasını Test 3 sonucuyla destekle

---

## Yürütme sırası

Önerilen sıra (bağımlılıklara göre):

1. **Test 1** (`compare_regularization.py`) — en yüksek etki, diğer testler için zemin
2. **Test 2** (`mode_transfer.py`) — signature figure, Test 1'in genişletilmiş hali
3. **Test 4** (`drift_monitor_sim.py`) — makale ruhunun ana gösterimi
4. **Test 5** (`bpm_offset_drift_sim.py`) — Test 4'ün doğal uzantısı
5. **Test 3** (`kxarc_sensitivity.py`) — bağımsız, sona bırakılabilir

Toplam tahmini süre: ~3-4 saat kodlama + koşma.

Test 1 ve 2'nin sonucu çıktıktan sonra **ara değerlendirme**: eğer regularize ΔR direkt inversion'a çok yaklaşırsa, makale çerçevesini bir kez daha gözden geçirmek gerekebilir.

---

## Test sonrası makale revizyonu

Tüm testler tamamlandıktan sonra `makale_taslagi.md`'de güncellemeler:

- Tablo 3'ün tüm boş hücreleri doldurulacak
- Şekil 3 (SVD spektra), Şekil 4 (mode transfer), Şekil 7 (drift monitor) eklenecek
- Tablo 4'ün BPM offset drift satırı doldurulacak
- Bölüm 4 (Numerical Experiments) yeniden yazılacak, her test bir alt-bölüm
- Bölüm 6.1'deki inverse crime tartışması Test 3 sonucuyla genişletilecek
- "TO BE DONE" yorumlarının hepsi temizlenecek

Bu rapor ve makale taslağı, çalışmanın referans çerçevesidir.
