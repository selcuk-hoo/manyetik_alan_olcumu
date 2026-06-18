# Simülasyon Planı — Yörünge-Trim Kademesinin §13 Sonrası Yeniden Değerlendirmesi

> **İlgili makale: yalnız `makale_trim_tr.tex`** (iki kademeli trim: EDM-kör
> yörünge trimi + spin-sürülü ölç-trim). Diğer taslaklar (tek-yörünge k=2,
> drift modu) bu hedefin dışında.
>
> **Soru.** Mekanik hizalama ~200–300 μm. Yörünge-trim kademesi (kapalı yörünge
> tabanlı, EDM-kör) **tek başına** — spin kademesi olmadan — sahte EDM'i
> CW/CCW+quad-flip zincirinin giriş seviyesine (~10⁻⁵ rad/s) indirebilir mi?
> Eski cevap "hayır" idi (yörünge tabanı ~2.5×10⁻⁴, simetrik alt-uzay) ve spin
> kademesi bu yüzden eklendi. **Ama o sayılar §13 estimator düzeltmesinden önceydi.**

## 1. §13'ün makale_trim_tr.tex'e etkisi (yeniden değerlendirme gerekçesi)

- Trim makalesindeki $f$ değerleri (trimsiz ~1.6×10⁻³, yörünge-trim tabanı
  ~2.5×10⁻⁴, simetrik alt-uzay ~1.0×10⁻⁴) büyük olasılıkla **betatron-kontamine
  estimator** (CO=False, düz polyfit, kısa t2) ile ölçülmüştü (§13.1).
- §13 ile kanıtlandı: dy-only gerçek seküler (CO=True, model fit) ~10⁻⁹;
  baskın kanal **dx·dy geometrik faz** ~10⁻⁵ (σ², demet=ideal); eski büyük
  değerler fit artefaktıydı.
- Dolayısıyla **yörünge-trim tabanının ve simetrik alt-uzay katkısının gerçek
  (model-fit) değeri yeniden ölçülmeli.** Eğer simetrik-alt-uzay tabanı doğru
  estimator'la ≪10⁻⁵ çıkarsa, **spin kademesi gereksizleşir**: yörünge-trim +
  CW/CCW yeterli olur. Kullanıcının hipotezi tam bu.

## 2. Yöntem (makale_trim_tr.tex §yörünge; k-mod/LASSO YOK)

- Mod parmak izi kalibrasyonu: $\mathbf{o}_k=[\mathbf{y}(A_\mathrm{cal})-
  \mathbf{y}_0]/A_\mathrm{cal}$ (kaydırıcı diferansiyel vuruşu; gradyan
  modülasyonu DEĞİL). Statik ofset farkta düşer.
- Trim: ölçülen yörünge $O=[\mathbf{o}_1\,\mathbf{o}_2\cdots]$'a düz LSQ fit
  (LASSO/Tikhonov DEĞİL); kestirim ters işaretle kaydırıcılara.
- Fit eşiği (kapalı form): $G_k>\sigma_b/\sigma_q \Leftrightarrow
  k_\mathrm{max}^2<Q_\mathrm{eff}^2+C\,\sigma_q/\sigma_b$. Yüksek-$G_k$ modlar
  (k=2) ofset-bağışık; bu "BPM ofseti k=2'ye sızmaz"ın trim-kademesi karşılığı.

## 3. Simülasyon adımları (hepsi §13 doğru estimator'ı = model fit ile)

1. **Truth:** rastgele dx,dy σ∈{200,300} μm, N=20–30 seed.
2. **Yörünge-trim uygula** (EDM-kör yörünge-trim mantığı, k≤k_max LSQ).
3. **Kalan misalignment** → sahte EDM'i **model-fit + CO=True** ile ölç
   (§13 model-fit estimator'ı), dx·dy kanalı dahil. ESKİ yörünge-trim spin
   doğrulaması kontamine estimator'la idi — **bu adım yeniden, doğru estimator'la.**
4. **Simetrik alt-uzay tabanı:** dik ayrıştırma (antisim/simetrik) ile simetrik
   parçanın sahte EDM'ini **model-fit + CO=True** ile yeniden ölç. Eski ~1.0×10⁻⁴
   gerçek mi, yoksa artefakt mı? (simetrik/antisim ayrıştırmayı doğru
   estimator'la yeniden koş.)
5. **Karar:** yörünge-trim sonrası kalan (simetrik dahil) sahte EDM ≤10⁻⁵ mü?
   - Evet → spin kademesi GEREKSİZ; makale "yörünge-trim + CW/CCW" olarak sadeleşir.
   - Hayır → simetrik taban gerçek; spin kademesi korunur.
6. **CW/CCW+quad-flip:** kalan üzerine → <1e-9 (bu oturumda 3 seed'de teyit
   edildi: ~1e-6 → ~1e-11).

## 4. Yanıtlanacak anahtar sorular

- §13 estimator'ıyla yörünge-trim gerçek tabanı nedir (eski 2.5×10⁻⁴ artefakt mı)?
- Simetrik alt-uzayın gerçek (model-fit, CO=True) sahte-EDM katkısı ≪10⁻⁵ mü?
- 200–300 μm'den yörünge-trim, dx·dy kanalını CW/CCW girişine (~10⁻⁵) indirir mi?
- Spin kademesi hâlâ gerekli mi, yoksa makale yörünge-trim + CW/CCW'ye sadeleşir mi?

## 5. Mevcut kod kaldıraçları

Kalıcı: `build_response_matrix.py`, `fourier_reconstruct.py`, `reconstruction.py`.
Yörünge-trim + simetrik ayrıştırma + dx·dy/§13 estimator ve CW/CCW mantığının
keşif scriptleri 2026-06 temizliğinde kaldırıldı (git geçmişinde; oturum içinde
`/tmp` altında `orbit_trim_dxdy.py`, `spin_trim_chain.py`, `recon_compare.py`
olarak yeniden üretildi). **Bu plan uygulandı; sonuçlar
`false_edm_harmonic_sinir.md §14`'te** (yörünge-trim ~2.7×, simetrik taban gerçek
→ spin kademesi korunur; 6 metot aynı tabana çarpar).

---

*Hazırlandı/revize: oturum `claude/awesome-babbage-nmi6w9`, 2026-06-16.
İlgili makale yalnız makale_trim_tr.tex; plan yörünge-trim kademesini §13 doğru
estimator'ıyla yeniden değerlendirmeye odaklanır (spin kademesinin hâlâ gerekli
olup olmadığı sorusu). k-mod/LASSO/drift/tek-yörünge yöntemleri kapsam dışı.*
