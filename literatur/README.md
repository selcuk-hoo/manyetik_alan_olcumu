# Literatür Referansları (özgünlük karşılaştırması)

Bu klasör, drift-monitör çalışmasının özgünlüğünü değerlendirmek için ilgili
makalelerin pdftotext ile çıkarılmış tam metinleridir. Gerçek metinden satır satır
karşılaştırma için (snippet'ten DEĞİL).

| Dosya | Makale | arXiv / kaynak |
|-------|--------|----------------|
| `ref_mirza_symmetric_circulant.md` | Mirza, Klingbeil, Singh, Forck — Closed orbit correction for symmetric/near-symmetric lattices (circulant ORM) | PRAB 22, 072804 / arXiv:1902.08683 |
| `ref_pac1993_2263.md` | Chung, Decker, Evans — Closed Orbit Correction Using SVD of the Response Matrix (temel SVD orbit-düzeltme) | PAC 1993, p.2263 (APS/Argonne) |
| `ref_fast_bba_ac.md` | Fast beam-based alignment using ac excitations | PRAB 23, 012802 / arXiv:2001.xxxxx |
| `ref_simultaneous_bba.md` | Simultaneous beam-based alignment measurement for multiple magnets | arXiv:2203.14869 |
| `ref_continuous_orm.md` | Ziemann & Ziemann — Noninvasively improving the ORM while continuously correcting the orbit (dither ile **corrector↔orbit** ORM online güncelleme; misalignment/drift izleme DEĞİL) | arXiv:2104.05300 |
| `ref_allelectric_quad_misplacement.md` | **Hacıömeroğlu & Semertzidis** (yazarın KENDİ 2017 makalesi) — all-electric halkada misplacement→sahte-EDM (ileri hesap, FARKLI makine; izleme/gözlenebilirlik DEĞİL) | arXiv:1709.01208 |
| `_BIZIM_drift_makalesi_eski_derleme.md` | **BİZİM kendi taslağımız** (reframe öncesi derleme) — rakip DEĞİL | — |

## Karşılaştırma durumu (metne-dayalı, snippet DEĞİL)

| Makale | Bizi preempt ediyor mu? | Neden |
|--------|------------------------|-------|
| Mirza (circulant) | **HAYIR** | Orbit *düzeltme* (BPM↔corrector); misalignment gözlenebilirliği/drift/kör nokta yok (terim sayıları 0). Tek örtüşme: response-matrix circulant gözlemi (= eski fig8, zaten kaldırıldı). |
| 2104.05300 (sürekli ORM) | **HAYIR** | Dither ile *corrector↔orbit* ORM'sini online günceller (düzeltme modeli); misalignment/alignment/drift/BPM-ofset/EDM = 0. "Sürekli" = matris-tahmini, drift izleme değil. |
| 1709.01208 | rakip değil | Yazarın KENDİ all-electric ileri-sistematik makalesi. |
| PAC1993 (SVD orbit corr.) | **KISMEN** | SVD-gözlenebilirlik TEMELİNİ kurar (mode-space, κ, decoupled/gözlenemez modlar, harmonik modlar ≈ G_k, periyodiklik). Bizim analitik çekirdeğimiz 1993'te var → özgün değil. AMA drift-kaynağı/EDM/ofset-iptali yok. |
| BBA'lar (fast-ac, simultaneous) | (okunacak) | aktif per-quad BBA. |

**Özgünlük residue'sü (Mirza+PAC1993+2104.05300 sonrası):** SVD-gözlenebilirlik
*makinesi* özgün değil (PAC1993). Kalan tek savunulabilir çekirdek: o makinenin
**fiziksel drift kaynağına + sahte-EDM alt-uzayına** uygulanması — koherent
yer-hareketinin yapısal körlüğü + ofset-iptalli drift izleme. BBA + EDM
literatürüne karşı hâlâ sınanmalı.

**Not:** Telif metinler; yalnız özel araştırma deposunda referans amaçlı.

---

## Deep-research (ChatGPT + Gemini) ile çıkan KRİTİK yeni prior-art (henüz okunmadı)

Bu ikisi bizim bulamadığımız ve residue'müzü en çok tehdit eden makaleler — **mutlaka okunmalı:**

| Makale | Neyi tehdit ediyor | Link |
|--------|---------------------|------|
| **Rossbach 1989**, "Closed-orbit distortions of periodic FODO lattices due to plane ground waves", Particle Accelerators 23, 121 | **C+D**: FODO'da yer-dalgalarının COD'u; uzun dalga (koherent) → küçük orbit, rezonans nN±Q'da. Bizim "koherent→simetrik→kör" sonucumuzun fizik çekirdeği. | CERN inspire |
| **Wegscheider et al. 2023**, PRAB 26, 032803 | **B**: ORM'den kuadrupol hatası geri-çatımında quasidegeneracy / null-space (düzeltme değil, REKONSTRÜKSİYON gözlenebilirliği) — bizim B iddiamızın doğrudan eşi. | GSI repo |

Diğer surfacelenen: Tiefenback 1985 (FODO sim/antisim mod), Shiltsev 1995 (ATL),
Parkhomchuk-Shiltsev-Stupakov 1994 (korelasyon→COD), Khan 2017 (σ_k=Q/π\|Q²−k²\|),
Safranek 1997 (LOCO null-space), Huang 2022 / Xu 2025 (pasif/ML misalignment).

**İki AI'ın harf-notları (ÇELİŞİYOR → notlar yumuşak, asıl olan substance):**
A: NONE/PARTIAL · B: PARTIAL/FULL · C: NONE/PARTIAL · D: FULL/(PARTIAL-NONE) · E: PARTIAL/FULL
