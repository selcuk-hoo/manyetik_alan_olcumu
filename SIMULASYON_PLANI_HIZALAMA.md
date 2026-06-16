# Simülasyon Planı — Tek-Yörünge ile Sahte-EDM Harmoniğinin Ölçümü (200–300 μm → hedef altı)

> **Asıl problem.** Kuadrupoller mekanik olarak ~200–300 μm RMS hizalanabilir.
> Soru: yalnız **tek bir kapalı-yörünge (BPM) ölçümünden**, bu gürültü içinde,
> sahte-EDM'i süren harmonik içeriği (k=2 ve düşük antisim modlar) ölçüp
> düzeltebilir, böylece kalan sahte EDM'i CW/CCW+quad-flip zincirinin giriş
> seviyesine (~10⁻⁵ rad/s) indirebilir miyiz? Bu sefer **belki spin-trim'e
> gerek kalmadan** (çünkü doğru ölçümle başlangıç ~10⁻⁵, eski sanılan 10⁻³ değil).
>
> **YÖNTEM = tek-yörünge k=2 tahmini** (paper_draft.tex / makale_tr.tex /
> makale_aciklamasi.md). **k-modülasyon YOK, LASSO/Tikhonov YOK, drift modu YOK,
> spin-trim YOK** — yalnız tek statik kapalı-yörünge + R-Fourier projeksiyonu.

---

## 1. Yöntemin temeli: R = G·diag(Kⱼ) ve ofset-bypass

$$\mathbf{y}=R\,\Delta q+\mathbf{b}+\boldsymbol{\eta},\qquad R=G\,\mathrm{diag}(K_j)$$

- $\mathrm{diag}(K_j)$: kaçıklık→kick (QF +, QD −; $(-1)^j$ bazda gömülü).
- $G$: betatron Green fonksiyonu (kick→yörünge),
  $G_{ij}=\frac{\sqrt{\beta_i\beta_j}}{2\sin\pi Q_y}\cos(|\mu_i-\mu_j|-\pi Q_y)$.
- **BPM ofseti $\mathbf{b}$, $G$'den SONRA ölçüme eklenir → zincirin dışında,
  kuvvetlenmez.** Quad kaçıklığı ise $G$'nin rezonansıyla kuvvetlenir; $Q_y\approx2.68$
  için k=2 en yakın → $\|M_{k=2}\|=167$ ($\|M_{k=4}\|=16$, 10× fark).
- Sonuç: **k=2 kestirimine ofset katkısı 167'ye bölünür** → tek-yörünge ölçümü
  ofseti iptal etmeye GEREK olmadan k=2'yi temiz verir. ("BPM ofseti k=2'ye sızmaz.")

## 2. Kestirici (k-mod ve LASSO YOK)

FODO-antisim baz $F_k[j]=(-1)^j\cos(2\pi k\lfloor j/2\rfloor/24)$, $\Delta q=F\hat a$,
$M_k=R F_k$. Hedefli projeksiyon (tek adım):
$$\hat a_{k}=\frac{M_k^{\!\top}\mathbf{y}}{\|M_k\|^2}.$$
Mod kümesi bilinmiyorsa **CLEAN** (oracle-free; R-LS ile istatistiksel özdeş,
200 MC'de 1.0×). x ve y bağımsız 48×48; ikisi de ayrı çatılır.

## 3. Bu oturumun güncellemesi: baskın kanal dx·dy

§13'te kanıtlandı: baskın sahte EDM **dy-only k=2 değil, dx·dy geometrik-faz
çapraz kanalı** (σ², ~1e-5, demet=ideal). Dolayısıyla ölçülmesi gereken,
**hem dx hem dy'nin** sahte-EDM-süren harmonik içeriği (her iki düzlemde k=2
ve düşük antisim modlar). Tek-yörünge yöntemi her iki düzleme bağımsız uygulanır;
düzeltme sonrası kalan dx·dy çarpımı sahte EDM'i belirler.

## 4. Simülasyon adımları

1. **Truth:** rastgele dx,dy σ∈{200,300} μm; N=20–30 seed.
2. **Tek yörünge üret:** $\mathbf{y}_x,\mathbf{y}_y = R_{x,y}\Delta q_{x,y}
   +\mathbf{b}(\sigma_b{\sim}100\mu m)+\eta(\sigma_n{\sim}1\mu m)$ (analitik R
   ya da semplektik izleyici kapalı yörüngesi).
3. **Çat:** R-Fourier projeksiyonu / CLEAN, her düzlem için $\hat a_k$
   ($k\le k_\mathrm{max}$). k-mod ve LASSO kullanılmaz.
4. **Düzeltme:** kestirilen $\hat a_k$ dipol düzelticilerle (ya da kaydırıcı)
   ters işaretle uygulanır → kalan misalignment harmoniği.
5. **Metrikler:**
   - k=2 (ve k≤k_max) genlik/faz geri-çatım hatası (paper_draft hedefi: %0.6
     genlik / 0.046 rad — 200-300μm'de hâlâ tutuyor mu?);
   - k=2 kestiriminde BPM ofset sızıntısı (167'ye bölünme niceliği, N seed);
   - **kalan sahte EDM** = düzeltilmiş dx,dy → dx·dy kanalı
     (`test_dxdy_geometric_phase.py`, §13) → ~10⁻⁵ altına indi mi?
6. **CW/CCW+quad-flip:** kalan üzerine simetri zinciri → <1e-9 (§ önceki testler).
7. **Taban kontrolü (trim makalesinden):** tek-yörünge yalnız ANTİSİM gözlenebilir
   alt-uzayı (gözlenebilir k≤k_max) görür; SİMETRİK 23-boyut (yörünge kazancı düşük)
   görünmez → bu alt-uzayın kalan sahte EDM'e katkısı CW/CCW zincirine girmeden
   önce ne kadar? (trim makalesi: ~10⁻⁴ taban; ama o spin-trim için; burada
   CW/CCW zincirinin girişine ~10⁻⁵ yeterli mi sorusu.)

## 5. Yanıtlanacak anahtar sorular

- 200–300 μm'den, 100 μm BPM ofseti + 1 μm gürültü altında, **tek yörünge +
  R-Fourier projeksiyonu** (k-mod/LASSO YOK) ile sahte-EDM-süren harmonik
  (k=2, her iki düzlem) hedef altına ölçülüp düzeltilebilir mi?
- Ofset-bypass (k=2'ye sızmama) 200-300μm ölçeğinde de geçerli mi?
- Düzeltme sonrası dx·dy sahte EDM'i ~10⁻⁵ altında mı? + CW/CCW → <1e-9?
- Simetrik alt-uzayın (tek-yörüngenin göremediği) kalan katkısı zincir için
  yeterince küçük mü — yoksa spin-trim yine gerekli mi?

## 6. Mevcut kod kaldıraçları

`build_response_matrix.py` (R_dy, R_dx), `fourier_reconstruct.py` (FODO baz +
R-LS/CLEAN projeksiyonu), `bozoki_ls.py` (baz karşılaştırması), `test_bpm_offset.py`
(ofset-bypass), `test_dxdy_geometric_phase.py` (§13 sahte-EDM bağlantısı).
**Bilinçle DIŞARIDA:** k-modülasyon, LASSO/Tikhonov, drift modu, spin-trim.

---

*Hazırlandı/revize: oturum `claude/awesome-babbage-nmi6w9`, 2026-06-16.
Yöntem paper_draft.tex / makale_tr.tex / makale_aciklamasi.md (tek-yörünge k=2
R-Fourier projeksiyonu) ile hizalı. Drift modu (makale-taslagi-2.md) ve spin-trim
(makale_trim_tr.tex) AYRI makalelerdir; bu plan onları kullanmaz.*
