#!/usr/bin/env python3
"""bozoki_ls.py — Bozoki (1989) LS yöntemi vs kmod+CLEAN karşılaştırması.

Bozoki 1989 (BNL, NSLS):
  Çevreli halka, Q≈1.26, 4-kat süper-periyot.
  Dominant harmonik k=1.  BPM ölçümünü β_i^{1/2}'ye bölerek normalize etmişler
  (η_i = y_i / √β_i) ve S = Σ(η_i − a cos φ_i − b sin φ_i)² minimize etmişler.
  k=1 güçlü olduğu için trim dipol telafisi yetmiş.

Bizim durum (pEDM, N_Q=48):
  Hedef harmonik k=2, genlik ~10 μm.
  Nuisance harmonikler k=4,6,8 genlik ~300 μm.
  BPM sistematik ofseti σ=100 μm >> k=2 sinyali.
  → Tek yörünge yöntemleri başarısız.
  → kmod (BPM ofseti iptali): ofset yok → CLEAN başarılı.

Monte Carlo karşılaştırma (σ_bpm=100 μm sabit):
  Yöntem 1 — Bozoki LS   : η=y/√β, azimut DFT fit, TEK yörünge
  Yöntem 2 — Doğrudan LS : R·F·a=y, FODO-antisim baz, TEK yörünge
  Yöntem 3 — kmod+CLEAN  : R·F·a=y_kmod, ofset iptali, İTERATİF

Tüm yöntemlerde aynı R orbit tepki matrisi kullanılır.
kmod avantajı YALNIZCA BPM sistematik ofsetinin iptalinden gelir:
  Tek yörünge : y_obs = R·q + b + ε     (b bozar; σ_b=100 μm >> 10 μm)
  kmod        : y_obs = R·q + ε          (b iptal; gürültü tabanı √2 μm)
"""

import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from fourier_reconstruct import clean_reconstruct, fodo_basis

# ── kafes sabitleri ───────────────────────────────────────────────────────────
with open("params.json") as fh:
    cfg = json.load(fh)

N_Q = 2 * int(cfg["nFODO"])     # 48
G   = float(cfg["g1"])           # 0.21 T/m
L_Q = float(cfg["quadLen"])      # 0.4 m
M2, AMU = 0.938272046, 1.792847356
p_magic = M2 / math.sqrt(AMU)
Brho    = p_magic / 0.29979246   # T·m
KL      = G * L_Q / Brho         # m⁻¹  (QF = QD, aynı büyüklük)

# ── R matrisi (nominal tam-kerteli orbit tepki) ───────────────────────────────
R1 = np.load("R_dy_1.npy")          # 48×48, tam kerteli
# kmod da aynı R1'i kullanır; avantaj sadece BPM ofset iptalinden gelir

# ── β_y (FFT Q_y = 1.732) ────────────────────────────────────────────────────
Q_y      = 1.732
cot_y    = math.cos(math.pi * Q_y) / math.sin(math.pi * Q_y)
KL_sgn   = np.where(np.arange(N_Q) % 2 == 0, +KL, -KL)
beta_y   = np.diag(R1) / (KL_sgn * cot_y / 2)
sqrt_bet = np.sqrt(np.abs(beta_y))      # √β_i  [m^{1/2}]

# ── Fourier bazları ───────────────────────────────────────────────────────────
CANDIDATE_KS = [0, 1, 2, 3, 4, 5, 6, 7, 8]

F_k2, _ = fodo_basis(N_Q, [2], antisym=True)  # (48, 2): cos/sin

# ── Nuisance misalignment deseni (k=4,6,8 — params.json'dan) ─────────────────
dq_nuisance = np.zeros(N_Q)
for h in cfg.get("dy_harmonics", []):
    if h["k"] != 2:
        Fh, _ = fodo_basis(N_Q, [h["k"]], antisym=True)
        dq_nuisance += Fh @ np.array([h.get("amp_cos", 0.0), h.get("amp_sin", 0.0)])

# ── Yöntem 1: Bozoki LS ──────────────────────────────────────────────────────
# η = y / √β  →  fit a·cos(2θ) + b·sin(2θ) (azimut açısı)
theta   = 2 * np.pi * np.arange(N_Q) / N_Q
Phi_boz = np.column_stack([np.cos(2 * theta), np.sin(2 * theta)])  # 48×2


def bozoki_k2(y_obs: np.ndarray) -> float:
    """Bozoki: η=y/√β → azimut LS → k=2 genlik [m]."""
    eta = y_obs / sqrt_bet
    a, _, _, _ = np.linalg.lstsq(Phi_boz, eta, rcond=None)
    return float(np.sqrt(a[0]**2 + a[1]**2))


# ── Yöntem 2: Doğrudan R-matris LS ──────────────────────────────────────────
# R1 · F_k2 · a = y  →  a → A_k2
M_dir = R1 @ F_k2   # 48×2


def direct_ls_k2(y_obs: np.ndarray) -> float:
    """Doğrudan LS: M·a=y, M=R1·F_k2 (tek yörünge, FODO-antisim baz)."""
    a, _, _, _ = np.linalg.lstsq(M_dir, y_obs, rcond=None)
    return float(np.sqrt(a[0]**2 + a[1]**2))


# ── Yöntem 3: kmod + CLEAN ────────────────────────────────────────────────────
# kmod sinyal: y_kmod = R1·q + ε,  BPM ofseti tam iptal (b=0)
def kmod_clean_k2(y_kmod: np.ndarray) -> float:
    """kmod + CLEAN: R1·F·a = y_kmod (ofset iptali, aynı R1 kullanılır)."""
    accum, _, _ = clean_reconstruct(
        [R1], [y_kmod], CANDIDATE_KS, antisym=True,
        gain=0.2, max_iter=300, tol=1e-4)
    a2 = accum[2]
    return float(np.sqrt(a2[0]**2 + a2[1]**2))


# ── Monte Carlo taraması ──────────────────────────────────────────────────────
print("Monte Carlo başlıyor...")
RNG         = np.random.default_rng(42)
N_TRIALS    = 80
SIGMA_BPM   = 100e-6   # 100 μm sistematik BPM ofseti
SIGMA_NOISE =   1e-6   #   1 μm ölçüm gürültüsü

A_range = np.logspace(-7, -3.5, 20)   # 0.1 μm → 316 μm

err_boz = np.zeros((len(A_range), N_TRIALS))
err_dir = np.zeros((len(A_range), N_TRIALS))
err_cln = np.zeros((len(A_range), N_TRIALS))

for ia, A_k2 in enumerate(A_range):
    dq    = F_k2 @ np.array([A_k2, 0.0]) + dq_nuisance   # k=2 + nuisance
    y_sig = R1 @ dq                      # gerçek orbit sinyali (b yok)

    for it in range(N_TRIALS):
        b   = RNG.normal(0, SIGMA_BPM,   N_Q)
        eps = RNG.normal(0, SIGMA_NOISE, N_Q)

        y_obs   = y_sig + b + eps          # tek yörünge: sinyal + BPM ofseti + gürültü
        y_kmod  = y_sig + eps              # kmod: BPM ofseti iptal, sadece gürültü

        err_boz[ia, it] = abs(bozoki_k2(y_obs)       - A_k2) / A_k2
        err_dir[ia, it] = abs(direct_ls_k2(y_obs)    - A_k2) / A_k2
        err_cln[ia, it] = abs(kmod_clean_k2(y_kmod)  - A_k2) / A_k2

    snr = A_k2 / SIGMA_BPM
    print(f"  A={A_k2*1e6:7.2f} μm  SNR={snr:.4f}  "
          f"Bozoki={np.median(err_boz[ia])*100:6.1f}%  "
          f"DirLS={np.median(err_dir[ia])*100:6.1f}%  "
          f"CLEAN={np.median(err_cln[ia])*100:6.1f}%")

# ── Tablo yazdır ──────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  {'A_k2 [μm]':>10}  {'SNR':>6}  "
      f"{'Bozoki LS':>12}  {'Direkt LS':>12}  {'kmod+CLEAN':>12}")
print(f"  {'-'*10}  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}")
for ia, A_k2 in enumerate(A_range):
    snr = A_k2 / SIGMA_BPM
    print(f"  {A_k2*1e6:10.2f}  {snr:6.4f}  "
          f"{np.median(err_boz[ia])*100:11.1f}%  "
          f"{np.median(err_dir[ia])*100:11.1f}%  "
          f"{np.median(err_cln[ia])*100:11.1f}%")
print(f"{'='*72}")
print(f"  σ_BPM={SIGMA_BPM*1e6:.0f} μm,  σ_noise={SIGMA_NOISE*1e6:.0f} μm,  "
      f"N_Q={N_Q},  N_trials={N_TRIALS}")
print(f"  Nuisance: k=4,6,8 genlik ≈ 300 μm  (tek yörünge sinyalini bastırır)")
print(f"  kmod: BPM ofseti tam iptali (y_kmod=R·q+ε, b=0); CLEAN nuisance soyar")

# ── Grafik ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

snr_vals = A_range / SIGMA_BPM

METHODS = [
    ('bozoki', err_boz, 'Bozoki LS\n(tek yörünge, η=y/√β)',    '#d62728', '--'),
    ('direct', err_dir, 'R-matris LS\n(tek yörünge, R·F·a=y)', '#ff7f0e', ':'),
    ('clean',  err_cln, 'kmod + CLEAN\n(ofset iptali, R·F·a=y)', '#2ca02c', '-'),
]

# ── Sol panel: hata vs SNR ────────────────────────────────────────────────────
ax = axes[0]
for _, err, label, color, ls in METHODS:
    med = np.median(err, axis=1)
    p25 = np.percentile(err, 25, axis=1)
    p75 = np.percentile(err, 75, axis=1)
    ax.semilogy(snr_vals, med * 100, color=color, ls=ls, lw=2, label=label)
    ax.fill_between(snr_vals, p25 * 100, p75 * 100, color=color, alpha=0.12)

ax.axhline(10, color='gray', lw=0.8, ls='-', alpha=0.5)
ax.text(snr_vals[1], 13, '%10 hata eşiği', color='gray', fontsize=8, va='bottom')
ax.axvline(1.0, color='gray', lw=0.8, ls=':', alpha=0.5)
ax.text(1.05, 0.9, 'SNR=1', color='gray', fontsize=8, va='top',
        transform=ax.get_xaxis_transform(), rotation=90)

ax.set_xlabel('SNR = $A_{k=2}$ / $σ_{\\rm BPM}$', fontsize=11)
ax.set_ylabel('$k=2$ genlik hatası (%, medyan ± IQR)', fontsize=11)
ax.set_title('Yöntem karşılaştırması\n'
             f'$σ_{{\\rm BPM}}$={SIGMA_BPM*1e6:.0f} μm,  '
             f'nuisance k=4,6,8 @ 300 μm', fontsize=10)
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(snr_vals[0], snr_vals[-1])
ax.grid(True, alpha=0.25)

# ── Sağ panel: hata vs mutlak genlik [μm] ────────────────────────────────────
ax2 = axes[1]
A_um = A_range * 1e6

for _, err, label, color, ls in METHODS:
    med = np.median(err, axis=1)
    p25 = np.percentile(err, 25, axis=1)
    p75 = np.percentile(err, 75, axis=1)
    ax2.loglog(A_um, med * 100, color=color, ls=ls, lw=2, label=label)
    ax2.fill_between(A_um, p25 * 100, p75 * 100, color=color, alpha=0.12)

# Referans çizgiler
ax2.axhline(10, color='gray', lw=0.8, ls='-', alpha=0.5)
ax2.axvline(SIGMA_BPM * 1e6, color='#d62728', lw=0.8, ls=':', alpha=0.6)
ax2.text(SIGMA_BPM*1e6*1.05, 500,
         f'$σ_{{\\rm BPM}}$={SIGMA_BPM*1e6:.0f} μm',
         color='#d62728', fontsize=8, va='top', rotation=90)

noise_floor_kmod = math.sqrt(2) * SIGMA_NOISE  # kmod gürültü tabanı (orbit uzayında)
ax2.axvline(noise_floor_kmod * 1e6, color='#2ca02c', lw=0.8, ls=':', alpha=0.6)
ax2.text(noise_floor_kmod*1e6*1.05, 500,
         f'kmod gürültü\n√2·σ={noise_floor_kmod*1e6:.1f} μm',
         color='#2ca02c', fontsize=8, va='top', rotation=90)

# Gerçek k=2 genliği (params.json)
A_k2_true = 10.0  # μm (params.json: amp_cos=1e-5 m)
for h in cfg.get("dy_harmonics", []):
    if h["k"] == 2:
        A_k2_true = math.sqrt(h.get("amp_cos", 0)**2 + h.get("amp_sin", 0)**2) * 1e6
ax2.axvline(A_k2_true, color='purple', lw=1.2, ls='--', alpha=0.7)
ax2.text(A_k2_true*1.1, 500,
         f'pEDM k=2\n({A_k2_true:.0f} μm)', color='purple', fontsize=8, va='top')

ax2.set_xlabel('$A_{k=2}$ [μm]', fontsize=11)
ax2.set_ylabel('$k=2$ genlik hatası (%)', fontsize=11)
ax2.set_title('Mutlak genlik ölçeğinde\n(ok yönü: küçülürken hata artar)', fontsize=10)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.25, which='both')

plt.suptitle('Bozoki (1989) LS — kmod+CLEAN karşılaştırması\n'
             f'pEDM halkası, N_Q={N_Q} quad, {N_TRIALS} Monte Carlo deneme',
             fontsize=11, y=1.02)

plt.tight_layout()
fig.savefig('fig_bozoki_vs_clean.png', dpi=150, bbox_inches='tight')
print(f"\nGrafik kaydedildi: fig_bozoki_vs_clean.png")

# ── Kritik A_k2 değerleri (hata < %10 için minimum genlik) ───────────────────
print(f"\n{'='*55}")
print(f"  %10 hata eşiğini geçen minimum A_k2:")
for _, err, label, color, ls in METHODS:
    med = np.median(err, axis=1)
    idx = np.where(med < 0.10)[0]
    if len(idx) > 0:
        A_min = A_range[idx[0]] * 1e6
        print(f"  {label.split(chr(10))[0]:35s}  A_min ≈ {A_min:.1f} μm  "
              f"(SNR_min ≈ {A_min/SIGMA_BPM*1e-6:.2f})")
    else:
        print(f"  {label.split(chr(10))[0]:35s}  — hiçbir noktada %10 altında —")
print(f"{'='*55}")
print(f"""
Sonuç:
  • Bozoki / direkt LS: BPM ofseti (σ={SIGMA_BPM*1e6:.0f} μm) k=2 sinyalini
    bastırır; A_k2 ≫ σ_BPM olmadan doğru tahmin mümkün değil.
  • kmod+CLEAN: BPM ofseti tam iptal edilir (y=R·q+ε, b=0);
    nuisance CLEAN ile soyulur; gürültü tabanı ≈ √2·σ_noise = {noise_floor_kmod*1e6:.1f} μm.
  • pEDM'deki A_k2 = {A_k2_true:.0f} μm kmod+CLEAN ile başarıyla kestirilebilir.
  • Bozoki 1989'daki k=1 baskın harmonik kendi SNR'ı yüksek olduğu için
    tek yörünge yöntemi çalışmıştı; bizim zayıf k=2 durumumuzda çalışmaz.
""")
