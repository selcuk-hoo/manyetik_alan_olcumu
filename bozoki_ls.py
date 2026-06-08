#!/usr/bin/env python3
"""bozoki_ls.py — Bozoki (1989) LS vs CLEAN+kmod Monte Carlo karşılaştırması.

Gerçekçi senaryo (params.json'dan):
  Hedef   : k=2 @ A_k2 (tarama: 0.1 – 316 μm)
  Arka plan: k=4@300μm, k=6@300μm, k=8@200μm (deterministik, params.json)
  BPM ofseti: σ_b = 100 μm
  Gürültü  : σ_ε = 1 μm

Yöntemler:
  1. Bozoki LS    : η=y/√β, azimut baz cos(2θ)/sin(2θ), TEK yörünge
  2. R-matris LS  : R·F_k2·a=y, SADECE k=2 fit — k=4,6,8 modellenmiyor
  3. CLEAN        : iteratif k=0..8, TEK yörünge (b var)
  4. CLEAN + kmod : CLEAN + BPM ofseti tam iptali (b=0)

Neden CLEAN > R-matris LS:
  k=4,6,8 R-uzayında k=2'ye neredeyse ortogonal (~0.01) ama sıfır değil;
  300 μm × 0.01 = ~3 μm sabit kontaminasyon.  A_k2 < ~30 μm bölgesinde
  bu %10'u geçer → R-LS yeter değil; CLEAN doğru modeller ve çıkarır.
"""

import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from fourier_reconstruct import clean_reconstruct, fodo_basis

with open("params.json") as fh:
    cfg = json.load(fh)

N_Q  = 2 * int(cfg["nFODO"])
G    = float(cfg["g1"])
L_Q  = float(cfg["quadLen"])
M2, AMU = 0.938272046, 1.792847356
Brho = (M2 / math.sqrt(AMU)) / 0.29979246
KL   = G * L_Q / Brho
R1   = np.load("R_dy_1.npy")

Q_y    = 1.732
cot_y  = math.cos(math.pi * Q_y) / math.sin(math.pi * Q_y)
KL_sgn = np.where(np.arange(N_Q) % 2 == 0, +KL, -KL)
beta_y = np.diag(R1) / (KL_sgn * cot_y / 2)
sqrt_bet = np.sqrt(np.abs(beta_y))

F_k2, _ = fodo_basis(N_Q, [2], antisym=True)
CANDIDATE_KS = list(range(9))

# arka plan: params.json dy_harmonics, k != 2
dq_bg = np.zeros(N_Q)
bg_info = []
for h in cfg.get("dy_harmonics", []):
    if h["k"] != 2:
        Fh, _ = fodo_basis(N_Q, [h["k"]], antisym=True)
        amp = math.sqrt(h.get("amp_cos", 0)**2 + h.get("amp_sin", 0)**2)
        dq_bg += Fh @ np.array([h.get("amp_cos", 0.0), h.get("amp_sin", 0.0)])
        bg_info.append((h["k"], amp * 1e6))

print("Arka plan harmonikleri (dy_harmonics, k≠2):")
for k, a in bg_info:
    print(f"  k={k}: {a:.0f} μm")

# yöntem fonksiyonları
theta   = 2 * np.pi * np.arange(N_Q) / N_Q
Phi_boz = np.column_stack([np.cos(2*theta), np.sin(2*theta)])
M_dir   = R1 @ F_k2

def _boz(y):
    eta = y / sqrt_bet
    a, _, _, _ = np.linalg.lstsq(Phi_boz, eta, rcond=None)
    return math.sqrt(a[0]**2 + a[1]**2)

def _rls(y):
    a, _, _, _ = np.linalg.lstsq(M_dir, y, rcond=None)
    return math.sqrt(a[0]**2 + a[1]**2)

def _clean(y_in):
    acc, _, _ = clean_reconstruct([R1], [y_in], CANDIDATE_KS,
                                   antisym=True, gain=0.2, max_iter=300, tol=1e-4)
    a = acc[2]
    return math.sqrt(a[0]**2 + a[1]**2)

# gürültüsüz kontaminasyon testi
A2d = 10e-6
y_d = R1 @ (F_k2 @ [A2d, 0.0] + dq_bg)
boz_d   = _boz(y_d)
rls_d   = _rls(y_d)
cln_d   = _clean(y_d)
contam  = abs(rls_d - cln_d) * 1e6    # μm sabit kontaminasyon

print(f"\n{'='*65}")
print(f"  GÜRÜLTÜSÜZ (A_k2={A2d*1e6:.0f} μm, arka plan k=4,6,8):")
print(f"    Bozoki LS         : {boz_d*1e6:7.2f} μm → %{abs(boz_d-A2d)/A2d*100:.0f} baz sapması")
print(f"    R-matris LS       : {rls_d*1e6:7.3f} μm → %{abs(rls_d-A2d)/A2d*100:.1f} (k=4,6,8 kontam.)")
print(f"    CLEAN (k=0..8)    : {cln_d*1e6:7.3f} μm → %{abs(cln_d-A2d)/A2d*100:.1f}")
print(f"  → Sabit kontaminasyon: {contam:.2f} μm  "
      f"(A_k2={A2d*1e6:.0f}μm'de %{contam/(A2d*1e6)*100:.1f};  "
      f"A_k2={contam*10:.0f}μm'de %10 olur)")
print(f"{'='*65}\n")

# Monte Carlo
print("Monte Carlo başlıyor...")
RNG         = np.random.default_rng(42)
N_TRIALS    = 80
SIGMA_BPM   = 100e-6
SIGMA_NOISE =   1e-6
A_range = np.logspace(-7, -3.5, 20)

err_boz = np.zeros((len(A_range), N_TRIALS))
err_rls = np.zeros((len(A_range), N_TRIALS))
err_cnk = np.zeros((len(A_range), N_TRIALS))
err_cln = np.zeros((len(A_range), N_TRIALS))

for ia, A_k2 in enumerate(A_range):
    y_sig = R1 @ (F_k2 @ [A_k2, 0.0] + dq_bg)
    for it in range(N_TRIALS):
        b    = RNG.normal(0, SIGMA_BPM,   N_Q)
        eps  = RNG.normal(0, SIGMA_NOISE, N_Q)
        y_o  = y_sig + b + eps
        y_km = y_sig + eps
        err_boz[ia, it] = abs(_boz(y_o)    - A_k2) / A_k2
        err_rls[ia, it] = abs(_rls(y_o)    - A_k2) / A_k2
        err_cnk[ia, it] = abs(_clean(y_o)  - A_k2) / A_k2
        err_cln[ia, it] = abs(_clean(y_km) - A_k2) / A_k2
    print(f"  A={A_k2*1e6:7.2f}μm  SNR={A_k2/SIGMA_BPM:.4f}  "
          f"Bozoki={np.median(err_boz[ia])*100:6.1f}%  "
          f"R-LS={np.median(err_rls[ia])*100:6.1f}%  "
          f"CLEAN={np.median(err_cnk[ia])*100:6.1f}%  "
          f"CLEAN+kmod={np.median(err_cln[ia])*100:6.1f}%")

# tablo
print(f"\n{'='*80}")
print(f"  {'A_k2':>8}  {'SNR':>6}  {'Bozoki':>9}  "
      f"{'R-LS':>9}  {'CLEAN':>8}  {'CLEAN+kmod':>10}")
print(f"  {'-'*8}  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*10}")
for ia, A_k2 in enumerate(A_range):
    print(f"  {A_k2*1e6:7.2f}μm  {A_k2/SIGMA_BPM:6.4f}  "
          f"{np.median(err_boz[ia])*100:8.1f}%  "
          f"{np.median(err_rls[ia])*100:8.1f}%  "
          f"{np.median(err_cnk[ia])*100:7.1f}%  "
          f"{np.median(err_cln[ia])*100:9.1f}%")
print(f"{'='*80}")

# grafik
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
A_um = A_range * 1e6
snr  = A_range / SIGMA_BPM
METHODS = [
    (err_boz, 'Bozoki LS (azimut baz)',             '#d62728', '--'),
    (err_rls, 'R-matris LS (k=2 only)',             '#ff7f0e', ':'),
    (err_cnk, 'CLEAN k=0..8 (tek yörünge, b var)',  '#1f77b4', '-.'),
    (err_cln, 'CLEAN + kmod (b iptal)',              '#2ca02c', '-'),
]
for ax, xv, xl in [(axes[0], snr,  r'SNR = $A_{k=2}$ / $σ_\mathrm{BPM}$'),
                   (axes[1], A_um, r'$A_{k=2}$ [μm]')]:
    for err, lbl, col, ls in METHODS:
        med=np.median(err,axis=1); p25=np.percentile(err,25,axis=1); p75=np.percentile(err,75,axis=1)
        fn = ax.semilogy if ax is axes[0] else ax.loglog
        fn(xv, med*100, color=col, ls=ls, lw=2, label=lbl)
        ax.fill_between(xv, p25*100, p75*100, color=col, alpha=0.12)
    ax.axhline(10, color='gray', lw=0.8, ls='-', alpha=0.5)
    ax.set_xlabel(xl, fontsize=11)
    ax.set_ylabel(r'$k=2$ genlik hatası (%, medyan±IQR)', fontsize=11)
    ax.legend(fontsize=8.5, loc='upper right')
    ax.grid(True, alpha=0.25, which='both')

# sabit kontaminasyon çizgisi (R-LS için teorik plateau)
axes[1].plot(A_um, contam / (A_range*1e6) * 100, color='#ff7f0e',
             lw=0.9, ls='--', alpha=0.5, label=f'%{contam:.1f}μm/A plateau')

A_k2_true = 10.0
for h in cfg.get("dy_harmonics", []):
    if h["k"] == 2:
        A_k2_true = math.sqrt(h.get("amp_cos",0)**2 + h.get("amp_sin",0)**2) * 1e6
axes[1].axvline(A_k2_true, color='purple', lw=1.2, ls='--', alpha=0.7)
axes[1].text(A_k2_true*1.1, 80, f'pEDM k=2\n({A_k2_true:.0f}μm)', color='purple', fontsize=8)
for k, amp in bg_info:
    axes[1].axvline(amp, color='gray', lw=0.6, ls=':', alpha=0.5)

bg_str = ", ".join(f"k={k}@{a:.0f}μm" for k, a in bg_info)
plt.suptitle(
    f'Bozoki LS — R-matris LS — CLEAN (gerçekçi arka plan: {bg_str})\n'
    f'pEDM N_Q={N_Q},  σ_b={SIGMA_BPM*1e6:.0f}μm,  σ_ε={SIGMA_NOISE*1e6:.0f}μm,  {N_TRIALS} MC',
    fontsize=9, y=1.02)
plt.tight_layout()
fig.savefig('fig_bozoki_vs_clean.png', dpi=150, bbox_inches='tight')
print("\nGrafik kaydedildi: fig_bozoki_vs_clean.png")

i10 = int(np.argmin(np.abs(A_range - A_k2_true*1e-6)))
thresh = {}
for name, err in [('Bozoki',err_boz),('R-LS',err_rls),('CLEAN',err_cnk),('CLEAN+kmod',err_cln)]:
    idx = np.where(np.median(err, axis=1) < 0.10)[0]
    thresh[name] = A_range[idx[0]]*1e6 if len(idx) else None

print(f"""
{'='*65}
  A_k2 = {A_range[i10]*1e6:.0f} μm (pEDM):
    Bozoki LS        : {np.median(err_boz[i10])*100:.1f}%  (baz uyumsuzluğu)
    R-matris LS      : {np.median(err_rls[i10])*100:.1f}%  (k=4,6,8 kontam. {contam:.2f} μm sabit)
    CLEAN (kmod yok) : {np.median(err_cnk[i10])*100:.1f}%  (kontam. giderildi, σ_b sınırlı)
    CLEAN + kmod     : {np.median(err_cln[i10])*100:.1f}%  (hem kontam. hem b iptal)

  %10 hata eşiği minimum A_k2:
""" + "\n".join(
    f"    {n:15s}: {'%.1f μm' % v if v else '—'}" for n,v in thresh.items()
) + f"""

  Sonuçlar:
  [1] Bozoki: gürültüsüz bile ~%300 iç baz sapması → kullanılamaz.
  [2] R-matris LS: k=4,6,8 modellenmediği için {contam:.2f} μm sabit
      kontaminasyon; A_k2 < {contam*10:.0f} μm'de %10'u geçer.
  [3] CLEAN: k=4,6,8 iteratif soyulur → kontaminasyon giderilir;
      artık sınır BPM ofseti (σ_b={SIGMA_BPM*1e6:.0f}μm) olur.
  [4] CLEAN + kmod: b de iptal; gürültü tabanı √2·{SIGMA_NOISE*1e6:.0f} = {math.sqrt(2)*SIGMA_NOISE*1e6:.1f}μm.
{'='*65}
""")
