#!/usr/bin/env python3
"""
bx_profile_analysis.py  —  B_x alan profili ve nT dönüşümü

Kaynaklar:
  - Quad gradyanı G = 0.21 T/m  (integrator.py varsayılan g1)
  - Quad uzunluğu L = 0.40 m
  - Çevre C ≈ 600 m  (24 FODO × ~25 m/hücre)
  - n_FODO = 24,  n_Q = 48  (QF+QD per hücre)

Dönüşüm ilişkisi:
  Quad kaçıklığı δy → lokal B_x = G × δy  (quad merkezinde)

Bu betik:
  1. Mode_tolerance_analysis sonuçlarından B_x lokal alan tablosu
  2. Tüm test modlarının s boyunca B_x profilini çizer
  3. Ölçüm hassasiyetini nm → nT'ye çevirir
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Ring parametreleri ────────────────────────────────────────────────────────
G      = 0.21      # T/m  — quad gradyanı (integrator.py varsayılanı g1)
L_quad = 0.40      # m    — quad uzunluğu
N_FODO = 24        # FODO hücresi sayısı
N_Q    = 48        # toplam quad sayısı (QF+QD)
C      = 600.0     # m    — yaklaşık çevre (24 × 25 m)
f_fill = N_Q * L_quad / C   # doluluk oranı = 3.2%

# Her quad için s koordinatı
s_q  = np.linspace(0, C, N_Q, endpoint=False)

# ── Fourier modu tanımı ───────────────────────────────────────────────────────
def Fcos(k, n=N_Q):
    j = np.arange(n)
    if k == 0: return (-1.0)**j
    return (-1.0)**j * np.cos(2*np.pi*k*(j//2)/(n//2))

def Fsin(k, n=N_Q):
    j = np.arange(n)
    return (-1.0)**j * np.sin(2*np.pi*k*(j//2)/(n//2))

def dq_mode(k, A_m, phi=0.0):
    """k modunun quad kaçıklığı deseni [m]"""
    return A_m * (np.cos(phi)*Fcos(k) + np.sin(phi)*Fsin(k))

# ── B_x lokal alan tablosu ────────────────────────────────────────────────────
print("=" * 65)
print("DÖNÜŞÜM: Quad kaçıklık amplitüdü → lokal B_x (quad merkezinde)")
print(f"  G = {G} T/m,  L_quad = {L_quad} m,  doluluk = {f_fill*100:.1f}%")
print()
print(f"  {'Kaynak':<35}  {'A_k [nm]':>10}  {'B_x [nT]':>10}")
print("  " + "-"*58)

cases = [
    # (isim, A_um)
    ("k=2 sinyal amplitüdü",           10.0),
    ("k=3 sinyal amplitüdü",            8.0),
    ("k=2 ölçüm hassasiyeti (σ_off=100μm)", 0.857),
    ("k=3 ölçüm hassasiyeti (σ_off=100μm)", 3.263),
    ("k=2 ölçüm hassasiyeti (σ_off=20μm)",  0.171),
    ("k=3 ölçüm hassasiyeti (σ_off=20μm)",  0.653),
    ("k=2 ölçüm hassasiyeti (σ_off=5μm)",   0.043),
    ("k=3 ölçüm hassasiyeti (σ_off=5μm)",   0.163),
    ("k=4 kirletici amplitüdü",       300.0),
    ("k=5 kirletici amplitüdü",       250.0),
    ("k=6 kirletici amplitüdü",       300.0),
]
for name, A_um in cases:
    A_nm = A_um * 1e3
    Bx_nT = G * (A_um * 1e-6) * 1e9  # nT
    print(f"  {name:<35}  {A_nm:>10.1f}  {Bx_nT:>10.1f}")

print()
print("  NOT: 'Lokal B_x' = quad içindeki alan (yalnızca quad uzunluğu boyunca).")
print("  Halka genelinde ortalama = lokal × doluluk oranı × (~%3.2)")

# ── σ_off gereksinimi: 1 nT sınırı için ────────────────────────────────────
import importlib.util
spec = importlib.util.spec_from_file_location("mpf", "make_paper_figures.py")
mpf = importlib.util.module_from_spec(spec); spec.loader.exec_module(mpf)
R = mpf._load_R()
n_q = R.shape[0]

def Mk_norm(k):
    Mc = R @ Fcos(k, n_q); Ms = R @ Fsin(k, n_q)
    if k == 0: return float(np.linalg.norm(Mc))
    return float(np.sqrt(np.linalg.norm(Mc)**2 + np.linalg.norm(Ms)**2))

print()
print("=" * 65)
print("1 nT ölçüm hassasiyeti için gereken BPM sistematik ofseti:")
print()
for k_tgt in [2, 3, 4]:
    Mk = Mk_norm(k_tgt)
    # σAk = σ_off / Mk  →  σ_off = σAk × Mk
    # B_x = G × σAk = 1 nT  →  σAk = 1e-9 / G
    sigma_Ak_m = 1e-9 / G   # 1 nT → kaçıklık çözünürlüğü [m]
    sigma_off_needed = sigma_Ak_m * Mk
    sigma_off_um = sigma_off_needed * 1e6
    print(f"  k={k_tgt}:  σAk = {sigma_Ak_m*1e9:.1f} nm  →  σ_offset gerekli = {sigma_off_um:.2f} μm")

# ── B_x profili: tüm test modları ────────────────────────────────────────────
print()
print("=" * 65)
print("B_x profil özeti (tüm test modları)")
print()

# Sadece sinyal modları (k=2,3)
signal_modes = {2: (10e-6, 0.3), 3: (8e-6, 1.1)}
contam_modes = {4: (300e-6, 0.7), 5: (250e-6, 1.3),
                6: (300e-6, 1.2), 7: (180e-6, 0.5), 8: (200e-6, 2.1)}

dq_signal = np.zeros(N_Q)
for k, (A, phi) in signal_modes.items():
    dq_signal += dq_mode(k, A, phi)

dq_total = dq_signal.copy()
for k, (A, phi) in contam_modes.items():
    dq_total += dq_mode(k, A, phi)

Bx_signal = G * dq_signal * 1e6   # μT
Bx_total  = G * dq_total  * 1e6   # μT

print(f"  Sadece k=2,3 sinyal modları:")
print(f"    RMS  B_x = {np.sqrt(np.mean(Bx_signal**2)):.2f} μT")
print(f"    Peak B_x = {np.max(np.abs(Bx_signal)):.2f} μT")
print()
print(f"  Tüm modlar (k=2..8):")
print(f"    RMS  B_x = {np.sqrt(np.mean(Bx_total**2)):.2f} μT")
print(f"    Peak B_x = {np.max(np.abs(Bx_total)):.2f} μT")
A_peak_total = np.sqrt(np.sum([(A**2) for A, _ in list(signal_modes.values()) +
                                                   list(contam_modes.values())]))
print(f"    (Teorik en kötü durum pik: {G*A_peak_total*1e6:.1f} μT, "
      f"tüm fazlar aynı olsaydı)")

# ── Şekil: B_x profili ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle(r"$B_x$ profili: Quad kaçıklık modları (lokal alan, quad merkezleri)",
             fontsize=13)

ax = axes[0]
ax.bar(s_q, Bx_signal, width=C/N_Q*0.7, color="#2166ac", alpha=0.8,
       label=r"$k=2$ (10 μm) + $k=3$ (8 μm)")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel(r"Lokal $B_x$ [μT]")
ax.set_title("Sinyal modları (k=2, 3)")
ax.legend(fontsize=9)
ax.set_xlim(0, C)

ax2 = axes[1]
ax2.bar(s_q, Bx_total, width=C/N_Q*0.7, color="#d6604d", alpha=0.7,
        label="k=2..8 (gerçekçi amplitüdler)")
ax2.bar(s_q, Bx_signal, width=C/N_Q*0.7, color="#2166ac", alpha=0.8,
        label=r"$k=2+3$ sinyal")
ax2.axhline(0, color="k", lw=0.5)
ax2.set_xlabel("s [m]")
ax2.set_ylabel(r"Lokal $B_x$ [μT]")
ax2.set_title("Tüm modlar: sinyal (k=2,3) + kirletici (k=4..8)")
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig("bx_profile.png", dpi=150)
plt.close()
print()
print("bx_profile.png  ✓")

# ── Özet tablo: nm → nT dönüşümü ─────────────────────────────────────────────
print()
print("=" * 65)
print("ÖZET: Ölçüm hassasiyeti (BPM sistematik ofseti limiti)")
print()
print(f"  {'σ_offset':>10}  {'k=2 [nm]':>10}  {'B_x,k=2 [nT]':>14}  "
      f"{'k=3 [nm]':>10}  {'B_x,k=3 [nT]':>14}")
print("  " + "-"*65)

# Test 2 sonuçlarından (mode_tolerance_analysis.py)
t2_data = {
    10:  (90.3,   322.8),
    50:  (430.3,  1665.6),
    100: (857.3,  3262.8),
    200: (1727.9, 6414.7),
    500: (4186.3, 16854.1),
}
for sig_um, (rms2_nm, rms3_nm) in t2_data.items():
    Bx2 = G * rms2_nm * 1e-9 * 1e9  # nT = G [T/m] × nm × 1e-9 m/nm × 1e9 nT/T
    Bx3 = G * rms3_nm * 1e-9 * 1e9
    print(f"  {sig_um:>6} μm  →  {rms2_nm:>10.1f}  {Bx2:>14.1f}  "
          f"{rms3_nm:>10.1f}  {Bx3:>14.1f}")

print()
print("  KAYNAKLAR: BPM ofseti → yörünge → kaçıklık → B_x (lokal, quad içi)")
print("  1 nT için gereken σ_offset (Tablo 2'deki Mk normları ile):")
for k_tgt in [2, 3]:
    Mk = Mk_norm(k_tgt)
    sigma_Ak_m = 1e-9 / G
    sigma_off_um = sigma_Ak_m * Mk * 1e6
    print(f"    k={k_tgt}: σ_offset ≲ {sigma_off_um:.1f} μm")
print()
print("  NOT: 'Spin-eşdeğer B_x' (harici sinüsoidal alan ile karşılaştırma)")
print("  compare_field_harmonics.py çıktısı (R_B) gerektirir:")
print("  B_x_eff = dSy/dt(quad) / R_B [nT]")
print("  (R_B değeri Mac'te çalıştırılacak simülasyondan gelecek)")
