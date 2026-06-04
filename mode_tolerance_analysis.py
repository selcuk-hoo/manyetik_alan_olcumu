#!/usr/bin/env python3
"""
mode_tolerance_analysis.py  —  k=2,3 mod ölçüm toleransı analizi

Üç soruyu yanıtlar:
  1. Modlar arası sızıntı: gerçekçi amplitüdlerde k≥4 modlar k=2,3 tahminini ne kadar kirletiyor?
  2. BPM sistematik ofseti: kalibrasyon hatasının k=2,3 ölçümüne etkisi
  3. R model hatası (fiziksel modeller):
       a) BPM kazanç hatası  — satır-bazlı çarpımsal: R_meas[i,j] = (1+δ_i) R_true[i,j]
       b) Quad gradyan hatası — sütun-bazlı çarpımsal: R_meas[:,j] = R_true[:,j] × (1+εK_j)
     (Element-bazlı bağımsız bozulma fiziksel değildir; bu iki model kullanılır.)

kappa_spin = 7e-6 rad/s/m  (false_edm_mode_scan ampirik değeri, k=1..5, ±%7)
"""
import numpy as np, sys, importlib.util, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

spec = importlib.util.spec_from_file_location("mpf", "make_paper_figures.py")
mpf = importlib.util.module_from_spec(spec); spec.loader.exec_module(mpf)
R = mpf._load_R()
n_q = R.shape[0]   # 48 BPM / quad

KAPPA = 7e-6   # rad/s/m

rng = np.random.default_rng(42)

# ── Fourier modları (unnormalized FODO-antisymmetric) ───────────────────────
def Fcos(k):
    j = np.arange(n_q)
    if k == 0: return (-1.0)**j
    return (-1.0)**j * np.cos(2*np.pi*k*(j//2)/(n_q//2))

def Fsin(k):
    j = np.arange(n_q)
    return (-1.0)**j * np.sin(2*np.pi*k*(j//2)/(n_q//2))

def Mk_norm(k):
    """||M_k|| = sqrt(||RFcos||^2 + ||RFsin||^2)  (cos+sin birlikte)"""
    Mc = R @ Fcos(k); Ms = R @ Fsin(k)
    if k == 0: return float(np.linalg.norm(Mc))
    return float(np.sqrt(np.linalg.norm(Mc)**2 + np.linalg.norm(Ms)**2))

def estimate_k(y_bpm, k, R_est=None):
    """lstsq projeksiyonu: A_k ve faz tahmini (R_est verilmezse global R kullanılır)"""
    if R_est is None: R_est = R
    Mc = R_est @ Fcos(k); Ms = R_est @ Fsin(k)
    M2 = np.column_stack([Mc, Ms])
    a2, _, _, _ = np.linalg.lstsq(M2, y_bpm, rcond=None)
    return float(np.sqrt(a2[0]**2 + a2[1]**2)), float(np.arctan2(a2[1], a2[0]))


# ════════════════════════════════════════════════════════════════════════════
# TEST 1: Modlar arası sızıntı — GERÇEKÇİ amplitüdler
# ════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("TEST 1: Modlar arası sızıntı (gerçekçi amplitüdler, gürültüsüz)")
print()
print("  Senaryo: k=2 (10μm) + k=3 (8μm) sinyal modu")
print("           k=4..8 kirletici modlar gerçekçi amplitüdlerde")
print("           Gürültü yok, R mükemmel biliniyor")
print()

# Gerçekçi amplitüdler: büyük kaçıklık modları (düzeltilmemiş halka)
signal_modes = {2: (10e-6, 0.3), 3: (8e-6, 1.1)}
contam_modes = {4: (300e-6, 0.7), 5: (250e-6, 1.3),
                6: (300e-6, 1.2), 7: (180e-6, 0.5), 8: (200e-6, 2.1)}

# Toplam yörünge: sinyal + kirletici
dq_total = np.zeros(n_q)
for k, (A, phi) in {**signal_modes, **contam_modes}.items():
    dq_total += A * (np.cos(phi)*Fcos(k) + np.sin(phi)*Fsin(k))
y_orb = R @ dq_total

# Sadece kirletici modlar (sinyal yok)
dq_contam = np.zeros(n_q)
for k, (A, phi) in contam_modes.items():
    dq_contam += A * (np.cos(phi)*Fcos(k) + np.sin(phi)*Fsin(k))
y_contam_only = R @ dq_contam

print(f"  {'Durum':<35}  {'k=2 tahmini':>14}  {'k=3 tahmini':>14}")
print("  " + "-"*66)

A2_true, _ = estimate_k(R @ (10e-6*(np.cos(0.3)*Fcos(2)+np.sin(0.3)*Fsin(2))), 2)
A3_true, _ = estimate_k(R @ (8e-6*(np.cos(1.1)*Fcos(3)+np.sin(1.1)*Fsin(3))), 3)
print(f"  {'Sadece sinyal (referans)':<35}  {A2_true*1e9:>12.1f} nm  {A3_true*1e9:>12.1f} nm")

A2_full, _ = estimate_k(y_orb, 2)
A3_full, _ = estimate_k(y_orb, 3)
err2 = (A2_full - signal_modes[2][0]) * 1e9
err3 = (A3_full - signal_modes[3][0]) * 1e9
print(f"  {'Sinyal + kirletici (k=4..8)':<35}  {A2_full*1e9:>12.1f} nm  {A3_full*1e9:>12.1f} nm")
print(f"  {'  → hata':<35}  {err2:>+12.1f} nm  {err3:>+12.1f} nm")
print()

# Her kirletici modun ayrı ayrı katkısı
print("  Kirletici modların ayrı ayrı k=2,3'e sızıntısı:")
print(f"  {'mod j  Amp':>14}  {'k=2 siz':>12}  {'k=3 siz':>12}")
for k, (A, phi) in sorted(contam_modes.items()):
    dq_k = A * (np.cos(phi)*Fcos(k) + np.sin(phi)*Fsin(k))
    y_k  = R @ dq_k
    A2k, _ = estimate_k(y_k, 2)
    A3k, _ = estimate_k(y_k, 3)
    print(f"  k={k}: {A*1e6:>5.0f}μm  →  k=2: {A2k*1e9:>+9.1f} nm  "
          f"k=3: {A3k*1e9:>+9.1f} nm")

dSy2 = KAPPA * Mk_norm(2) * abs(err2)*1e-9
dSy3 = KAPPA * Mk_norm(3) * abs(err3)*1e-9
print(f"\n  Toplam sızıntı hatası → false-EDM katkısı:")
print(f"    k=2: {abs(err2):.0f} nm hata → dSy/dt = {dSy2:.2e} rad/s")
print(f"    k=3: {abs(err3):.0f} nm hata → dSy/dt = {dSy3:.2e} rad/s")


# ════════════════════════════════════════════════════════════════════════════
# TEST 2: BPM sistematik ofseti
# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("TEST 2: BPM sistematik ofseti (quad kaçıklık yok)")
print("  σ_offset: her BPM'nin mekanik/elektronik sabit ofseti [RMS]")
print("  500 Monte Carlo örneği")
print()
print(f"  {'σ_offset':>10}  {'k=2 false signal [nm]':>22}  {'dSy/dt':>12}  "
      f"{'k=3 false signal [nm]':>22}  {'dSy/dt':>12}")
print("  " + "-"*88)

for sig_um in [10, 50, 100, 200, 500]:
    sig = sig_um * 1e-6
    A2_list, A3_list = [], []
    for _ in range(500):
        b = rng.normal(0, sig, n_q)
        A2, _ = estimate_k(b, 2); A3, _ = estimate_k(b, 3)
        A2_list.append(A2); A3_list.append(A3)
    rms2 = float(np.sqrt(np.mean(np.array(A2_list)**2)))
    rms3 = float(np.sqrt(np.mean(np.array(A3_list)**2)))
    ds2 = KAPPA * Mk_norm(2) * rms2
    ds3 = KAPPA * Mk_norm(3) * rms3
    print(f"  {sig_um:>6} μm  →  k=2: {rms2*1e9:>12.1f} nm  {ds2:>12.2e} rad/s  "
          f"k=3: {rms3*1e9:>12.1f} nm  {ds3:>12.2e} rad/s")


# ════════════════════════════════════════════════════════════════════════════
# TEST 3a: R model hatası — BPM KAZANÇ HATASI (fiziksel model)
# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("TEST 3a: R model hatası — BPM kazanç hatası (satır-bazlı)")
print("  R_meas[i,j] = (1 + δ_i) × R_true[i,j],  δ_i ~ N(0, σ_gain)")
print("  Fiziksel anlam: BPM i'nin kazanç kalibrasyonu σ_gain oranında yanlış")
print("  Gercek k=2=10μm, k=3=8μm; kirletici modlar yok; gürültüsüz BPM")
print()

dq_sig = np.zeros(n_q)
for k, (A, phi) in signal_modes.items():
    dq_sig += A * (np.cos(phi)*Fcos(k) + np.sin(phi)*Fsin(k))

print(f"  {'σ_gain':>8}  {'k=2 std':>10}  {'k=2 bias':>10}  "
      f"{'k=3 std':>10}  {'k=3 bias':>10}")
print("  " + "-"*56)
for sig_pct in [0, 0.5, 1, 2, 5]:
    sig = sig_pct / 100.0
    A2_list, A3_list = [], []
    for _ in range(300):
        delta = rng.normal(0, sig, n_q)           # per-BPM gain error
        R_meas = R * (1.0 + delta[:, None])       # row-wise: each BPM row scaled
        y_true = R @ dq_sig                        # true orbit with ideal R
        A2, _ = estimate_k(y_true, 2, R_est=R_meas)
        A3, _ = estimate_k(y_true, 3, R_est=R_meas)
        A2_list.append(A2); A3_list.append(A3)
    a2 = np.array(A2_list); a3 = np.array(A3_list)
    b2 = (a2.mean() - signal_modes[2][0]) * 1e9
    b3 = (a3.mean() - signal_modes[3][0]) * 1e9
    print(f"  {sig_pct:>5} %  →  k=2 std={a2.std()*1e9:>7.1f}nm  bias={b2:>+8.1f}nm  "
          f"k=3 std={a3.std()*1e9:>7.1f}nm  bias={b3:>+8.1f}nm")


# ════════════════════════════════════════════════════════════════════════════
# TEST 3b: R model hatası — QUAD GRADYAN HATASI (fiziksel model)
# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("TEST 3b: R model hatası — quad gradyan hatası (sütun-bazlı)")
print("  R_meas[:,j] = R_true[:,j] × (1 + εK_j),  εK_j ~ N(0, σ_grad)")
print("  Fiziksel anlam: j'inci quadın entegre gradyanı σ_grad oranında yanlış")
print()

print(f"  {'σ_grad':>8}  {'k=2 std':>10}  {'k=2 bias':>10}  "
      f"{'k=3 std':>10}  {'k=3 bias':>10}")
print("  " + "-"*56)
for sig_pct in [0, 0.5, 1, 2, 5]:
    sig = sig_pct / 100.0
    A2_list, A3_list = [], []
    for _ in range(300):
        eps = rng.normal(0, sig, n_q)             # per-quad gradient error
        R_meas = R * (1.0 + eps[None, :])         # col-wise: each quad column scaled
        y_true = R @ dq_sig
        A2, _ = estimate_k(y_true, 2, R_est=R_meas)
        A3, _ = estimate_k(y_true, 3, R_est=R_meas)
        A2_list.append(A2); A3_list.append(A3)
    a2 = np.array(A2_list); a3 = np.array(A3_list)
    b2 = (a2.mean() - signal_modes[2][0]) * 1e9
    b3 = (a3.mean() - signal_modes[3][0]) * 1e9
    print(f"  {sig_pct:>5} %  →  k=2 std={a2.std()*1e9:>7.1f}nm  bias={b2:>+8.1f}nm  "
          f"k=3 std={a3.std()*1e9:>7.1f}nm  bias={b3:>+8.1f}nm")


# ════════════════════════════════════════════════════════════════════════════
# ÖZET: Gerçekçi floor tablosu
# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("ÖZET: Limitasyon hiyerarşisi (tipik değerler)")
print()

scenarios = [
    ("Gaussian BPM gürültüsü, tek çekim",   "σ_noise=10μm",    1e-5/Mk_norm(2), 1e-5/Mk_norm(3)),
    ("Gaussian BPM gürültüsü, 10^4 tur",    "σ_noise=0.1μm",   1e-7/Mk_norm(2), 1e-7/Mk_norm(3)),
    ("BPM sistematik ofset (BBA öncesi)",   "σ_off=200μm",     200e-6/Mk_norm(2)*1.1, 200e-6/Mk_norm(3)*1.1),
    ("BPM sistematik ofset (BBA sonrası)",  "σ_off=20μm",      20e-6/Mk_norm(2)*1.1,  20e-6/Mk_norm(3)*1.1),
    ("%1 BPM kazanç hatası",                "σ_gain=1%",       10e-6*0.01, 8e-6*0.01),
    ("%1 quad gradyan hatası",              "σ_grad=1%",       10e-6*0.012, 8e-6*0.025),
    ("Sızıntı (k=4..8, 200-300μm)",         "toplam",          abs(err2)*1e-9, abs(err3)*1e-9),
]
print(f"  {'Kaynak':<42}  {'σAk2 [nm]':>10}  {'dSy2 [rad/s]':>14}  "
      f"{'σAk3 [nm]':>10}  {'dSy3 [rad/s]':>14}")
print("  " + "-"*98)
for name, param, dAk2, dAk3 in scenarios:
    ds2 = KAPPA * Mk_norm(2) * dAk2
    ds3 = KAPPA * Mk_norm(3) * dAk3
    print(f"  {name:<42}  {dAk2*1e9:>10.1f}  {ds2:>14.2e}  "
          f"{dAk3*1e9:>10.1f}  {ds3:>14.2e}")

print()
print("  NOT: BPM sistematik ofseti ölçüm limitasyonunun ana kaynağı.")
print("  Beam-based alignment (BBA) ile σ_off ~200μm → ~20μm indirilebilir.")
print("  kappa_spin = 7e-6 rad/s/m (false_edm_mode_scan, k=1..5)")
