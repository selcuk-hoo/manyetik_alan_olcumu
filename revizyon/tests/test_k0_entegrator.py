"""Katman 0 — sayısal entegratör (GL4). Tek bir saptırıcı elemanı içinde, element sınırı olmadan,
gl4_step_element doğrudan çağrılır. Parçacık referans yörüngeden 2 mm radyal ve 1 mm düşey
saparak fırlatılır; böylece betatron hareketi ve gerçek bir alan değişimi görür.

Testler:
  - toplam enerji  W = γ m c² + q Φ(R),  Φ = −E0 R0 ln(R/R0)  (n = 1 için E_r = E0 R0/R)
  - |S| = 1 korunumu
  - zaman adımı yakınsaması: hata ∝ dt⁴ (GL4 dördüncü mertebe)
  - zamanda geri gitme: GL4 simetrik bir yöntem → ileri n adım + geri n adım = başlangıç
  - faz uzayı hacmi (Liouville): 6×6 tur haritası Jacobian determinantı = 1
"""
import numpy as np
from ortak import load_lib, fparams, step, R0, E0_MAGIC, P_MAG, M_P, Q_E, C_LIGHT

LIB = load_lib("rev")
FP = fparams()                       # ideal saptırıcı, n = 1
DT = 1e-11


def y0(dx=2e-3, dz=1e-3):
    return np.array([R0 + dx, 0.0, dz, 0.0, -P_MAG, 1e-4 * P_MAG, 0.0, -1.0, 0.0])


def total_energy(y):
    p = np.linalg.norm(y[3:6]); R = np.hypot(y[0], y[1])
    W_kin = np.sqrt((p * C_LIGHT) ** 2 + (M_P * C_LIGHT ** 2) ** 2)
    phi = -E0_MAGIC * R0 * np.log(R / R0)
    return W_kin + Q_E * phi


def test_enerji_korunumu():
    """10⁵ adım (1 μs, ~0.1 yay boyu): toplam enerjinin bağıl değişimi."""
    y = y0(); W0 = total_energy(y)
    worst = 0.0
    for _ in range(10):
        y = step(LIB, y, FP, 0, DT, n=10_000)
        worst = max(worst, abs(total_energy(y) - W0) / W0)
    print(f"max |ΔW/W| = {worst:.2e}")
    assert worst < 1e-12


def test_spin_boyu():
    y = step(LIB, y0(), FP, 0, DT, n=100_000)
    s = np.linalg.norm(y[6:9])
    print(f"|S| − 1 = {s - 1:.2e}")
    assert abs(s - 1) < 1e-12


def test_dt_yakinsamasi():
    """Sabit T = 0.64 μs; adım 64 ns'den 1 ns'ye yarılanır. Ardışık hata oranı ≈ 16 (dt⁴).
    Not: çalışma adımı dt = 1e-11 s'de kesme hatası yuvarlama hatasının altında kalır; mertebe ancak
    çok büyük adımlarla görülebilir (ilk denemede 0.4 ns'lik adımlar yuvarlama tabanındaydı)."""
    T = 6.4e-7
    hs = [6.4e-8, 3.2e-8, 1.6e-8, 8e-9, 4e-9, 2e-9, 1e-9]
    F = [step(LIB, y0(), FP, 0, h, n=int(round(T / h))) for h in hs]
    errs = [np.linalg.norm(F[k][:3] - F[-1][:3]) for k in range(len(hs) - 1)]
    ratios = [errs[k] / errs[k + 1] for k in range(len(errs) - 2)]
    print("konum hataları:", ["%.2e" % e for e in errs], " oranlar:", ["%.1f" % r for r in ratios])
    assert all(14 < r < 18 for r in ratios), ratios
    print(f"dt = 1e-11 s için kesme hatası tahmini (T = 0.64 μs): {errs[-2]*(1e-11/2e-9)**4:.1e} m")


def test_zamanda_geri_gitme():
    y = y0()
    yf = step(LIB, y, FP, 0, DT, n=20_000)
    yb = step(LIB, yf, FP, 0, -DT, n=20_000)
    dpos = np.max(np.abs(yb[:3] - y[:3])); dmom = np.max(np.abs(yb[3:6] - y[3:6])) / P_MAG
    print(f"geri dönüş hatası: konum {dpos:.2e} m, momentum (bağıl) {dmom:.2e}")
    assert dpos < 1e-12 and dmom < 1e-12


def test_faz_uzayi_hacmi():
    """6×6 harita Jacobian'ı (merkezi fark) determinantı 1 olmalı (statik alanda Liouville)."""
    base = y0(); n = 20_000
    scale = np.array([1e-6, 1e-6, 1e-6, 1e-6 * P_MAG, 1e-6 * P_MAG, 1e-6 * P_MAG])
    J = np.zeros((6, 6))
    for j in range(6):
        d = np.zeros(9); d[j] = scale[j]
        yp = step(LIB, base + d, FP, 0, DT, n=n); ym = step(LIB, base - d, FP, 0, DT, n=n)
        J[:, j] = ((yp[:6] - ym[:6]) / (2 * scale[j])) * (1 / scale[:6]) * scale[j]
    det = np.linalg.det(J)
    print(f"det J − 1 = {det - 1:.2e}")
    assert abs(det - 1) < 1e-6
