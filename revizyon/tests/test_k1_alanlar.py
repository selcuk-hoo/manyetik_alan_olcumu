"""Katman 1 — alanlar. Her eleman tipinde Maxwell denklemleri (statik, kaynaksız bölge):
∇·E = 0, ∇×E = 0, ∇·B = 0, ∇×B = 0; ayrıca kaydırma/ölçekleme/döndürme özdeşlikleri.

Tolerans: türevler merkezi farkla (h = 10 μm) alınır; eşik, alanın tipik gradyanına göre bağıl
verilir (kuadrupolde G, saptırıcıda E0/R0). Her testin docstring'i neyi ve neden test ettiğini söyler.
"""
import numpy as np
import pytest
from ortak import load_lib, fparams, fields, div_curl, R0, E0_MAGIC

LIB = load_lib("rev")
RNG = np.random.default_rng(12345)


def quad_points(n=20):
    """Kuadrupol (düz eleman) içinde rastgele noktalar: X = R0 + x, |x|,|z| ≤ 5 mm."""
    return [np.array([R0 + RNG.uniform(-5e-3, 5e-3), RNG.uniform(0, 0.4), RNG.uniform(-5e-3, 5e-3)])
            for _ in range(n)]


def arc_points(n=20):
    """Saptırıcı içinde rastgele noktalar: yay açısı 0..0.13 rad, R = R0 ± 5 mm, Z ± 5 mm."""
    pts = []
    for _ in range(n):
        th = RNG.uniform(0, 0.13); R = R0 + RNG.uniform(-5e-3, 5e-3)
        pts.append(np.array([R * np.cos(th), R * np.sin(th), RNG.uniform(-5e-3, 5e-3)]))
    return pts


QUAD_CASES = {
    "ideal QF": dict(etype=2, fp=fparams()),
    "ideal QD": dict(etype=3, fp=fparams()),
    "QF dx,dy kaydırılmış": dict(etype=2, fp=fparams(dx=37e-6, dy=-21e-6)),
    "QF tilt 1 mrad": dict(etype=2, fp=fparams(qtilt=1e-3, dx=10e-6, dy=5e-6)),
    "QF dG %1": dict(etype=2, fp=fparams(dG=0.01)),
    "QF sekstupol": dict(etype=2, fp=fparams(sext=1.0, K2=-0.015)),
    "QF_MOD (hücre 0)": dict(etype=4, fp=fparams(dx=10e-6, dy=10e-6)),
}


@pytest.mark.parametrize("name", list(QUAD_CASES))
def test_kuadrupol_maxwell(name):
    """Kuadrupol alanı ∇·B = 0 ve ∇×B = 0 sağlamalı (tilt ve sekstupol dahil)."""
    c = QUAD_CASES[name]
    G = abs(c["fp"][6])
    for r in quad_points():
        div, curl, _ = div_curl(LIB, c["fp"], c["etype"], r, which="B")
        assert abs(div) < 1e-6 * G, f"{name}: ∇·B = {div:.3e} (G={G})"
        assert np.max(np.abs(curl)) < 1e-6 * G, f"{name}: ∇×B = {curl}"


def test_kuadrupol_kaydirma_ozdesligi():
    """Kaydırılmış kuadrupolün alanı, kaydırılmamış alanın kaydırılmış noktadaki değeridir:
    B(x; d) = B(x − d; 0). Bu, kaçıklığın doğru işaretle uygulandığını sınar."""
    dx, dy = 37e-6, -21e-6
    f0, fd = fparams(), fparams(dx=dx, dy=dy)
    for r in quad_points():
        _, Bd = fields(LIB, r, fd, 2)
        _, B0 = fields(LIB, r - np.array([dx, 0, dy]), f0, 2)
        assert np.allclose(Bd, B0, rtol=0, atol=1e-12), (Bd, B0)


def test_kuadrupol_tilt_tam_donmeyle():
    """Birinci mertebe tilt formülü, tam döndürülmüş kuadrupol alanıyla O(φ²) içinde uyuşmalı.
    Tam döndürme: B'(x) = Rot(φ) B(Rot(−φ) x) (x–z düzleminde, s ekseni etrafında)."""
    phi = 1e-3
    fpt, f0 = fparams(qtilt=phi), fparams()
    c, s = np.cos(phi), np.sin(phi)
    worst = 0.0
    for r in quad_points():
        _, Bt = fields(LIB, r, fpt, 2)
        x, z = r[0] - R0, r[2]
        xr, zr = c * x + s * z, -s * x + c * z           # Rot(−φ)
        _, B0 = fields(LIB, np.array([R0 + xr, r[1], zr]), f0, 2)
        Bx, Bz = B0[0], B0[2]
        Bex = np.array([c * Bx - s * Bz, 0.0, s * Bx + c * Bz])   # Rot(φ)
        worst = max(worst, np.max(np.abs(Bt - Bex)) / (abs(fpt[6]) * 5e-3))
    # tilt yönü (işaret) konvansiyonu farklıysa fark O(φ) olur; O(φ²) bekleniyor
    assert worst < 10 * phi ** 2, f"tilt formülü ile tam döndürme farkı (bağıl) = {worst:.2e}"


DEFL_CASES = {
    "ideal (n=1)": fparams(),
    "n = 1.2": fparams(n=1.2),
    "alan hatası ε = 1e-4": fparams(eps=1e-4),
    "elektrik roll 1e-4": fparams(roll=1e-4),
    "elektrik roll 1e-4, n=1.2": fparams(roll=1e-4, n=1.2),
}


@pytest.mark.parametrize("name", list(DEFL_CASES))
def test_saptirici_maxwell(name):
    """Saptırıcı elektrik alanı ∇·E = 0 ve ∇×E = 0 sağlamalı. Tipik gradyan |E0|/R0 ölçek alınır.
    n ≠ 1 açılımı (Z/R)⁴'te kesildiği için orada küçük bir artık beklenir."""
    fp = DEFL_CASES[name]
    scale = abs(E0_MAGIC) / R0
    worst_div = worst_curl = 0.0
    for r in arc_points():
        div, curl, _ = div_curl(LIB, fp, 0, r, which="E")
        worst_div = max(worst_div, abs(div) / scale)
        worst_curl = max(worst_curl, np.max(np.abs(curl)) / scale)
    print(f"{name}: max|∇·E|/(E0/R0) = {worst_div:.2e}, max|∇×E|/(E0/R0) = {worst_curl:.2e}")
    assert worst_div < 1e-5 and worst_curl < 1e-5, (worst_div, worst_curl)


def test_saptirici_alan_hatasi_olcekler():
    """ε alan hatası E'yi tam (1+ε) ile ölçeklemeli."""
    eps = 1e-4
    for r in arc_points():
        E0, _ = fields(LIB, r, fparams(), 0)
        Ee, _ = fields(LIB, r, fparams(eps=eps), 0)
        assert np.allclose(Ee, E0 * (1 + eps), rtol=1e-12, atol=0)


def test_saptirici_roll_duşey_alan():
    """Referans yörüngede (R = R0, Z = 0) roll φ düşey alan E_z = E0·sin φ vermeli ve
    radyal alanı cos φ ile ölçeklemeli."""
    phi = 1e-4
    th = 0.05
    r = np.array([R0 * np.cos(th), R0 * np.sin(th), 0.0])
    E, _ = fields(LIB, r, fparams(roll=phi), 0)
    Er = E[0] * np.cos(th) + E[1] * np.sin(th)
    assert np.isclose(E[2], E0_MAGIC * np.sin(phi), rtol=1e-12)
    assert np.isclose(Er, E0_MAGIC * np.cos(phi), rtol=1e-12)


def test_manyetik_duzelticiler_duzgun():
    """Saptırıcı konumundaki manyetik düzelticiler düzgün alan: B_v → B[2], B_r → B[0] (yerel radyal)."""
    r = arc_points(1)[0]
    _, B = fields(LIB, r, fparams(Bv=2e-6, Br=-3e-6), 0)
    assert np.isclose(B[2], 2e-6) and np.isclose(B[0], -3e-6) and abs(B[1]) < 1e-18


def test_eski_ve_yeni_motor_ayni_alan():
    """Revizyon motoru, ek parametreler sıfırken eski motorla aynı alanı vermeli (bütün tipler)."""
    old = load_lib("old")
    fp = fparams(dx=10e-6, dy=-7e-6, qtilt=2e-4, sext=1.0, K2=-0.015, dtilt=1e-4)
    for et, pts in ((0, arc_points(5)), (1, quad_points(5)), (2, quad_points(5)),
                    (3, quad_points(5)), (4, quad_points(5))):
        for r in pts:
            Eo, Bo = fields(old, r, fp, et); En, Bn = fields(LIB, r, fp, et)
            assert np.array_equal(Eo, En) and np.array_equal(Bo, Bn), (et, Eo, En, Bo, Bn)
