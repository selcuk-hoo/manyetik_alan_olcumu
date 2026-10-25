"""Katman 5 — revizyon betiklerinin Python doğrusal cebir katmanı (C++ gerektirmez, hızlı).

Bu katman, Adım 1'de yapılan türden model/kod hatalarını (örn. ρ=0'da yanlışlıkla eklenen bağımsız
BPM hareketi) erken yakalamak içindir.
"""
import os, sys
import numpy as np
import pytest
from ortak import REV, BASE, NQ

sys.path.insert(0, REV)
import adim1_bpm_model as A1
import adim4_correlated as A4
import adim5_harmonics as A5


def test_projektorler():
    """P_s + P_a = I, P_s P_a = 0, P_s² = P_s (dik izdüşüm)."""
    Ps, Pa = A1.PS, A1.PA
    assert np.allclose(Ps + Pa, np.eye(NQ))
    assert np.allclose(Ps @ Pa, 0)
    assert np.allclose(Ps @ Ps, Ps) and np.allclose(Ps, Ps.T)
    # aynı tanım üç betikte de aynı olmalı
    assert np.allclose(A4.PS, Ps) and np.allclose(A4.PA, Pa)


def test_harmonik_desenlerin_turu():
    """adim5 desenleri gerçekten saf simetrik / saf antisimetrik olmalı."""
    for m in range(13):
        for ph in (0.0, np.pi / 2):
            vs = A5.pattern(m, "sym", ph); va = A5.pattern(m, "anti", ph)
            assert np.allclose(A1.PA @ vs, 0) and np.allclose(A1.PS @ va, 0)


@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
def test_gurultusuz_tam_geri_bulma(rho):
    """Gürültüsüz, model tam doğruyken ρ-bilen iki demet kestirici (E4) sürüklenmeyi geri bulmalı
    (TSVD'nin attığı alt uzay hariç; çok küçük rcond ile)."""
    rng = np.random.default_rng(0)
    for pl in "xy":
        Rc, Rw = A1.R[pl]["ccw"], A1.R[pl]["cw"]
        dm = rng.normal(0, 1e-6, NQ)
        y = np.concatenate([(Rc - rho * np.eye(NQ)) @ dm, (Rw - rho * np.eye(NQ)) @ dm])
        est = np.linalg.pinv(np.vstack([Rc - rho * np.eye(NQ), Rw - rho * np.eye(NQ)]), rcond=1e-12) @ y
        assert np.linalg.norm(est - dm) / np.linalg.norm(dm) < 1e-6


def test_adim1_rho0_lab_sabit_bpm_hareketi_yok():
    """Adım 1 'k0' ailesinde ρ=0, BPM hareketi u=0 demektir: ρ=0 ve gürültü 0'da E3 artığı yalnızca
    TSVD kesmesinden gelmeli ve aile 'eq'den farklı olmalı (ilk sürümdeki hatanın regresyon testi)."""
    r_k0 = A1.make_case(0, "sym", 5e-6, 0.0, 0.0, "k0", 1e-12)
    r_eq = A1.make_case(0, "sym", 5e-6, 0.0, 0.0, "eq", 1e-12)
    assert np.std(r_k0["x"]["E3"]) < 1e-12           # tam ters + gürültüsüz → sıfır artık
    assert np.std(r_eq["x"]["E3"]) > 1e-7            # eşit-rms ailesi bağımsız BPM hareketi içerir


def test_rijit_bpm_uniform_sifir_modu():
    """Rijit bağlı BPM'de (R − I) dikey düzlemde neredeyse tekildir: bütün kuadrupollerin birlikte
    düşey kayması demetçe izlenir ve BPM'lerde görünmez (Adım 1, 5)."""
    Rc = A1.R["y"]["ccw"]
    u = np.ones(NQ) / np.sqrt(NQ)
    assert np.linalg.norm((Rc - np.eye(NQ)) @ u) < 0.1          # Adım 5: rijit kazanç 0.056
    s = np.linalg.svd(Rc - np.eye(NQ), compute_uv=False)
    assert s[-1] / s[0] < 1e-3


def test_seed_tekrar_uretilebilirlik():
    a = A1.make_case(3, "gen", 5e-6, 0.5, 14e-9, "k0", 1e-2)
    b = A1.make_case(3, "gen", 5e-6, 0.5, 14e-9, "k0", 1e-2)
    assert all(np.array_equal(a[pl][m], b[pl][m]) for pl in "xy" for m in a[pl])
    p1 = A4.pattern("rw", 5, "y"); p2 = A4.pattern("rw", 5, "y")
    assert np.array_equal(p1, p2)


def test_cw_ccw_matris_isaret_yapisi():
    """Kuadrupol tepki matrisleri CW ile CCW arasında güçlü negatif korelasyonlu (manyetik tekme
    yönle işaret değiştirir); sütun ortalaması ≈ −0.95 (Adım 3 işaret kontrolü)."""
    for pl in "xy":
        Rc, Rw = A1.R[pl]["ccw"], A1.R[pl]["cw"]
        c = np.mean([np.corrcoef(Rc[:, k], Rw[:, k])[0, 1] for k in range(NQ)])
        assert -0.97 < c < -0.93, c
