"""Katman 3 (tek eleman) — Thomas–BMT spin denkleminin fiziksel özdeşlikleri.

compute_rhs ile verilen bir durumdaki anlık türevler (dp/dt, dS/dt) okunur ve analitik
beklentiyle karşılaştırılır. Her testte spin momentuma paralel (boyuna) başlatılır; böylece
"spin momentuma göre ne kadar döner" doğrudan ölçülür:
    momentum yönünün dönme hızı  ω_p = |d p̂/dt|
    spinin dönme hızı            ω_s = |dS/dt|        (S birim vektör, S ∥ p̂)
Beklentiler (G = anomali, γ Lorentz faktörü):
    saf manyetik alan, B ⊥ v :  ω_s / ω_p = 1 + Gγ
    saf elektrik alan, sihirli momentum: ω_s / ω_p = 1  (donmuş spin)
    sihirli olmayan momentum, elektrik alan: göreli hız = (q/m)|G − 1/(γ²−1)| β|E|/c
    EDM: Ω_EDM = (q/m)(η/2)|E|/c  (radyal E için S_z'yi büyütür)
"""
import ctypes
import numpy as np
import pytest
from ortak import (load_lib, fparams, R0, E0_MAGIC, P_MAG, M_P, Q_E, C_LIGHT, G_P, GAMMA0,
                   BETA0, T_REV, CIRC, L_DEF, NQ)

LIB = load_lib("rev")
EDM_ETA = 1.88e-15


def rhs(y, fp, etype):
    yb = (ctypes.c_double * 9)(*y); d = (ctypes.c_double * 9)()
    f = np.ascontiguousarray(fp, dtype=float)
    LIB.compute_rhs(0.0, yb, f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), etype, d, 9)
    return np.array(d[:])


def state(pos, p_vec, s_vec):
    return np.concatenate([pos, p_vec, s_vec])


def rates(y, fp, etype):
    """ω_p (momentum yönü dönme hızı), dS/dt, dp̂/dt."""
    d = rhs(y, fp, etype)
    p = y[3:6]; pn = np.linalg.norm(p); ph = p / pn
    dp = d[3:6]
    dph = (dp - ph * (ph @ dp)) / pn
    return dph, d[6:9]


@pytest.mark.parametrize("direction", [+1, -1])
def test_manyetik_alanda_spin_1_arti_Ggamma(direction):
    """Kaydırılmış kuadrupol (Z = 0'da düzgün radyal B) içinde, boyuna spin momentumdan
    (1 + Gγ) kat hızlı dönmeli — iki demet yönünde de."""
    fp = fparams(dy=50e-6)
    pos = np.array([R0, 0.0, 0.0])
    p = np.array([0.0, direction * P_MAG, 0.0]); s = np.array([0.0, direction, 0.0])
    dph, dS = rates(state(pos, p, s), fp, 2)
    ratio = np.linalg.norm(dS) / np.linalg.norm(dph)
    expected = 1 + G_P * GAMMA0
    print(f"yön {direction:+d}: ω_s/ω_p = {ratio:.9f}, beklenen 1+Gγ = {expected:.9f}")
    assert np.isclose(ratio, expected, rtol=1e-9)
    # dönme aynı yönde olmalı (spin ve momentum aynı tarafa döner)
    assert dS @ dph > 0


@pytest.mark.parametrize("direction", [+1, -1])
def test_elektrik_alanda_donmus_spin(direction):
    """İdeal saptırıcıda sihirli momentumla, boyuna spin momentumu birebir izlemeli:
    dS/dt = dp̂/dt (göreli presesyon sıfır)."""
    fp = fparams(dir=direction)
    pos = np.array([R0, 0.0, 0.0])
    p = np.array([0.0, direction * P_MAG, 0.0]); s = np.array([0.0, direction, 0.0])
    dph, dS = rates(state(pos, p, s), fp, 0)
    rel = np.linalg.norm(dS - dph) / np.linalg.norm(dph)
    print(f"yön {direction:+d}: |dS/dt − dp̂/dt| / |dp̂/dt| = {rel:.2e}")
    assert rel < 1e-9


def test_dusey_elektrik_alanda_donmus_spin():
    """Roll'lu saptırıcıda (düşey E bileşeni) da spin momentumu izlemeli: dikey düzlemde de
    elektrik alan sihirli momentumda spini momentuma göre döndürmez (Adım 2 mekanizmasının temeli)."""
    fp = fparams(roll=1e-3)
    pos = np.array([R0, 0.0, 0.0]); p = np.array([0.0, -P_MAG, 0.0]); s = np.array([0.0, -1.0, 0.0])
    dph, dS = rates(state(pos, p, s), fp, 0)
    assert abs(dph[2]) > 0                                  # gerçekten düşey kuvvet var
    rel = np.linalg.norm(dS - dph) / np.linalg.norm(dph)
    assert rel < 1e-9, rel


@pytest.mark.parametrize("delta", [1e-4, -1e-4, 1e-3])
def test_sihirli_olmayan_momentumda_goreli_presesyon(delta):
    """Momentum sihirli değerden δ kadar saparsa, spin elektrik alanda momentuma göre
    Ω_rel = (q/m)|G − 1/(γ²−1)| β|E|/c hızıyla döner (BMT, B = 0, β·E = 0)."""
    fp = fparams()
    pmag = P_MAG * (1 + delta)
    g = np.sqrt(1 + (pmag / (M_P * C_LIGHT)) ** 2); b = np.sqrt(1 - 1 / g ** 2)
    pos = np.array([R0, 0.0, 0.0]); p = np.array([0.0, -pmag, 0.0]); s = np.array([0.0, -1.0, 0.0])
    dph, dS = rates(state(pos, p, s), fp, 0)
    measured = np.linalg.norm(dS - dph)
    expected = (Q_E / M_P) * abs(G_P - 1 / (g ** 2 - 1)) * b * abs(E0_MAGIC) / C_LIGHT
    print(f"δ = {delta:+.0e}: göreli presesyon {measured:.6e} rad/s, beklenen {expected:.6e}")
    assert np.isclose(measured, expected, rtol=1e-6)


def test_edm_terimi_hizi():
    """EDM açıkken ideal saptırıcıda boyuna spin dikey yönde Ω = (q/m)(η/2)|E|/c ile döner.
    Halka ortalaması = bu değer × (saptırıcıların çevredeki payı)."""
    fp0 = fparams(); fp1 = fparams(edm=1.0)
    pos = np.array([R0, 0.0, 0.0]); p = np.array([0.0, -P_MAG, 0.0]); s = np.array([0.0, -1.0, 0.0])
    d0 = rhs(state(pos, p, s), fp0, 0); d1 = rhs(state(pos, p, s), fp1, 0)
    dSz = d1[8] - d0[8]
    expected = (Q_E / M_P) * 0.5 * EDM_ETA * abs(E0_MAGIC) / C_LIGHT
    frac = NQ * L_DEF / CIRC
    print(f"saptırıcı içinde dS_z/dt = {abs(dSz):.4e} rad/s, beklenen {expected:.4e}; "
          f"halka ortalaması ≈ {expected*frac:.4e} (makale 9.81e-10), saptırıcı payı {frac:.4f}")
    assert np.isclose(abs(dSz), expected, rtol=1e-9)


def test_spin_boyu_korunur():
    """dS/dt ⟂ S olmalı (|S| sabit): Ω × S yapısının sonucu."""
    rng = np.random.default_rng(3)
    for et, fp in ((0, fparams(roll=1e-3, edm=1.0)), (2, fparams(dx=1e-4, dy=-1e-4, qtilt=1e-3))):
        for _ in range(10):
            s = rng.normal(size=3); s /= np.linalg.norm(s)
            pos = np.array([R0 + 1e-3, 0.0, 1e-3]); p = np.array([1e-3 * P_MAG, -P_MAG, 0.0])
            dS = rhs(state(pos, p, s), fp, et)[6:9]
            assert abs(dS @ s) < 1e-12 * np.linalg.norm(dS) + 1e-30
