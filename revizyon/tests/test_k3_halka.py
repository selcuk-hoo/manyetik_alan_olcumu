"""Katman 3 (tam halka) — YAVAŞ. Spin dinamiğinin halka ölçeğinde analitik kontrolleri.

1. EDM'nin geri bulunması: kaçıklık yokken EDM açık → dS_y/dt = (q/m)(η/2)|E0|/c × (saptırıcı payı).
2. Düzgün roll: bütün saptırıcılarda aynı φ → birinci mertebe sahte EDM
       |f| = Gγ · 2πφ / T_dev     (Σθ_E = 2πφ; kapalı yörünge Σθ_B = −Σθ_E ister;
                                   spin manyetik tekmede (1+Gγ), elektrikte 1 kat döner)
   ve CW ile CCW'de aynı işaret.
Çalıştırma: python3 -m pytest -q -s -m slow test_k3_halka.py
"""
import os, sys, json, tempfile
import numpy as np
import pytest
from concurrent.futures import ProcessPoolExecutor
from ortak import BASE, REV, NQ, Q_E, M_P, C_LIGHT, E0_MAGIC, G_P, GAMMA0, T_REV, L_DEF, CIRC

for p in (REV, BASE, os.path.join(BASE, "berry_data"), os.path.join(BASE, "kmod_drivers")):
    sys.path.insert(0, p)
pytestmark = pytest.mark.slow
EDM_ETA = 1.88e-15


def _measure(task):
    """Tam halka sahte EDM ölçümü (fast_measure ile aynı reçete, n_turns=28, n_iter=4);
    edm=True ise gerçek EDM terimi açılır."""
    os.chdir(tempfile.mkdtemp())
    import integrator_rev
    sys.modules["integrator"] = integrator_rev
    direction, roll, edm = task
    integrator_rev.DEFL_ROLL = np.full(NQ, roll) if roll else None
    integrator_rev.DEFL_DE = None
    from false_edm_mode_scan import setup_fields, _make_state, measure_dSy_dt_model
    from false_edm_4d import find_co_4d, _T_rev
    cfg = json.load(open(os.path.join(BASE, "params.json"))); cfg["direction"] = float(direction)
    fields, y0, beta0, R0, p_mag, d = setup_fields(cfg)
    fields.EDMSwitch = 1.0 if edm else 0.0
    z = np.zeros(NQ)
    v_co, resid = find_co_4d(fields, p_mag, d, z, z, z, _T_rev(fields, beta0, R0), n_turns=28, n_iter=4)
    launch = _make_state(v_co, p_mag, d, [0.0, 0.0, d])
    fields.poincare_quad_index = 0.0
    _, poin, pt = integrator_rev.integrate_particle(launch, 0.0, 3e-4, float(cfg["dt"]), fields=fields,
                                                    return_steps=4000, quad_dx=z, quad_dy=z)
    return float(measure_dSy_dt_model(np.asarray(poin[:, 7], float), np.asarray(pt, float)))


@pytest.fixture(scope="module")
def res():
    tasks = {"edm_ccw": (-1, 0.0, True), "edm_cw": (+1, 0.0, True),
             "roll_ccw": (-1, 1e-8, False), "roll_cw": (+1, 1e-8, False)}
    with ProcessPoolExecutor(4) as pool:
        out = dict(zip(tasks, pool.map(_measure, tasks.values())))
    json.dump(out, open(os.path.join(REV, "sonuclar", "test_k3_halka.json"), "w"), indent=1)
    return out


def test_edm_geri_bulunur(res):
    frac = NQ * L_DEF / CIRC
    expected = (Q_E / M_P) * 0.5 * EDM_ETA * abs(E0_MAGIC) / C_LIGHT * frac
    for k in ("edm_ccw", "edm_cw"):
        print(f"{k}: dS_y/dt = {res[k]:.5e} rad/s, analitik ±{expected:.5e}")
        assert np.isclose(abs(res[k]), expected, rtol=0.01)
    # gerçek EDM demet yönüyle işaret değiştirmeli
    assert np.sign(res["edm_ccw"]) == -np.sign(res["edm_cw"])


def test_duzgun_roll_birinci_mertebe(res):
    phi = 1e-8
    expected = G_P * GAMMA0 * 2 * np.pi * phi / T_REV
    for k in ("roll_ccw", "roll_cw"):
        print(f"{k}: |f| = {abs(res[k]):.6e} rad/s, analitik {expected:.6e}")
        assert np.isclose(abs(res[k]), expected, rtol=0.01)
    assert np.sign(res["roll_ccw"]) == np.sign(res["roll_cw"])     # çift (yönden bağımsız)
