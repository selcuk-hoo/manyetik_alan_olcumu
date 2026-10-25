"""Katman 2 — doğrusal optik ve kapalı yörünge (yalnız yörünge, spin yok). YAVAŞ: C++ izleme.

Kritik soru: Tepki matrisleri R (R_d?_1.npy, R_d?_cw.npy) eksenden fırlatılan bir parçacığın
t2 = 1e-4 s (~22 tur) boyunca BPM konumlarındaki ORTALAMASIYLA üretildi. Parçacık kapalı yörünge
etrafında betatron salınımı yaptığı için bu ortalama, kapalı yörüngeye ancak yaklaşık eşittir.
Burada aynı tek-kuadrupol tepkisi üç yolla hesaplanır:
  (a) R üretim yöntemi:        eksenden fırlatma, t2 = 1e-4 s ortalaması
  (b) uzun ortalama:            eksenden fırlatma, t2 = 1e-3 s
  (c) GERÇEK kapalı yörünge:    4D kapalı yörünge bulucu (find_co_4d) → yörünge üzerinden fırlatma
ve (a)'nın (c)'den bağıl farkı raporlanır. Ayrıca doğrusallık ve süperpozisyon sınanır.
Çalıştırma: python3 -m pytest -q -s -m slow test_k2_yorunge.py
"""
import os, sys, json, tempfile
import numpy as np
import pytest
from concurrent.futures import ProcessPoolExecutor

from ortak import BASE, REV, NQ
for p in (BASE, os.path.join(BASE, "berry_data"), os.path.join(BASE, "kmod_drivers")):
    sys.path.insert(0, p)

pytestmark = pytest.mark.slow
QI = 7                      # test kuadrupolü (QD, hücre 3)
DQ = 1e-4                   # R üretimindeki pertürbasyonla aynı


def _orbit(task):
    """task = (mode, plane, amp, t2, direction, extra) → (x_BPM, y_BPM) [m]"""
    os.chdir(tempfile.mkdtemp())
    mode, plane, amp, t2, direction, extra = task
    import build_response_matrix as brm
    cfg = json.load(open(os.path.join(BASE, "params.json")))
    cfg["direction"] = float(direction); cfg["t2"] = t2
    dx = np.zeros(NQ); dy = np.zeros(NQ)
    for qi, a in ([(QI, amp)] + list(extra)):
        (dy if plane == "y" else dx)[qi] += a
    if mode == "axis":
        al, st = brm.setup_fields(cfg)
        return brm.run_sim(al, st, cfg, dy, dx)
    # gerçek kapalı yörünge
    from false_edm_mode_scan import setup_fields, _make_state
    from false_edm_4d import find_co_4d, _T_rev
    from integrator import integrate_particle
    fields, y0, beta0, R0, p_mag, d = setup_fields(cfg)
    v_co, resid = find_co_4d(fields, p_mag, d, dx, dy, np.zeros(NQ), _T_rev(fields, beta0, R0),
                             n_turns=28, n_iter=4)
    launch = _make_state(v_co, p_mag, d, [0.0, 0.0, d])
    fields.poincare_quad_index = 999.0
    for f in ("cod_data.txt", "rf.txt"):
        if os.path.exists(f):
            os.remove(f)
    integrate_particle(launch, 0.0, t2, float(cfg["dt"]), fields=fields, return_steps=10,
                       quad_dy=dy, quad_dx=dx)
    return brm.read_cod_quads(NQ // 2)


@pytest.fixture(scope="module")
def runs():
    tasks = {
        "ref_axis": ("axis", "y", 0.0, 1e-4, -1, []),
        "y_axis_1e-4": ("axis", "y", DQ, 1e-4, -1, []),
        "y_axis_1e-3": ("axis", "y", DQ, 1e-3, -1, []),
        "y_co": ("co", "y", DQ, 1e-4, -1, []),
        "x_axis_1e-4": ("axis", "x", DQ, 1e-4, -1, []),
        "x_co": ("co", "x", DQ, 1e-4, -1, []),
        "y_axis_lin10": ("axis", "y", 10 * DQ, 1e-4, -1, []),
        "y_axis_lin01": ("axis", "y", 0.1 * DQ, 1e-4, -1, []),
        "y_axis_super": ("axis", "y", DQ, 1e-4, -1, [(20, DQ)]),
        "y_axis_q20": ("axis", "y", 0.0, 1e-4, -1, [(20, DQ)]),
    }
    keys = list(tasks)
    with ProcessPoolExecutor(4) as pool:
        res = dict(zip(keys, pool.map(_orbit, [tasks[k] for k in keys])))
    return res


def _col(res, key, plane):
    i = 1 if plane == "y" else 0
    return (np.asarray(res[key][i]) - np.asarray(res["ref_axis"][i])) / DQ


def test_R_matrisi_kayitli_ile_ayni(runs):
    """Kayıtlı R_dy_1.npy'nin QI sütunu, aynı yöntemle yeniden hesaplanan sütunla uyuşmalı."""
    R = np.load(os.path.join(BASE, "R_dy_1.npy"))
    col = _col(runs, "y_axis_1e-4", "y")
    rel = np.linalg.norm(col - R[:, QI]) / np.linalg.norm(R[:, QI])
    print(f"kayıtlı R sütunu ile fark (bağıl): {rel:.2e}")
    assert rel < 1e-6


def test_tur_ortalamasi_raporu(runs):
    """R üretim yöntemi (eksenden fırlatma, 22 tur ortalaması) ile gerçek kapalı yörünge arasındaki
    bağıl fark — bir 'model hatası' kaynağı. Büyüklüğü raporlanır; %1'in altında beklenir."""
    out = {}
    for plane in ("y", "x"):
        a = _col(runs, f"{plane}_axis_1e-4", plane)
        i = 1 if plane == "y" else 0
        co = np.asarray(runs[f"{plane}_co"][i]) / DQ          # gerçek CO (referans ideal = 0)
        rel = float(np.linalg.norm(a - co) / np.linalg.norm(co))
        out[plane] = rel
        print(f"{plane}: |R(22 tur ortalaması) − R(gerçek CO)| / |R(CO)| = {rel:.3e}")
    if "y_axis_1e-3" in runs:
        b = _col(runs, "y_axis_1e-3", "y")
        co = np.asarray(runs["y_co"][1]) / DQ
        print(f"y: t2=1e-3 (220 tur) ortalaması ile fark = {np.linalg.norm(b-co)/np.linalg.norm(co):.3e}")
    json.dump(out, open(os.path.join(REV, "sonuclar", "test_k2_tur_ortalamasi.json"), "w"), indent=1)
    assert max(out.values()) < 0.01, out


def test_dogrusallik(runs):
    """Yörünge kaçıklıkla doğrusal olmalı: 0.1×, 1×, 10× pertürbasyon aynı R sütununu vermeli."""
    c1 = _col(runs, "y_axis_1e-4", "y")
    for k, s in (("y_axis_lin10", 10), ("y_axis_lin01", 0.1)):
        c = _col(runs, k, "y") / s
        rel = np.linalg.norm(c - c1) / np.linalg.norm(c1)
        print(f"ölçek {s}: bağıl fark {rel:.2e}")
        assert rel < 1e-3


def test_superpozisyon(runs):
    """İki kuadrupolün birlikte kaydırılması = ayrı ayrı tepkilerin toplamı."""
    both = _col(runs, "y_axis_super", "y")
    s = _col(runs, "y_axis_1e-4", "y") + _col(runs, "y_axis_q20", "y")
    rel = np.linalg.norm(both - s) / np.linalg.norm(s)
    print(f"süperpozisyon bağıl fark {rel:.2e}")
    assert rel < 1e-3
