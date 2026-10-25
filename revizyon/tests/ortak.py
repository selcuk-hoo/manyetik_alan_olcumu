"""ortak.py — doğrulama testleri için ortak yardımcılar.

C++ kütüphanesindeki alan (get_electromagnetic_fields), sağ taraf (compute_rhs) ve tek adım
(gl4_step_element) fonksiyonlarına ctypes ile doğrudan erişim sağlar. Böylece fizik bileşenleri
tam halka simülasyonu gerekmeden tek tek test edilebilir.

field_params dizisi (34 eleman; revizyon motorunun tam düzeni):
  0 R0, 1 E0, 2 n (E0_power), 3 B0ver, 4 B0rad, 5 B0long, 6 quadG1, 7 sextK1, 8 quadSwitch,
  9 sextSwitch, 10 EDMSwitch, 11 direction, 12 nFODO, 13 quadLen, 14 poincare, 15 rfSwitch,
  16 rfVoltage, 17 h, 18 driftLen, 19 quadModA, 20 quadModF, 21 Br_harm_amp, 22 Br_harm_N,
  23 quad dy, 24 quadG0, 25 quad dx, 26 dipole_tilt, 27 quad_tilt, 28 quad_dG, 29 θ_e,
  30 defl ε, 31 defl roll, 32 defl B_v, 33 defl B_r
Eleman tipleri: 0 saptırıcı, 1 sürüklenme, 2 QF, 3 QD, 4 QF_MOD (hücre 0).
"""
import ctypes, json, os
import numpy as np

TESTS = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(TESTS)
BASE = os.path.dirname(REV)
CFG = json.load(open(os.path.join(BASE, "params.json")))

# fiziksel sabitler (integrator.cpp ile AYNI değerler)
C_LIGHT = 299792458.0
M_P = 1.672621777e-27
Q_E = 1.602176565e-19
G_P = 1.792847356
M_GEV = 0.938272046
P_MAGIC_GEV = M_GEV / np.sqrt(G_P)
E_MAGIC_GEV = np.sqrt(P_MAGIC_GEV ** 2 + M_GEV ** 2)
BETA0 = P_MAGIC_GEV / E_MAGIC_GEV
GAMMA0 = 1.0 / np.sqrt(1.0 - BETA0 ** 2)
P_MAG = GAMMA0 * M_P * C_LIGHT * BETA0          # [kg·m/s]
R0 = CFG["R0"]
E0_MAGIC = -(P_MAGIC_GEV * BETA0 / R0) * 1e9    # build_response_matrix.setup_fields ile aynı [V/m]
NQ = 2 * int(CFG["nFODO"])
L_DEF = 2 * np.pi * R0 / NQ
CIRC = NQ * L_DEF + 2 * NQ * CFG["driftLen"] + NQ * CFG["quadLen"]
T_REV = CIRC / (BETA0 * C_LIGHT)


def load_lib(which="rev"):
    path = (os.path.join(REV, "lib_integrator_rev.so") if which == "rev"
            else os.path.join(BASE, "lib_integrator.so"))
    lib = ctypes.CDLL(path)
    dp = ctypes.POINTER(ctypes.c_double)
    lib.get_electromagnetic_fields.argtypes = [ctypes.c_double, dp, dp, ctypes.c_int, dp, dp]
    lib.compute_rhs.argtypes = [ctypes.c_double, dp, dp, ctypes.c_int, dp, ctypes.c_int]
    lib.gl4_step_element.argtypes = [ctypes.c_double, dp, dp, ctypes.c_int, ctypes.c_double, ctypes.c_int]
    return lib


def fparams(**kw):
    """Varsayılan (params.json ile tutarlı) 34 elemanlı field_params; kw ile indeks üzerine yaz."""
    fp = np.zeros(34)
    fp[0] = R0; fp[1] = E0_MAGIC; fp[2] = CFG.get("E0_power", 1.0)
    fp[6] = CFG["g1"]; fp[8] = 1.0; fp[11] = -1.0; fp[12] = CFG["nFODO"]
    fp[13] = CFG["quadLen"]; fp[18] = CFG["driftLen"]; fp[24] = CFG.get("g0", CFG["g1"])
    names = {"R0": 0, "E0": 1, "n": 2, "B0ver": 3, "B0rad": 4, "B0long": 5, "G": 6, "K2": 7,
             "sext": 9, "edm": 10, "dir": 11, "dy": 23, "G0": 24, "dx": 25, "dtilt": 26,
             "qtilt": 27, "dG": 28, "eps": 30, "roll": 31, "Bv": 32, "Br": 33}
    for k, v in kw.items():
        fp[names[k]] = v
    return fp


def _arr(a):
    a = np.ascontiguousarray(a, dtype=float)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def fields(lib, r, fp, etype, t=0.0):
    r_, rp = _arr(r); f_, fpp = _arr(fp)
    E = np.zeros(3); B = np.zeros(3)
    _, Ep = _arr(E); _, Bp = _arr(B)
    Eb = (ctypes.c_double * 3)(); Bb = (ctypes.c_double * 3)()
    lib.get_electromagnetic_fields(t, rp, fpp, etype, Eb, Bb)
    return np.array(Eb[:]), np.array(Bb[:])


def step(lib, y, fp, etype, h, n=1, t=0.0):
    """GL4 ile n adım ilerlet; y (9) yerinde değil kopya olarak döner."""
    y = np.ascontiguousarray(y, dtype=float).copy()
    f_, fpp = _arr(fp)
    yb = (ctypes.c_double * 9)(*y)
    for i in range(n):
        lib.gl4_step_element(t + i * h, yb, fpp, etype, h, 9)
    return np.array(yb[:])


def div_curl(lib, fp, etype, r, h=1e-5, which="E"):
    """Merkezi farklarla ∇·F ve ∇×F (F = E ya da B), global Kartezyen koordinatlarda."""
    J = np.zeros((3, 3))                       # J[i, j] = ∂F_i/∂x_j
    for j in range(3):
        d = np.zeros(3); d[j] = h
        Fp = fields(lib, r + d, fp, etype)[0 if which == "E" else 1]
        Fm = fields(lib, r - d, fp, etype)[0 if which == "E" else 1]
        J[:, j] = (Fp - Fm) / (2 * h)
    div = np.trace(J)
    curl = np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])
    return div, curl, J
