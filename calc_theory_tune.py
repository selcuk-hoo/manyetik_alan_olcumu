import numpy as np
import json
import math

with open("params.json") as f:
    cfg = json.load(f)

R0 = cfg["R0"]
nFODO = cfg["nFODO"]
L_q = cfg["quadLen"]
L_d = cfg["driftLen"]
G1 = cfg["g1"]

M2 = 0.938272046
AMU = 1.792847356
C_light = 299792458.0
M1 = 1.672621777e-27

p_magic = M2 / math.sqrt(AMU)
E_tot = math.sqrt(p_magic**2 + M2**2)
beta0 = p_magic / E_tot
gamma0 = 1 / math.sqrt(1 - beta0**2)
p_mag = gamma0 * M1 * C_light * beta0
Brho = p_magic / 0.29979246 # T*m

K_q = G1 / Brho # Quadrupole strength k [m^-2]

L_def = (2 * math.pi * R0) / (2 * nFODO)
K_x_def = (2 - beta0**2) / (R0**2)
K_y_def = 0.0

def drift_mat(L):
    return np.array([[1, L], [0, 1]])

def quad_F_mat(K, L):
    sqrtK = math.sqrt(abs(K))
    return np.array([
        [math.cos(sqrtK * L), (1/sqrtK)*math.sin(sqrtK * L)],
        [-sqrtK*math.sin(sqrtK * L), math.cos(sqrtK * L)]
    ])

def quad_D_mat(K, L):
    sqrtK = math.sqrt(abs(K))
    return np.array([
        [math.cosh(sqrtK * L), (1/sqrtK)*math.sinh(sqrtK * L)],
        [sqrtK*math.sinh(sqrtK * L), math.cosh(sqrtK * L)]
    ])

def def_x_mat(K, L):
    sqrtK = math.sqrt(K)
    return np.array([
        [math.cos(sqrtK * L), (1/sqrtK)*math.sin(sqrtK * L)],
        [-sqrtK*math.sin(sqrtK * L), math.cos(sqrtK * L)]
    ])

def def_y_mat(K, L):
    return np.array([[1, L], [0, 1]])

M_drift = drift_mat(L_d)
M_qf_x = quad_F_mat(K_q, L_q)
M_qf_y = quad_D_mat(K_q, L_q)

M_qd_x = quad_D_mat(K_q, L_q)
M_qd_y = quad_F_mat(K_q, L_q)

M_def_x = def_x_mat(K_x_def, L_def)
M_def_y = def_y_mat(K_y_def, L_def)

# FODO Cell: ARC1 -> DRIFT -> QF -> DRIFT -> ARC2 -> DRIFT -> QD -> DRIFT
def calc_cell_matrix(M_arc, M_qf, M_qd):
    M = np.eye(2)
    M = M_drift @ M
    M = M_qd @ M
    M = M_drift @ M
    M = M_arc @ M
    M = M_drift @ M
    M = M_qf @ M
    M = M_drift @ M
    M = M_arc @ M
    return M

M_cell_x = calc_cell_matrix(M_def_x, M_qf_x, M_qd_x)
M_cell_y = calc_cell_matrix(M_def_y, M_qf_y, M_qd_y)

tr_x = np.trace(M_cell_x)
tr_y = np.trace(M_cell_y)

mu_x = math.acos(tr_x / 2.0)
mu_y = math.acos(tr_y / 2.0)

Q_x = nFODO * mu_x / (2 * math.pi)
Q_y = nFODO * mu_y / (2 * math.pi)

print(f"Brho: {Brho:.5f}")
print(f"K_q: {K_q:.5f}")
print(f"beta0: {beta0:.5f}")
print(f"Theoretical Q_x: {Q_x:.5f}")
print(f"Theoretical Q_y: {Q_y:.5f}")
