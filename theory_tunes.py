#!/usr/bin/env python3
import numpy as np
import json
import math
import os

def calculate_theoretical_tunes():
    # Load parameters
    with open("params.json") as f:
        cfg = json.load(f)

    R0 = cfg["R0"]
    nFODO = cfg["nFODO"]
    L_q = cfg["quadLen"]
    L_d = cfg["driftLen"]
    G1 = cfg["g1"]

    # Physical constants
    M2 = 0.938272046
    AMU = 1.792847356
    C_light = 299792458.0
    M1 = 1.672621777e-27

    p_magic = M2 / math.sqrt(AMU)
    E_tot = math.sqrt(p_magic**2 + M2**2)
    beta0 = p_magic / E_tot
    gamma0 = 1 / math.sqrt(1 - beta0**2)
    Brho = p_magic / 0.29979246

    K_q = G1 / Brho

    # Deflector lengths and focusing indices
    # Electric ring, n=1 -> K_x = (2 - beta0^2 - 1) / R0^2 = 1 / (gamma0^2 R0^2)
    L_def = (2 * math.pi * R0) / (2 * nFODO)
    K_x_def = (1 - beta0**2) / (R0**2)
    K_y_def = 0.0

    # Transfer matrix generators
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

    def calc_cell_matrix(M_arc, M_qf, M_qd):
        M = np.eye(2)
        M = drift_mat(L_d) @ M
        M = M_qd @ M
        M = drift_mat(L_d) @ M
        M = M_arc @ M
        M = drift_mat(L_d) @ M
        M = M_qf @ M
        M = drift_mat(L_d) @ M
        M = M_arc @ M
        return M

    # Calculate full cell matrices
    M_cell_x = calc_cell_matrix(def_x_mat(K_x_def, L_def), quad_F_mat(K_q, L_q), quad_D_mat(K_q, L_q))
    M_cell_y = calc_cell_matrix(drift_mat(L_def), quad_D_mat(K_q, L_q), quad_F_mat(K_q, L_q))

    # Calculate tunes from trace
    mu_x = math.acos(np.trace(M_cell_x) / 2.0)
    mu_y = math.acos(np.trace(M_cell_y) / 2.0)

    Q_x = nFODO * mu_x / (2 * math.pi)
    Q_y = nFODO * mu_y / (2 * math.pi)

    print("=== Teorik Transfer Matrisi ile Tune Hesabı ===")
    print(f"R0       : {R0} m")
    print(f"Brho     : {Brho:.5f} T*m")
    print(f"nFODO    : {nFODO}")
    print(f"Quad K   : {K_q:.5f} m^-2")
    print(f"Def K_x  : {K_x_def:.7f} m^-2")
    print("-----------------------------------------------")
    print(f"Teorik Q_x : {Q_x:.5f}")
    print(f"Teorik Q_y : {Q_y:.5f}")
    print("===============================================")

if __name__ == "__main__":
    calculate_theoretical_tunes()
