import os
import numpy as np
import matplotlib.pyplot as plt
import base64
import gzip
import io

# ── Style Configurations ───────────────────────────────────────────────────
def apply_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

BLUE   = "#2166ac"
RED    = "#d6604d"
ORANGE = "#f4a742"
GRAY   = "#888888"
GREEN  = "#4dac26"
PURPLE = "#762a83"

# ── Fourier Basis (Unnormalized) ───────────────────────────────────────────
def Fcos(k, n_q=48):
    """FODO-antisymmetric cosine mode (unnormalized)."""
    j = np.arange(n_q)
    if k == 0:
        return (-1.0) ** j
    return (-1.0) ** j * np.cos(2 * np.pi * k * (j // 2) / (n_q // 2))

def Fsin(k, n_q=48):
    """FODO-antisymmetric sine mode (unnormalized)."""
    j = np.arange(n_q)
    return (-1.0) ** j * np.sin(2 * np.pi * k * (j // 2) / (n_q // 2))

def RF_unit_norm(R, k):
    """||RF_k|| with unit-normalized F_k (Table 2 values)."""
    Fc = Fcos(k, R.shape[0])
    Fc = Fc / np.linalg.norm(Fc)
    return np.linalg.norm(R @ Fc)

def M_col_norm(R, k):
    """||M_k|| = sqrt(N_cells) * ||RF_k||  ≈ 167 for k=2."""
    return np.sqrt(R.shape[0] // 2) * RF_unit_norm(R, k)

# ── Projection Estimator ───────────────────────────────────────────────────
def project_amplitude(y, R, k):
    """Estimate amplitude of harmonic k from orbit measurement y (metres)."""
    Mc = R @ Fcos(k, R.shape[0])
    Ms = R @ Fsin(k, R.shape[0])
    M2 = np.column_stack([Mc, Ms])
    a2, _, _, _ = np.linalg.lstsq(M2, y, rcond=None)
    return float(np.sqrt(a2[0]**2 + a2[1]**2)), float(np.arctan2(a2[1], a2[0]))

def make_orbit(R, k_true=2, A_true=10e-6, phi_true=0.3,
               contaminants=None, b_sigma=0.0, rng=None):
    """Generate test orbit with known harmonic pattern + BPM offset."""
    if rng is None:
        rng = np.random.default_rng(0)
    if contaminants is None:
        contaminants = {4: (300e-6, 0.7), 6: (300e-6, 1.2), 8: (200e-6, 2.1)}
    n_q = R.shape[0]
    dq = A_true * (np.cos(phi_true) * Fcos(k_true, n_q)
                   + np.sin(phi_true) * Fsin(k_true, n_q))
    for k_c, (A_c, phi_c) in contaminants.items():
        dq += A_c * (np.cos(phi_c) * Fcos(k_c, n_q)
                     + np.sin(phi_c) * Fsin(k_c, n_q))
    b = rng.normal(0, b_sigma, n_q) if b_sigma > 0 else np.zeros(n_q)
    return R @ dq + b

# ── Response Matrix Loader ─────────────────────────────────────────────────
_R_FALLBACK_B64 = (
    "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"
)

def load_R(base_dir=None):
    """Load R_dy_1.npy if present and valid; otherwise use the embedded reference."""
    if base_dir is None:
        base_dir = os.getcwd()
    try:
        file_path = os.path.join(base_dir, "R_dy_1.npy")
        if os.path.exists(file_path):
            Rmat = np.load(file_path)
            if np.max(np.abs(Rmat)) > 1e-10:
                print("R loaded from R_dy_1.npy")
                return Rmat
        raise ValueError("R_dy_1.npy missing or contains a zero matrix")
    except Exception as exc:
        print(f"  Note: {exc}")
        print("  Using embedded reference R")
        raw = gzip.decompress(base64.b64decode(_R_FALLBACK_B64))
        return np.load(io.BytesIO(raw))
