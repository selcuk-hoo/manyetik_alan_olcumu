#!/usr/bin/env python3
"""make_fig2.py — Figure 2: Fourier mode orbit gain (Scatter Plot)"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_plot_utils import apply_paper_style, load_R, RF_unit_norm, M_col_norm, BLUE, RED, GRAY

def make_fig2():
    apply_paper_style()
    R = load_R()
    
    k_list = list(range(1, 13))
    rf_norms = [RF_unit_norm(R, k) for k in k_list]
    m_norms  = [M_col_norm(R, k)   for k in k_list]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    xpos = np.array(k_list)
    
    # Plot all points
    ax.scatter(xpos, rf_norms, color=BLUE, s=60, zorder=3)
    
    # Optional: connect with a faint line to guide the eye
    ax.plot(xpos, rf_norms, color=GRAY, linestyle=':', alpha=0.5, zorder=2)
               
    ax.set_xticks(xpos)
    ax.set_xlabel("Fourier mode $k$")
    ax.set_ylabel(r"$\|RF_k\|$ (unit-normalized $F_k$)")
    ax.set_title(r"Fourier mode orbit gain  ($Q_y \approx 2.68$)")
    ax.legend(frameon=False)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig2_orbit_gain.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"fig2_orbit_gain.png generated at {out_path}")

if __name__ == "__main__":
    make_fig2()
