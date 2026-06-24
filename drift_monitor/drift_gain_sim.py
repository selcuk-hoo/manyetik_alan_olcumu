#!/usr/bin/env python3
"""
drift_gain_sim.py — BPM kazanç (gain) hatalarının drift monitöre etkisi.

Gerçek BPM'ler $y_{\\rm ölç,i} = (1+g_i)\\,y_{\\rm gerçek,i} + \\text{ofset}_i +
\\text{gürültü}$ okur; $g_i$ kalibrasyon-kaynaklı per-BPM kazanç hatasıdır
(tipik %1-5). Bu, drift kurtarımını nasıl etkiler?

Drift modu zaman farkı aldığından sabit ofset $(1+g_i)b_{0,i}$ yine iptal olur
(gain'den bağımsız). Geriye kalan etki çarpımsaldır:
  δq̂ = R⁻¹·diag(1+g)·R·δq = δq + R⁻¹·diag(g)·R·δq,
yani hata terimi R⁻¹·diag(g)·R·(drift). Bu tamamen analitik (izleyici gerekmez);
R analitik Courant-Snyder matrisidir (fodo_lattice).

Çıktı: takip hatası vs gain RMS tablosu + fig9_bpm_gain.png.
Kullanım: python3 drift_gain_sim.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
import fodo_lattice as fl

COL = 3.375
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "legend.frameon": False, "lines.linewidth": 1.1,
                     "lines.markersize": 3, "savefig.dpi": 600,
                     "savefig.bbox": "tight"})

OFF, NOISE, RAMP, DQ0, NEP, NSEED = 50e-6, 1e-6, 10e-6, 100e-6, 10, 30


def analytic_R(cfg, plane="y"):
    Kx = fl.calibrate_K_x_arc(cfg) if plane == "x" else None
    beta, phi, Q = fl.compute_twiss_at_quads(cfg, plane, K_x_arc=Kx)
    return fl.build_response_matrix(beta, phi, Q, fl.signed_KL(cfg, plane))


def track_err(R, Rinv, sigma_g, seed):
    """BPM gain RMS sigma_g altında drift takip hatası [m]."""
    N = R.shape[0]
    rng = np.random.default_rng(3000 + seed)
    g = rng.normal(0, sigma_g, N) if sigma_g > 0 else np.zeros(N)   # per-BPM gain
    dq0 = rng.normal(0, DQ0, N)
    b0 = rng.normal(0, OFF, N)
    ramp = rng.normal(0, RAMP, N)

    def meas(dq):
        return (1.0 + g) * (R @ dq + b0) + rng.normal(0, NOISE, N)
    y0 = meas(dq0)
    errs = []
    for t in range(1, NEP + 1):
        dqt = dq0 + ramp * (t / NEP)
        dqhat = Rinv @ (meas(dqt) - y0)
        errs.append(np.sqrt(np.mean((dqhat - (dqt - dq0)) ** 2)))
    return np.mean(errs)


def main():
    os.chdir(os.path.join(_DIR, ".."))
    with open("params.json") as f:
        cfg = json.load(f)
    R = analytic_R(cfg, "y"); Rinv = np.linalg.inv(R)

    sig_g = np.array([0.0, 0.01, 0.02, 0.05, 0.10])
    print(f"{'gain RMS':>9} {'y-takip[μm]':>12}")
    rows = []
    for sg in sig_g:
        e = np.median([track_err(R, Rinv, sg, s) for s in range(NSEED)]) * 1e6
        rows.append((sg, e))
        print(f"{sg*100:>7.0f}% {e:>12.2f}")
    rows = np.array(rows)
    np.save(os.path.join(_DIR, "bpm_gain.npy"), rows)

    fig, ax = plt.subplots(figsize=(COL, 2.6))
    ax.plot(rows[:, 0] * 100, rows[:, 1], "o-", color="C0")
    ax.axhline(10, color="C3", ls="--", lw=0.9, label=r"hedef 10$\,\mu$m")
    ax.axvline(2, color="gray", ls=":", lw=0.8, label="tipik BPM (\\%2)" if False
               else "tipik BPM %2")
    ax.set_xlabel(r"BPM kazanç hatası RMS  $\sigma_g$  [%]")
    ax.set_ylabel(r"drift takip hatası [$\mu$m]")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(_DIR, "fig9_bpm_gain.png"))
    print("Kaydedildi: drift_monitor/fig9_bpm_gain.png")


if __name__ == "__main__":
    main()
