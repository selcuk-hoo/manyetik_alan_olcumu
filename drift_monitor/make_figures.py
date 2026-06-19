#!/usr/bin/env python3
"""
make_figures.py — drift_makalesi.md için tüm şekilleri üretir.

Şekiller (makale Şekil Listesi ile uyumlu):
  Şekil 1: R ve ΔR singüler-değer spektrumları (ΔR ≈ ε R kayması)        — §2.4/§3.2
  Şekil 2: Drift izleme zaman serisi (gerçek vs kestirilen + mutlak)      — §3.4
  Şekil 3: β-beating tarama eğrisi (tracking hatası vs ε_β)              — §3.7
  Şekil 4: Per-mod SVD (mod indeksi → 1/σ ve simetrik güç, çift eksen)   — §4.3

Şekil 5 (iki-katmanlı mimari şeması) ayrı: make_fig5_architecture.py.

Tüm hesaplar drift_monitor/fodo_lattice.py üzerinden; C++ bağımlılığı yok.
Parametreler ../params.json ve test_params.json'dan okunur.

Kullanım:
    python3 make_figures.py            # 4 şekli de üretir
    python3 make_figures.py 1 3        # yalnız Şekil 1 ve 3
"""

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
import fodo_lattice as fl

# --------------------------------------------------------------------------
# Ortak yapılandırma ve yardımcılar
# --------------------------------------------------------------------------
with open(os.path.join(_DIR, "..", "params.json")) as _f:
    CFG = json.load(_f)
with open(os.path.join(_DIR, "test_params.json")) as _f:
    TP = json.load(_f)

EPS = float(TP.get("EPS", 0.02))          # iki-gradient ayrımı g2=g1(1+ε)
OUTDIR = _DIR                              # PNG'ler buraya yazılır
DPI = 150


def build_R(config, g, plane, K_x_arc=None):
    """Verilen gradient g ve düzlem için Courant-Snyder tepki matrisi."""
    cfg = dict(config)
    cfg["g1"] = g
    beta, phi, Q = fl.compute_twiss_at_quads(cfg, plane, K_x_arc=K_x_arc)
    KL = fl.signed_KL(cfg, plane)
    return fl.build_response_matrix(beta, phi, Q, KL)


def sym_frac(v):
    """Bir vektörün simetrik (hücre içi QF/QD aynı yön) alt-uzaydaki güç oranı."""
    N = len(v)
    s = v.copy()
    for c in range(N // 2):
        m = 0.5 * (v[2 * c] + v[2 * c + 1])
        s[2 * c] = m
        s[2 * c + 1] = m
    return np.sum(s ** 2) / np.sum(v ** 2)


def _save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  kaydedildi: {os.path.relpath(path)}")


# --------------------------------------------------------------------------
# Şekil 1 — R ve ΔR singüler-değer spektrumları
# --------------------------------------------------------------------------
def fig1_svd_spectra():
    """ΔR = R(g1) − R(g1(1+ε)) ≈ ε R ölçeklemesini ve κ farkını gösterir."""
    print("Şekil 1: SVD spektrumları (R vs ΔR)")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, plane in zip(axes, ("y", "x")):
        Kxarc = fl.calibrate_K_x_arc(CFG) if plane == "x" else None
        R1 = build_R(CFG, CFG["g1"], plane, Kxarc)
        R2 = build_R(CFG, CFG["g1"] * (1.0 + EPS), plane, Kxarc)
        dR = R1 - R2

        sR = np.linalg.svd(R1, compute_uv=False)
        sdR = np.linalg.svd(dR, compute_uv=False)
        idx = np.arange(1, len(sR) + 1)

        ax.semilogy(idx, sR, "o-", ms=4, color="C0", label=r"$R$")
        ax.semilogy(idx, sdR, "s-", ms=4, color="C3", label=r"$\Delta R = R_1-R_2$")
        # ε·R referans çizgisi (beklenen kayma)
        ax.semilogy(idx, EPS * sR, "--", color="gray", lw=1,
                    label=r"$\varepsilon\,\sigma(R)$ (beklenen)")

        kR = sR[0] / sR[-1]
        kdR = sdR[0] / sdR[-1]
        ax.set_title(f"Düzlem {plane}:  "
                     rf"$\kappa(R)\approx{kR:.0f}$,  "
                     rf"$\kappa(\Delta R)\approx{kdR:.0f}$")
        ax.set_xlabel("Singüler değer indeksi")
        ax.set_ylabel(r"$\sigma_i$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(r"Şekil 1 — Bulk singüler değerler $\Delta R \approx \varepsilon R$ "
                 rf"($\varepsilon={EPS}$) izler; en küçük modlar daha da çöker "
                 r"$\Rightarrow \kappa(\Delta R)\gg\kappa(R)$",
                 fontsize=10.5)
    _save(fig, "fig1_svd_spektrum.png")


# --------------------------------------------------------------------------
# Şekil 2 — Drift izleme zaman serisi
# --------------------------------------------------------------------------
def fig2_drift_tracking():
    """Test 4: gerçek vs kestirilen drift RMS ve mutlak rek. hatası."""
    print("Şekil 2: drift izleme zaman serisi (Test 4)")
    t4 = TP["test4"]
    DQ0 = float(t4.get("DQ0_RMS", 100e-6))
    OFF = float(t4["BPM_OFFSET"])
    NOISE = float(t4["BPM_NOISE"])
    RAMP = float(t4["DRIFT_RAMP"])
    NEP = int(t4["N_EPOCHS"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rng = np.random.default_rng(2026)

    for col, plane in enumerate(("y", "x")):
        Kxarc = fl.calibrate_K_x_arc(CFG) if plane == "x" else None
        R = build_R(CFG, CFG["g1"], plane, Kxarc)
        N = R.shape[0]
        Rinv = np.linalg.inv(R)

        dq0 = rng.normal(0, DQ0, N)
        b0 = rng.normal(0, OFF, N)
        ramp = rng.normal(0, RAMP, N)
        y0 = R @ dq0 + b0 + rng.normal(0, NOISE, N)

        times = np.arange(0, NEP + 1)
        true_rms = np.zeros_like(times, dtype=float)
        hat_rms = np.zeros_like(times, dtype=float)
        track_err = np.zeros_like(times, dtype=float)
        abs_err = np.zeros_like(times, dtype=float)

        for t in times:
            dqt = dq0 + ramp * (t / NEP)
            d_true = dqt - dq0
            yt = R @ dqt + b0 + rng.normal(0, NOISE, N)
            d_hat = Rinv @ (yt - y0)
            dq_abs = Rinv @ yt
            true_rms[t] = np.sqrt(np.mean(d_true ** 2))
            hat_rms[t] = np.sqrt(np.mean(d_hat ** 2))
            track_err[t] = np.sqrt(np.mean((d_hat - d_true) ** 2))
            abs_err[t] = np.sqrt(np.mean((dq_abs - dqt) ** 2))

        ax = axes[col]
        ax.plot(times, true_rms * 1e6, "o-", color="C0", label="Gerçek drift")
        ax.plot(times, hat_rms * 1e6, "s--", color="C1", ms=5,
                label="Kestirilen drift")
        ax.plot(times, abs_err * 1e6, "^:", color="C3",
                label="Mutlak rek. hatası")
        ax.axhline(OFF * 1e6, color="gray", ls="--", lw=1,
                   label=rf"BPM ofseti = {OFF*1e6:.0f} μm")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMS [μm]")
        ax.set_title(f"Düzlem {plane}:  drift takip hatası "
                     f"≈ {np.mean(track_err[1:])*1e6:.1f} μm")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Şekil 2 — Kalibrasyon-referans drift izleme: "
                 "ofset mutlak rek.'u boğar, drift modu izler", fontsize=11)
    _save(fig, "fig2_drift_izleme.png")


# --------------------------------------------------------------------------
# Şekil 3 — β-beating tarama eğrisi
# --------------------------------------------------------------------------
def fig3_betabeat():
    """Test 8: tracking hatası vs örgü-modeli β-beating seviyesi."""
    print("Şekil 3: β-beating tarama eğrisi (Test 8)")
    OFF, NOISE, RAMP, DQ0, NEP, NSEED = 50e-6, 1e-6, 10e-6, 100e-6, 10, 15

    def nominal(plane):
        Kxarc = fl.calibrate_K_x_arc(CFG) if plane == "x" else None
        beta, phi, Q = fl.compute_twiss_at_quads(CFG, plane, K_x_arc=Kxarc)
        KL = fl.signed_KL(CFG, plane)
        return beta, phi, Q, KL

    def track_err(plane, eps_beta, seed):
        beta, phi, Q, KL = nominal(plane)
        N = len(beta)
        Rm = fl.build_response_matrix(beta, phi, Q, KL)
        Rm_inv = np.linalg.inv(Rm)
        rb = np.random.default_rng(50000 + seed)
        bt = beta * (1 + rb.normal(0, eps_beta, N))
        pt = phi + rb.normal(0, eps_beta, N)
        Rt = fl.build_response_matrix(bt, pt, Q, KL)
        rng = np.random.default_rng(1000 + seed)
        dq0 = rng.normal(0, DQ0, N)
        b0 = rng.normal(0, OFF, N)
        ramp = rng.normal(0, RAMP, N)
        y0 = Rt @ dq0 + b0 + rng.normal(0, NOISE, N)
        errs = []
        for t in range(1, NEP + 1):
            dqt = dq0 + ramp * (t / NEP)
            yt = Rt @ dqt + b0 + rng.normal(0, NOISE, N)
            dqhat = Rm_inv @ (yt - y0)
            errs.append(np.sqrt(np.mean((dqhat - (dqt - dq0)) ** 2)))
        return np.mean(errs)

    eps_grid = np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.10])
    ey = np.array([np.median([track_err("y", e, s) for s in range(NSEED)])
                   for e in eps_grid])
    ex = np.array([np.median([track_err("x", e, s) for s in range(NSEED)])
                   for e in eps_grid])

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(eps_grid * 100, ey * 1e6, "o-", color="C0", label="Düzlem y")
    ax.plot(eps_grid * 100, ex * 1e6, "s-", color="C1", label="Düzlem x")
    ax.axhline(10, color="C3", ls="--", label="Hedef: 10 μm")
    ax.axvline(1.0, color="gray", ls=":", lw=1,
               label="LOCO-gerçekçi (%1)")
    ax.set_xlabel("β-beating seviyesi  $\\varepsilon_\\beta$  [%]")
    ax.set_ylabel("Drift takip hatası RMS [μm]")
    ax.set_title("Şekil 3 — Örgü-modeli hatası altında sağlamlık (Test 8)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    _save(fig, "fig3_betabeat.png")


# --------------------------------------------------------------------------
# Şekil 4 — Per-mod SVD: gürültü duyarlılığı ve simetrik karakter
# --------------------------------------------------------------------------
def fig4_permode():
    """§4.3: mod indeksine karşı 1/σ (sol) ve simetrik güç % (sağ)."""
    print("Şekil 4: per-mod SVD analizi")
    plane = "y"
    R = build_R(CFG, CFG["g1"], plane)
    U, S, Vt = np.linalg.svd(R)
    N = len(S)
    idx = np.arange(N)
    inv_sigma = 1.0 / S
    symp = np.array([sym_frac(Vt[i]) for i in range(N)]) * 100.0

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    color1 = "C0"
    ax1.semilogy(idx, inv_sigma, "o-", color=color1, ms=4)
    ax1.set_xlabel("Singüler mod indeksi (büyük σ → küçük σ)")
    ax1.set_ylabel(r"Gürültü duyarlılığı $1/\sigma_i$", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "C3"
    ax2.plot(idx, symp, "s--", color=color2, ms=4)
    ax2.set_ylabel("Simetrik alt-uzay gücü [%]", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 105)

    kappa = S[0] / S[-1]
    best = symp[:8].mean()
    worst = symp[-8:].mean()
    ax1.set_title(rf"Şekil 4 — Per-mod SVD (düzlem y): $\kappa={kappa:.0f}$; "
                  f"en iyi 8 mod %{best:.0f} sim., en kötü 8 mod %{worst:.0f} sim.")
    _save(fig, "fig4_permode_svd.png")


# --------------------------------------------------------------------------
# Şekil 6 — ε-taraması: gürültü büyütmesi vs kondisyon sayısı
# --------------------------------------------------------------------------
def fig6_epsilon_sweep():
    """ε taraması: ‖ΔR⁻¹‖ = 1/σ_min(ΔR) 1/ε ile ölçeklenir (asıl sonuç);
    κ(ΔR) ise ε'den kabaca bağımsızdır (κ(ΔR) ≈ κ(R'), R' = g·∂R/∂g)."""
    print("Şekil 6: ε-taraması (gürültü büyütmesi ∝ 1/ε; κ(ΔR) ~ sabit)")
    eps_grid = np.array([0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    for plane, c in (("y", "C0"), ("x", "C1")):
        Kxarc = fl.calibrate_K_x_arc(CFG) if plane == "x" else None
        R = build_R(CFG, CFG["g1"], plane, Kxarc)
        inv_smin = []   # ‖ΔR⁻¹‖ = 1/σ_min(ΔR)
        kappa_dR = []   # κ(ΔR)
        for eps in eps_grid:
            dR = R - build_R(CFG, CFG["g1"] * (1.0 + eps), plane, Kxarc)
            s = np.linalg.svd(dR, compute_uv=False)
            inv_smin.append(1.0 / s[-1])
            kappa_dR.append(s[0] / s[-1])
        inv_smin = np.array(inv_smin)

        # Sol panel: gürültü büyütmesi (log-log) + 1/ε referansı
        axL.loglog(eps_grid, inv_smin, "o-", color=c, label=f"Düzlem {plane}")
        # 1/ε referansı (ε=0.02 noktasından normalize)
        ref = inv_smin[eps_grid == 0.02][0] * (0.02 / eps_grid)
        axL.loglog(eps_grid, ref, ":", color=c, lw=1, alpha=0.7)

        # Sağ panel: κ(ΔR) (semilogx)
        axR.semilogx(eps_grid, kappa_dR, "s-", color=c, label=f"Düzlem {plane}")

    axL.set_xlabel(r"İki-gradient ayrımı $\varepsilon$")
    axL.set_ylabel(r"$\|\Delta R^{-1}\| = 1/\sigma_{\min}(\Delta R)$")
    axL.set_title(r"Gürültü büyütmesi $\propto 1/\varepsilon$ (nokta çizgi: $1/\varepsilon$ ref.)")
    axL.grid(True, which="both", alpha=0.3)
    axL.legend(fontsize=9)

    axR.set_xlabel(r"İki-gradient ayrımı $\varepsilon$")
    axR.set_ylabel(r"$\kappa(\Delta R)$")
    axR.set_title(r"$\kappa(\Delta R)$ $\varepsilon$'dan kabaca bağımsız "
                  r"($\approx\kappa(R')$)")
    axR.grid(True, which="both", alpha=0.3)
    axR.legend(fontsize=9)

    fig.suptitle(r"Şekil 6 — $\varepsilon$ taraması: ofset-iptal eden "
                 r"estimatörün gürültü büyütmesi $\propto 1/\varepsilon$ patlar; "
                 r"$\kappa(\Delta R)$ ise sabit", fontsize=10.5)
    _save(fig, "fig6_epsilon_sweep.png")


# --------------------------------------------------------------------------
FIGS = {1: fig1_svd_spectra, 2: fig2_drift_tracking,
        3: fig3_betabeat, 4: fig4_permode, 6: fig6_epsilon_sweep}


def main():
    which = [int(a) for a in sys.argv[1:]] or sorted(FIGS)
    print("=" * 60)
    print("drift_makalesi şekilleri üretiliyor:", which)
    print("=" * 60)
    for n in which:
        FIGS[n]()
    print("Tamamlandı.")


if __name__ == "__main__":
    main()
