#!/usr/bin/env python3
"""
make_figures.py — drift_makalesi.md için tüm şekilleri PRD tek-sütun formatında üretir.

Şekiller (makale Şekil Listesi ile uyumlu):
  Şekil 1: R ve ΔR singüler-değer spektrumları (ΔR ≈ ε R bulk kayması)    — §2.4/§3.2
  Şekil 2: Drift izleme zaman serisi (gerçek vs kestirilen + mutlak)      — §3.4
  Şekil 3: β-beating tarama eğrisi (tracking hatası vs ε_β)              — §3.7
  Şekil 4: Per-mod SVD (mod indeksi → 1/σ ve simetrik güç, çift eksen)   — §4.3
  Şekil 6: ε taraması (‖ΔR⁻¹‖ ∝ 1/ε; κ(ΔR) sabit)                       — §3.2

Şekil 5 (iki-katmanlı mimari şeması) ayrı: make_fig5_architecture.py.

PRD biçim kuralları (bu betikte uygulanır):
  * Tek-sütun genişlik: 3.375 in (246 pt). Çok-panelli şekiller dikey istiflenir.
  * Serif yazı tipi + Computer-Modern mathtext; küçük puntolar (8/7 pt).
  * Şekil-içi başlık YOK; tüm açıklama makale caption'ında. Paneller (a),(b).
  * savefig 600 dpi (PRD çizgi-grafik standardı).

Tüm hesaplar drift_monitor/fodo_lattice.py üzerinden; C++ bağımlılığı yok.

Kullanım:
    python3 make_figures.py            # tüm şekiller
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
# PRD tek-sütun stili
# --------------------------------------------------------------------------
COL = 3.375          # PRD tek-sütun genişliği [inç] (246 pt)
SAVE_DPI = 600       # PRD çizgi-grafik

plt.rcParams.update({
    "font.family": "serif",          # DejaVu Serif: Türkçe + tüm glifler
    "mathtext.fontset": "cm",        # Computer-Modern matematik
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "legend.handlelength": 1.6,
    "legend.frameon": False,
    "lines.linewidth": 1.1,
    "lines.markersize": 2.6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.linewidth": 0.4,
    "savefig.dpi": SAVE_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

with open(os.path.join(_DIR, "..", "params.json")) as _f:
    CFG = json.load(_f)
with open(os.path.join(_DIR, "test_params.json")) as _f:
    TP = json.load(_f)

EPS = float(TP.get("EPS", 0.02))


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


def _panel_label(ax, text, dx=-0.22, dy=1.02):
    """Panel etiketi (a),(b) — PRD konvansiyonu."""
    ax.text(dx, dy, text, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="left")


def _save(fig, name):
    path = os.path.join(_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  kaydedildi: {os.path.relpath(path)}")


# --------------------------------------------------------------------------
# Şekil 1 — R ve ΔR singüler-değer spektrumları (dikey istif)
# --------------------------------------------------------------------------
def fig1_svd_spectra():
    print("Şekil 1: SVD spektrumları (R vs ΔR)")
    fig, axes = plt.subplots(2, 1, figsize=(COL, 4.4), sharex=True)

    for row, (ax, plane, lab) in enumerate(zip(axes, ("y", "x"), ("(a)", "(b)"))):
        Kxarc = fl.calibrate_K_x_arc(CFG) if plane == "x" else None
        R1 = build_R(CFG, CFG["g1"], plane, Kxarc)
        R2 = build_R(CFG, CFG["g1"] * (1.0 + EPS), plane, Kxarc)
        dR = R1 - R2
        sR = np.linalg.svd(R1, compute_uv=False)
        sdR = np.linalg.svd(dR, compute_uv=False)
        idx = np.arange(1, len(sR) + 1)

        ax.semilogy(idx, sR, "o-", color="C0", label=r"$R$")
        ax.semilogy(idx, sdR, "s-", color="C3", label=r"$\Delta R$")
        ax.semilogy(idx, EPS * sR, "--", color="gray", lw=0.8,
                    label=r"$\varepsilon\,\sigma(R)$")
        ax.set_ylabel(r"$\sigma_i$")
        ax.grid(True, which="both", alpha=0.3)
        kR, kdR = sR[0] / sR[-1], sdR[0] / sdR[-1]
        ax.text(0.97, 0.93,
                rf"düzlem {plane}: $\kappa(R)\!\approx\!{kR:.0f}$, "
                rf"$\kappa(\Delta R)\!\approx\!{kdR:.0f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
        _panel_label(ax, lab)
        if row == 0:
            ax.legend(loc="lower left", ncol=3, columnspacing=1.0)

    axes[-1].set_xlabel("singüler değer indeksi")
    fig.tight_layout(h_pad=0.6)
    _save(fig, "fig1_svd_spektrum.png")


# --------------------------------------------------------------------------
# Şekil 2 — Drift izleme zaman serisi (dikey istif)
# --------------------------------------------------------------------------
def fig2_drift_tracking():
    print("Şekil 2: drift izleme zaman serisi (Test 4)")
    t4 = TP["test4"]
    DQ0 = float(t4.get("DQ0_RMS", 100e-6))
    OFF, NOISE, RAMP = float(t4["BPM_OFFSET"]), float(t4["BPM_NOISE"]), float(t4["DRIFT_RAMP"])
    NEP = int(t4["N_EPOCHS"])

    fig, axes = plt.subplots(2, 1, figsize=(COL, 4.4), sharex=True)
    rng = np.random.default_rng(2026)

    for row, (ax, plane, lab) in enumerate(zip(axes, ("y", "x"), ("(a)", "(b)"))):
        Kxarc = fl.calibrate_K_x_arc(CFG) if plane == "x" else None
        R = build_R(CFG, CFG["g1"], plane, Kxarc)
        N = R.shape[0]
        Rinv = np.linalg.inv(R)
        dq0 = rng.normal(0, DQ0, N)
        b0 = rng.normal(0, OFF, N)
        ramp = rng.normal(0, RAMP, N)
        y0 = R @ dq0 + b0 + rng.normal(0, NOISE, N)

        times = np.arange(0, NEP + 1)
        true_rms = np.zeros_like(times, float)
        hat_rms = np.zeros_like(times, float)
        track_err = np.zeros_like(times, float)
        abs_err = np.zeros_like(times, float)
        for t in times:
            dqt = dq0 + ramp * (t / NEP)
            d_true = dqt - dq0
            yt = R @ dqt + b0 + rng.normal(0, NOISE, N)
            d_hat = Rinv @ (yt - y0)
            true_rms[t] = np.sqrt(np.mean(d_true ** 2))
            hat_rms[t] = np.sqrt(np.mean(d_hat ** 2))
            track_err[t] = np.sqrt(np.mean((d_hat - d_true) ** 2))
            abs_err[t] = np.sqrt(np.mean((Rinv @ yt - dqt) ** 2))

        ax.plot(times, true_rms * 1e6, "o-", color="C0", label="gerçek drift")
        ax.plot(times, hat_rms * 1e6, "s--", color="C1", label="kestirilen")
        ax.plot(times, abs_err * 1e6, "^:", color="C3", label="mutlak rek.")
        ax.axhline(OFF * 1e6, color="gray", ls="--", lw=0.8,
                   label=rf"ofset {OFF*1e6:.0f}$\,\mu$m")
        ax.set_yscale("log")
        ax.set_ylabel(r"RMS [$\mu$m]")
        ax.grid(True, which="both", alpha=0.3)
        ax.text(0.03, 0.5, f"düzlem {plane}", transform=ax.transAxes, fontsize=6.5)
        _panel_label(ax, lab)
        if row == 0:
            ax.legend(loc="center right", ncol=1)

    axes[-1].set_xlabel("epoch")
    fig.tight_layout(h_pad=0.6)
    _save(fig, "fig2_drift_izleme.png")


# --------------------------------------------------------------------------
# Şekil 3 — β-beating tarama eğrisi (tek panel)
# --------------------------------------------------------------------------
def fig3_betabeat():
    print("Şekil 3: β-beating tarama eğrisi (Test 8)")
    OFF, NOISE, RAMP, DQ0, NEP, NSEED = 50e-6, 1e-6, 10e-6, 100e-6, 10, 15

    def nominal(plane):
        Kxarc = fl.calibrate_K_x_arc(CFG) if plane == "x" else None
        beta, phi, Q = fl.compute_twiss_at_quads(CFG, plane, K_x_arc=Kxarc)
        return beta, phi, Q, fl.signed_KL(CFG, plane)

    def track_err(plane, eps_beta, seed):
        beta, phi, Q, KL = nominal(plane)
        N = len(beta)
        Rm_inv = np.linalg.inv(fl.build_response_matrix(beta, phi, Q, KL))
        rb = np.random.default_rng(50000 + seed)
        Rt = fl.build_response_matrix(beta * (1 + rb.normal(0, eps_beta, N)),
                                      phi + rb.normal(0, eps_beta, N), Q, KL)
        rng = np.random.default_rng(1000 + seed)
        dq0 = rng.normal(0, DQ0, N)
        b0 = rng.normal(0, OFF, N)
        ramp = rng.normal(0, RAMP, N)
        y0 = Rt @ dq0 + b0 + rng.normal(0, NOISE, N)
        errs = []
        for t in range(1, NEP + 1):
            dqt = dq0 + ramp * (t / NEP)
            yt = Rt @ dqt + b0 + rng.normal(0, NOISE, N)
            errs.append(np.sqrt(np.mean((Rm_inv @ (yt - y0) - (dqt - dq0)) ** 2)))
        return np.mean(errs)

    eps_grid = np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.10])
    ey = np.array([np.median([track_err("y", e, s) for s in range(NSEED)]) for e in eps_grid])
    ex = np.array([np.median([track_err("x", e, s) for s in range(NSEED)]) for e in eps_grid])

    fig, ax = plt.subplots(figsize=(COL, 2.5))
    ax.plot(eps_grid * 100, ey * 1e6, "o-", color="C0", label="düzlem y")
    ax.plot(eps_grid * 100, ex * 1e6, "s-", color="C1", label="düzlem x")
    ax.axhline(10, color="C3", ls="--", lw=0.9, label=r"hedef 10$\,\mu$m")
    ax.axvline(1.0, color="gray", ls=":", lw=0.8, label="LOCO (%1)")
    ax.set_xlabel(r"$\beta$-beating  $\varepsilon_\beta$  [%]")
    ax.set_ylabel(r"drift takip hatası [$\mu$m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "fig3_betabeat.png")


# --------------------------------------------------------------------------
# Şekil 4 — Per-mod SVD (çift eksen, tek panel)
# --------------------------------------------------------------------------
def fig4_permode():
    print("Şekil 4: per-mod SVD analizi")
    R = build_R(CFG, CFG["g1"], "y")
    U, S, Vt = np.linalg.svd(R)
    N = len(S)
    idx = np.arange(N)
    inv_sigma = 1.0 / S
    symp = np.array([sym_frac(Vt[i]) for i in range(N)]) * 100.0

    fig, ax1 = plt.subplots(figsize=(COL, 2.7))
    ax1.semilogy(idx, inv_sigma, "o-", color="C0", ms=2.6)
    ax1.set_xlabel(r"singüler mod indeksi (büyük $\sigma\!\to\!$ küçük $\sigma$)")
    ax1.set_ylabel(r"$1/\sigma_i$", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(idx, symp, "s--", color="C3", ms=2.6)
    ax2.set_ylabel("simetrik güç [%]", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.set_ylim(0, 105)
    fig.tight_layout()
    _save(fig, "fig4_permode_svd.png")


# --------------------------------------------------------------------------
# Şekil 6 — ε taraması (dikey istif)
# --------------------------------------------------------------------------
def fig6_epsilon_sweep():
    print("Şekil 6: ε-taraması (‖ΔR⁻¹‖ ∝ 1/ε; κ(ΔR) sabit)")
    eps_grid = np.array([0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10])
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(COL, 4.4))

    for plane, c, m in (("y", "C0", "o"), ("x", "C1", "s")):
        Kxarc = fl.calibrate_K_x_arc(CFG) if plane == "x" else None
        R = build_R(CFG, CFG["g1"], plane, Kxarc)
        inv_smin, kappa_dR = [], []
        for eps in eps_grid:
            s = np.linalg.svd(R - build_R(CFG, CFG["g1"] * (1.0 + eps), plane, Kxarc),
                              compute_uv=False)
            inv_smin.append(1.0 / s[-1])
            kappa_dR.append(s[0] / s[-1])
        inv_smin = np.array(inv_smin)

        axT.loglog(eps_grid, inv_smin, m + "-", color=c, label=f"düzlem {plane}")
        ref = inv_smin[eps_grid == 0.02][0] * (0.02 / eps_grid)
        axT.loglog(eps_grid, ref, ":", color=c, lw=0.8)
        axB.loglog(eps_grid, kappa_dR, m + "-", color=c, label=f"düzlem {plane}")

    axT.set_ylabel(r"$\|\Delta R^{-1}\|=1/\sigma_{\min}$")
    axT.grid(True, which="both", alpha=0.3)
    axT.legend(loc="upper right")
    axT.text(0.03, 0.06, r"nokta çizgi: $\propto 1/\varepsilon$",
             transform=axT.transAxes, fontsize=6.5)
    _panel_label(axT, "(a)")

    axB.set_xlabel(r"iki-gradient ayrımı $\varepsilon$")
    axB.set_ylabel(r"$\kappa(\Delta R)$")
    axB.grid(True, which="both", alpha=0.3)
    axB.legend(loc="upper left")
    _panel_label(axB, "(b)")

    fig.tight_layout(h_pad=0.6)
    _save(fig, "fig6_epsilon_sweep.png")


# --------------------------------------------------------------------------
# Şekil 8 — SVD spektrumu = G_k'nın kesin gerçekleşmesi (σ_k ∝ G_k)
# --------------------------------------------------------------------------
def fig8_svd_gain():
    """SVD spektrumu = yörünge gain yasasının kendisi (iki panel, ortak x=k).

    Her SVD modunun iki sayısı var: tekil değeri σ_i ve kick-harmoniği k(i).
    Kick harmoniği (-1)^j çarpanıyla (Nyquist kayması) bulunur — orbit gain
    kaçıklık desenine değil KICK'e uygulanır. Yatay eksen fiziksel harmonik k:

      (a) σ_i ↔ G_k: tekil değerler, modun k'sindeki rezonans kazancı
          G_k=C/|Q_eff²−k²|'ya oturur (k≈Q_eff'te tepe, yüksek-k'de bastırma).
      (b) χ_i ↔ k:   bastırılan yüksek-k uç simetrik (χ>0), iyi ölçülen
          düşük-k uç antisimetrik (χ<0). Gürültü tabanı / no-go bağlantısı.
    """
    print("Şekil 8: SVD spektrumu = gain yasası (σ_i↔G_k, χ_i↔k)")
    C_GAIN, QEFF2 = 24.8, 5.03
    R = build_R(CFG, CFG["g1"], "y")
    U, S, Vt = np.linalg.svd(R)
    N = len(S)
    sign = (-1.0) ** np.arange(N)

    def kick_k(v):
        # DC bin'i (k=0) silmeyin: üniform-antisimetrik mod (v≈(-1)^j·sabit) için
        # sign*v üniformdur, gerçek kick harmoniği k=0'dır. F[0]'ı sıfırlamak bu modu
        # yanlışlıkla k=24'e etiketler (sahte aykırı nokta) — bkz. fig8 doğrulaması.
        F = np.abs(np.fft.rfft(sign * v))
        return int(np.argmax(F))

    ks = np.array([kick_k(Vt[i]) for i in range(N)])                 # her modun k'si
    Gk = C_GAIN / np.abs(QEFF2 - ks ** 2.0)                          # modun kazancı
    chi = np.array([2.0 * sym_frac(Vt[i]) - 1.0 for i in range(N)])  # χ∈[-1,1]
    corr = np.corrcoef(Gk, S)[0, 1]                                  # lineer Pearson
    a = float(np.sum(S * Gk) / np.sum(Gk * Gk))                      # σ≈a·G_k ölçeği

    # Teorik eğri yalnız tamsayı k'de tanımlı (fiziksel harmonikler); pol yok
    # çünkü Q_eff=√5.03≈2.24 hiçbir tamsayıya düşmez.
    kk = np.arange(0, N // 2 + 1)
    Gk_line = C_GAIN / np.abs(QEFF2 - kk ** 2.0)

    fig, (axa, axb) = plt.subplots(2, 1, figsize=(COL, 4.2), sharex=True)

    # ---- Panel (a): σ_i ve teorik a·G_k, ortak yatay eksen k --------------
    axa.plot(kk, a * Gk_line, "--", color="0.4", lw=0.9, zorder=2,
             label=rf"$a\,G_k,\ a={a:.2f}$")
    axa.scatter(ks, S, s=16, color="C0", edgecolors="k", linewidths=0.3,
                zorder=3, label=r"$\sigma_i$ (mod)")
    axa.set_ylabel(r"tekil değer $\sigma_i$")
    axa.set_ylim(0, max(S.max(), (a * Gk_line).max()) * 1.05)
    # LOG ALTERNATİFİ: spektrum ~193× yayılır; bastırılmış kuyruğu görmek için
    # aşağıdaki iki satırı açın (lineer set_ylim'i kapatın):
    # axa.set_yscale("log")
    # axa.set_ylim(S.min() * 0.6, max(S.max(), (a * Gk_line).max()) * 1.6)
    axa.text(0.97, 0.60, rf"Pearson $={corr:.2f}$", transform=axa.transAxes,
             fontsize=7, va="top", ha="right")
    axa.legend(loc="upper right", fontsize=6.5)
    axa.grid(True, alpha=0.3)
    _panel_label(axa, "(a)")

    # ---- Panel (b): χ_i, ortak yatay eksen k ------------------------------
    axb.axhline(0.0, color="gray", lw=0.7, ls=":")
    axb.scatter(ks, chi, c=chi, cmap="coolwarm", s=16, vmin=-1, vmax=1,
                edgecolors="k", linewidths=0.3, zorder=3)
    axb.set_ylim(-1.08, 1.08)
    axb.set_ylabel(r"simetri $\chi_i$")
    axb.set_xlabel(r"kick harmoniği $k$")
    axb.set_xlim(-0.6, N // 2 + 0.6)
    axb.grid(True, alpha=0.3)
    _panel_label(axb, "(b)")

    fig.tight_layout()
    _save(fig, "fig8_svd_gain.png")


# --------------------------------------------------------------------------
FIGS = {1: fig1_svd_spectra, 2: fig2_drift_tracking,
        3: fig3_betabeat, 4: fig4_permode, 6: fig6_epsilon_sweep,
        8: fig8_svd_gain}


def main():
    which = [int(a) for a in sys.argv[1:]] or sorted(FIGS)
    print("=" * 60)
    print("drift_makalesi şekilleri (PRD tek-sütun) üretiliyor:", which)
    print("=" * 60)
    for n in which:
        FIGS[n]()
    print("Tamamlandı.")


if __name__ == "__main__":
    main()
