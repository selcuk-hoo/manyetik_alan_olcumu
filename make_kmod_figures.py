#!/usr/bin/env python3
"""make_kmod_figures.py — all-quad AC-BBA makalesi figürleri.

Üretir:
  fig_kmod_obs.png      : (a) ΔR tekil-değer spektrumu (simetrik içerikle renkli)
                          (b) mod-ayrımlı geri-çatım korelasyonu (AC-BBA vs ΔR)
  fig_kmod_linchpin.png : kalan sahte-EDM vs β-beating (estimator + ölçek yasası)

Gözlenebilirlik nicelikleri ac_bba_observability'den yeniden hesaplanır (hızlı);
estimator sonuçları (pahalı) sabit gömülü (bkz. kmod_bba_sonuclar.md §4).
Kullanım: python3 make_kmod_figures.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ac_bba_observability as obs
import ac_bba_linchpin as lin

BASE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 7.5, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})


def fig_observability():
    with open(os.path.join(BASE, "params.json")) as f:
        cfg = json.load(f)
    nFODO = int(cfg["nFODO"]); n_q = 2 * nFODO

    # ── ΔR tekil değerleri + her modun simetrik içeriği ──
    dR, _, _ = obs.ak.build_analytic_dR(cfg, cfg["g1"], cfg["g1"] * 1.02, "y")
    U, s, Vt = np.linalg.svd(dR)
    sym_frac = np.array([
        (lambda sc, dc: np.sum(sc**2) / (np.sum(sc**2) + np.sum(dc**2)))(
            *obs.sym_antisym(Vt[k])) for k in range(n_q)])

    # ── mod-ayrımlı geri-çatım korelasyonu (gözlenebilirlik testi) ──
    T, *_ = obs.build_T(cfg, "y", 0.02)
    sig_off, sig_noise, n_seed = 100e-6, 1e-6, 40
    R = {("acbba", "sym"): [], ("acbba", "antisym"): [],
         ("dR", "sym"): [], ("dR", "antisym"): []}
    for seed in range(n_seed):
        rng = np.random.default_rng(1000 + seed)
        for kind in ("sym", "antisym"):
            o = obs.make_pattern(kind, nFODO, rng, sigma=1e-4)
            A = T * o[None, :] + rng.normal(0, sig_noise, T.shape)
            R[("acbba", kind)].append(obs.corr(o, obs.recon_acbba(T, A)))
            delta = dR @ o + rng.normal(0, np.sqrt(2) * sig_noise, n_q)
            R[("dR", kind)].append(obs.corr(o, obs.recon_dR(dR, delta)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # (a) tekil-değer spektrumu, simetrik içerikle renklendirilmiş
    sc = ax1.scatter(np.arange(1, n_q + 1), s, c=sym_frac, cmap="coolwarm",
                     s=22, vmin=0, vmax=1, edgecolor="k", linewidth=0.3, zorder=3)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"tekil değer indeksi (büyük$\to$küçük)")
    ax1.set_ylabel(r"$\Delta R$ tekil değeri $s_k$")
    ax1.set_title(r"(a) $\Delta R$ inversiyonu: simetrik = küçük-$s$ = kör")
    cb = fig.colorbar(sc, ax=ax1, pad=0.02)
    cb.set_label("modun simetrik içeriği", fontsize=7.5)
    ax1.text(0.05, 0.08, rf"$\kappa={s[0]/s[-1]:.1e}$", transform=ax1.transAxes,
             fontsize=8, bbox=dict(fc="white", ec="0.6", alpha=0.85))
    ax1.grid(True, which="both", alpha=0.25)

    # (b) mod-ayrımlı geri-çatım korelasyonu
    labels = ["simetrik", "antisim"]
    acb = [np.mean(R[("acbba", "sym")]), np.mean(R[("acbba", "antisym")])]
    acberr = [np.std(R[("acbba", "sym")]), np.std(R[("acbba", "antisym")])]
    dRm = [np.mean(R[("dR", "sym")]), np.mean(R[("dR", "antisym")])]
    dRerr = [np.std(R[("dR", "sym")]), np.std(R[("dR", "antisym")])]
    x = np.arange(2); w = 0.36
    ax2.bar(x - w/2, acb, w, yerr=acberr, capsize=3, color="C0",
            label="per-quad AC-BBA")
    ax2.bar(x + w/2, dRm, w, yerr=dRerr, capsize=3, color="C3",
            label=r"$\Delta R$ inversiyonu")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel(r"geri-çatım korelasyonu $r$")
    ax2.set_ylim(0, 1.08)
    ax2.set_title("(b) gözlenebilirlik: AC-BBA simetriği görür")
    ax2.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax2.legend(loc="center", bbox_to_anchor=(0.5, 0.74), framealpha=0.9)
    ax2.grid(True, axis="y", alpha=0.25)
    for xi, v in zip(x - w/2, acb):
        ax2.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    for xi, v in zip(x + w/2, dRm):
        ax2.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)

    fig.tight_layout()
    out = os.path.join(BASE, "fig_kmod_obs.png")
    fig.savefig(out); plt.close(fig)
    print("Kaydedildi:", out)


def fig_linchpin():
    # estimator sonuçları (pahalı; kmod_bba_sonuclar.md §4)
    eps_pts = np.array([0.0, 1.0, 5.0])               # %
    f_pts = np.array([4.2e-11, 1.03e-9, 1.74e-8])     # rad/s
    f_err = np.array([0.9e-12, 1.5e-10, 7.6e-10])
    A_eff = 1.176e4                                    # rad/s/m² (ε=%1 noktasından)

    # ölçek yasası eğrisi: σ_res(ε) lineer modelden
    eps_grid = np.linspace(0.0, 5.0, 26)
    sig_res = []
    for eps in eps_grid:
        rr = []
        for sd in range(6):
            rng = np.random.default_rng(5000 + sd)
            dx = rng.normal(0, 10e-6, lin.NQ); dy = rng.normal(0, 10e-6, lin.NQ)
            ex, ey = lin.bba_residual_offset(lin.CFG, dx, dy, 0.02, eps/100.0,
                                             1e-6, 1.0, rng)
            rr.append(np.sqrt(np.mean(np.concatenate([ex, ey])**2)))
        sig_res.append(np.mean(rr))
    sig_res = np.array(sig_res)
    f_curve = A_eff * sig_res**2

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.plot(eps_grid, f_curve * 1e9, "-", color="C0", lw=1.4,
            label=r"ölçek yasası $A_\mathrm{eff}\sigma_\mathrm{res}^2$")
    ax.errorbar(eps_pts, f_pts * 1e9, yerr=f_err * 1e9, fmt="o", color="C3",
                ms=6, capsize=3, zorder=5, label="estimator (doğrudan)")
    ax.axhline(1.0, color="green", lw=1.0, ls="--", label="hedef 1 nrad/s")
    ax.axhline(9.81e-1, color="purple", lw=0.9, ls=":",
               label="gerçek EDM (0.98 nrad/s)")
    ax.set_yscale("log")
    ax.set_xlabel(r"optik-model $\beta$-beating $\varepsilon$ [%]")
    ax.set_ylabel(r"kalan sahte-EDM [nrad/s]")
    ax.set_title("LINCHPIN: kalan sahte-EDM")
    ax.axvspan(0, 1.0, color="green", alpha=0.07)
    ax.text(0.5, 3e-2, "hedef\naltı", ha="center", fontsize=7, color="green")
    ax.legend(loc="lower right", fontsize=6.8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out = os.path.join(BASE, "fig_kmod_linchpin.png")
    fig.savefig(out); plt.close(fig)
    print("Kaydedildi:", out)


if __name__ == "__main__":
    os.chdir(BASE)
    fig_observability()
    fig_linchpin()
