#!/usr/bin/env python3
"""make_kmod_figures.py — all-quad AC-BBA makalesi figürleri.

Üretir:
  fig_kmod_obs.png       : (a) ΔR tekil-değer spektrumu (simetrik içerikle renkli)
                           (b) mod-ayrımlı geri-çatım korelasyonu (AC-BBA vs ΔR)
  fig_kmod_linchpin.png  : kalan sahte-EDM vs β-beating (estimator + ölçek yasası)
  fig_kmod_sigma.png     : σ² mekanizma doğrulaması (sahte-EDM vs hizalama σ, p=2.00)
  fig_kmod_syst.png      : (a) tilt ψ-taraması (0.2 mrad→1 nrad/s)
                           (b) CW/CCW tilt marjinali EVEN(söner) vs ODD(kalır)

Gözlenebilirlik nicelikleri ac_bba_observability'den yeniden hesaplanır (hızlı);
estimator sonuçları (pahalı) sabit gömülü (bkz. kmod_bba_sonuclar.md §4, §7;
ham veri: kmod_drivers/ sürücüleri). Kullanım: python3 make_kmod_figures.py
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


def fig_sigma():
    """σ² mekanizma doğrulaması: sahte-EDM vs hizalama σ, log-log eğim p=2.00.

    Estimator (azaltılmış-ayar, beyaz misalignment, 3 seed/σ; kmod_drivers/fast_est
    calib + sig25). Eğer /tmp'de sig25.json varsa 2.5μm noktası oradan; yoksa gömülü.
    """
    sig = np.array([10e-6, 5e-6, 2.5e-6])
    f = np.array([9.12e-7, 2.28e-7, 5.9e-8])          # gömülü (calib + sig25 ortalamaları)
    ferr = np.array([6.4e-7, 1.6e-7, 4.0e-8])
    j = "/tmp/kmod_recover/sig25.json"
    if os.path.exists(j):                              # taze 2.5μm varsa kullan
        d = json.load(open(j)); fs = np.array(d["f"])
        f[2] = fs.mean(); ferr[2] = fs.std()
    p = np.polyfit(np.log(sig), np.log(f), 1)[0]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.errorbar(sig * 1e6, f, yerr=ferr, fmt="o", color="C0", ms=7, capsize=3,
                zorder=5, label="estimator (4D-CO + model-fit)")
    ss = np.array([2.0, 12.0])
    ax.plot(ss, f[0] * (ss / 10.0) ** 2, "--", color="C3", lw=1.3,
            label=r"$\propto \sigma^2$ (geometrik faz)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"quad hizalama hatası RMS $\sigma$ [$\mu$m]")
    ax.set_ylabel(r"$|dS_y/dt|$ sahte-EDM [rad/s]")
    ax.set_title(rf"$\sigma^2$ mekanizma: üs $p={p:.2f}$")
    ax.legend(loc="upper left"); ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out = os.path.join(BASE, "fig_kmod_sigma.png")
    fig.savefig(out); plt.close(fig)
    print(f"Kaydedildi: {out}  (p={p:.3f})")


def fig_systematics():
    """(a) tilt ψ-taraması (0.3μm kalan; 6 seed) (b) CW/CCW EVEN/ODD (10μm; 20 seed).
    Veri: kmod_drivers/{tiltscan,cwccw_ens}; özet sabitler kmod_bba_sonuclar.md §7.3."""
    # (a) ψ-taraması: |f| ort ± sem [nrad/s], 0.3μm kalan misalignment
    psi = np.array([0.0, 0.1, 0.2, 0.5])
    fpsi = np.array([1.302, 1.065, 1.298, 3.995])     # nrad/s
    fsem = np.array([0.57, 0.29, 0.35, 0.88])
    # (b) CW/CCW tilt marjinali RMS [nrad/s] (10μm mis + 1mrad tilt, 20 seed)
    even, even_e = 30.7, 6.3
    odd, odd_e = 57.3, 12.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # (a)
    ax1.axhline(1.0, color="green", ls="--", lw=1.0, label="hedef 1 nrad/s")
    ax1.axhline(0.981, color="purple", ls=":", lw=0.9, label="gerçek EDM")
    ax1.axvspan(0, 0.3, color="green", alpha=0.08)
    ax1.errorbar(psi, fpsi, yerr=fsem, fmt="o-", color="C0", ms=6, capsize=3,
                 lw=1.0, zorder=5)
    ax1.text(0.15, 0.45, "tilt katkısı\n≈0", ha="center", fontsize=7, color="green")
    ax1.annotate("0.5 mrad:\ntilt baskın", xy=(0.5, 3.995), xytext=(0.33, 2.3),
                 fontsize=7, color="C3",
                 arrowprops=dict(arrowstyle="->", color="C3", lw=0.8))
    ax1.set_xlabel(r"quad tilt RMS $\psi$ [mrad]")
    ax1.set_ylabel(r"kalan sahte-EDM [nrad/s]")
    ax1.set_title("(a) tilt ψ-taraması (0.3 μm kalan)")
    ax1.set_ylim(0, 5.2); ax1.legend(loc="upper left", fontsize=6.8)
    ax1.grid(True, alpha=0.25)

    # (b)
    bars = ax2.bar([0, 1], [even, odd], yerr=[even_e, odd_e], capsize=4,
                   color=["C2", "C3"], width=0.6)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["EVEN\n(CW/CCW söner)", "ODD\n(CW/CCW kalır)"])
    ax2.set_ylabel(r"tilt marjinali RMS [nrad/s]")
    ax2.set_title("(b) CW/CCW tilt'i gidermez (ODD>EVEN)")
    for x, v in zip([0, 1], [even, odd]):
        ax2.text(x, v + 13, f"{v:.0f}", ha="center", fontsize=8)
    ax2.text(0.5, odd * 1.18, f"ODD/EVEN = {odd/even:.1f}", ha="center",
             fontsize=7.5, color="C3")
    ax2.set_ylim(0, 85); ax2.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    out = os.path.join(BASE, "fig_kmod_syst.png")
    fig.savefig(out); plt.close(fig)
    print("Kaydedildi:", out)


if __name__ == "__main__":
    os.chdir(BASE)
    fig_observability()
    fig_linchpin()
    fig_sigma()
    fig_systematics()
