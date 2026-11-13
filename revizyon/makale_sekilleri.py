#!/usr/bin/env python3
"""makale_sekilleri.py — revize makale (makale_orbit_bastirma_en_v2.tex) için yeni şekiller.

Çıktılar (repo kökü, makalenin \\includegraphics yolu): fig_v2_*.png / .pdf
  optics     : bir FODO hücresi boyunca β_x, β_y ve eleman şeması (Hakem B)
  modes      : (a) ölçülen harmonik kazancı sim/antisim + √β tahmini, (b) SVD tekil değerleri ve
               sağ tekil vektörlerin simetrik içeriği (Hakem B: düşük SVD modları ↔ simetrik)
  suppression: donmuş spin aracıyla C (ham / tek demet / iki demet), Şekil-4 kurulumu
  bpm        : (a) BPM modeli ve kestirici (ρ=0,1), (b) rijit BPM: C vs σ_δ
  deflector  : (a) saptırıcı sürüklenmesinin simetrik kestirime sızması, (b) trim sonrası C vs genlik
  time       : bir haftalık senaryo (sahte EDM, iki demet / tek demet / makale modeli / yalnız BBA)
Bütün sayılar mevcut sonuç dosyalarından; C++ izlemesi yapılmaz.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REV = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(REV)
RES = os.path.join(REV, "sonuclar")
sys.path[:0] = [REV, BASE, os.path.join(BASE, "drift_monitor")]
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", verm="#D55E00", purple="#CC79A7",
          sky="#56B4E9", grey="#595959", yellow="#F0E442")
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8,
                     "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "lines.linewidth": 1.4, "figure.dpi": 200, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})
COL1, COL2 = (3.4, 2.5), (7.0, 2.6)
TGT = 1e-9


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(BASE, f"fig_v2_{name}.{ext}"), bbox_inches="tight")
    plt.close(fig); print("yazıldı fig_v2_" + name)


def panel(ax, s):
    ax.text(-0.16, 1.04, s, transform=ax.transAxes, va="bottom", fontsize=9, fontweight="bold")


def C_pairs(rows, key):
    """rows → {key(r): {dir: f}} → C list."""
    d = {}
    for r in rows:
        d.setdefault(key(r), {})[r["direction"]] = r["f"]
    return {k: abs(v[1] - v[-1]) / 2 / TGT for k, v in d.items() if len(v) == 2}


# ───────────────────────── optics ─────────────────────────
def optics():
    import fodo_lattice as F
    cfg = json.load(open(os.path.join(BASE, "params.json")))
    Ld, Lq, R0, nF = cfg["driftLen"], cfg["quadLen"], cfg["R0"], int(cfg["nFODO"])
    La = np.pi * R0 / nF
    K = cfg["g1"] / F.compute_Brho(F.magic_momentum_proton())
    Kx = F.calibrate_K_x_arc(cfg)
    seq = [("QF", Lq), ("D", Ld), ("A", La), ("D", Ld), ("QD", Lq), ("D", Ld), ("A", La), ("D", Ld)]

    def mat(t, L, pl):
        if t == "QF": return F.thick_quad_matrix(K, L, focusing=(pl == "x"))
        if t == "QD": return F.thick_quad_matrix(K, L, focusing=(pl == "y"))
        if t == "A": return F.arc_matrix(L, Kx if pl == "x" else 0.0)
        return F.drift_matrix(L)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for pl, col, lab in (("x", OI["blue"], r"$\beta_x$"), ("y", OI["verm"], r"$\beta_y$")):
        M = np.eye(2)
        for t, L in seq:
            M = mat(t, L, pl) @ M
        b0, a0, mu = F.twiss_from_periodic_matrix(M)
        g0 = (1 + a0 ** 2) / b0
        s, bet = [0.0], [b0]; T = np.array([b0, a0, g0]); s0 = 0.0
        for t, L in seq:
            n = 40
            for i in range(n):
                m = mat(t, L / n, pl)
                C, S, Cp, Sp = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
                Tm = np.array([[C * C, -2 * C * S, S * S], [-C * Cp, C * Sp + S * Cp, -S * Sp],
                               [Cp * Cp, -2 * Cp * Sp, Sp * Sp]])
                T = Tm @ T; s0 += L / n; s.append(s0); bet.append(T[0])
        ax.plot(s, bet, color=col, label=lab)
    y0 = -6; s0 = 0.0
    for t, L in seq:
        c = {"QF": OI["blue"], "QD": OI["verm"], "A": OI["orange"], "D": None}[t]
        if c:
            h = 5 if t != "A" else 3
            yy = y0 if t != "QD" else y0 - h
            ax.add_patch(plt.Rectangle((s0, yy if t != "A" else y0 - 1.5), L, h, color=c, lw=0))
        s0 += L
    ax.axhline(y0, color="k", lw=0.5)
    ax.text(s0 * 0.005, y0 + 6, "QF", fontsize=6); ax.text(s0 * 0.48, y0 - 10.5, "QD", fontsize=6)
    ax.text(s0 * 0.2, y0 + 3, "deflector", fontsize=6); ax.text(s0 * 0.7, y0 + 3, "deflector", fontsize=6)
    ax.set_xlabel("$s$ in one FODO cell [m]"); ax.set_ylabel(r"$\beta$ [m]")
    ax.set_ylim(-18, 85); ax.legend(loc="upper center", ncol=2, frameon=False)
    save(fig, "optics")


# ───────────────────────── modes ─────────────────────────
def modes():
    import adim1_bpm_model as A1
    import fodo_lattice as F
    g = json.load(open(os.path.join(RES, "adim5_gain.json")))
    cfg = json.load(open(os.path.join(BASE, "params.json")))
    fig, (axa, axb) = plt.subplots(1, 2, figsize=COL2)
    m = np.arange(13)
    for pl, col, mk in (("x", OI["blue"], "o"), ("y", OI["verm"], "s")):
        rs = [next(r for r in g if r["m"] == k and r["kind"] == "sym")[pl]["ccw"] /
              next(r for r in g if r["m"] == k and r["kind"] == "anti")[pl]["ccw"] for k in m]
        axa.plot(m, rs, mk + "-", color=col, ms=3.5, label=f"measured, {pl}")
        b = np.asarray(F.compute_twiss_at_quads(cfg, pl)[0])[:2]
        q = abs(np.sqrt(b[0]) - np.sqrt(b[1])) / (np.sqrt(b[0]) + np.sqrt(b[1]))
        axa.axhline(q, color=col, ls=":", lw=1)
    axa.text(7.2, 0.115, r"$\frac{|\sqrt{\beta_F}-\sqrt{\beta_D}|}{\sqrt{\beta_F}+\sqrt{\beta_D}}$", fontsize=7)
    axa.axhline(0.01, color=OI["grey"], ls="--", lw=1)
    axa.text(0.3, 0.0115, r"smooth-focusing estimate ($\sim$1/100)", fontsize=6.5, color=OI["grey"])
    axa.set_yscale("log"); axa.set_xlabel("cell harmonic $m$")
    axa.set_ylabel("orbit gain, symmetric / antisymmetric"); axa.legend(frameon=False, loc="lower right")
    panel(axa, "(a)")
    for pl, col, mk in (("y", OI["verm"], "s"), ("x", OI["blue"], "o")):
        Rm = A1.R[pl]["ccw"]
        U, s, Vt = np.linalg.svd(Rm)
        frac = np.array([np.linalg.norm(A1.PS @ Vt[i]) ** 2 for i in range(48)])
        sc = axb.scatter(np.arange(1, 49), s / s[0], c=frac, cmap="viridis", vmin=0, vmax=1, s=10,
                         marker=mk, label=f"{pl}")
    cb = fig.colorbar(sc, ax=axb, pad=0.01); cb.set_label("symmetric content $|P_s v_i|^2$", fontsize=7)
    axb.set_yscale("log"); axb.set_xlabel("singular value index $i$"); axb.set_ylabel("$s_i/s_1$")
    axb.legend(frameon=False, loc="lower left"); panel(axb, "(b)")
    save(fig, "modes")


# ───────────────────────── suppression ─────────────────────────
def suppression():
    rows = [json.loads(l) for l in open(os.path.join(RES, "donmus_fig4.jsonl"))]
    fig4 = [r for r in rows if r["kind"] == "fig4"]
    fig, ax = plt.subplots(figsize=COL1)
    cases = (("raw", "no correction", OI["grey"], "o"), ("one", "single-beam orbit corr.", OI["green"], "s"),
             ("two", "two-beam orbit corr.", OI["blue"], "D"))
    for i, (c, lab, col, mk) in enumerate(cases):
        Cs = C_pairs([r for r in fig4 if r["case"] == c], lambda r: r["seed"])
        v = np.array(list(Cs.values()))
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=col, marker=mk, s=16,
                   edgecolor="k", linewidth=0.3, zorder=5)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=col, lw=2)
    ax.axhline(1, color=OI["verm"], ls="--", lw=1); ax.text(2.35, 1.3, "target", color=OI["verm"], ha="right")
    ax.axhline(0.008, color=OI["grey"], ls=":", lw=1); ax.text(2.35, 0.0045, "numerical floor", color=OI["grey"],
                                                               ha="right", fontsize=6.5)
    ax.set_xticks(range(3), ["none", "single beam", "two beams"]); ax.set_yscale("log")
    ax.set_ylabel(r"$C=\frac{1}{2}|f_{\rm CW}-f_{\rm CCW}|$ [units of target]")
    ax.set_xlim(-0.5, 2.5)
    save(fig, "suppression")


# ───────────────────────── bpm ─────────────────────────
def bpm():
    a1 = [json.loads(l) for l in open(os.path.join(RES, "donmus_adim1_kisa.jsonl"))]
    t2 = [json.loads(l) for l in open(os.path.join(RES, "donmus_t2_kisa.jsonl"))]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=COL2)
    cases = [(0.0, "E0", "none"), (0.0, "E1", "E1"), (0.0, "E3", "E3"), (1.0, "E1", "E1"), (1.0, "E2", "E2"),
             (1.0, "E3", "E3"), (1.0, "E4", "E4")]
    cols = {"E0": OI["grey"], "E1": OI["green"], "E2": OI["orange"], "E3": OI["blue"], "E4": OI["purple"]}
    for i, (rho, m, lab) in enumerate(cases):
        Cs = np.array(list(C_pairs([r for r in a1 if r["rho"] == rho and r["method"] == m],
                                   lambda r: r["seed"]).values()))
        axa.scatter(np.full(len(Cs), i) + np.linspace(-0.15, 0.15, len(Cs)), Cs, s=10, color=cols[m],
                    edgecolor="k", linewidth=0.2, zorder=5)
        axa.plot([i - 0.28, i + 0.28], [np.median(Cs)] * 2, color=cols[m], lw=2)
    axa.set_xticks(range(len(cases)), [c[2] for c in cases])
    axa.axvline(2.5, color="k", lw=0.5)
    axa.text(1.0, 60, r"lab-fixed ($\rho=0$)", ha="center", fontsize=7)
    axa.text(4.5, 60, r"rigid ($\rho=1$)", ha="center", fontsize=7)
    axa.axhline(1, color=OI["verm"], ls="--", lw=1); axa.set_yscale("log"); axa.set_ylim(1e-4, 300)
    axa.set_ylabel("$C$ [units of target]"); panel(axa, "(a)")
    for m, col, lab in (("E4", OI["purple"], "two beams (E4)"), ("E2", OI["orange"], "single beam (E2)")):
        sds = sorted({r["sig_d"] for r in t2}); med = []
        for sd in sds:
            Cs = np.array(list(C_pairs([r for r in t2 if r["sig_d"] == sd and r["method"] == m],
                                       lambda r: r["seed"]).values()))
            axb.scatter(np.full(len(Cs), sd * 1e6), Cs, s=8, color=col, alpha=0.5)
            med.append(np.median(Cs))
        axb.plot(np.array(sds) * 1e6, med, "o-", color=col, label=lab)
    axb.axhline(1, color=OI["verm"], ls="--", lw=1); axb.axhline(0.008, color=OI["grey"], ls=":", lw=1)
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_xlabel(r"BPM--quadrupole relative drift $\sigma_\delta$ [$\mu$m]"); axb.set_ylabel("$C$ [units of target]")
    axb.legend(frameon=False, loc="upper left"); panel(axb, "(b)")
    save(fig, "bpm")


# ───────────────────────── deflector ─────────────────────────
def deflector():
    t1 = [json.loads(l) for l in open(os.path.join(RES, "donmus_t1_kisa.jsonl"))]
    gl = [json.loads(l) for l in open(os.path.join(RES, "donmus_t1_genlik.jsonl"))]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=COL2)
    for g, ls, labg in (("", "-", "exact model"), ("_g20", "--", r"0.2% gradient error")):
        d = json.load(open(os.path.join(RES, f"adim2_scan{g}_n14nm.json")))
        rows = [r for r in d["rows"] if r["plane"] == "x" and r["sigma_defl"] > 0]
        sd = np.array([r["sigma_defl"] for r in rows])
        axa.plot(sd, [r["Q_med"] for r in rows], ls=ls, color=OI["grey"], marker="o", ms=3,
                 label="quadrupole-only fit, " + labg)
        axa.plot(sd, [r["QD_med"] for r in rows], ls=ls, color=OI["blue"], marker="D", ms=3,
                 label="with deflector columns, " + labg)
    axa.set_xscale("log"); axa.set_yscale("log")
    axa.set_xlabel(r"deflector field drift $\sigma_\varepsilon$")
    axa.set_ylabel(r"symmetric reconstruction error [$\mu$m]")
    axa.legend(frameon=False, fontsize=5.8, loc="upper left")
    panel(axa, "(a)")
    for case, col, lab in (("trim", OI["blue"], "trim, exact model"), ("trim_g20", OI["verm"], r"trim, 0.2% gradient error"),
                           ("QD_g20", OI["grey"], r"no trim, 0.2% gradient error")):
        A, med = [], []
        for a in (3e-7, 1e-6, 3e-6, 1e-5):
            src = [r for r in t1 if r["case"] == case] if a == 3e-7 else [r for r in gl if r["amp"] == a and r["case"] == case]
            Cs = np.array(list(C_pairs(src, lambda r: r["seed"]).values()))
            if len(Cs) == 0:
                continue
            axb.scatter(np.full(len(Cs), a), Cs, s=8, color=col, alpha=0.5)
            A.append(a); med.append(np.median(Cs))
        axb.plot(A, med, "o-", color=col, label=lab)
    axb.axhline(1, color=OI["verm"], ls="--", lw=1); axb.axhline(0.008, color=OI["grey"], ls=":", lw=1)
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_xlabel(r"deflector drift rms $\sigma_\varepsilon=\sigma_\phi$"); axb.set_ylabel("$C$ [units of target]")
    axb.legend(frameon=False, loc="upper left"); panel(axb, "(b)")
    save(fig, "deflector")


def _plot_scan(ax, sc):
    ax.text(0.5, 0.5, "see adim2_scan", transform=ax.transAxes)


# ───────────────────────── time ─────────────────────────
def time_scenario():
    rows = [json.loads(l) for l in open(os.path.join(RES, "donmus_adim6_kisa.jsonl"))]
    S = json.load(open(os.path.join(RES, "adim6_series.json")))
    fig, (axa, axb) = plt.subplots(1, 2, figsize=COL2)
    col = {"W1_E4": OI["purple"], "W1_E2": OI["orange"], "W0_E3": OI["blue"], "W1_bba_only": OI["green"],
           "W1_none": OI["grey"]}
    lab = {"W1_E4": "rigid BPM, two beams", "W1_E2": "rigid BPM, single beam", "W0_E3": "monitors move, fixed-monitor fit (E3)",
           "W1_bba_only": "periodic BBA only", "W1_none": "no correction"}
    for k, r in S["runs"].items():
        h = np.array(r["hours"])
        axa.semilogy(h, r["sym_med"], color=col[k], label=lab[k])
        axa.fill_between(h, r["sym_p16"], r["sym_p84"], color=col[k], alpha=0.12, lw=0)
    axa.set_xlabel("time [h]"); axa.set_ylabel(r"symmetric residual [$\mu$m]"); panel(axa, "(a)")
    t0 = C_pairs([r for r in rows if r["hour"] == 0], lambda r: r["seed"])
    for k in col:
        w, m = k.split("_", 1)
        hs = [0, 24, 96, 168]; med = []
        for h in hs:
            src = [r for r in rows if r["hour"] == 0] if h == 0 else [r for r in rows if r["world"] == w and r["method"] == m and r["hour"] == h]
            Cs = np.array(list(C_pairs(src, lambda r: r["seed"]).values()))
            axb.scatter(np.full(len(Cs), h), np.maximum(Cs, 1e-3), color=col[k], s=6, alpha=0.5)
            med.append(np.median(Cs))
        axb.plot(hs, med, "o-", color=col[k], ms=3)
    axb.axhline(1, color=OI["verm"], ls="--", lw=1)
    axb.set_yscale("log"); axb.set_xlabel("time [h]"); axb.set_ylabel("$C$ [units of target]"); panel(axb, "(b)")
    h0, l0 = axa.get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12))
    save(fig, "time")


ALL = dict(optics=optics, modes=modes, suppression=suppression, bpm=bpm, deflector=deflector, time=time_scenario)
if __name__ == "__main__":
    for k in (sys.argv[1:] or ALL):
        ALL[k]()
