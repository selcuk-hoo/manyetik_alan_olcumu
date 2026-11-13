#!/usr/bin/env python3
"""firlatma_sekiller.py — fırlatma yöntemi karşılaştırma raporu için şekiller (sonuclar/firlatma_*.pdf/png).

fA: eski − yeni farkı, kapalı yörünge büyüklüğüne karşı (Şekil 4 desenleri, her yön ayrı)
fB: Şekil 4 — eski ve yeni C, seed başına
fC: önceki raporlardaki ana sayılar, eski ve yeni yöntem
fD: yeni yöntemin başlangıç kuralından bağımsızlığı
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REV = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(REV)
OUT = os.path.join(REV, "sonuclar")
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, TARGET = "#0b0b0b", "#52514e", "#d9d8d4", "#e34948"
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False, "legend.fontsize": 8,
})


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"firlatma_{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig); print("yazıldı:", name)


def fA():
    rows = json.load(open(os.path.join(OUT, "firlatma_fig4_yorunge.json")))
    ham = {}
    p = os.path.join(OUT, "firlatma_eski_ham.json")
    if os.path.exists(p):
        ham = {(r["seed"], r["direction"]): r["f_target"] for r in json.load(open(p))}
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    lab = {"raw": ("düzeltme yok", C4), "one": ("tek demet düzeltmesi", C2), "two": ("iki demet düzeltmesi", C1)}
    for case in ("raw", "one", "two"):
        xs, ys, mk = [], [], []
        for case_, s, d, orb, old, new in rows:
            if case_ != case:
                continue
            if case == "raw":
                old = ham.get((s, d))
            if old is None:
                continue
            xs.append(orb); ys.append(max(abs(old - new), 1e-4)); mk.append(d)
        for d, m, nm in ((1, "o", "CW"), (-1, "s", "CCW")):
            sel = [i for i in range(len(xs)) if mk[i] == d]
            ax.scatter([xs[i] for i in sel], [ys[i] for i in sel], marker=m, s=30, color=lab[case][1],
                       edgecolor="white", linewidth=0.8, label=f"{lab[case][0]}, {nm}")
    xx = np.array([3e-3, 1e4])
    ax.plot(xx, 0.08 * xx, color=INK2, lw=1, ls="--")
    ax.text(2e2, 0.08 * 2e2 * 2.2, "eğim 1", fontsize=7.5, color=INK2)
    ax.axhline(0.008, color=INK2, lw=1, ls=":"); ax.text(4e-3, 0.0045, "sayısal taban", fontsize=7, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"fırlatma anındaki kapalı yörünge büyüklüğü  rms$_x\cdot$rms$_y$  [$\mu$m$^2$]")
    ax.set_ylabel(r"|eski $-$ yeni|  [× hedef]")
    ax.set_title("eski yöntemin hatası kapalı yörüngeyle büyüyor", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left", ncol=1)
    save(fig, "fA_fark_yorunge")


def fB():
    old = json.load(open(os.path.join(BASE, "kmod_drivers", "twobeam_oc_14nm.json")))
    new = [json.loads(l) for l in open(os.path.join(OUT, "donmus_fig4.jsonl"))]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, case, tt in ((axs[0], "one", "tek demet düzeltmesi"), (axs[1], "two", "iki demet düzeltmesi")):
        x = np.arange(5); w = 0.36
        o = [old[str(s)][case]["C"] for s in range(5)]
        n = []
        for s in range(5):
            f = {r["direction"]: r["f"] for r in new if r["kind"] == "fig4" and r["case"] == case and r["seed"] == s}
            n.append(abs(f[1] - f[-1]) / 2e-9)
        ax.bar(x - w / 2, o, w, color=C4, edgecolor="white", linewidth=1.5, label="eski (makale)")
        ax.bar(x + w / 2, n, w, color=C1, edgecolor="white", linewidth=1.5, label="yeni (donmuş spin)")
        ax.set_yscale("log"); ax.set_xticks(x, [f"seed {s}" for s in range(5)], fontsize=7.5)
        ax.grid(axis="x", visible=False); ax.set_title(tt, fontsize=9)
        ax.axhline(1, color=TARGET, lw=1.2, ls="--")
        if case == "two":
            ax.axhline(0.008, color=INK2, lw=1, ls=":")
    axs[0].set_ylabel(r"$C$ [× hedef]"); axs[1].legend(loc="upper right", bbox_to_anchor=(1.0, 0.8), fontsize=7)
    save(fig, "fB_sekil4")


def fC():
    items = [  # (etiket, eski değerler, yeni medyan, yeni aralık)
        ("Şekil 4, tek demet", [54.9, 71.5, 12.1, 0.60, 47.4], [29.4, 19.6, 4.37, 0.017, 11.9]),
        ("Şekil 4, iki demet", [0.0064, 0.0046, 0.0096, 0.0043, 0.0009], [0.0015, 0.0015, 0.0018, 0.0022, 0.0010]),
        ("Adım 1: ρ=1, makale kestiricisi (E3)", [0.54, 3.1, 2.5], None),
        ("T2: rijit BPM, iki demet, σδ=1 μm", [0.17, 0.35, 0.27], None),
        ("Adım 6: makale modeli, 96. saat", [7.5, 15, 21], None),
        ("T1: yalnız kuadrupol, %0.2 β-beat", [1.28, 0.74, 2.80], None),
    ]
    newfiles = {
        2: ("donmus_adim1_kisa.jsonl", lambda r: r["rho"] == 1.0 and r["method"] == "E3"),
        3: ("donmus_t2_kisa.jsonl", lambda r: r["sig_d"] == 1e-6 and r["method"] == "E4"),
        4: ("donmus_adim6_kisa.jsonl", lambda r: r["world"] == "W0" and r["hour"] == 96),
        5: ("donmus_t1_kisa.jsonl", lambda r: r["case"] == "QD_g20"),
    }
    vals = []
    for i, (lab, o, n) in enumerate(items):
        if n is None:
            fn, sel = newfiles[i]
            rows = [json.loads(l) for l in open(os.path.join(OUT, fn)) if sel(json.loads(l))]
            n = []
            for s in sorted({r["seed"] for r in rows}):
                f = {r["direction"]: r["f"] for r in rows if r["seed"] == s}
                if len(f) == 2:
                    n.append(abs(f[1] - f[-1]) / 2e-9)
        vals.append((lab, o, n))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = np.arange(len(vals))[::-1]
    for yi, (lab, o, n) in zip(y, vals):
        ax.scatter(o, [yi + 0.12] * len(o), color=C4, s=14, alpha=0.6)
        ax.scatter(n, [yi - 0.12] * len(n), color=C1, s=14, alpha=0.6)
        mo, mn = np.median(o), np.median(n)
        ax.plot([mo, mn], [yi, yi], color=INK2, lw=1)
        ax.scatter([mo], [yi], color=C4, s=60, marker="D", edgecolor="white", zorder=5)
        ax.scatter([mn], [yi], color=C1, s=60, marker="D", edgecolor="white", zorder=5)
    ax.scatter([], [], color=C4, marker="D", s=40, label="eski yöntem (medyan ◆, seed'ler •)")
    ax.scatter([], [], color=C1, marker="D", s=40, label="yeni yöntem")
    ax.set_yticks(y, [v[0] for v in vals], fontsize=7.5); ax.set_xscale("log")
    ax.axvline(1, color=TARGET, lw=1.2, ls="--"); ax.text(1.1, y[0] + 0.35, "hedef", color=TARGET, fontsize=7.5)
    ax.axvline(0.008, color=INK2, lw=1, ls=":")
    ax.set_xlabel(r"$C$ [× hedef]"); ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, fontsize=7)
    ax.set_title("raporlardaki ana sayılar: eski ve yeni yöntem", fontsize=9)
    save(fig, "fC_ozet")


def fD():
    p = os.path.join(OUT, "teshis_baslangic_kurali.json")
    R = json.load(open(p))
    abc = json.load(open(os.path.join(OUT, "teshis_enerji_ABC.json")))
    en = {x["k"]: x["f_target"] for x in abc if x["mode"] == "energy" and abs(x["w_h"]) < 1e-4}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ks = [r["k"] for r in R]
    axs[0].plot(ks, [r["f0_target"] for r in R], "o-", color=C4, ms=5, label="başlangıç noktası (eski kural, δ = 0)")
    axs[0].plot(ks, [r["f_target"] for r in R], "D-", color=C1, ms=6, label="donmuş spin sonucu")
    axs[0].axhline(0, color=INK2, lw=0.8)
    axs[0].set_xticks(ks); axs[0].set_xlabel("fırlatma hücresi"); axs[0].set_ylabel("$f$ [× hedef]")
    axs[0].legend(fontsize=7); axs[0].set_title("başlangıç kuralı: |p| = p$_{\\rm sihirli}$", fontsize=9)
    x = np.arange(len(ks)); w = 0.36
    axs[1].bar(x - w / 2, [r["f_target"] for r in R], w, color=C1, edgecolor="white", label="eski kuraldan başlayarak")
    axs[1].bar(x + w / 2, [en[k] for k in ks], w, color=C3, edgecolor="white", label="enerji kuralından başlayarak")
    lo = min(min(r["f_target"] for r in R), min(en.values())); hi = max(max(r["f_target"] for r in R), max(en.values()))
    axs[1].set_ylim(lo - 0.3, hi + 0.3); axs[1].set_xticks(x, [str(k) for k in ks])
    axs[1].set_xlabel("fırlatma hücresi"); axs[1].set_ylabel("donmuş spin $f^*$ [× hedef]")
    axs[1].legend(fontsize=7, loc="lower right"); axs[1].grid(axis="x", visible=False)
    axs[1].set_title("sonuç başlangıç kuralından bağımsız (eksen dar)", fontsize=9)
    save(fig, "fD_baslangic")


ALL = dict(fA=fA, fB=fB, fC=fC, fD=fD)
if __name__ == "__main__":
    for k in (sys.argv[1:] or ALL):
        ALL[k]()
