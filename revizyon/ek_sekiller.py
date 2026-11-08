#!/usr/bin/env python3
"""ek_sekiller.py — ek rapor için şekiller (sonuclar/ek_sekil_*.pdf/png).

Her şekil mevcut sonuç dosyalarından ya da hızlı doğrusal hesaptan üretilir; C++ izlemesi yapılmaz.
Çalıştırma: python3 ek_sekiller.py           (hepsi)
            python3 ek_sekiller.py s8 s12    (yalnız seçilenler)
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch

REV = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(REV)
OUT = os.path.join(REV, "sonuclar")
sys.path[:0] = [REV, os.path.join(REV, "tests"), os.path.join(BASE, "drift_monitor")]

# Referans kategorik palet (dataviz/palette.md, açık mod), sabit sıra
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
TARGET = "#e34948"

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
        fig.savefig(os.path.join(OUT, f"ek_sekil_{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("yazıldı:", name)


def target_line(ax, y=1.0, label="hedef (1 nrad/s)"):
    ax.axhline(y, color=TARGET, lw=1.2, ls="--")
    ax.text(ax.get_xlim()[1], y, " " + label, color=TARGET, va="bottom", ha="right", fontsize=7.5)


def bars(ax, x, h, color, w, label=None):
    return ax.bar(x, h, width=w, color=color, edgecolor="white", linewidth=1.5, label=label)


# ─────────────────────────── S1: kanal normları (soru 1) ───────────────────────────
def s1():
    import adim1_bpm_model as A1
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, pl in zip(axs, "xy"):
        Rc, Rw = A1.R[pl]["ccw"], A1.R[pl]["cw"]
        vals = {}
        for nm, P in (("simetrik", A1.PS), ("antisimetrik", A1.PA)):
            vals[nm] = (np.linalg.norm((Rc + Rw) @ P) / 2, np.linalg.norm((Rc - Rw) @ P) / 2)
        x = np.arange(2); w = 0.36
        bars(ax, x - w / 2, [vals["simetrik"][0], vals["antisimetrik"][0]], C1, w, "toplam kanalı (CW+CCW)")
        bars(ax, x + w / 2, [vals["simetrik"][1], vals["antisimetrik"][1]], C2, w, "fark kanalı (CW−CCW)")
        for i, nm in enumerate(("simetrik", "antisimetrik")):
            for j, dx in enumerate((-w / 2, w / 2)):
                ax.text(i + dx, vals[nm][j] + 0.6, f"{vals[nm][j]:.1f}", ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(x, ["simetrik kaçıklık", "antisimetrik kaçıklık"])
        ax.set_title(f"{pl} düzlemi", fontsize=9)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("yörünge tepkisi (Frobenius normu)")
    axs[0].legend(loc="upper left")
    save(fig, "s1_kanallar")


# ─────────────────────── S2: tekil değerler ve rcond (soru 4, 12) ───────────────────────
def s2():
    import adim1_bpm_model as A1
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, pl in zip(axs, "xy"):
        Rc, Rw = A1.R[pl]["ccw"], A1.R[pl]["cw"]
        s1_ = np.linalg.svd(Rc, compute_uv=False); s2_ = np.linalg.svd(np.vstack([Rc, Rw]), compute_uv=False)
        i = np.arange(1, 49)
        ax.semilogy(i, s1_ / s1_[0], "o", ms=3.5, color=C2, label="tek demet $R_{\\rm CCW}$")
        ax.semilogy(i, s2_ / s2_[0], "o", ms=3.5, color=C1, label="iki demet $[R_{\\rm CCW};R_{\\rm CW}]$")
        ax.axhline(0.01, color=INK2, lw=1, ls="--")
        ax.text(1, 0.012, "rcond = 0.01 (makale)", fontsize=7.5, color=INK2)
        ax.axhline(0.003, color=INK2, lw=1, ls=":")
        ax.text(1, 0.0034, "0.003", fontsize=7.5, color=INK2)
        n1 = int((s1_ < 0.01 * s1_[0]).sum()); n2 = int((s2_ < 0.01 * s2_[0]).sum())
        ax.set_title(f"{pl}: 0.01 eşiği tek demette {n1},\niki demette {n2} doğrultu atar", fontsize=8.5)
        ax.set_xlabel("tekil değer sırası")
    axs[0].set_ylabel("$s_i/s_1$"); axs[0].legend(loc="upper right")
    save(fig, "s2_tekil")


# ─────────────── S3: f ve ω_h'nin δ'ya bağımlılığı (soru 6, 8) ───────────────
def s3():
    r = json.load(open(os.path.join(OUT, "teshis_enerji_ABC.json")))
    a = sorted([x for x in r if x["mode"] == "energy" and x["k"] == 0 and abs(x["delta"]) in (0, 1e-7, 1e-6)
                and "C_test" not in x], key=lambda x: x["delta"])
    fr = [x for x in r if x["mode"] == "energy" and x["k"] == 0 and abs(x["w_h"]) < 1e-4][0]
    d = np.array([x["delta"] for x in a]); f = np.array([x["f_target"] for x in a]); w = np.array([x["w_h"] for x in a])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, y, lab, col in ((axs[0], w, r"yatay presesyon $\omega_h$ [rad/s]", C1),
                            (axs[1], f, r"sahte EDM $f$ [× hedef]", C2)):
        ax.plot(d * 1e7, y, "o-", color=col, ms=5)
        yf = fr["w_h"] if col == C1 else fr["f_target"]
        ax.plot([fr["delta"] * 1e7], [yf], "D", color=INK, ms=7, zorder=5)
        ax.annotate("donmuş spin\n($\\omega_h=0$)", (fr["delta"] * 1e7, yf), xytext=(8, 25 if col == C1 else -40),
                    textcoords="offset points", fontsize=7.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2))
        ax.set_xlabel(r"enerji sapması $\delta$ [$10^{-7}$]"); ax.set_ylabel(lab)
        ax.axhline(0, color=INK2, lw=0.8)
    axs[0].set_title(r"eğim $-5.24\times10^{6}$ rad/s (kaçıklıktan bağımsız)", fontsize=8.5)
    axs[1].set_title(r"eğim $\approx-290\times$hedef / $10^{-7}$ ($\propto\sigma$)", fontsize=8.5)
    save(fig, "s3_delta")


# ─────────────── S4: hücre şeması + |p| profili + fırlatma bağımsızlığı (soru 9) ───────────────
def s4():
    fig = plt.figure(figsize=(7.4, 5.0))
    ax0 = fig.add_axes([0.04, 0.70, 0.92, 0.26])
    # hücre şeması: elem 1..7,0 sırası
    el = [("sürükl.", "d", 1.0), ("QF", "q", 0.6), ("sürükl.", "d", 1.0), ("ARC2\n(saptırıcı)", "a", 2.4),
          ("sürükl.", "d", 1.0), ("QD", "q", 0.6), ("sürükl.", "d", 1.0), ("ARC1\n(saptırıcı)", "a", 2.4)]
    x = 0
    col = {"d": "#f1f0ec", "q": C1, "a": C4}
    for nm, t, L in el:
        ax0.add_patch(Rectangle((x, 0.3), L, 0.4, facecolor=col[t], edgecolor="white", linewidth=2))
        ax0.text(x + L / 2, 0.5, nm, ha="center", va="center", fontsize=7.5, color=INK if t != "q" else "white")
        x += L
    ax0.annotate("fırlatma noktası\n(alansız bölge)", (0.02, 0.7), xytext=(0.3, 1.25), fontsize=7.5,
                 arrowprops=dict(arrowstyle="->", color=INK), color=INK)
    ax0.annotate("|p| burada ölçüldü\n(ARC2 girişi)", (3.6, 0.7), xytext=(3.0, 1.25), fontsize=7.5,
                 arrowprops=dict(arrowstyle="->", color=INK), color=INK)
    ax0.text(x / 2 + 3.0, 1.15, "saptırıcı içinde E alanı radyal harekete göre iş yapar;\n"
             "sert kenarda potansiyel basamağı yok → |p| hücreden hücreye değişir",
             fontsize=7.5, color=INK2, ha="center")
    ax0.annotate("", (x, 0.15), (0, 0.15), arrowprops=dict(arrowstyle="->", color=INK2))
    ax0.text(x / 2, 0.02, "demet yönü (bir FODO hücresi; halkada 24 hücre)", ha="center", fontsize=7.5, color=INK2)
    ax0.set_xlim(-0.2, x + 0.2); ax0.set_ylim(0, 1.6); ax0.axis("off")

    p = json.load(open(os.path.join(OUT, "teshis_p_azimut.json")))
    ax1 = fig.add_axes([0.08, 0.10, 0.38, 0.48])
    cells = [0, 6, 12, 18]
    for i, (r, c, lab) in enumerate(((p[0], C2, r"$\sigma=10~\mu$m"), (p[1], C1, "kaçıklık yok"))):
        v = [r["cells"][str(k)]["delta_mean"] * 1e7 for k in cells]
        bars(ax1, np.arange(4) + (i - 0.5) * 0.36, v, c, 0.36, lab)
    ax1.set_xticks(range(4), [f"hücre {k}" for k in cells]); ax1.axhline(0, color=INK2, lw=0.8)
    ax1.set_ylabel(r"$|p|/p_{\rm sihirli}-1$  [$10^{-7}$]"); ax1.legend(loc="lower right")
    ax1.set_ylim(-6.6, 0.3); ax1.text(1.5, -6.4, "kaçıklık yokken |fark| < 2e−12 (çubuklar görünmez)", fontsize=7, color=INK2, ha="center")
    ax1.set_title("aynı kapalı yörüngede |p| (eski prosedür)", fontsize=8.5); ax1.grid(axis="x", visible=False)

    r = json.load(open(os.path.join(OUT, "teshis_enerji_ABC.json")))
    ax2 = fig.add_axes([0.58, 0.10, 0.40, 0.48])
    ks = [0, 6, 12, 18]
    pm = [next(x["f_target"] for x in r if x["mode"] == "pmag" and x["k"] == k) for k in ks]
    en = [next(x["f_target"] for x in r if x["mode"] == "energy" and x["k"] == k and x["delta"] == 0.0) for k in ks]
    fz = [next(x["f_target"] for x in r if x["mode"] == "energy" and x["k"] == k and abs(x["w_h"]) < 1e-4) for k in ks]
    ax2.plot(ks, pm, "o-", color=C2, ms=5, label=r"eski: $|p|=p_{\rm sihirli}$")
    ax2.plot(ks, en, "s-", color=C4, ms=5, label="toplam enerji 'tasarımda'")
    ax2.plot(ks, fz, "D-", color=C1, ms=5, label="donmuş spin (82.0 ± 0.1)")
    ax2.axhline(0, color=INK2, lw=0.8)
    ax2.set_xticks(ks, [str(k) for k in ks]); ax2.set_xlabel("fırlatma hücresi")
    ax2.set_ylabel("sahte EDM $f$ [× hedef]"); ax2.legend(loc="upper right", fontsize=7)
    ax2.set_title(r"aynı desen, farklı fırlatma noktası ($\sigma$=10 μm)", fontsize=8.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"ek_sekil_s4_firlatma.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig); print("yazıldı: s4_firlatma")


# ─────────────── S5: σ² ve eski/yeni oranı (soru 10) ───────────────
def s5():
    r = json.load(open(os.path.join(OUT, "donmus_sigma.json")))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for s, c in zip((0, 1, 2), (C1, C2, C3)):
        rr = sorted([x for x in r if x["seed"] == s], key=lambda x: x["sigma_um"])
        sg = [x["sigma_um"] for x in rr]
        axs[0].loglog(sg, [abs(x["f"]) / 1e-9 for x in rr], "o-", color=c, ms=5, label=f"seed {s}: donmuş spin")
        axs[0].loglog(sg, [abs(x["f0"]) / 1e-9 for x in rr], "o--", color=c, ms=4, mfc="white", lw=1.2,
                      label=f"seed {s}: sabit fırlatma")
        axs[1].plot(sg, [x["f0"] / x["f"] for x in rr], "o-", color=c, ms=5, label=f"seed {s}")
    axs[0].set_xlabel(r"$\sigma$ [$\mu$m]"); axs[0].set_ylabel(r"$|f|$ [× hedef]")
    axs[0].set_title(r"ikisi de $\propto\sigma^2$ (eğim 2)", fontsize=8.5); axs[0].legend(fontsize=6, ncol=1, loc="upper left")
    axs[1].axhline(1, color=INK2, lw=0.8)
    axs[1].set_xlabel(r"$\sigma$ [$\mu$m]"); axs[1].set_ylabel(r"$f_{\rm sabit}/f_{\rm donmuş}$")
    axs[1].set_title("oran σ'dan bağımsız ama 1 değil", fontsize=8.5); axs[1].legend()
    save(fig, "s5_sigma2")


# ─────────────── S6: lab-sabit ve rijit BPM çizimi (soru 13, 29) ───────────────
def s6():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, rigid in zip(axs, (False, True)):
        ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(-0.2, 6.2)
        ax.plot([0, 10], [2.0, 2.0], color=INK2, lw=0.8, ls=":")
        ax.text(8.0, 1.65, "ideal eksen", fontsize=7, color=INK2)
        qc = 2.6                                       # kaymış kuadrupol merkezi
        ax.add_patch(Rectangle((1.5, qc + 0.9), 2.0, 0.6, fc=C1, ec="white"))
        ax.add_patch(Rectangle((1.5, qc - 1.5), 2.0, 0.6, fc=C1, ec="white"))
        ax.text(2.5, qc + 1.7, "kuadrupol", ha="center", fontsize=7.5, color=C1)
        ax.plot([1.3, 3.7], [qc, qc], color=C1, lw=1, ls="--")
        ax.annotate("", (1.1, qc), (1.1, 2.0), arrowprops=dict(arrowstyle="<->", color=INK2))
        ax.text(0.55, 2.2, "d", fontsize=9, color=INK2)
        ax.plot([0, 10], [2.15, 2.15], color=C2, lw=2)
        ax.text(4.0, 2.25, "demet (x küçük)", fontsize=7, color=C2)
        by = qc if rigid else 2.0
        ax.add_patch(Rectangle((6.6, by - 0.5), 0.5, 1.0, fc=C3, ec="white"))
        ax.text(6.85, by + 0.65, "BPM", ha="center", fontsize=7.5, color=C3)
        if rigid:
            ax.plot([3.5, 6.6], [qc + 1.2, qc + 1.2], color=C3, lw=2)
            ax.plot([6.85, 6.85], [qc + 0.5, qc + 1.2], color=C3, lw=2)
            ax.text(5.0, qc + 1.35, "rijit bağlantı", fontsize=7, color=C3, ha="center")
            t1, t2, t3 = "BPM kuadrupolle birlikte kayar", r"okuma $r=x-d\approx-d$: büyük sinyal", r"matris $(R-I)\,d$"
        else:
            ax.plot([6.85, 6.85], [-0.1, 1.5], color=INK2, lw=2)
            ax.text(7.0, 0.1, "zemine sabit", fontsize=7, color=INK2)
            t1, t2, t3 = "BPM yerinde durur", r"okuma $r=x$: simetrik kaymada küçük", r"matris $R\,d$"
        ax.text(0.2, 5.9, t1, fontsize=7.5, color=INK)
        ax.text(0.2, 5.3, t2, fontsize=8, color=INK)
        ax.text(0.2, 4.7, t3, fontsize=8, color=INK2)
        ax.set_title(("(b) rijit BPM, ρ = 1" if rigid else "(a) laboratuvara sabit BPM, ρ = 0"), fontsize=9)
    save(fig, "s6_bpm")


# ─────────────── S7: TSVD eşiği seçimi (soru 14) ───────────────
def s7():
    r = json.load(open(os.path.join(OUT, "ek_rcond_secimi.json")))
    import adim1_bpm_model as A1
    d = json.load(open(os.path.join(OUT, "adim1_recon.json")))
    orc = {}
    for x in d["rows"]:
        if x["family"] == "k0" and x["kind"] == "sym" and x["sig_n"] == 14e-9:
            for m in ("E1", "E2", "E3", "E4"):
                orc[(x["rho"], m)] = (x[f"{m}_best_sym_med"], x[f"{m}_fix_sym_med"])
    lab = [f"ρ={x['rho']:.0f}, {x['method']}" + (" (yanlış model)" if x["rho"] == 1 and x["method"] == "E3" else "") for x in r]
    fix = [orc[(x["rho"], x["method"])][1] for x in r]
    ora = [orc[(x["rho"], x["method"])][0] for x in r]
    mor = [x["morozov_sym_med"] for x in r]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    xx = np.arange(len(r)); w = 0.26
    bars(ax, xx - w, fix, C4, w, "sabit rcond = 0.01")
    bars(ax, xx, mor, C1, w, "uyuşmazlık ilkesi (gerçek kaçıklık bilinmeden)")
    bars(ax, xx + w, ora, C3, w, "kahin (gerçek kaçıklıkla en iyi)")
    ax.set_yscale("log"); ax.set_xticks(xx, lab, fontsize=7.5); ax.grid(axis="x", visible=False)
    ax.set_ylabel(r"kalan simetrik rms [$\mu$m]"); ax.legend(loc="upper left", fontsize=7)
    ax.set_title("5 μm simetrik sürüklenme, 14 nm, 200 seed medyanı", fontsize=8.5)
    save(fig, "s7_rcond")


# ─────────────── S8: yanlış ρ̂ (soru 15) ───────────────
def s8():
    L = json.load(open(os.path.join(OUT, "ek_testler_rho_lin.json")))["rows"]
    rows = [json.loads(l) for l in open(os.path.join(OUT, "ek_testler_edm.jsonl"))]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for m, c, lab in (("E4", C1, "iki demet (E4)"), ("E2", C2, "tek demet (E2)")):
        rr = [x for x in L if x["method"] == m]
        axs[0].plot([x["rhohat"] for x in rr], [x["sym_med_um"] for x in rr], "o-", color=c, ms=5, label=lab)
        axs[0].fill_between([x["rhohat"] for x in rr], [x["sym_p16"] for x in rr], [x["sym_p84"] for x in rr],
                            color=c, alpha=0.12, lw=0)
        for rh in (0.9, 0.95):
            Cs = []
            for s in range(6):
                f = {x["direction"]: x["f"] for x in rows if x["kind"] == "rhohat" and x["case"] == f"{m}_{rh}" and x["seed"] == s}
                if len(f) == 2:
                    Cs.append(abs(f[1] - f[-1]) / 2e-9)
            off = -0.006 if m == "E4" else 0.006
            axs[1].scatter([rh + off] * len(Cs), np.maximum(Cs, 1e-4), color=c, s=18, alpha=0.8,
                           label=lab if rh == 0.9 else None)
        ref = 0.003 if m == "E4" else 0.006
        axs[1].scatter([1.0 + (-0.006 if m == "E4" else 0.006)], [ref], marker="D", color=c, s=30)
    axs[0].set_yscale("log"); axs[0].set_xlabel(r"kestiricinin varsaydığı $\hat\rho$ (gerçek ρ = 1)")
    axs[0].set_ylabel(r"kalan simetrik rms [$\mu$m]"); axs[0].legend(); axs[0].set_title("doğrusal, 200 seed", fontsize=8.5)
    axs[1].set_yscale("log"); axs[1].set_xlim(0.87, 1.03); axs[1].set_ylim(1e-4, 3)
    axs[1].axhline(0.008, color=INK2, lw=1, ls=":"); axs[1].text(0.965, 0.0095, "sayısal taban", fontsize=7, color=INK2)
    target_line(axs[1])
    axs[1].set_xticks([0.9, 0.95, 1.0]); axs[1].set_xlabel(r"$\hat\rho$")
    axs[1].set_ylabel(r"sahte EDM $C$ [× hedef]"); axs[1].legend(loc=(0.45, 0.6))
    axs[1].set_title("C++, donmuş spin (nokta = seed; ◆ = Adım 1 medyanı)", fontsize=8.5)
    save(fig, "s8_rhohat")


# ─────────────── S9: E3'ün harmonik başına etkisi (soru 17) ───────────────
def s9():
    import adim1_bpm_model as A1
    rc = A1.best_rcond("k0", "sym", 14e-9, 1.0, "E3")
    P = np.zeros(13); P0 = np.zeros(13)
    for s in range(10):
        res = A1.make_case(s, "sym", 5e-6, 1.0, 14e-9, "k0", rc)
        for pl in "xy":
            P += np.abs(np.fft.rfft((A1.PS @ res[pl]["E3"])[::2])) ** 2
            P0 += np.abs(np.fft.rfft((A1.PS @ res[pl]["E0"])[::2])) ** 2
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    m = np.arange(13)
    cols = [C2 if k in (2, 3) else C1 for k in m]
    ax.bar(m, P / P0, color=cols, edgecolor="white", linewidth=1.5)
    ax.axhline(1, color=INK2, lw=0.8)
    ax.text(2.5, 0.32, "spinin en duyarlı\nolduğu harmonikler", ha="center", fontsize=7.5, color=C2)
    ax.set_xticks(m); ax.set_xlabel("hücre harmoniği m"); ax.grid(axis="x", visible=False)
    ax.set_ylabel("kalan güç / başlangıç gücü")
    ax.set_title("ρ = 1, makale kestiricisi E3: yalnız m = 2–3'ü düzeltiyor (10 seed, iki düzlem)", fontsize=8.5)
    save(fig, "s9_e3_harmonik")


# ─────────────── S10: roll vektör çizimi (soru 18) ───────────────
def s10():
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.axis("off"); ax.set_aspect("equal"); ax.set_xlim(-0.4, 1.5); ax.set_ylim(-0.35, 1.0)
    ax.annotate("", (1.2, 0), (0, 0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
    ax.annotate("", (0, 0.85), (0, 0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
    ax.text(1.22, -0.06, "x (radyal)", fontsize=8, color=INK2); ax.text(0.03, 0.88, "y (düşey)", fontsize=8, color=INK2)
    phi = 0.35
    ax.annotate("", (1.0, 0), (0, 0), arrowprops=dict(arrowstyle="-|>", color=C1, lw=2))
    ax.text(0.55, -0.13, r"$\vec E$ (hizalı): $E_0\,\hat x$", color=C1, fontsize=8)
    ax.annotate("", (np.cos(phi), np.sin(phi)), (0, 0), arrowprops=dict(arrowstyle="-|>", color=C2, lw=2))
    ax.text(np.cos(phi) + 0.03, np.sin(phi) + 0.02, r"roll $\phi$ sonrası", color=C2, fontsize=8)
    ax.plot([np.cos(phi), np.cos(phi)], [0, np.sin(phi)], color=C2, lw=1, ls="--")
    ax.text(np.cos(phi) + 0.03, np.sin(phi) / 2 - 0.03, r"$E_y=E_0\sin\phi\approx E_0\phi$", fontsize=8, color=INK)
    a = np.linspace(0, phi, 20); ax.plot(0.3 * np.cos(a), 0.3 * np.sin(a), color=INK2, lw=1)
    ax.text(0.32, 0.05, r"$\phi$", fontsize=9)
    ax.text(-0.38, -0.32, r"$E_x=E_0\cos\phi\approx E_0(1-\phi^2/2)$: değişim 2. mertebe", fontsize=7, color=INK2)
    save(fig, "s10_roll")


# ─────────────── S11: yüksek gerilim kararlılığı (soru 25) ───────────────
def s11():
    items = [("bu çalışmanın\nsenaryosu (T1)", 3e-7, C1), ("HIRFL-CSRm\nHV (10 ppm)", 1e-5, C4),
             ("COSY 200 kV\ndalgalanma", 1e-5, C4), ("nEDM FuG 200 kV\nkararlılık", 1e-4, C4)]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    y = np.arange(len(items))[::-1]
    ax.barh(y, [v for _, v, _ in items], color=[c for *_, c in items], edgecolor="white", height=0.55)
    for yi, (_, v, _) in zip(y, items):
        ax.text(v * 1.3, yi, f"{v:.0e}".replace("e-0", "e−"), va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [n for n, *_ in items], fontsize=7.5); ax.set_xscale("log"); ax.set_xlim(1e-7, 1e-3)
    ax.set_xlabel("bağıl alan / gerilim kararlılığı"); ax.grid(axis="y", visible=False)
    ax.set_title("varsayılan saptırıcı sürüklenmesi bugünkü kaynaklardan 30–300 kat küçük (kaynaklar ek raporda)",
                 fontsize=8.5)
    save(fig, "s11_hv")


# ─────────────── S12: T2 — C ve harmonik içerik (soru 28–30) ───────────────
def s12():
    rows = [json.loads(l) for l in open(os.path.join(OUT, "donmus_t2_kisa.jsonl"))]
    import t2_bpm_rigid as T2, adim1_bpm_model as A1
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for k, c, lab in (("E4", C1, "iki demet (E4)"), ("E2", C2, "tek demet (E2)")):
        meds = []
        for sd in T2.EDM_SD:
            Cs = []
            for s in sorted({r["seed"] for r in rows}):
                f = {r["direction"]: r["f"] for r in rows if r["seed"] == s and r["sig_d"] == sd and r["method"] == k}
                if len(f) == 2:
                    Cs.append(abs(f[1] - f[-1]) / 2e-9)
            axs[0].scatter([sd * 1e6] * len(Cs), Cs, color=c, s=12, alpha=0.5)
            meds.append(np.median(Cs))
        axs[0].plot(np.array(T2.EDM_SD) * 1e6, meds, "o-", color=c, ms=6, label=lab)
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel(r"BPM–kuadrupol göreli kayma $\sigma_\delta$ [$\mu$m]")
    axs[0].set_ylabel(r"$C$ [× hedef]"); axs[0].axhline(0.008, color=INK2, lw=1, ls=":")
    axs[0].text(0.105, 0.0095, "sayısal taban", fontsize=7, color=INK2)
    target_line(axs[0]); axs[0].legend(loc="upper left"); axs[0].set_title("8 seed (noktalar), medyan (çizgi)", fontsize=8.5)
    m = np.arange(13)
    for k, c, off, lab in (("E4", C1, -0.2, "iki demet"), ("E2", C2, 0.2, "tek demet")):
        rc = T2._best_rc(5e-6, 14e-9, 1e-6, k); P = np.zeros(13)
        for s in range(8):
            for pl in "xy":
                P += np.abs(np.fft.rfft((A1.PS @ T2.residual(s, pl, 5e-6, 1e-6, 14e-9, k, rc))[::2])) ** 2
        axs[1].bar(m + off, P / P.sum(), width=0.4, color=c, edgecolor="white", linewidth=1, label=lab)
    axs[1].axvspan(1.5, 3.5, color=C4, alpha=0.12, lw=0)
    axs[1].text(2.5, 0.55, "spin-duyarlı", ha="center", fontsize=7.5, color=INK2)
    axs[1].set_xticks(m); axs[1].set_xlabel("hücre harmoniği m"); axs[1].set_ylabel("kalan simetrik gücün payı")
    axs[1].set_title(r"$\sigma_\delta$ = 1 μm: iki demetin artığı m = 0–1'de", fontsize=8.5)
    axs[1].legend(); axs[1].grid(axis="x", visible=False)
    save(fig, "s12_t2")


# ─────────────── S13: β ağırlığı ve kazanç oranı (soru 33) ───────────────
def s13():
    import fodo_lattice as F
    cfg = json.load(open(os.path.join(BASE, "params.json")))
    g = json.load(open(os.path.join(OUT, "adim5_gain.json")))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axs[0]
    bx = np.asarray(F.compute_twiss_at_quads(cfg, "x")[0])[:2]; by = np.asarray(F.compute_twiss_at_quads(cfg, "y")[0])[:2]
    for i, (pl, b, c) in enumerate((("x", bx, C1), ("y", by, C2))):
        sq = np.sqrt(b)
        ax.bar([i * 3 + 0, i * 3 + 1], [sq[0], -sq[1]], color=c, edgecolor="white", width=0.8)
        ax.bar([i * 3 + 0.5], [sq[0] - sq[1]], color=INK2, width=0.3)
        ax.text(i * 3 + 0.5, 9.0 - 2.0, f"net/toplam\n{(sq[0]-sq[1])/(sq[0]+sq[1]):+.2f}",
                ha="center", fontsize=7, color=INK)
        ax.text(i * 3 + 0.5, 10.2, f"{pl} düzlemi", ha="center", fontsize=8)
    ax.set_xticks([0, 1, 3, 4], ["QF", "QD", "QF", "QD"]); ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel(r"tekme × $\sqrt{\beta}$ (simetrik desen)"); ax.set_ylim(-10.5, 11); ax.grid(axis="x", visible=False)
    ax.set_title("QF ve QD tekmeleri tam götürmez", fontsize=8.5)
    ax = axs[1]
    m = np.arange(13)
    for pl, c in (("x", C1), ("y", C2)):
        rs = []
        for k in m:
            s = next(r for r in g if r["m"] == k and r["kind"] == "sym")[pl]["ccw"]
            a = next(r for r in g if r["m"] == k and r["kind"] == "anti")[pl]["ccw"]
            rs.append(s / a)
        ax.plot(m, rs, "o-", color=c, ms=4, label=f"{pl}: ölçülen")
    ax.axhline(0.15, color=INK2, lw=1, ls="--"); ax.text(6.5, 0.17, r"$\sqrt{\beta}$ tahmini 0.15", fontsize=7.5, color=INK2)
    ax.axhline(0.01, color=TARGET, lw=1, ls=":"); ax.text(0.2, 0.012, "makalenin '100 kat' iddiası", fontsize=7.5, color=TARGET)
    ax.set_yscale("log"); ax.set_xlabel("hücre harmoniği m"); ax.set_ylabel("simetrik / antisimetrik kazanç")
    ax.set_title("yörünge görünürlüğü oranı (x ve y üst üste)", fontsize=8.5); ax.legend(loc="center right")
    save(fig, "s13_beta")


# ─────────────── S14: BBA artığı ölçeklemesi (soru 37) ───────────────
def s14():
    rows = [json.loads(l) for l in open(os.path.join(OUT, "donmus_adim6_kisa.jsonl"))]
    Cs = []
    for s in range(5):
        f = {r["direction"]: r["f"] for r in rows if r["seed"] == s and r["hour"] == 0}
        if len(f) == 2:
            Cs.append(abs(f[1] - f[-1]) / 2e-9)
    sg = np.linspace(0.05, 1.2, 100)
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    ax.plot(sg, max(Cs) * (sg / 0.3) ** 2, color=C2, label="en kötü seed × (σ/0.3)²")
    ax.plot(sg, np.median(Cs) * (sg / 0.3) ** 2, color=C1, label="medyan × (σ/0.3)²")
    ax.scatter([0.3] * len(Cs), Cs, color=INK, s=14, zorder=5, label="ölçülen (5 seed, t = 0)")
    ax.set_yscale("log"); ax.set_xlabel(r"BBA artığı $\sigma_{\rm BBA}$ [$\mu$m]"); ax.set_ylabel(r"$C$ [× hedef]")
    target_line(ax); ax.axvspan(0.4, 0.5, color=C3, alpha=0.15, lw=0)
    ax.text(0.45, 0.004, "önerilen\nsınır", ha="center", fontsize=7.5, color=INK2)
    ax.legend(loc="upper left", fontsize=7); ax.set_title("σ² ölçeklemesiyle BBA gereksinimi", fontsize=8.5)
    save(fig, "s14_bba")


# ─────────────── S15: BPM sürüklenmesi literatür (soru 27, 31) ───────────────
def s15():
    items = [("SLS 2.0 elektronik şartı\n(0.1 μm/saat → gün)", 2.4, C4), ("APS MBA şartı\n(7 günde)", 1.0, C4),
             ("SPring-8 prototip\n(uzun süre)", 1.0, C4), ("SRRC BPM–girder\n(vardiya, iyileştirilmiş)", 0.1, C4),
             ("T2: iki demet hedefin\n100× altında", 1.0, C1), ("Adım 6 varsayımı\n(günde)", 0.2, C1)]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    y = np.arange(len(items))[::-1]
    ax.barh(y, [v for _, v, _ in items], color=[c for *_, c in items], edgecolor="white", height=0.55)
    for yi, (_, v, _) in zip(y, items):
        ax.text(v * 1.15, yi, f"{v:g} μm", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [n for n, *_ in items], fontsize=7.5); ax.set_xscale("log"); ax.set_xlim(0.05, 6)
    ax.set_xlabel("BPM sürüklenmesi / göreli kayma [μm]"); ax.grid(axis="y", visible=False)
    ax.set_title("literatür (turuncu) ve bu çalışma (mavi); süreler farklı, kaba karşılaştırma", fontsize=8.5)
    save(fig, "s15_bpm_lit")


ALL = dict(s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, s6=s6, s7=s7, s8=s8, s9=s9, s10=s10, s11=s11, s12=s12,
           s13=s13, s14=s14, s15=s15)

if __name__ == "__main__":
    for k in (sys.argv[1:] or ALL):
        ALL[k]()
