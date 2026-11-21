#!/usr/bin/env python3
"""sekil_kusurlu_polarite.py — Omarov birleşiminden sonra kalan sahte EDM, statik saptırıcı hatası σ_d'ye karşı.
Veri: sonuclar/kusurlu_polarite.jsonl (bulut 3 desen + laptop v3). Çıktı: ../fig_kusurlu_polarite.{png,pdf}
(a) iki polarite koşusu arasında 0.3 μm kuadrupol sürüklenmesi: düzeltme yok / tek demet / iki demet izleyici;
(b) diğer kusurlar: gradyan farkı η = 1e-4, 1e-3; saptırıcı 1e-7 sürüklenmesi (trimli/trimsiz); kusursuz çevirme.
Çubuklar: medyan; hata çubuğu: %16–84.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REV = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(REV)
rows = [json.loads(l) for l in open(os.path.join(REV, "sonuclar", "kusurlu_polarite.jsonl"))]
f = {}
for r in rows:
    f.setdefault((r["sig_d"], r["seed"], r["state"], r["gs"]), {})[r["direction"]] = r["f_target"]
odd = {k: (v[1] - v[-1]) / 2 for k, v in f.items() if len(v) == 2}


def E(sd, st, gs):
    ss = sorted({k[1] for k in odd if k[0] == sd})
    return np.array([abs(odd[(sd, s, "S0", 1.0)] + odd[(sd, s, st, gs)]) / 2 for s in ss
                     if (sd, s, "S0", 1.0) in odd and (sd, s, st, gs) in odd])


SD = [1e-6, 1e-5, 3e-5, 1e-4]
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.6))
for a, cases, title in (
        (ax[0], [("dq_yok", -1.0, "drift, no correction", "#595959", "o"),
                 ("dq_E2", -1.0, "single-beam monitor", "#E69F00", "s"),
                 ("dq_E4", -1.0, "two-beam monitor", "#0072B2", "D")],
         r"(a) 0.3 $\mu$m quadrupole drift between polarity runs"),
        (ax[1], [("S0", -1.001, r"$\eta=10^{-3}$", "#D55E00", "^"), ("S0", -1.0001, r"$\eta=10^{-4}$", "#CC79A7", "v"),
                 ("dd_yok", -1.0, r"deflector drift $10^{-7}$", "#595959", "o"),
                 ("dd_E4", -1.0, "same, two-beam fit + trim", "#009E73", "s"),
                 ("dc5_yok", -1.0, r"common field shift $10^{-5}$", "#000000", "P"),
                 ("dc5_E4", -1.0, "same, common trim from fit", "#0072B2", "*"),
                 ("S0", -1.0, "perfect reversal", "#56B4E9", "x")],
         "(b) other imperfections")):
    for i, (st, gs, lab, col, mk) in enumerate(cases):
        x, m, lo, hi = [], [], [], []
        for sd in SD:
            e = E(sd, st, gs)
            if len(e):
                x.append(sd * (1 + 0.06 * (i - len(cases) / 2))); m.append(np.median(e))
                lo.append(np.median(e) - np.percentile(e, 16)); hi.append(np.percentile(e, 84) - np.median(e))
        a.errorbar(x, m, yerr=[lo, hi], fmt=mk + "-", color=col, ms=4, lw=1, capsize=2, label=lab)
    a.axhline(1, color="#D55E00", ls="--", lw=0.8)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"static deflector error $\sigma_d$ (field and roll, rms)")
    a.set_ylabel(r"$E=\frac{1}{2}|\mathrm{odd}_+ + \mathrm{odd}_-|$ (units of target)")
    a.set_title(title, fontsize=8)
    a.legend(frameon=False, fontsize=6.3, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(os.path.join(BASE, f"fig_kusurlu_polarite.{e}"), dpi=200, bbox_inches="tight")
print("ok")
