#!/usr/bin/env python3
"""makale_sekil_kanallar.py — Şekil 2 (dört sahte-EDM kanalı, donmuş spin, CCW) → fig_v2_channels.{png,pdf}
Veri: revizyon/sonuclar/makale_yeniden_olcum.jsonl (kind=ch; σ=10 μm 3 seed; σ=2.5,5,10 μm seed 0)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REV = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(REV)
R = [json.loads(l) for l in open(os.path.join(REV, "sonuclar", "makale_yeniden_olcum.jsonl"))]
R = [r for r in R if r["kind"] == "ch"]
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})
CH = ["ss", "sa", "as", "aa"]; LAB = ["ss", "sa", "as", "aa", "full"]
col = dict(ss="#0072B2", sa="#009E73", **{"as": "#E69F00"}, aa="#D55E00", full="#595959")
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
a = ax[0]
for i, c in enumerate(LAB):
    v = [abs(r["f_target"]) for r in R if r["sigma"] == 1e-5 and r["channel"] == c]
    a.bar(i, np.sqrt(np.mean(np.square(v))), color=col[c], alpha=.35, width=.6)
    a.plot([i] * len(v), v, "o", ms=3, color=col[c])
a.set_yscale("log"); a.set_xticks(range(5)); a.set_xticklabels(LAB)
a.set_ylabel(r"$|f|$ (units of target)"); a.set_title(r"$\sigma=10\,\mu$m, 3 patterns (bars: rms)", fontsize=8)
a.text(-0.16, 1.04, "(a)", transform=a.transAxes, fontweight="bold")
b = ax[1]; sg = np.array([2.5, 5, 10])
for c in CH:
    y = [abs(next(r["f_target"] for r in R if r["seed"] == 0 and r["channel"] == c and abs(r["sigma"] - s * 1e-6) < 1e-12)) for s in sg]
    b.loglog(sg, y, "o-", color=col[c], label=c, ms=3)
b.loglog(sg, 0.4 * (sg / 2.5) ** 2 * 8, "k:", lw=1, label=r"$\propto\sigma^2$")
b.set_xlabel(r"misalignment $\sigma$ ($\mu$m)"); b.set_ylabel(r"$|f|$ (units of target)")
b.set_title("seed 0", fontsize=8); b.legend(frameon=False, fontsize=7)
b.text(-0.16, 1.04, "(b)", transform=b.transAxes, fontweight="bold")
fig.tight_layout()
for e in ("png", "pdf"): fig.savefig(os.path.join(BASE, f"fig_v2_channels.{e}"), bbox_inches="tight")
print("ok")
