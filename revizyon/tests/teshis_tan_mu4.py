#!/usr/bin/env python3
"""teshis_tan_mu4.py — Simetrik/antisimetrik yörünge kazanç oranı için analitik yasa testi.

İddia (türetme, ince mercek FODO): düşük harmonikte G_sym/G_anti ≈ (√βF−√βD)/(√βF+√βD) = tan(μ/4),
μ = hücre başına faz ilerlemesi. Burada analitik örgü (drift_monitor/fodo_lattice.py, C++ yok) ile
gradyan g taranır; her g'de Courant–Snyder tepki matrisi kurulur, harmonik m=1..4 simetrik (QF,QD aynı
yön) ve antisimetrik (zıt yön) desenler için kapalı yörünge rms oranı ölçülür.
Yatay düzlemde ark odaklaması nominal Q_x=2.6824'e bir kez kalibre edilip sabit tutulur.
Çıktı: sonuclar/teshis_tan_mu4.json, ../fig_tan_mu4.{png,pdf}
"""
import os, sys, json
import numpy as np
REV = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); BASE = os.path.dirname(REV)
sys.path.insert(0, os.path.join(BASE, "drift_monitor"))
import fodo_lattice as FL

cfg0 = json.load(open(os.path.join(BASE, "params.json")))
nF = int(cfg0["nFODO"]); c = np.arange(nF)
Kx = FL.calibrate_K_x_arc(cfg0)


def ratio(cfg, plane, m):
    b, ph, Q = FL.compute_twiss_at_quads(cfg, plane, K_x_arc=Kx if plane == "x" else None)
    R = FL.build_response_matrix(b, ph, Q, FL.signed_KL(cfg, plane))
    out = []
    for f in (np.cos, np.sin):
        w = f(2 * np.pi * m * c / nF)
        s = np.repeat(w, 2); a = s * np.tile([1, -1], nF)
        out.append((np.sum((R @ s) ** 2), np.sum((R @ a) ** 2)))
    gs = np.sqrt(sum(o[0] for o in out) / sum(o[1] for o in out))
    mu = 2 * np.pi * Q / nF
    sb = np.sqrt(b[:2])
    return dict(Q=Q, mu_deg=np.degrees(mu), ratio=gs, beta_formula=abs(sb[0] - sb[1]) / (sb[0] + sb[1]),
                tan_mu4=np.tan(mu / 4), bF=b[0], bD=b[1])


rows = []
for g in np.round(np.arange(0.04, 1.2, 0.02), 3):
    cfg = dict(cfg0); cfg["g1"] = float(g)
    for pl in ("x", "y"):
        try:
            for m in (1, 2, 3, 4):
                r = ratio(cfg, pl, m)
                if not np.isfinite(r["Q"]):
                    raise ValueError
                rows.append(dict(g=float(g), plane=pl, m=m, **{k: float(v) for k, v in r.items()}))
        except Exception:
            pass
json.dump(rows, open(os.path.join(REV, "sonuclar", "teshis_tan_mu4.json"), "w"), indent=0)
print(" g    düz  m   Q     μ(°)   oran   √β-formül  tan(μ/4)")
for r in rows:
    if r["m"] in (1, 3) and (abs(r["g"] - 0.21) < 1e-9 or abs(r["g"] * 10 % 1) < 1e-9):
        print(f"{r['g']:.2f}  {r['plane']}  {r['m']}  {r['Q']:.3f} {r['mu_deg']:6.1f}  {r['ratio']:.3f}   {r['beta_formula']:.3f}     {r['tan_mu4']:.3f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
for k, pl in enumerate(("x", "y")):
    a = ax[k]
    for m, col in zip((1, 2, 3, 4), ("#0072B2", "#009E73", "#E69F00", "#D55E00")):
        R = [r for r in rows if r["plane"] == pl and r["m"] == m]
        a.plot([r["mu_deg"] for r in R], [r["ratio"] for r in R], "o", ms=2.5, color=col, label=f"Courant-Snyder response, m={m}")
    R = sorted([r for r in rows if r["plane"] == pl and r["m"] == 1], key=lambda r: r["mu_deg"])
    a.plot([r["mu_deg"] for r in R], [r["beta_formula"] for r in R], "k-", lw=1, label=r"$(\sqrt{\beta_F}-\sqrt{\beta_D})/(\sqrt{\beta_F}+\sqrt{\beta_D})$")
    mu = np.linspace(1, 175, 200); a.plot(mu, np.tan(np.radians(mu) / 4), "k:", lw=1, label=r"$\tan(\mu/4)$")
    r0 = [r for r in rows if r["plane"] == pl and abs(r["g"] - 0.21) < 1e-9 and r["m"] == 1]
    if r0: a.axvline(r0[0]["mu_deg"], color="grey", lw=0.8, ls="--")
    a.set_xlabel(r"phase advance per cell $\mu$ (deg)"); a.set_ylabel(r"$G_\mathrm{sym}/G_\mathrm{anti}$")
    a.set_title({"x": "horizontal", "y": "vertical"}[pl], fontsize=8)
    if k == 0: a.legend(frameon=False, fontsize=6)
fig.tight_layout()
for e in ("png", "pdf"): fig.savefig(os.path.join(BASE, f"fig_tan_mu4.{e}"), bbox_inches="tight", dpi=200)
