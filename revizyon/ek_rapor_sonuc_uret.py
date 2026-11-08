#!/usr/bin/env python3
"""ek_rapor_sonuc_uret.py — ek_testler_edm.jsonl'dan ek_rapor_sonuclar.tex makrolarını üretir.
Yorum makroları (\\RhoYorum, \\RhoBesYorum, \\EdYorum) ek_rapor_yorumlar.tex'ten okunur (varsa)."""
import os, json
import numpy as np
REV = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(REV, "sonuclar", "ek_testler_edm.jsonl"))]


def C(kind, case):
    Cs, ev, extra = [], [], []
    for s in sorted({r["seed"] for r in rows}):
        rr = [r for r in rows if r["kind"] == kind and r["case"] == case and r["seed"] == s]
        f = {r["direction"]: r["f"] for r in rr}
        if len(f) == 2:
            Cs.append(abs(f[1] - f[-1]) / 2e-9); ev.append(abs(f[1] + f[-1]) / 2e-9); extra.append(rr[0])
    return np.array(Cs), np.array(ev), extra


def fmt(x):
    return f"{x:.2g}" if x < 0.1 else f"{x:.2f}" if x < 10 else f"{x:.0f}"


def rng_str(Cs):
    if len(Cs) == 0:
        return "bekleniyor"
    return f"{fmt(np.median(Cs))} [{fmt(Cs.min())}, {fmt(Cs.max())}]"


M = {}
for m, nm in (("E4", "Ceq"), ("E2", "Cet")):
    for rh, sfx in ((0.9, "nine"), (0.95, "nf")):
        Cs, _, _ = C("rhohat", f"{m}_{rh}")
        M[nm + sfx] = (rng_str(Cs) + (f" (n={len(Cs)})" if len(Cs) else ""))
for m, nm in (("E1", "RhoBirEbir"), ("E2", "RhoBirEiki"), ("E3", "RhoBirEuc"), ("E4", "RhoBirEdort")):
    Cs, _, _ = C("rho05", m)
    M[nm] = f"{rng_str(Cs)} & {int((Cs > 1).sum())}/{len(Cs)}" if len(Cs) else "bekleniyor & --"
for case, nm in (("Ed_zero", "EdSifir"), ("Ed_center", "EdMerkez")):
    Cs, ev, ex = C("Ed", case)
    if len(Cs):
        e_ = np.median([e['corr_eps_rms'] for e in ex]); r_ = np.median([e['corr_roll_rms'] for e in ex])
        M[nm + "Kuv"] = f"${e_/1e-6:.1f}$ / ${r_/1e-6:.1f}\\times10^{{-6}}$"
        ee = np.median(ev); ex_ = int(np.floor(np.log10(ee)))
        M[nm + "C"] = f"{rng_str(Cs)} & {int((Cs > 1).sum())}/{len(Cs)} & ${ee/10**ex_:.1f}\\times10^{{{ex_}}}$"
    else:
        M[nm + "Kuv"] = "--"; M[nm + "C"] = "bekleniyor & -- & --"
yor = os.path.join(REV, "ek_rapor_yorumlar.tex")
with open(os.path.join(REV, "ek_rapor_sonuclar.tex"), "w") as fo:
    fo.write("% otomatik: ek_rapor_sonuc_uret.py\n")
    for k, v in M.items():
        fo.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    if os.path.exists(yor):
        fo.write(open(yor).read())
    else:
        for k in ("RhoYorum", "RhoBesYorum", "EdYorum"):
            fo.write(f"\\newcommand{{\\{k}}}{{\\textit{{(C++ sonuçları bekleniyor.)}}}}\n")
print(open(os.path.join(REV, "ek_rapor_sonuclar.tex")).read())
