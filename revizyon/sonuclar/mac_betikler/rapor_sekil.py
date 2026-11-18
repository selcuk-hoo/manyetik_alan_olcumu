#!/usr/bin/env python3
"""Mac koşu raporu için şekil + LaTeX tablo parçaları (veri: revizyon/sonuclar/*)."""
import json, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/Users/selcuk/works/manyetik_alan_olcumu/revizyon/sonuclar"
OUT = os.path.dirname(os.path.abspath(__file__))
CH = ["ss", "sa", "as", "aa"]
RENK = {"ss": "#2a78d6", "sa": "#eb6834", "as": "#1baf7a", "aa": "#eda100"}   # doğrulanmış kategorik sıra
ISARET = {"ss": "o", "sa": "s", "as": "D", "aa": "^"}
METIN, IKINCIL, ZEMIN = "#0b0b0b", "#52514e", "#fcfcfb"

kan = {}
for r in map(json.loads, open(os.path.join(R, "bba_gecis_kanallar.jsonl"))):
    kan[(r["pass_"], r["channel"])] = r["f_target"]
gecis = sorted({p for p, _ in kan if all((p, c) in kan for c in CH + ["full"])})
st = json.load(open(os.path.join(R, "bba_iter_cpp_donmus_state.json")))
hist = {h["pass"]: h for h in st["hist"]}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": IKINCIL, "axes.labelcolor": METIN,
                     "xtick.color": IKINCIL, "ytick.color": IKINCIL, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": ZEMIN, "axes.facecolor": ZEMIN})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
x = np.array(gecis)
full = np.array([kan[(p, "full")] for p in gecis])
a1.plot(x, np.abs(full), "-", color="#2a78d6", lw=2, zorder=2)
for p, v in zip(x, full):
    a1.plot(p, abs(v), "o", ms=7, mfc="#2a78d6" if v > 0 else ZEMIN, mec="#2a78d6", mew=1.5, zorder=3)
a1.axhline(1, color=IKINCIL, lw=1, ls="--")
a1.text(x[-1], 1.15, "hedef (1 nrad/s)", ha="right", va="bottom", color=IKINCIL, fontsize=8)
a1.set_yscale("log"); a1.set_xlabel("BBA geçişi"); a1.set_ylabel("|sahte EDM| / hedef")
a1.set_title("(a) Toplam sahte EDM", loc="left", fontsize=9, color=METIN)
a1.grid(axis="y", color="#e4e3df", lw=0.6); a1.set_axisbelow(True)
for c in CH:
    v = np.array([kan[(p, c)] for p in gecis])
    a2.plot(x, np.abs(v), "-", color=RENK[c], lw=2, zorder=2)
    for p, w in zip(x, v):
        a2.plot(p, abs(w), ISARET[c], ms=6, mfc=RENK[c] if w > 0 else ZEMIN, mec=RENK[c], mew=1.5, zorder=3)
    a2.annotate(c, (x[-1], abs(v[-1])), xytext=(6, 0), textcoords="offset points", va="center",
                color=METIN, fontsize=8)
a2.axhline(1, color=IKINCIL, lw=1, ls="--")
a2.set_yscale("log"); a2.set_xlabel("BBA geçişi")
a2.set_title("(b) Kanallar (dx × dy)", loc="left", fontsize=9, color=METIN)
a2.grid(axis="y", color="#e4e3df", lw=0.6); a2.set_axisbelow(True)
for a in (a1, a2):
    a.set_xticks(x)
a2.set_xlim(x[0] - 0.3, x[-1] + 1.0)
fig.text(0.01, 0.005, "Dolu işaret: pozitif, boş işaret: negatif değer.", fontsize=7.5, color=IKINCIL)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(OUT, "bba_kanallar.pdf")); fig.savefig(os.path.join(OUT, "bba_kanallar.png"), dpi=150)

# LaTeX tabloları
with open(os.path.join(OUT, "tablo_kanal.tex"), "w") as t:
    for p in gecis:
        v = [kan[(p, c)] for c in CH]
        t.write(f"{p} & " + " & ".join(f"${w:+.2f}$" for w in v) +
                f" & ${kan[(p, 'full')]:+.2f}$ & ${sum(v):+.2f}$ & {sum(map(abs, v)) / max(abs(kan[(p, 'full')]), 1e-9):.1f} \\\\\n")
with open(os.path.join(OUT, "tablo_bba.tex"), "w") as t:
    for p, h in sorted(hist.items()):
        t.write(f"{p} & {h['f']/1e-9:.2f} & {h['sym_dx']*1e6:.2f} & {h['sym_dy']*1e6:.3f} & {h['orbit_um']:.1f} & "
                f"{h['minqual_x']:.2f} / {h['minqual_y']:.2f} \\\\\n")
yo = collections.defaultdict(list)
for r in map(json.loads, open(os.path.join(R, "makale_yeniden_olcum.jsonl"))):
    yo[(r["kind"], round(r["sigma"] * 1e6, 1), r["channel"], r["direction"])].append(r["f_target"])
with open(os.path.join(OUT, "tablo_yo.tex"), "w") as t:
    for k in sorted(yo, key=lambda k: (k[0], k[1], k[2], k[3])):
        v = np.array(yo[k])
        t.write(f"{k[0]} & {k[1]:g} & {k[2]} & ${k[3]:+d}$ & {len(v)} & ${v.mean():+.3f}$ & {np.sqrt((v**2).mean()):.3f} \\\\\n")
print("geçişler:", gecis)
