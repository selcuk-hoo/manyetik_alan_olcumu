#!/usr/bin/env python3
"""donmus_adim6.py — Adım 6 (zaman senaryosu) C++ noktalarının donmuş spin aracıyla, daha fazla seed'le
yeniden ölçümü.

Zaman serisi adim6_time.simulate ile aynıdır (ATL zemin hareketi A = 1e-5 μm²/(s·m), BBA artığı 0.3 μm,
BPM–kuadrupol göreli kayması 0.2 μm/√gün, BBA eşiği σ_δ* = 0.3 μm, 10 nm gürültü tabanı). Seçilen
saatlerdeki etkin kaçıklık m_eff = q − c iki yönde donmuş spin noktasında ölçülür; C = ½|f_CW − f_CCW|.
t = 0 bütün senaryolarda aynı (BBA artığı) olduğundan yalnız bir kez ölçülür.
"""
import os, sys, json, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

REV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REV)
from sahte_edm_donmus import measure_frozen

HOURS = [0, 24, 96, 168]


def task(t):
    seed, w, m, h, d, n_turns, n_iter = t
    import adim6_time as A6
    o = A6.simulate(seed, w, m, record_hours=[h])
    dx, dy = (np.array(a) for a in o["m_eff"][h])
    r = measure_frozen(dx, dy, direction=d, n_turns=n_turns, n_iter=n_iter)
    return dict(seed=seed, world=w, method=m, hour=h, direction=d, f=r["f"], f_target=r["f"] / 1e-9,
                w1=r["w1"], sym=float(np.mean([np.std(A6.A4.PS @ dx), np.std(A6.A4.PS @ dy)]) * 1e6))


def main(seeds, workers, ayar):
    import adim6_time as A6
    n_turns, n_iter = (14, 2) if ayar == "kisa" else (28, 4)
    out = os.path.join(REV, "sonuclar", f"donmus_adim6_{ayar}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["seed"], r["world"], r["method"], r["hour"], r["direction"]) for r in map(json.loads, open(out))}
    tasks = [(s, w, m, h, d, n_turns, n_iter) for s in seeds for w, m in A6.SCEN for h in HOURS
             for d in (-1, +1) if not (h == 0 and (w, m) != ("W1", "none")) and (s, w, m, h, d) not in done]
    print(f"{len(tasks)} görev (her biri 2 izleme)", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(task, tasks):
            open(out, "a").write(json.dumps(r) + "\n")
            print(f"seed {r['seed']} {r['world']}-{r['method']:8s} {r['hour']:3d} sa yön {r['direction']:+d}: "
                  f"f* = {r['f_target']:10.4f} × hedef (sim {r['sym']:.3f} μm)", flush=True)
    summarize(out)


def summarize(out):
    import adim6_time as A6
    rows = [json.loads(l) for l in open(out)]
    print(f"\n{os.path.basename(out)}: C = ½|f_CW − f_CCW| [× hedef]")
    for w, m in [("W1", "none")] + [s for s in A6.SCEN if s != ("W1", "none")]:
        for h in HOURS:
            ww, mm = ("W1", "none") if h == 0 else (w, m)
            if h == 0 and (w, m) != ("W1", "none"):
                continue
            Cs, sy = [], []
            for s in sorted(set(r["seed"] for r in rows)):
                rr = [r for r in rows if r["seed"] == s and r["world"] == ww and r["method"] == mm and r["hour"] == h]
                f = {r["direction"]: r["f"] for r in rr}
                if len(f) == 2:
                    Cs.append(abs(f[1] - f[-1]) / 2 / 1e-9); sy.append(rr[0]["sym"])
            if Cs:
                Cs = np.array(Cs)
                lab = "t=0 (BBA artığı)" if h == 0 else f"{w}-{m} {h} sa"
                print(f"  {lab:22s}: n={len(Cs):2d}  medyan {np.median(Cs):.4g}  [16–84%: {np.percentile(Cs,16):.3g}, "
                      f"{np.percentile(Cs,84):.3g}]  max {Cs.max():.3g}  hedef-üstü {int((Cs > 1).sum())}/{len(Cs)}  "
                      f"(sim medyan {np.median(sy):.3f} μm)")


def plot(out):
    """adim6_time.run_plot ile aynı üç panel; (b) donmuş spin sonuçlarıyla. → sonuclar/donmus_adim6.{pdf,png}"""
    import adim6_time as A6
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(l) for l in open(out)]
    E = []
    for key in sorted({(r["seed"], r["world"], r["method"], r["hour"]) for r in rows}):
        f = {r["direction"]: r["f"] for r in rows if (r["seed"], r["world"], r["method"], r["hour"]) == key}
        if len(f) == 2:
            E.append(dict(seed=key[0], world=key[1], method=key[2], hour=key[3], C=abs(f[1] - f[-1]) / 2 / 1e-9))
    nseed = len({e["seed"] for e in E})
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
    col = {"W1_E4": "#D55E00", "W1_E2": "#E69F00", "W0_E3": "#0072B2", "W1_bba_only": "#009E73",
           "W1_none": "#595959"}
    lab = {"W1_E4": "rigid BPM, two beams", "W1_E2": "rigid BPM, one beam",
           "W0_E3": "BPM on own stand, paper model", "W1_bba_only": "periodic BBA only",
           "W1_none": "no monitoring, no BBA"}
    S = json.load(open(os.path.join(A6.OUTDIR, "adim6_series.json")))
    for k, r in S["runs"].items():
        h = np.array(r["hours"])
        axs[0].semilogy(h, r["sym_med"], color=col[k], label=lab[k])
        axs[0].fill_between(h, r["sym_p16"], r["sym_p84"], color=col[k], alpha=0.12)
    axs[0].set_xlabel("time [h]"); axs[0].set_ylabel(r"symmetric residual [$\mu$m]")
    axs[0].set_title("(a) residual misalignment (linear, 20 seeds)", fontsize=8.5)
    h0, l0 = axs[0].get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    t0 = [e for e in E if e["hour"] == 0]
    for k in col:
        w, m = k.split("_", 1)
        pts = [e for e in E if e["world"] == w and e["method"] == m and e["hour"] > 0] + \
              [dict(e, world=w, method=m) for e in t0]
        hs = sorted(set(e["hour"] for e in pts))
        med = [np.median([e["C"] for e in pts if e["hour"] == hh]) for hh in hs]
        axs[1].semilogy(hs, med, "o-", color=col[k], ms=3)
        for e in pts:
            axs[1].scatter(e["hour"], max(e["C"], 1e-5), color=col[k], s=6, alpha=0.4)
    axs[1].axhline(1, color="#D55E00", ls="--", lw=1)
    axs[1].set_xlabel("time [h]"); axs[1].set_ylabel(r"false EDM $C$ [target]")
    axs[1].set_title(f"(b) false EDM (C++, frozen spin, {nseed} seeds)", fontsize=8.5)
    D = json.load(open(os.path.join(A6.OUTDIR, "adim6_detect.json")))
    for st, ls in ((1e-6, ":"), (10e-6, "-"), (100e-6, "--")):
        for fl, c in ((0.0, "#595959"), (10e-9, "#0072B2"), (100e-9, "#D55E00")):
            g = [x for x in D["grid"] if x["sig_turn"] == st and x["floor"] == fl]
            axs[2].loglog([x["T_avg"] for x in g], [x["min_E4"] * 1e6 for x in g], ls, color=c, lw=1,
                          label=f"{st*1e6:.0f} $\\mu$m/turn, floor {fl*1e9:.0f} nm")
    axs[2].set_xlabel(r"averaging time $T_{\rm avg}$ [s]")
    axs[2].set_ylabel(r"min. detectable sym. drift (3$\sigma$) [$\mu$m]")
    axs[2].set_title("(c) two-beam detection limit", fontsize=8.5)
    axs[2].legend(fontsize=5.5, ncol=1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(REV, "sonuclar", f"donmus_adim6.{ext}"), dpi=200, bbox_inches="tight")
    print("donmus_adim6.pdf/png yazıldı")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--ayar", choices=["kisa", "tam"], default="kisa")
    ap.add_argument("--ozet", action="store_true")
    ap.add_argument("--cizim", action="store_true")
    a = ap.parse_args()
    if a.cizim:
        plot(os.path.join(REV, "sonuclar", f"donmus_adim6_{a.ayar}.jsonl"))
    elif a.ozet:
        summarize(os.path.join(REV, "sonuclar", f"donmus_adim6_{a.ayar}.jsonl"))
    else:
        main(a.seeds, a.workers, a.ayar)
