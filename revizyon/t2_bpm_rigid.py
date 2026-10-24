#!/usr/bin/env python3
"""t2_bpm_rigid.py — Tartışma testi T2: BPM kuadrupole rijit bağlı (ρ=1), göreli kayma δ.

δ = (BPM elektriksel merkezi) − (kuadrupol manyetik merkezi) değişimi, BBA kalibrasyonundan beri.
Kaynakları: BPM–kuadrupol bağlantısının mekanik kayması + BPM elektroniğinin ofset kayması
(ikisi okumaya aynı biçimde girer, ayrılamaz). Her BPM için bağımsız, rms σ_δ.

Okuma (rijit bağlı BPM): Δr_b = R_b Δm − (Δm + δ) + n_b = (R_b − I) Δm − δ + n_b
Kestiriciler (BPM'in kuadrupolle hareket ettiğini bilen model, R − I):
  E2: tek demet  pinv(R_ccw − I)
  E4: iki demet  pinv([R_ccw − I; R_cw − I])
  E3: iki demet, makale modeli (BPM sabit sanır) — karşılaştırma
Düzeltme: kuadrupol içi manyetik düzeltici (Mq) → etkin artık m' = Δm − m̂.

Modlar: recon (doğrusal, 200 seed) | edm (C++, seçili σ_δ) | plot
"""
import os, sys, json, argparse
import numpy as np

REV = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(REV)
for p in (REV, BASE, os.path.join(BASE, "kmod_drivers"), os.path.join(BASE, "berry_data")):
    sys.path.insert(0, p)
import adim1_bpm_model as A1

OUTDIR = A1.OUTDIR
NQ = 48
I = np.eye(NQ)
RC = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5]
SIG_D = [0.0, 0.01e-6, 0.03e-6, 0.1e-6, 0.3e-6, 1e-6, 3e-6]
TARGET = 1e-9


def mats(pl):
    Rc, Rw = A1.R[pl]["ccw"], A1.R[pl]["cw"]
    return {"E2": (Rc - I, "one"), "E4": (np.vstack([Rc - I, Rw - I]), "two"),
            "E3": (np.vstack([Rc, Rw]), "two")}


_P = {}


def pinv(pl, k, rc):
    key = (pl, k, rc)
    if key not in _P:
        M = mats(pl)[k][0]
        _P[key] = np.linalg.pinv(M, rcond=rc)
    return _P[key]


def case(seed, pl, drift, sig_d, sig_n):
    rng = np.random.default_rng(60_000 + 2 * seed + (pl == "y"))
    dm = A1.PS @ rng.normal(0, 1, NQ); dm *= drift / np.std(dm)
    delta = rng.normal(0, 1, NQ) * sig_d
    n1 = rng.normal(0, sig_n, NQ); n2 = rng.normal(0, sig_n, NQ)
    Rc, Rw = A1.R[pl]["ccw"], A1.R[pl]["cw"]
    r1 = (Rc - I) @ dm - delta + n1; r2 = (Rw - I) @ dm - delta + n2
    return dm, r1, r2


def residual(seed, pl, drift, sig_d, sig_n, k, rc):
    dm, r1, r2 = case(seed, pl, drift, sig_d, sig_n)
    y = r1 if mats(pl)[k][1] == "one" else np.concatenate([r1, r2])
    return dm - pinv(pl, k, rc) @ y


def run_recon(nseed=200):
    out = {"params": {"nseed": nseed, "sig_d_m": SIG_D}, "rows": []}
    for drift in (1e-6, 5e-6):
        for sig_n in (14e-9, 1e-6):
            for sd in SIG_D:
                row = {"drift": drift, "sig_n": sig_n, "sig_d": sd}
                for k in ("E2", "E4", "E3"):
                    best, bv = None, None
                    for rc in RC:
                        tot = [np.std(residual(s, pl, drift, sd, sig_n, k, rc))
                               for s in range(nseed) for pl in "xy"]
                        if bv is None or np.median(tot) < bv:
                            best, bv = rc, np.median(tot)
                    sym = [np.std(A1.PS @ residual(s, pl, drift, sd, sig_n, k, best))
                           for s in range(nseed) for pl in "xy"]
                    anti = [np.std(A1.PA @ residual(s, pl, drift, sd, sig_n, k, best))
                            for s in range(nseed) for pl in "xy"]
                    row[k] = {"rcond": best, "sym_um": float(np.median(sym) * 1e6),
                              "sym_p16": float(np.percentile(sym, 16) * 1e6),
                              "sym_p84": float(np.percentile(sym, 84) * 1e6),
                              "anti_um": float(np.median(anti) * 1e6)}
                out["rows"].append(row)
                print(f"sürüklenme {drift*1e6:.0f} μm, gürültü {sig_n*1e9:5.0f} nm, σ_δ {sd*1e6:5.2f} μm | "
                      + " | ".join(f"{k} sim {row[k]['sym_um']:.3f} anti {row[k]['anti_um']:.3f}"
                                   for k in ("E2", "E4", "E3")), flush=True)
    json.dump(out, open(os.path.join(OUTDIR, "t2_recon.json"), "w"), indent=1)


EDM_JSONL = os.path.join(OUTDIR, "t2_edm.jsonl")
EDM_SD = [0.1e-6, 0.3e-6, 1e-6]


def _best_rc(drift, sig_n, sd, k):
    for r in json.load(open(os.path.join(OUTDIR, "t2_recon.json")))["rows"]:
        if (r["drift"], r["sig_n"], r["sig_d"]) == (drift, sig_n, sd):
            return r[k]["rcond"]
    raise KeyError


def _edm_task(task):
    import tempfile
    os.chdir(tempfile.mkdtemp())
    from fast_est import fast_measure
    seed, sd, k = task
    drift, sig_n = 5e-6, 14e-9
    rc = _best_rc(drift, sig_n, sd, k)
    dx = residual(seed, "x", drift, sd, sig_n, k, rc)
    dy = residual(seed, "y", drift, sd, sig_n, k, rc)
    fcw, _ = fast_measure(dx, dy, direction=+1, signed=True, n_turns=28, n_iter=4)
    fccw, _ = fast_measure(dx, dy, direction=-1, signed=True, n_turns=28, n_iter=4)
    return dict(seed=seed, sig_d=sd, method=k, rcond=rc, f_cw=fcw / TARGET, f_ccw=fccw / TARGET,
                C=abs(fcw - fccw) / 2 / TARGET,
                sym_x=float(np.std(A1.PS @ dx) * 1e6), sym_y=float(np.std(A1.PS @ dy) * 1e6))


def run_edm(seeds, workers):
    from concurrent.futures import ProcessPoolExecutor
    done = set()
    if os.path.exists(EDM_JSONL):
        done = {(d["seed"], d["sig_d"], d["method"]) for d in map(json.loads, open(EDM_JSONL))}
    tasks = [(s, sd, k) for s in seeds for sd in EDM_SD for k in ("E4", "E2")
             if (s, sd, k) not in done]
    print(f"{len(tasks)} görev", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(_edm_task, tasks):
            open(EDM_JSONL, "a").write(json.dumps(r) + "\n")
            print(f"  seed {r['seed']} σ_δ {r['sig_d']*1e6:.1f} μm {r['method']}: C={r['C']:.4f}× "
                  f"(sim {r['sym_x']:.3f}/{r['sym_y']:.3f} μm)", flush=True)


def run_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    d = json.load(open(os.path.join(OUTDIR, "t2_recon.json")))
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    col = {"E2": "#E69F00", "E4": "#D55E00", "E3": "#0072B2"}
    lab = {"E2": "one beam, quad-referenced model", "E4": "two beams, quad-referenced model",
           "E3": "two beams, paper model (fixed BPM)"}
    for ax, sig_n in zip(axs, (14e-9, 1e-6)):
        for drift, ls in ((5e-6, "-"), (1e-6, ":")):
            rows = [r for r in d["rows"] if r["sig_n"] == sig_n and r["drift"] == drift and r["sig_d"] > 0]
            sd = np.array([r["sig_d"] for r in rows]) * 1e6
            for k in ("E2", "E4", "E3"):
                ax.loglog(sd, [r[k]["sym_um"] for r in rows], ls, marker="o", ms=3, color=col[k],
                          label=f"{lab[k]}, drift {drift*1e6:.0f} $\\mu$m")
        ax.loglog([0.01, 3], [0.01, 3], "k--", lw=0.8, alpha=0.6)
        ax.text(0.012, 0.016, r"error $=\sigma_\delta$", fontsize=7)
        ax.set_xlabel(r"BPM–quadrupole relative drift $\sigma_\delta$ [$\mu$m]")
        ax.set_title(f"BPM noise {'14 nm' if sig_n < 1e-7 else '1 $\\mu$m'}", fontsize=9)
        ax.grid(alpha=0.3, which="both", lw=0.4)
    axs[0].set_ylabel(r"symmetric residual [$\mu$m]")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, -0.2))
    fig.suptitle("BPM rigidly mounted on quadrupole (ρ = 1); 200 seeds, median", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"t2.{ext}"), dpi=200, bbox_inches="tight")
    print("t2.pdf/png yazıldı")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["recon", "edm", "plot"])
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    {"recon": run_recon, "edm": lambda: run_edm(a.seeds, a.workers), "plot": run_plot}[a.mode]()
