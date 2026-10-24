#!/usr/bin/env python3
"""adim5_harmonics.py — PRAB revizyonu Adım 5: harmonik başına yörünge kazancı ve spin duyarlılığı.

Desenler (hücre harmoniği m = 0..12, k = hücre indisi 0..23):
  sym  : v[2k] = v[2k+1] = cos(2π m k / 24 + ψ)        (QF, QD aynı yönde)
  anti : v[2k] = −v[2k+1] = cos(2π m k / 24 + ψ)       (QF, QD zıt yönde)
Beklenti: QF/QD gradyanları zıt işaretli → simetrik kaçıklığın tekmeleri 48 kuadrupol boyunca
işaret değiştirir → yörünge harmoniği 24 − m; antisimetrik → m. m = 12'de ikisi aynı (12).

Yörünge kazancı: ‖R v‖/‖v‖ (C++ ile ölçülmüş R; BPM kuadrupol girişinde, lab-sabit),
  ayrıca CW+CCW toplamı, farkı ve rijit-bağlı BPM okuması (R − I).
Sahte EDM (bilineer, dx ve dy aynı harmonik ve aynı tür):
  inphase : dx = dy = A·cos(mθ)          (x ∝ y)
  quad    : dx = A·cos(mθ), dy = A·sin(mθ)   (dik faz; m = 0, 12'de sin ≡ 0 → tanımsız)
  A: desen rms'i 5 μm; sonuç f / (5 μm)² [hedef / μm²]. C = ½|f_CW − f_CCW|, çift = ½(f_CW + f_CCW).

Modlar: gain | edm | plot
"""
import os, sys, json, argparse
import numpy as np

REV = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(REV)
for p in (REV, BASE, os.path.join(BASE, "kmod_drivers"), os.path.join(BASE, "berry_data")):
    sys.path.insert(0, p)
OUTDIR = os.path.join(REV, "sonuclar")
NQ = 48
TARGET = 1e-9
AMP = 5e-6
MS = list(range(13))


def pattern(m, kind, phase):
    k = np.arange(24)
    c = np.cos(2 * np.pi * m * k / 24 + phase)
    v = np.zeros(NQ)
    v[0::2] = c
    v[1::2] = c if kind == "sym" else -c
    return v


def unit(v):
    """rms (ortalama dahil) ile normalize; m=0 simetrik desen sabittir (std=0 ama rms=1)."""
    s = np.sqrt(np.mean(v ** 2))
    return v / s if s > 1e-12 else None


def load_R():
    return {pl: {t: np.load(os.path.join(BASE, f"R_{k}_{'1' if t=='ccw' else 'cw'}.npy"))
                 for t in ("ccw", "cw")} for pl, k in (("x", "dx"), ("y", "dy"))}


def orbit_harmonic(y):
    """BPM okumasının baskın azimutal harmoniği (48 nokta, 0..24)."""
    F = np.abs(np.fft.rfft(y))
    return int(np.argmax(F))


def run_gain():
    R = load_R(); I = np.eye(NQ); out = []
    for m in MS:
        for kind in ("sym", "anti"):
            row = {"m": m, "kind": kind}
            for pl in ("x", "y"):
                g = {"ccw": [], "cw": [], "sum": [], "diff": [], "rigid_ccw": []}
                harm = []
                for ph in (0.0, np.pi / 2):
                    v = unit(pattern(m, kind, ph))
                    if v is None:
                        continue
                    a, b = R[pl]["ccw"] @ v, R[pl]["cw"] @ v
                    n = np.linalg.norm(v)
                    g["ccw"].append(np.linalg.norm(a) / n); g["cw"].append(np.linalg.norm(b) / n)
                    g["sum"].append(np.linalg.norm(a + b) / n); g["diff"].append(np.linalg.norm(a - b) / n)
                    g["rigid_ccw"].append(np.linalg.norm(a - v) / n)
                    harm.append(orbit_harmonic(a))
                row[pl] = {k: float(np.mean(v_)) for k, v_ in g.items()}
                row[pl]["orbit_harmonic"] = harm
            out.append(row)
            print(f"m={m:2d} {kind:4s} | y: kazanç CCW {row['y']['ccw']:.3f}, toplam {row['y']['sum']:.3f}, "
                  f"fark {row['y']['diff']:.3f}, rijit {row['y']['rigid_ccw']:.3f}, "
                  f"yörünge harmoniği {row['y']['orbit_harmonic']} | x: CCW {row['x']['ccw']:.3f}")
    json.dump(out, open(os.path.join(OUTDIR, "adim5_gain.json"), "w"), indent=1)


JSONL = os.path.join(OUTDIR, "adim5_edm.jsonl")


def configs():
    cfg = []
    for m in MS:
        for kind in ("sym", "anti"):
            cfg.append((m, kind, "inphase"))
            if m not in (0, 12):
                cfg.append((m, kind, "quad"))
    return cfg


def _task(task):
    import tempfile
    os.chdir(tempfile.mkdtemp())
    from fast_est import fast_measure
    m, kind, pairing = task
    dx = unit(pattern(m, kind, 0.0)) * AMP
    dy = unit(pattern(m, kind, 0.0 if pairing == "inphase" else -np.pi / 2)) * AMP   # cos → sin
    fcw, _ = fast_measure(dx, dy, direction=+1, signed=True, n_turns=28, n_iter=4)
    fccw, _ = fast_measure(dx, dy, direction=-1, signed=True, n_turns=28, n_iter=4)
    s = (AMP * 1e6) ** 2
    return dict(m=m, kind=kind, pairing=pairing, f_cw=fcw / TARGET, f_ccw=fccw / TARGET,
                C=abs(fcw - fccw) / 2 / TARGET, even=(fcw + fccw) / 2 / TARGET,
                C_per_um2=abs(fcw - fccw) / 2 / TARGET / s, fmax_per_um2=max(abs(fcw), abs(fccw)) / TARGET / s)


def run_edm(workers):
    from concurrent.futures import ProcessPoolExecutor
    done = set()
    if os.path.exists(JSONL):
        done = {(d["m"], d["kind"], d["pairing"]) for d in map(json.loads, open(JSONL))}
    tasks = [c for c in configs() if c not in done]
    print(f"{len(tasks)} görev", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(_task, tasks):
            open(JSONL, "a").write(json.dumps(r) + "\n")
            print(f"  m={r['m']:2d} {r['kind']:4s} {r['pairing']:7s}: C/μm² = {r['C_per_um2']:.4g}, "
                  f"max|f|/μm² = {r['fmax_per_um2']:.4g}", flush=True)


def run_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    G = json.load(open(os.path.join(OUTDIR, "adim5_gain.json")))
    E = [json.loads(l) for l in open(JSONL)] if os.path.exists(JSONL) else []
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4))
    col = {"sym": "#0072B2", "anti": "#D55E00"}
    for pl, ax in zip(("y", "x"), axs):
        for kind in ("sym", "anti"):
            rows = [r for r in G if r["kind"] == kind]
            m = [r["m"] for r in rows]
            ax.semilogy(m, [r[pl]["ccw"] for r in rows], "-o", color=col[kind], ms=3,
                        label=f"orbit gain, {kind} (one beam)")
            ax.semilogy(m, [r[pl]["sum"] for r in rows], ":", color=col[kind], lw=1,
                        label=f"orbit gain, {kind} (CW+CCW sum)")
        ax.set_xlabel("cell harmonic $m$")
        ax.set_ylabel(r"orbit gain $\|Rv\|/\|v\|$")
        ax.set_title(f"({'a' if pl=='y' else 'b'}) {'vertical' if pl=='y' else 'horizontal'} orbit; "
                     "false EDM on right axis", fontsize=8.5)
        if E and pl == "y":
            ax2 = ax.twinx()
            for kind, mk in (("sym", "s"), ("anti", "^")):
                for pairing, fill in (("inphase", "none"), ("quad", None)):
                    rr = sorted([r for r in E if r["kind"] == kind and r["pairing"] == pairing],
                                key=lambda r: r["m"])
                    if not rr:
                        continue
                    ax2.semilogy([r["m"] for r in rr], [max(r["C_per_um2"], 1e-6) for r in rr],
                                 mk + "--", color=col[kind], ms=5, mfc=fill or col[kind], lw=0.8,
                                 label=f"false EDM C, {kind}, {'in-phase' if pairing=='inphase' else 'quadrature'}")
            ax2.set_ylabel(r"false EDM $C$ per $\mu$m$^2$ [target]")
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, fontsize=6.3,
                       bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"adim5.{ext}"), dpi=200, bbox_inches="tight")
    print("adim5.pdf/png yazıldı")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gain", "edm", "plot"])
    ap.add_argument("-w", "--workers", type=int, default=4)
    a = ap.parse_args()
    {"gain": run_gain, "edm": lambda: run_edm(a.workers), "plot": run_plot}[a.mode]()
