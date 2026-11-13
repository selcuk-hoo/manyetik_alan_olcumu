#!/usr/bin/env python3
"""makale_yeniden_olcum.py — makale revizyonu: eski prosedürle ölçülmüş şekillerin donmuş spin aracıyla
yeniden ölçümü (kısaltılmış ayar 14/2).

Görev türleri (kurulumlar makaledeki üreticilerle aynı):
  ch    : Şekil 2 (kanallar) — fig_channels_gen._pattern(seed, σ) (rng(seed), dx sonra dy, [0]=0),
          f_ss/f_sa/f_as/f_aa/full, CCW (makaledeki gibi tek yön). (a) σ=10 μm, seed 0–2;
          (b) seed 0, σ = 2.5, 5 μm.
  rawcw : Şekil 4 'raw' — paper_runs 'sigma' desenleri (rng 3000+seed), σ = 2.5, 5, 10 μm, seed 0–2, CW
          (CCW yönü donmus_sigma.json'da, tam ayar) → C = ½|f_CW − f_CCW|.
  oc1   : Şekil 4 'orbit corr.' — twobeam_oc kurulumu, tek demet (R_CCW, rcond 0.01, 14 nm, nominal),
          σ = 5, 20 μm, seed 0–2, iki yön (σ = 10 μm donmus_fig4.jsonl'dan).
  tb20  : Şekil 4 elması — twobeam_oc_14nm_g20 kurulumu (makine %0.2 gradyan hatası, model nominal,
          14 nm, rcond 0.01), seed 0–14, iki yön, izleme dG ile.
"""
import os, sys, json, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

REV = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(REV)
sys.path[:0] = [REV, BASE]
OUT = os.path.join(REV, "sonuclar", "makale_yeniden_olcum.jsonl")
NQ = 48
RC = 0.01


def projectors():
    Ps = np.zeros((NQ, NQ))
    for k in range(NQ // 2):
        a, b = 2 * k, 2 * k + 1
        Ps[a, a] = Ps[b, b] = Ps[a, b] = Ps[b, a] = 0.5
    return Ps, np.eye(NQ) - Ps


def R(pl, d, g20=False):
    if g20:
        from build_rtrue import path
        return np.load(path("d" + pl, d, 0.002))
    return np.load(os.path.join(BASE, f"R_d{pl}_{'1' if d == 'ccw' else 'cw'}.npy"))


def oc_pattern(seed, sig):
    rng = np.random.default_rng(seed)
    m_dx = rng.normal(0, sig, NQ); m_dx[0] = 0.0
    m_dy = rng.normal(0, sig, NQ); m_dy[0] = 0.0
    nr = np.random.default_rng(50000 + seed)
    n = {k: nr.normal(0, 14e-9, NQ) for k in ("xa", "xb", "ya", "yb")}
    return m_dx, m_dy, n


def residual(kind, seed, sig):
    m_dx, m_dy, n = oc_pattern(seed, sig)
    pinv = lambda M: np.linalg.pinv(M, rcond=RC)
    if kind == "oc1":
        cx = m_dx - pinv(R("x", "ccw")) @ (R("x", "ccw") @ m_dx + n["xa"])
        cy = m_dy - pinv(R("y", "ccw")) @ (R("y", "ccw") @ m_dy + n["ya"])
        return cx, cy
    # tb20: gerçek makine g20, model nominal
    Sx = np.vstack([R("x", "ccw"), R("x", "cw")]); Sy = np.vstack([R("y", "ccw"), R("y", "cw")])
    yx = np.concatenate([R("x", "ccw", True) @ m_dx + n["xa"], R("x", "cw", True) @ m_dx + n["xb"]])
    yy = np.concatenate([R("y", "ccw", True) @ m_dy + n["ya"], R("y", "cw", True) @ m_dy + n["yb"]])
    return m_dx - pinv(Sx) @ yx, m_dy - pinv(Sy) @ yy


def tasks_all():
    T = []
    for s in range(3):
        for ch in ("ss", "sa", "as", "aa", "full"):
            T.append(("ch", s, 10e-6, ch, -1))
    for sg in (2.5e-6, 5e-6):
        for ch in ("ss", "sa", "as", "aa"):
            T.append(("ch", 0, sg, ch, -1))
    for sg in (2.5e-6, 5e-6, 10e-6):
        for s in range(3):
            T.append(("rawcw", s, sg, "-", +1))
    for sg in (5e-6, 20e-6):
        for s in range(3):
            for d in (-1, 1):
                T.append(("oc1", s, sg, "-", d))
    for s in range(15):
        for d in (-1, 1):
            T.append(("tb20", s, 10e-6, "-", d))
    return T


def task(t):
    kind, seed, sig, ch, d = t
    from sahte_edm_donmus import measure_frozen
    dG = None
    if kind == "ch":
        Ps, Pa = projectors()
        rng = np.random.default_rng(seed)
        dx = rng.normal(0.0, sig, NQ); dx[0] = 0.0
        dy = rng.normal(0.0, sig, NQ); dy[0] = 0.0
        part = {"s": Ps, "a": Pa, "f": np.eye(NQ)}
        ci, cj = {"ss": "ss", "sa": "sa", "as": "as", "aa": "aa", "full": "ff"}[ch]
        dx, dy = part[ci] @ dx, part[cj] @ dy
    elif kind == "rawcw":
        rng = np.random.default_rng(3000 + seed)
        dx = rng.normal(0, sig, NQ); dy = rng.normal(0, sig, NQ)
    else:
        dx, dy = residual(kind, seed, sig)
        if kind == "tb20":
            from build_rtrue import dG_pattern
            dG = dG_pattern(0.002)
    r = measure_frozen(dx, dy, direction=d, dG=dG, n_turns=14, n_iter=2)
    Ps, _ = projectors()
    return dict(kind=kind, seed=seed, sigma=sig, channel=ch, direction=d, f=r["f"], f_target=r["f"] / 1e-9,
                w1=r["w1"], sym_um=float(np.mean([np.std(Ps @ dx), np.std(Ps @ dy)]) * 1e6))


def main(workers):
    done = set()
    if os.path.exists(OUT):
        done = {(r["kind"], r["seed"], r["sigma"], r["channel"], r["direction"]) for r in map(json.loads, open(OUT))}
    T = [t for t in tasks_all() if t not in done]
    print(f"{len(T)} görev", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(task, T):
            open(OUT, "a").write(json.dumps(r) + "\n")
            print(f"{r['kind']:5s} seed {r['seed']:2d} σ={r['sigma']*1e6:4.1f} {r['channel']:4s} yön {r['direction']:+d}: "
                  f"f* = {r['f_target']:12.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("-w", "--workers", type=int, default=4)
    main(ap.parse_args().workers)
