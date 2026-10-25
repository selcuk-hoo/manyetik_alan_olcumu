#!/usr/bin/env python3
"""donmus_fig4.py — makale Şekil 4'ün donmuş spin ölçüm aracıyla tekrarı + σ=10 μm'de C.

Kurulum kmod_drivers/twobeam_oc.py ile aynı (14 nm gürültü, nominal model, BPM ofseti yok):
  desen: rng(seed): m_dx = N(0,10 μm), m_dy = N(0,10 μm), kuad 0 = 0
  raw : düzeltme yok
  one : tek demet (R_CCW), TSVD rcond 0.01 — makalenin 'orbit corr.' noktası
  two : iki demet ([R_CCW; R_CW]), TSVD rcond 0.01 — makalenin elması
Her durum iki yönde donmuş spin noktasında ölçülür; C = ½|f_CW − f_CCW|.
Ek: σ=10 μm sigma desenleri (seed 0–2) için CW yönü (CCW donmus_sigma.json'da) → C.
"""
import os, sys, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor

REV = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(REV)
sys.path.insert(0, REV)
from sahte_edm_donmus import measure_frozen, OUT_SIGMA

OUT = os.path.join(REV, "sonuclar", "donmus_fig4.jsonl")
NQ = 48
RCOND = 0.01


def residual(seed, case, sig=14e-9):
    Rx_c = np.load(os.path.join(BASE, "R_dx_1.npy")); Ry_c = np.load(os.path.join(BASE, "R_dy_1.npy"))
    Rx_w = np.load(os.path.join(BASE, "R_dx_cw.npy")); Ry_w = np.load(os.path.join(BASE, "R_dy_cw.npy"))
    rng = np.random.default_rng(seed)
    m_dx = rng.normal(0, 10e-6, NQ); m_dx[0] = 0.0
    m_dy = rng.normal(0, 10e-6, NQ); m_dy[0] = 0.0
    nr = np.random.default_rng(50000 + seed)
    nxa, nxb = nr.normal(0, sig, NQ), nr.normal(0, sig, NQ)
    nya, nyb = nr.normal(0, sig, NQ), nr.normal(0, sig, NQ)
    if case == "raw":
        return m_dx, m_dy
    if case == "one":
        cx = m_dx - np.linalg.pinv(Rx_c, rcond=RCOND) @ (Rx_c @ m_dx + nxa)
        cy = m_dy - np.linalg.pinv(Ry_c, rcond=RCOND) @ (Ry_c @ m_dy + nya)
        return cx, cy
    Sx = np.vstack([Rx_c, Rx_w]); Sy = np.vstack([Ry_c, Ry_w])
    cx = m_dx - np.linalg.pinv(Sx, rcond=RCOND) @ np.concatenate([Rx_c @ m_dx + nxa, Rx_w @ m_dx + nxb])
    cy = m_dy - np.linalg.pinv(Sy, rcond=RCOND) @ np.concatenate([Ry_c @ m_dy + nya, Ry_w @ m_dy + nyb])
    return cx, cy


def task(t):
    kind, seed, case, d = t
    if kind == "fig4":
        dx, dy = residual(seed, case)
    else:                                        # sigma deseni, 10 μm
        rng = np.random.default_rng(3000 + seed); dx = rng.normal(0, 10e-6, NQ); dy = rng.normal(0, 10e-6, NQ)
    r = measure_frozen(dx, dy, direction=d)
    return dict(kind=kind, seed=seed, case=case, direction=d, f=r["f"], f_target=r["f"] / 1e-9,
                delta_star=r["delta_star"], w0=r["w0"], w1=r["w1"])


def main():
    done = set()
    if os.path.exists(OUT):
        done = {(r["kind"], r["seed"], r["case"], r["direction"]) for r in map(json.loads, open(OUT))}
    tasks = [("sigma", s, "raw", +1) for s in (0, 1, 2)]
    tasks += [("fig4", s, c, d) for c in ("raw", "one", "two") for s in range(5) for d in (-1, +1)]
    tasks = [t for t in tasks if t not in done]
    print(f"{len(tasks)} görev (her biri 2 izleme)", flush=True)
    with ProcessPoolExecutor(4) as pool:
        for r in pool.map(task, tasks):
            open(OUT, "a").write(json.dumps(r) + "\n")
            print(f"{r['kind']:5s} seed {r['seed']} {r['case']:3s} yön {r['direction']:+d}: f* = {r['f_target']:10.4f} × hedef "
                  f"(ω1 = {r['w1']:+.1e})", flush=True)
    summarize()


def summarize():
    rows = [json.loads(l) for l in open(OUT)]
    ccw10 = {r["seed"]: r["f"] for r in json.load(open(OUT_SIGMA)) if r["sigma_um"] == 10.0 and r["direction"] == -1}
    print("\nσ = 10 μm sigma desenleri: C = ½|f_CW − f_CCW|")
    for r in rows:
        if r["kind"] == "sigma" and r["seed"] in ccw10:
            fc, fw = ccw10[r["seed"]], r["f"]
            print(f"  seed {r['seed']}: f_CCW = {fc/1e-9:8.2f}, f_CW = {fw/1e-9:8.2f}, C = {abs(fw-fc)/2/1e-9:8.2f}, "
                  f"çift = {(fw+fc)/2/1e-9:8.2f} × hedef")
    print("\nŞekil 4 (twobeam_oc kurulumu, 5 seed): C × hedef")
    for c in ("raw", "one", "two"):
        Cs = []
        for s in range(5):
            f = {r["direction"]: r["f"] for r in rows if r["kind"] == "fig4" and r["case"] == c and r["seed"] == s}
            if len(f) == 2:
                Cs.append(abs(f[1] - f[-1]) / 2 / 1e-9)
        if Cs:
            print(f"  {c:3s}: medyan {np.median(Cs):.4g}  [{min(Cs):.4g}, {max(Cs):.4g}]  değerler {['%.4g' % x for x in Cs]}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["ozet"]:
        summarize()
    else:
        main()
