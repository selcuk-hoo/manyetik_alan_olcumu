#!/usr/bin/env python3
"""build_rtrue.py — β-beat'li "gerçek makine" tepki matrisleri (CW + CCW).

Gradyan hatası deseni twobeam_oc.py ile AYNI: rng(0).normal(0, σ_g, 48), quad 0 = 0
(cell-0 QF quad_dG okumaz, CLAUDE.md tuzak #8). σ_g=0.002 → ~%1 β-beat
(makale Şekil 4 elması ile aynı koşul).

Kullanım: python3 revizyon/build_rtrue.py --sig-g 0.002 -w 4
Çıktı: revizyon/sonuclar/Rtrue_{dx,dy}_{ccw,cw}_g{σ_g·1e4}.npy
"""
import os, sys, json, argparse, time
import numpy as np
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); os.chdir(BASE)
import build_response_matrix as brm

OUTDIR = os.path.join(BASE, "revizyon", "sonuclar")


def dG_pattern(sig_g):
    g = np.random.default_rng(0).normal(0.0, sig_g, 48); g[0] = 0.0
    return g


def path(plane, d, sig_g):
    return os.path.join(OUTDIR, f"Rtrue_{plane}_{d}_g{int(round(sig_g*1e4))}.npy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig-g", type=float, default=0.002)
    ap.add_argument("-w", "--workers", type=int, default=4)
    a = ap.parse_args()
    dG = dG_pattern(a.sig_g)
    for d, tag in ((-1.0, "ccw"), (1.0, "cw")):
        if os.path.exists(path("dy", tag, a.sig_g)):
            print(tag, "mevcut, atlandı"); continue
        cfg = json.load(open("params.json")); cfg["direction"] = d
        t0 = time.time()
        Ry, Rx = brm.build_matrices(cfg, delta_q=1e-4, n_workers=a.workers,
                                    quad_dG_pert=dG, label=f"Rtrue {tag}")
        np.save(path("dy", tag, a.sig_g), Ry); np.save(path("dx", tag, a.sig_g), Rx)
        print(f"{tag} bitti ({time.time()-t0:.0f}s)", flush=True)
