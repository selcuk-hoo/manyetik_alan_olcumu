#!/usr/bin/env python3
"""firlatma_eski_ham.py — Şekil 4 'ham' (düzeltilmemiş) desenleri ESKİ prosedürle (makale: fast_measure,
|p| = p_sihirli, 28/4) — donmus_fig4.jsonl'daki yeni değerlerle seed seed karşılaştırma için."""
import os, sys, json, tempfile
import numpy as np
REV = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(REV)
sys.path[:0] = [REV, os.path.join(BASE, "kmod_drivers"), os.path.join(BASE, "berry_data"), BASE]
OUT = os.path.join(REV, "sonuclar", "firlatma_eski_ham.json")


def run(t):
    s, d = t
    os.chdir(tempfile.mkdtemp())
    import donmus_fig4 as D
    from fast_est import fast_measure
    dx, dy = D.residual(s, "raw")
    f, _ = fast_measure(dx, dy, direction=d, signed=True, n_turns=28, n_iter=4)
    return dict(seed=s, direction=d, f_target=f / 1e-9)


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(4) as p:
        R = list(p.map(run, [(s, d) for s in range(5) for d in (-1, 1)]))
    json.dump(R, open(OUT, "w"), indent=1)
    for r in R:
        print(r)
