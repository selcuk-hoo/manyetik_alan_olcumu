#!/usr/bin/env python3
"""teshis_baslangic_dogrulama.py — Test 4'ün üçüncü adımı: eski kuraldan başlayarak bulunan δ*'da doğrudan
ölçüm (ω_h ≈ 0 olmalı; f orada okunur). k = 0 ve 6."""
import os, sys, json, tempfile
import numpy as np
B = "/home/user/manyetik_alan_olcumu"
sys.path[:0] = [B + "/revizyon", B + "/revizyon/tests", B, B + "/berry_data", B + "/kmod_drivers"]
from teshis_baslangic_kurali import pmag_launcher
OUT = B + "/revizyon/sonuclar/teshis_baslangic_dogrulama.json"


def run(k):
    os.chdir(tempfile.mkdtemp())
    import sahte_edm_donmus as SD
    SD.launcher = pmag_launcher
    prev = {r["k"]: r for r in json.load(open(B + "/revizyon/sonuclar/teshis_baslangic_kurali.json"))}[k]
    rng = np.random.default_rng(3000)
    dx = np.roll(rng.normal(0, 10e-6, 48), -2 * k); dy = np.roll(rng.normal(0, 10e-6, 48), -2 * k)
    f, w, res = SD._one(dx, dy, -1, prev["delta_star"], None, None, 28, 4, 3e-4)
    return dict(k=k, delta=prev["delta_star"], f_target=f / 1e-9, w=w, f_interp=prev["f_target"])


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(2) as p:
        R = list(p.map(run, [0, 6]))
    json.dump(R, open(OUT, "w"), indent=1)
    for r in R:
        print(f"k={r['k']}: δ* = {r['delta']:+.3e} → doğrudan f = {r['f_target']:.3f} × hedef, ω_h = {r['w']:+.2e} "
              f"(ara değerleme {r['f_interp']:.3f})")
