"""Donmuş spin aracında kısaltılmış ayar (n_turns=14, n_iter=2) ↔ tam ayar (28, 4) karşılaştırması
ve kaçıklıksız sayısal taban. Beklenen: seed 0, σ=10 μm, CCW: f* = 82.04 × hedef (tam ayar)."""
import sys, json, numpy as np
sys.path.insert(0, "/home/user/manyetik_alan_olcumu/revizyon")
from sahte_edm_donmus import measure_frozen
from concurrent.futures import ProcessPoolExecutor

def task(t):
    name, n_turns, n_iter, d, zero = t
    rng = np.random.default_rng(3000); dx = rng.normal(0, 10e-6, 48); dy = rng.normal(0, 10e-6, 48)
    if zero: dx = dy = np.zeros(48)
    r = measure_frozen(dx, dy, direction=d, n_turns=n_turns, n_iter=n_iter)
    return name, r["f"] / 1e-9, r["w1"]

T = [("seed0 kısaltılmış CCW", 14, 2, -1, False), ("sıfır kaçıklık kısaltılmış CCW", 14, 2, -1, True),
     ("sıfır kaçıklık tam CCW", 28, 4, -1, True), ("sıfır kaçıklık kısaltılmış CW", 14, 2, +1, True)]
with ProcessPoolExecutor(4) as p:
    out = list(p.map(task, T))
for n, f, w in out:
    print(f"{n}: f* = {f:.5f} × hedef (ω1 = {w:+.1e})")
json.dump(out, open("/home/user/manyetik_alan_olcumu/revizyon/sonuclar/teshis_kisaltilmis_ayar.json", "w"), indent=1)
