#!/usr/bin/env python3
"""teshis_baslangic_kurali.py — donmuş spin yönteminin başlangıç kuralından bağımsızlığı.

Donmuş spin aracı (sahte_edm_donmus.measure_frozen) δ = 0 başlangıç noktası olarak "enerji korunumlu"
kuralı kullanır. Burada aynı prosedür, başlangıç kuralı olarak ESKİ yöntemin kuralıyla
(|p| = p_sihirli·(1+δ)) çalıştırılır. Desen: σ = 10 μm, seed 0 (rng 3000), CCW, tam ayar (28/4),
fırlatma hücresi k = 0, 6, 12, 18 (desen 2k kuadrupol kaydırılır).
Beklenti: dört sonuç da enerji kuralıyla bulunan değere (82.04 × hedef) eşit olmalı.
"""
import os, sys, json, tempfile
import numpy as np
B = "/home/user/manyetik_alan_olcumu"
sys.path[:0] = [B + "/revizyon", B + "/revizyon/tests", B, B + "/berry_data", B + "/kmod_drivers"]
OUT = B + "/revizyon/sonuclar/teshis_baslangic_kurali.json"


def pmag_launcher(delta):
    def ms(v, p_mag, direction, spin):
        p = p_mag * (1.0 + delta)
        return [v[0], v[1], 0.0, p * direction * v[2], p * direction * v[3], p * direction,
                spin[0], spin[1], spin[2]]
    return ms


def run(k):
    os.chdir(tempfile.mkdtemp())
    import sahte_edm_donmus as SD
    SD.launcher = pmag_launcher                      # başlangıç kuralını değiştir
    rng = np.random.default_rng(3000)
    dx = np.roll(rng.normal(0, 10e-6, 48), -2 * k); dy = np.roll(rng.normal(0, 10e-6, 48), -2 * k)
    r = SD.measure_frozen(dx, dy, direction=-1, n_turns=28, n_iter=4)
    r.update(k=k, f_target=r["f"] / 1e-9, f0_target=r["f0"] / 1e-9)
    return r


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(4) as p:
        R = list(p.map(run, [0, 6, 12, 18]))
    json.dump(R, open(OUT, "w"), indent=1)
    for r in R:
        print(f"k={r['k']:2d}: başlangıç (eski kural) f0 = {r['f0_target']:9.2f}, ω0 = {r['w0']:+.3e}  →  "
              f"donmuş f* = {r['f_target']:.3f} × hedef (ω1 = {r['w1']:+.1e}, δ* = {r['delta_star']:+.3e})")
