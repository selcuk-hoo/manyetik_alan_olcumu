#!/usr/bin/env python3
"""σ=10 μm'de CW-tek sahte EDM'in mutlak seviyesi: çok-seed.
JSON ile AYNI ayar (fast_measure_g varsayılan: n_turns=14, t2=3e-4).
Amaç: 10-15 seed'de ~1e-5 mi ~1e-6 mı görülüyor? (Omarov fit: 1.5e-5 @ σ=10.)
Kullanım (lokal): python3 kmod_drivers/sigma_abs_check.py [nseed]
"""
import sys, os, numpy as np, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (BASE, os.path.join(BASE,"kmod_drivers"), os.path.join(BASE,"berry_data")):
    sys.path.insert(0, p)
os.chdir(BASE)
from paper_runs import fast_measure_g, NQ
nseed = int(sys.argv[1]) if len(sys.argv) > 1 else 15
sg = 10e-6; vals = []; t0 = time.time()
for seed in range(nseed):
    rng = np.random.default_rng(3000 + seed)
    dx = rng.normal(0, sg, NQ); dy = rng.normal(0, sg, NQ)
    f, _ = fast_measure_g(dx, dy, g_scale=1.0)
    vals.append(f); v = np.array(vals)
    rms = np.sqrt((v**2).mean())
    print(f"seed {seed:2d}: f={f:.3e}  | koşu RMS={rms:.3e}  ({rms/1e-9:.0f}× hedef)  "
          f"[Omarov 1.5e-5 / RMS = {1.5e-5/rms:.1f}×]  t={time.time()-t0:.0f}s", flush=True)
v = np.array(vals)
print(f"\n=== {len(v)} seed, σ=10μm ===")
print(f"mean={v.mean():.3e}  RMS={np.sqrt((v**2).mean()):.3e}  medyan={np.median(v):.3e}  max={v.max():.3e}")
