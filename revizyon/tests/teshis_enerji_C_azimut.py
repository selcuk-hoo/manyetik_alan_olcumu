import sys, json, numpy as np
sys.path.insert(0, "/home/user/manyetik_alan_olcumu/revizyon/tests")
import teshis_enerji_ABC as T
from concurrent.futures import ProcessPoolExecutor
res = json.load(open(T.OUT))
a = sorted([r for r in res if r["mode"]=="energy" and r["k"]==0 and abs(r["delta"])<=1e-7+1e-15 and not r.get("C_test")], key=lambda r:r["delta"])
slope = np.polyfit([r["delta"] for r in a], [r["w_h"] for r in a], 1)[0]
tasks = []
for k in (6, 12, 18):
    w0 = [r for r in res if r["mode"]=="energy" and r["k"]==k and r["delta"]==0.0][0]["w_h"]
    tasks.append(("energy", float(-w0/slope), k))
with ProcessPoolExecutor(3) as p:
    for r in p.map(T.run, tasks):
        r["C_test"] = True; res.append(r); json.dump(res, open(T.OUT, "w"), indent=1)
        print(f"C k={r['k']:2d}: δ* = {r['delta']:+.3e} → f = {r['f_target']:.2f} × hedef, ω_h = {r['w_h']:+.3e}", flush=True)
