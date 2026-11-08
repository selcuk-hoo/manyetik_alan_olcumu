#!/usr/bin/env python3
"""ek_rcond_secimi.py — ek rapor madde 14: TSVD eşiği gerçekte nasıl seçilir?

'Kahin' seçim (Adım 1): gerçek kaçıklığı bilerek 200 seed'in medyan simetrik artığını en küçük yapan eşik.
Uygulanabilir seçim (Morozov uyuşmazlık ilkesi): yalnız bilinen BPM gürültüsü σ_n kullanılır; ızgaradaki en
BÜYÜK rcond seçilir, öyle ki fit artığı ‖S m̂ − y‖ ≤ τ·σ_n·√N_ölçüm (τ = 1.1). Gerçek kaçıklık kullanılmaz;
her seed kendi eşiğini kendi okumasından seçer. Senaryo Adım 1 ('k0', 5 μm saf simetrik, 14 nm).
"""
import os, sys, json
import numpy as np
REV = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, REV)
import adim1_bpm_model as A1
NQ = 48; I = np.eye(NQ); TAU = 1.1
GRID = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]


def one(seed, rho, method, sig_n=14e-9, rms=5e-6):
    rng = np.random.default_rng(10_000 + seed)
    out = {}
    for pl in ("x", "y"):
        dm = A1.drift_pattern(rng, "sym", rms); A1.drift_pattern(rng, "gen", rms)
        n1 = rng.normal(0, sig_n, NQ); n2 = rng.normal(0, sig_n, NQ)
        Rc, Rw = A1.R[pl]["ccw"], A1.R[pl]["cw"]
        r1 = Rc @ dm - rho * dm + n1; r2 = Rw @ dm - rho * dm + n2
        rh = rho if method in ("E2", "E4") else 0.0
        if method in ("E1", "E2"):
            S = Rc - rh * I; y = r1
        else:
            S = np.vstack([Rc - rh * I, Rw - rh * I]); y = np.concatenate([r1, r2])
        U, s, Vt = np.linalg.svd(S, full_matrices=False)
        best = None
        for rc in GRID:                      # büyükten küçüğe: ilk uyan = en çok düzenlileştiren
            k = s >= rc * s[0]
            m = Vt[k].T @ ((U[:, k].T @ y) / s[k])
            if np.linalg.norm(S @ m - y) <= TAU * sig_n * np.sqrt(len(y)):
                best = (rc, m); break
        if best is None:
            best = (GRID[-1], m)
        out[pl] = (best[0], dm - best[1])
    return out


rows = []
for rho, m in [(0.0, "E1"), (0.0, "E3"), (1.0, "E2"), (1.0, "E4"), (1.0, "E3")]:
    sy, rcs = [], []
    for s in range(200):
        o = one(s, rho, m)
        sy.append(np.mean([np.std(A1.PS @ o[pl][1]) for pl in "xy"]) * 1e6); rcs += [o[pl][0] for pl in "xy"]
    orc = A1.best_rcond("k0", "sym", 14e-9, rho, m)
    rows.append(dict(rho=rho, method=m, morozov_sym_med=float(np.median(sym := np.array(sy))),
                     morozov_p84=float(np.percentile(sym, 84)), rcond_mode=float(max(set(rcs), key=rcs.count)),
                     oracle_rcond=orc))
    print(f"ρ={rho} {m}: Morozov seçimi kalan sim medyan {rows[-1]['morozov_sym_med']:.3f} μm "
          f"(84%: {rows[-1]['morozov_p84']:.3f}), en sık rcond {rows[-1]['rcond_mode']:.0e}; kahin rcond {orc:.0e}", flush=True)
json.dump(rows, open(os.path.join(REV, "sonuclar", "ek_rcond_secimi.json"), "w"), indent=1)
