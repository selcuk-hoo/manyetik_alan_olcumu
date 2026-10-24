#!/usr/bin/env python3
"""adim4_ground_bpm.py — Adım 4 eki: yer hareketi (ATL/rastgele yürüyüş) hem kuadrupolleri hem
BPM'leri taşırsa (aynı zemin), BPM'ler uzun dalgada kuadrupolle birlikte hareket eder (Adım 1 ρ≈1).

Sürüklenme (BBA sonrası, fark okuması):
  zemin   g(s): periyodik rastgele yürüyüş (düzlem çıkarılmış)
  kuad    Δm = g + q_loc     (q_loc: kuadrupol yerel hareketi, iid)
  BPM     u  = g + b_loc     (b_loc: BPM yerel hareketi, iid)
Okuma   Δr_b = R_b Δm − u + n
Kestiriciler (Adım 1): E1 tek-demet standart, E3 iki-demet (makale, BPM sabit sanır),
  E2/E4: BPM'in zemini izlediğini bilen (ρ=1 modeli: R−I) tek/iki demet.
Yerel/zemin oranı f_loc = rms(q_loc)/rms(g) taranır. Toplam kuadrupol sürüklenmesi 5 μm rms.
Metrik: kalan simetrik ve antisimetrik rms (200 seed medyan), en iyi rcond.
"""
import os, sys, json
import numpy as np
REV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REV)
import adim4_correlated as A

R = A.load_R(); I = np.eye(48)
RC = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]


def pinv(M, rc):
    n = np.linalg.norm(M, axis=0)
    return np.linalg.pinv(M / n, rcond=rc) / n[:, None]


def run(sig_n=14e-9, nseed=200, drift=5e-6):
    out = []
    for floc in (0.0, 0.1, 0.3, 1.0, 3.0):
        acc = {}
        for pl in "xy":
            Rc, Rw = R[pl]["ccw"], R[pl]["cw"]
            mats = {"E1": (Rc, "one"), "E3": (np.vstack([Rc, Rw]), "two"),
                    "E2": (Rc - I, "one"), "E4": (np.vstack([Rc - I, Rw - I]), "two")}
            for rc in RC:
                P = {k: pinv(M, rc) for k, (M, _) in mats.items()}
                for s in range(nseed):
                    rng = np.random.default_rng(90_000 + s * 2 + (pl == "y"))
                    g = A.remove_plane(A.gen("rw", rng), pl); g /= np.std(g)
                    ql = rng.normal(0, 1, 48); bl = rng.normal(0, 1, 48)
                    scale = drift / np.sqrt(1 + floc ** 2)
                    dm = scale * (g + floc * ql); u = scale * (g + floc * bl)
                    n1 = rng.normal(0, sig_n, 48); n2 = rng.normal(0, sig_n, 48)
                    r1 = Rc @ dm - u + n1; r2 = Rw @ dm - u + n2
                    for k, (_, kind) in mats.items():
                        y = r1 if kind == "one" else np.concatenate([r1, r2])
                        res = dm - P[k] @ y
                        acc.setdefault((k, rc), {"sym": [], "anti": [], "tot": []})
                        acc[(k, rc)]["sym"].append(np.std(A.PS @ res))
                        acc[(k, rc)]["anti"].append(np.std(A.PA @ res))
                        acc[(k, rc)]["tot"].append(np.std(res))
        row = {"f_loc": floc, "sig_n": sig_n}
        for k in ("E1", "E2", "E3", "E4"):
            best = min(RC, key=lambda rc: np.median(acc[(k, rc)]["tot"]))
            row[k] = {"rcond": best, "sym_um": float(np.median(acc[(k, best)]["sym"]) * 1e6),
                      "anti_um": float(np.median(acc[(k, best)]["anti"]) * 1e6)}
        out.append(row)
        print(f"σn={sig_n*1e9:.0f}nm yerel/zemin={floc:.1f}: "
              + " | ".join(f"{k} sim {row[k]['sym_um']:.3f} anti {row[k]['anti_um']:.3f}"
                           for k in ("E1", "E2", "E3", "E4")), flush=True)
    return out


if __name__ == "__main__":
    res = {"14nm": run(14e-9), "1um": run(1e-6)}
    json.dump(res, open(os.path.join(A.OUTDIR, "adim4_ground_bpm.json"), "w"), indent=1)
