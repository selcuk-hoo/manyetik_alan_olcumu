#!/usr/bin/env python3
"""BBA geçiş desenlerinin kanal ayrıştırması: her geçişin kalan (res_x, res_y) desenini
ss/sa/as/aa (simetrik/antisimetrik dx × dy) ve full kanallarında donmuş-spin sahte EDM'i (İŞARETLİ) ölçer.
Soru: geçiş 1'deki düşük f kanallar arası sönümlenme mi? (geçiş 2'de f 4.17→13.40× arttı)
Kullanım: bba_kanal.py -w 2 [--passes 0,1,2,3,4]
Çıktı: revizyon/sonuclar/bba_gecis_kanallar.jsonl (kaldığı yerden devam eder)"""
import os, sys, json, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BASE = "/Users/selcuk/works/manyetik_alan_olcumu"
REV = os.path.join(BASE, "revizyon")
sys.path[:0] = [REV, BASE, os.path.join(BASE, "kmod_drivers")]
OUT = os.path.join(REV, "sonuclar", "bba_gecis_kanallar.jsonl")
STATE = os.path.join(REV, "sonuclar", "bba_iter_cpp_donmus_state.json")
NQ = 48


def projectors():
    Ps = np.zeros((NQ, NQ))
    for k in range(NQ // 2):
        a, b = 2 * k, 2 * k + 1
        Ps[a, a] = Ps[b, b] = Ps[a, b] = Ps[b, a] = 0.5
    return Ps, np.eye(NQ) - Ps


def desenler():
    rng = np.random.default_rng(0)      # classic_bba_iter_donmus.py ile aynı: seed 0, σ=10 μm
    m_dx = rng.normal(0.0, 10e-6, NQ); m_dx[0] = 0.0
    m_dy = rng.normal(0.0, 10e-6, NQ); m_dy[0] = 0.0
    d = {0: (m_dx, m_dy)}
    for h in json.load(open(STATE))["hist"]:
        d[h["pass"]] = (np.array(h["res_x"]), np.array(h["res_y"]))
    return d


def task(t):
    p, ch = t
    from sahte_edm_donmus import measure_frozen
    dx, dy = desenler()[p]
    Ps, Pa = projectors()
    part = {"s": Ps, "a": Pa, "f": np.eye(NQ)}
    ci, cj = {"ss": "ss", "sa": "sa", "as": "as", "aa": "aa", "full": "ff"}[ch]
    r = measure_frozen(part[ci] @ dx, part[cj] @ dy, direction=-1, n_turns=28, n_iter=4)  # BBA ile aynı ayar
    return dict(pass_=p, channel=ch, f=r["f"], f_target=r["f"] / 1e-9, co_resid=r["co_resid"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--workers", type=int, default=2)
    ap.add_argument("--passes", default="0,1,2,3,4")
    a = ap.parse_args()
    gecisler = [int(x) for x in a.passes.split(",")]
    done = set()
    if os.path.exists(OUT):
        done = {(r["pass_"], r["channel"]) for r in map(json.loads, open(OUT))}
    T = [(p, ch) for p in gecisler for ch in ("full", "ss", "sa", "as", "aa") if (p, ch) not in done]
    print(f"{len(T)} görev", flush=True)
    with ProcessPoolExecutor(a.workers, initializer=__import__("build_response_matrix")._worker_init) as pool:
        for r in pool.map(task, T):
            open(OUT, "a").write(json.dumps(r) + "\n")
            print(f"geçiş {r['pass_']} {r['channel']:4s}: f* = {r['f_target']:+9.3f}", flush=True)
